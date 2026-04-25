import functools
import importlib.util
import os
import tempfile
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import pyarrow.parquet as pq
import streamlit as st
import streamlit.components.v1 as components
from backtesting import Backtest
from pathlib import Path

st.set_page_config(page_title="Optimus Prime", layout="wide")

APP_DIR = Path(__file__).parent
DB_DIR  = APP_DIR.parent / "Database"

def _negate_metric(stats, key):
    return -stats[key]


def _optimize(bt, param_ranges, maximize_fn, max_tries, constraint, random_state=42):
    """Sequential grid search — avoids multiprocessing pickling issues on Streamlit Cloud."""
    import itertools, random, math

    keys  = list(param_ranges.keys())
    lists = [list(v) if (hasattr(v, '__iter__') and not isinstance(v, str)) else [v]
             for v in param_ranges.values()]
    combos = list(itertools.product(*lists))

    if len(combos) > max_tries:
        combos = random.Random(random_state).sample(combos, max_tries)

    best_stats, best_score, records = None, None, []

    for combo in combos:
        params = dict(zip(keys, combo))
        if constraint is not None:
            try:
                if not constraint(pd.Series(params)):
                    continue
            except Exception:
                continue
        try:
            stats = bt.run(**params)
            score = maximize_fn(stats) if callable(maximize_fn) else stats[maximize_fn]
            if not math.isfinite(score):
                continue
            records.append((*combo, score))
            if best_score is None or score > best_score:
                best_score, best_stats = score, stats
        except Exception:
            continue

    if not records or best_stats is None:
        return None, pd.Series(dtype=float)

    idx     = pd.MultiIndex.from_tuples([r[:-1] for r in records], names=keys)
    heatmap = pd.Series([r[-1] for r in records], index=idx)
    return best_stats, heatmap

OPTIMIZE_METRICS = {
    "Sharpe Ratio":       ("Sharpe Ratio",        False),
    "Return [%]":         ("Return [%]",           False),
    "Return (Ann.) [%]":  ("Return (Ann.) [%]",    False),
    "Calmar Ratio":       ("Calmar Ratio",          False),
    "Win Rate [%]":       ("Win Rate [%]",          False),
    "Profit Factor":      ("Profit Factor",         False),
    "Max Drawdown (min)": ("Max. Drawdown [%]",     True),
}

# ── STRATEGY DISCOVERY ────────────────────────────────────────────────────────
def load_strategies():
    strats = {}
    for f in sorted((APP_DIR / "strategies").glob("*.py")):
        if f.stem.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        strats[mod.NAME] = mod
    return strats

STRATEGIES = load_strategies()

# ── COMMODITY DISCOVERY ───────────────────────────────────────────────────────
@st.cache_data
def get_commodities():
    valid = []
    for p in sorted(DB_DIR.glob("rollex_*.parquet")):
        try:
            schema = pq.read_schema(p)
            if "rollex_open" in schema.names:
                valid.append(p.stem.replace("rollex_", "").upper())
        except Exception:
            pass
    return valid

COMMODITIES = get_commodities()

# ── DATA LOADER ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data(commodity):
    df = pd.read_parquet(DB_DIR / f"rollex_{commodity}.parquet")
    df = df.rename(columns={
        "rollex_open": "Open",
        "rollex_high": "High",
        "rollex_low":  "Low",
        "rollex_px":   "Close",
    })
    return df[["Open", "High", "Low", "Close"]].dropna()

# ── HELPERS ───────────────────────────────────────────────────────────────────
def stats_tables(stats):
    drop = {"_equity_curve", "_trades", "_strategy", "_broker"}
    rows = []
    for k, v in stats.items():
        if k in drop:
            continue
        rows.append({"Metric": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)})
    half = len(rows) // 2
    l, r = st.columns(2)
    with l:
        st.dataframe(pd.DataFrame(rows[:half]),  hide_index=True, use_container_width=True)
    with r:
        st.dataframe(pd.DataFrame(rows[half:]), hide_index=True, use_container_width=True)


def bt_plot(bt):
    fd, tmp = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        bt.plot(filename=tmp, open_browser=False)
        with open(tmp, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=750, scrolling=True)
    finally:
        os.unlink(tmp)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("Optimus Prime")
st.caption("Roll-adjusted continuous futures backtester")

if not COMMODITIES:
    st.error("No commodities with OHLC data found. Run rollex_builder --full first.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    commodity     = st.selectbox("Commodity", COMMODITIES)
    strategy_name = st.selectbox("Strategy",  list(STRATEGIES.keys()))
    mod           = STRATEGIES[strategy_name]

    st.divider()
    st.subheader("Engine")
    cash       = st.number_input("Starting Cash ($)", value=10_000, step=1_000, min_value=1_000)
    commission = st.number_input("Commission", value=0.002, step=0.001, format="%.3f", min_value=0.0)
    size       = st.number_input("Trade Size (fraction of equity)", min_value=0.01, max_value=1.0, value=0.95, step=0.05, format="%.2f")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
df_full = load_data(commodity)
min_date = df_full.index.min().date()
max_date = df_full.index.max().date()

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Backtest", "Optimize"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    c_params, c_engine = st.columns([2, 1])

    with c_params:
        st.subheader("Strategy Parameters")
        params = mod.params_ui(st)

    with c_engine:
        st.subheader("Date Range")
        dc1, dc2 = st.columns(2)
        bt_from = dc1.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="bt_from")
        bt_to   = dc2.date_input("To",   value=max_date, min_value=min_date, max_value=max_date, key="bt_to")
        date_range = st.slider(
            "Drag to adjust",
            min_value=min_date, max_value=max_date,
            value=(bt_from, bt_to),
            key="bt_slider",
        )

    st.divider()
    run = st.button("Run Backtest", type="primary", key="run_bt")

    if run:
        df = df_full.loc[str(date_range[0]):str(date_range[1])]
        st.caption(f"{commodity}  |  {date_range[0]} to {date_range[1]}  |  {len(df):,} trading days")

        with st.spinner("Running..."):
            StrategyClass = mod.build(params, size=size)
            bt    = Backtest(df, StrategyClass, cash=cash, commission=commission, exclusive_orders=True)
            stats = bt.run()

        trades = stats["_trades"]

        st.subheader("Performance Summary")
        stats_tables(stats)
        st.divider()

        st.subheader("Strategy Chart")
        bt_plot(bt)

        if not trades.empty:
            with st.expander(f"Trade Log  ({len(trades)} trades)"):
                display = trades[["EntryTime","ExitTime","Size","EntryPrice","ExitPrice","ReturnPct","PnL","Duration"]].copy()
                display["ReturnPct"] = (display["ReturnPct"] * 100).round(2)
                display["PnL"]       = display["PnL"].round(2)
                display = display.rename(columns={"ReturnPct": "Return %"})
                st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption(f"{commodity}  |  {min_date} to {max_date}  |  {len(df_full):,} trading days")
        st.info("Configure parameters above and click Run Backtest.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — OPTIMIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not hasattr(mod, "optimize_params_ui"):
        st.warning("This strategy does not support optimization yet.")
    else:
        c_opt, c_cfg = st.columns([2, 1])

        with c_opt:
            st.subheader("Parameter Ranges")
            opt_cfg = mod.optimize_params_ui(st)

        with c_cfg:
            st.subheader("Date Range")
            oc1, oc2 = st.columns(2)
            opt_from = oc1.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="opt_from")
            opt_to   = oc2.date_input("To",   value=max_date, min_value=min_date, max_value=max_date, key="opt_to")
            opt_dates = st.slider(
                "Drag to adjust",
                min_value=min_date, max_value=max_date,
                value=(opt_from, opt_to),
                key="opt_slider",
            )
            st.subheader("Maximize")
            metric_label = st.selectbox("Metric", list(OPTIMIZE_METRICS.keys()))
            st.subheader("Max Tries")
            max_tries = st.slider("Combinations to sample", min_value=50, max_value=2000, value=300, step=50)

        st.divider()
        run_opt = st.button("Run Optimizer", type="primary", key="run_opt")

        if run_opt:
            metric_key, minimize = OPTIMIZE_METRICS[metric_label]
            maximize_fn = functools.partial(_negate_metric, key=metric_key) if minimize else metric_key

            df_opt = df_full.loc[str(opt_dates[0]):str(opt_dates[1])]  # opt_dates comes from slider synced to calendars
            st.caption(f"{commodity}  |  {opt_dates[0]} to {opt_dates[1]}  |  {len(df_opt):,} trading days")

            StrategyClass = mod.build(
                {k: list(v)[0] if hasattr(v, '__iter__') and not isinstance(v, str) else v
                 for k, v in opt_cfg["ranges"].items()},
                size=size
            )
            bt_opt = Backtest(df_opt, StrategyClass, cash=cash, commission=commission, exclusive_orders=True)

            total_combos = 1
            for v in opt_cfg["ranges"].values():
                total_combos *= len(list(v)) if hasattr(v, '__iter__') and not isinstance(v, str) else 1
            sampling = int(max_tries) < total_combos
            label = f"Sampling {int(max_tries):,} of {total_combos:,} combinations..." if sampling else f"Running all {total_combos:,} combinations..."

            with st.spinner(label):
                constraint  = opt_cfg.get("constraint")
                best_stats, heatmap = _optimize(
                    bt_opt, opt_cfg["ranges"], maximize_fn,
                    int(max_tries), constraint,
                )

            # ── BEST PARAMS ───────────────────────────────────────────────────
            st.subheader("Best Result")
            best_strategy = best_stats["_strategy"]
            best_params   = {k: getattr(best_strategy, k) for k in opt_cfg["ranges"]}
            bp_df = pd.DataFrame([{"Parameter": k, "Best Value": v} for k, v in best_params.items()])
            st.dataframe(bp_df, hide_index=True, use_container_width=False)

            st.divider()

            # ── BEST STATS ────────────────────────────────────────────────────
            st.subheader("Performance Summary — Best Config")
            stats_tables(best_stats)
            st.divider()

            # ── HEATMAP ───────────────────────────────────────────────────────
            hx = opt_cfg["heatmap_x"]
            hy = opt_cfg["heatmap_y"]
            st.subheader(f"Heatmap  —  {metric_label}  ({hx} vs {hy})")

            if heatmap is not None and not heatmap.empty:
                hm_df   = heatmap.reset_index()
                val_col = hm_df.columns[-1]
                hm_2d   = hm_df.groupby([hx, hy])[val_col].max().unstack(hy)
                hm_2d.index   = hm_2d.index.astype(str)
                hm_2d.columns = hm_2d.columns.astype(str)

                fig = px.imshow(
                    hm_2d,
                    labels=dict(x=hy, y=hx, color=metric_label),
                    color_continuous_scale="RdYlGn" if not minimize else "RdYlGn_r",
                    aspect="auto",
                    text_auto=".2f",
                )
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Heatmap data not available — try widening the parameter ranges.")

            # ── BEST CONFIG CHART ─────────────────────────────────────────────
            st.subheader("Strategy Chart — Best Config")
            bt_plot(bt_opt)

            with st.expander(f"Trade Log  ({len(best_stats['_trades'])} trades)"):
                trades = best_stats["_trades"]
                if not trades.empty:
                    display = trades[["EntryTime","ExitTime","Size","EntryPrice","ExitPrice","ReturnPct","PnL","Duration"]].copy()
                    display["ReturnPct"] = (display["ReturnPct"] * 100).round(2)
                    display["PnL"]       = display["PnL"].round(2)
                    display = display.rename(columns={"ReturnPct": "Return %"})
                    st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("Configure parameter ranges above and click Run Optimizer.")
