import importlib.util
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
import streamlit as st
from backtesting import Backtest
from pathlib import Path

st.set_page_config(page_title="Optimus Prime", page_icon="⚡", layout="wide")

APP_DIR = Path(__file__).parent
DB_DIR  = APP_DIR.parent / "Database"

# ── STRATEGY DISCOVERY ────────────────────────────────────────────────────────
@st.cache_resource
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

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## ⚡ Optimus Prime")
st.caption("Roll-adjusted continuous futures backtester  |  Source: ICE Connect via ICEBREAKER-Rollex")

if not COMMODITIES:
    st.error("No commodities with OHLC data found. Run `rollex_builder --full` first.")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    commodity     = st.selectbox("Commodity", COMMODITIES)
    strategy_name = st.selectbox("Strategy",  list(STRATEGIES.keys()))

    st.divider()

    mod = STRATEGIES[strategy_name]
    st.subheader(f"Strategy Parameters")
    params = mod.params_ui(st)

    st.divider()
    st.subheader("Engine")
    cash       = st.number_input("Starting Cash ($)", value=10_000, step=1_000, min_value=1_000)
    commission = st.number_input("Commission", value=0.002, step=0.001, format="%.3f", min_value=0.0)
    size       = st.slider("Trade Size (fraction of equity)", 0.1, 1.0, 0.95, step=0.05)

    st.divider()
    run = st.button("▶  Run Backtest", type="primary", use_container_width=True)

# ── LOAD + PRICE CHART ────────────────────────────────────────────────────────
df = load_data(commodity)

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=df.index, y=df["Close"],
    line=dict(color="#4A90D9", width=1.5),
    name="Close (Rollex adjusted)",
    hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
))
fig_price.update_layout(
    title=dict(text=f"{commodity} — Continuous Roll-Adjusted Close", font=dict(size=14)),
    height=280,
    margin=dict(l=0, r=0, t=40, b=0),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="white"),
    xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
    yaxis=dict(showgrid=True, gridcolor="#1e2130"),
    hovermode="x unified",
)
st.plotly_chart(fig_price, use_container_width=True)

st.caption(f"Data: {df.index.min().date()} → {df.index.max().date()}  |  {len(df):,} trading days")

# ── RUN BACKTEST ──────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running backtest..."):
        StrategyClass = mod.build(params, size=size)
        bt    = Backtest(df, StrategyClass, cash=cash, commission=commission, exclusive_orders=True)
        stats = bt.run()

    trades = stats["_trades"]
    eq     = stats["_equity_curve"]["Equity"]

    st.divider()

    # ── KEY METRICS ───────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ret    = stats["Return [%]"]
    bh_ret = stats["Buy & Hold Return [%]"]
    c1.metric("Total Return",   f"{ret:.1f}%",
              delta=f"B&H {bh_ret:.1f}%",
              delta_color="normal")
    c2.metric("Ann. Return",    f"{stats['Return (Ann.) [%]']:.1f}%")
    c3.metric("Max Drawdown",   f"{stats['Max. Drawdown [%]']:.1f}%")
    c4.metric("Sharpe Ratio",   f"{stats['Sharpe Ratio']:.2f}")
    c5.metric("Win Rate",       f"{stats['Win Rate [%]']:.1f}%")
    c6.metric("# Trades",       int(stats["# Trades"]))

    # ── SECONDARY METRICS ─────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Profit Factor",  f"{stats['Profit Factor']:.2f}")
    s2.metric("Expectancy",     f"{stats['Expectancy [%]']:.2f}%")
    s3.metric("Best Trade",     f"{stats['Best Trade [%]']:.1f}%")
    s4.metric("Worst Trade",    f"{stats['Worst Trade [%]']:.1f}%")

    st.divider()

    # ── EQUITY CURVE ──────────────────────────────────────────────────────────
    bh = df["Close"].reindex(eq.index) / df["Close"].iloc[0] * cash

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name="Strategy",
        line=dict(color="#00d4aa", width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.05)",
        hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra>Strategy</extra>",
    ))
    fig_eq.add_trace(go.Scatter(
        x=bh.index, y=bh.values,
        name="Buy & Hold",
        line=dict(color="#888", width=1.5, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>$%{y:,.0f}<extra>Buy & Hold</extra>",
    ))
    fig_eq.update_layout(
        title="Equity Curve",
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#1e2130", tickprefix="$"),
        legend=dict(orientation="h", y=1.08),
        hovermode="x unified",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── TRADE LOG ─────────────────────────────────────────────────────────────
    if not trades.empty:
        st.subheader("Trade Log")
        display = trades[[
            "EntryTime", "ExitTime", "Size",
            "EntryPrice", "ExitPrice",
            "ReturnPct", "PnL", "Duration"
        ]].copy()
        display["ReturnPct"] = (display["ReturnPct"] * 100).round(2)
        display["PnL"]       = display["PnL"].round(2)
        display = display.rename(columns={"ReturnPct": "Return %"})
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── FULL STATS ────────────────────────────────────────────────────────────
    with st.expander("Full Stats"):
        drop_cols = ["_equity_curve", "_trades", "_strategy", "_broker"]
        stat_df   = pd.Series({
            k: v for k, v in stats.items() if k not in drop_cols
        }).to_frame("Value")
        st.dataframe(stat_df, use_container_width=True)

else:
    st.info("Configure your strategy in the sidebar and click **▶ Run Backtest**.")
