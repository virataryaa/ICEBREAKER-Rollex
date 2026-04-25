import pandas as pd
from backtesting import Strategy

NAME = "Seasonal"

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",  5: "May",  6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _in_window(month, start, end):
    """True if month falls within [start, end], wrapping the year boundary."""
    if start == end:
        return month == start
    if start < end:
        return start <= month <= end
    return month >= start or month <= end   # e.g. Oct → Feb


class SeasonalStrategy(Strategy):
    long_entry_month  = 10   # October  — KC pre-Brazil-harvest seasonal long
    long_exit_month   = 2    # February
    short_entry_month = 5    # May      — Brazilian harvest pressure
    short_exit_month  = 8    # August
    sl_pct = 0.0             # 0 = disabled
    tp_pct = 0.0             # 0 = disabled
    _size  = 0.95

    def init(self):
        pass

    def next(self):
        month = self.data.index[-1].month
        entry = self.data.Close[-1]

        in_long  = _in_window(month, self.long_entry_month,  self.long_exit_month)
        in_short = _in_window(month, self.short_entry_month, self.short_exit_month)

        if in_long and not self.position.is_long:
            self.position.close()
            kw = dict(size=self._size)
            if self.sl_pct > 0: kw["sl"] = entry * (1 - self.sl_pct)
            if self.tp_pct > 0: kw["tp"] = entry * (1 + self.tp_pct)
            self.buy(**kw)
        elif in_short and not self.position.is_short:
            self.position.close()
            kw = dict(size=self._size)
            if self.sl_pct > 0: kw["sl"] = entry * (1 + self.sl_pct)
            if self.tp_pct > 0: kw["tp"] = entry * (1 - self.tp_pct)
            self.sell(**kw)
        elif not in_long and not in_short and self.position:
            self.position.close()


def params_ui(st):
    months = list(_MONTH_NAMES.keys())
    fmt    = lambda m: f"{m} – {_MONTH_NAMES[m]}"
    labels = [fmt(m) for m in months]
    l2n    = {fmt(m): m for m in months}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    le = c1.selectbox("Long Entry",   labels, index=9,  key="sle")   # Oct
    lx = c2.selectbox("Long Exit",    labels, index=1,  key="slx")   # Feb
    se = c3.selectbox("Short Entry",  labels, index=4,  key="sse")   # May
    sx = c4.selectbox("Short Exit",   labels, index=7,  key="ssx")   # Aug
    sl_pct = c5.number_input("Stop Loss % (0 = off)",   min_value=0.0, max_value=50.0, value=0.0, step=0.5) / 100
    tp_pct = c6.number_input("Take Profit % (0 = off)", min_value=0.0, max_value=50.0, value=0.0, step=0.5) / 100

    return dict(
        long_entry_month  = l2n[le],
        long_exit_month   = l2n[lx],
        short_entry_month = l2n[se],
        short_exit_month  = l2n[sx],
        sl_pct = sl_pct,
        tp_pct = tp_pct,
    )


def optimize_params_ui(st):
    months = list(_MONTH_NAMES.keys())
    fmt    = _MONTH_NAMES.__getitem__

    c1, c2, c3, c4 = st.columns(4)
    le_vals = c1.multiselect("Long Entry",  months, default=[9, 10, 11], format_func=fmt)
    lx_vals = c2.multiselect("Long Exit",   months, default=[1,  2,  3], format_func=fmt)
    se_vals = c3.multiselect("Short Entry", months, default=[4,  5,  6], format_func=fmt)
    sx_vals = c4.multiselect("Short Exit",  months, default=[7,  8,  9], format_func=fmt)

    d1, d2 = st.columns(2)
    sl_vals = d1.multiselect("Stop Loss % (0 = off)",   [0, 2, 3, 5, 8],    default=[0, 5])
    tp_vals = d2.multiselect("Take Profit % (0 = off)", [0, 5, 8, 10, 15],  default=[0])

    le_list = le_vals if le_vals else [10]
    lx_list = lx_vals if lx_vals else [2]
    se_list = se_vals if se_vals else [5]
    sx_list = sx_vals if sx_vals else [8]
    sl_list = [v / 100 for v in sl_vals] if sl_vals else [0.0]
    tp_list = [v / 100 for v in tp_vals] if tp_vals else [0.0]

    combos = len(le_list) * len(lx_list) * len(se_list) * len(sx_list) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(long_entry_month=le_list, long_exit_month=lx_list,
                           short_entry_month=se_list, short_exit_month=sx_list,
                           sl_pct=sl_list, tp_pct=tp_list),
        "constraint": None,
        "heatmap_x":  "long_entry_month",
        "heatmap_y":  "long_exit_month",
    }


def build(params, size=0.95):
    SeasonalStrategy.long_entry_month  = params["long_entry_month"]
    SeasonalStrategy.long_exit_month   = params["long_exit_month"]
    SeasonalStrategy.short_entry_month = params["short_entry_month"]
    SeasonalStrategy.short_exit_month  = params["short_exit_month"]
    SeasonalStrategy.sl_pct = params["sl_pct"]
    SeasonalStrategy.tp_pct = params["tp_pct"]
    SeasonalStrategy._size  = size
    return SeasonalStrategy
