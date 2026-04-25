import numpy as np
import pandas as pd
from backtesting import Strategy

NAME = "Z-Score"


def _zscore(arr, period):
    s    = pd.Series(arr)
    mean = s.rolling(period).mean()
    std  = s.rolling(period).std()
    return ((s - mean) / std.replace(0, np.nan)).values


class ZScoreStrategy(Strategy):
    period    = 20
    entry_z   = 2.0
    exit_z    = 0.5
    sl_pct    = 0.05
    tp_pct    = 0.05
    _size     = 0.95

    def init(self):
        self.z = self.I(_zscore, self.data.Close, self.period)

    def next(self):
        entry = self.data.Close[-1]
        z     = self.z[-1]

        if np.isnan(z):
            return

        # Mean reversion: buy when deeply oversold, sell when deeply overbought
        if z < -self.entry_z and not self.position.is_long:
            self.position.close()
            self.buy(size=self._size, sl=entry * (1 - self.sl_pct), tp=entry * (1 + self.tp_pct))
        elif z > self.entry_z and not self.position.is_short:
            self.position.close()
            self.sell(size=self._size, sl=entry * (1 + self.sl_pct), tp=entry * (1 - self.tp_pct))
        elif self.position.is_long and z >= self.exit_z:
            self.position.close()
        elif self.position.is_short and z <= -self.exit_z:
            self.position.close()


def params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    period  = int(c1.number_input("Lookback Period", min_value=5,  max_value=300, value=20,  step=1))
    entry_z = c2.number_input("Entry Z",  min_value=0.5, max_value=5.0, value=2.0, step=0.1, format="%.1f")
    exit_z  = c3.number_input("Exit Z",   min_value=0.0, max_value=3.0, value=0.5, step=0.1, format="%.1f")
    sl_pct  = c4.number_input("Stop Loss %",   min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    tp_pct  = c5.number_input("Take Profit %", min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    return dict(period=period, entry_z=entry_z, exit_z=exit_z, sl_pct=sl_pct, tp_pct=tp_pct)


def optimize_params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    p_min      = int(c1.number_input("Period Min",   value=10, min_value=5, key="zpmin"))
    p_max      = int(c2.number_input("Period Max",   value=60, min_value=6, key="zpmax"))
    p_step     = int(c3.number_input("Period Step",  value=10, min_value=1, key="zpstp"))
    ez_vals    = c4.multiselect("Entry Z", [1.0, 1.5, 2.0, 2.5, 3.0], default=[1.5, 2.0, 2.5])
    xz_vals    = c5.multiselect("Exit Z",  [0.0, 0.25, 0.5, 0.75, 1.0], default=[0.25, 0.5, 0.75])

    d1, d2 = st.columns(2)
    sl_vals = d1.multiselect("Stop Loss % to test",   [1,2,3,4,5,6,8,10], default=[3,5,8])
    tp_vals = d2.multiselect("Take Profit % to test", [1,2,3,4,5,6,8,10], default=[3,5,8])

    p_range = range(p_min, p_max + 1, p_step)
    ez_list = ez_vals if ez_vals else [2.0]
    xz_list = xz_vals if xz_vals else [0.5]
    sl_list = [v / 100 for v in sl_vals] if sl_vals else [0.05]
    tp_list = [v / 100 for v in tp_vals] if tp_vals else [0.05]

    combos = len(p_range) * len(ez_list) * len(xz_list) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(period=p_range, entry_z=ez_list, exit_z=xz_list,
                           sl_pct=sl_list, tp_pct=tp_list),
        "constraint": lambda p: p.exit_z < p.entry_z,
        "heatmap_x":  "period",
        "heatmap_y":  "entry_z",
    }


def build(params, size=0.95):
    ZScoreStrategy.period  = params["period"]
    ZScoreStrategy.entry_z = params["entry_z"]
    ZScoreStrategy.exit_z  = params["exit_z"]
    ZScoreStrategy.sl_pct  = params["sl_pct"]
    ZScoreStrategy.tp_pct  = params["tp_pct"]
    ZScoreStrategy._size   = size
    return ZScoreStrategy
