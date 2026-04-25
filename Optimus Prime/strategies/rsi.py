import numpy as np
import pandas as pd
from backtesting import Strategy

NAME = "RSI"


def _rsi(arr, period):
    s     = pd.Series(arr)
    delta = s.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    period     = int(c1.number_input("RSI Period",       min_value=2,  max_value=200, value=14, step=1))
    oversold   = int(c2.number_input("Oversold",         min_value=5,  max_value=49,  value=30, step=1))
    overbought = int(c3.number_input("Overbought",       min_value=51, max_value=95,  value=70, step=1))
    sl_pct     = c4.number_input("Stop Loss %",  min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    tp_pct     = c5.number_input("Take Profit %", min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    return dict(period=period, oversold=oversold, overbought=overbought,
                sl_pct=sl_pct, tp_pct=tp_pct)


def optimize_params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    p_min   = int(c1.number_input("Period Min",  value=5,  min_value=2, key="opmin"))
    p_max   = int(c2.number_input("Period Max",  value=30, min_value=3, key="opmax"))
    p_step  = int(c3.number_input("Period Step", value=5,  min_value=1, key="opstp"))
    os_vals = c4.multiselect("Oversold",   [15,20,25,30,35,40], default=[20,25,30])
    ob_vals = c5.multiselect("Overbought", [60,65,70,75,80,85], default=[65,70,75])

    d1, d2 = st.columns(2)
    sl_vals = d1.multiselect("Stop Loss % to test",   [1,2,3,4,5,6,8,10], default=[3,5,8])
    tp_vals = d2.multiselect("Take Profit % to test", [1,2,3,4,5,6,8,10], default=[3,5,8])

    p_range = range(p_min, p_max + 1, p_step)
    os_list = os_vals if os_vals else [30]
    ob_list = ob_vals if ob_vals else [70]
    sl_list = [v / 100 for v in sl_vals] if sl_vals else [0.05]
    tp_list = [v / 100 for v in tp_vals] if tp_vals else [0.05]

    combos = len(p_range) * len(os_list) * len(ob_list) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(period=p_range, oversold=os_list, overbought=ob_list,
                           sl_pct=sl_list, tp_pct=tp_list),
        "constraint": lambda p: p.oversold < p.overbought,
        "heatmap_x":  "oversold",
        "heatmap_y":  "overbought",
    }


def build(params, size=0.95):
    _size = size

    class RSIStrategy(Strategy):
        period     = params["period"]
        oversold   = params["oversold"]
        overbought = params["overbought"]
        sl_pct     = params["sl_pct"]
        tp_pct     = params["tp_pct"]

        def init(self):
            self.rsi = self.I(_rsi, self.data.Close, self.period)

        def next(self):
            entry = self.data.Close[-1]
            if self.rsi[-1] < self.oversold and not self.position.is_long:
                self.position.close()
                self.buy(size=_size, sl=entry * (1 - self.sl_pct), tp=entry * (1 + self.tp_pct))
            elif self.rsi[-1] > self.overbought and not self.position.is_short:
                self.position.close()
                self.sell(size=_size, sl=entry * (1 + self.sl_pct), tp=entry * (1 - self.tp_pct))

    return RSIStrategy
