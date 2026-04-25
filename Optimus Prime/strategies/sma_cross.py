import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

NAME = "SMA Crossover"


def _sma(arr, n):
    return pd.Series(arr).rolling(n).mean().values


class SmaCross(Strategy):
    n1     = 10
    n2     = 20
    sl_pct = 0.05
    tp_pct = 0.05
    _size  = 0.95

    def init(self):
        self.sma1 = self.I(_sma, self.data.Close, self.n1)
        self.sma2 = self.I(_sma, self.data.Close, self.n2)

    def next(self):
        entry = self.data.Close[-1]
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy(size=self._size, sl=entry * (1 - self.sl_pct), tp=entry * (1 + self.tp_pct))
        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell(size=self._size, sl=entry * (1 + self.sl_pct), tp=entry * (1 - self.tp_pct))


def params_ui(st):
    c1, c2, c3, c4 = st.columns(4)
    n1     = int(c1.number_input("Fast MA (days)", min_value=2,  max_value=500, value=10,  step=1))
    n2     = int(c2.number_input("Slow MA (days)", min_value=3,  max_value=500, value=20,  step=1))
    sl_pct = c3.number_input("Stop Loss %",   min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    tp_pct = c4.number_input("Take Profit %", min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    return dict(n1=n1, n2=n2, sl_pct=sl_pct, tp_pct=tp_pct)


def optimize_params_ui(st):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    n1_min  = int(c1.number_input("Fast MA Min",  value=5,   min_value=2, key="on1min"))
    n1_max  = int(c2.number_input("Fast MA Max",  value=50,  min_value=3, key="on1max"))
    n1_step = int(c3.number_input("Fast MA Step", value=5,   min_value=1, key="on1stp"))
    n2_min  = int(c4.number_input("Slow MA Min",  value=20,  min_value=3, key="on2min"))
    n2_max  = int(c5.number_input("Slow MA Max",  value=200, min_value=4, key="on2max"))
    n2_step = int(c6.number_input("Slow MA Step", value=20,  min_value=1, key="on2stp"))

    d1, d2 = st.columns(2)
    sl_vals = d1.multiselect("Stop Loss % to test",   [1,2,3,4,5,6,8,10], default=[3,5,8])
    tp_vals = d2.multiselect("Take Profit % to test", [1,2,3,4,5,6,8,10], default=[3,5,8])

    n1_range = range(n1_min, n1_max + 1, n1_step)
    n2_range = range(n2_min, n2_max + 1, n2_step)
    sl_list  = [v / 100 for v in sl_vals] if sl_vals else [0.05]
    tp_list  = [v / 100 for v in tp_vals] if tp_vals else [0.05]

    combos = len(n1_range) * len(n2_range) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(n1=n1_range, n2=n2_range, sl_pct=sl_list, tp_pct=tp_list),
        "constraint": lambda p: p.n1 < p.n2,
        "heatmap_x":  "n1",
        "heatmap_y":  "n2",
    }


def build(params, size=0.95):
    SmaCross.n1     = params["n1"]
    SmaCross.n2     = params["n2"]
    SmaCross.sl_pct = params["sl_pct"]
    SmaCross.tp_pct = params["tp_pct"]
    SmaCross._size  = size
    return SmaCross
