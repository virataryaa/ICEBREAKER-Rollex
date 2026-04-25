import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

NAME = "Bollinger Bands"


def _bb_upper(arr, period, n_std):
    s = pd.Series(arr)
    return (s.rolling(period).mean() + n_std * s.rolling(period).std()).values


def _bb_lower(arr, period, n_std):
    s = pd.Series(arr)
    return (s.rolling(period).mean() - n_std * s.rolling(period).std()).values


class BollingerStrategy(Strategy):
    period = 20
    n_std  = 2.0
    mode   = "Breakout"
    sl_pct = 0.05
    tp_pct = 0.05
    _size  = 0.95

    def init(self):
        close      = self.data.Close
        self.upper = self.I(_bb_upper, close, self.period, self.n_std)
        self.lower = self.I(_bb_lower, close, self.period, self.n_std)

    def next(self):
        entry    = self.data.Close[-1]
        price    = self.data.Close
        breakout = self.mode == "Breakout"

        if breakout:
            if crossover(price, self.upper) and not self.position.is_long:
                self.position.close()
                self.buy(size=self._size, sl=entry * (1 - self.sl_pct), tp=entry * (1 + self.tp_pct))
            elif crossover(self.lower, price) and not self.position.is_short:
                self.position.close()
                self.sell(size=self._size, sl=entry * (1 + self.sl_pct), tp=entry * (1 - self.tp_pct))
        else:
            if crossover(price, self.lower) and not self.position.is_long:
                self.position.close()
                self.buy(size=self._size, sl=entry * (1 - self.sl_pct), tp=entry * (1 + self.tp_pct))
            elif crossover(self.upper, price) and not self.position.is_short:
                self.position.close()
                self.sell(size=self._size, sl=entry * (1 + self.sl_pct), tp=entry * (1 - self.tp_pct))


def params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    period = int(c1.number_input("Period",  min_value=5,   max_value=200, value=20,  step=1))
    n_std  = c2.number_input("Std Dev",     min_value=0.5, max_value=5.0, value=2.0, step=0.1, format="%.1f")
    mode   = c3.selectbox("Mode", ["Breakout", "Mean Reversion"])
    sl_pct = c4.number_input("Stop Loss %",   min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    tp_pct = c5.number_input("Take Profit %", min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    return dict(period=period, n_std=n_std, mode=mode, sl_pct=sl_pct, tp_pct=tp_pct)


def optimize_params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    p_min      = int(c1.number_input("Period Min",  value=10, min_value=5, key="obpmin"))
    p_max      = int(c2.number_input("Period Max",  value=50, min_value=6, key="obpmax"))
    p_step     = int(c3.number_input("Period Step", value=10, min_value=1, key="obpstp"))
    n_std_vals = c4.multiselect("Std Dev",          [1.0,1.5,2.0,2.5,3.0], default=[1.5,2.0,2.5])
    mode_vals  = c5.multiselect("Mode",             ["Breakout","Mean Reversion"], default=["Breakout","Mean Reversion"])

    d1, d2 = st.columns(2)
    sl_vals = d1.multiselect("Stop Loss % to test",   [1,2,3,4,5,6,8,10], default=[3,5,8])
    tp_vals = d2.multiselect("Take Profit % to test", [1,2,3,4,5,6,8,10], default=[3,5,8])

    p_range   = range(p_min, p_max + 1, p_step)
    std_list  = n_std_vals if n_std_vals else [2.0]
    mode_list = mode_vals  if mode_vals  else ["Breakout"]
    sl_list   = [v / 100 for v in sl_vals] if sl_vals else [0.05]
    tp_list   = [v / 100 for v in tp_vals] if tp_vals else [0.05]

    combos = len(p_range) * len(std_list) * len(mode_list) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(period=p_range, n_std=std_list, mode=mode_list,
                           sl_pct=sl_list, tp_pct=tp_list),
        "constraint": None,
        "heatmap_x":  "period",
        "heatmap_y":  "n_std",
    }


def build(params, size=0.95):
    BollingerStrategy.period = params["period"]
    BollingerStrategy.n_std  = params["n_std"]
    BollingerStrategy.mode   = params["mode"]
    BollingerStrategy.sl_pct = params["sl_pct"]
    BollingerStrategy.tp_pct = params["tp_pct"]
    BollingerStrategy._size  = size
    return BollingerStrategy
