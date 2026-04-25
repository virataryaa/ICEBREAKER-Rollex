import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

NAME = "Keltner Channel"


def _ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _atr(high, low, close, period):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().values


def _kc_upper(close, high, low, ema_period, atr_period, mult):
    mid = pd.Series(_ema(close, ema_period))
    atr = pd.Series(_atr(high, low, close, atr_period))
    return (mid + mult * atr).values


def _kc_lower(close, high, low, ema_period, atr_period, mult):
    mid = pd.Series(_ema(close, ema_period))
    atr = pd.Series(_atr(high, low, close, atr_period))
    return (mid - mult * atr).values


class KeltnerStrategy(Strategy):
    ema_period = 20
    atr_period = 10
    mult       = 2.0
    mode       = "Breakout"
    sl_pct     = 0.05
    tp_pct     = 0.05
    _size      = 0.95

    def init(self):
        c, h, l   = self.data.Close, self.data.High, self.data.Low
        self.upper = self.I(_kc_upper, c, h, l, self.ema_period, self.atr_period, self.mult)
        self.lower = self.I(_kc_lower, c, h, l, self.ema_period, self.atr_period, self.mult)

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
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    ema_period = int(c1.number_input("EMA Period", min_value=5,  max_value=200, value=20,  step=1))
    atr_period = int(c2.number_input("ATR Period", min_value=2,  max_value=100, value=10,  step=1))
    mult       = c3.number_input("ATR Mult",   min_value=0.5, max_value=5.0,  value=2.0, step=0.25, format="%.2f")
    mode       = c4.selectbox("Mode", ["Breakout", "Mean Reversion"])
    sl_pct     = c5.number_input("Stop Loss %",   min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    tp_pct     = c6.number_input("Take Profit %", min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    return dict(ema_period=ema_period, atr_period=atr_period, mult=mult,
                mode=mode, sl_pct=sl_pct, tp_pct=tp_pct)


def optimize_params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    ep_min  = int(c1.number_input("EMA Min",    value=10, min_value=5, key="kepmin"))
    ep_max  = int(c2.number_input("EMA Max",    value=50, min_value=6, key="kepmax"))
    ep_step = int(c3.number_input("EMA Step",   value=10, min_value=1, key="kepstp"))
    ap_vals = c4.multiselect("ATR Period", [5, 7, 10, 14, 20], default=[7, 10, 14])
    mt_vals = c5.multiselect("ATR Mult",   [1.0, 1.5, 2.0, 2.5, 3.0], default=[1.5, 2.0, 2.5])

    d1, d2, d3 = st.columns(3)
    mode_vals = d1.multiselect("Mode", ["Breakout", "Mean Reversion"], default=["Breakout", "Mean Reversion"])
    sl_vals   = d2.multiselect("Stop Loss % to test",   [1,2,3,4,5,6,8,10], default=[3,5,8])
    tp_vals   = d3.multiselect("Take Profit % to test", [1,2,3,4,5,6,8,10], default=[3,5,8])

    ep_range  = range(ep_min, ep_max + 1, ep_step)
    ap_list   = ap_vals   if ap_vals   else [10]
    mt_list   = mt_vals   if mt_vals   else [2.0]
    mode_list = mode_vals if mode_vals else ["Breakout"]
    sl_list   = [v / 100 for v in sl_vals] if sl_vals else [0.05]
    tp_list   = [v / 100 for v in tp_vals] if tp_vals else [0.05]

    combos = len(ep_range) * len(ap_list) * len(mt_list) * len(mode_list) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(ema_period=ep_range, atr_period=ap_list, mult=mt_list,
                           mode=mode_list, sl_pct=sl_list, tp_pct=tp_list),
        "constraint": None,
        "heatmap_x":  "ema_period",
        "heatmap_y":  "mult",
    }


def build(params, size=0.95):
    KeltnerStrategy.ema_period = params["ema_period"]
    KeltnerStrategy.atr_period = params["atr_period"]
    KeltnerStrategy.mult       = params["mult"]
    KeltnerStrategy.mode       = params["mode"]
    KeltnerStrategy.sl_pct     = params["sl_pct"]
    KeltnerStrategy.tp_pct     = params["tp_pct"]
    KeltnerStrategy._size      = size
    return KeltnerStrategy
