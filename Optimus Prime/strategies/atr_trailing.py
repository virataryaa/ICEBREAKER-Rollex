import numpy as np
import pandas as pd
from backtesting import Strategy

NAME = "ATR Trailing Stop"


def _atr(high, low, close, period):
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().values


def _chandelier_long(close, high, low, period, atr_period, mult):
    atr     = pd.Series(_atr(high, low, close, atr_period))
    highest = pd.Series(close).rolling(period).max()
    return (highest - mult * atr).values


def _chandelier_short(close, high, low, period, atr_period, mult):
    atr    = pd.Series(_atr(high, low, close, atr_period))
    lowest = pd.Series(close).rolling(period).min()
    return (lowest + mult * atr).values


class ATRTrailingStop(Strategy):
    period     = 22
    atr_period = 22
    mult       = 3.0
    sl_pct     = 0.0   # 0 = disabled; exit is purely the chandelier stop
    _size      = 0.95

    def init(self):
        c, h, l = self.data.Close, self.data.High, self.data.Low
        self.long_stop  = self.I(_chandelier_long,  c, h, l, self.period, self.atr_period, self.mult)
        self.short_stop = self.I(_chandelier_short, c, h, l, self.period, self.atr_period, self.mult)

    def next(self):
        entry      = self.data.Close[-1]
        long_stop  = self.long_stop[-1]
        short_stop = self.short_stop[-1]

        if np.isnan(long_stop) or np.isnan(short_stop):
            return

        # Exit if stop crossed
        if self.position.is_long and entry < long_stop:
            self.position.close()
        elif self.position.is_short and entry > short_stop:
            self.position.close()

        # Enter if flat
        if not self.position:
            if entry > short_stop:
                kw = dict(size=self._size)
                if self.sl_pct > 0:
                    kw["sl"] = entry * (1 - self.sl_pct)
                self.buy(**kw)
            elif entry < long_stop:
                kw = dict(size=self._size)
                if self.sl_pct > 0:
                    kw["sl"] = entry * (1 + self.sl_pct)
                self.sell(**kw)


def params_ui(st):
    c1, c2, c3, c4 = st.columns(4)
    period     = int(c1.number_input("Lookback Period",       min_value=5,   max_value=200, value=22,  step=1))
    atr_period = int(c2.number_input("ATR Period",            min_value=2,   max_value=100, value=22,  step=1))
    mult       = c3.number_input("ATR Multiplier",            min_value=0.5, max_value=8.0, value=3.0, step=0.25, format="%.2f")
    sl_pct     = c4.number_input("Hard Stop Loss % (0 = off)", min_value=0.0, max_value=50.0, value=0.0, step=0.5) / 100
    return dict(period=period, atr_period=atr_period, mult=mult, sl_pct=sl_pct)


def optimize_params_ui(st):
    c1, c2, c3 = st.columns(3)
    p_min  = int(c1.number_input("Period Min",  value=10, min_value=5, key="atpmin"))
    p_max  = int(c2.number_input("Period Max",  value=40, min_value=6, key="atpmax"))
    p_step = int(c3.number_input("Period Step", value=5,  min_value=1, key="atpstp"))

    d1, d2, d3 = st.columns(3)
    ap_vals = d1.multiselect("ATR Period",      [7, 10, 14, 20, 22],           default=[10, 14, 22])
    mt_vals = d2.multiselect("ATR Multiplier",  [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], default=[2.0, 3.0, 3.5])
    sl_vals = d3.multiselect("Hard SL % (0=off)", [0, 3, 5, 8, 10],            default=[0, 5])

    p_range = range(p_min, p_max + 1, p_step)
    ap_list = ap_vals if ap_vals else [22]
    mt_list = mt_vals if mt_vals else [3.0]
    sl_list = [v / 100 for v in sl_vals] if sl_vals else [0.0]

    combos = len(p_range) * len(ap_list) * len(mt_list) * len(sl_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(period=p_range, atr_period=ap_list, mult=mt_list, sl_pct=sl_list),
        "constraint": None,
        "heatmap_x":  "period",
        "heatmap_y":  "mult",
    }


def build(params, size=0.95):
    ATRTrailingStop.period     = params["period"]
    ATRTrailingStop.atr_period = params["atr_period"]
    ATRTrailingStop.mult       = params["mult"]
    ATRTrailingStop.sl_pct     = params["sl_pct"]
    ATRTrailingStop._size      = size
    return ATRTrailingStop
