import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

NAME = "MACD"


def _ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _macd_line(arr, fast, slow):
    s = pd.Series(arr)
    return (s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()).values


def _signal_line(arr, fast, slow, signal):
    macd = pd.Series(_macd_line(arr, fast, slow))
    return macd.ewm(span=signal, adjust=False).mean().values


class MACDStrategy(Strategy):
    fast   = 12
    slow   = 26
    signal = 9
    sl_pct = 0.05
    tp_pct = 0.05
    _size  = 0.95

    def init(self):
        close        = self.data.Close
        self.macd    = self.I(_macd_line,  close, self.fast, self.slow)
        self.sig     = self.I(_signal_line, close, self.fast, self.slow, self.signal)

    def next(self):
        entry = self.data.Close[-1]
        if crossover(self.macd, self.sig) and not self.position.is_long:
            self.position.close()
            self.buy(size=self._size, sl=entry * (1 - self.sl_pct), tp=entry * (1 + self.tp_pct))
        elif crossover(self.sig, self.macd) and not self.position.is_short:
            self.position.close()
            self.sell(size=self._size, sl=entry * (1 + self.sl_pct), tp=entry * (1 - self.tp_pct))


def params_ui(st):
    c1, c2, c3, c4, c5 = st.columns(5)
    fast   = int(c1.number_input("Fast EMA",    min_value=2,  max_value=100, value=12, step=1))
    slow   = int(c2.number_input("Slow EMA",    min_value=3,  max_value=300, value=26, step=1))
    signal = int(c3.number_input("Signal",      min_value=2,  max_value=100, value=9,  step=1))
    sl_pct = c4.number_input("Stop Loss %",  min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    tp_pct = c5.number_input("Take Profit %", min_value=0.1, max_value=50.0, value=5.0, step=0.5) / 100
    return dict(fast=fast, slow=slow, signal=signal, sl_pct=sl_pct, tp_pct=tp_pct)


def optimize_params_ui(st):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    fast_min  = int(c1.number_input("Fast Min",   value=8,  min_value=2, key="mfmin"))
    fast_max  = int(c2.number_input("Fast Max",   value=20, min_value=3, key="mfmax"))
    fast_step = int(c3.number_input("Fast Step",  value=4,  min_value=1, key="mfstp"))
    slow_min  = int(c4.number_input("Slow Min",   value=20, min_value=3, key="msmin"))
    slow_max  = int(c5.number_input("Slow Max",   value=40, min_value=4, key="msmax"))
    slow_step = int(c6.number_input("Slow Step",  value=5,  min_value=1, key="msstp"))

    d1, d2, d3, d4 = st.columns(4)
    sig_vals = d1.multiselect("Signal",              [5, 7, 9, 12, 15], default=[7, 9, 12])
    sl_vals  = d2.multiselect("Stop Loss % to test", [1,2,3,4,5,6,8,10], default=[3,5,8])
    tp_vals  = d3.multiselect("Take Profit % to test",[1,2,3,4,5,6,8,10], default=[3,5,8])

    fast_range = range(fast_min, fast_max + 1, fast_step)
    slow_range = range(slow_min, slow_max + 1, slow_step)
    sig_list   = sig_vals if sig_vals else [9]
    sl_list    = [v / 100 for v in sl_vals] if sl_vals else [0.05]
    tp_list    = [v / 100 for v in tp_vals] if tp_vals else [0.05]

    combos = len(fast_range) * len(slow_range) * len(sig_list) * len(sl_list) * len(tp_list)
    st.caption(f"Grid size: {combos:,} combinations")

    return {
        "ranges":     dict(fast=fast_range, slow=slow_range, signal=sig_list,
                           sl_pct=sl_list, tp_pct=tp_list),
        "constraint": lambda p: p.fast < p.slow,
        "heatmap_x":  "fast",
        "heatmap_y":  "slow",
    }


def build(params, size=0.95):
    MACDStrategy.fast   = params["fast"]
    MACDStrategy.slow   = params["slow"]
    MACDStrategy.signal = params["signal"]
    MACDStrategy.sl_pct = params["sl_pct"]
    MACDStrategy.tp_pct = params["tp_pct"]
    MACDStrategy._size  = size
    return MACDStrategy
