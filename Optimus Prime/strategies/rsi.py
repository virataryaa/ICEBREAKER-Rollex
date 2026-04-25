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
    period     = st.slider("RSI Period",       5,  50, 14)
    oversold   = st.slider("Oversold Level",  10,  45, 30)
    overbought = st.slider("Overbought Level", 55,  90, 70)
    sl_pct     = st.slider("Stop Loss %",      1,  20,  5) / 100
    tp_pct     = st.slider("Take Profit %",    1,  20,  5) / 100
    return dict(period=period, oversold=oversold, overbought=overbought,
                sl_pct=sl_pct, tp_pct=tp_pct)


def build(params, size=0.95):
    period     = params["period"]
    oversold   = params["oversold"]
    overbought = params["overbought"]
    sl_pct     = params["sl_pct"]
    tp_pct     = params["tp_pct"]

    class RSIStrategy(Strategy):
        def init(self):
            self.rsi = self.I(_rsi, self.data.Close, period)

        def next(self):
            entry = self.data.Close[-1]
            if self.rsi[-1] < oversold and not self.position.is_long:
                self.position.close()
                self.buy(size=size, sl=entry * (1 - sl_pct), tp=entry * (1 + tp_pct))
            elif self.rsi[-1] > overbought and not self.position.is_short:
                self.position.close()
                self.sell(size=size, sl=entry * (1 + sl_pct), tp=entry * (1 - tp_pct))

    return RSIStrategy
