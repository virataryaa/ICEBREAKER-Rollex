import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover

NAME = "SMA Crossover"


def _sma(arr, n):
    return pd.Series(arr).rolling(n).mean().values


def params_ui(st):
    n1     = st.slider("Fast MA (days)", 5, 100, 10)
    n2     = st.slider("Slow MA (days)", 10, 300, 20)
    sl_pct = st.slider("Stop Loss %",   1, 20, 5) / 100
    tp_pct = st.slider("Take Profit %", 1, 20, 5) / 100
    return dict(n1=n1, n2=n2, sl_pct=sl_pct, tp_pct=tp_pct)


def build(params, size=0.95):
    n1     = params["n1"]
    n2     = params["n2"]
    sl_pct = params["sl_pct"]
    tp_pct = params["tp_pct"]

    class SmaCross(Strategy):
        def init(self):
            close     = self.data.Close
            self.sma1 = self.I(_sma, close, n1)
            self.sma2 = self.I(_sma, close, n2)

        def next(self):
            entry = self.data.Close[-1]
            if crossover(self.sma1, self.sma2):
                self.position.close()
                self.buy(size=size, sl=entry * (1 - sl_pct), tp=entry * (1 + tp_pct))
            elif crossover(self.sma2, self.sma1):
                self.position.close()
                self.sell(size=size, sl=entry * (1 + sl_pct), tp=entry * (1 - tp_pct))

    return SmaCross
