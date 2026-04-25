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


def params_ui(st):
    period = st.slider("Period",        10,  50, 20)
    n_std  = st.slider("Std Dev",      1.0, 3.0, 2.0, step=0.1)
    mode   = st.selectbox("Mode", ["Breakout", "Mean Reversion"])
    sl_pct = st.slider("Stop Loss %",   1,  20,  5) / 100
    tp_pct = st.slider("Take Profit %", 1,  20,  5) / 100
    return dict(period=period, n_std=n_std, mode=mode, sl_pct=sl_pct, tp_pct=tp_pct)


def build(params, size=0.95):
    period   = params["period"]
    n_std    = params["n_std"]
    mode     = params["mode"]
    sl_pct   = params["sl_pct"]
    tp_pct   = params["tp_pct"]
    breakout = mode == "Breakout"

    class BollingerStrategy(Strategy):
        def init(self):
            close      = self.data.Close
            self.upper = self.I(_bb_upper, close, period, n_std)
            self.lower = self.I(_bb_lower, close, period, n_std)

        def next(self):
            entry = self.data.Close[-1]
            price = self.data.Close

            if breakout:
                # Buy when price breaks above upper, sell when breaks below lower
                if crossover(price, self.upper) and not self.position.is_long:
                    self.position.close()
                    self.buy(size=size, sl=entry * (1 - sl_pct), tp=entry * (1 + tp_pct))
                elif crossover(self.lower, price) and not self.position.is_short:
                    self.position.close()
                    self.sell(size=size, sl=entry * (1 + sl_pct), tp=entry * (1 - tp_pct))
            else:
                # Mean reversion: buy bounce off lower, sell fade off upper
                if crossover(price, self.lower) and not self.position.is_long:
                    self.position.close()
                    self.buy(size=size, sl=entry * (1 - sl_pct), tp=entry * (1 + tp_pct))
                elif crossover(self.upper, price) and not self.position.is_short:
                    self.position.close()
                    self.sell(size=size, sl=entry * (1 + sl_pct), tp=entry * (1 - tp_pct))

    return BollingerStrategy
