import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import backtrader as bt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class BaseStrategy(bt.Strategy):
    params = (
        ('order_percentage', 0.95),
        ('ticker1', 'SPY'),
        ('ticker2', ''),
    )

    def __init__(self):
        self.order = None
        self.trades = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            trade_info = {
                'datetime': self.datas[0].datetime.date(0),
                'type': 'buy' if order.isbuy() else 'sell',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'net_value': order.executed.value + order.executed.comm if order.isbuy()
                             else order.executed.value - order.executed.comm
            }
            self.trades.append(trade_info)
        self.order = None

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


# --- Pairs Trading Strategy ---
class PairsTradingStrategy(BaseStrategy):
    params = (
        ('fast', 20),
        ('slow', 60),
        ('zscore_high', 2.0),
        ('zscore_low', -2.0),
        ('ticker1', '00700.HK'),  # H-share
        ('ticker2', '600519.SS'),  # A-share
        ('use_regime', True)
    )

    def __init__(self):
        super().__init__()
        self.data1 = self.datas[0]
        self.data2 = self.datas[1]

        self.spread = np.log(1+bt.indicators.PctChange(self.data1.close)) - np.log(1+bt.indicators.PctChange(self.data2.close))
        self.zscore = (self.spread - bt.indicators.SMA(self.spread, period=self.p.slow)) / \
                      bt.indicators.StdDev(self.spread, period=self.p.fast)

        self.use_regime = self.p.use_regime
        self.regimes = []

    def next(self):
        if self.order:
            return

        z = self.zscore[0]
        date = self.datas[0].datetime.date(0)

        # Markov regime detection
        if self.use_regime and len(self.spread.get(size=100)) == 100:
            df_spread = pd.Series([self.spread[i] for i in range(-99, 1)])
            try:
                model = MarkovRegression(df_spread, k_regimes=2, trend='c', switching_trend=True)
                result = model.fit(disp=False)
                probs = result.filtered_marginal_probabilities
                current_state = probs.idxmax(axis=1).iloc[-1]
                self.regimes.append((date, current_state))
                if current_state == 1:
                    return  # Skip high volatility regime
            except Exception as e:
                pass  # Default to normal behavior

        # Trade logic
        if not self.position:
            if z < self.p.zscore_low:
                size1 = int(self.broker.cash * self.p.order_percentage / self.data1.close)
                size2 = int(self.broker.cash * self.p.order_percentage / self.data2.close)
                self.sell(data=self.data2, size=size2)
                self.buy(data=self.data1, size=size1)
                self.log(f"LONG TRADE ENTERED, Spread Z-Score: {z:.2f}")
        else:
            if abs(z) < 0.5:
                self.close(data=self.data1)
                self.close(data=self.data2)
                self.log(f"POSITION CLOSED, Spread Z-Score: {z:.2f}")


def get_stock_data(ticker, start_date, end_date, demo_mode=False):
    try:
        if demo_mode:
            return get_demo_data(ticker)
        else:
            df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'datetime'}, inplace=True)
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            return df
    except Exception as e:
        st.error(f"Failed to download data for {ticker}: {str(e)}")
        return None


def get_demo_data(ticker):
    dates = pd.date_range(start=datetime(2020, 1, 1), end=datetime(2021, 12, 31), freq='B')
    opens = [300 + i*0.5 for i in range(len(dates))]
    closes = [o + np.random.normal(0, 4) for o in opens]
    highs = [max(o, c) + abs(np.random.normal(0, 1)) for o, c in zip(opens, closes)]
    lows = [min(o, c) - abs(np.random.normal(0, 1)) for o, c in zip(opens, closes)]
    volumes = [int(np.random.uniform(1e6, 2e6)) for _ in dates]
    return pd.DataFrame({
        'datetime': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def run_backtest(strategy_class, ticker1, ticker2, start_date, end_date, demo_mode=False, **params):
    cerebro = bt.Cerebro()
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    df1 = get_stock_data(ticker1, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), demo_mode)
    df2 = get_stock_data(ticker2, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), demo_mode)

    if df1 is None or df2 is None or len(df1) == 0 or len(df2) == 0:
        st.error("Missing data for one or both tickers.")
        return None, None

    data1 = bt.feeds.PandasData(dataname=df1, datetime='datetime', open='open', high='high', low='low', close='close', volume='volume')
    data2 = bt.feeds.PandasData(dataname=df2, datetime='datetime', open='open', high='high', low='low', close='close', volume='volume')

    cerebro.adddata(data1)
    cerebro.adddata(data2)

    cerebro.broker.setcash(100000.0)
    cerebro.addstrategy(strategy_class, **params)

    results = cerebro.run()
    strat = results[0]

    # Generate plot
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    spread_series = np.array(strat.spread.array)
    zscores = np.array(strat.zscore.array)

    ax[0].plot(spread_series, label="Spread", color='blue')
    ax[0].axhline(np.mean(spread_series), color='gray', linestyle='--')
    ax[0].set_title("Spread Between Assets")
    ax[0].grid(True)

    ax[1].plot(zscores, label="Z-Score", color='green')
    ax[1].axhline(strat.params.zscore_high, color='red', linestyle='--', label='+2σ Threshold')
    ax[1].axhline(strat.params.zscore_low, color='red', linestyle='--', label='-2σ Threshold')
    ax[1].legend()
    ax[1].set_title("Z-Score of Spread")
    ax[1].grid(True)

    plt.tight_layout()

    # ADF Test
    adf_result = adfuller(spread_series)
    st.write(f"Augmented Dickey-Fuller p-value: {adf_result[1]:.4f} → {'Mean-Reverting' if adf_result[1] < 0.05 else 'Not Mean-Reverting'}")

    if hasattr(strat, 'regimes') and strat.use_regime:
        st.write("Detected Regime Changes:")
        for d, r in strat.regimes[-10:]:
            st.write(f"{d}: Regime {r}")

    return fig, None


def main():
    st.set_page_config(page_title="Pairs Trading Backtester (A/H Shares)", layout="wide")
    st.title("Statistical Arbitrage Pairs Trading Backtester (China A/H Shares)")

    st.sidebar.header("Strategy Configuration")
    selected_strategy_name = "PairsTradingStrategy"
    selected_strategy_class = PairsTradingStrategy
    demo_mode = st.sidebar.checkbox('Demo Mode (Use Static Data)', False)

    strategy_params = {}
    st.sidebar.subheader('Strategy Parameters')
    use_regime = st.sidebar.checkbox('Enable Markov Regime Detection', value=True)
    fast_ma = st.sidebar.number_input("Short-Term Moving Avg (Fast)", value=20, step=1)
    slow_ma = st.sidebar.number_input("Long-Term Moving Avg (Slow)", value=60, step=1)
    z_up = st.sidebar.number_input("Z-Score Upper Bound", value=2.0, step=0.1)
    z_down = st.sidebar.number_input("Z-Score Lower Bound", value=-2.0, step=0.1)

    strategy_params.update({
        'fast': fast_ma,
        'slow': slow_ma,
        'zscore_high': z_up,
        'zscore_low': z_down,
        'use_regime': use_regime
    })

    st.sidebar.header("Backtest Configuration")
    ticker1 = st.sidebar.text_input("Ticker 1 (H-share, e.g., 0700.HK)", "0700.HK")
    ticker2 = st.sidebar.text_input("Ticker 2 (A-share, e.g., 600519.SS)", "600519.SS")

    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    if st.sidebar.button("Run Backtest"):
        st.info(f"Running pairs trading backtest between {ticker1} and {ticker2} from {start_date} to {end_date}")
        progress_bar = st.progress(0)
        try:
            fig, _ = run_backtest(
                selected_strategy_class,
                ticker1=ticker1,
                ticker2=ticker2,
                start_date=start_date,
                end_date=end_date,
                demo_mode=demo_mode,
                **strategy_params
            )
            st.pyplot(fig)
            st.success("Backtest completed!")
        except Exception as e:
            st.error(f"Error during backtest: {e}")
        progress_bar.progress(100)

if __name__ == "__main__":
    main()
