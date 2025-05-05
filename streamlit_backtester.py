import streamlit as st
import pandas as pd
import numpy as np
import backtrader as bt
import yfinance as yf
import matplotlib

# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math
from scipy import stats


# Create a base strategy class that will be inherited by all other strategies
class BaseStrategy(bt.Strategy):
    """
    Base strategy class that implements common functionality for all strategies
    """
    params = (
        ('order_percentage', 0.95),  # Percentage of portfolio to allocate per trade
        ('ticker', 'SPY'),  # Ticker for the strategy
    )

    def __init__(self):
        # Common initialization
        self.order = None
        self.price = None
        self.comm = None

        # Trade tracking
        self.trades = []

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                trade_info = {
                    'datetime': self.datas[0].datetime.date(0),
                    'type': 'buy',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'value': order.executed.value,
                    'commission': order.executed.comm,
                    'net_value': order.executed.value + order.executed.comm
                }
                self.trades.append(trade_info)
            else:  # Sell
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                trade_info = {
                    'datetime': self.datas[0].datetime.date(0),
                    'type': 'sell',
                    'price': order.executed.price,
                    'size': order.executed.size,
                    'value': order.executed.value,
                    'commission': order.executed.comm,
                    'net_value': order.executed.value - order.executed.comm
                }
                self.trades.append(trade_info)

        # Reset the order attribute
        self.order = None

    def log(self, txt):
        """
        Logging function for debugging
        """
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


# Create strategies that inherit from the base strategy
class MovingAverageCrossStrategy(BaseStrategy):
    """
    A strategy that generates signals based on moving average crossovers
    """
    params = (
        ('fast', 10),  # Fast moving average period
        ('slow', 30),  # Slow moving average period
    )

    def __init__(self):
        # Call the parent class's init method
        super(MovingAverageCrossStrategy, self).__init__()

        # Initialize indicators
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow
        )

        # Cross signals
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        # Check if we already have an open order
        if self.order:
            return

        # Check if we are in a position
        if not self.position:
            # Not in a position, look for buy signal
            if self.crossover > 0:  # Fast MA crosses above slow MA
                # Calculate order size
                size = int(self.broker.cash * self.params.order_percentage / self.data.close)
                # Submit buy order
                self.order = self.buy(size=size)
                self.log(f'BUY ORDER CREATED, Size: {size}')
        else:
            # In a position, look for sell signal
            if self.crossover < 0:  # Fast MA crosses below slow MA
                # Submit sell order
                self.order = self.sell(size=self.position.size)
                self.log(f'SELL ORDER CREATED, Size: {self.position.size}')


class MomentumStrategy(BaseStrategy):
    """
    A strategy that buys stocks showing strong momentum and sells on momentum weakening
    """
    params = (
        ('lookback', 90),  # Lookback period for momentum calculation
        ('percentile', 80),  # Percentile threshold for entry (0-100)
        ('exit_percentile', 50),  # Percentile threshold for exit (0-100)
    )

    def __init__(self):
        # Call the parent class's init method
        super(MomentumStrategy, self).__init__()

        # Calculate momentum (rate of change over lookback period)
        self.momentum = bt.indicators.RateOfChange(self.data.close, period=self.params.lookback)

        # Historical momentum list
        self.historical_momentum = []

    def next(self):
        # Check if we already have an open order
        if self.order:
            return

        # Add current momentum to historical list
        if not math.isnan(self.momentum[0]):
            self.historical_momentum.append(self.momentum[0])

        # Wait until we have enough data
        if len(self.historical_momentum) < 30:  # Need some history for percentile calculation
            return

        # Calculate momentum percentile
        current_momentum = self.momentum[0]
        momentum_percentile = stats.percentileofscore(self.historical_momentum, current_momentum)

        # Check for entry/exit conditions
        if not self.position:
            # Not in a position, look for buy signal
            if momentum_percentile > self.params.percentile:
                # Calculate order size
                size = int(self.broker.cash * self.params.order_percentage / self.data.close)
                # Submit buy order
                self.order = self.buy(size=size)
                self.log(f'BUY ORDER CREATED, Size: {size}, Momentum Percentile: {momentum_percentile:.2f}')
        else:
            # In a position, look for sell signal
            if momentum_percentile < self.params.exit_percentile:
                # Submit sell order
                self.order = self.sell(size=self.position.size)
                self.log(
                    f'SELL ORDER CREATED, Size: {self.position.size}, Momentum Percentile: {momentum_percentile:.2f}')


class RSIStrategy(BaseStrategy):
    """
    A strategy that buys when RSI is oversold and sells when RSI is overbought
    """
    params = (
        ('period', 14),  # RSI calculation period
        ('overbought', 70),  # Overbought threshold
        ('oversold', 30),  # Oversold threshold
    )

    def __init__(self):
        # Call the parent class's init method
        super(RSIStrategy, self).__init__()

        # Calculate RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.period)

    def next(self):
        # Check if we already have an open order
        if self.order:
            return

        # Check if we are in a position
        if not self.position:
            # Not in a position, look for buy signal (oversold condition)
            if self.rsi[0] < self.params.oversold:
                # Calculate order size
                size = int(self.broker.cash * self.params.order_percentage / self.data.close)
                # Submit buy order
                self.order = self.buy(size=size)
                self.log(f'BUY ORDER CREATED, Size: {size}, RSI: {self.rsi[0]:.2f}')
        else:
            # In a position, look for sell signal (overbought condition)
            if self.rsi[0] > self.params.overbought:
                # Submit sell order
                self.order = self.sell(size=self.position.size)
                self.log(f'SELL ORDER CREATED, Size: {self.position.size}, RSI: {self.rsi[0]:.2f}')


# Add our strategies
strategies = {
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
    "MomentumStrategy": MomentumStrategy,
    "RSIStrategy": RSIStrategy
}


def get_stock_data(ticker, start_date, end_date, demo_mode=False):
    """
    Get historical stock data for backtesting

    Parameters:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        demo_mode: If True, load from static CSV instead of Yahoo Finance

    Returns:
        DataFrame: Historical price data with OHLCV format
    """
    try:
        if demo_mode:
            # Use static demo data
            df = get_demo_data(ticker)

            # Filter by date range
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[(df['datetime'] >= pd.Timestamp(start_date)) & (df['datetime'] <= pd.Timestamp(end_date))]

            if len(df) == 0:
                st.warning(f"No demo data available for the selected date range. Using all available demo data.")
                df = get_demo_data(ticker)  # Use all demo data

            return df
        else:
            # Get data from Yahoo Finance
            df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)

            # Rename columns to match backtrader's expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'datetime'})

            return df
    except Exception as e:
        st.error(f"Failed to download data for {ticker}: {str(e)}")
        return None


def get_demo_data(ticker):
    """
    Return static demo data for the given ticker

    Parameters:
        ticker: Stock ticker symbol (only SPY is supported in demo mode)

    Returns:
        DataFrame: Static historical price data
    """
    # Create a default demo dataset for SPY (normally you would load this from a file)
    # This is a simplified example, in production you would load from a saved CSV

    # For this example, we'll generate some synthetic data
    # In reality, you should save actual data from Yahoo Finance to a CSV and load it here

    # Generate some dates
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2021, 12, 31)

    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

    # Create synthetic price data with some trend and volatility
    np.random.seed(42)  # For reproducibility

    # Start price
    price = 300.0

    # Lists to store data
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    # Generate synthetic OHLCV data
    for i in range(len(dates)):
        # Daily volatility
        daily_vol = price * 0.01

        # Generate OHLC
        open_price = price
        high_price = open_price + abs(np.random.normal(0, daily_vol))
        low_price = open_price - abs(np.random.normal(0, daily_vol))
        close_price = np.random.normal((high_price + low_price) / 2, daily_vol / 2)

        # Ensure logical prices (high >= open, close, low and low <= open, close)
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Add some trend
        if i % 100 < 50:  # Uptrend for 50 days
            price = close_price * (1 + np.random.normal(0.0005, 0.001))  # Slight upward bias
        else:  # Downtrend for 50 days
            price = close_price * (1 + np.random.normal(-0.0005, 0.001))  # Slight downward bias

        # Volume (higher during big price moves)
        volume = int(np.random.normal(1000000, 200000) * (1 + abs((close_price - open_price) / open_price) * 10))

        # Append to lists
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(volume)

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

    return df


def run_backtest(strategy_class, ticker, start_date, end_date, demo_mode=False, **params):
    """
    Run a backtest for the specified strategy and parameters

    Parameters:
        strategy_class: The strategy class to backtest
        ticker: Stock ticker symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        demo_mode: If True, use static demo data
        **params: Strategy parameters

    Returns:
        tuple: (Matplotlib figure, DataFrame of trades)
    """
    # Initialize Cerebro engine
    cerebro = bt.Cerebro()

    # Set commission
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Get data
    df = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), demo_mode)

    if df is None or len(df) == 0:
        st.error(f"No data available for {ticker} during the specified period.")
        return None, None

    # Create a backtrader data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime='datetime',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1  # Not available in our data
    )

    # Add data to cerebro
    cerebro.adddata(data)

    # Add the strategy
    # Update the params with ticker
    params.update({'ticker': ticker})
    cerebro.addstrategy(strategy_class, **params)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Set initial cash
    cerebro.broker.setcash(100000.0)

    # Run the backtest
    results = cerebro.run()

    # Get the strategy instance
    strat = results[0]

    # Extract results
    returns = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()

    # Convert returns to DataFrame for visualization
    returns_df = pd.DataFrame(list(returns.items()), columns=['Date', 'Return'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    returns_df.set_index('Date', inplace=True)

    # Calculate cumulative returns
    returns_df['Cumulative'] = (1 + returns_df['Return']).cumprod()

    # Extract trade data
    trades_df = pd.DataFrame(strat.trades) if hasattr(strat, 'trades') and strat.trades else pd.DataFrame()

    # Plot results with dynamic figure size
    fig = plt.figure(figsize=(10, 8))  # Default figsize, will be resized via Streamlit

    # Plot equity curve
    ax1 = fig.add_subplot(211)
    ax1.plot(returns_df.index, returns_df['Cumulative'], label='Equity Curve')
    if demo_mode:
        ax1.set_title(f'Backtest Results (DEMO DATA): {ticker} - {strategy_class.__name__}')
    else:
        ax1.set_title(f'Backtest Results: {ticker} - {strategy_class.__name__}')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True)

    # Add buy/sell markers if we have trade data
    if not trades_df.empty:
        # Convert dates for plotting
        trades_df['plot_date'] = pd.to_datetime(trades_df['datetime'])

        # Plot buy signals
        buy_signals = trades_df[trades_df['type'] == 'buy']
        if not buy_signals.empty:
            for idx, trade in buy_signals.iterrows():
                try:
                    date = trade['plot_date']
                    # Find the equity value on this date
                    equity_value = returns_df.loc[date:date, 'Cumulative'].iloc[0]

                    # Use a small fixed offset for the arrow tail
                    y_offset = equity_value * 0.01  # Small percentage offset

                    ax1.annotate('', xy=(date, equity_value), xytext=(date, equity_value - y_offset),
                                 arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
                except (KeyError, IndexError) as e:
                    # Handle date not found in returns
                    continue

        # Plot sell signals
        sell_signals = trades_df[trades_df['type'] == 'sell']
        if not sell_signals.empty:
            for idx, trade in sell_signals.iterrows():
                try:
                    date = trade['plot_date']
                    # Find the equity value on this date
                    equity_value = returns_df.loc[date:date, 'Cumulative'].iloc[0]

                    # Use a small fixed offset for the arrow tail
                    y_offset = equity_value * 0.01  # Small percentage offset

                    ax1.annotate('', xy=(date, equity_value), xytext=(date, equity_value + y_offset),
                                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
                except (KeyError, IndexError) as e:
                    # Handle date not found in returns
                    continue

    # Plot returns
    ax2 = fig.add_subplot(212)
    ax2.bar(returns_df.index, returns_df['Return'], color='blue')
    ax2.set_ylabel('Daily Returns')
    ax2.grid(True)

    # Add text with summary statistics
    stats_text = (
        f"Initial Capital: $100,000\n"
        f"Final Value: ${cerebro.broker.getvalue():,.2f}\n"
        f"Total Return: {cerebro.broker.getvalue() / 100000 - 1:.2%}\n"
        f"Sharpe Ratio: {sharpe.get('sharperatio', 0):.2f}\n"
        f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2%}"
    )

    plt.figtext(0.15, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig, trades_df


def main():
    """
    Main function to run the Streamlit application
    """
    st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")

    st.title('US Equities Trading Strategy Backtester')

    # Sidebar for strategy selection and parameters
    st.sidebar.header('Strategy Configuration')

    # Demo mode toggle
    demo_mode = st.sidebar.checkbox('Demo Mode (Use Static Data)', False)

    if demo_mode:
        st.sidebar.info("Demo mode is enabled. Using static data instead of live Yahoo Finance data.")

    # Strategy selection
    selected_strategy_name = st.sidebar.selectbox('Select Strategy', list(strategies.keys()))
    selected_strategy_class = strategies[selected_strategy_name]

    # Get strategy parameters
    strategy_params = {}
    st.sidebar.subheader('Strategy Parameters')

    # Dynamically get parameters from the selected strategy
    for param_name, param_default in selected_strategy_class.params._getitems():
        if param_name not in ['ticker']:  # Skip ticker as we'll set it separately
            # Set the appropriate input type based on the default parameter type
            if isinstance(param_default, bool):
                strategy_params[param_name] = st.sidebar.checkbox(
                    f'{param_name}',
                    value=param_default
                )
            elif isinstance(param_default, int):
                strategy_params[param_name] = st.sidebar.number_input(
                    f'{param_name}',
                    value=param_default,
                    step=1
                )
            elif isinstance(param_default, float):
                strategy_params[param_name] = st.sidebar.number_input(
                    f'{param_name}',
                    value=param_default,
                    step=0.01
                )
            else:
                strategy_params[param_name] = st.sidebar.text_input(
                    f'{param_name}',
                    value=str(param_default)
                )

    # Get ticker and date range
    st.sidebar.header('Backtest Configuration')

    if demo_mode:
        ticker = st.sidebar.selectbox('Select demo ticker:', ['SPY'])
        st.sidebar.warning("Only SPY is available in demo mode")
    else:
        ticker = st.sidebar.text_input('Enter ticker symbol (e.g., AAPL, MSFT, AMZN, ...)', 'SPY')

    start_date = st.sidebar.date_input(
        'Select start date:',
        datetime.now() - timedelta(days=365)
    )

    end_date = st.sidebar.date_input(
        'Select end date:',
        datetime.now()
    )

    # Run backtest button
    if st.sidebar.button('Run Backtest'):
        if demo_mode:
            st.info(
                f"Running backtest on {ticker} (DEMO DATA) from {start_date} to {end_date} using {selected_strategy_name}")
        else:
            st.info(f"Running backtest on {ticker} from {start_date} to {end_date} using {selected_strategy_name}")

        # Create a progress bar
        progress_bar = st.progress(0)

        # Update progress
        progress_bar.progress(10)

        # Run the backtest
        try:
            fig, trades_df = run_backtest(selected_strategy_class, ticker, start_date, end_date, demo_mode,
                                          **strategy_params)

            # Update progress
            progress_bar.progress(90)

            if fig:
                # Create tabs for results and trades
                tab1, tab2 = st.tabs(["Performance", "Trade Log"])

                with tab1:
                    # Use Streamlit's built-in functionality for responsive charts
                    st.pyplot(fig, use_container_width=True)

                    if demo_mode:
                        st.warning("This backtest is using DEMO DATA and does not reflect real market conditions.")

                    st.success("Backtest completed successfully!")

                with tab2:
                    if not trades_df.empty:
                        # Format the dataframe for display
                        display_df = trades_df.copy()
                        display_df['datetime'] = pd.to_datetime(display_df['datetime']).dt.strftime('%Y-%m-%d')
                        display_df['price'] = display_df['price'].map('${:.2f}'.format)
                        display_df['value'] = display_df['value'].map('${:.2f}'.format)
                        display_df['commission'] = display_df['commission'].map('${:.2f}'.format)
                        display_df['net_value'] = display_df['net_value'].map('${:.2f}'.format)
                        display_df = display_df.rename(columns={
                            'datetime': 'Date',
                            'type': 'Action',
                            'price': 'Price',
                            'size': 'Shares',
                            'value': 'Value',
                            'commission': 'Commission',
                            'net_value': 'Net Value'
                        })

                        # Display the trade log
                        st.write("### Trade Log")
                        if demo_mode:
                            st.warning(
                                "DEMO DATA: This trade log is based on synthetic data and does not reflect real trades.")

                        st.dataframe(display_df, use_container_width=True)

                        # Add download button
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="Download Trade Data",
                            data=csv,
                            file_name=f"{ticker}_{selected_strategy_name}_trades.csv",
                            mime="text/csv"
                        )
                    else:
                        st.write("No trades were executed during this backtest period.")
            else:
                st.error("Backtest failed. Please check your inputs and try again.")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"An error occurred during the backtest: {str(e)}")
            st.error(f"Error details:\n```\n{error_details}\n```")

        # Complete progress
        progress_bar.progress(100)

    # Add some helpful information
    st.markdown("""
    ## About This Application

    This application allows you to backtest various trading strategies on US equities data. The backtest results include:

    - Portfolio equity curve
    - Daily returns
    - Performance statistics (Total Return, Sharpe Ratio, Max Drawdown)

    ### Available Strategies:

    1. **Moving Average Cross Strategy**: Generates signals based on fast and slow moving average crossovers
    2. **Momentum Strategy**: Buys stocks showing strong momentum and sells when momentum weakens
    3. **RSI Strategy**: Buys when RSI is oversold and sells when RSI is overbought

    ### To Run a Backtest:

    1. Select a strategy from the dropdown menu
    2. Adjust strategy parameters as needed
    3. Enter a ticker symbol (e.g., AAPL, MSFT, SPY)
    4. Set the backtest date range
    5. Click "Run Backtest"

    ### Demo Mode:

    Toggle "Demo Mode" in the sidebar to use static data instead of live Yahoo Finance data. 
    This is useful for demonstration purposes or when you don't have internet access.
    """)


if __name__ == "__main__":
    main()