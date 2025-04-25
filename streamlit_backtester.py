import streamlit as st
import pandas as pd
import backtrader as bt
import yfinance as yf
import matplotlib

# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import inspect
from datetime import datetime, timedelta


# Create a strategies module to house your trading strategies
class MovingAverageCrossStrategy(bt.Strategy):
    """
    A strategy that generates signals based on moving average crossovers
    """
    params = (
        ('fast', 10),  # Fast moving average period
        ('slow', 30),  # Slow moving average period
        ('order_percentage', 0.95),  # Percentage of portfolio to allocate per trade
        ('ticker', 'SPY'),  # Ticker for the strategy
    )

    def __init__(self):
        # Initialize indicators
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow
        )

        # Cross signals
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

        # Track order, position, and trade status
        self.order = None
        self.price = None
        self.comm = None

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

    def log(self, txt):
        """
        Logging function for debugging
        """
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


class MomentumStrategy(bt.Strategy):
    """
    A strategy that buys stocks showing strong momentum and sells on momentum weakening
    """
    params = (
        ('lookback', 90),  # Lookback period for momentum calculation
        ('percentile', 80),  # Percentile threshold for entry (0-100)
        ('exit_percentile', 50),  # Percentile threshold for exit (0-100)
        ('order_percentage', 0.95),  # Percentage of portfolio to allocate per trade
        ('ticker', 'SPY'),  # Ticker for the strategy
    )

    def __init__(self):
        # Calculate momentum (rate of change over lookback period)
        self.momentum = bt.indicators.RateOfChange(self.data.close, period=self.params.lookback)

        # Track order, position, and trade status
        self.order = None
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

    def log(self, txt):
        """
        Logging function for debugging
        """
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


class RSIStrategy(bt.Strategy):
    """
    A strategy that buys when RSI is oversold and sells when RSI is overbought
    """
    params = (
        ('period', 14),  # RSI calculation period
        ('overbought', 70),  # Overbought threshold
        ('oversold', 30),  # Oversold threshold
        ('order_percentage', 0.95),  # Percentage of portfolio to allocate per trade
        ('ticker', 'SPY'),  # Ticker for the strategy
    )

    def __init__(self):
        # Calculate RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.period)

        # Track order, position, and trade status
        self.order = None

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

    def log(self, txt):
        """
        Logging function for debugging
        """
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')


# Import necessary libraries for strategies
import math
from scipy import stats

# Add our strategies
strategies = {
    "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
    "MomentumStrategy": MomentumStrategy,
    "RSIStrategy": RSIStrategy
}


def get_stock_data(ticker, start_date, end_date):
    """
    Get historical stock data for backtesting

    Parameters:
        ticker: Stock ticker symbol
        start_date: Start date for data retrieval
        end_date: End date for data retrieval

    Returns:
        DataFrame: Historical price data with OHLCV format
    """
    try:
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


def run_backtest(strategy_class, ticker, start_date, end_date, **params):
    """
    Run a backtest for the specified strategy and parameters

    Parameters:
        strategy_class: The strategy class to backtest
        ticker: Stock ticker symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        **params: Strategy parameters

    Returns:
        Matplotlib figure: Plot of backtest results
    """
    # Debugging: Print all parameters before processing
    st.write("Debug - Parameters received:", params)

    # Check for problematic parameters
    for param_name, param_value in params.items():
        st.write(f"Parameter '{param_name}': {type(param_value).__name__} = {param_value}")

        # Convert any tuples to appropriate values (defensive approach)
        if isinstance(param_value, tuple):
            st.warning(f"Found tuple for parameter '{param_name}': {param_value}")
            # Try to convert tuple to a single value if it has only one element
            if len(param_value) == 1:
                params[param_name] = param_value[0]
                st.write(f"Converted to: {params[param_name]}")
            # Otherwise, use the first value (or implement other logic as needed)
            else:
                params[param_name] = param_value[0]  # Using first value as a simple fix
                st.write(f"Using first value: {params[param_name]}")

    # Initialize Cerebro engine
    cerebro = bt.Cerebro()

    # Set commission to 0 (can be adjusted for realistic backtest)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Get data
    df = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None or len(df) == 0:
        st.error(f"No data available for {ticker} during the specified period.")
        return None

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

    # Plot results
    fig = plt.figure(figsize=(10, 8))

    # Plot equity curve
    ax1 = fig.add_subplot(211)
    ax1.plot(returns_df.index, returns_df['Cumulative'], label='Equity Curve')
    ax1.set_title(f'Backtest Results: {ticker} - {strategy_class.__name__}')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend()
    ax1.grid(True)

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

    return fig


def main():
    """
    Main function to run the Streamlit application
    """
    st.set_page_config(page_title="Trading Strategy Backtester", layout="wide")

    st.title('US Equities Trading Strategy Backtester')

    # Sidebar for strategy selection and parameters
    st.sidebar.header('Strategy Configuration')

    # Strategy selection
    selected_strategy_name = st.sidebar.selectbox('Select Strategy', list(strategies.keys()))
    selected_strategy_class = strategies[selected_strategy_name]

    # Helper function to convert string to appropriate number type
    def to_number(s):
        """Convert string to appropriate number type if possible"""
        if isinstance(s, (int, float)):  # Already a number
            return s

        if not isinstance(s, str):  # If not a string (e.g., tuple), return as is
            return s

        try:
            n = float(s)
            return int(n) if n.is_integer() else n
        except ValueError:
            return s

    # Get strategy parameters
    strategy_params = {}
    st.sidebar.subheader('Strategy Parameters')

    # Dynamically get parameters from the selected strategy
    for param_name, param_default in selected_strategy_class.params._getitems():
        if param_name not in ['ticker']:  # Skip ticker as we'll set it separately
            param_type = type(param_default).__name__
            st.sidebar.text(f"Parameter type: {param_type}")  # Debug info

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
                input_value = st.sidebar.text_input(
                    f'{param_name}',
                    value=str(param_default)
                )
                # Simple safe conversion
                if input_value.isdigit():
                    strategy_params[param_name] = int(input_value)
                elif input_value.replace('.', '', 1).isdigit() and input_value.count('.') < 2:
                    strategy_params[param_name] = float(input_value)
                else:
                    strategy_params[param_name] = input_value
    # Debug: Show the parameters that will be used
    st.sidebar.write("Parameters to be used:", strategy_params)

    # Convert parameters to appropriate types
    strategy_params = {param_name: to_number(strategy_params[param_name]) for param_name in strategy_params}

    # Get ticker and date range
    st.sidebar.header('Backtest Configuration')

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
        st.info(f"Running backtest on {ticker} from {start_date} to {end_date} using {selected_strategy_name}")

        # Create a progress bar
        progress_bar = st.progress(0)

        # Update progress
        progress_bar.progress(10)

        # Run the backtest
        try:
            fig = run_backtest(selected_strategy_class, ticker, start_date, end_date, **strategy_params)

            # Update progress
            progress_bar.progress(90)

            if fig:
                st.pyplot(fig)
                st.success("Backtest completed successfully!")
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
    """)


if __name__ == "__main__":
    main()