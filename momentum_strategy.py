import numpy as np
import pandas as pd
import yfinance as yf
import math
from scipy import stats
from statistics import mean
import xlsxwriter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def get_sp500_tickers():
    """
    Get S&P 500 component tickers
    """
    try:
        # Use pandas to read S&P 500 data from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        table = pd.read_html(url)[0]
        tickers = table['Symbol'].tolist()
        names = table['Security'].tolist()

        # Create a dataframe with the tickers and names
        stock_list = pd.DataFrame({'ticker': tickers, 'name': names})
        print(f"Successfully retrieved {len(stock_list)} S&P 500 stocks")
        return stock_list
    except Exception as e:
        print(f"Failed to retrieve S&P 500 stocks: {str(e)}")
        # Return empty dataframe if there's an error
        return pd.DataFrame(columns=['ticker', 'name'])


def get_stock_data(ticker, start_date, end_date):
    """
    Get historical stock data for a single ticker

    Parameters:
        ticker: Stock ticker symbol (e.g., AAPL)
        start_date: Start date for data retrieval
        end_date: End date for data retrieval

    Returns:
        DataFrame: Historical price data including date, open, close, etc.
        None: If data retrieval fails
    """
    try:
        # Get data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date)

        # Only keep necessary columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Add ticker and name columns
        df['ticker'] = ticker

        return df
    except Exception as e:
        print(f"Failed to retrieve data for {ticker}: {str(e)}")
        return None


def calculate_momentum(df, ticker, name):
    """
    Calculate momentum indicators for different time periods

    Parameters:
        df: DataFrame with historical price data
        ticker: Stock ticker symbol
        name: Company name

    Returns:
        dict: Momentum indicators for different time periods
    """
    if df is None or len(df) < 2:
        return None

    latest_price = df['Close'].iloc[-1]

    # Calculate returns over different time periods
    returns = {
        'ticker': ticker,
        'name': name,
        'price': latest_price
    }

    # 1-month return (approximately 21 trading days)
    if len(df) >= 21:
        returns['month1_return'] = (latest_price - df['Close'].iloc[-21]) / df['Close'].iloc[-21]

    # 3-month return (approximately 63 trading days)
    if len(df) >= 63:
        returns['month3_return'] = (latest_price - df['Close'].iloc[-63]) / df['Close'].iloc[-63]

    # 6-month return (approximately 126 trading days)
    if len(df) >= 126:
        returns['month6_return'] = (latest_price - df['Close'].iloc[-126]) / df['Close'].iloc[-126]

    # 1-year return
    returns['year1_return'] = (latest_price - df['Close'].iloc[0]) / df['Close'].iloc[0]

    return returns


def portfolio_input():
    """
    Get portfolio size input from user and validate

    Returns:
        float: Validated portfolio size
    """
    while True:
        portfolio_size = input("Enter your portfolio amount (USD): ")
        try:
            val = float(portfolio_size)
            if val <= 0:
                print("Amount must be greater than 0")
                continue
            return val
        except ValueError:
            print("Please enter a valid number!")


def create_excel_report(top_momentum, portfolio_size, total_investment, cash_remaining):
    """
    Create and format an Excel report with the momentum portfolio

    Parameters:
        top_momentum: DataFrame with top momentum stocks
        portfolio_size: Total portfolio amount
        total_investment: Total amount invested
        cash_remaining: Cash remaining after investment

    Returns:
        None
    """
    # Create Excel writer
    report_file = 'US_Momentum_Strategy_Portfolio.xlsx'
    writer = pd.ExcelWriter(report_file, engine='xlsxwriter')

    # Write data
    top_momentum.to_excel(writer, sheet_name='Momentum Portfolio', index=False)

    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Momentum Portfolio']

    # Define professional formats
    format_dict = {
        'font_name': 'Arial',
        'border': 1
    }

    header_format = workbook.add_format({
        **format_dict,
        'bold': True,
        'font_color': '#FFFFFF',
        'bg_color': '#1F497D',
        'align': 'center',
        'valign': 'vcenter'
    })

    # Number formats
    price_format = workbook.add_format({
        **format_dict,
        'num_format': '$#,##0.00',
        'align': 'right'
    })

    percent_format = workbook.add_format({
        **format_dict,
        'num_format': '0.0%',
        'align': 'right'
    })

    int_format = workbook.add_format({
        **format_dict,
        'num_format': '#,##0',
        'align': 'right'
    })

    # Set column widths and formats
    columns_config = [
        ('Ticker', 10, header_format),
        ('Name', 40, header_format),
        ('Price', 12, price_format),
        ('Shares to Buy', 12, int_format),
        ('1-Year Return', 12, percent_format),
        ('1-Year Percentile', 12, percent_format),
        ('6-Month Return', 12, percent_format),
        ('6-Month Percentile', 12, percent_format),
        ('3-Month Return', 12, percent_format),
        ('3-Month Percentile', 12, percent_format),
        ('1-Month Return', 12, percent_format),
        ('1-Month Percentile', 12, percent_format),
        ('HQM Score', 12, percent_format),
        ('Actual Investment', 14, price_format)
    ]

    for idx, (col_name, width, fmt) in enumerate(columns_config):
        # Set column width
        worksheet.set_column(idx, idx, width, fmt)
        # Rewrite headers
        worksheet.write(0, idx, col_name, header_format)

    # Add summary information
    summary_text = [
        f"Report Generation Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"Total Portfolio Amount: ${float(portfolio_size):,.2f}",
        f"Actual Investment Amount: ${total_investment:,.2f}",
        f"Cash Balance: ${cash_remaining:,.2f}",
        f"Number of Stocks: {len(top_momentum)}"
    ]

    for row, text in enumerate(summary_text, start=len(top_momentum) + 3):
        worksheet.write(row, 0, text)

    # Save Excel file
    writer.close()
    print(f"\nPortfolio report generated: {report_file}")


def run_momentum_strategy():
    """
    Execute the momentum strategy:
    1. Get stock data
    2. Calculate momentum indicators
    3. Rank stocks by momentum
    4. Select top momentum stocks
    5. Calculate trading orders
    6. Generate report
    """
    # Get current date and date one year ago
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Get S&P 500 stocks
    stock_list = get_sp500_tickers()

    if len(stock_list) == 0:
        print("No stocks found. Exiting.")
        return

    # Initialize data collector
    momentum_data = []

    # Process each stock
    print("\nProcessing stocks...")
    for i, row in stock_list.iterrows():
        ticker = row['ticker']
        name = row['name']

        print(f"Processing {i + 1}/{len(stock_list)}: {ticker} - {name}")

        # Get data and calculate momentum
        df = get_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if df is not None and len(df) > 0:
            stock_momentum = calculate_momentum(df, ticker, name)

            # Only save valid data
            if stock_momentum and all(k in stock_momentum for k in ['month1_return', 'year1_return']):
                momentum_data.append(stock_momentum)

    # Convert to DataFrame
    if len(momentum_data) == 0:
        print("No valid momentum data found. Exiting.")
        return

    momentum_df = pd.DataFrame(momentum_data)

    # Add market info (NYSE/NASDAQ)
    momentum_df['market'] = 'US'

    # Define analysis time periods
    time_periods = ['year1', 'month6', 'month3', 'month1']

    # Calculate percentiles for each time period
    for period in time_periods:
        col_name = f'{period}_return'
        percentile_col = f'{period}_percentile'

        # Filter valid returns
        valid_returns = momentum_df[col_name].dropna()

        # Calculate percentiles (0-1 range)
        momentum_df[percentile_col] = momentum_df[col_name].apply(
            lambda x: stats.percentileofscore(valid_returns, x) / 100 if pd.notnull(x) else None
        )

    # Calculate combined HQM score (High Quality Momentum)
    momentum_df['hqm_score'] = momentum_df[[f'{period}_percentile' for period in time_periods]].mean(axis=1)

    # Clean data - remove rows with missing HQM score
    momentum_df = momentum_df.dropna(subset=['hqm_score'])

    # Add rank information
    momentum_df['rank'] = momentum_df['hqm_score'].rank(ascending=False)

    # Select top momentum stocks
    top_count = 50
    top_momentum = momentum_df.sort_values('hqm_score', ascending=False).head(top_count)

    # Reset index for cleaner output
    top_momentum.reset_index(drop=True, inplace=True)

    # Add selection reason
    top_momentum['selection_reason'] = "High Quality Momentum Stock"

    # Example output
    print(
        f"\nSelected {len(top_momentum)} momentum stocks, HQM score range: {top_momentum['hqm_score'].min():.2f}-{top_momentum['hqm_score'].max():.2f}")

    # Get portfolio size from user
    print("\n=== Portfolio Configuration ===")
    portfolio_size = portfolio_input()

    # Calculate allocation per stock
    position_size = float(portfolio_size) / len(top_momentum)
    print(f"Amount allocated per stock: ${position_size:,.2f}")

    # Calculate shares to buy (considering round lots for US stocks)
    top_momentum['shares_to_buy'] = (position_size / top_momentum['price']).apply(
        lambda x: math.floor(x) if not pd.isna(x) else 0  # US stocks can be bought in any quantity
    )

    # Calculate actual investment amount
    top_momentum['actual_investment'] = top_momentum['shares_to_buy'] * top_momentum['price']

    # Display summary information
    total_investment = top_momentum['actual_investment'].sum()
    cash_remaining = float(portfolio_size) - total_investment
    print(f"\nTotal actual investment: ${total_investment:,.2f}")
    print(f"Remaining cash: ${cash_remaining:,.2f} ({cash_remaining / float(portfolio_size):.1%})")

    # Create Excel report
    create_excel_report(top_momentum, portfolio_size, total_investment, cash_remaining)

    # Create a visualization of the HQM scores
    plt.figure(figsize=(10, 6))
    top_momentum.sort_values('hqm_score').plot(kind='barh', x='ticker', y='hqm_score', legend=False)
    plt.title('Top Momentum Stocks by HQM Score')
    plt.xlabel('HQM Score')
    plt.ylabel('Stock Ticker')
    plt.tight_layout()
    plt.savefig('momentum_scores.png')
    print("HQM score visualization saved as momentum_scores.png")

    return top_momentum


def main():
    """
    Main function to execute the momentum strategy
    """
    print("=== US Equity Momentum Strategy ===")
    print("This program identifies high-momentum stocks from the S&P 500 and creates an investment portfolio.")

    run_momentum_strategy()


if __name__ == "__main__":
    main()