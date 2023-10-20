import yfinance as yf
import pandas as pd
import numpy as np

# Function to calculate portfolio statistics
def calculate_portfolio_stats(portfolio_tickers, weights):
    # Download historical data
    data = yf.download(portfolio_tickers, start='2010-01-01', end='2023-01-01')['Adj Close']
    
    # Download historical data for the market (using S&P 500 as a proxy)
    sp500 = yf.download('SPY', start='2010-01-01', end='2023-01-01')['Adj Close']

    # Drop missing data
    data.dropna(inplace=True)
    sp500.dropna(inplace=True)

    # Calculate daily returns for portfolio and market
    returns = data.pct_change()
    market_returns = sp500.pct_change()

    # Drop the first row as it will contain NaN values after calculating returns
    returns.dropna(inplace=True)
    market_returns.dropna(inplace=True)

    # Match dates between market returns and portfolio returns
    common_dates = returns.index.intersection(market_returns.index)
    returns = returns.loc[common_dates]
    market_returns = market_returns.loc[common_dates]
    
    # Calculate portfolio returns
    portfolio_returns = returns.dot(weights)

    # Calculate Beta
    cov_matrix = np.cov(portfolio_returns, market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    # Use a 10Y risk-free rate of 4.10% per annum, converted to daily rate
    risk_free_rate = 0.041 / 252

    # Calculate expected portfolio return and standard deviation
    expected_return = portfolio_returns.mean() * 252
    std_dev = portfolio_returns.std() * (252 ** 0.5)

    # Calculate the Sharpe Ratio
    sharpe_ratio = (expected_return - risk_free_rate) / std_dev
    
    # Calculate Sortino Ratio
    downside_std_dev = np.sqrt((portfolio_returns[portfolio_returns < 0] ** 2).mean()) * np.sqrt(252)
    sortino_ratio = (expected_return - risk_free_rate) / downside_std_dev

    # Calculate Maximum Drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    max_drawdown = drawdown.min()

    # Calculate Value at Risk (VaR) at 95% confidence level
    var_95 = np.percentile(portfolio_returns, 5)

    return expected_return, std_dev, sharpe_ratio, var_95, beta, sortino_ratio, max_drawdown

# Define the first portfolio and weights
portfolio1_tickers = [
    'NVDA', 'SMCI', 'NVO', 'ASML', 'CSU.TO',
    'V', 'RMS.PA', 'TSM', 'FICO', 'MKL', 
    'ROP', 'MC.PA', 'ULTA', 'BRK-B'
]
weights1 = [1/len(portfolio1_tickers)] * len(portfolio1_tickers)
expected_return1, std_dev1, sharpe_ratio1, var_951, beta1, sortino_ratio1, max_drawdown1 = calculate_portfolio_stats(portfolio1_tickers, weights1)

# Print the results for the first portfolio
print("------ First Portfolio ------")
print(f"Expected Portfolio Return: {expected_return1*100:.2f}%")
print(f"Portfolio Standard Deviation: {std_dev1*100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio1:.2f}")
print(f"Sortino Ratio: {sortino_ratio1:.2f}")
print(f"Beta: {beta1:.2f}")
print(f"Maximum Drawdown: {max_drawdown1*100:.2f}%")
print(f"Value at Risk (95% confidence): {var_951*100:.2f}%")
