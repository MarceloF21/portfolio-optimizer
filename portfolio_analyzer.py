#!/usr/bin/env python3
"""
================================================================================
QUANTITATIVE PORTFOLIO ANALYZER & OPTIMIZER
================================================================================
Author: Marcelo Farez
Description: Advanced portfolio analysis using Modern Portfolio Theory (MPT)
             with SQL database integration, risk metrics, and optimization.

Features:
- Historical data retrieval and storage (SQLite)
- Portfolio optimization (Max Sharpe, Min Volatility)
- Efficient Frontier generation
- Risk metrics (VaR, Beta, Max Drawdown, Sortino Ratio)
- Professional visualizations

Requirements: pandas, numpy, yfinance, matplotlib, seaborn, scipy
================================================================================
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configuration
DB_PATH = 'portfolio_analyzer.db'
RISK_FREE_RATE = 0.045  # 4.5% annual

class PortfolioAnalyzer:
    """
    Comprehensive portfolio analysis and optimization engine.
    Implements Modern Portfolio Theory for optimal asset allocation.
    """

    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize analyzer with stock universe.

        Parameters:
        -----------
        tickers : dict
            Dictionary mapping ticker symbols to sectors
        start_date : datetime, optional
            Analysis start date (default: 3 years ago)
        end_date : datetime, optional
            Analysis end date (default: today)
        """
        self.tickers = tickers
        self.ticker_list = list(tickers.keys())
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=3*365))
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.prices = None
        self.returns = None

    def initialize_database(self):
        """Create database schema for storing price and metric data."""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date DATE,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                adjusted_close REAL,
                PRIMARY KEY (ticker, date)
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_metrics (
                ticker TEXT PRIMARY KEY,
                expected_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                beta REAL,
                var_95 REAL,
                max_drawdown REAL,
                last_updated DATE
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS efficient_frontier (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                expected_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                weights TEXT,
                created_date DATE
            )
        ''')
        self.conn.commit()

    def fetch_data(self):
        """Retrieve historical price data from Yahoo Finance."""
        print(f"Downloading data for {len(self.ticker_list)} stocks...")

        import yfinance as yf
        all_data = []

        for ticker in self.ticker_list:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=self.start_date, end=self.end_date)

                for date, row in hist.iterrows():
                    all_data.append((
                        ticker, date.strftime('%Y-%m-%d'),
                        row['Open'], row['High'], row['Low'],
                        row['Close'], int(row['Volume']), row['Close']
                    ))
                print(f"  ✓ {ticker}: {len(hist)} records")
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")

        self.cursor.executemany('''
            INSERT OR REPLACE INTO stock_prices 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', all_data)
        self.conn.commit()

    def load_data(self):
        """Load price data from database into DataFrame."""
        query = f"""
            SELECT ticker, date, close_price 
            FROM stock_prices 
            WHERE ticker IN ({','.join([''?'']*len(self.ticker_list))})
            ORDER BY date
        """
        df = pd.read_sql(query, self.conn, params=self.ticker_list)
        df['date'] = pd.to_datetime(df['date'])
        self.prices = df.pivot(index='date', columns='ticker', values='close_price')
        self.returns = self.prices.pct_change().dropna()

    def calculate_metrics(self):
        """Calculate comprehensive risk and return metrics."""
        annual_returns = self.returns.mean() * 252
        annual_vol = self.returns.std() * np.sqrt(252)

        # Market proxy (equal-weighted)
        market_returns = self.returns.mean(axis=1)

        metrics = []
        for ticker in self.ticker_list:
            stock_ret = self.returns[ticker]

            # Beta
            cov = np.cov(stock_ret, market_returns)[0, 1]
            beta = cov / np.var(market_returns)

            # VaR (95%)
            var_95 = np.percentile(stock_ret, 5)

            # Max Drawdown
            cumret = (1 + stock_ret).cumprod()
            running_max = cumret.expanding().max()
            max_dd = ((cumret - running_max) / running_max).min()

            # Sharpe Ratio
            sharpe = (annual_returns[ticker] - RISK_FREE_RATE) / annual_vol[ticker]

            metrics.append({
                'ticker': ticker,
                'expected_return': annual_returns[ticker],
                'volatility': annual_vol[ticker],
                'sharpe_ratio': sharpe,
                'beta': beta,
                'var_95': var_95,
                'max_drawdown': max_dd
            })

        # Store in database
        data = [(m['ticker'], m['expected_return'], m['volatility'],
                m['sharpe_ratio'], m['beta'], m['var_95'], 
                m['max_drawdown'], datetime.now().strftime('%Y-%m-%d')) 
               for m in metrics]

        self.cursor.executemany('''
            INSERT OR REPLACE INTO portfolio_metrics 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data)
        self.conn.commit()

        return pd.DataFrame(metrics)

    def optimize_portfolio(self, objective='sharpe', bounds=None):
        """
        Optimize portfolio weights using mean-variance optimization.

        Parameters:
        -----------
        objective : str
            'sharpe' for maximum Sharpe ratio, 'volatility' for minimum volatility
        bounds : list of tuples, optional
            Min/max weights for each asset (default: 0-25%)
        """
        annual_ret = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        n = len(self.ticker_list)

        if bounds is None:
            bounds = [(0, 0.25)] * n

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        initial_weights = np.array([1/n] * n)

        def neg_sharpe(w):
            port_ret = np.sum(annual_ret * w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(port_ret - RISK_FREE_RATE) / port_vol

        def port_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        if objective == 'sharpe':
            result = minimize(neg_sharpe, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)
        else:
            result = minimize(port_vol, initial_weights, method='SLSQP',
                            bounds=bounds, constraints=constraints)

        weights = result.x
        port_ret = np.sum(annual_ret * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_ret - RISK_FREE_RATE) / port_vol

        return {
            'weights': weights,
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe_ratio': sharpe
        }

    def generate_efficient_frontier(self, n_points=50):
        """Generate efficient frontier by optimizing at different return targets."""
        annual_ret = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        n = len(self.ticker_list)

        min_vol_result = self.optimize_portfolio('volatility')
        max_ret = annual_ret.max()

        target_returns = np.linspace(min_vol_result['expected_return'], max_ret * 0.95, n_points)
        frontier = []

        for target in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(annual_ret * w) - target},
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]

            result = minimize(
                lambda w: np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))),
                np.array([1/n] * n),
                method='SLSQP',
                bounds=[(0, 0.25)] * n,
                constraints=constraints
            )

            if result.success:
                vol = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
                sharpe = (target - RISK_FREE_RATE) / vol
                frontier.append({
                    'return': target,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'weights': result.x
                })

        return frontier

    def plot_efficient_frontier(self, save_path=None):
        """Create professional efficient frontier visualization."""
        annual_ret = self.returns.mean() * 252
        annual_vol = self.returns.std() * np.sqrt(252)

        max_sharpe = self.optimize_portfolio('sharpe')
        min_vol = self.optimize_portfolio('volatility')
        frontier = self.generate_efficient_frontier()

        fig, ax = plt.subplots(figsize=(12, 8))

        # Individual stocks
        for ticker in self.ticker_list:
            ax.scatter(annual_vol[ticker], annual_ret[ticker], s=100, alpha=0.7)
            ax.annotate(ticker, (annual_vol[ticker], annual_ret[ticker]))

        # Efficient frontier
        ax.plot([p['volatility'] for p in frontier], 
               [p['return'] for p in frontier], 'b-', linewidth=2)

        # Optimized portfolios
        ax.scatter(max_sharpe['volatility'], max_sharpe['expected_return'], 
                  c='gold', s=300, marker='*', label=f"Max Sharpe: {max_sharpe['sharpe_ratio']:.3f}")
        ax.scatter(min_vol['volatility'], min_vol['expected_return'], 
                  c='lime', s=300, marker='*', label='Min Volatility')

        ax.set_xlabel('Volatility')
        ax.set_ylabel('Expected Return')
        ax.set_title('Efficient Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Define universe
    universe = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
        'AMZN': 'Consumer Discretionary', 'JPM': 'Financials', 'BAC': 'Financials',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'XOM': 'Energy',
        'CVX': 'Energy', 'WMT': 'Consumer Staples', 'PG': 'Consumer Staples',
        'DIS': 'Communication Services', 'VZ': 'Communication Services',
        'BA': 'Industrials', 'CAT': 'Industrials'
    }

    # Initialize and run analysis
    analyzer = PortfolioAnalyzer(universe)
    analyzer.initialize_database()
    # analyzer.fetch_data()  # Uncomment to download fresh data
    analyzer.load_data()

    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    print("\nRisk Metrics:")
    print(metrics.to_string())

    # Optimize portfolios
    max_sharpe_port = analyzer.optimize_portfolio('sharpe')
    print(f"\nMax Sharpe Portfolio: {max_sharpe_port['sharpe_ratio']:.3f}")

    min_vol_port = analyzer.optimize_portfolio('volatility')
    print(f"Min Vol Portfolio: {min_vol_port['volatility']:.2%}")

    # Generate visualization
    analyzer.plot_efficient_frontier('efficient_frontier.png')
