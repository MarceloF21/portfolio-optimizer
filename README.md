# Portfolio Optimizer – Modern Portfolio Theory

**Author:** Marcelo Farez  
**Technologies:** Python, Pandas, NumPy, SciPy, SQLite, Matplotlib/Seaborn  
**Domain:** Quantitative Finance, Portfolio Optimization, Risk Management

### What it does
Builds optimal stock portfolios using **Modern Portfolio Theory** (Markowitz).  
Finds the best mix of 16 stocks across 8 sectors for **maximum Sharpe ratio** or **minimum volatility**.

### Key Results
- **Max Sharpe Ratio:** 1.853  
- **Expected Annual Return:** 44.00%  
- **Volatility:** 21.32%  
- **Database:** 12,544 price records (3 years)  
- **Efficient Frontier:** 42 optimized points  
- **Risk metrics calculated:** VaR, Beta, Max Drawdown, Sortino Ratio

### Files in this repo
- `portfolio_analyzer.py`        → Main optimization & analysis code  
- `portfolio_analyzer.db`        → SQLite database with historical prices  
- `portfolio_analysis_dashboard.png` → 6-panel performance charts  
- `risk_analysis_detailed.png`   → Detailed risk metrics visuals  
- `executive_summary.png`        → One-page project summary  
- `requirements.txt`             → Python dependencies

### How to run (optional)
1. `pip install -r requirements.txt`  
2. Uncomment `analyzer.fetch_data()` in the script to download fresh data  
3. Run `python portfolio_analyzer.py`

### Skills shown
- Mean-variance optimization (SciPy SLSQP)  
- Constrained portfolio allocation (max 25% per stock)  
- Institutional risk metrics  
- Professional data visualization  
- SQLite database design

Contact: marcelodavid1404@gmail.com  
GitHub: github.com/MarceloF21/portfolio-optimizer
