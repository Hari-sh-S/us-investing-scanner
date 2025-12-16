# 🇺🇸 US Investing Scanner

Advanced backtesting platform for US stock markets built with Streamlit.

## Features

- **Multiple Universes**: S&P 500, NASDAQ 100, DOW 30, Russell 2000, sector indices
- **Custom Scoring**: Momentum, volatility, Sharpe, Sortino formulas
- **Regime Filters**: EMA, MACD, SuperTrend based market timing
- **Uncorrelated Assets**: Allocate to GLD, TLT, etc. during downturns
- **Comprehensive Metrics**: CAGR, Sharpe, Max Drawdown, Win Rate
- **Benchmark Comparison**: Compare against S&P 500, NASDAQ, DOW, etc.
- **Export Reports**: Excel and PDF reports

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run us_app.py
```

## Available Universes

| Category | Indices |
|----------|---------|
| Broad Market | S&P 500, NASDAQ 100, DOW 30, Russell 2000 |
| Cap-Based | S&P MidCap 400, S&P SmallCap 600 |
| Sectors | Technology, Healthcare, Financials, Energy, Consumer |
| Thematic | NYSE FANG+, NASDAQ Financial |

## Configuration

- **Starting Capital**: $10,000 - $100,000,000
- **Portfolio Size**: 1-50 stocks
- **Rebalancing**: Weekly or Monthly
- **Time Period**: Custom date range

## Scoring Formulas

Examples:
- `6 Month Performance` - Momentum strategy
- `6 Month Sharpe` - Risk-adjusted returns
- `6 Month Performance - 3 Month Volatility` - Quality momentum

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to [share.streamlit.io](https://share.streamlit.io)
3. Deploy `us_app.py`

## License

MIT License
