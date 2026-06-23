# AIC Alpaca Research Wrapper

This project provides a small read-only wrapper around `alpaca-py` for research and exploratory data analysis in Jupyter notebooks.

## Setup

Install the project and notebook tooling:

```bash
uv sync --group dev
```

Create a local `.env` file from the example and add your Alpaca paper credentials:

```bash
cp .env.example .env
```

```dotenv
ALPACA_API_KEY=your-api-key
ALPACA_SECRET_KEY=your-secret-key
ALPACA_PAPER=true
ALPACA_STOCK_FEED=iex
ALPACA_CRYPTO_FEED=us
```

## Notebook Usage

```python
from aic import AlpacaResearchClient

client = AlpacaResearchClient.from_env()

stocks = client.stock_bars(["AAPL", "MSFT"], start="2025-01-01", timeframe="1Day")
crypto = client.crypto_bars("BTC/USD", start="2025-01-01", timeframe="1Day")
positions = client.positions()
```

Methods return pandas DataFrames by default. Pass `raw=True` to receive the underlying `alpaca-py` response object instead.

`AlpacaResearchClient.from_env()` searches for `.env` in the current directory and parent directories, so it works when a notebook kernel starts inside `notebooks/`.

Run the starter notebook:

```bash
uv run jupyter lab notebooks/alpaca_quickstart.ipynb
```

For a deeper exploratory data analysis walkthrough:

```bash
uv run jupyter lab notebooks/alpaca_eda_examples.ipynb
```
