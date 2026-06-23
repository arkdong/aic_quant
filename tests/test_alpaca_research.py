from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import aic.alpaca_research as alpaca_research
from aic import AlpacaResearchClient, AlpacaSettings, MissingAlpacaCredentialsError
from aic.alpaca_research import (
    DataFeed,
    parse_adjustment,
    parse_crypto_feed,
    parse_stock_feed,
    parse_timeframe,
    normalize_to_dataframe,
)


def test_settings_loads_primary_env_vars(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("ALPACA_STOCK_FEED", "sip")
    monkeypatch.setenv("ALPACA_CRYPTO_FEED", "us")

    settings = AlpacaSettings.from_env(env_file=None)

    assert settings.api_key == "key"
    assert settings.secret_key == "secret"
    assert settings.paper is False
    assert settings.stock_feed == "sip"
    assert settings.crypto_feed == "us"


def test_settings_supports_apca_aliases(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "alias-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "alias-secret")

    settings = AlpacaSettings.from_env(env_file=None)

    assert settings.api_key == "alias-key"
    assert settings.secret_key == "alias-secret"
    assert settings.paper is True
    assert settings.stock_feed == "iex"


def test_settings_finds_env_file_from_notebook_subdirectory(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    notebook_dir = project_dir / "notebooks"
    notebook_dir.mkdir(parents=True)
    (project_dir / ".env").write_text(
        "ALPACA_API_KEY=key-from-file\n"
        "ALPACA_SECRET_KEY=secret-from-file\n"
        "ALPACA_PAPER=true\n",
        encoding="utf-8",
    )
    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(notebook_dir)

    settings = AlpacaSettings.from_env()

    assert settings.api_key == "key-from-file"
    assert settings.secret_key == "secret-from-file"


def test_missing_credentials_error_names_env_vars(monkeypatch):
    for key in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(MissingAlpacaCredentialsError) as exc_info:
        AlpacaSettings.from_env(env_file=None)

    message = str(exc_info.value)
    assert "ALPACA_API_KEY" in message
    assert "ALPACA_SECRET_KEY" in message
    assert "APCA_API_KEY_ID" in message
    assert "APCA_API_SECRET_KEY" in message


def test_parsers_accept_friendly_values():
    timeframe = parse_timeframe("5Min")

    assert timeframe.amount_value == 5
    assert timeframe.unit_value.value == "Min"
    assert parse_stock_feed("iex") == DataFeed.IEX
    assert parse_stock_feed("delayed-sip").value == "delayed_sip"
    assert parse_crypto_feed("us").value == "us"
    assert parse_adjustment("all").value == "all"


def test_normalize_dataframe_resets_multi_index():
    index = pd.MultiIndex.from_tuples(
        [("AAPL", pd.Timestamp("2025-01-02"))],
        names=["symbol", "timestamp"],
    )
    source = type("BarSet", (), {"df": pd.DataFrame({"close": [100.0]}, index=index)})()

    frame = normalize_to_dataframe(source)

    assert list(frame.columns) == ["symbol", "timestamp", "close"]
    assert frame.loc[0, "symbol"] == "AAPL"


def test_normalize_empty_positions_list():
    frame = normalize_to_dataframe([])

    assert frame.empty


def test_stock_bars_builds_request_without_live_call(monkeypatch):
    created_clients = []

    class FakeStockDataClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        def get_stock_bars(self, request):
            self.request = request
            return {"ok": True}

    monkeypatch.setattr(alpaca_research, "StockHistoricalDataClient", FakeStockDataClient)

    client = AlpacaResearchClient(AlpacaSettings(api_key="key", secret_key="secret"))
    payload = client.stock_bars(
        ["AAPL", "MSFT"],
        start="2025-01-01",
        timeframe="5Min",
        limit=50,
        raw=True,
    )

    request = created_clients[0].request
    assert payload == {"ok": True}
    assert created_clients[0].kwargs == {"api_key": "key", "secret_key": "secret"}
    assert request.symbol_or_symbols == ["AAPL", "MSFT"]
    assert request.start == datetime(2025, 1, 1)
    assert request.limit == 50
    assert request.timeframe.amount_value == 5
    assert request.feed == DataFeed.IEX


def test_crypto_bars_builds_request_and_feed_without_live_call(monkeypatch):
    created_clients = []

    class FakeCryptoDataClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created_clients.append(self)

        def get_crypto_bars(self, request, feed):
            self.request = request
            self.feed = feed
            return {"ok": True}

    monkeypatch.setattr(alpaca_research, "CryptoHistoricalDataClient", FakeCryptoDataClient)

    client = AlpacaResearchClient(AlpacaSettings(api_key="key", secret_key="secret"))
    payload = client.crypto_bars("BTC/USD", start="2025-01-01", timeframe="1Day", raw=True)

    request = created_clients[0].request
    assert payload == {"ok": True}
    assert created_clients[0].kwargs == {"api_key": "key", "secret_key": "secret"}
    assert request.symbol_or_symbols == "BTC/USD"
    assert request.start == datetime(2025, 1, 1)
    assert request.timeframe.amount_value == 1
    assert created_clients[0].feed.value == "us"
