"""A small read-only wrapper around alpaca-py for notebook research."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.data.enums import Adjustment, CryptoFeed, DataFeed, MarketType, MostActivesBy
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    CryptoQuoteRequest,
    CryptoSnapshotRequest,
    CryptoTradesRequest,
    MarketMoversRequest,
    MostActivesRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockSnapshotRequest,
    StockTradesRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from dotenv import find_dotenv, load_dotenv

EnvPath = str | Path | None
SymbolInput = str | Sequence[str]
DateLike = str | date | datetime | Any


class MissingAlpacaCredentialsError(ValueError):
    """Raised when required Alpaca credentials are missing from the environment."""


@dataclass(frozen=True)
class AlpacaSettings:
    """Configuration needed to initialize Alpaca SDK clients."""

    api_key: str
    secret_key: str
    paper: bool = True
    stock_feed: str = "iex"
    crypto_feed: str = "us"

    def __post_init__(self) -> None:
        if not self.api_key or not self.secret_key:
            raise MissingAlpacaCredentialsError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in .env, or use APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY aliases."
            )

        stock_feed = str(self.stock_feed).strip().lower()
        crypto_feed = str(self.crypto_feed).strip().lower()
        parse_stock_feed(stock_feed)
        parse_crypto_feed(crypto_feed)
        object.__setattr__(self, "stock_feed", stock_feed)
        object.__setattr__(self, "crypto_feed", crypto_feed)

    @classmethod
    def from_env(cls, env_file: EnvPath = ".env", *, override: bool = False) -> "AlpacaSettings":
        """Load settings from environment variables and an optional .env file."""

        if env_file is not None:
            load_dotenv(dotenv_path=resolve_env_file(env_file), override=override)

        api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")

        if not api_key or not secret_key:
            raise MissingAlpacaCredentialsError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in .env, or use APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY aliases."
            )

        return cls(
            api_key=api_key,
            secret_key=secret_key,
            paper=parse_bool(os.getenv("ALPACA_PAPER", "true")),
            stock_feed=os.getenv("ALPACA_STOCK_FEED", "iex"),
            crypto_feed=os.getenv("ALPACA_CRYPTO_FEED", "us"),
        )


class AlpacaResearchClient:
    """Read-only Alpaca client wrapper optimized for pandas and notebooks."""

    def __init__(self, settings: AlpacaSettings):
        self.settings = settings
        self._trading_client: TradingClient | None = None
        self._stock_data_client: StockHistoricalDataClient | None = None
        self._crypto_data_client: CryptoHistoricalDataClient | None = None
        self._screener_client: ScreenerClient | None = None

    @classmethod
    def from_env(
        cls, env_file: EnvPath = ".env", *, override: bool = False
    ) -> "AlpacaResearchClient":
        """Create a client from .env and current process environment values."""

        return cls(AlpacaSettings.from_env(env_file=env_file, override=override))

    @property
    def trading_client(self) -> TradingClient:
        """The underlying alpaca-py trading client, initialized lazily."""

        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
                paper=self.settings.paper,
            )
        return self._trading_client

    @property
    def stock_data_client(self) -> StockHistoricalDataClient:
        """The underlying alpaca-py stock historical data client."""

        if self._stock_data_client is None:
            self._stock_data_client = StockHistoricalDataClient(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
            )
        return self._stock_data_client

    @property
    def crypto_data_client(self) -> CryptoHistoricalDataClient:
        """The underlying alpaca-py crypto historical data client."""

        if self._crypto_data_client is None:
            self._crypto_data_client = CryptoHistoricalDataClient(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
            )
        return self._crypto_data_client

    @property
    def screener_client(self) -> ScreenerClient:
        """The underlying alpaca-py screener client."""

        if self._screener_client is None:
            self._screener_client = ScreenerClient(
                api_key=self.settings.api_key,
                secret_key=self.settings.secret_key,
            )
        return self._screener_client

    def stock_bars(
        self,
        symbols: SymbolInput,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        timeframe: str | TimeFrame = "1Day",
        limit: int | None = None,
        adjustment: str | Adjustment | None = None,
        feed: str | DataFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = StockBarsRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            timeframe=parse_timeframe(timeframe),
            start=parse_datetime(start),
            end=parse_datetime(end),
            limit=limit,
            adjustment=parse_adjustment(adjustment),
            feed=parse_stock_feed(feed or self.settings.stock_feed),
        )
        return self._format(self.stock_data_client.get_stock_bars(request), raw=raw)

    def stock_quotes(
        self,
        symbols: SymbolInput,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        limit: int | None = None,
        feed: str | DataFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = StockQuotesRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            start=parse_datetime(start),
            end=parse_datetime(end),
            limit=limit,
            feed=parse_stock_feed(feed or self.settings.stock_feed),
        )
        return self._format(self.stock_data_client.get_stock_quotes(request), raw=raw)

    def stock_trades(
        self,
        symbols: SymbolInput,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        limit: int | None = None,
        feed: str | DataFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = StockTradesRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            start=parse_datetime(start),
            end=parse_datetime(end),
            limit=limit,
            feed=parse_stock_feed(feed or self.settings.stock_feed),
        )
        return self._format(self.stock_data_client.get_stock_trades(request), raw=raw)

    def stock_snapshot(
        self,
        symbols: SymbolInput,
        *,
        feed: str | DataFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = StockSnapshotRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            feed=parse_stock_feed(feed or self.settings.stock_feed),
        )
        return self._format(self.stock_data_client.get_stock_snapshot(request), raw=raw)

    def crypto_bars(
        self,
        symbols: SymbolInput,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        timeframe: str | TimeFrame = "1Day",
        limit: int | None = None,
        feed: str | CryptoFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = CryptoBarsRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            timeframe=parse_timeframe(timeframe),
            start=parse_datetime(start),
            end=parse_datetime(end),
            limit=limit,
        )
        payload = self.crypto_data_client.get_crypto_bars(
            request,
            feed=parse_crypto_feed(feed or self.settings.crypto_feed),
        )
        return self._format(payload, raw=raw)

    def crypto_quotes(
        self,
        symbols: SymbolInput,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        limit: int | None = None,
        feed: str | CryptoFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = CryptoQuoteRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            start=parse_datetime(start),
            end=parse_datetime(end),
            limit=limit,
        )
        payload = self.crypto_data_client.get_crypto_quotes(
            request,
            feed=parse_crypto_feed(feed or self.settings.crypto_feed),
        )
        return self._format(payload, raw=raw)

    def crypto_trades(
        self,
        symbols: SymbolInput,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        limit: int | None = None,
        feed: str | CryptoFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = CryptoTradesRequest(
            symbol_or_symbols=coerce_symbols(symbols),
            start=parse_datetime(start),
            end=parse_datetime(end),
            limit=limit,
        )
        payload = self.crypto_data_client.get_crypto_trades(
            request,
            feed=parse_crypto_feed(feed or self.settings.crypto_feed),
        )
        return self._format(payload, raw=raw)

    def crypto_snapshot(
        self,
        symbols: SymbolInput,
        *,
        feed: str | CryptoFeed | None = None,
        raw: bool = False,
    ) -> Any:
        request = CryptoSnapshotRequest(symbol_or_symbols=coerce_symbols(symbols))
        payload = self.crypto_data_client.get_crypto_snapshot(
            request,
            feed=parse_crypto_feed(feed or self.settings.crypto_feed),
        )
        return self._format(payload, raw=raw)

    def account(self, *, raw: bool = False) -> Any:
        return self._format(self.trading_client.get_account(), raw=raw)

    def positions(self, *, raw: bool = False) -> Any:
        return self._format(self.trading_client.get_all_positions(), raw=raw)

    def assets(self, *, raw: bool = False) -> Any:
        return self._format(self.trading_client.get_all_assets(), raw=raw)

    def calendar(
        self,
        *,
        start: DateLike | None = None,
        end: DateLike | None = None,
        raw: bool = False,
    ) -> Any:
        filters = None
        if start is not None or end is not None:
            filters = GetCalendarRequest(start=parse_date(start), end=parse_date(end))
        return self._format(self.trading_client.get_calendar(filters=filters), raw=raw)

    def clock(self, *, raw: bool = False) -> Any:
        return self._format(self.trading_client.get_clock(), raw=raw)

    def most_actives(
        self,
        *,
        top: int = 10,
        by: str | MostActivesBy = "volume",
        raw: bool = False,
    ) -> Any:
        request = MostActivesRequest(top=top, by=parse_most_actives_by(by))
        return self._format(self.screener_client.get_most_actives(request), raw=raw)

    def market_movers(
        self,
        *,
        top: int = 10,
        market_type: str | MarketType = "stocks",
        raw: bool = False,
    ) -> Any:
        request = MarketMoversRequest(
            top=top,
            market_type=parse_market_type(market_type),
        )
        return self._format(self.screener_client.get_market_movers(request), raw=raw)

    @staticmethod
    def _format(payload: Any, *, raw: bool) -> Any:
        if raw:
            return payload
        return normalize_to_dataframe(payload)


def parse_bool(value: str | bool | int | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y", "paper"}:
        return True
    if token in {"0", "false", "f", "no", "n", "live"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def resolve_env_file(env_file: str | Path) -> str | Path:
    """Resolve .env from the current directory or a parent directory."""

    env_path = Path(env_file).expanduser()
    if env_path.is_absolute() or env_path.exists():
        return env_path

    found = find_dotenv(filename=str(env_path), usecwd=True)
    return found or env_path


def coerce_symbols(symbols: SymbolInput) -> str | list[str]:
    if isinstance(symbols, str):
        symbol = symbols.strip()
        if not symbol:
            raise ValueError("symbols cannot be empty")
        return symbol

    if not isinstance(symbols, Sequence):
        raise TypeError("symbols must be a string or sequence of strings")

    cleaned = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
    if not cleaned:
        raise ValueError("symbols cannot be empty")
    return cleaned


def parse_datetime(value: DateLike | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        return datetime.fromisoformat(token.replace("Z", "+00:00"))
    raise TypeError(f"Unsupported datetime value: {value!r}")


def parse_date(value: DateLike | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        return date.fromisoformat(token[:10])
    raise TypeError(f"Unsupported date value: {value!r}")


def parse_timeframe(value: str | TimeFrame) -> TimeFrame:
    if isinstance(value, TimeFrame):
        return value

    token = str(value).strip()
    match = re.fullmatch(r"(?P<amount>\d+)?\s*(?P<unit>[A-Za-z]+)", token)
    if not match:
        raise ValueError(
            "Unsupported timeframe. Use values like 1Min, 5Min, 1Hour, "
            "1Day, 1Week, or 1Month."
        )

    amount = int(match.group("amount") or "1")
    unit_token = match.group("unit").lower()
    unit_aliases = {
        "m": TimeFrameUnit.Minute,
        "min": TimeFrameUnit.Minute,
        "mins": TimeFrameUnit.Minute,
        "minute": TimeFrameUnit.Minute,
        "minutes": TimeFrameUnit.Minute,
        "h": TimeFrameUnit.Hour,
        "hr": TimeFrameUnit.Hour,
        "hour": TimeFrameUnit.Hour,
        "hours": TimeFrameUnit.Hour,
        "d": TimeFrameUnit.Day,
        "day": TimeFrameUnit.Day,
        "days": TimeFrameUnit.Day,
        "w": TimeFrameUnit.Week,
        "week": TimeFrameUnit.Week,
        "weeks": TimeFrameUnit.Week,
        "mo": TimeFrameUnit.Month,
        "mon": TimeFrameUnit.Month,
        "month": TimeFrameUnit.Month,
        "months": TimeFrameUnit.Month,
    }
    try:
        unit = unit_aliases[unit_token]
    except KeyError as exc:
        raise ValueError(f"Unsupported timeframe unit: {match.group('unit')!r}") from exc

    return TimeFrame(amount, unit)


def parse_stock_feed(value: str | DataFeed | None) -> DataFeed | None:
    return enum_from_value(DataFeed, value, label="stock feed")


def parse_crypto_feed(value: str | CryptoFeed | None) -> CryptoFeed | None:
    return enum_from_value(CryptoFeed, value, label="crypto feed")


def parse_adjustment(value: str | Adjustment | None) -> Adjustment | None:
    return enum_from_value(Adjustment, value, label="adjustment")


def parse_most_actives_by(value: str | MostActivesBy) -> MostActivesBy:
    parsed = enum_from_value(MostActivesBy, value, label="most actives ranking")
    if parsed is None:
        raise ValueError("most actives ranking cannot be None")
    return parsed


def parse_market_type(value: str | MarketType) -> MarketType:
    parsed = enum_from_value(MarketType, value, label="market type")
    if parsed is None:
        raise ValueError("market type cannot be None")
    return parsed


def enum_from_value(enum_type: type[Any], value: Any, *, label: str) -> Any:
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value

    token = str(value).strip()
    normalized = _normalize_enum_token(token)
    for member in enum_type:
        candidates = {
            _normalize_enum_token(member.name),
            _normalize_enum_token(str(member.value)),
        }
        if normalized in candidates:
            return member

    expected = ", ".join(str(member.value) for member in enum_type)
    raise ValueError(f"Unsupported {label}: {value!r}. Expected one of: {expected}")


def normalize_to_dataframe(payload: Any) -> pd.DataFrame:
    """Convert Alpaca SDK responses or model objects into a reset-index DataFrame."""

    if isinstance(payload, pd.DataFrame):
        return reset_index_for_notebooks(payload)

    frame = getattr(payload, "df", None)
    if isinstance(frame, pd.DataFrame):
        return reset_index_for_notebooks(frame)

    return plain_to_dataframe(model_to_plain(payload))


def reset_index_for_notebooks(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.RangeIndex) and df.index.name is None:
        return df.copy()
    return df.reset_index()


def model_to_plain(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, (str, int, float, bool, date, datetime)) or value is pd.NA:
        return value
    if isinstance(value, Mapping):
        return {key: model_to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [model_to_plain(item) for item in value]
    if is_dataclass(value):
        return model_to_plain(asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_to_plain(model_dump(mode="json"))
        except TypeError:
            return model_to_plain(model_dump())

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return model_to_plain(to_dict())

    if hasattr(value, "__dict__"):
        return {
            key: model_to_plain(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and not callable(item)
        }

    return value


def plain_to_dataframe(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return reset_index_for_notebooks(value)
    if isinstance(value, list):
        return list_to_dataframe(value)
    if isinstance(value, Mapping):
        return mapping_to_dataframe(value)
    return pd.DataFrame([{"value": value}])


def list_to_dataframe(items: list[Any]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    if all(isinstance(item, Mapping) for item in items):
        return pd.DataFrame(items)
    return pd.DataFrame({"value": items})


def mapping_to_dataframe(data: Mapping[Any, Any]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()

    if all(isinstance(value, Mapping) for value in data.values()):
        return pd.DataFrame(
            [{"symbol": key, **dict(value)} for key, value in data.items()]
        )

    scalar_values = {
        key: value
        for key, value in data.items()
        if not isinstance(value, list)
    }
    list_values = {
        key: value
        for key, value in data.items()
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value)
    }

    if len(list_values) == 1:
        _, rows = next(iter(list_values.items()))
        frame = pd.DataFrame(rows)
        for key, value in scalar_values.items():
            frame[key] = value
        return frame

    if len(list_values) > 1:
        frames: list[pd.DataFrame] = []
        for section, rows in list_values.items():
            frame = pd.DataFrame(rows)
            frame.insert(0, "section", section)
            for key, value in scalar_values.items():
                frame[key] = value
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return pd.DataFrame([dict(data)])


def _normalize_enum_token(value: str) -> str:
    return re.sub(r"[\s\-]+", "_", value.strip().lower())
