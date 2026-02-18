import pandas as pd

from intraday_loader import (
    fetch_intraday_alpha_vantage,
    fetch_intraday_twelve_data,
    fetch_nyse_latest_open_prices,
)


class DummyResponse:
    def __init__(self, payload, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, responses):
        self._responses = responses

    def get(self, url, params=None, timeout=20):
        symbol = (params or {}).get("symbol")
        return DummyResponse(self._responses[symbol])


def test_fetch_intraday_twelve_data_parses_rows():
    session = DummySession(
        {
            "AAPL": {
                "values": [
                    {
                        "datetime": "2026-02-18 16:00:00",
                        "open": "190.0",
                        "high": "191.0",
                        "low": "189.0",
                        "close": "190.5",
                        "volume": "12345",
                    },
                    {
                        "datetime": "2026-02-18 15:55:00",
                        "open": "189.5",
                        "high": "190.2",
                        "low": "189.4",
                        "close": "190.0",
                        "volume": "10000",
                    },
                ]
            }
        }
    )

    df = fetch_intraday_twelve_data(
        "AAPL",
        api_key="x",
        session=session,
    )

    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume", "symbol", "provider"]
    assert df["datetime"].is_monotonic_increasing
    assert df["symbol"].unique().tolist() == ["AAPL"]
    assert pd.api.types.is_numeric_dtype(df["close"])


def test_fetch_intraday_alpha_vantage_parses_rows():
    session = DummySession(
        {
            "IBM": {
                "Time Series (5min)": {
                    "2026-02-18 16:00:00": {
                        "1. open": "250.0",
                        "2. high": "251.0",
                        "3. low": "249.5",
                        "4. close": "250.4",
                        "5. volume": "54321",
                    },
                    "2026-02-18 15:55:00": {
                        "1. open": "249.8",
                        "2. high": "250.1",
                        "3. low": "249.2",
                        "4. close": "250.0",
                        "5. volume": "45000",
                    },
                }
            }
        }
    )

    df = fetch_intraday_alpha_vantage(
        "IBM",
        api_key="x",
        session=session,
    )

    assert df["datetime"].is_monotonic_increasing
    assert df["provider"].iloc[0] == "alpha_vantage"
    assert float(df["open"].iloc[0]) == 249.8


def test_fetch_nyse_latest_open_prices_multi_ticker(monkeypatch):
    dates = pd.to_datetime(["2026-02-17", "2026-02-18"])
    cols = pd.MultiIndex.from_product(
        [["AAPL", "IBM"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    raw[("AAPL", "Open")] = [189, 190]
    raw[("IBM", "Open")] = [248, 250]

    monkeypatch.setattr("intraday_loader.yf.download", lambda *args, **kwargs: raw)

    df = fetch_nyse_latest_open_prices(
        tickers=["AAPL", "IBM"],
        batch_size=50,
        show_progress=False,
    )

    assert list(df.columns) == ["date", "ticker", "open", "provider"]
    assert set(df["ticker"].tolist()) == {"AAPL", "IBM"}
    assert str(df["date"].max()) == "2026-02-18"
    assert set(df["provider"].unique()) == {"yfinance"}


def test_fetch_nyse_latest_open_prices_single_ticker(monkeypatch):
    dates = pd.to_datetime(["2026-02-17", "2026-02-18"])
    raw = pd.DataFrame(
        {
            "Open": [248, 250],
            "High": [249, 251],
            "Low": [247, 249],
            "Close": [248.5, 250.4],
            "Adj Close": [248.4, 250.3],
            "Volume": [1900, 2000],
        },
        index=dates,
    )

    monkeypatch.setattr("intraday_loader.yf.download", lambda *args, **kwargs: raw)

    df = fetch_nyse_latest_open_prices(
        tickers=["IBM"],
        show_progress=False,
    )

    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "IBM"
    assert str(df.iloc[0]["date"]) == "2026-02-18"
    assert float(df.iloc[0]["open"]) == 250.0
