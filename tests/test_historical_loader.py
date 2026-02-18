from historical_loader import download_historical_daily_prices, fetch_nyse_tickers
import pandas as pd


class DummyResponse:
    def __init__(self, payload=None, text=""):
        self._payload = payload
        self.text = text

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class DummySession:
    def __init__(self, nyse_text, json_by_symbol):
        self.nyse_text = nyse_text
        self.json_by_symbol = json_by_symbol

    def get(self, url, params=None, timeout=20):
        if "otherlisted.txt" in url:
            return DummyResponse(text=self.nyse_text)
        symbol = (params or {}).get("symbol")
        return DummyResponse(payload=self.json_by_symbol[symbol])


def test_fetch_nyse_tickers_filters_invalid_symbols():
    nyse_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "AAPL|Apple Inc.|N|AAPL|N|100|N|AAPL\n"
        "BRK.B|Berkshire|N|BRK.B|N|100|N|BRK.B\n"
        "MSFT|Microsoft|Q|MSFT|N|100|N|MSFT\n"
        "IBM|IBM|N|IBM|N|100|N|IBM\n"
        "File Creation Time: 02182026\n"
    )
    session = DummySession(nyse_text=nyse_text, json_by_symbol={})

    tickers = fetch_nyse_tickers(session=session)
    assert tickers == ["AAPL", "IBM"]


def test_download_historical_daily_prices_yfinance_multi_ticker(monkeypatch):
    dates = pd.to_datetime(["2026-02-17", "2026-02-18"])
    cols = pd.MultiIndex.from_product(
        [["AAPL", "IBM"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    raw[("AAPL", "Open")] = [189, 190]
    raw[("AAPL", "High")] = [190, 191]
    raw[("AAPL", "Low")] = [188, 189]
    raw[("AAPL", "Close")] = [189.3, 190.5]
    raw[("AAPL", "Adj Close")] = [189.2, 190.4]
    raw[("AAPL", "Volume")] = [1100, 1000]
    raw[("IBM", "Open")] = [248, 250]
    raw[("IBM", "High")] = [249, 251]
    raw[("IBM", "Low")] = [247, 249]
    raw[("IBM", "Close")] = [248.5, 250.4]
    raw[("IBM", "Adj Close")] = [248.4, 250.3]
    raw[("IBM", "Volume")] = [1900, 2000]

    monkeypatch.setattr("historical_loader.yf.download", lambda *args, **kwargs: raw)

    df = download_historical_daily_prices(
        start_date="2026-02-17",
        end_date="2026-02-18",
        tickers=["AAPL", "IBM"],
        batch_size=50,
    )

    assert df["Ticker"].nunique() == 2
    assert list(df.columns) == ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Provider"]
    assert df.iloc[0]["Ticker"] == "AAPL"
    assert str(df["Date"].max()) == "2026-02-18"
    assert set(df["Provider"].unique()) == {"yfinance"}


def test_download_historical_daily_prices_yfinance_single_ticker(monkeypatch):
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

    monkeypatch.setattr("historical_loader.yf.download", lambda *args, **kwargs: raw)

    df = download_historical_daily_prices(
        start_date="2026-02-17",
        end_date="2026-02-18",
        tickers=["IBM"],
    )

    assert len(df) == 2
    assert str(df["Date"].min()) == "2026-02-17"
    assert str(df["Date"].max()) == "2026-02-18"
    assert df["Ticker"].unique().tolist() == ["IBM"]


def test_download_historical_daily_prices_retries_missing_ticker_in_batch(monkeypatch):
    dates = pd.to_datetime(["2026-02-17", "2026-02-18"])
    cols_aapl = pd.MultiIndex.from_product(
        [["AAPL"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    first = pd.DataFrame(index=dates, columns=cols_aapl, dtype=float)
    first[("AAPL", "Open")] = [189, 190]
    first[("AAPL", "High")] = [190, 191]
    first[("AAPL", "Low")] = [188, 189]
    first[("AAPL", "Close")] = [189.3, 190.5]
    first[("AAPL", "Adj Close")] = [189.2, 190.4]
    first[("AAPL", "Volume")] = [1100, 1000]

    cols_ibm = pd.MultiIndex.from_product(
        [["IBM"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    second = pd.DataFrame(index=dates, columns=cols_ibm, dtype=float)
    second[("IBM", "Open")] = [248, 250]
    second[("IBM", "High")] = [249, 251]
    second[("IBM", "Low")] = [247, 249]
    second[("IBM", "Close")] = [248.5, 250.4]
    second[("IBM", "Adj Close")] = [248.4, 250.3]
    second[("IBM", "Volume")] = [1900, 2000]

    calls = {"count": 0}

    def fake_download(batch, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            assert batch == ["AAPL", "IBM"]
            return first
        assert batch == ["IBM"]
        return second

    monkeypatch.setattr("historical_loader.yf.download", fake_download)

    df = download_historical_daily_prices(
        start_date="2026-02-17",
        end_date="2026-02-18",
        tickers=["AAPL", "IBM"],
        batch_size=2,
        retry_backoff_seconds=0,
    )

    assert calls["count"] == 2
    assert set(df["Ticker"].unique()) == {"AAPL", "IBM"}


def test_download_historical_daily_prices_retries_on_exception(monkeypatch):
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

    calls = {"count": 0}

    def fake_download(batch, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("rate limit")
        assert batch == ["IBM"]
        return raw

    monkeypatch.setattr("historical_loader.yf.download", fake_download)

    df = download_historical_daily_prices(
        start_date="2026-02-17",
        end_date="2026-02-18",
        tickers=["IBM"],
        max_retries_per_ticker=2,
        retry_backoff_seconds=0,
    )

    assert calls["count"] == 2
    assert df["Ticker"].unique().tolist() == ["IBM"]


def test_download_historical_daily_prices_raises_when_retry_budget_exhausted(monkeypatch):
    def fake_download(batch, **kwargs):
        raise RuntimeError("rate limit")

    monkeypatch.setattr("historical_loader.yf.download", fake_download)

    try:
        download_historical_daily_prices(
            start_date="2026-02-17",
            end_date="2026-02-18",
            tickers=["AAPL"],
            max_retries_per_ticker=1,
            retry_backoff_seconds=0,
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "Failed to download 1 ticker(s) after retries: AAPL" in str(exc)


def test_download_historical_daily_prices_retries_no_data_before_confirming(monkeypatch):
    dates = pd.to_datetime(["2026-02-17", "2026-02-18"])
    cols = pd.MultiIndex.from_product(
        [["IBM"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    empty_ibm = pd.DataFrame(index=dates, columns=cols, dtype=float)
    full_ibm = empty_ibm.copy()
    full_ibm[("IBM", "Open")] = [248, 250]
    full_ibm[("IBM", "High")] = [249, 251]
    full_ibm[("IBM", "Low")] = [247, 249]
    full_ibm[("IBM", "Close")] = [248.5, 250.4]
    full_ibm[("IBM", "Adj Close")] = [248.4, 250.3]
    full_ibm[("IBM", "Volume")] = [1900, 2000]

    calls = {"count": 0}

    def fake_download(batch, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return empty_ibm
        return full_ibm

    monkeypatch.setattr("historical_loader.yf.download", fake_download)

    df = download_historical_daily_prices(
        start_date="2026-02-17",
        end_date="2026-02-18",
        tickers=["IBM"],
        no_data_confirmations=2,
        retry_backoff_seconds=0,
    )

    assert calls["count"] == 2
    assert len(df) == 2
    assert df["Ticker"].unique().tolist() == ["IBM"]


def test_download_historical_daily_prices_deduplicates_repeated_ticker_rows(monkeypatch):
    dates = pd.to_datetime(["2026-02-17", "2026-02-18"])
    cols = pd.MultiIndex.from_product(
        [["IBM"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    raw = pd.DataFrame(index=dates, columns=cols, dtype=float)
    raw[("IBM", "Open")] = [248, 250]
    raw[("IBM", "High")] = [249, 251]
    raw[("IBM", "Low")] = [247, 249]
    raw[("IBM", "Close")] = [248.5, 250.4]
    raw[("IBM", "Adj Close")] = [248.4, 250.3]
    raw[("IBM", "Volume")] = [1900, 2000]

    calls = {"count": 0}

    def fake_download(batch, **kwargs):
        calls["count"] += 1
        return raw

    monkeypatch.setattr("historical_loader.yf.download", fake_download)

    df = download_historical_daily_prices(
        start_date="2026-02-17",
        end_date="2026-02-18",
        tickers=["IBM", "IBM"],
        retry_backoff_seconds=0,
    )

    assert len(df) == 2
    assert df[["Ticker", "Date"]].duplicated().sum() == 0
