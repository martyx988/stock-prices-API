from ticker_loader import fetch_nyse_tickers_with_names


class DummyResponse:
    def __init__(self, text=""):
        self.text = text

    def raise_for_status(self):
        return None


class DummySession:
    def __init__(self, nyse_text):
        self.nyse_text = nyse_text

    def get(self, url, params=None, timeout=20):
        return DummyResponse(text=self.nyse_text)


def test_fetch_nyse_tickers_with_names_filters_and_sorts():
    nyse_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "IBM|International Business Machines|N|IBM|N|100|N|IBM\n"
        "BRK.B|Berkshire Hathaway|N|BRK.B|N|100|N|BRK.B\n"
        "AAPL|Apple Inc.|N|AAPL|N|100|N|AAPL\n"
        "MSFT|Microsoft Corporation|Q|MSFT|N|100|N|MSFT\n"
        "AAPL|Apple Inc. Duplicate|N|AAPL|N|100|N|AAPL\n"
        "File Creation Time: 02182026\n"
    )
    session = DummySession(nyse_text=nyse_text)

    df = fetch_nyse_tickers_with_names(session=session)

    assert list(df.columns) == ["Ticker", "CompanyName"]
    assert df["Ticker"].tolist() == ["AAPL", "IBM"]
    assert df.iloc[0]["CompanyName"] == "Apple Inc."


def test_fetch_nyse_tickers_with_names_limit():
    nyse_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "AAPL|Apple Inc.|N|AAPL|N|100|N|AAPL\n"
        "IBM|International Business Machines|N|IBM|N|100|N|IBM\n"
        "TSLA|Tesla Inc.|N|TSLA|N|100|N|TSLA\n"
        "File Creation Time: 02182026\n"
    )
    session = DummySession(nyse_text=nyse_text)

    df = fetch_nyse_tickers_with_names(limit=2, session=session)

    assert len(df) == 2
    assert df["Ticker"].tolist() == ["AAPL", "IBM"]
