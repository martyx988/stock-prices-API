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
        "GLD|SPDR Gold Shares|N|GLD|Y|100|N|GLD\n"
        "ETNN|Random Income ETN|N|ETNN|Y|100|N|ETNN\n"
        "ETCX|Silver Basket ETC|N|ETCX|Y|100|N|ETCX\n"
        "BRK.B|Berkshire Hathaway|N|BRK.B|N|100|N|BRK.B\n"
        "AAPL|Apple Inc.|N|AAPL|N|100|N|AAPL\n"
        "MSFT|Microsoft Corporation|Q|MSFT|N|100|N|MSFT\n"
        "AAPL|Apple Inc. Duplicate|N|AAPL|N|100|N|AAPL\n"
        "File Creation Time: 02182026\n"
    )
    session = DummySession(nyse_text=nyse_text)

    df = fetch_nyse_tickers_with_names(session=session)

    assert list(df.columns) == ["Ticker", "CompanyName", "Type"]
    assert df["Ticker"].tolist() == ["AAPL", "ETCX", "ETNN", "GLD", "IBM"]
    assert df.iloc[0]["CompanyName"] == "Apple Inc."
    assert dict(zip(df["Ticker"], df["Type"])) == {
        "AAPL": "STOCK",
        "ETCX": "ETC",
        "ETNN": "ETN",
        "GLD": "ETF",
        "IBM": "STOCK",
    }


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


def test_fetch_nyse_tickers_with_names_include_types_filter():
    nyse_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "AAPL|Apple Inc.|N|AAPL|N|100|N|AAPL\n"
        "GLD|SPDR Gold Shares|N|GLD|Y|100|N|GLD\n"
        "ETNN|Random Income ETN|N|ETNN|Y|100|N|ETNN\n"
        "ETCX|Silver Basket ETC|N|ETCX|Y|100|N|ETCX\n"
        "File Creation Time: 02182026\n"
    )
    session = DummySession(nyse_text=nyse_text)

    df = fetch_nyse_tickers_with_names(include_types={"STOCK", "ETF"}, session=session)

    assert df["Ticker"].tolist() == ["AAPL", "GLD"]
    assert df["Type"].tolist() == ["STOCK", "ETF"]


def test_fetch_nyse_tickers_with_names_includes_arca_by_default():
    nyse_text = (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "AAPL|Apple Inc.|N|AAPL|N|100|N|AAPL\n"
        "GLD|SPDR Gold Shares|P|GLD|Y|100|N|GLD\n"
        "File Creation Time: 02182026\n"
    )
    session = DummySession(nyse_text=nyse_text)

    df = fetch_nyse_tickers_with_names(session=session)

    assert df["Ticker"].tolist() == ["AAPL", "GLD"]
    assert df["Type"].tolist() == ["STOCK", "ETF"]
