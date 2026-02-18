import io
import re
from typing import Optional

import pandas as pd
import requests


def _classify_security_type(etf_flag: str, company_name: str) -> str:
    flag = str(etf_flag).strip().upper()
    name = str(company_name).strip().upper()
    if flag == "Y":
        if re.search(r"\bETN\b|EXCHANGE TRADED NOTE", name):
            return "ETN"
        if re.search(r"\bETC\b|EXCHANGE TRADED COMMODITY", name):
            return "ETC"
        return "ETF"
    return "STOCK"


def fetch_nyse_tickers_with_names(
    limit: Optional[int] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
    include_types: Optional[set[str]] = None,
    include_exchanges: Optional[set[str]] = None,
) -> pd.DataFrame:
    client = session or requests
    response = client.get(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        timeout=timeout,
    )
    response.raise_for_status()

    ticker_df = pd.read_csv(io.StringIO(response.text), sep="|")
    # NYSE family: N=NYSE, P=NYSE Arca, A=NYSE American
    allowed_exchanges = {value.upper() for value in (include_exchanges or {"N", "P", "A"})}
    nyse = ticker_df[ticker_df["Exchange"].astype(str).str.upper().isin(allowed_exchanges)][
        ["ACT Symbol", "Security Name", "ETF"]
    ].copy()
    nyse.columns = ["Ticker", "CompanyName", "ETF"]
    nyse["Ticker"] = nyse["Ticker"].astype(str).str.strip()
    nyse["CompanyName"] = nyse["CompanyName"].astype(str).str.strip()
    nyse["Type"] = nyse.apply(
        lambda row: _classify_security_type(row["ETF"], row["CompanyName"]),
        axis=1,
    )
    nyse = nyse[nyse["Ticker"].str.isalpha() & (nyse["Ticker"].str.len() <= 5)]
    allowed_types = {value.upper() for value in (include_types or {"STOCK", "ETF", "ETN", "ETC"})}
    nyse = nyse[nyse["Type"].isin(allowed_types)]
    nyse = nyse.drop_duplicates(subset=["Ticker"], keep="first")
    nyse = nyse[["Ticker", "CompanyName", "Type"]]
    nyse = nyse.sort_values("Ticker").reset_index(drop=True)

    if limit is not None:
        return nyse.head(limit).reset_index(drop=True)
    return nyse
