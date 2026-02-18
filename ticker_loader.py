import io
from typing import Optional

import pandas as pd
import requests


def fetch_nyse_tickers_with_names(
    limit: Optional[int] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> pd.DataFrame:
    client = session or requests
    response = client.get(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        timeout=timeout,
    )
    response.raise_for_status()

    ticker_df = pd.read_csv(io.StringIO(response.text), sep="|")
    nyse = ticker_df[ticker_df["Exchange"] == "N"][["ACT Symbol", "Security Name"]].copy()
    nyse.columns = ["Ticker", "CompanyName"]
    nyse["Ticker"] = nyse["Ticker"].astype(str).str.strip()
    nyse["CompanyName"] = nyse["CompanyName"].astype(str).str.strip()
    nyse = nyse[nyse["Ticker"].str.isalpha() & (nyse["Ticker"].str.len() <= 5)]
    nyse = nyse.drop_duplicates(subset=["Ticker"], keep="first")
    nyse = nyse.sort_values("Ticker").reset_index(drop=True)

    if limit is not None:
        return nyse.head(limit).reset_index(drop=True)
    return nyse
