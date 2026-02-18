import os
import time
import contextlib
from datetime import date
from typing import Optional
import io

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from historical_loader import fetch_nyse_tickers
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is not available
    tqdm = None

load_dotenv()


def _require_api_key(provided_key: Optional[str], env_name: str) -> str:
    api_key = provided_key or os.getenv(env_name)
    if not api_key:
        raise ValueError(f"Missing API key: {env_name}")
    return api_key


def fetch_intraday_twelve_data(
    symbol: str,
    interval: str = "5min",
    outputsize: int = 100,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> pd.DataFrame:
    key = _require_api_key(api_key, "TWELVE_DATA_API_KEY")
    client = session or requests

    response = client.get(
        "https://api.twelvedata.com/time_series",
        params={
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": key,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    values = data.get("values")
    if not isinstance(values, list) or not values:
        message = data.get("message", "Unexpected response format")
        raise RuntimeError(f"Twelve Data request failed for {symbol}: {message}")

    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol
    df["provider"] = "twelve_data"
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_intraday_alpha_vantage(
    symbol: str,
    interval: str = "5min",
    outputsize: str = "compact",
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> pd.DataFrame:
    key = _require_api_key(api_key, "ALPHA_VANTAGE_API_KEY")
    client = session or requests

    response = client.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": key,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info for {symbol}: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage request failed for {symbol}: {data['Error Message']}")

    series_key = f"Time Series ({interval})"
    series = data.get(series_key)
    if not isinstance(series, dict) or not series:
        raise RuntimeError(f"Alpha Vantage request failed for {symbol}: Unexpected response format")

    rows = []
    for dt_str, payload in series.items():
        rows.append(
            {
                "datetime": pd.to_datetime(dt_str),
                "open": pd.to_numeric(payload.get("1. open"), errors="coerce"),
                "high": pd.to_numeric(payload.get("2. high"), errors="coerce"),
                "low": pd.to_numeric(payload.get("3. low"), errors="coerce"),
                "close": pd.to_numeric(payload.get("4. close"), errors="coerce"),
                "volume": pd.to_numeric(payload.get("5. volume"), errors="coerce"),
            }
        )

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df["symbol"] = symbol
    df["provider"] = "alpha_vantage"
    return df


def fetch_daily_alpha_vantage_latest_date(
    symbol: str,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> str:
    key = _require_api_key(api_key, "ALPHA_VANTAGE_API_KEY")
    client = session or requests
    response = client.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": key,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage info for {symbol}: {data['Information']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit for {symbol}: {data['Note']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage request failed for {symbol}: {data['Error Message']}")

    series = data.get("Time Series (Daily)")
    if not isinstance(series, dict) or not series:
        raise RuntimeError(f"Alpha Vantage request failed for {symbol}: Unexpected response format")
    return sorted(series.keys(), reverse=True)[0]


def _extract_open_prices_from_daily_batch(raw: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "provider"])

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        for ticker in batch:
            if ticker not in level0:
                continue
            ticker_df = raw[ticker].dropna(how="all")
            if ticker_df.empty or "Open" not in ticker_df.columns:
                continue
            latest_idx = ticker_df.index.max()
            open_value = pd.to_numeric(ticker_df.loc[latest_idx, "Open"], errors="coerce")
            if pd.isna(open_value):
                continue
            rows.append(
                {
                    "date": pd.to_datetime(latest_idx).date(),
                    "ticker": ticker,
                    "open": float(open_value),
                    "provider": "yfinance",
                }
            )
    else:
        ticker = batch[0]
        ticker_df = raw.dropna(how="all")
        if not ticker_df.empty and "Open" in ticker_df.columns:
            latest_idx = ticker_df.index.max()
            open_value = pd.to_numeric(ticker_df.loc[latest_idx, "Open"], errors="coerce")
            if not pd.isna(open_value):
                rows.append(
                    {
                        "date": pd.to_datetime(latest_idx).date(),
                        "ticker": ticker,
                        "open": float(open_value),
                        "provider": "yfinance",
                    }
                )

    return pd.DataFrame(rows, columns=["date", "ticker", "open", "provider"])


def _format_eta(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def fetch_nyse_latest_open_prices(
    tickers: Optional[list[str]] = None,
    nyse_limit: Optional[int] = None,
    batch_size: int = 100,
    show_progress: bool = True,
    silence_yfinance_output: bool = True,
) -> pd.DataFrame:
    if tickers is None:
        tickers = fetch_nyse_tickers(limit=nyse_limit)
    if not tickers:
        raise ValueError("No tickers to fetch")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    unique_tickers = list(dict.fromkeys(tickers))
    frames: list[pd.DataFrame] = []
    total_tickers = len(unique_tickers)
    completed = 0
    start_ts = time.time()
    progress_bar = None
    total_batches = (total_tickers + batch_size - 1) // batch_size
    if show_progress and tqdm is not None:
        progress_bar = tqdm(total=total_tickers, desc="Intraday open download", unit="ticker")
    devnull_stream = open(os.devnull, "w") if silence_yfinance_output else None

    try:
        for batch_index, start in enumerate(range(0, total_tickers, batch_size)):
            batch = unique_tickers[start : start + batch_size]
            batch_label = f"batch {batch_index + 1}/{total_batches}"
            if progress_bar is not None:
                progress_bar.set_description(f"Intraday open download ({batch_label})")
                progress_bar.set_postfix_str("status=downloading...")
                progress_bar.refresh()
            elif show_progress:
                elapsed = time.time() - start_ts
                print(
                    f"Starting {batch_label} ({len(batch)} tickers) | "
                    f"completed {completed}/{total_tickers} | elapsed {_format_eta(elapsed)}"
                )

            if silence_yfinance_output and devnull_stream is not None:
                with contextlib.redirect_stdout(devnull_stream), contextlib.redirect_stderr(devnull_stream):
                    raw = yf.download(
                        batch,
                        period="5d",
                        interval="1d",
                        auto_adjust=False,
                        progress=False,
                        group_by="ticker",
                        threads=True,
                    )
            else:
                raw = yf.download(
                    batch,
                    period="5d",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )

            frame = _extract_open_prices_from_daily_batch(raw, batch)
            if not frame.empty:
                frames.append(frame)

            completed += len(batch)
            elapsed = time.time() - start_ts
            avg_per_ticker = elapsed / completed if completed else 0
            eta_seconds = avg_per_ticker * (total_tickers - completed)
            if progress_bar is not None:
                progress_bar.update(len(batch))
                progress_bar.set_postfix_str(f"status=done | ETA {_format_eta(eta_seconds)}")
            elif show_progress:
                pct = completed / total_tickers * 100
                print(
                    f"[{completed}/{total_tickers}] {pct:5.1f}% complete | "
                    f"elapsed {_format_eta(elapsed)} | ETA {_format_eta(eta_seconds)}"
                )
    finally:
        if devnull_stream is not None:
            devnull_stream.close()

    if progress_bar is not None:
        progress_bar.close()

    if not frames:
        raise RuntimeError("No opening prices returned from yfinance")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    latest_date = combined["date"].max()
    return combined[combined["date"] == latest_date].reset_index(drop=True)


def smoke_test_connections(
    twelve_symbol: str = "AAPL",
    alpha_symbol: str = "IBM",
) -> list[tuple[bool, str]]:
    results: list[tuple[bool, str]] = []

    try:
        td = fetch_intraday_twelve_data(twelve_symbol, interval="1day", outputsize=2)
        latest = td["datetime"].max().strftime("%Y-%m-%d")
        results.append((True, f"Twelve Data OK. Latest date for {twelve_symbol}: {latest}"))
    except Exception as exc:
        results.append((False, f"Twelve Data failed: {exc}"))

    try:
        latest = fetch_daily_alpha_vantage_latest_date(alpha_symbol)
        results.append((True, f"Alpha Vantage OK. Latest date for {alpha_symbol}: {latest}"))
    except Exception as exc:
        results.append((False, f"Alpha Vantage failed: {exc}"))

    return results
