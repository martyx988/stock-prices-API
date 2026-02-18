import io
import os
import time
import contextlib
from collections import deque
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is not available
    tqdm = None


def fetch_nyse_tickers(
    limit: Optional[int] = None,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
) -> list[str]:
    client = session or requests
    response = client.get(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        timeout=timeout,
    )
    response.raise_for_status()
    ticker_df = pd.read_csv(io.StringIO(response.text), sep="|")

    nyse = ticker_df[ticker_df["Exchange"] == "N"]["ACT Symbol"].dropna().astype(str).tolist()
    nyse = [ticker for ticker in nyse if ticker.isalpha() and len(ticker) <= 5]
    nyse = sorted(set(nyse))

    if limit is not None:
        return nyse[:limit]
    return nyse


def _extract_yfinance_batch(
    raw: pd.DataFrame, batch: list[str]
) -> tuple[pd.DataFrame, set[str], set[str], set[str]]:
    """Return normalized frame and ticker classification for one yfinance batch."""
    frames: list[pd.DataFrame] = []
    success: set[str] = set()
    no_data: set[str] = set()
    unresolved: set[str] = set()

    if raw.empty:
        return pd.DataFrame(), success, no_data, set(batch)

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        index_name = raw.index.name or "index"
        for ticker in batch:
            if ticker not in level0:
                unresolved.add(ticker)
                continue
            ticker_df = raw[ticker].dropna(how="all")
            if ticker_df.empty:
                no_data.add(ticker)
            else:
                success.add(ticker)
                ticker_out = ticker_df.reset_index().rename(columns={index_name: "Date"})
                ticker_out["Ticker"] = ticker
                frames.append(ticker_out)
    else:
        ticker = batch[0]
        ticker_df = raw.dropna(how="all")
        if ticker_df.empty:
            no_data.add(ticker)
        else:
            success.add(ticker)
            index_name = raw.index.name or "index"
            ticker_out = ticker_df.reset_index().rename(columns={index_name: "Date"})
            ticker_out["Ticker"] = ticker
            frames.append(ticker_out)

    if not frames:
        return pd.DataFrame(), success, no_data, unresolved

    combined = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame(
        {
            "Date": pd.to_datetime(combined["Date"]).dt.date,
            "Ticker": combined["Ticker"],
            "Open": pd.to_numeric(combined.get("Open"), errors="coerce"),
            "High": pd.to_numeric(combined.get("High"), errors="coerce"),
            "Low": pd.to_numeric(combined.get("Low"), errors="coerce"),
            "Close": pd.to_numeric(combined.get("Close"), errors="coerce"),
            "Volume": pd.to_numeric(combined.get("Volume"), errors="coerce"),
            "Adj Close": pd.to_numeric(combined.get("Adj Close"), errors="coerce"),
            "Provider": "yfinance",
        }
    )
    return out, success, no_data, unresolved


def _format_eta(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def download_historical_daily_prices(
    start_date: str,
    end_date: str,
    tickers: Optional[list[str]] = None,
    nyse_limit: Optional[int] = None,
    batch_size: int = 50,
    delay_seconds: float = 0.0,
    session: Optional[requests.Session] = None,
    timeout: int = 20,
    combined_csv_path: Optional[str] = None,
    individual_dir: Optional[str] = None,
    show_progress: bool = True,
    max_retries_per_ticker: int = 4,
    retry_backoff_seconds: float = 1.5,
    no_data_confirmations: int = 2,
    silence_yfinance_output: bool = True,
    yfinance_threads: bool = True,
) -> pd.DataFrame:
    if tickers is None:
        tickers = fetch_nyse_tickers(limit=nyse_limit, session=session, timeout=timeout)
    if not tickers:
        raise ValueError("No tickers to download")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if max_retries_per_ticker < 0:
        raise ValueError("max_retries_per_ticker must be >= 0")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be >= 0")
    if no_data_confirmations < 1:
        raise ValueError("no_data_confirmations must be >= 1")

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    end_exclusive = end + timedelta(days=1)
    frames = []
    pending = deque(dict.fromkeys(tickers))
    total_tickers = len(pending)
    finalized = 0
    attempt_counts = {ticker: 0 for ticker in pending}
    no_data_counts = {ticker: 0 for ticker in pending}
    failed_tickers: set[str] = set()
    start_ts = time.time()
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(total=total_tickers, desc="Historical download", unit="ticker")
    devnull_stream = open(os.devnull, "w") if silence_yfinance_output else None

    try:
        batch_index = 0
        while pending:
            batch_index += 1
            batch = [pending.popleft() for _ in range(min(batch_size, len(pending)))]
            batch_label = f"batch {batch_index}"
            if show_progress and progress_bar is not None:
                progress_bar.set_description(f"Historical download ({batch_label})")
                progress_bar.set_postfix_str(f"status=downloading | pending={len(pending) + len(batch)}")
            elif show_progress:
                elapsed = time.time() - start_ts
                print(
                    f"Starting {batch_label} ({len(batch)} tickers) | "
                    f"finalized {finalized}/{total_tickers} | elapsed {_format_eta(elapsed)}"
                )

            try:
                if silence_yfinance_output and devnull_stream is not None:
                    with contextlib.redirect_stdout(devnull_stream), contextlib.redirect_stderr(devnull_stream):
                        raw = yf.download(
                            batch,
                            start=start.isoformat(),
                            end=end_exclusive.isoformat(),
                            auto_adjust=False,
                            progress=False,
                            group_by="ticker",
                            threads=yfinance_threads,
                        )
                else:
                    raw = yf.download(
                        batch,
                        start=start.isoformat(),
                        end=end_exclusive.isoformat(),
                        auto_adjust=False,
                        progress=False,
                        group_by="ticker",
                        threads=yfinance_threads,
                    )
            except Exception:
                retry_candidates = set(batch)
                newly_finalized = 0
                max_attempt_in_batch = 0
                for ticker in retry_candidates:
                    no_data_counts[ticker] = 0
                    attempt_counts[ticker] += 1
                    max_attempt_in_batch = max(max_attempt_in_batch, attempt_counts[ticker])
                    if attempt_counts[ticker] <= max_retries_per_ticker:
                        pending.append(ticker)
                    else:
                        failed_tickers.add(ticker)
                        newly_finalized += 1
                finalized += newly_finalized
                if progress_bar is not None and newly_finalized:
                    progress_bar.update(newly_finalized)
                if retry_backoff_seconds and retry_candidates:
                    backoff = retry_backoff_seconds * (2 ** (max_attempt_in_batch - 1))
                    time.sleep(min(backoff, 60.0))
                continue

            frame, success_tickers, no_data_batch, unresolved = _extract_yfinance_batch(raw, batch)
            if not frame.empty:
                frames.append(frame)
            no_data_confirmed: set[str] = set()
            no_data_retry: set[str] = set()

            for ticker in success_tickers:
                no_data_counts[ticker] = 0

            for ticker in no_data_batch:
                no_data_counts[ticker] += 1
                if no_data_counts[ticker] >= no_data_confirmations:
                    no_data_confirmed.add(ticker)
                else:
                    no_data_retry.add(ticker)

            newly_finalized = len(success_tickers) + len(no_data_confirmed)
            finalized += newly_finalized
            elapsed = time.time() - start_ts
            avg_per_ticker = elapsed / finalized if finalized else 0
            eta_seconds = avg_per_ticker * (total_tickers - finalized)
            if progress_bar is not None:
                if newly_finalized:
                    progress_bar.update(newly_finalized)
                progress_bar.set_postfix_str(
                    f"status=done | pending={len(pending) + len(unresolved) + len(no_data_retry)} | ETA {_format_eta(eta_seconds)}"
                )
            elif show_progress:
                pct = finalized / total_tickers * 100
                print(
                    f"[{finalized}/{total_tickers}] {pct:5.1f}% complete | "
                    f"elapsed {_format_eta(elapsed)} | ETA {_format_eta(eta_seconds)}"
                )

            max_attempt_in_batch = 0
            for ticker in no_data_retry:
                pending.append(ticker)
                max_attempt_in_batch = max(max_attempt_in_batch, no_data_counts[ticker])

            for ticker in unresolved:
                no_data_counts[ticker] = 0
                attempt_counts[ticker] += 1
                max_attempt_in_batch = max(max_attempt_in_batch, attempt_counts[ticker])
                if attempt_counts[ticker] <= max_retries_per_ticker:
                    pending.append(ticker)
                else:
                    failed_tickers.add(ticker)
                    finalized += 1
                    if progress_bar is not None:
                        progress_bar.update(1)

            if retry_backoff_seconds and (unresolved or no_data_retry):
                backoff = retry_backoff_seconds * (2 ** (max_attempt_in_batch - 1))
                time.sleep(min(backoff, 60.0))
            elif delay_seconds and pending:
                time.sleep(delay_seconds)
    finally:
        if devnull_stream is not None:
            devnull_stream.close()

    if progress_bar is not None:
        progress_bar.close()

    if failed_tickers:
        failed_sample = ", ".join(sorted(failed_tickers)[:20])
        raise RuntimeError(
            f"Failed to download {len(failed_tickers)} ticker(s) after retries: {failed_sample}"
        )

    if not frames:
        raise RuntimeError("No historical price data returned from yfinance")

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.date
    combined = combined.drop_duplicates(subset=["Ticker", "Date"], keep="last")
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if combined_csv_path:
        combined.to_csv(combined_csv_path, index=False)

    if individual_dir:
        os.makedirs(individual_dir, exist_ok=True)
        for ticker, group in combined.groupby("Ticker"):
            group.to_csv(os.path.join(individual_dir, f"{ticker}.csv"), index=False)

    return combined
