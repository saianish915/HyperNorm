import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import os

Tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN"]
Start_Date = "2021-01-01"
End_Date = datetime.date.today().isoformat()
Output_File = "tech_stock_dataset.csv"


def fetch_stock_data(ticker, start, end):
    """Download raw OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end,
                     progress=False, auto_adjust=False)
    if df.empty:
        print(f"  [WARN] No data returned for {ticker}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    return df


def compute_features(df):
    """Compute returns and basic features from raw OHLCV data."""
    f = df.copy()

    # Standard daily return close to close
    f["daily_return"] = f["Close"].pct_change()

    # Log return is more statistically well behaved for modeling
    f["log_return"] = np.log(f["Close"] / f["Close"].shift(1))

    # Gap between yesterday's close and today's open captures overnight news
    f["overnight_gap"] = (f["Open"] - f["Close"].shift(1)) / f["Close"].shift(1)

    # Intraday return measures what happened during the trading day
    f["intraday_return"] = (f["Close"] - f["Open"]) / f["Open"]

    # Volume relative to 10-day average flags unusual institutional activity
    f["volume_ratio"] = f["Volume"] / f["Volume"].rolling(10).mean()

    return f


def compute_forward_returns(df):
    """
    Compute forward returns at three horizons.
    These are the prediction targets for all models.
    Shift by negative values to get future returns.
    """
    # 1 day forward return
    df["forward_return_1d"] = df["Close"].shift(-1) / df["Close"] - 1

    # 1 week forward return (5 trading days)
    df["forward_return_1w"] = df["Close"].shift(-5) / df["Close"] - 1

    # 1 month forward return (20 trading days)
    df["forward_return_1m"] = df["Close"].shift(-20) / df["Close"] - 1

    return df


def build_dataset(tickers, start, end):
    """Build the full dataset for all tickers and combine into one file."""
    all_dfs = []

    for ticker in tickers:
        print(f"Processing {ticker}...")

        raw = fetch_stock_data(ticker, start, end)
        if raw.empty:
            continue

        df = compute_features(raw)
        df = compute_forward_returns(df)

        # Add ticker column so everyone can filter by stock
        df.insert(0, "ticker", ticker)

        df = df.reset_index().rename(columns={"index": "date", "Date": "date"})
        df["date"] = pd.to_datetime(df["date"])

        all_dfs.append(df)
        print(f"  {len(df)} rows loaded "
              f"({df['date'].min().date()} to {df['date'].max().date()})")

    if not all_dfs:
        print("[ERROR] No data fetched")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)

    return combined


def main():
    print("Tech Stock Dataset Builder")
    print(f"Tickers : {', '.join(TICKERS)}")
    print(f"Period  : {START_DATE} to {END_DATE}")
    print()

    dataset = build_dataset(TICKERS, START_DATE, END_DATE)

    if dataset.empty:
        return

    dataset.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDataset saved to {OUTPUT_FILE}")
    print(f"Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(f"\nColumns:")
    for col in dataset.columns:
        print(f"  {col}")
    print(f"\nStocks:")
    for ticker in dataset["ticker"].unique():
        n = len(dataset[dataset["ticker"] == ticker])
        print(f"  {ticker}: {n} trading days")
    print(f"\nDate range: "
          f"{dataset['date'].min().date()} to {dataset['date'].max().date()}")
    print(f"\nNote: Last 20 rows per stock will have NaN in forward_return_1m")
    print(f"      Last 5 rows per stock will have NaN in forward_return_1w")
    print(f"      Last 1 row per stock will have NaN in forward_return_1d")
    print(f"      Drop NaN rows before training depending on horizon used")


if __name__ == "__main__":
    main()
