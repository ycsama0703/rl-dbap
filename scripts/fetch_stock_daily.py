#!/usr/bin/env python
"""
Fetch daily price/volume for a list of tickers via yfinance and save a compact parquet.

Example:
  python scripts/fetch_stock_daily.py \
    --mapping data/permno_ticker.csv \
    --start 2015-01-01 --end 2024-12-31 \
    --out data/stock_daily.parquet

`--mapping` CSV format (header required):
    permno,ticker
    10107,AAPL
    ...
If no mapping is provided, tickers from --tickers will be used (permno will be NA).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download daily Close / AdjClose / Volume for given tickers via yfinance."
    )
    p.add_argument("--mapping", type=str, default="", help="Optional CSV with columns permno,ticker.")
    p.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,GOOGL",
        help="Comma-separated tickers (used if mapping not provided).",
    )
    p.add_argument("--start", type=str, default="2015-01-01", help="Start date (inclusive).")
    p.add_argument("--end", type=str, default="2024-12-31", help="End date (inclusive).")
    p.add_argument("--out", type=str, default="data/stock_daily.parquet", help="Output parquet path.")
    return p.parse_args()


def load_mapping(path: Path, tickers_fallback: list[str]) -> pd.DataFrame:
    """Load permno<->ticker mapping."""
    if not path or not path.exists():
        return pd.DataFrame(
            {
                "permno": pd.Series([pd.NA] * len(tickers_fallback), dtype="Int64"),
                "ticker": tickers_fallback,
            }
        )

    df = pd.read_csv(path)
    if not {"permno", "ticker"} <= set(df.columns):
        raise ValueError("mapping CSV must contain columns permno,ticker")

    df = df[["permno", "ticker"]]
    df["permno"] = df["permno"].astype("Int64")
    return df


def normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize yfinance output to long form with columns:
    date, ticker, Close, AdjClose, Volume
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker case: ticker is column level=1
        try:
            # pandas >= 2.1
            df_long = df.stack(level=1, future_stack=True).reset_index()
        except TypeError:
            # pandas < 2.1
            df_long = df.stack(level=1).reset_index()

        # rename by position, not by guessed names
        df_long = df_long.rename(
            columns={
                df_long.columns[0]: "date",
                df_long.columns[1]: "ticker",
            }
        )
    else:
        # Single ticker case
        df_long = df.reset_index()
        df_long = df_long.rename(columns={df_long.columns[0]: "date"})
        df_long["ticker"] = None

    # unify column names
    df_long = df_long.rename(columns={"Adj Close": "AdjClose"})

    required = {"date", "ticker", "Close", "AdjClose", "Volume"}
    missing = required - set(df_long.columns)
    if missing:
        raise RuntimeError(f"Missing columns after download: {missing}")

    return df_long[["date", "ticker", "Close", "AdjClose", "Volume"]]
def main() -> None:
    args = parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    mapping = load_mapping(Path(args.mapping), tickers)

    tickers_dl = mapping["ticker"].dropna().astype(str).unique().tolist()
    if not tickers_dl:
        raise RuntimeError("No tickers provided.")

    # yfinance 'end' is exclusive -> shift by +1 day
    end_exclusive = (
        pd.to_datetime(args.end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    df_raw = yf.download(
        tickers_dl,
        start=args.start,
        end=end_exclusive,
        interval="1d",
        auto_adjust=False,   # IMPORTANT: keep AdjClose from yfinance
        progress=False,
    )

    if df_raw.empty:
        raise RuntimeError("No data downloaded; check tickers or network.")

    df_long = normalize_download(df_raw)

    # attach permno
    mapping_dict = dict(zip(mapping["ticker"].astype(str), mapping["permno"]))
    df_long["permno"] = (
        df_long["ticker"].map(mapping_dict).astype("Int64")
    )

    df_long = df_long[
        ["permno", "ticker", "date", "Close", "AdjClose", "Volume"]
    ].sort_values(["ticker", "date"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(out_path, index=False)

    print(f"[ok] wrote {len(df_long):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
