import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

# 1. 定义指数 ticker
ticker = "^GSPC"  # S&P 500 Index

# 2. 下载历史数据
df = yf.download(
    ticker,
    start="1990-01-01",
    auto_adjust=False,
    progress=False
)

# 3. 只保留成交量
df = df.reset_index()[["Date", "Volume"]]

# 4. 标准化列名，与你数据库一致
df = df.rename(columns={
    "Date": "fdate",
    "Volume": "market_volume"
})

# ⭐ 关键：彻底移除 MultiIndex 残留
df.columns = ["fdate", "market_volume"]

# 5. 确保日期格式统一
df["fdate"] = pd.to_datetime(df["fdate"])

# 6. log 成交量（核心步骤）
df["ln_market_volume"] = np.log(df["market_volume"])

# 7. 创建输出目录并保存
out_dir = Path("./data")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / "sp500_market_volume.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(df.tail())
print(f"Saved to: {out_path.resolve()}")