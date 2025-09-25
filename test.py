from pathlib import Path
import pandas as pd

raw_dir = Path("data/raw")
for fp in sorted(raw_dir.glob("*.parquet")):
    df = pd.read_parquet(fp, engine="fastparquet", columns=None)
    print(fp.name, "has outaum?", "outaum" in df.columns)
