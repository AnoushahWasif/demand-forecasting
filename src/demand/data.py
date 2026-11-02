"""Download Online Retail II, check it, and cache both sheets as one Parquet file.

The zip's size and SHA-256 are checked before it is opened, and the one expected member is
read with a size cap, so a swapped or oversized file never reaches the xlsx parser.
"""

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
MAX_ZIP_BYTES = 60_000_000  # the real file is 45.6 MB
MAX_XLSX_BYTES = 100_000_000
MEMBER = "online_retail_II.xlsx"

COLUMNS = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
           "Customer ID", "Country"]  # fmt: skip
DTYPES = {"Invoice": str, "StockCode": str, "Description": str, "Country": str}

# The two sheets overlap: "Year 2009-2010" runs to 9 Dec 2010 and "Year 2010-2011" starts on
# 1 Dec 2010 with the same 1,088 invoices. Each sheet is used only for its own side of this date.
SHEET_SPLIT = pd.Timestamp("2010-12-01")


def data_dir() -> Path:
    return Path(os.environ.get("DEMAND_DATA_DIR", "data"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_zip(path: Path, expected_sha256: str = SHA256) -> None:
    size = path.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise ValueError(f"{path} is {size:,} bytes, more than the {MAX_ZIP_BYTES:,} allowed")
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}")


def download(dest: Path | None = None) -> Path:
    folder = (dest or data_dir()) / "raw"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "online_retail_ii.zip"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    check_zip(path)
    return path


def extract_xlsx(zip_path: Path, out_dir: Path) -> Path:
    """Copy out only the expected member, by name, with a size cap. Other members are ignored."""
    out = out_dir / MEMBER
    with zipfile.ZipFile(zip_path) as zf:
        try:
            info = zf.getinfo(MEMBER)
        except KeyError as exc:
            raise ValueError(f"{zip_path} has no {MEMBER}") from exc
        if info.file_size > MAX_XLSX_BYTES:
            raise ValueError(f"{MEMBER} would unpack to {info.file_size:,} bytes")
        written = 0
        with zf.open(info) as src, out.open("wb") as dst:
            # file_size comes from the zip header, which can lie, so count while copying too.
            for chunk in iter(lambda: src.read(1 << 20), b""):
                written += len(chunk)
                if written > MAX_XLSX_BYTES:
                    dst.close()
                    out.unlink()
                    raise ValueError(f"{MEMBER} is larger than its header says")
                dst.write(chunk)
    return out


def check_columns(df: pd.DataFrame, where: str) -> None:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}")


def combine_sheets(early: pd.DataFrame, late: pd.DataFrame) -> pd.DataFrame:
    for name, df in (("early sheet", early), ("late sheet", late)):
        check_columns(df, name)
    early = early[early["InvoiceDate"] < SHEET_SPLIT].assign(sheet="2009-2010")
    late = late[late["InvoiceDate"] >= SHEET_SPLIT].assign(sheet="2010-2011")
    df = pd.concat([early, late], ignore_index=True)[[*COLUMNS, "sheet"]]
    return df.astype({"Quantity": "int64", "Price": "float64", "Customer ID": "Float64"})


def read_xlsx(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, dtype=DTYPES)
    names = sorted(sheets)
    if names != ["Year 2009-2010", "Year 2010-2011"]:
        raise ValueError(f"unexpected sheets: {names}")
    return combine_sheets(sheets[names[0]], sheets[names[1]])


def load(dest: Path | None = None) -> pd.DataFrame:
    """Transactions from both sheets, overlap removed. Cached: the xlsx takes minutes to read."""
    folder = (dest or data_dir()) / "raw"
    cache = folder / "transactions.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    zip_path = download(dest)
    xlsx = extract_xlsx(zip_path, folder)
    df = read_xlsx(xlsx)
    df.to_parquet(cache, index=False)
    xlsx.unlink()
    return df
