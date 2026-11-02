import zipfile

import pandas as pd
import pytest

from demand import data
from demand.data import COLUMNS, check_zip, combine_sheets, extract_xlsx


def sheet(dates):
    n = len(dates)
    return pd.DataFrame(
        {
            "Invoice": [str(500000 + i) for i in range(n)],
            "StockCode": "85123A",
            "Description": "MUG",
            "Quantity": 1,
            "InvoiceDate": pd.to_datetime(dates, format="mixed"),
            "Price": 1.0,
            "Customer ID": 1.0,
            "Country": "UK",
        }
    )


def test_overlapping_days_are_taken_from_one_sheet_only():
    early = sheet(["2010-11-30", "2010-12-01 09:00", "2010-12-09"])
    late = sheet(["2010-12-01 09:00", "2010-12-09", "2011-01-05"])
    df = combine_sheets(early, late)
    assert len(df) == 4
    assert df.loc[df["InvoiceDate"] < "2010-12-01", "sheet"].unique().tolist() == ["2009-2010"]
    assert df.loc[df["InvoiceDate"] >= "2010-12-01", "sheet"].unique().tolist() == ["2010-2011"]
    assert list(df.columns) == [*COLUMNS, "sheet"]


def test_a_sheet_missing_a_column_is_refused():
    with pytest.raises(ValueError, match="late sheet: missing columns"):
        combine_sheets(sheet(["2010-01-01"]), sheet(["2011-01-01"]).drop(columns="Price"))


def test_a_changed_zip_is_refused_before_opening(tmp_path):
    path = tmp_path / "online_retail_ii.zip"
    path.write_bytes(b"not the real file")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        check_zip(path)


def test_an_oversized_zip_is_refused_before_hashing(tmp_path, monkeypatch):
    path = tmp_path / "big.zip"
    path.write_bytes(b"x" * 100)
    monkeypatch.setattr(data, "MAX_ZIP_BYTES", 10)
    monkeypatch.setattr(data, "sha256_of", lambda p: pytest.fail("hashed an oversized file"))
    with pytest.raises(ValueError, match="more than"):
        check_zip(path)


def test_download_checks_a_cached_file_too(tmp_path, monkeypatch):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "online_retail_ii.zip").write_bytes(b"tampered")
    monkeypatch.setattr(data.urllib.request, "urlretrieve", lambda *a: pytest.fail("no network"))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.download(tmp_path)


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)


def test_only_the_expected_member_is_extracted(tmp_path):
    zpath = tmp_path / "a.zip"
    make_zip(zpath, {data.MEMBER: b"xlsx bytes", "../../evil.py": b"print(1)"})
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out = extract_xlsx(zpath, out_dir)
    assert out.read_bytes() == b"xlsx bytes"
    assert [p.name for p in out_dir.iterdir()] == [data.MEMBER]
    assert not (tmp_path / "evil.py").exists()


def test_a_zip_without_the_member_is_refused(tmp_path):
    zpath = tmp_path / "a.zip"
    make_zip(zpath, {"other.xlsx": b"x"})
    with pytest.raises(ValueError, match="has no"):
        extract_xlsx(zpath, tmp_path)


def test_a_member_that_would_unpack_too_big_is_refused(tmp_path, monkeypatch):
    zpath = tmp_path / "a.zip"
    make_zip(zpath, {data.MEMBER: b"0" * 5000})
    monkeypatch.setattr(data, "MAX_XLSX_BYTES", 1000)
    with pytest.raises(ValueError, match="would unpack"):
        extract_xlsx(zpath, tmp_path)
    assert not (tmp_path / data.MEMBER).exists()


def test_data_dir_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("DEMAND_DATA_DIR", str(tmp_path))
    assert data.data_dir() == tmp_path
