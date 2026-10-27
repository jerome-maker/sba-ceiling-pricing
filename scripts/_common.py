"""共用設定：資料路徑解析與借款人鍵正規化。

資料檔案刻意不放 Google Drive。預設從 ~/sba_data 讀，可用環境變數 SBA_DIR 覆寫。
"""
import os
import glob
import pandas as pd

SBA_DIR = os.environ.get("SBA_DIR", os.path.join(os.path.expanduser("~"), "sba_data"))

# 剔除規則見 CLAUDE.md 4.3：CANCLD 是未撥款，不是貸款
NOT_A_LOAN = {"CANCLD"}

# 洩漏變數，一律不得進入特徵集（CLAUDE.md 4.1）
LEAKY_COLS = [
    "LoanStatus", "ChargeOffDate", "PaidInFullDate",
    "GrossChargeOffAmount", "AsOfDate",
]


def find_file(pattern):
    """以樣式在 SBA_DIR 找檔案，回傳最新的一個（檔名含 asof 日期）。"""
    hits = sorted(glob.glob(os.path.join(SBA_DIR, pattern)))
    if not hits:
        raise FileNotFoundError(
            f"在 {SBA_DIR} 找不到符合 {pattern} 的檔案。\n"
            f"先執行：bash scripts/00_fetch_sba.sh"
        )
    return hits[-1]


def load_7a(years="FY2010_FY2019", usecols=None, drop_cancelled=True):
    """讀入一個 7(a) 檔案。years 例如 'FY2010_FY2019'、'FY2000_FY2009'。"""
    path = find_file(f"FOIA_7a_{years}_asof_*.csv")
    df = pd.read_csv(path, dtype=str, low_memory=False, usecols=usecols)
    if drop_cancelled and "LoanStatus" in df.columns:
        df = df[~df["LoanStatus"].isin(NOT_A_LOAN)].copy()
    return df, path


def norm_name(s):
    """借款人名稱正規化：去標點、去公司型態字尾，供實體比對用。"""
    s = s.fillna("").str.upper().str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    s = s.str.replace(
        r"\b(INC|LLC|L L C|CORP|CORPORATION|CO|COMPANY|LTD|LP|LLP|PC|PA|DBA|THE)\b",
        " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def borrower_key(df):
    """企業鍵＝正規化名稱 + 州 + 郵遞區號前五碼。

    已知限制：鍵含郵遞區號，企業搬遷會斷鏈，會低估再借款率。
    正式分析要另跑「名稱+州」版本做敏感度（見 CLAUDE.md 4.4）。
    """
    return (norm_name(df["BorrName"]) + "|"
            + df["BorrState"].fillna("") + "|"
            + df["BorrZip"].fillna("").str[:5])


def to_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df
