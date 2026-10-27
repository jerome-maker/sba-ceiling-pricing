"""論文 B 第二輪：ΔAUC 的拔靴法信賴區間 ＋ 殘差利差十分位駝峰的拆解。

兩件事：
  (1) ΔAUC（加入殘差利差的增量）目前只有點估計 +0.006，不知道顯不顯著。
      用**依借款人分群的成對拔靴法**：固定已配適的模型，對測試集重抽，
      每次重算 AUC_res - AUC_base。捕捉的是測試樣本的抽樣變異，
      不含模型訓練的變異——這是標準做法，但要在論文裡講清楚。
  (2) 殘差利差十分位 vs 違約率呈駝峰（中間十分位最高），不符單調預期。
      先查組成效果：十分位在產品別、核貸世代上的分布是否不均。

第一次執行會重算殘差並存檔，之後直接讀快取。
用法：python scripts/08_bootstrap_and_decile.py [--nboot 1000] [--refresh]
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from _common import load_7a, borrower_key, to_dates, SBA_DIR

warnings.filterwarnings("ignore")
DEVICE = "cuda"
NBOOT = int(sys.argv[sys.argv.index("--nboot") + 1]) if "--nboot" in sys.argv else 1000
CACHE = os.path.join(SBA_DIR, "resid_loans.pkl")

X1 = ["approval_ym", "lender_id", "term_months", "processing", "is_revolver",
      "is_fixed_rate", "log_amount", "guarantee_share", "has_collateral",
      "business_type", "business_age", "naics4", "state", "jobs", "is_franchise"]
PRODUCT_PROXIES = ["term_months", "processing", "is_revolver"]
X2_BASE = [c for c in X1 if c not in PRODUCT_PROXIES + ["approval_ym"]] + ["approval_fy"]

# ---------------------------------------------------------------- 資料與殘差
if os.path.exists(CACHE) and "--refresh" not in sys.argv:
    df = pd.read_pickle(CACHE)
    print(f"讀取快取 {CACHE}：{len(df):,} 筆")
else:
    COLS = ["BorrName", "BorrState", "BorrZip", "LocationID", "ApprovalDate", "ApprovalFY",
            "GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "InitialInterestRate",
            "FixedorVariableInterestInd", "RevolverStatus", "CollateralInd", "BusinessType",
            "BusinessAge", "NaicsCode", "ProjectState", "JobsSupported", "FranchiseCode",
            "ProcessingMethod", "LoanStatus", "ChargeOffDate", "PaidInFullDate", "AsOfDate"]
    raw = pd.concat([load_7a(years=f, usecols=COLS, drop_cancelled=True)[0]
                     for f in ["FY2000_FY2009", "FY2010_FY2019"]], ignore_index=True)
    raw = to_dates(raw, ["ApprovalDate", "ChargeOffDate", "PaidInFullDate", "AsOfDate"])
    g = pd.to_numeric(raw["GrossApproval"], errors="coerce")
    s = pd.to_numeric(raw["SBAGuaranteedApproval"], errors="coerce")
    df = pd.DataFrame({
        "borrower_key": borrower_key(raw),
        "approval_fy": pd.to_numeric(raw["ApprovalFY"], errors="coerce"),
        "approval_ym": raw["ApprovalDate"].dt.year * 12 + raw["ApprovalDate"].dt.month,
        "rate": pd.to_numeric(raw["InitialInterestRate"], errors="coerce"),
        "term_months": pd.to_numeric(raw["TermInMonths"], errors="coerce"),
        "log_amount": np.log1p(g), "guarantee_share": s / g,
        "jobs": pd.to_numeric(raw["JobsSupported"], errors="coerce"),
        "is_fixed_rate": (raw["FixedorVariableInterestInd"] == "F").astype("int8"),
        "is_revolver": (raw["RevolverStatus"] == "Y").astype("int8"),
        "has_collateral": (raw["CollateralInd"] == "Y").astype("int8"),
        "is_franchise": raw["FranchiseCode"].notna().astype("int8"),
        "business_type": raw["BusinessType"].astype("category"),
        "business_age": raw["BusinessAge"].astype("category"),
        "naics4": raw["NaicsCode"].str[:4].astype("category"),
        "state": raw["ProjectState"].astype("category"),
        "lender_id": raw["LocationID"].astype("category"),
        "processing": raw["ProcessingMethod"].astype("category"),
        "status": raw["LoanStatus"].values})
    end = np.where(raw["LoanStatus"] == "CHGOFF", raw["ChargeOffDate"],
          np.where(raw["LoanStatus"] == "P I F", raw["PaidInFullDate"], raw["AsOfDate"]))
    df["t_m"] = (pd.to_datetime(pd.Series(end, index=raw.index))
                 - raw["ApprovalDate"]).dt.days / 30.44
    df["obs_m"] = (raw["AsOfDate"] - raw["ApprovalDate"]).dt.days / 30.44
    df = df[df["rate"].notna() & (df["rate"] > 0) & df["t_m"].notna() & (df["t_m"] >= 0)].copy()

    PAR = {"objective": "reg:squarederror", "tree_method": "hist", "device": DEVICE,
           "max_depth": 8, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
           "min_child_weight": 50, "verbosity": 0}
    rng = np.random.default_rng(0)
    fold = rng.integers(0, 5, len(df))
    df["rate_hat"] = np.nan
    for k in range(5):
        tr_i, te_i = fold != k, fold == k
        a = xgb.QuantileDMatrix(df.loc[tr_i, X1], label=df.loc[tr_i, "rate"],
                                enable_categorical=True)
        b = xgb.QuantileDMatrix(df.loc[te_i, X1], enable_categorical=True, ref=a)
        df.loc[te_i, "rate_hat"] = xgb.train(PAR, a, num_boost_round=600).predict(b)
    df["resid"] = df["rate"] - df["rate_hat"]
    ss = 1 - ((df["rate"]-df["rate_hat"])**2).sum() / ((df["rate"]-df["rate"].mean())**2).sum()
    print(f"第一階段樣本外 R² = {ss:.4f}   殘差標準差 = {df['resid'].std():.3f} 個百分點")
    df.to_pickle(CACHE)
    print(f"已存快取 {CACHE}")

H = 5
ok = (df["status"] == "CHGOFF") | (df["status"] == "P I F") | (df["obs_m"] >= H * 12)
d = df[ok].copy()
d["y"] = ((d["status"] == "CHGOFF") & (d["t_m"] <= H * 12)).astype(int)

# ---------------------------------------------------------------- (2) 十分位拆解
print("\n" + "=" * 70)
print("(2) 殘差利差十分位的駝峰拆解")
print("=" * 70)
d["dec"] = pd.qcut(d["resid"], 10, labels=False, duplicates="drop")
d["term_bin"] = pd.cut(d["term_months"], [0, 24, 60, 84, 120, 240, 300, 420],
                       labels=["<=24m", "25-60m", "61-84m", "85-120m",
                               "121-240m", "241-300m", ">300m"])
d["era"] = pd.cut(d["approval_fy"], [0, 2004, 2009, 2014, 9999],
                  labels=["2000-04", "2005-09", "2010-14", "2015+"])

print("\n違約率 by 十分位（重現駝峰）")
t = d.groupby("dec", observed=True).agg(n=("y", "size"), 殘差=("resid", "mean"),
                                         違約率=("y", "mean"))
t["殘差"] = t["殘差"].round(3); t["違約率%"] = (t["違約率"] * 100).round(2)
print(t[["n", "殘差", "違約率%"]].to_string())

print("\n組成：各十分位的產品別佔比（%）")
print(pd.crosstab(d["dec"], d["term_bin"], normalize="index").mul(100).round(1).to_string())
print("\n組成：各十分位的核貸世代佔比（%）")
print(pd.crosstab(d["dec"], d["era"], normalize="index").mul(100).round(1).to_string())
print("\n組成：各十分位的循環信用比例（%）")
print(d.groupby("dec", observed=True)["is_revolver"].mean().mul(100).round(1).to_string())

print("\n**分層後**：各產品別內，違約率 by 十分位（%）— 駝峰是否消失？")
sub = d[d["term_bin"].isin(["25-60m", "61-84m", "85-120m", "241-300m"])]
piv = sub.pivot_table(index="dec", columns="term_bin", values="y",
                      aggfunc="mean", observed=True).mul(100).round(2)
print(piv.to_string())

print("\n**分世代後**：各世代內，違約率 by 十分位（%）")
piv2 = d.pivot_table(index="dec", columns="era", values="y",
                     aggfunc="mean", observed=True).mul(100).round(2)
print(piv2.to_string())

# ---------------------------------------------------------------- (1) 拔靴法
print("\n" + "=" * 70)
print(f"(1) ΔAUC 的成對拔靴法（依借款人分群，B={NBOOT}）")
print("=" * 70)
PAR2 = {"objective": "binary:logistic", "tree_method": "hist", "device": DEVICE,
        "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 50, "verbosity": 0}
tr = d[d["approval_fy"].between(2000, 2015)]
te = d[d["approval_fy"].between(2016, 2017)]
ov = set(tr["borrower_key"]) & set(te["borrower_key"])
te = te[~te["borrower_key"].isin(ov)].copy()
print(f"訓練 {len(tr):,} 筆   測試 {len(te):,} 筆   違約率 {te['y'].mean()*100:.2f}%")

preds = {}
for tag, cols in [("base", X2_BASE), ("res", X2_BASE + ["resid"])]:
    a = xgb.QuantileDMatrix(tr[cols], label=tr["y"], enable_categorical=True)
    b = xgb.QuantileDMatrix(te[cols], label=te["y"], enable_categorical=True, ref=a)
    preds[tag] = xgb.train(PAR2, a, num_boost_round=300).predict(b)
y = te["y"].values
a0, a1 = roc_auc_score(y, preds["base"]), roc_auc_score(y, preds["res"])
print(f"點估計：基準 {a0:.4f}   ＋殘差利差 {a1:.4f}   ΔAUC {a1-a0:+.4f}")

# 依借款人分群重抽
keys = te["borrower_key"].values
uniq, inv = np.unique(keys, return_inverse=True)
idx_by_key = [np.where(inv == i)[0] for i in range(len(uniq))]
rng = np.random.default_rng(42)
small = (te["log_amount"] < te["log_amount"].median()).values


def boot_stats(idx):
    yy = y[idx]
    if yy.sum() < 10 or yy.sum() == len(yy):
        return None
    d0, d1 = roc_auc_score(yy, preds["base"][idx]), roc_auc_score(yy, preds["res"][idx])
    out = [d1 - d0]
    for m in (small[idx], ~small[idx]):
        if m.sum() > 500 and 10 <= yy[m].sum() < m.sum():
            out.append(roc_auc_score(yy[m], preds["res"][idx][m])
                       - roc_auc_score(yy[m], preds["base"][idx][m]))
        else:
            out.append(np.nan)
    return out


t0 = time.perf_counter()
B = []
for _ in range(NBOOT):
    pick = rng.integers(0, len(uniq), len(uniq))
    idx = np.concatenate([idx_by_key[i] for i in pick])
    r = boot_stats(idx)
    if r is not None:
        B.append(r)
B = np.array(B)
print(f"拔靴完成 {len(B)} 次，耗時 {time.perf_counter()-t0:.1f} 秒\n")

names = ["ΔAUC（全體）", "ΔAUC（小額，金額<中位數）", "ΔAUC（大額）"]
rows = []
for j, nm in enumerate(names):
    v = B[:, j][~np.isnan(B[:, j])]
    rows.append((nm, round(v.mean(), 4), round(np.percentile(v, 2.5), 4),
                 round(np.percentile(v, 97.5), 4), round((v <= 0).mean(), 4)))
print(pd.DataFrame(rows, columns=["統計量", "拔靴平均", "95%下界", "95%上界",
                                  "P(Δ<=0)"]).to_string(index=False))

diff = B[:, 1] - B[:, 2]
diff = diff[~np.isnan(diff)]
print(f"\n小額 vs 大額的 ΔAUC 差距：{diff.mean():+.4f}  "
      f"95% CI [{np.percentile(diff,2.5):+.4f}, {np.percentile(diff,97.5):+.4f}]  "
      f"P(差距<=0) = {(diff <= 0).mean():.4f}")
print("\n註：此拔靴僅涵蓋測試樣本的抽樣變異，未含模型訓練的變異。")
