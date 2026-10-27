"""論文 B 的核心檢定：放款利率的「殘差利差」是否含有可觀察變數之外的資訊。

設定：中小企業沒有公開財務報表，硬資訊不存在，放款機構只能依賴軟資訊，
而**訂價是軟資訊唯一的可觀察痕跡**。

兩階段：
  第一階段  以所有核貸當下可觀察的變數（放款機構、核貸年月、產品、金額、擔保、
            產業、地區、企業特徵）預測利率，取殘差＝殘差利差。
            用 XGBoost 而非 OLS：高基數類別可原生處理，且更有彈性＝吸收更多，
            對「殘差仍有資訊」這個主張是**更保守**的檢定。
  第二階段  檢定殘差利差對違約的**增量**預測力（ΔAUC / ΔPR-AUC），
            並做不透明度的異質性檢定——理論預測軟資訊在不透明借款人身上最強。

**尚未處理的識別問題**：高利率可能是「揭露風險」也可能是「造成風險」
（還款負擔提高）。本腳本只做關聯，不宣稱因果。緩解方向見 docs/handover.md。

用法：python scripts/07_residual_spread.py
"""
import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from _common import load_7a, borrower_key, to_dates

warnings.filterwarnings("ignore")
DEVICE = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
FILES = ["FY2000_FY2009", "FY2010_FY2019"]
HORIZONS = (3, 5)

COLS = ["BorrName", "BorrState", "BorrZip", "LocationID", "ApprovalDate", "ApprovalFY",
        "GrossApproval", "SBAGuaranteedApproval", "TermInMonths", "InitialInterestRate",
        "FixedorVariableInterestInd", "RevolverStatus", "CollateralInd", "BusinessType",
        "BusinessAge", "NaicsCode", "ProjectState", "JobsSupported", "FranchiseCode",
        "ProcessingMethod", "LoanStatus", "ChargeOffDate", "PaidInFullDate", "AsOfDate"]

# 第一階段：所有核貸當下可觀察的變數。含產品代理變數是刻意的——
# 要把機械性訂價（產品別、規模、期間、放款機構慣例）全部吸掉。
X1 = ["approval_ym", "lender_id", "term_months", "processing", "is_revolver",
      "is_fixed_rate", "log_amount", "guarantee_share", "has_collateral",
      "business_type", "business_age", "naics4", "state", "jobs", "is_franchise"]
# 第二階段的基準特徵集（不含利率本身）。
# 產品代理變數會把基準 AUC 推到 0.97（見 CLAUDE.md 4.1.1），增量檢定在那裡沒有功效，
# 所以主規格用排除產品代理變數的誠實基準（AUC ~0.73），另跑飽和基準作對照。
PRODUCT_PROXIES = ["term_months", "processing", "is_revolver"]
X2_FULL = [c for c in X1 if c != "approval_ym"] + ["approval_fy"]
X2_BASE = [c for c in X2_FULL if c not in PRODUCT_PROXIES]

t0 = time.perf_counter()
parts = []
for f in FILES:
    d, path = load_7a(years=f, usecols=COLS, drop_cancelled=True)
    print(f"讀入 {os.path.basename(path)}：{len(d):,} 筆")
    parts.append(d)
raw = pd.concat(parts, ignore_index=True)
raw = to_dates(raw, ["ApprovalDate", "ChargeOffDate", "PaidInFullDate", "AsOfDate"])

g = pd.to_numeric(raw["GrossApproval"], errors="coerce")
s = pd.to_numeric(raw["SBAGuaranteedApproval"], errors="coerce")
df = pd.DataFrame({
    "borrower_key": borrower_key(raw),
    "approval_fy": pd.to_numeric(raw["ApprovalFY"], errors="coerce"),
    "approval_ym": (raw["ApprovalDate"].dt.year * 12 + raw["ApprovalDate"].dt.month),
    "rate": pd.to_numeric(raw["InitialInterestRate"], errors="coerce"),
    "term_months": pd.to_numeric(raw["TermInMonths"], errors="coerce"),
    "log_amount": np.log1p(g),
    "guarantee_share": s / g,
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
    "status": raw["LoanStatus"].values,
})
end = np.where(raw["LoanStatus"] == "CHGOFF", raw["ChargeOffDate"],
      np.where(raw["LoanStatus"] == "P I F", raw["PaidInFullDate"], raw["AsOfDate"]))
df["t_m"] = (pd.to_datetime(pd.Series(end, index=raw.index)) - raw["ApprovalDate"]).dt.days / 30.44
df["obs_m"] = (raw["AsOfDate"] - raw["ApprovalDate"]).dt.days / 30.44
df = df[df["rate"].notna() & (df["rate"] > 0) & df["t_m"].notna() & (df["t_m"] >= 0)].copy()
print(f"合計可用 {len(df):,} 筆   準備 {time.perf_counter()-t0:.1f} 秒\n")

print("=== 利率分布 ===")
print(df["rate"].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]).round(2).to_string())

# ---------------------------------------------------------------- 第一階段
PAR = {"objective": "reg:squarederror", "tree_method": "hist", "device": DEVICE,
       "max_depth": 8, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
       "min_child_weight": 50, "verbosity": 0}
# 交叉配適（cross-fitting）：用樣本內預測會讓第一階段把真正的私有訊號也配適掉，
# 殘差被壓縮，第二階段的檢定會偏向虛無。改用 5 折的樣本外預測取殘差。
t1 = time.perf_counter()
K = 5
rng = np.random.default_rng(0)
fold = rng.integers(0, K, len(df))
df["rate_hat"] = np.nan
for k in range(K):
    tr_i, te_i = fold != k, fold == k
    a = xgb.QuantileDMatrix(df.loc[tr_i, X1], label=df.loc[tr_i, "rate"],
                            enable_categorical=True)
    b = xgb.QuantileDMatrix(df.loc[te_i, X1], enable_categorical=True, ref=a)
    m1 = xgb.train(PAR, a, num_boost_round=600)
    df.loc[te_i, "rate_hat"] = m1.predict(b)
df["resid"] = df["rate"] - df["rate_hat"]
ss_res = float(((df["rate"] - df["rate_hat"]) ** 2).sum())
ss_tot = float(((df["rate"] - df["rate"].mean()) ** 2).sum())
print(f"\n=== 第一階段：利率 ~ 可觀察變數，{K} 折交叉配適（{time.perf_counter()-t1:.1f} 秒）===")
print(f"樣本外 R² = {1 - ss_res/ss_tot:.4f}   殘差標準差 = {df['resid'].std():.3f} 個百分點")
print("\n殘差利差分布（百分點）")
print(df["resid"].describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
imp = pd.Series(m1.get_score(importance_type="gain")).sort_values(ascending=False)
print("\n第一階段特徵重要性（gain）")
print(imp.head(8).round(1).to_string())

# ---------------------------------------------------------------- 描述性
print("\n=== 殘差利差十分位 vs 實際違約率 ===")
ok5 = (df["status"] == "CHGOFF") | (df["status"] == "P I F") | (df["obs_m"] >= 60)
d5 = df[ok5].copy()
d5["y5"] = ((d5["status"] == "CHGOFF") & (d5["t_m"] <= 60)).astype(int)
d5["dec"] = pd.qcut(d5["resid"], 10, labels=False, duplicates="drop")
t = d5.groupby("dec", observed=True).agg(n=("y5", "size"), 殘差利差=("resid", "mean"),
                                          實際利率=("rate", "mean"), 違約率=("y5", "mean"))
t["殘差利差"] = t["殘差利差"].round(3)
t["實際利率"] = t["實際利率"].round(2)
t["5年違約率%"] = (t["違約率"] * 100).round(2)
print(t[["n", "殘差利差", "實際利率", "5年違約率%"]].to_string())
lo, hi = t["5年違約率%"].iloc[0], t["5年違約率%"].iloc[-1]
print(f"最高十分位 / 最低十分位 = {hi/lo:.1f} 倍")

# ---------------------------------------------------------------- 第二階段
PAR2 = {"objective": "binary:logistic", "tree_method": "hist", "device": DEVICE,
        "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 50, "verbosity": 0}


def stage2(H, cols, tag):
    ok = (df["status"] == "CHGOFF") | (df["status"] == "P I F") | (df["obs_m"] >= H * 12)
    d = df[ok].copy()
    d["y"] = ((d["status"] == "CHGOFF") & (d["t_m"] <= H * 12)).astype(int)
    tr = d[d["approval_fy"].between(2000, 2015)]
    te = d[d["approval_fy"].between(2016, 2017)]
    ov = set(tr["borrower_key"]) & set(te["borrower_key"])
    te = te[~te["borrower_key"].isin(ov)]
    a = xgb.QuantileDMatrix(tr[cols], label=tr["y"], enable_categorical=True)
    b = xgb.QuantileDMatrix(te[cols], label=te["y"], enable_categorical=True, ref=a)
    m = xgb.train(PAR2, a, num_boost_round=300)
    p = m.predict(b)
    y = te["y"].values
    return dict(tag=tag, H=H, n=len(te), rate=y.mean() * 100,
                auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                te=te, p=p, y=y)


print("\n=== 第二階段：殘差利差的增量預測力（訓練 FY2000-2015 → 測試 FY2016-2017）===")
rows = []
for H in HORIZONS:
    for cols, spec in [(X2_BASE, "主規格(排除產品代理)"), (X2_FULL, "飽和基準(含產品代理)")]:
        base = stage2(H, cols, f"{spec} 基準")
        res = stage2(H, cols + ["resid"], f"{spec} ＋殘差利差")
        raw_r = stage2(H, cols + ["rate"], f"{spec} ＋原始利率")
        for r in (base, res, raw_r):
            rows.append({"視野": f"{H}年", "規格": r["tag"], "測試筆數": r["n"],
                         "實際%": round(r["rate"], 2), "AUC": round(r["auc"], 4),
                         "PR-AUC": round(r["ap"], 4)})
        rows[-2]["ΔAUC"] = round(res["auc"] - base["auc"], 4)
        rows[-2]["ΔPR-AUC"] = round(res["ap"] - base["ap"], 4)
        rows[-1]["ΔAUC"] = round(raw_r["auc"] - base["auc"], 4)
        rows[-1]["ΔPR-AUC"] = round(raw_r["ap"] - base["ap"], 4)
        if H == 5 and cols is X2_BASE:
            keep5 = (base, res)
print(pd.DataFrame(rows).fillna("").to_string(index=False))

# ---------------------------------------------------------------- 異質性
print("\n=== 異質性：殘差利差的增量在不透明借款人身上是否更大？（5 年視野）===")
base, res = keep5
te = res["te"].copy()
te["p_base"], te["p_res"], te["y"] = base["p"], res["p"], res["y"]
opaque = {
    "新創/未滿1年": te["business_age"].astype(str).str.contains("Startup|New,", na=False),
    "無擔保": te["has_collateral"] == 0,
    "非加盟": te["is_franchise"] == 0,
    "金額低於中位數": te["log_amount"] < te["log_amount"].median(),
}
out = []
for name, mask in opaque.items():
    for lab, m in [(name, mask.values), (f"非{name}", (~mask).values)]:
        if m.sum() > 3000 and te["y"].values[m].sum() >= 30:
            out.append((lab, int(m.sum()), int(te["y"].values[m].sum()),
                        round(roc_auc_score(te["y"].values[m], te["p_base"].values[m]), 4),
                        round(roc_auc_score(te["y"].values[m], te["p_res"].values[m]), 4)))
h = pd.DataFrame(out, columns=["子群", "筆數", "違約數", "基準AUC", "＋殘差AUC"])
h["ΔAUC"] = (h["＋殘差AUC"] - h["基準AUC"]).round(4)
print(h.to_string(index=False))
print("\n理論預測：不透明子群的 ΔAUC 應大於對應的透明子群。")
