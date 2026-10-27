"""論文 B 投稿 Small Business Economics 前的擴充：識別與軟資訊理論的異質性預測。

A  放款機構固定效果：「貼上限的資訊」是否只是某些放款機構固定貼上限的機構政策？
   線性機率模型 y = β·at_cap + 控制 + 固定效果，依序加入
   (1) 只有控制變數、(2) 放款機構 FE、(3) 放款機構×核貸年 FE、(4) 再加 NAICS-4 與州 FE。
   以交替投影吸收多組固定效果；標準誤依放款機構群集。
B  軟資訊理論的異質性：
   B1 放款機構規模（核貸前一年底 FDIC 資產）：小銀行（< 10 億美元）較能運用軟資訊（Stein 2002；Berger et al. 2005）
   B2 借貸距離（借款人與放款機構郵遞區號中心點距離）與跨州放款：軟資訊隨距離衰減（Agarwal & Hauswald 2010）
   以 at_cap × 組別 的交互作用在放款機構×年 FE 下估計（組別主效果多被 FE 吸收）。
   另以殘差利差的 ΔAUC（FY2009–15 訓練、FY2016–17 測試、剔除重複借款人）分組比較，依借款人拔靴。
C  安慰劑：七年以上貸款在「prime + 2.25」（低於其上限 0.5 個百分點的整數點）是否也有違約落差。
   若資訊只出現在具約束力的上限，支持「被上限壓住」的解讀，而非任何整數訂價點。

外部資料：FDIC BankFind 財務資料（年底 ASSET）、Census 2020 ZCTA 中心點（下載至 ~/sba_data）。
用法：python scripts/28_paperB_extensions.py [--nboot 500]
"""
import sys, os, io, json, time, warnings, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from _common import SBA_DIR, load_7a

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
NBOOT = int(sys.argv[sys.argv.index("--nboot") + 1]) if "--nboot" in sys.argv else 500
H = 5
SKIP = "--skip-ab" in sys.argv
t0 = time.perf_counter()
OUT = {}

# ------------------------------------------------------------------ 主樣本（同 10_cap_informativeness.py）
d = pd.read_pickle(os.path.join(SBA_DIR, "loans_with_cap.pkl"))
ex = d["processing"].astype(str).str.contains("Express|CAPLine|Community Advantage", case=False, na=False)
m = d[(d["is_fixed_rate"] == 0) & (~ex) & d["prime"].notna()].copy()
ok = (m["status"] == "CHGOFF") | (m["status"] == "P I F") | (m["obs_m"] >= H * 12)
m = m[ok].copy()
m["y"] = ((m["status"] == "CHGOFF") & (m["t_m"] <= H * 12)).astype(int)
print(f"主樣本 {len(m):,}；五年違約率 {m.y.mean()*100:.2f}%；貼上限 {m.at_cap.mean()*100:.1f}%")

# ------------------------------------------------------------------ 接回原始欄位（索引與 07 建檔時一致）
EXTRA = ["LoanStatus", "ApprovalFY", "InitialInterestRate", "BankFDICNumber", "BankNCUANumber", "BankZip", "BankState",
         "BorrZip", "BorrState", "SoldSecMrktInd"]
raw = pd.concat([load_7a(years=f, usecols=EXTRA, drop_cancelled=True)[0]
                 for f in ["FY2000_FY2009", "FY2010_FY2019"]], ignore_index=True)
raw = raw.loc[m.index]
chk_fy = (pd.to_numeric(raw["ApprovalFY"], errors="coerce").values == m["approval_fy"].values).mean()
chk_rt = np.isclose(pd.to_numeric(raw["InitialInterestRate"], errors="coerce").values, m["rate"].values).mean()
print(f"對齊檢查：核貸年一致 {chk_fy*100:.2f}%；利率一致 {chk_rt*100:.2f}%")
assert chk_fy > 0.999 and chk_rt > 0.999
m["fdic"] = pd.to_numeric(raw["BankFDICNumber"], errors="coerce").values
m["ncua"] = raw["BankNCUANumber"].notna().values
m["bank_state"] = raw["BankState"].values
m["borr_state"] = raw["BorrState"].values
m["sold"] = (raw["SoldSecMrktInd"] == "Y").astype(int).values
z5 = lambda s: s.astype(str).str.extract(r"(\d{5})")[0]
m["bzip"], m["lzip"] = z5(raw["BorrZip"]).values, z5(raw["BankZip"]).values

# FDIC 年底資產（千美元），取核貸前一年底
fd = []
for yy in range(2008, 2019):
    url = (f"https://banks.data.fdic.gov/api/financials?filters=REPDTE:{yy}1231"
           f"&fields=CERT,ASSET&limit=10000&format=json")
    js = json.load(urllib.request.urlopen(url, timeout=120))
    fd.append(pd.DataFrame([r["data"] for r in js["data"]]).assign(year=yy + 1))
    time.sleep(0.5)
fd = pd.concat(fd)[["CERT", "ASSET", "year"]].rename(columns={"CERT": "fdic", "ASSET": "asset"})
m = m.merge(fd, left_on=["fdic", "approval_fy"], right_on=["fdic", "year"], how="left").set_axis(m.index)
assets_bn = m["asset"] / 1e6
m["lender_type"] = np.select(
    [m["fdic"].notna() & (assets_bn < 1), m["fdic"].notna() & (assets_bn >= 1) & (assets_bn < 50),
     m["fdic"].notna() & (assets_bn >= 50), m["ncua"]],
    ["small bank (<$1bn)", "mid bank ($1-50bn)", "large bank (>=$50bn)", "credit union"], default="non-bank / unmatched")
print(m["lender_type"].value_counts().to_string())

# 距離
zc = pd.read_csv(os.path.join(SBA_DIR, "2020_Gaz_zcta_national.txt"), sep="\t", dtype={"GEOID": str})
zc.columns = [c.strip() for c in zc.columns]
zc = zc.set_index("GEOID")[["INTPTLAT", "INTPTLONG"]]
la1, lo1 = zc.reindex(m["bzip"]).values.T
la2, lo2 = zc.reindex(m["lzip"]).values.T
r = np.pi / 180
a = np.sin((la2 - la1) * r / 2) ** 2 + np.cos(la1 * r) * np.cos(la2 * r) * np.sin((lo2 - lo1) * r / 2) ** 2
m["km"] = 2 * 6371 * np.arcsin(np.sqrt(a))
m["dist_grp"] = pd.cut(m["km"], [-1, 25, 100, 500, 1e5], labels=["<=25km", "25-100km", "100-500km", ">500km"]).astype(str)
m["out_state"] = (m["bank_state"] != m["borr_state"]).astype(int)
print(f"距離可計算 {m.km.notna().mean()*100:.1f}%；中位數 {m.km.median():.0f} km；跨州 {m.out_state.mean()*100:.1f}%")
amt = np.expm1(m["log_amount"])
m["small_loan"] = (amt <= amt.median()).astype(int)
m["term_b"] = pd.cut(m["term_months"], [0, 36, 60, 84, 120, 240, 1e4]).astype(str)

# ------------------------------------------------------------------ 多組固定效果的線性機率模型
NUM = ["log_amount", "guarantee_share", "jobs", "has_collateral", "is_franchise"]


def demean(M, groups, tol=1e-8, it=200):
    M = M.copy()
    codes = [pd.factorize(g)[0] for g in groups]
    for _ in range(it):
        mx = 0.0
        for c in codes:
            cnt = np.bincount(c)
            for j in range(M.shape[1]):
                mu = np.bincount(c, weights=M[:, j]) / cnt
                M[:, j] -= mu[c]
            mx = max(mx, np.abs(np.bincount(codes[0], weights=M[:, 0]) / np.bincount(codes[0])).max())
        if mx < tol:
            break
    return M


def lpm(df, xvars, fes, cluster):
    df = df.dropna(subset=xvars + NUM + ["y"]).copy()
    cat = ["business_type", "business_age", "term_b"]
    D = pd.get_dummies(df[cat].astype(str), drop_first=True).astype(float)
    Xall = np.column_stack([df[xvars].astype(float).values, df[NUM].astype(float).values, D.values])
    Y = df["y"].astype(float).values[:, None]
    groups = [df[f].astype(object).where(df[f].notna(), "NA").astype(str).values for f in fes] if fes else []
    if groups:
        Z = demean(np.column_stack([Y, Xall]), groups)
        Yd, Xd = Z[:, :1], Z[:, 1:]
    else:
        Yd, Xd = Y - Y.mean(), Xall - Xall.mean(0)
    keep = Xd.std(0) > 1e-10
    Xd = Xd[:, keep]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    b = XtX_inv @ Xd.T @ Yd
    u = (Yd - Xd @ b)[:, 0]
    cl = pd.factorize(df[cluster].astype(str))[0]
    G = cl.max() + 1
    S = np.zeros((Xd.shape[1], Xd.shape[1]))
    sc = np.zeros((G, Xd.shape[1]))
    np.add.at(sc, cl, Xd * u[:, None])
    S = sc.T @ sc
    n, k = Xd.shape
    V = XtX_inv @ S @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    nx = sum(keep[:len(xvars)])
    return {xvars[i]: (float(b[i, 0]), float(np.sqrt(V[i, i]))) for i in range(nx)}, len(df), G


print("\n######## A. 放款機構固定效果")
specs = [("controls only", []), ("+ lender FE", ["lender_id"]),
         ("+ lender×year FE", ["lender_year"]), ("+ lender×year, NAICS-4, state FE", ["lender_year", "naics4", "state"])]
m["lender_year"] = m["lender_id"].astype(str) + "_" + m["approval_fy"].astype(str)
rowsA = []
for lab, fes in ([] if SKIP else specs):
    est, n, G = lpm(m, ["at_cap"], fes, "lender_id")
    bb, se = est["at_cap"]
    rowsA.append(dict(spec=lab, beta_pp=round(bb * 100, 3), se_pp=round(se * 100, 3), t=round(bb / se, 2), n=n, clusters=G))
    print(f"  {lab:38s} β = {bb*100:+.3f} pp  (se {se*100:.3f}, t {bb/se:.2f})  n {n:,}  lenders {G:,}")
# 單一貸款放款機構無法在 FE 內提供變異：報告在 FE 內實際有 at_cap 變異的放款機構比例
var_l = m.groupby("lender_year")["at_cap"].agg(lambda s: s.nunique() > 1)
print(f"  有 at_cap 組內變異的「放款機構×年」比例：{var_l.mean()*100:.1f}%（涵蓋貸款 {m['lender_year'].map(var_l).mean()*100:.1f}%）")
OUT["A"] = rowsA

print("\n######## B. 異質性：at_cap × 組別（放款機構×年、NAICS-4、州 FE）")
rowsB = []
for var, ref in ([] if SKIP else [("lender_type", "large bank (>=$50bn)"), ("dist_grp", ">500km"), ("out_state", None), ("small_loan", None)]):
    s = m.copy()
    if ref is not None:
        s = s[s[var].astype(str) != "nan"]
        levels = [l for l in s[var].dropna().unique() if l != ref and l != "nan"]
        xv = ["at_cap"]
        for l in levels:
            nm = f"at_cap_x_{l}"
            s[nm] = s["at_cap"] * (s[var] == l).astype(int)
            s[f"main_{l}"] = (s[var] == l).astype(int)
            xv.append(nm)
        xv += [f"main_{l}" for l in levels]          # 主效果（多數會被放款機構×年 FE 吸收）
    else:
        s["at_cap_x"] = s["at_cap"] * s[var]
        s["main"] = s[var]
        xv = ["at_cap", "at_cap_x", "main"]
    est, n, G = lpm(s, xv, ["lender_year", "naics4", "state"], "lender_id")
    for k_, (bb, se) in est.items():
        rowsB.append(dict(split=var, term=k_, beta_pp=round(bb * 100, 3), se_pp=round(se * 100, 3), t=round(bb / se, 2)))
        print(f"  [{var}] {k_:40s} {bb*100:+.3f} pp (se {se*100:.3f}, t {bb/se:.2f})")
    gm = s.groupby(var if ref is not None else var).agg(n=("y", "size"), at_cap=("at_cap", "mean"), dr=("y", "mean"))
    print("   ", gm.round(4).to_dict("index"))
OUT["B_lpm"] = rowsB

# ------------------------------------------------------------------ B 的 ΔAUC 分組
print("\n######## B'. 殘差利差的增量預測力，分組（FY2009–15 訓練／FY2016–17 測試）")
XB = ["lender_id", "log_amount", "guarantee_share", "has_collateral", "business_type", "business_age",
      "naics4", "state", "jobs", "is_franchise", "approval_fy"]
for c in ["lender_id", "naics4", "state", "business_type", "business_age"]:
    m[c] = m[c].astype("category")
tr = m[m["approval_fy"] <= 2015]
te = m[m["approval_fy"].isin([2016, 2017]) & ~m["borrower_key"].isin(set(tr["borrower_key"]))]
PAR = {"objective": "binary:logistic", "tree_method": "hist", "device": "cuda", "max_depth": 6, "eta": 0.05,
       "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 50, "verbosity": 0}
preds = {}
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for nm, X in [("base", XB), ("res", XB + ["resid"])]:
        A = xgb.QuantileDMatrix(tr[X], label=tr["y"], enable_categorical=True)
        B = xgb.QuantileDMatrix(te[X], enable_categorical=True, ref=A)
        preds[nm] = xgb.train(PAR, A, num_boost_round=300).predict(B)
    fb = any("GPU" in str(x.message) and "CPU" in str(x.message) for x in w)
print(f"測試集 {len(te):,}；GPU 退回 CPU：{'有' if fb else '無'}；"
      f"基準 AUC {roc_auc_score(te.y, preds['base']):.4f}（論文表 3 為 0.7089）")
te = te.assign(pb=preds["base"], pr=preds["res"])
bk = te["borrower_key"].astype(str).values
ub, inv = np.unique(bk, return_inverse=True)
rng = np.random.default_rng(7)
rowsC = []
for var in (["dist_grp", "out_state", "small_loan"] if SKIP else ["lender_type", "dist_grp", "out_state", "small_loan"]):
    for lev in sorted(te[var].astype(object).where(te[var].notna(), "nan").astype(str).unique()):
        idx = np.flatnonzero(te[var].astype(object).where(te[var].notna(), "nan").astype(str).values == lev)
        if len(idx) < 3000 or te["y"].values[idx].sum() < 50:
            continue
        y, pb, pr = te["y"].values[idx], te["pb"].values[idx], te["pr"].values[idx]
        d0 = roc_auc_score(y, pr) - roc_auc_score(y, pb)
        inv_i = inv[idx]
        uu = np.unique(inv_i)
        bs = []
        for _ in range(NBOOT):
            pick = rng.choice(uu, len(uu), replace=True)
            cnt = np.bincount(pick, minlength=len(ub))[inv_i]
            sel = np.repeat(np.arange(len(idx)), cnt)
            if y[sel].sum() < 10:
                continue
            bs.append(roc_auc_score(y[sel], pr[sel]) - roc_auc_score(y[sel], pb[sel]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rowsC.append(dict(split=var, level=lev, n=len(idx), defaults=int(y.sum()), dAUC=round(d0, 4),
                          lo=round(lo, 4), hi=round(hi, 4), p_le0=round(float(np.mean(np.array(bs) <= 0)), 3)))
        print(f"  [{var}] {lev:24s} n {len(idx):6,}  ΔAUC {d0:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
OUT["B_auc"] = rowsC

# ------------------------------------------------------------------ C 安慰劑
print("\n######## C. 安慰劑：七年以上貸款在 prime + 2.25（非上限）")
L = m[m["term_months"] >= 84].copy()
L["spread"] = L["rate"] - L["prime"]
L["at_round"] = (np.abs(L["spread"] - 2.25) < 0.01).astype(int)
L = L[L["at_cap"] == 0]
estC, n, G = lpm(L, ["at_round"], ["lender_year", "naics4", "state"], "lender_id")
bb, se = estC["at_round"]
print(f"  非上限整數點 prime+2.25：占 {L.at_round.mean()*100:.1f}%；β = {bb*100:+.3f} pp (se {se*100:.3f}, t {bb/se:.2f})")
estC2, _, _ = lpm(m[m["term_months"] >= 84], ["at_cap"], ["lender_year", "naics4", "state"], "lender_id")
print(f"  同一子樣本的上限 at_cap：β = {estC2['at_cap'][0]*100:+.3f} pp (t {estC2['at_cap'][0]/estC2['at_cap'][1]:.2f})")
OUT["C"] = dict(share_round=float(L.at_round.mean()), beta_round=bb, se_round=se,
                beta_cap_same_subsample=estC2["at_cap"][0], se_cap_same_subsample=estC2["at_cap"][1])

path = os.path.join(SBA_DIR, "results_28_paperB_extensions" + ("_part2" if SKIP else "") + ".json")
json.dump(OUT, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(f"\n結果存於 {path}；總耗時 {time.perf_counter()-t0:.0f} 秒")
