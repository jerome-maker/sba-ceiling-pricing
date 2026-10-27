"""論文 B（SBE）補強：放款機構層級的異質性與上限定義的穩健性。

全部在「放款機構×核貸年、NAICS-4、州」固定效果下估計線性機率模型，標準誤依放款機構群集。

R1 貼上限容差：|利率 − 實務上限| < 0.01（主規格）、< 0.05、< 0.125
R2 法條上限：依 §120.214(d)(e) 的期限分層上限（prime + 2.25／2.75 ＋ 小額加成）
H1 放款機構的選擇性：以放款機構「其他貸款」的貼上限比例（留一法）衡量；
   很少貼上限的機構一旦貼上限，應更具資訊（選擇性高）。at_cap × 低貼上限比例（低於中位數）
H2 二級市場出售：保證部分出售者（SoldSecMrktInd = Y），放款機構保留的風險較少，
   篩選與訂價誘因可能不同。at_cap × sold
H3 委託授權（Preferred Lenders Program）：不經 SBA 逐案審核，放款機構的判斷權重較大。
   at_cap × PLP（相對 7a General 等非委託）
用法：python scripts/32_paperB_more_robustness.py
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import SBA_DIR, load_7a

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
H = 5
t0 = time.perf_counter()
d = pd.read_pickle(os.path.join(SBA_DIR, "loans_with_cap.pkl"))
ex = d["processing"].astype(str).str.contains("Express|CAPLine|Community Advantage", case=False, na=False)
m = d[(d["is_fixed_rate"] == 0) & (~ex) & d["prime"].notna()].copy()
ok = (m["status"] == "CHGOFF") | (m["status"] == "P I F") | (m["obs_m"] >= H * 12)
m = m[ok].copy()
m["y"] = ((m["status"] == "CHGOFF") & (m["t_m"] <= H * 12)).astype(int)
raw = pd.concat([load_7a(years=f, usecols=["LoanStatus", "ApprovalFY", "SoldSecMrktInd"], drop_cancelled=True)[0]
                 for f in ["FY2000_FY2009", "FY2010_FY2019"]], ignore_index=True).loc[m.index]
assert (pd.to_numeric(raw["ApprovalFY"]).values == m["approval_fy"].values).mean() > 0.999
m["sold"] = (raw["SoldSecMrktInd"] == "Y").astype(int).values
m["plp"] = (m["processing"].astype(str) == "Preferred Lenders Program").astype(int)
m["term_b"] = pd.cut(m["term_months"], [0, 36, 60, 84, 120, 240, 1e4]).astype(str)
m["lender_year"] = m["lender_id"].astype(str) + "_" + m["approval_fy"].astype(str)
# 上限定義
m["at_cap_05"] = (np.abs(m["rate"] - m["cap"]) < 0.05).astype(int)
m["at_cap_125"] = (np.abs(m["rate"] - m["cap"]) < 0.125).astype(int)
m["at_cap_stat"] = (np.abs(m["rate"] - m["cap_cfr"]) < 0.01).astype(int)
# 留一法的放款機構貼上限比例
g = m.groupby("lender_id", observed=True)["at_cap"]
cnt, sm = g.transform("size"), g.transform("sum")
m["loo_share"] = np.where(cnt > 1, (sm - m["at_cap"]) / (cnt - 1).clip(lower=1), np.nan)
med = float(np.nanmedian(m["loo_share"]))
m["selective"] = (m["loo_share"] < med).astype(int)
print(f"樣本 {len(m):,}；出售 {m.sold.mean()*100:.1f}%；PLP {m.plp.mean()*100:.1f}%；留一貼上限比例中位數 {med:.3f}")

NUM = ["log_amount", "guarantee_share", "jobs", "has_collateral", "is_franchise"]
FES = ["lender_year", "naics4", "state"]


def demean(M, groups, tol=1e-8, it=200):
    M = M.copy()
    codes = [pd.factorize(g_)[0] for g_ in groups]
    for _ in range(it):
        for c in codes:
            cntc = np.bincount(c)
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, weights=M[:, j]) / cntc)[c]
        if np.abs(np.bincount(codes[0], weights=M[:, 0]) / np.bincount(codes[0])).max() < tol:
            break
    return M


def lpm(df, xvars):
    df = df.dropna(subset=xvars + NUM + ["y"]).copy()
    D = pd.get_dummies(df[["business_type", "business_age", "term_b"]].astype(str), drop_first=True).astype(float)
    Xall = np.column_stack([df[xvars].astype(float).values, df[NUM].astype(float).values, D.values])
    groups = [df[f].astype(object).where(df[f].notna(), "NA").astype(str).values for f in FES]
    Z = demean(np.column_stack([df["y"].astype(float).values, Xall]), groups)
    Yd, Xd = Z[:, :1], Z[:, 1:]
    keep = Xd.std(0) > 1e-10
    Xd = Xd[:, keep]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    b = XtX_inv @ Xd.T @ Yd
    u = (Yd - Xd @ b)[:, 0]
    cl = pd.factorize(df["lender_id"].astype(str))[0]
    G = cl.max() + 1
    sc = np.zeros((G, Xd.shape[1]))
    np.add.at(sc, cl, Xd * u[:, None])
    n, k = Xd.shape
    V = XtX_inv @ (sc.T @ sc) @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    names = [x for x, kk in zip(xvars, keep[:len(xvars)]) if kk]
    return {nm: (float(b[i, 0] * 100), float(np.sqrt(V[i, i]) * 100)) for i, nm in enumerate(names)}, len(df)


OUT = {}
def show(tag, est, n):
    OUT[tag] = dict(n=n, coef={k: dict(b_pp=v[0], se_pp=v[1], t=v[0] / v[1]) for k, v in est.items()})
    for k, (bb, se) in est.items():
        print(f"  [{tag}] {k:26s} {bb:+.3f} pp (se {se:.3f}, t {bb/se:.2f})")


print("\n######## R1/R2 上限定義")
for v in ["at_cap", "at_cap_05", "at_cap_125", "at_cap_stat"]:
    print(f"   {v}: 比例 {m[v].mean()*100:.1f}%")
    show(v, *lpm(m, [v]))

print("\n######## H1–H3 交互作用")
for tag, mv in [("selective", "selective"), ("sold", "sold"), ("plp", "plp")]:
    s = m.copy()
    s["at_cap_x"] = s["at_cap"] * s[mv]
    show(tag, *lpm(s, ["at_cap", "at_cap_x", mv]))
    gm = s.groupby(mv).agg(n=("y", "size"), at_cap=("at_cap", "mean"), dr=("y", "mean"))
    print("    ", gm.round(4).to_dict("index"))
    OUT[tag]["groups"] = gm.reset_index().to_dict("records")
OUT["loo_median"] = med
json.dump(OUT, open(os.path.join(SBA_DIR, "results_32_paperB_more.json"), "w", encoding="utf-8"), indent=1, default=float)
print(f"\n總耗時 {time.perf_counter()-t0:.0f} 秒")
