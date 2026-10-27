"""[36] 沿用 35 的樣本與估計函數。論文 B 模擬審查第一輪（2026-10-27，docs/review_paperB_JRE_round1.md）要求的新分析。

R2  同一利率、不同上限：小額加成（≤2.5 萬 +2.00；2.5–5 萬 +1.00；>5 萬 0）讓同一利差在不同金額區間
    分屬「在上限」與「不在上限」。主迴歸加入「利差格點（0.25）× 固定效果」與「金額區間固定效果」，
    at_cap 只剩「利差 × 金額區間」交互格的變異，與利率水準分開識別。
    另做 5 萬美元附近的窄區間（3.5–6.5 萬）差異中的差異：利差 = 2.75 時，>5 萬（在上限）與
    ≤5 萬（低於上限 1.00）的違約差，減去利差 = 2.25 時同一金額區間的差（兩側皆不在上限）。
R3  疫情紓困：五年視窗在 2020-03 前結束的世代（核貸月 + 60 ≤ 2020-03）重估；三年視野（全世代與
    視窗在 2020-03 前結束者）。
R6  保證損失佔比：五年內沖銷金額（GrossChargeOffAmount）中，貼上限貸款的佔比；
    乘上保證比例近似 SBA 的保證損失。
R7  超過實務上限的貸款組成（dist_to_cap < −0.01）。
R8  固定利率在該期間的比例；三年視野下的主估計。
全部在「放款機構×核貸年、NAICS-4、州」固定效果下估計線性機率模型，標準誤依放款機構群集。
用 CPU：工作是 19 萬列的交替投影與小型最小平方，搬到 GPU 沒有好處（全域規則第七節第 1 項）。
用法：python scripts/35_paperB_review_round1.py
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import SBA_DIR, load_7a

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.perf_counter()
COVID_YM = 2020 * 12 + 3          # 2020-03
d = pd.read_pickle(os.path.join(SBA_DIR, "loans_with_cap.pkl"))
ex = d["processing"].astype(str).str.contains("Express|CAPLine|Community Advantage", case=False, na=False)
base = d[(~ex) & d["prime"].notna() & d["approval_fy"].between(2009, 2019)].copy()
OUT = {"fixed_rate_share_FY2009_2019": float(base["is_fixed_rate"].mean())}
print(f"FY2009–2019 標準 7(a) 有 prime 的貸款 {len(base):,}；固定利率比例 {base.is_fixed_rate.mean()*100:.1f}%")
v = base[base["is_fixed_rate"] == 0].copy()


def sample(h):
    ok = (v["status"] == "CHGOFF") | (v["status"] == "P I F") | (v["obs_m"] >= h * 12)
    s = v[ok].copy()
    s["y"] = ((s["status"] == "CHGOFF") & (s["t_m"] <= h * 12)).astype(int)
    s["pre_covid"] = (s["approval_ym"] + h * 12 <= COVID_YM).astype(int)
    return s


m = sample(5)
print(f"五年主樣本 {len(m):,}（應為 193,733）；違約率 {m.y.mean()*100:.2f}%")
assert len(m) == 193_733
amt = np.expm1(m["log_amount"])
m["band"] = np.select([amt <= 25_000, amt <= 50_000], ["le25k", "25-50k"], "gt50k")
m["spread"] = m["rate"] - m["prime"]
m["spread_bin"] = (np.round(m["spread"] * 4) / 4).clip(-1, 6).astype(str)
m["term_b"] = pd.cut(m["term_months"], [0, 36, 60, 84, 120, 240, 1e4]).astype(str)
m["lender_year"] = m["lender_id"].astype(str) + "_" + m["approval_fy"].astype(str)
m["amt"] = amt

NUM = ["log_amount", "guarantee_share", "jobs", "has_collateral", "is_franchise"]
FES = ["lender_year", "naics4", "state"]


def demean(M, groups, tol=1e-8, it=300):
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


def lpm(df, xvars, extra_fes=()):
    df = df.dropna(subset=xvars + NUM + ["y"]).copy()
    D = pd.get_dummies(df[["business_type", "business_age", "term_b"]].astype(str), drop_first=True).astype(float)
    Xall = np.column_stack([df[xvars].astype(float).values, df[NUM].astype(float).values, D.values])
    groups = [df[f].astype(object).where(df[f].notna(), "NA").astype(str).values for f in list(FES) + list(extra_fes)]
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
    return {nm: (float(b[i, 0] * 100), float(np.sqrt(V[i, i]) * 100)) for i, nm in enumerate(names)}, len(df), float(df["y"].mean())


def show(tag, res):
    est, n, ybar = res
    OUT[tag] = dict(n=n, mean_y_pct=ybar * 100, coef={k: dict(b_pp=b, se_pp=s, t=b / s) for k, (b, s) in est.items()})
    for k, (b, s) in est.items():
        print(f"  [{tag}] n={n:,} 違約率 {ybar*100:.2f}%  {k:14s} {b:+.3f} pp (se {s:.3f}, t {b/s:.2f})")


# ---------------------------------------------------------------- 36：上限以下與以上的利差梯度
# 只用 >5 萬美元的貸款（上限一律 prime + 2.75），在放款機構×年、NAICS-4、州 FE 下估計各利差格的違約差，
# 參考組為利差 < 2.00。上限以上（> 2.76）是違反法規的訂價，用來看梯度在上限處是否只是被截斷。
g = m[m.band == "gt50k"].copy()
sp = g["spread"]
g["sb"] = np.select([sp < 2.0, sp < 2.24, sp < 2.26, sp < 2.74, sp < 2.76], ["lt2.00", "2.00-2.24", "2.25", "2.26-2.74", "2.75cap"], "gt2.75")
names = ["2.00-2.24", "2.25", "2.26-2.74", "2.75cap", "gt2.75"]
for nm in names:
    g["d_" + nm] = (g["sb"] == nm).astype(int)
desc = g.groupby("sb").agg(n=("y", "size"), dr=("y", "mean")).reindex(["lt2.00"] + names)
print(desc.to_string())
OUT["G_desc"] = desc.reset_index().to_dict("records")
show("G_gradient", lpm(g, ["d_" + nm for nm in names]))
g5 = g[g.pre_covid == 1].copy()
show("G_gradient_pre_covid", lpm(g5, ["d_" + nm for nm in names]))
json.dump(OUT, open(os.path.join(SBA_DIR, "results_36_paperB_gradient.json"), "w", encoding="utf-8"), indent=1, default=float)
print(f"總耗時 {time.perf_counter()-t0:.0f} 秒（CPU）")
