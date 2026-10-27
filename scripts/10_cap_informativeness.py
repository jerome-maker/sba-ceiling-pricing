"""論文 B 主軸：貼上限（at-cap）訂價的**資訊含量隨時間改變**。

假說：法定上限的約束力取決於貨幣環境。
  prime 低（2009-2015 釘在 3.25%）→ 上限 = prime+2.75+加成 的絕對水準低
  → 想加碼的放款機構都被壓到上限 → 貼上限是常態、不具選擇性 → 資訊含量低
  prime 回升 → 上限鬆開 → 只有真的想收高價的才會貼上限 → 資訊含量高

四組證據：
  A 貼上限比例 vs prime 水準（約束力的直接量測）
  B 貼上限的違約率落差，**控制可觀察變數後**，逐年估計
  C 落差 vs prime 的關係
  D **橫斷面檢定**：同一時點下，小額貸款的上限比大額高 2 個百分點
    （120.215 的加成），所以同一年內不同規模面對的約束強度不同。
    這給了時間序列之外的變異——只靠 11 年的 prime 路徑，變異來源太單薄。

已知限制（論文必寫）：
  * 時間序列只有 11 個世代、prime 路徑只有一條，總體變異的有效樣本很小
  * 各世代的 5 年結果落在不同總體環境（FY2015-19 世代的視窗含 COVID 與 PPP）。
    PPP 會**壓低**這些世代的違約，方向與本文發現相反，故不致造出此結果
  * 利率資料只有 FY2009 起（CLAUDE.md 4.7）

用法：python scripts/10_cap_informativeness.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from _common import SBA_DIR

warnings.filterwarnings("ignore")
DEVICE = "cuda"
H = 5
CTRL = ["log_amount", "guarantee_share", "term_months", "jobs", "is_fixed_rate",
        "has_collateral", "is_franchise", "business_type", "business_age",
        "naics4", "state", "lender_id", "processing"]

d = pd.read_pickle(os.path.join(SBA_DIR, "loans_with_cap.pkl"))
ex = d["processing"].astype(str).str.contains("Express|CAPLine|Community Advantage",
                                              case=False, na=False)
m = d[(d["is_fixed_rate"] == 0) & (~ex) & d["prime"].notna()].copy()
ok = (m["status"] == "CHGOFF") | (m["status"] == "P I F") | (m["obs_m"] >= H * 12)
m = m[ok].copy()
m["y"] = ((m["status"] == "CHGOFF") & (m["t_m"] <= H * 12)).astype(int)
amt = np.expm1(m["log_amount"])
m["tier"] = np.where(amt <= 25_000, "≤25k", np.where(amt <= 50_000, "25-50k", ">50k"))
m["cap_level"] = m["cap"]                      # prime + 2.75 + 加成
for c in CTRL:
    if str(m[c].dtype) in ("object", "category"):
        m[c] = m[c].astype("category")
print(f"樣本 {len(m):,} 筆（FY{int(m.approval_fy.min())}–{int(m.approval_fy.max())}）"
      f"   5 年違約率 {m['y'].mean()*100:.2f}%   貼上限 {m['at_cap'].mean()*100:.1f}%\n")

# ---------------------------------------------------------------- A 約束力
print("=== A. 貼上限比例 vs prime（約束力的直接量測）===")
a = m.groupby("approval_fy", observed=True).agg(
    n=("y", "size"), prime=("prime", "mean"), 貼上限=("at_cap", "mean"),
    平均上限=("cap_level", "mean"), 違約率=("y", "mean"))
a["prime"] = a["prime"].round(2); a["平均上限"] = a["平均上限"].round(2)
a["貼上限%"] = (a["貼上限"] * 100).round(1); a["違約率%"] = (a["違約率"] * 100).round(2)
print(a[["n", "prime", "平均上限", "貼上限%", "違約率%"]].to_string())
r = np.corrcoef(a["prime"], a["貼上限"])[0, 1]
print(f"\nprime 與貼上限比例的相關 = {r:+.3f}   "
      f"（假說預測為負：prime 越低、越多人被壓到上限）")

# ---------------------------------------------------------------- B 控制後的落差
print("\n=== B. 貼上限的違約率落差（控制可觀察變數後），逐年 ===")
print("作法：每年用**不含 at_cap** 的模型預測違約，再比較兩組的平均殘差 (y - ŷ)。")
PAR = {"objective": "binary:logistic", "tree_method": "hist", "device": DEVICE,
       "max_depth": 5, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
       "min_child_weight": 50, "verbosity": 0}
rows = []
for fy in sorted(m["approval_fy"].dropna().unique()):
    s = m[m["approval_fy"] == fy]
    if len(s) < 3000 or s["y"].sum() < 60:
        continue
    # 5 折交叉配適，避免樣本內過度配適把落差吃掉
    rng = np.random.default_rng(int(fy))
    fold = rng.integers(0, 5, len(s))
    yhat = np.empty(len(s))
    for k in range(5):
        tr, te = fold != k, fold == k
        A = xgb.QuantileDMatrix(s.iloc[np.flatnonzero(tr)][CTRL],
                                label=s["y"].values[tr], enable_categorical=True)
        B = xgb.QuantileDMatrix(s.iloc[np.flatnonzero(te)][CTRL],
                                enable_categorical=True, ref=A)
        yhat[te] = xgb.train(PAR, A, num_boost_round=200).predict(B)
    res = s["y"].values - yhat
    g1, g0 = res[s["at_cap"].values == 1], res[s["at_cap"].values == 0]
    gap = g1.mean() - g0.mean()
    se = np.sqrt(g1.var(ddof=1) / len(g1) + g0.var(ddof=1) / len(g0))
    raw = s.loc[s["at_cap"] == 1, "y"].mean() - s.loc[s["at_cap"] == 0, "y"].mean()
    rows.append((int(fy), len(s), round(s["prime"].mean(), 2),
                 round(s["at_cap"].mean() * 100, 1), round(raw * 100, 2),
                 round(gap * 100, 2), round(se * 100, 2), round(gap / se, 2)))
b = pd.DataFrame(rows, columns=["FY", "n", "prime", "貼上限%", "原始落差pp",
                                "控制後落差pp", "SE", "t"])
print(b.to_string(index=False))

# ---------------------------------------------------------------- C 落差 vs prime
print("\n=== C. 控制後落差 vs prime ===")
x, yv, w = b["prime"].values, b["控制後落差pp"].values, b["n"].values
r2 = np.corrcoef(x, yv)[0, 1]
beta = np.polyfit(x, yv, 1)
print(f"相關係數 = {r2:+.3f}")
print(f"以 prime 迴歸落差：斜率 = {beta[0]:+.3f} pp／每 1 個百分點的 prime")
print("假說預測斜率為正：prime 越高、上限越鬆，貼上限越具資訊含量")

# ---------------------------------------------------------------- D 橫斷面
print("\n=== D. 橫斷面檢定：同一年內，上限較高的規模組是否落差較大？ ===")
print("（120.215 的加成讓 ≤25k 的上限比 >50k 高 2 個百分點）")
out = []
for tier in ["≤25k", "25-50k", ">50k"]:
    s = m[m["tier"] == tier]
    if len(s) < 2000:
        continue
    out.append((tier, len(s), round(s["cap_level"].mean(), 2),
                round(s["at_cap"].mean() * 100, 1),
                round((s.loc[s["at_cap"] == 1, "y"].mean()
                       - s.loc[s["at_cap"] == 0, "y"].mean()) * 100, 2),
                round(s["y"].mean() * 100, 2)))
print(pd.DataFrame(out, columns=["規模組", "n", "平均上限", "貼上限%",
                                 "原始落差pp", "違約率%"]).to_string(index=False))

print("\n各規模組內的逐年貼上限比例（%）— 約束力是否同步鬆動")
piv = m.pivot_table(index="approval_fy", columns="tier", values="at_cap",
                    aggfunc="mean", observed=True).mul(100).round(1)
print(piv.to_string())

# ---------------------------------------------------------------- E 組成檢查
print("\n=== E. 組成檢查：貼上限組的可觀察特徵是否逐年改變 ===")
comp = m[m["at_cap"] == 1].groupby("approval_fy", observed=True).agg(
    金額中位=("log_amount", lambda s: np.expm1(s).median()),
    期限中位=("term_months", "median"), 有擔保=("has_collateral", "mean"),
    加盟=("is_franchise", "mean"), 就業中位=("jobs", "median"))
comp["有擔保%"] = (comp["有擔保"] * 100).round(1)
comp["加盟%"] = (comp["加盟"] * 100).round(1)
comp["金額中位"] = comp["金額中位"].round(0)
print(comp[["金額中位", "期限中位", "有擔保%", "加盟%", "就業中位"]].to_string())
print("\n若貼上限組的組成大致穩定，資訊含量的變化就不是組成效果造成的。")
