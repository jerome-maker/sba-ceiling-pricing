"""用 SBA 法定利率上限把「公式化訂價」正式化。

法源（primary source，非二手整理）——13 CFR，1996 年訂、**2022 年 8 月才移除**，
因此涵蓋本專案 FY2000–2019 的全部資料期間：

  § 120.214 變動利率
    (c) 基準利率為 prime、1 個月 LIBOR + 3、或 SBA Optional Peg Rate；
        採「當月第一個營業日」的值
    (d) 期限未滿 7 年：上限 = 基準 + 2.25 個百分點
    (e) 期限 7 年以上：上限 = 基準 + 2.75 個百分點
    另明定「SBA 的最高利率只適用於 SBA 收到貸款申請當日的**初始利率**」
    —— 正好對應資料中的 InitialInterestRate

  § 120.215 小額貸款加成
    逾 $25,000 未逾 $50,000：上限再加 1.00 個百分點
    $25,000 以下：上限再加 2.00 個百分點

合成後的上限表（本資料期間適用）：

  | 金額            | 期限 <7 年   | 期限 ≥7 年   |
  |-----------------|-------------|-------------|
  | > $50,000       | prime+2.25  | prime+2.75  |
  | $25,001–$50,000 | prime+3.25  | prime+3.75  |
  | ≤ $25,000       | prime+4.25  | prime+4.75  |

**樣本限制（刻意保守）**：
  * 只留**變動利率**貸款——120.214 明文規範的就是變動利率；固定利率的上限
    另有公式（2018-11 起的 Fixed Base Rate 制），尚未查證，不納入主分析。
  * 排除 **SBA Express 等特別方案**——其上限由 SOP 另訂（較高），未取得 primary
    source，不硬套 120.214。

**已知量測誤差，必須寫進論文**：
  1. 法規用「申請日」，資料只有核貸日（ApprovalDate），兩者差數週
  2. 基準利率可選 prime / LIBOR+3 / Optional Peg，這裡一律用 prime
  3. prime 取核貸當月第一個營業日（FRED DPRIME 日資料）

用法：python scripts/09_rate_cap.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from _common import SBA_DIR

warnings.filterwarnings("ignore")
CACHE = os.path.join(SBA_DIR, "resid_loans.pkl")
PRIME = os.path.join(SBA_DIR, "dprime.csv")

df = pd.read_pickle(CACHE)
print(f"讀取 {len(df):,} 筆（含交叉配適殘差）")

# ---------------------------------------------------------------- prime：當月第一個營業日
pr = pd.read_csv(PRIME).rename(columns={"observation_date": "d", "DPRIME": "prime"})
pr["d"] = pd.to_datetime(pr["d"])
pr["prime"] = pd.to_numeric(pr["prime"], errors="coerce")
pr = pr.dropna().sort_values("d")
pr["ym"] = pr["d"].dt.year * 12 + pr["d"].dt.month
first_bd = pr.groupby("ym").first()["prime"]
df["prime"] = df["approval_ym"].map(first_bd)
print(f"prime 對上 {df['prime'].notna().mean()*100:.1f}%   "
      f"區間 {df['prime'].min():.2f}–{df['prime'].max():.2f}%")

# ---------------------------------------------------------------- 法定上限
amount = np.expm1(df["log_amount"])
# ★ 2026-09-22 實測修正：CFR 的「未滿 7 年用 +2.25」在資料裡看不到。
#   契約期未滿 84 個月者，若套 +2.25 有 31–53% 會「超過上限」，且超額
#   高度集中在剛好 0.50（＝2.75 與 2.25 之差），其中 98.9% 契約期 <84 個月。
#   一律用 +2.75 後，超過上限從 4.78% 降到 1.11%。
#   → **實務上綁住的上限是 prime + 2.75 + 小額加成，與期限無關。**
#   制度上的原因（SOP 50 10 是否覆寫了 CFR 的期限分層）尚未查證，需查各版 SOP。
#   舊規格保留在 cap_cfr，供穩健性對照。
small_bump = np.where(amount <= 25_000, 2.00,
                      np.where(amount <= 50_000, 1.00, 0.00))   # 120.215
df["cap_cfr"] = df["prime"] + np.where(df["term_months"] < 84, 2.25, 2.75) + small_bump
df["cap"] = df["prime"] + 2.75 + small_bump
df["dist_to_cap"] = df["cap"] - df["rate"]        # 0 = 貼著上限；負 = 超過上限
df["at_cap"] = (df["dist_to_cap"].abs() < 0.01).astype(int)

# 主樣本：變動利率、非 Express 等特別方案
is_express = df["processing"].astype(str).str.contains("Express|CAPLine|Community Advantage",
                                                       case=False, na=False)
main = df[(df["is_fixed_rate"] == 0) & (~is_express) & df["prime"].notna()].copy()
print(f"\n主樣本（變動利率、排除 Express/CAPLine/CA）：{len(main):,} 筆 "
      f"（占全體 {len(main)/len(df)*100:.1f}%）")

print("\n=== 與法定上限的距離（個百分點）===")
print(main["dist_to_cap"].describe(percentiles=[.01, .05, .25, .5, .75, .95]).round(3).to_string())
print(f"\n貼著上限（|距離| < 0.01）：{main['at_cap'].mean()*100:.1f}%")
print(f"超過上限（距離 < −0.01）：{(main['dist_to_cap'] < -0.01).mean()*100:.1f}%  "
      f"← 應該很少；偏高代表基準利率或申請日的量測誤差")
print(f"低於上限 1 個百分點以上：{(main['dist_to_cap'] > 1).mean()*100:.1f}%")

# ---------------------------------------------------------------- 違約
H = 5
ok = (main["status"] == "CHGOFF") | (main["status"] == "P I F") | (main["obs_m"] >= H * 12)
d = main[ok].copy()
d["y"] = ((d["status"] == "CHGOFF") & (d["t_m"] <= H * 12)).astype(int)
print(f"\n可評估違約的樣本 {len(d):,} 筆，5 年違約率 {d['y'].mean()*100:.2f}%")

print("\n=== 核心結果：貼上限 vs 未貼上限 ===")
g = d.groupby("at_cap", observed=True).agg(n=("y", "size"), 平均利率=("rate", "mean"),
                                            平均prime=("prime", "mean"), 違約率=("y", "mean"))
g["平均利率"] = g["平均利率"].round(2); g["平均prime"] = g["平均prime"].round(2)
g["5年違約率%"] = (g["違約率"] * 100).round(2)
print(g[["n", "平均利率", "平均prime", "5年違約率%"]]
      .rename(index={0: "未貼上限", 1: "貼著法定上限"}).to_string())

print("\n=== 依與上限的距離分組 ===")
b = pd.cut(d["dist_to_cap"], [-99, -0.01, 0.01, 0.26, 0.51, 1.01, 2.01, 99],
           labels=["超過上限", "貼上限(±0.01)", "低0.01-0.25", "低0.26-0.50",
                   "低0.51-1.00", "低1.01-2.00", "低2.00以上"])
t = d.groupby(b, observed=True).agg(n=("y", "size"), 平均利率=("rate", "mean"),
                                     殘差利差=("resid", "mean"), 違約率=("y", "mean"))
t["平均利率"] = t["平均利率"].round(2); t["殘差利差"] = t["殘差利差"].round(3)
t["5年違約率%"] = (t["違約率"] * 100).round(2)
print(t[["n", "平均利率", "殘差利差", "5年違約率%"]].to_string())

print("\n=== 交叉檢查：貼上限 vs 我先前用的『落在當月模式值』===")
mode_by_ym = d.groupby("approval_ym")["rate"].agg(lambda s: s.mode().iat[0] if len(s) else np.nan)
d["at_mode"] = (d["rate"] == d["approval_ym"].map(mode_by_ym)).astype(int)
print(pd.crosstab(d["at_cap"], d["at_mode"], normalize="all").mul(100).round(1)
      .rename(index={0: "未貼上限", 1: "貼上限"},
              columns={0: "非模式值", 1: "模式值"}).to_string())
print("\n兩指標交叉下的 5 年違約率（%）")
print(d.pivot_table(index="at_cap", columns="at_mode", values="y", aggfunc="mean",
                    observed=True).mul(100).round(2)
      .rename(index={0: "未貼上限", 1: "貼上限"},
              columns={0: "非模式值", 1: "模式值"}).to_string())

print("\n=== 異質性：小額 vs 大額（金額中位數切分）===")
d["small"] = (d["log_amount"] < d["log_amount"].median()).astype(int)
p = d.pivot_table(index="small", columns="at_cap", values="y", aggfunc="mean",
                  observed=True).mul(100).round(2)
n = d.pivot_table(index="small", columns="at_cap", values="y", aggfunc="size", observed=True)
print("5 年違約率（%）"); print(p.rename(index={0: "大額", 1: "小額"},
                                        columns={0: "未貼上限", 1: "貼上限"}).to_string())
print("\n樣本數"); print(n.rename(index={0: "大額", 1: "小額"},
                                  columns={0: "未貼上限", 1: "貼上限"}).to_string())

print("\n=== 依核貸世代（確認不是單一時期的現象）===")
d["era"] = pd.cut(d["approval_fy"], [0, 2004, 2009, 2014, 9999],
                  labels=["2000-04", "2005-09", "2010-14", "2015+"])
e = d.pivot_table(index="era", columns="at_cap", values="y", aggfunc="mean",
                  observed=True).mul(100).round(2)
en = d.pivot_table(index="era", columns="at_cap", values="y", aggfunc="size", observed=True)
print("5 年違約率（%）"); print(e.rename(columns={0: "未貼上限", 1: "貼上限"}).to_string())
print("\n樣本數"); print(en.rename(columns={0: "未貼上限", 1: "貼上限"}).to_string())

out = os.path.join(SBA_DIR, "loans_with_cap.pkl")
df.to_pickle(out)
print(f"\n已存檔（含 cap / dist_to_cap / at_cap）：{out}")
