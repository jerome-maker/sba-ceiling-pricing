#!/usr/bin/env bash
# 下載 SBA 7(a)/504 FOIA 逐筆放款資料到本機（不放 Google Drive）
#
# 用法：
#   bash scripts/00_fetch_sba.sh            # 只抓主檔 FY2010-2019 + 資料字典
#   bash scripts/00_fetch_sba.sh all        # 全部六個檔案
#   SBA_DIR=/d/data/sba bash scripts/00_fetch_sba.sh
#
# 注意：SBA 每季更新，檔名含 asof_YYMMDD。若下列檔名 404，
#       到 https://data.sba.gov/dataset/7a-504-foia 取最新檔名後更新本檔。
set -euo pipefail

SBA_DIR="${SBA_DIR:-$HOME/sba_data}"
BASE="https://data.sba.gov/sites/default/files/uploaded_resources"
ASOF="260630"

mkdir -p "$SBA_DIR"
echo "資料目錄：$SBA_DIR"

CORE=(
  "FOIA_7a_FY2010_FY2019_asof_${ASOF}.csv"
  "7a_504_foia_data_dictionary.xlsx"
)
EXTRA=(
  "FOIA_7a_FY1991_FY1999_asof_${ASOF}.csv"
  "FOIA_7a_FY2000_FY2009_asof_${ASOF}.csv"
  "FOIA_7a_FY2020_Present_asof_${ASOF}.csv"
  "FOIA_504_FY1991_FY2009_asof_${ASOF}.csv"
  "FOIA_504_FY2010_Present_asof_${ASOF}.csv"
)

FILES=("${CORE[@]}")
if [[ "${1:-}" == "all" ]]; then
  FILES+=("${EXTRA[@]}")
fi

for f in "${FILES[@]}"; do
  out="$SBA_DIR/$f"
  if [[ -s "$out" ]]; then
    echo "已存在，略過：$f"
    continue
  fi
  echo "下載：$f"
  curl -fL --retry 3 --max-time 1800 -o "$out" "$BASE/$f"
done

echo
echo "完成。目錄內容："
ls -lh "$SBA_DIR"
echo
echo "接著執行：python scripts/01_profile_sba.py"
