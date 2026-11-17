#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
kaggle competitions download -c home-credit-default-risk -p data/raw
(cd data/raw && unzip -o home-credit-default-risk.zip && rm -f home-credit-default-risk.zip)
mv -f data/raw/HomeCredit_columns_description.csv docs/ 2>/dev/null || true
python - <<'PY'
from credit_risk import data
print("converted to parquet:", data.csv_to_parquet())
PY
