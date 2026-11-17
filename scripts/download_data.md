# Getting the real Home Credit data

The raw data is **not** in this repo (competition rules; also too big). Steps:

1. Create a free Kaggle account, open the competition **"Home Credit Default Risk"** and click **Join / accept the rules**
   (downloads fail with 403 until you do).
2. Kaggle -> *Settings* -> *API* -> *Create new token* -> save `kaggle.json` to `~/.kaggle/kaggle.json` (`chmod 600`),
   or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.
3. Run:

```bash
pip install kaggle
bash scripts/download_data.sh          # downloads + unzips into data/raw/ and converts CSV -> Parquet
CR_DATA_SOURCE=real make run-a         # Stage A on application_train only
CR_DATA_SOURCE=real make run-b         # Stage B: + bureau, previous_application, instalments, POS, cards
make docs                              # regenerate model card, audit, business summary from the real numbers
```

Files used: `application_train`, `bureau`, `previous_application`, `installments_payments`, `POS_CASH_balance`,
`credit_card_balance`. (`bureau_balance` and `application_test` are not used; `HomeCredit_columns_description.csv` is the data dictionary.)

Memory tip: on 8 GB machines run Stage B on Kaggle Notebooks/Colab, or keep Parquet + downcast types (done by `csv_to_parquet`).
