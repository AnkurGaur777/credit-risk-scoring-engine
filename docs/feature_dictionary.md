# Feature dictionary (engineered features)

Raw columns follow `HomeCredit_columns_description`.

| feature | how computed | why it should matter |
|---|---|---|
| CREDIT_INCOME_RATIO | AMT_CREDIT / AMT_INCOME_TOTAL | Larger loans relative to income are harder to repay. |
| ANNUITY_INCOME_RATIO | AMT_ANNUITY / AMT_INCOME_TOTAL | Payment burden: share of income consumed each period. |
| CREDIT_GOODS_RATIO | AMT_CREDIT / AMT_GOODS_PRICE | Above 1 means the borrower finances more than the goods cost (fees, add-ons). |
| ANNUITY_CREDIT_RATIO | AMT_ANNUITY / AMT_CREDIT | Proxy for loan term: high ratio = short term = heavy instalments. |
| INCOME_PER_FAMILY_MEMBER | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS | Disposable income is shared across the household. |
| YEARS_EMPLOYED | -DAYS_EMPLOYED / 365.25 (sentinel -> NaN) | Job stability. |
| EMPLOYED_TO_AGE_RATIO | YEARS_EMPLOYED / AGE_YEARS | Share of adult life in work. Age-derived: excluded by default policy. |
| YEARS_REGISTRATION | -DAYS_REGISTRATION / 365.25 | Residential/registration stability. |
| YEARS_ID_PUBLISH | -DAYS_ID_PUBLISH / 365.25 | Recently re-issued ID can signal instability. |
| YEARS_PHONE_CHANGE | -DAYS_LAST_PHONE_CHANGE / 365.25 | Recent phone change is a mild instability signal. |
| EXT_SOURCE_1_MISSING | EXT_SOURCE_1 is NaN | Missing external score is informative, not random. |
| EXT_SOURCE_3_MISSING | EXT_SOURCE_3 is NaN | Missing external score is informative, not random. |
| HOUSING_FEATURES_MISSING_SHARE | share of *_AVG housing columns that are NaN | Proxy for how much is known about the home. |
| EXT_SOURCE_MEAN | mean of available EXT_SOURCE_1..3 | Combined external credit opinion. |
| EXT_SOURCE_MIN | min of available EXT_SOURCE_1..3 | Weakest external opinion. |
| EXT_SOURCE_PROD | product of available EXT_SOURCE_1..3 | Interaction: all three must be good. |
| EXT_SOURCE_WEIGHTED | weighted mean (1:2, 2:3, 3:4) | Weights roughly follow individual predictive strength. |
| DOCUMENTS_PROVIDED | sum of FLAG_DOCUMENT_* columns | Documentation completeness. |
| BUREAU_ENQUIRIES_TOTAL | sum of AMT_REQ_CREDIT_BUREAU_* columns | Credit-seeking intensity. |
| BUREAU_ENQUIRIES_RECENT | HOUR + DAY + WEEK + MON enquiries | Recent credit hunger. |
| BUREAU_COUNT | number of past bureau credits | Credit experience. |
| BUREAU_ACTIVE_COUNT | number of Active bureau credits | Current outstanding obligations. |
| BUREAU_DEBT_SUM | sum AMT_CREDIT_SUM_DEBT | Total debt at other lenders. |
| BUREAU_OVERDUE_SUM | sum AMT_CREDIT_SUM_OVERDUE | Overdue amounts at other lenders. |
| BUREAU_MAX_DAYS_OVERDUE | max CREDIT_DAY_OVERDUE | Worst past delinquency. |
| BUREAU_DEBT_CREDIT_RATIO | sum debt / sum credit | Bureau utilisation. |
| BUREAU_AVG_CREDIT_AGE_YEARS | mean -DAYS_CREDIT / 365.25 | Length of credit history. |
| PREV_COUNT | number of previous applications | Application history. |
| PREV_REFUSED_RATIO | refused / all previous applications | Other lenders' (and our own) past rejections. |
| PREV_APPROVED_COUNT | approved previous applications | Successful borrowing history. |
| PREV_AVG_CREDIT | mean AMT_CREDIT of approved | Typical past loan size. |
| PREV_DAYS_SINCE_LAST | min -DAYS_DECISION | Time since last application. |
| INST_LATE_SHARE | share of instalments paid after due date | Core behavioural risk signal. |
| INST_AVG_DAYS_LATE | mean max(0, entry - due) | Severity of lateness. |
| INST_PAYMENT_RATIO | sum paid / sum due | Under-payment. |
| POS_DPD_MONTHS | months with SK_DPD > 0 | Delinquency on POS/cash loans. |
| POS_MAX_DPD | max SK_DPD | Worst delinquency on POS/cash loans. |
| CC_UTILISATION_MEAN | mean AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL | Card utilisation. |
| CC_DPD_MONTHS | months with SK_DPD > 0 on cards | Delinquency on cards. |
| CC_DRAWINGS_RATIO | mean drawings / limit | Cash-drawing behaviour. |
