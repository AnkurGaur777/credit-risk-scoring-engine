"""Synthetic look-alike of the Home Credit tables (same column names, same quirks).

NOT real data. It exists so the whole pipeline can be run and tested without a Kaggle account.
A latent 'creditworthiness' q drives the application features; a latent 'behaviour' b only shows up in
the history tables, so Stage B (extra tables) genuinely adds signal, as in the real data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(z: np.ndarray, target_rate: float) -> float:
    lo, hi = -12.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(n: int = 20000, seed: int = C.SEED, target_rate: float = 0.08) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100001, 100001 + n)
    q = rng.normal(size=n)  # creditworthiness (high = good), seen through application fields
    b = rng.normal(size=n)  # behaviour (high = bad), seen only in history tables
    q = q - 0.25 * b  # mild correlation between the two

    age = np.clip(rng.normal(43, 11, n), 21, 69)
    gender = rng.choice(["F", "M", "XNA"], n, p=[0.655, 0.344, 0.001])
    income = np.clip(np.exp(rng.normal(np.log(140000) + 0.12 * q, 0.5)), 25000, 4_000_000).round(-2)
    contract = rng.choice(["Cash loans", "Revolving loans"], n, p=[0.905, 0.095])
    credit = (income * np.clip(rng.lognormal(np.log(3.2), 0.5, n), 0.6, 12)).round(-3)
    term = rng.uniform(8, 40, n)
    annuity = (credit / term).round(1)
    goods = (credit * rng.uniform(0.82, 1.0, n)).round(-3)
    goods[rng.random(n) < 0.0009] = np.nan
    pensioner = rng.random(n) < (0.17 + 0.004 * (age - 43).clip(0))
    emp_years = np.clip(rng.gamma(2.0, 3.0, n) * (1 + 0.06 * q), 0, age - 18)
    days_emp = np.where(pensioner, C.DAYS_EMPLOYED_SENTINEL, -(emp_years * 365.25).round()).astype(float)
    income_type = np.where(
        pensioner,
        "Pensioner",
        rng.choice(
            ["Working", "Commercial associate", "State servant", "Student"], n, p=[0.62, 0.27, 0.105, 0.005]
        ),
    )
    education = rng.choice(
        [
            "Secondary / secondary special",
            "Higher education",
            "Incomplete higher",
            "Lower secondary",
            "Academic degree",
        ],
        n,
        p=[0.71, 0.24, 0.033, 0.012, 0.005],
    )
    family = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        n,
        p=[0.64, 0.145, 0.095, 0.065, 0.055],
    )
    housing = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment"],
        n,
        p=[0.89, 0.048, 0.037, 0.016, 0.009],
    )
    occ_pool = np.array(
        [
            "Laborers",
            "Sales staff",
            "Core staff",
            "Managers",
            "Drivers",
            "High skill tech staff",
            "Accountants",
            "Medicine staff",
            "Cooking staff",
            "Security staff",
        ]
    )
    occupation = rng.choice(occ_pool, n).astype(object)
    occupation[(rng.random(n) < 0.31) | pensioner] = np.nan
    org = rng.choice(
        [
            "Business Entity Type 3",
            "Self-employed",
            "Other",
            "Medicine",
            "Government",
            "School",
            "Trade: type 7",
        ],
        n,
    ).astype(object)
    org[pensioner] = "XNA"
    fam_members = np.clip(rng.poisson(1.2, n) + 1, 1, 9).astype(float)
    children = np.clip(fam_members - 1 - (family == "Married"), 0, 8)
    own_car = (rng.random(n) < _sigmoid(-0.6 + 0.4 * np.log(income / 140000) + 0.1 * q)).astype(int)
    own_realty = (rng.random(n) < 0.69).astype(int)

    def ext(loading, miss_p):
        v = 0.5 + 0.2 * (loading * q + np.sqrt(1 - loading**2) * rng.normal(size=n))
        v = np.clip(v, 0.01, 0.99)
        v[rng.random(n) < miss_p] = np.nan
        return v

    e1, e2, e3 = ext(0.55, 0.56), ext(0.72, 0.002), ext(0.62, 0.20)

    enq_year = rng.poisson(np.clip(1.6 + 0.35 * b, 0.1, None)).astype(float)
    enq_mon = rng.poisson(0.3, n).astype(float)
    enq_week = rng.poisson(0.03, n).astype(float)
    enq_qrt = rng.poisson(0.6, n).astype(float)
    enq_day = rng.poisson(0.007, n).astype(float)
    enq_hour = rng.poisson(0.006, n).astype(float)
    enq_missing = rng.random(n) < 0.135
    enq = {
        "HOUR": enq_hour,
        "DAY": enq_day,
        "WEEK": enq_week,
        "MON": enq_mon,
        "QRT": enq_qrt,
        "YEAR": enq_year,
    }
    for v in enq.values():
        v[enq_missing] = np.nan

    apt = np.where(rng.random(n) < 0.50, np.nan, np.clip(rng.beta(2, 8, n) + 0.01 * q, 0, 1))
    expl = np.where(rng.random(n) < 0.49, np.nan, np.clip(rng.normal(0.98, 0.06, n), 0.5, 1.0))
    floors = np.where(rng.random(n) < 0.68, np.nan, np.clip(rng.beta(2, 4, n), 0, 1))
    docs3 = (rng.random(n) < 0.71).astype(int)
    docs6 = (rng.random(n) < 0.09).astype(int)

    # ---- target ---------------------------------------------------------------------------
    ai = annuity / income
    ci = credit / income
    z = (
        -0.60 * q  # via EXT_SOURCE_*, education etc.
        + 0.80 * b  # history-table behaviour only
        + 0.25 * (ai - 0.18) / 0.09
        + 0.18 * (ci - 3.2) / 2.0
        - 0.22 * (emp_years - 6) / 6
        + 0.25 * (contract == "Revolving loans") * -1
        + 0.30 * (education == "Lower secondary")
        + 0.20 * (education == "Secondary / secondary special")
        - 0.35 * (education == "Higher education")
        + 0.25 * (housing == "With parents")
        + 0.18 * (housing == "Rented apartment")
        - 0.10 * own_car
        + 0.08 * (np.nan_to_num(enq_year, nan=1.6) - 1.6)
        + 0.12 * (occupation == "Laborers")
        + 0.10 * (occupation == "Drivers")
        - 0.015 * (age - 43)  # younger -> riskier (audit will show an age gap)
        + 0.10 * (gender == "M")  # latent gender gap (audit will find it via proxies)
        + 0.12 * (np.isnan(e3))
    )
    z = z + rng.normal(0, 0.35, n)
    intercept = _solve_intercept(z, target_rate)
    p = _sigmoid(z + intercept)
    target = (rng.random(n) < p).astype(int)

    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": target,
            "NAME_CONTRACT_TYPE": contract,
            "CODE_GENDER": gender,
            "FLAG_OWN_CAR": np.where(own_car == 1, "Y", "N"),
            "FLAG_OWN_REALTY": np.where(own_realty == 1, "Y", "N"),
            "CNT_CHILDREN": children,
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": education,
            "NAME_FAMILY_STATUS": family,
            "NAME_HOUSING_TYPE": housing,
            "DAYS_BIRTH": -(age * 365.25).round(),
            "DAYS_EMPLOYED": days_emp,
            "DAYS_REGISTRATION": -np.clip(rng.gamma(2.5, 2800, n), 0, 24000).round(),
            "DAYS_ID_PUBLISH": -np.clip(rng.gamma(2.2, 1100, n), 0, 7000).round(),
            "DAYS_LAST_PHONE_CHANGE": -np.clip(rng.gamma(1.5, 500, n), 0, 4000).round(),
            "OCCUPATION_TYPE": occupation,
            "CNT_FAM_MEMBERS": fam_members,
            "ORGANIZATION_TYPE": org,
            "EXT_SOURCE_1": e1,
            "EXT_SOURCE_2": e2,
            "EXT_SOURCE_3": e3,
            "APARTMENTS_AVG": apt,
            "YEARS_BEGINEXPLUATATION_AVG": expl,
            "FLOORSMAX_AVG": floors,
            "FLAG_DOCUMENT_3": docs3,
            "FLAG_DOCUMENT_6": docs6,
            **{f"AMT_REQ_CREDIT_BUREAU_{k}": v for k, v in enq.items()},
        }
    )

    # ---- history tables (Stage B) ---------------------------------------------------------
    sb = _sigmoid(1.2 * b)
    # bureau
    nb = rng.poisson(3.0, n)
    rep = np.repeat(np.arange(n), nb)
    m = len(rep)
    bsb = sb[rep]
    active = rng.random(m) < (0.30 + 0.25 * bsb)
    overdue_flag = rng.random(m) < (0.01 + 0.35 * bsb)
    credit_sum = np.exp(rng.normal(11.8, 1.1, m)).round(-2)
    debt = np.where(active, credit_sum * rng.uniform(0.1, 1.0, m), 0.0).round(-2)
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": ids[rep],
            "SK_ID_BUREAU": np.arange(5_000_000, 5_000_000 + m),
            "CREDIT_ACTIVE": np.where(active, "Active", "Closed"),
            "DAYS_CREDIT": -rng.integers(20, 3000, m).astype(float),
            "CREDIT_DAY_OVERDUE": np.where(overdue_flag, rng.integers(1, 120, m), 0),
            "DAYS_CREDIT_ENDDATE": np.where(
                active, rng.integers(10, 1500, m), -rng.integers(10, 2500, m)
            ).astype(float),
            "AMT_CREDIT_SUM": credit_sum,
            "AMT_CREDIT_SUM_DEBT": debt,
            "AMT_CREDIT_SUM_OVERDUE": np.where(
                overdue_flag, credit_sum * rng.uniform(0.001, 0.05, m), 0.0
            ).round(1),
        }
    )
    # previous applications
    npv = rng.poisson(2.6, n)
    rep_p = np.repeat(np.arange(n), npv)
    mp = len(rep_p)
    bsp = sb[rep_p]
    u = rng.random(mp)
    refuse_p = 0.08 + 0.45 * bsp
    status = np.where(
        u < refuse_p,
        "Refused",
        np.where(u < refuse_p + 0.08, "Canceled", np.where(u < refuse_p + 0.10, "Unused offer", "Approved")),
    )
    amt_app = np.exp(rng.normal(11.0, 1.0, mp)).round(-2)
    prev = pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(2_000_000, 2_000_000 + mp),
            "SK_ID_CURR": ids[rep_p],
            "NAME_CONTRACT_STATUS": status,
            "AMT_APPLICATION": amt_app,
            "AMT_CREDIT": np.where(status == "Approved", amt_app * rng.uniform(0.85, 1.1, mp), 0.0).round(-2),
            "AMT_ANNUITY": (amt_app / rng.uniform(6, 30, mp)).round(1),
            "DAYS_DECISION": -rng.integers(15, 2900, mp).astype(float),
        }
    )
    appr = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].reset_index(drop=True)
    # instalments for approved previous loans (only instalments due before the current application)
    k = 10
    rep_i = np.repeat(np.arange(len(appr)), k)
    kk = np.tile(np.arange(1, k + 1), len(appr))
    due = appr["DAYS_DECISION"].to_numpy()[rep_i] + 30.0 * kk
    keep = due <= -1
    rep_i, kk, due = rep_i[keep], kk[keep], due[keep]
    sbi = sb[(appr["SK_ID_CURR"].to_numpy()[rep_i] - ids[0])]
    late = rng.random(len(rep_i)) < (0.03 + 0.45 * sbi)
    delay = np.where(late, rng.exponential(12, len(rep_i)), -rng.integers(0, 6, len(rep_i)))
    inst_amt = (appr["AMT_CREDIT"].to_numpy()[rep_i] / k * 1.08).round(1)
    short = rng.random(len(rep_i)) < (0.01 + 0.20 * sbi)
    inst = pd.DataFrame(
        {
            "SK_ID_PREV": appr["SK_ID_PREV"].to_numpy()[rep_i],
            "SK_ID_CURR": appr["SK_ID_CURR"].to_numpy()[rep_i],
            "DAYS_INSTALMENT": due,
            "DAYS_ENTRY_PAYMENT": np.minimum(due + delay.round(), -1.0),
            "AMT_INSTALMENT": inst_amt,
            "AMT_PAYMENT": np.where(short, inst_amt * rng.uniform(0.2, 0.9, len(rep_i)), inst_amt).round(1),
        }
    )
    # POS cash balance on a subset of approved previous loans
    sub = appr.sample(frac=0.6, random_state=seed).reset_index(drop=True)
    mths = 8
    rep_s = np.repeat(np.arange(len(sub)), mths)
    sbs = sb[(sub["SK_ID_CURR"].to_numpy()[rep_s] - ids[0])]
    dpd = np.where(rng.random(len(rep_s)) < (0.01 + 0.35 * sbs), rng.integers(1, 60, len(rep_s)), 0)
    pos = pd.DataFrame(
        {
            "SK_ID_PREV": sub["SK_ID_PREV"].to_numpy()[rep_s],
            "SK_ID_CURR": sub["SK_ID_CURR"].to_numpy()[rep_s],
            "MONTHS_BALANCE": -(np.tile(np.arange(1, mths + 1), len(sub))).astype(float),
            "SK_DPD": dpd,
            "SK_DPD_DEF": np.where(dpd > 5, dpd, 0),
        }
    )
    # credit cards for ~35% of applicants
    has_card = np.where(rng.random(n) < 0.35)[0]
    mc = 12
    rep_c = np.repeat(has_card, mc)
    limit = np.repeat(np.exp(rng.normal(11.6, 0.6, len(has_card))).round(-3), mc)
    sbc = sb[rep_c]
    util = np.clip(rng.beta(2, 5, len(rep_c)) + 0.5 * (sbc - 0.5), 0, 1.3)
    cc = pd.DataFrame(
        {
            "SK_ID_PREV": np.repeat(np.arange(9_000_000, 9_000_000 + len(has_card)), mc),
            "SK_ID_CURR": ids[rep_c],
            "MONTHS_BALANCE": -(np.tile(np.arange(1, mc + 1), len(has_card))).astype(float),
            "AMT_BALANCE": (limit * util).round(1),
            "AMT_CREDIT_LIMIT_ACTUAL": limit,
            "AMT_DRAWINGS_CURRENT": (limit * util * rng.uniform(0, 0.4, len(rep_c))).round(1),
            "SK_DPD": np.where(
                rng.random(len(rep_c)) < (0.01 + 0.25 * sbc), rng.integers(1, 40, len(rep_c)), 0
            ),
        }
    )
    return {
        "application_train": app,
        "bureau": bureau,
        "previous_application": prev,
        "installments_payments": inst,
        "POS_CASH_balance": pos,
        "credit_card_balance": cc,
    }


def write(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic Home-Credit-like dataset")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", type=Path, default=C.ROOT / "data" / "synthetic")
    a = ap.parse_args()
    t = generate(a.n, a.seed)
    write(t, a.out)
    print(f"wrote {len(t)} tables to {a.out}; target rate {t['application_train']['TARGET'].mean():.3%}")


if __name__ == "__main__":
    main()
