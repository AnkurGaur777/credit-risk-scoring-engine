"""Request / response models with sensible validation ranges."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Applicant(BaseModel):
    """Application-time fields only (Stage A model). Field names follow the Home Credit data dictionary."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "NAME_CONTRACT_TYPE": "Cash loans",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "CNT_CHILDREN": 1,
                "AMT_INCOME_TOTAL": 135000,
                "AMT_CREDIT": 450000,
                "AMT_ANNUITY": 22500,
                "AMT_GOODS_PRICE": 430000,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "DAYS_EMPLOYED": -1800,
                "DAYS_REGISTRATION": -3200,
                "DAYS_ID_PUBLISH": -2100,
                "DAYS_LAST_PHONE_CHANGE": -400,
                "OCCUPATION_TYPE": "Laborers",
                "CNT_FAM_MEMBERS": 3,
                "EXT_SOURCE_1": 0.45,
                "EXT_SOURCE_2": 0.52,
                "EXT_SOURCE_3": 0.40,
                "AMT_REQ_CREDIT_BUREAU_YEAR": 2,
            }
        },
    )

    NAME_CONTRACT_TYPE: Literal["Cash loans", "Revolving loans"] = "Cash loans"
    FLAG_OWN_CAR: Literal["Y", "N"] | None = None
    FLAG_OWN_REALTY: Literal["Y", "N"] | None = None
    CNT_CHILDREN: int = Field(0, ge=0, le=20)
    AMT_INCOME_TOTAL: float = Field(..., gt=0, le=1e8)
    AMT_CREDIT: float = Field(..., gt=0, le=1e8)
    AMT_ANNUITY: float = Field(..., gt=0, le=1e7)
    AMT_GOODS_PRICE: float | None = Field(None, gt=0, le=1e8)
    NAME_INCOME_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    NAME_FAMILY_STATUS: str | None = None
    NAME_HOUSING_TYPE: str | None = None
    DAYS_EMPLOYED: float | None = Field(
        None, description="Days before application (negative). 365243 = not employed.", le=365243
    )
    DAYS_REGISTRATION: float | None = Field(None, le=0, ge=-40000)
    DAYS_ID_PUBLISH: float | None = Field(None, le=0, ge=-20000)
    DAYS_LAST_PHONE_CHANGE: float | None = Field(None, le=0, ge=-20000)
    OCCUPATION_TYPE: str | None = None
    ORGANIZATION_TYPE: str | None = None
    CNT_FAM_MEMBERS: float | None = Field(None, ge=1, le=30)
    EXT_SOURCE_1: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(None, ge=0, le=1)
    FLAG_DOCUMENT_3: int | None = Field(None, ge=0, le=1)
    FLAG_DOCUMENT_6: int | None = Field(None, ge=0, le=1)
    AMT_REQ_CREDIT_BUREAU_HOUR: float | None = Field(None, ge=0, le=50)
    AMT_REQ_CREDIT_BUREAU_DAY: float | None = Field(None, ge=0, le=50)
    AMT_REQ_CREDIT_BUREAU_WEEK: float | None = Field(None, ge=0, le=100)
    AMT_REQ_CREDIT_BUREAU_MON: float | None = Field(None, ge=0, le=100)
    AMT_REQ_CREDIT_BUREAU_QRT: float | None = Field(None, ge=0, le=300)
    AMT_REQ_CREDIT_BUREAU_YEAR: float | None = Field(None, ge=0, le=500)


class ReasonCode(BaseModel):
    feature: str
    reason: str
    impact_log_odds: float


class ScoreResponse(BaseModel):
    request_id: str
    model_version: str
    pd: float = Field(..., description="Calibrated probability of default")
    score: int = Field(..., description="Points on the scorecard scale (higher = safer)")
    risk_band: Literal["very low", "low", "medium", "high", "very high"]
    decision: Literal["approve", "decline"]
    cutoff_pd: float
    reason_codes: list[ReasonCode]
    latency_ms: float
    disclaimer: str = "Educational demonstration - not a production lending decision."


class Health(BaseModel):
    status: str
    model_version: str | None
    stage: str | None
    loaded: bool
