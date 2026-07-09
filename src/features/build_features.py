"""
피처 엔지니어링 함수 모음.

사용 원칙:
- 모든 변환 파라미터는 학습 데이터에서만 fit
- 시계열 피처는 shift(1) 후 rolling/lag 적용 (누수 방지)
- 인코더는 Pipeline 내에서만 사용 (분리 전 적용 금지)
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── 시계열 피처 ────────────────────────────────────────────────────────────────


def build_time_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int],
    windows: list[int],
) -> pd.DataFrame:
    """시계열 lag / rolling / 계절성 피처 생성.

    shift(1) 후 rolling을 적용하여 현재 시점 정보 유입을 방지한다.
    DatetimeIndex가 설정되어 있어야 한다.
    """
    df = df.copy()
    shifted = df[target_col].shift(1)  # 현재 시점 제거

    # Lag 피처
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # Rolling 집계
    for w in windows:
        df[f"{target_col}_roll_mean_{w}d"] = shifted.rolling(w).mean()
        df[f"{target_col}_roll_std_{w}d"] = shifted.rolling(w).std()
        df[f"{target_col}_roll_max_{w}d"] = shifted.rolling(w).max()

    # 계절성 (sin/cos 인코딩)
    df["day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    return df


# ── 정형 테이블 피처 ──────────────────────────────────────────────────────────


def build_preprocessor(
    num_cols: list[str],
    cat_cols: list[str],
) -> ColumnTransformer:
    """수치형 + 범주형 전처리 파이프라인 반환.

    반드시 sklearn Pipeline 내에서 사용해 train/val 분리 후 fit.
    """
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


# ── 피처 검증 ─────────────────────────────────────────────────────────────────


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index 계산 — 분포 드리프트 탐지.

    PSI 해석:
        < 0.1  : 안정
        0.1~0.2: 약간의 변화
        > 0.2  : 심각한 드리프트 — 재학습 또는 피처 점검 필요
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def validate_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = 0.2,
    corr_threshold: float = 0.95,
) -> dict:
    """피처 검증 리포트 생성.

    Returns:
        dict: {feature: {psi, high_corr_pairs, any_null}}
    """
    report: dict = {}

    for col in feature_cols:
        psi = compute_psi(df_train[col].dropna(), df_val[col].dropna())
        report[col] = {"psi": round(psi, 4), "psi_flag": psi > psi_threshold}

    # 중복성 확인
    corr_matrix = df_train[feature_cols].corr().abs()
    for i, col_a in enumerate(feature_cols):
        for col_b in feature_cols[i + 1 :]:
            if corr_matrix.loc[col_a, col_b] > corr_threshold:
                report.setdefault(col_a, {})["high_corr"] = col_b

    return report
