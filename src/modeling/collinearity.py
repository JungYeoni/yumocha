"""구조환경 세부/대영역 지수 간 다중공선성 진단 유틸.

방법론 메모(``reports/20260803_재정대응지수_구조환경지수_회귀설계_방법론_정리.md``
§3.3)의 "권장 진단 순서" 1~5번을 구현한다: pooled Pearson·Spearman, within(지역
평균 제거) 상관, 지역·연도 고정효과 잔차 상관, VIF·조건수. 실제 회귀에 투입할
변수 행렬과 같은 시도×연도 관측 단위에서 진단해야 하므로, 입력은 항상
"지역×연도 행, 지표 열" 형태의 wide 패널이다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

PANEL_KEY = ["지역", "연도"]


def pivot_scores_to_wide(
    long_scores: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """지역·연도·그룹 long 데이터를 지역·연도 행, 그룹 열의 wide 패널로 바꾼다."""

    required = {*PANEL_KEY, group_col, value_col}
    missing = sorted(required - set(long_scores.columns))
    if missing:
        raise KeyError(f"pivot 입력 필수 컬럼 누락: {missing}")

    wide = long_scores.pivot(index=PANEL_KEY, columns=group_col, values=value_col)
    if wide.isna().any().any():
        missing_cells = wide.isna().sum()
        raise ValueError(
            f"pivot 결과에 결측이 있습니다(지역×연도×그룹 완전격자가 아님): "
            f"{missing_cells.loc[missing_cells.gt(0)].to_dict()}"
        )
    wide.columns.name = None
    return wide.reset_index()


def _observation_count_matrix(wide: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    counts = pd.DataFrame(index=columns, columns=columns, dtype=int)
    for left in columns:
        for right in columns:
            counts.loc[left, right] = int(wide[[left, right]].dropna().shape[0])
    return counts


def compute_pooled_correlation(
    wide: pd.DataFrame, *, columns: Sequence[str], method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pooled 상관(Pearson 또는 Spearman)과 변수쌍별 관측치 수를 함께 반환한다."""

    if method not in {"pearson", "spearman"}:
        raise ValueError(f"method는 pearson 또는 spearman이어야 합니다: {method}")
    corr = wide[list(columns)].corr(method=method)
    counts = _observation_count_matrix(wide, columns)
    return corr, counts


def compute_within_region_correlation(
    wide: pd.DataFrame, *, columns: Sequence[str]
) -> pd.DataFrame:
    """각 지표에서 지역 평균을 뺀(1-way demean) 값끼리의 Pearson 상관을 계산한다."""

    demeaned = wide.copy()
    demeaned[list(columns)] = wide.groupby("지역")[list(columns)].transform(
        lambda values: values - values.mean()
    )
    return demeaned[list(columns)].corr(method="pearson")


def compute_two_way_fe_residuals(wide: pd.DataFrame, *, columns: Sequence[str]) -> pd.DataFrame:
    """지역 고정효과와 연도 고정효과를 뺀 잔차를 계산한다(균형패널 전제, 반복 이차 demean).

    ``값 - 지역평균 - 연도평균 + 전체평균``은 균형패널에서 시도·연도 더미를
    포함한 OLS(LSDV)의 잔차와 수학적으로 동일하다.
    """

    residuals = wide[list(columns)].copy()
    for column in columns:
        series = wide[column]
        region_mean = series.groupby(wide["지역"]).transform("mean")
        year_mean = series.groupby(wide["연도"]).transform("mean")
        grand_mean = series.mean()
        residuals[column] = series - region_mean - year_mean + grand_mean
    return residuals


def compute_vif(wide: pd.DataFrame, *, columns: Sequence[str]) -> pd.DataFrame:
    """변수 전체를 하나의 회귀 설계행렬로 놓고 VIF와 상관행렬 조건수를 계산한다."""

    design = wide[list(columns)].copy()
    if design.isna().any().any():
        raise ValueError("VIF 계산 대상에 결측이 있습니다.")
    if len(design) <= len(columns):
        raise ValueError(
            f"관측치 수({len(design)})가 변수 수({len(columns)})보다 많아야 VIF를 계산할 수 있습니다."
        )

    design_with_const = design.copy()
    design_with_const.insert(0, "const", 1.0)
    vif_values = []
    for index, column in enumerate(design_with_const.columns):
        if column == "const":
            continue
        vif_values.append(
            {
                "변수": column,
                "VIF": variance_inflation_factor(design_with_const.to_numpy(), index),
            }
        )
    vif_table = pd.DataFrame(vif_values)

    correlation = design.corr(method="pearson").to_numpy()
    eigenvalues = np.linalg.eigvalsh(correlation)
    if (eigenvalues <= 0).any():
        raise ValueError("상관행렬이 양의 정부호가 아니어서 조건수를 계산할 수 없습니다.")
    condition_number = float(np.sqrt(eigenvalues.max() / eigenvalues.min()))

    return vif_table, condition_number


def flag_high_correlation_pairs(corr: pd.DataFrame, *, threshold: float = 0.7) -> pd.DataFrame:
    """대각선을 제외한 상관행렬에서 |r| >= threshold인 변수쌍을 나열한다."""

    pairs = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= threshold:
                pairs.append({"변수1": left, "변수2": right, "상관계수": value})
    result = pd.DataFrame(pairs, columns=["변수1", "변수2", "상관계수"])
    return result.sort_values("상관계수", key=lambda s: s.abs(), ascending=False)
