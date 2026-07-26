"""구조환경지표·계획예산 지역×연도 추세 EDA 준비 유틸."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.evaluation.structural_validation import (
    normalize_nationwide_labels,
    require_columns,
    to_numeric_strict,
)


STRUCTURAL_ID_COLUMNS = ["지역", "대영역", "세부영역", "세부지표", "검증상태"]


def classify_basic_plan_period(year: int) -> str:
    """연도를 저출산·고령사회 기본계획 기간으로 구분한다."""

    if 2016 <= year <= 2020:
        return "제3차 기본계획(2016~2020)"
    if 2021 <= year <= 2024:
        return "제4차 기본계획(2021~2024)"
    return "분석기간 외"


def _add_iqr_outlier_flag(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    change_col: str,
    flag_col: str,
) -> pd.DataFrame:
    result = df.copy()
    grouped = result.groupby(list(group_cols))[change_col]
    q1 = grouped.transform(lambda values: values.quantile(0.25))
    q3 = grouped.transform(lambda values: values.quantile(0.75))
    iqr = q3 - q1
    result[flag_col] = result[change_col].notna() & (
        result[change_col].lt(q1 - 1.5 * iqr) | result[change_col].gt(q3 + 1.5 * iqr)
    )
    return result


def reshape_structural_indicators(
    df: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    years: Sequence[int] = tuple(range(2016, 2026)),
) -> pd.DataFrame:
    """구조환경지표 wide 표를 실측·결측 플래그가 있는 long 표로 변환한다."""

    year_cols = [str(year) for year in years]
    require_columns(
        df,
        [*STRUCTURAL_ID_COLUMNS, *year_cols],
        source_name="구조환경지표 검증본",
    )
    normalized = normalize_nationwide_labels(df)

    for indicator, group in normalized.groupby("세부지표"):
        actual_regions = set(group.loc[~group["지역"].eq("전국"), "지역"])
        missing_regions = sorted(set(expected_regions) - actual_regions)
        unexpected_regions = sorted(actual_regions - set(expected_regions))
        if missing_regions or unexpected_regions:
            raise ValueError(
                f"{indicator} 17개 시도 구성 불일치: "
                f"누락={missing_regions}, 예상외={unexpected_regions}"
            )

    long = normalized.melt(
        id_vars=STRUCTURAL_ID_COLUMNS,
        value_vars=year_cols,
        var_name="연도",
        value_name="측정값",
    )
    long["연도"] = long["연도"].astype(int)
    long["측정값"] = to_numeric_strict(long["측정값"])
    long["실측여부"] = long["측정값"].notna()
    long["결측상태"] = long["실측여부"].map({True: "실측", False: "원자료 결측"})
    long["관측치용도"] = long["지역"].eq("전국").map({True: "전국 참고", False: "17개 시도 분석"})
    long["기본계획기간"] = long["연도"].map(classify_basic_plan_period)

    duplicated = long.duplicated(["지역", "연도", "세부지표"], keep=False)
    if duplicated.any():
        keys = long.loc[duplicated, ["지역", "연도", "세부지표"]].drop_duplicates()
        raise ValueError(f"구조환경지표 지역×연도 키 중복: {keys.to_dict('records')}")

    long = long.sort_values(["세부지표", "지역", "연도"]).reset_index(drop=True)
    region_rows = long.loc[~long["지역"].eq("전국")].copy()
    grouped = region_rows.groupby(["세부지표", "지역"], sort=False)
    previous_year = grouped["연도"].shift(1)
    previous_value = grouped["측정값"].shift(1)
    consecutive = region_rows["연도"].sub(previous_year).eq(1)
    region_rows["전년대비변화"] = region_rows["측정값"].sub(previous_value).where(consecutive)
    region_rows = _add_iqr_outlier_flag(
        region_rows,
        group_cols=["세부지표", "연도"],
        change_col="전년대비변화",
        flag_col="급등락후보",
    )

    return long.merge(
        region_rows[["지역", "연도", "세부지표", "전년대비변화", "급등락후보"]],
        on=["지역", "연도", "세부지표"],
        how="left",
        validate="one_to_one",
    )


def prepare_budget_trends(
    budget_panel: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int] = tuple(range(2016, 2025)),
) -> pd.DataFrame:
    """#53 기초패널에 기본계획 기간·증감률·급등락 후보를 추가한다."""

    required = {"지역", "연도", "당해계획예산_백만원", "원자료_누락주의"}
    missing = sorted(required - set(budget_panel.columns))
    if missing:
        raise ValueError(f"계획예산 기초패널 필수 컬럼 누락: {missing}")

    result = budget_panel.copy()
    if result["지역"].eq("전국").any():
        raise ValueError("계획예산 지역 패널에 전국 행이 포함되어 있습니다.")

    actual_keys = set(result[["지역", "연도"]].itertuples(index=False, name=None))
    expected_keys = {(region, year) for region in expected_regions for year in expected_years}
    if actual_keys != expected_keys:
        raise ValueError(
            "계획예산 지역×연도 조합 불일치: "
            f"누락={sorted(expected_keys - actual_keys)[:10]}, "
            f"예상외={sorted(actual_keys - expected_keys)[:10]}"
        )

    result = result.sort_values(["지역", "연도"]).reset_index(drop=True)
    result["기본계획기간"] = result["연도"].map(classify_basic_plan_period)
    grouped = result.groupby("지역", sort=False)
    previous_year = grouped["연도"].shift(1)
    previous_budget = grouped["당해계획예산_백만원"].shift(1)
    consecutive = result["연도"].sub(previous_year).eq(1)
    result["전년대비증감률_pct"] = (
        result["당해계획예산_백만원"].div(previous_budget).sub(1).mul(100).where(consecutive)
    )
    return _add_iqr_outlier_flag(
        result,
        group_cols=["연도"],
        change_col="전년대비증감률_pct",
        flag_col="급등락후보",
    )


__all__ = [
    "classify_basic_plan_period",
    "prepare_budget_trends",
    "reshape_structural_indicators",
]
