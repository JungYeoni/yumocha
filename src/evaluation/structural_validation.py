"""
구조환경지표 원자료 재현 검증 공통 유틸(이슈 #38).

지역×연도 행렬 두 개를 대조해 원자료 재현값과 재정팀 산출값이 일치하는지
확인하는 21개 지표 검증 작업에서, reindex 후 max()가 결측을 건너뛰어
누락·중복을 놓치는 문제와 errors="coerce"가 파싱 실패를 조용히 삼키는
문제를 막기 위한 함수들을 모아둔다.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str], *, source_name: str) -> None:
    """df에 required 컬럼이 전부 있는지 확인하고, 없으면 어떤 컬럼이 없는지 바로 알려준다."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{source_name}: 필수 컬럼 누락 {missing} (실제 컬럼: {list(df.columns)})")


def require_sheets(
    available_sheets: Iterable[str], required: Iterable[str], *, source_name: str
) -> None:
    """엑셀 워크북 등에 required 시트가 전부 있는지 확인한다."""
    available = set(available_sheets)
    missing = [sheet for sheet in required if sheet not in available]
    if missing:
        raise KeyError(f"{source_name}: 필수 시트 누락 {missing} (실제 시트: {sorted(available)})")


def to_numeric_strict(
    series: pd.Series,
    *,
    allowed_missing_tokens: frozenset[str] = frozenset({"-", ""}),
) -> pd.Series:
    """정상 결측 토큰과 파싱 실패를 구분해서 숫자로 변환한다.

    allowed_missing_tokens에 해당하는 값만 결측(NaN)으로 허용한다. 그 외
    숫자로 변환되지 않는 값이 있으면 errors="coerce"처럼 조용히 NaN으로
    바꾸지 않고 즉시 ValueError를 낸다.
    """
    is_allowed_missing = series.astype(str).str.strip().isin(allowed_missing_tokens)
    cleaned = series.mask(is_allowed_missing)
    result = pd.to_numeric(cleaned, errors="coerce")

    unexpected_failures = result.isna() & cleaned.notna()
    if unexpected_failures.any():
        bad_values = series[unexpected_failures].unique().tolist()
        raise ValueError(
            f"숫자로 변환할 수 없는 값이 있습니다(허용된 결측 토큰 아님): {bad_values}"
        )

    return result


@dataclass
class ComparisonResult:
    """compare_region_year_matrices의 반환값. bool()로 통과 여부를 바로 확인할 수 있다."""

    label: str
    comparison_count: int
    max_abs_diff: float
    mismatch_count: int
    missing_combinations: pd.DataFrame
    max_diff_location: tuple[str, int] | None
    tolerance: float
    passed: bool
    detail: pd.DataFrame = field(repr=False)

    def __bool__(self) -> bool:
        return self.passed


def compare_region_year_matrices(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
    tolerance: float = 0.0,
    label: str,
) -> ComparisonResult:
    """지역을 인덱스로, 연도를 컬럼으로 갖는 두 행렬을 대조한다.

    reindex만으로 비교하면 누락된 지역·연도가 NaN이 되고 max()가 NaN을
    건너뛰어 "최대 오차 0.0"처럼 거짓으로 통과할 수 있다. 이를 막기 위해
    기대하는 지역·연도 조합이 두 행렬 모두에 실제로 존재하는지 먼저
    확인하고, 결측 조합·중복 인덱스·불일치 위치를 결과에 전부 남긴다.
    """
    for name, frame in (("left", left), ("right", right)):
        if frame.index.duplicated().any():
            dupes = frame.index[frame.index.duplicated()].unique().tolist()
            raise ValueError(f"{label} - {name}: 지역 인덱스 중복 {dupes}")

    missing_rows = []
    for region in expected_regions:
        for year in expected_years:
            in_left = region in left.index and year in left.columns
            in_right = region in right.index and year in right.columns
            if not (in_left and in_right):
                missing_rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "left_있음": in_left,
                        "right_있음": in_right,
                    }
                )
    missing_combinations = pd.DataFrame(
        missing_rows, columns=["지역", "연도", "left_있음", "right_있음"]
    )

    left_aligned = left.reindex(index=expected_regions, columns=expected_years)
    right_aligned = right.reindex(index=expected_regions, columns=expected_years)
    diff = (left_aligned - right_aligned).abs()

    comparable = diff.notna()
    comparison_count = int(comparable.sum().sum())

    max_abs_diff = float("nan")
    max_diff_location = None
    if comparison_count > 0:
        stacked = diff.stack()
        max_diff_location = stacked.idxmax()
        max_abs_diff = float(stacked.loc[max_diff_location])

    mismatch_mask = comparable & (diff > tolerance)
    mismatch_count = int(mismatch_mask.sum().sum())

    detail_rows = []
    for region in expected_regions:
        for year in expected_years:
            if region not in diff.index or year not in diff.columns:
                continue
            d = diff.loc[region, year]
            if pd.isna(d) or d > tolerance:
                detail_rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "left": left_aligned.loc[region, year],
                        "right": right_aligned.loc[region, year],
                        "절대오차": d,
                    }
                )
    detail = pd.DataFrame(detail_rows, columns=["지역", "연도", "left", "right", "절대오차"])

    fully_covered = comparison_count == len(expected_regions) * len(expected_years)
    passed = fully_covered and mismatch_count == 0

    return ComparisonResult(
        label=label,
        comparison_count=comparison_count,
        max_abs_diff=max_abs_diff,
        mismatch_count=mismatch_count,
        missing_combinations=missing_combinations,
        max_diff_location=max_diff_location,
        tolerance=tolerance,
        passed=passed,
        detail=detail,
    )


def to_verification_record(
    result: ComparisonResult, *, indicator_id: str, stage: str
) -> dict[str, object]:
    """ComparisonResult를 검증 결과 누적 테이블의 한 행으로 변환한다."""
    return {
        "지표ID": indicator_id,
        "검증단계": stage,
        "비교건수": result.comparison_count,
        "최대절대오차": result.max_abs_diff,
        "불일치건수": result.mismatch_count,
        "결측조합수": len(result.missing_combinations),
        "최대오차위치": result.max_diff_location,
        "판정": "정상" if result.passed else "확인 필요",
    }


def weighted_response_mean(
    df: pd.DataFrame,
    *,
    scores: dict[str, float],
    expected_regions: Sequence[str],
) -> pd.Series:
    """응답 비율표에서 항목별 점수로 가중평균을 계산한다(지역을 인덱스로 반환).

    사회조사류 응답 비율은 반올림 때문에 합이 정확히 100이 아닐 수 있어,
    100이 아니라 각 행의 항목(scores의 키) 실제 합으로 나눈다.
    """
    require_columns(df, scores.keys(), source_name="weighted_response_mean 입력")

    missing_regions = sorted(set(expected_regions) - set(df.index))
    if missing_regions:
        raise ValueError(f"weighted_response_mean: 지역 누락 {missing_regions}")

    category_cols = list(scores.keys())
    category_sum = df[category_cols].sum(axis=1)
    weighted_sum = sum(df[category] * score for category, score in scores.items())
    return weighted_sum / category_sum


__all__ = [
    "ComparisonResult",
    "compare_region_year_matrices",
    "require_columns",
    "require_sheets",
    "to_numeric_strict",
    "to_verification_record",
    "weighted_response_mean",
]
