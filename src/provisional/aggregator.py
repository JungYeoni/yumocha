"""Label-dependent provisional area aggregation for #81."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_full_grid(
    panel: pd.DataFrame,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
    *,
    category_column: str | None = None,
    expected_categories: Sequence[str] | None = None,
) -> None:
    key_columns = ["지역", "연도"]
    if category_column is not None:
        if expected_categories is None:
            raise ValueError("category_column 사용 시 expected_categories가 필요합니다.")
        key_columns.append(category_column)
    missing_columns = sorted(set(key_columns) - set(panel.columns))
    if missing_columns:
        raise KeyError(f"완전격자 검증 컬럼 누락: {missing_columns}")
    if panel.duplicated(key_columns).any():
        samples = panel.loc[panel.duplicated(key_columns, keep=False), key_columns].head(10)
        raise ValueError(f"완전격자 키 중복: {samples.to_dict(orient='records')}")

    regions = [str(value).strip() for value in expected_regions]
    years = [int(value) for value in expected_years]
    if len(regions) != len(set(regions)) or len(years) != len(set(years)):
        raise ValueError("예상 지역·연도에 중복이 있습니다.")
    if category_column is None:
        expected = {(region, year) for region in regions for year in years}
    else:
        categories = [str(value).strip() for value in expected_categories or []]
        if not categories or len(categories) != len(set(categories)):
            raise ValueError("예상 영역은 비어 있거나 중복될 수 없습니다.")
        expected = {
            (region, year, category)
            for region in regions
            for year in years
            for category in categories
        }

    actual = set(map(tuple, panel[key_columns].itertuples(index=False, name=None)))
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"완전격자 검증 실패. 누락={missing[:10]}, 예상외={unexpected[:10]}")


def _fill_empty_combinations(
    panel: pd.DataFrame,
    *,
    category_column: str,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
    expected_categories: Sequence[str],
) -> pd.DataFrame:
    """Add explicit zero rows for 지역·연도·카테고리 combinations with no detail rows.

    A combination genuinely absent from ``labeled_long`` (e.g. a region did not
    fund any project in a category that year) means the sum over an empty set,
    which is 0 — not an invented or estimated value. ``사업수=0`` keeps this
    distinguishable from a real observation, so nothing is fabricated.

    ``reindex``는 ``panel``에 있지만 기대 목록 밖인 지역·연도·카테고리 값을 조용히
    버린다 — 완전격자를 만드는 정상 동작(원본에 없는 조합을 0으로 채움)과 겉보기엔
    똑같아서, 실제 라벨 오타가 있어도 이후 ``validate_full_grid``가 못 잡는다(reindex가
    이미 정확히 기대 그리드로 맞춰놨기 때문). 그래서 reindex 전에 명시적으로 막는다.
    """
    unexpected_regions = sorted(set(panel["지역"]) - set(expected_regions))
    if unexpected_regions:
        raise ValueError(f"완전격자 채움 대상에 예상 밖 지역이 있습니다: {unexpected_regions}")
    unexpected_years = sorted(set(panel["연도"]) - {int(year) for year in expected_years})
    if unexpected_years:
        raise ValueError(f"완전격자 채움 대상에 예상 밖 연도가 있습니다: {unexpected_years}")
    unexpected_categories = sorted(set(panel[category_column]) - set(expected_categories))
    if unexpected_categories:
        raise ValueError(
            f"완전격자 채움 대상에 예상 밖 {category_column}이 있습니다: {unexpected_categories}"
        )

    full_index = pd.MultiIndex.from_product(
        [expected_regions, expected_years, expected_categories],
        names=["지역", "연도", category_column],
    )
    filled = panel.set_index(["지역", "연도", category_column]).reindex(full_index).reset_index()
    filled["당해계획예산_백만원_provisional"] = filled["당해계획예산_백만원_provisional"].fillna(
        0.0
    )
    filled["사업수"] = filled["사업수"].fillna(0).astype(int)
    filled["예산결측_사업수"] = filled["예산결측_사업수"].fillna(0).astype(int)
    return filled.sort_values(["지역", "연도", category_column]).reset_index(drop=True)


def aggregate_labels_to_panels(
    labeled_long: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
    expected_major_labels: Sequence[str],
    expected_sub_labels: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate an explicitly labeled detail file to major and sub-area panels.

    The function never invents missing *labels*. But a 지역·연도·카테고리
    combination with zero detail rows (e.g. a region funded nothing in a
    category that year) is filled with 사업수=0·예산=0 so the 765/1,836-row
    full grid holds — omitting the row would be less honest than an explicit,
    mathematically correct zero (sum/count over an empty set).
    """

    required = {"지역", "연도", "세부사업명", "대영역", "세부영역"}
    missing = sorted(required - set(labeled_long.columns))
    if missing:
        raise KeyError(f"라벨 입력 필수 컬럼 누락: {missing}")
    budget_column = next(
        (
            column
            for column in ("당해계획예산_백만원", "예산액", "예산액_숫자")
            if column in labeled_long.columns
        ),
        None,
    )
    if budget_column is None:
        raise KeyError("라벨 입력에 예산 컬럼이 없습니다.")

    frame = labeled_long.copy()
    frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype(int)
    numeric = pd.to_numeric(frame[budget_column], errors="coerce")
    invalid = frame[budget_column].notna() & numeric.isna()
    if invalid.any():
        raise ValueError("라벨 입력 예산에 비수치 값이 있습니다.")
    frame["_예산액"] = numeric
    if frame[["대영역", "세부영역"]].isna().any().any():
        raise ValueError("라벨 입력에 대영역 또는 세부영역 결측이 있습니다.")

    major = (
        frame.groupby(["지역", "연도", "대영역"], as_index=False, dropna=False)
        .agg(
            당해계획예산_백만원_provisional=("_예산액", lambda values: values.sum(min_count=1)),
            사업수=("세부사업명", "size"),
            예산결측_사업수=("_예산액", lambda values: int(values.isna().sum())),
        )
        .sort_values(["지역", "연도", "대영역"])
        .reset_index(drop=True)
    )
    sub = (
        frame.groupby(["지역", "연도", "세부영역"], as_index=False, dropna=False)
        .agg(
            당해계획예산_백만원_provisional=("_예산액", lambda values: values.sum(min_count=1)),
            사업수=("세부사업명", "size"),
            예산결측_사업수=("_예산액", lambda values: int(values.isna().sum())),
        )
        .sort_values(["지역", "연도", "세부영역"])
        .reset_index(drop=True)
    )
    major = _fill_empty_combinations(
        major,
        category_column="대영역",
        expected_regions=expected_regions,
        expected_years=expected_years,
        expected_categories=expected_major_labels,
    )
    sub = _fill_empty_combinations(
        sub,
        category_column="세부영역",
        expected_regions=expected_regions,
        expected_years=expected_years,
        expected_categories=expected_sub_labels,
    )
    validate_full_grid(
        major,
        expected_regions,
        expected_years,
        category_column="대영역",
        expected_categories=expected_major_labels,
    )
    validate_full_grid(
        sub,
        expected_regions,
        expected_years,
        category_column="세부영역",
        expected_categories=expected_sub_labels,
    )
    return major, sub
