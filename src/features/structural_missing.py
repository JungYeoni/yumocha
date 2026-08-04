from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from enum import Enum

import pandas as pd


class MissingType(str, Enum):
    OBSERVED = "실측"
    INTERMEDIATE = "중간 결측"
    LEADING = "선행 결측"
    TRAILING = "후행 결측"
    SINGLE_YEAR = "단년도 관측"
    ALL_MISSING = "전기간 미관측"
    STRUCTURAL = "원자료 자체 결측"


class ProcessingStrategy(str, Enum):
    NONE = "none"
    LINEAR_INTERPOLATION = "linear interpolation"
    HOLD_FIRST_OBSERVED = "hold first observed"
    HOLD_LAST_OBSERVED = "hold last observed"
    EXCLUDE_SINGLE_YEAR_SERIES = "exclude single-year series"
    EXCLUDE_ANALYSIS_PERIOD = "exclude analysis period"
    TREND_EXTRAPOLATION = "trend-based extrapolation"
    STRUCTURAL_MISSING = "preserve original NA"


class ExtrapolationStrategy(str, Enum):
    HOLD = "hold"
    TREND = "trend"
    EXCLUDE = "exclude"


DEFAULT_OUTPUT_COLUMNS = {
    "processed_value": "processed_value",
    "missing_type": "missing_type",
    "processing_strategy": "processing_strategy",
    "is_observed": "is_observed",
    "is_imputed": "is_imputed",
    "include_in_analysis": "include_in_analysis",
    "is_structural_missing": "is_structural_missing",
}


def _require_columns(df: pd.DataFrame, required: Iterable[str], source_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{source_name}: 필수 컬럼 누락 {missing} (실제 컬럼: {list(df.columns)})")


def _validate_panel_uniqueness(
    df: pd.DataFrame,
    region_col: str,
    indicator_col: str,
    year_col: str,
) -> None:
    duplicated = df.duplicated([region_col, indicator_col, year_col], keep=False)
    if duplicated.any():
        duplicate_keys = (
            df.loc[duplicated, [region_col, indicator_col, year_col]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"중복된 지역·지표·연도 입력이 있습니다: {duplicate_keys}")


def _materialize_output_names(output_column_map: dict[str, str] | None) -> dict[str, str]:
    result = DEFAULT_OUTPUT_COLUMNS.copy()
    if output_column_map:
        result.update(output_column_map)
    return result


def _interpolate_segment(
    series: pd.Series,
) -> pd.Series:
    return series.interpolate(method="linear", limit_area="inside")


def _validate_trend_extrapolation_series(
    result: pd.Series,
    expected_index: pd.Index,
    source_name: str,
) -> pd.Series:
    if not isinstance(result, pd.Series):
        raise TypeError(
            f"{source_name}: trend_extrapolation_func는 pandas.Series를 반환해야 합니다."
        )
    if not result.index.equals(expected_index):
        raise ValueError(
            f"{source_name}: 반환된 시리즈의 인덱스가 예상 연도 인덱스와 일치하지 않습니다."
        )
    if len(result) != len(expected_index):
        raise ValueError(
            f"{source_name}: 반환된 시리즈 길이({len(result)})가 대상 연도 수({len(expected_index)})와 일치하지 않습니다."
        )
    numeric = pd.to_numeric(result, errors="raise")
    if numeric.isna().any():
        raise ValueError(f"{source_name}: 반환값에 NA가 포함되어서는 안 됩니다.")
    return numeric


def _build_processing_strategy_label(
    strategy: ExtrapolationStrategy,
    axis: str,
) -> str:
    if strategy == ExtrapolationStrategy.HOLD:
        return (
            ProcessingStrategy.HOLD_FIRST_OBSERVED.value
            if axis == "leading"
            else ProcessingStrategy.HOLD_LAST_OBSERVED.value
        )
    if strategy == ExtrapolationStrategy.TREND:
        return ProcessingStrategy.TREND_EXTRAPOLATION.value
    return ProcessingStrategy.EXCLUDE_ANALYSIS_PERIOD.value


def _coerce_structural_missing_flag(series: pd.Series, source_name: str) -> pd.Series:
    truthy = {True, "True", "true", "TRUE", 1, "1", "Y", "y", "yes", "Yes", "YES"}
    falsy = {False, "False", "false", "FALSE", 0, "0", "N", "n", "no", "No", "NO"}
    coerced = []
    for value in series:
        if pd.isna(value):
            coerced.append(False)
            continue
        if value in truthy:
            coerced.append(True)
            continue
        if value in falsy:
            coerced.append(False)
            continue
        raise ValueError(
            f"{source_name}: structural_missing_col은 boolean, 0/1, 또는 'True'/'False' 형태여야 합니다. 잘못된 값: {value!r}"
        )
    return pd.Series(coerced, index=series.index, dtype="boolean")


def process_structural_indicator_panel(
    df: pd.DataFrame,
    *,
    region_col: str = "지역",
    indicator_col: str = "세부지표",
    year_col: str = "연도",
    value_col: str = "측정값",
    structural_missing_col: str | None = None,
    expected_years: Sequence[int] | None = None,
    leading_strategy: ExtrapolationStrategy = ExtrapolationStrategy.HOLD,
    trailing_strategy: ExtrapolationStrategy = ExtrapolationStrategy.HOLD,
    trend_extrapolation_func: Callable[[pd.Series, pd.Index], pd.Series] | None = None,
    output_column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """구조환경지표 지역·연도 패널의 결측 유형과 처리 결과를 생성한다.

    이 함수는 파일 I/O 없이 long 형식 판넬을 입력받아 다음을 모두 반환한다.

    - 원본 값을 보존하고 별도 처리값(`processed_value`)을 생성
    - 중간 결측은 선형 보간
    - 선행/후행 결측은 기본적으로 첫/마지막 관측값을 유지
    - 단년도 관측 계열은 분석에서 제외
    - 전기간 미관측은 그대로 NA로 유지
    - 원자료 자체 결측은 `structural_missing_col` 플래그가 있는 경우 특별히 보존

    `structural_missing_col`가 없으면 원자료 자체 결측을 구분할 근거가 없으므로
    해당 플래그가 있는 행만 구조적 결측으로 취급합니다.

    이 함수는 long 형식의 지역×지표×연도 패널을 전제로 하며, 누락된 연도 행을
    자동 생성하지 않습니다. `expected_years`를 지정하면 각 지역·지표 그룹이 해당
    연도 목록을 모두 포함하는지 검증합니다.
    """

    output_names = _materialize_output_names(output_column_map)
    required_columns = [region_col, indicator_col, year_col, value_col]
    if structural_missing_col:
        required_columns.append(structural_missing_col)
    _require_columns(df, required_columns, source_name="process_structural_indicator_panel")
    _validate_panel_uniqueness(df, region_col, indicator_col, year_col)

    expected_years_set = None
    if expected_years is not None:
        expected_years_set = {int(year) for year in expected_years}
        if len(expected_years_set) != len(list(expected_years)):
            raise ValueError("expected_years에는 중복 없는 연도 목록을 전달해야 합니다.")

    result = df.copy()
    result[year_col] = pd.to_numeric(result[year_col], errors="raise").astype(int)

    values = pd.to_numeric(result[value_col], errors="raise")
    result[value_col] = values

    if structural_missing_col is not None:
        result[output_names["is_structural_missing"]] = _coerce_structural_missing_flag(
            result[structural_missing_col],
            source_name="process_structural_indicator_panel",
        )
    else:
        result[output_names["is_structural_missing"]] = False

    final_rows = []
    for _, group in result.groupby([indicator_col, region_col], sort=False):
        group_result = group.sort_values(year_col).copy()
        if expected_years_set is not None:
            group_years = set(group_result[year_col].astype(int).tolist())
            missing_years = sorted(expected_years_set - group_years)
            extra_years = sorted(group_years - expected_years_set)
            if missing_years or extra_years:
                raise ValueError(
                    f"{region_col}={group_result[region_col].iloc[0]}, "
                    f"{indicator_col}={group_result[indicator_col].iloc[0]}에 대한 연도 패널 불일치: "
                    f"누락={missing_years}, 추가={extra_years}"
                )

        observed = group_result[value_col].notna()
        group_result[output_names["is_observed"]] = observed
        structural_missing = (
            group_result[output_names["is_structural_missing"]]
            if structural_missing_col
            else pd.Series(False, index=group_result.index)
        )
        observed_count = int(observed.sum())

        group_result[output_names["is_observed"]] = observed
        group_result[output_names["is_imputed"]] = False
        group_result[output_names["include_in_analysis"]] = False
        group_result[output_names["missing_type"]] = ""
        group_result[output_names["processing_strategy"]] = ProcessingStrategy.NONE.value
        group_result[output_names["processed_value"]] = group_result[value_col]

        if observed_count == 0:
            group_result[output_names["missing_type"]] = MissingType.ALL_MISSING.value
            group_result[output_names["processing_strategy"]] = ProcessingStrategy.NONE.value
            group_result[output_names["include_in_analysis"]] = False
            final_rows.append(group_result)
            continue

        nonstructural_missing = ~observed & ~structural_missing
        structural_mask = ~observed & structural_missing

        if observed_count == 1:
            group_result.loc[observed, output_names["missing_type"]] = MissingType.SINGLE_YEAR.value
            group_result.loc[observed, output_names["processing_strategy"]] = (
                ProcessingStrategy.EXCLUDE_SINGLE_YEAR_SERIES.value
            )
            group_result.loc[observed, output_names["include_in_analysis"]] = False

            first_observed_year = int(group_result.loc[observed, year_col].iloc[0])
            leading_mask = nonstructural_missing & group_result[year_col].lt(first_observed_year)
            trailing_mask = nonstructural_missing & group_result[year_col].gt(first_observed_year)
            interior_mask = nonstructural_missing & ~(leading_mask | trailing_mask)

            group_result.loc[leading_mask, output_names["missing_type"]] = MissingType.LEADING.value
            group_result.loc[trailing_mask, output_names["missing_type"]] = (
                MissingType.TRAILING.value
            )
            group_result.loc[interior_mask, output_names["missing_type"]] = (
                MissingType.INTERMEDIATE.value
            )
            group_result.loc[structural_mask, output_names["missing_type"]] = (
                MissingType.STRUCTURAL.value
            )

            group_result.loc[leading_mask, output_names["processing_strategy"]] = (
                ProcessingStrategy.EXCLUDE_ANALYSIS_PERIOD.value
            )
            group_result.loc[trailing_mask, output_names["processing_strategy"]] = (
                ProcessingStrategy.EXCLUDE_ANALYSIS_PERIOD.value
            )
            group_result.loc[interior_mask, output_names["processing_strategy"]] = (
                ProcessingStrategy.EXCLUDE_ANALYSIS_PERIOD.value
            )
            group_result.loc[structural_mask, output_names["processing_strategy"]] = (
                ProcessingStrategy.STRUCTURAL_MISSING.value
            )

            group_result.loc[leading_mask, output_names["include_in_analysis"]] = False
            group_result.loc[trailing_mask, output_names["include_in_analysis"]] = False
            group_result.loc[interior_mask, output_names["include_in_analysis"]] = False
            group_result.loc[structural_mask, output_names["include_in_analysis"]] = False

            final_rows.append(group_result)
            continue

        first_observed_year = int(group_result.loc[observed, year_col].min())
        last_observed_year = int(group_result.loc[observed, year_col].max())

        group_result.loc[observed, output_names["missing_type"]] = MissingType.OBSERVED.value
        group_result.loc[observed, output_names["processing_strategy"]] = (
            ProcessingStrategy.NONE.value
        )
        group_result.loc[observed, output_names["include_in_analysis"]] = True

        leading_mask = nonstructural_missing & group_result[year_col].lt(first_observed_year)
        trailing_mask = nonstructural_missing & group_result[year_col].gt(last_observed_year)
        interior_mask = nonstructural_missing & ~(leading_mask | trailing_mask)

        group_result.loc[leading_mask, output_names["missing_type"]] = MissingType.LEADING.value
        group_result.loc[trailing_mask, output_names["missing_type"]] = MissingType.TRAILING.value
        group_result.loc[interior_mask, output_names["missing_type"]] = (
            MissingType.INTERMEDIATE.value
        )
        group_result.loc[structural_mask, output_names["missing_type"]] = (
            MissingType.STRUCTURAL.value
        )

        group_result.loc[leading_mask, output_names["processing_strategy"]] = (
            _build_processing_strategy_label(leading_strategy, "leading")
        )
        group_result.loc[trailing_mask, output_names["processing_strategy"]] = (
            _build_processing_strategy_label(trailing_strategy, "trailing")
        )
        group_result.loc[interior_mask, output_names["processing_strategy"]] = (
            ProcessingStrategy.LINEAR_INTERPOLATION.value
        )
        group_result.loc[structural_mask, output_names["processing_strategy"]] = (
            ProcessingStrategy.STRUCTURAL_MISSING.value
        )

        group_result.loc[interior_mask, output_names["include_in_analysis"]] = True
        group_result.loc[leading_mask, output_names["include_in_analysis"]] = (
            leading_strategy != ExtrapolationStrategy.EXCLUDE
        )
        group_result.loc[trailing_mask, output_names["include_in_analysis"]] = (
            trailing_strategy != ExtrapolationStrategy.EXCLUDE
        )
        group_result.loc[structural_mask, output_names["include_in_analysis"]] = False

        processed = group_result[value_col].copy()
        boundary = structural_missing | structural_missing.shift(-1, fill_value=False)
        segment_id = boundary.cumsum()
        for _, segment in group_result.groupby(segment_id, sort=False):
            if segment[output_names["is_structural_missing"]].all():
                continue
            segment_target = segment[value_col].where(
                ~segment[output_names["is_structural_missing"]]
            )
            processed.loc[segment_target.index] = _interpolate_segment(segment_target)

        if leading_strategy == ExtrapolationStrategy.HOLD:
            first_value = group_result.loc[
                group_result[year_col] == first_observed_year, value_col
            ].iloc[0]
            processed.loc[leading_mask] = first_value
        elif leading_strategy == ExtrapolationStrategy.TREND:
            if trend_extrapolation_func is None:
                raise NotImplementedError(
                    "선행 결측의 추세 기반 역외삽을 적용하려면 trend_extrapolation_func를 제공해야 합니다."
                )
            reference_series = group_result.loc[observed, value_col].copy()
            reference_series.index = group_result.loc[observed, year_col].astype(int)
            target_years = pd.Index(group_result.loc[leading_mask, year_col].astype(int))
            processed.loc[leading_mask] = _validate_trend_extrapolation_series(
                trend_extrapolation_func(reference_series, target_years),
                target_years,
                "leading",
            ).to_numpy()

        if trailing_strategy == ExtrapolationStrategy.HOLD:
            last_value = group_result.loc[
                group_result[year_col] == last_observed_year, value_col
            ].iloc[0]
            processed.loc[trailing_mask] = last_value
        elif trailing_strategy == ExtrapolationStrategy.TREND:
            if trend_extrapolation_func is None:
                raise NotImplementedError(
                    "후행 결측의 추세 기반 역외삽을 적용하려면 trend_extrapolation_func를 제공해야 합니다."
                )
            reference_series = group_result.loc[observed, value_col].copy()
            reference_series.index = group_result.loc[observed, year_col].astype(int)
            target_years = pd.Index(group_result.loc[trailing_mask, year_col].astype(int))
            processed.loc[trailing_mask] = _validate_trend_extrapolation_series(
                trend_extrapolation_func(reference_series, target_years),
                target_years,
                "trailing",
            ).to_numpy()

        group_result[output_names["processed_value"]] = processed
        imputed_mask = group_result[output_names["processed_value"]].notna() & ~observed
        group_result.loc[imputed_mask, output_names["is_imputed"]] = True

        final_rows.append(group_result)

    final = pd.concat(final_rows, axis=0, ignore_index=True)
    return final.sort_values([indicator_col, region_col, year_col]).reset_index(drop=True)
