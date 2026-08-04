import pandas as pd
import pytest

from src.features.structural_missing import (
    ExtrapolationStrategy,
    MissingType,
    ProcessingStrategy,
    process_structural_indicator_panel,
)


def test_observed_values_are_preserved():
    df = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "세부지표": ["지표", "지표"],
            "연도": [2020, 2021],
            "측정값": [1.0, 2.0],
        }
    )

    result = process_structural_indicator_panel(df)

    assert result.loc[result["연도"].eq(2020), "processed_value"].iloc[0] == 1.0
    assert result.loc[result["연도"].eq(2021), "processed_value"].iloc[0] == 2.0
    assert (result["missing_type"] == MissingType.OBSERVED.value).all()
    assert (result["is_imputed"] == False).all()
    assert (result["include_in_analysis"] == True).all()


def test_intermediate_missing_linear_interpolation():
    df = pd.DataFrame(
        {
            "지역": ["부산"] * 4,
            "세부지표": ["지표"] * 4,
            "연도": [2019, 2020, 2021, 2022],
            "측정값": [1.0, None, None, 4.0],
        }
    )

    result = process_structural_indicator_panel(df)
    interpolated = result.loc[result["연도"].isin([2020, 2021]), "processed_value"].tolist()

    assert interpolated == [2.0, 3.0]
    assert (result.loc[result["연도"].isin([2020, 2021]), "missing_type"] == MissingType.INTERMEDIATE.value).all()
    assert (result.loc[result["연도"].isin([2020, 2021]), "processing_strategy"] == ProcessingStrategy.LINEAR_INTERPOLATION.value).all()


def test_leading_missing_holds_first_observed():
    df = pd.DataFrame(
        {
            "지역": ["대구"] * 4,
            "세부지표": ["지표"] * 4,
            "연도": [2017, 2018, 2019, 2020],
            "측정값": [None, 5.0, 6.0, None],
        }
    )

    result = process_structural_indicator_panel(df)

    assert result.loc[result["연도"].eq(2017), "processed_value"].iloc[0] == 5.0
    assert result.loc[result["연도"].eq(2020), "processed_value"].iloc[0] == 6.0
    assert result.loc[result["연도"].eq(2017), "missing_type"].iloc[0] == MissingType.LEADING.value
    assert result.loc[result["연도"].eq(2017), "processing_strategy"].iloc[0] == ProcessingStrategy.HOLD_FIRST_OBSERVED.value


def test_leading_missing_trend_extrapolation_requires_function():
    df = pd.DataFrame(
        {
            "지역": ["부산"] * 4,
            "세부지표": ["지표"] * 4,
            "연도": [2017, 2018, 2019, 2020],
            "측정값": [None, 4.0, 6.0, None],
        }
    )

    with pytest.raises(NotImplementedError, match="trend_extrapolation_func"):
        process_structural_indicator_panel(
            df,
            leading_strategy=ExtrapolationStrategy.TREND,
        )


def test_trailing_missing_holds_last_observed():
    df = pd.DataFrame(
        {
            "지역": ["경기"] * 4,
            "세부지표": ["지표"] * 4,
            "연도": [2017, 2018, 2019, 2020],
            "측정값": [2.0, 3.0, None, None],
        }
    )

    result = process_structural_indicator_panel(df)

    assert result.loc[result["연도"].eq(2019), "processed_value"].iloc[0] == 3.0
    assert result.loc[result["연도"].eq(2020), "processed_value"].iloc[0] == 3.0
    assert (result.loc[result["연도"].isin([2019, 2020]), "missing_type"] == MissingType.TRAILING.value).all()
    assert (result.loc[result["연도"].isin([2019, 2020]), "processing_strategy"] == ProcessingStrategy.HOLD_LAST_OBSERVED.value).all()


def test_trailing_missing_exclude_analysis_period():
    df = pd.DataFrame(
        {
            "지역": ["경기"] * 4,
            "세부지표": ["지표"] * 4,
            "연도": [2017, 2018, 2019, 2020],
            "측정값": [2.0, 3.0, None, None],
        }
    )

    result = process_structural_indicator_panel(
        df,
        trailing_strategy=ExtrapolationStrategy.EXCLUDE,
    )

    assert (result.loc[result["연도"].isin([2019, 2020]), "include_in_analysis"] == False).all()
    assert (result.loc[result["연도"].isin([2019, 2020]), "processing_strategy"] == ProcessingStrategy.EXCLUDE_ANALYSIS_PERIOD.value).all()


def test_single_year_series_excluded():
    df = pd.DataFrame(
        {
            "지역": ["울산"],
            "세부지표": ["지표"],
            "연도": [2020],
            "측정값": [3.5],
        }
    )

    result = process_structural_indicator_panel(df)

    assert result["missing_type"].iloc[0] == MissingType.SINGLE_YEAR.value
    assert result["include_in_analysis"].iloc[0] == False
    assert result["processed_value"].iloc[0] == 3.5


def test_all_missing_series_preserved():
    df = pd.DataFrame(
        {
            "지역": ["세종"] * 2,
            "세부지표": ["지표"] * 2,
            "연도": [2019, 2020],
            "측정값": [None, None],
        }
    )

    result = process_structural_indicator_panel(df)

    assert result["missing_type"].eq(MissingType.ALL_MISSING.value).all()
    assert result["processed_value"].isna().all()
    assert result["include_in_analysis"].eq(False).all()


def test_structural_missing_preserved_with_flag():
    df = pd.DataFrame(
        {
            "지역": ["강원", "강원"],
            "세부지표": ["지표", "지표"],
            "연도": [2019, 2020],
            "측정값": [None, 5.0],
            "원자료_자체결측": [True, False],
        }
    )

    result = process_structural_indicator_panel(
        df,
        structural_missing_col="원자료_자체결측",
    )

    assert result.loc[result["연도"].eq(2019), "missing_type"].iloc[0] == MissingType.STRUCTURAL.value
    assert pd.isna(result.loc[result["연도"].eq(2019), "processed_value"]).iloc[0]
    assert result.loc[result["연도"].eq(2019), "include_in_analysis"].iloc[0] == False


def test_processing_isolated_across_regions_and_indicators():
    df = pd.DataFrame(
        {
            "지역": ["서울", "서울", "부산", "부산"],
            "세부지표": ["지표A", "지표A", "지표A", "지표B"],
            "연도": [2019, 2020, 2019, 2020],
            "측정값": [1.0, None, None, 2.0],
        }
    )

    result = process_structural_indicator_panel(df)

    seoul_2020 = result.query("지역 == '서울' and 세부지표 == '지표A' and 연도 == 2020").iloc[0]
    busan_2019 = result.query("지역 == '부산' and 세부지표 == '지표A' and 연도 == 2019").iloc[0]

    assert pd.isna(seoul_2020["processed_value"])
    assert seoul_2020["include_in_analysis"] == False
    assert pd.isna(busan_2019["processed_value"])
    assert busan_2019["include_in_analysis"] == False


def test_unsorted_input_is_sorted_by_year():
    df = pd.DataFrame(
        {
            "지역": ["대전"] * 3,
            "세부지표": ["지표"] * 3,
            "연도": [2020, 2018, 2019],
            "측정값": [None, 1.0, None],
        }
    )

    result = process_structural_indicator_panel(df)

    assert result["연도"].tolist() == [2018, 2019, 2020]
    assert result.loc[result["연도"].eq(2018), "processed_value"].iloc[0] == 1.0


def test_duplicate_region_indicator_year_raises():
    df = pd.DataFrame(
        {
            "지역": ["대전", "대전"],
            "세부지표": ["지표", "지표"],
            "연도": [2020, 2020],
            "측정값": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="중복된 지역·지표·연도 입력"):
        process_structural_indicator_panel(df)


def test_expected_years_requires_complete_panel():
    df = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "세부지표": ["지표", "지표"],
            "연도": [2019, 2021],
            "측정값": [1.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="연도 패널 불일치"):
        process_structural_indicator_panel(df, expected_years=[2019, 2020, 2021])


def test_structural_missing_flag_coercion_and_invalid_value():
    df = pd.DataFrame(
        {
            "지역": ["강원", "강원"],
            "세부지표": ["지표", "지표"],
            "연도": [2019, 2020],
            "측정값": [None, 5.0],
            "원자료_자체결측": ["True", "False"],
        }
    )

    result = process_structural_indicator_panel(
        df,
        structural_missing_col="원자료_자체결측",
    )

    assert result.loc[result["연도"].eq(2019), "missing_type"].iloc[0] == MissingType.STRUCTURAL.value
    assert result.loc[result["연도"].eq(2019), "include_in_analysis"].iloc[0] == False

    df_invalid = df.copy()
    df_invalid.loc[0, "원자료_자체결측"] = "unknown"
    with pytest.raises(ValueError, match="structural_missing_col은 boolean"):
        process_structural_indicator_panel(
            df_invalid,
            structural_missing_col="원자료_자체결측",
        )


def test_trend_extrapolation_callback_contract():
    df = pd.DataFrame(
        {
            "지역": ["광주"] * 4,
            "세부지표": ["지표"] * 4,
            "연도": [2017, 2018, 2019, 2020],
            "측정값": [None, 2.0, 3.0, None],
        }
    )

    def invalid_callback(series, target_index):
        return pd.Series([1.0], index=[999])

    with pytest.raises(ValueError, match="인덱스가 예상 연도 인덱스와 일치하지 않습니다"):
        process_structural_indicator_panel(
            df,
            leading_strategy=ExtrapolationStrategy.TREND,
            trend_extrapolation_func=invalid_callback,
        )

    def callback_with_na(series, target_index):
        return pd.Series([None], index=target_index)

    with pytest.raises(ValueError, match="NA가 포함되어서는 안 됩니다"):
        process_structural_indicator_panel(
            df,
            leading_strategy=ExtrapolationStrategy.TREND,
            trend_extrapolation_func=callback_with_na,
        )


def test_isolated_region_indicator_paneling():
    df = pd.DataFrame(
        {
            "지역": ["서울", "서울", "부산", "부산"],
            "세부지표": ["지표A", "지표A", "지표A", "지표B"],
            "연도": [2019, 2020, 2019, 2020],
            "측정값": [1.0, None, None, 2.0],
        }
    )

    result = process_structural_indicator_panel(df)

    seoul_2020 = result.query("지역 == '서울' and 세부지표 == '지표A' and 연도 == 2020").iloc[0]
    busan_2019 = result.query("지역 == '부산' and 세부지표 == '지표A' and 연도 == 2019").iloc[0]
    busan_2020 = result.query("지역 == '부산' and 세부지표 == '지표B' and 연도 == 2020").iloc[0]

    assert pd.isna(seoul_2020["processed_value"])
    assert seoul_2020["include_in_analysis"] == False
    assert pd.isna(busan_2019["processed_value"])
    assert busan_2019["include_in_analysis"] == False
    assert busan_2020["processed_value"] == 2.0
    assert busan_2020["include_in_analysis"] == False
