"""구조환경지표 검증 공통 유틸(src.evaluation.structural_validation) 단위 테스트."""

import pandas as pd
import pytest

from src.evaluation.structural_validation import (
    compare_region_year_matrices,
    require_columns,
    require_sheets,
    to_numeric_strict,
    to_verification_record,
    weighted_response_mean,
)


def test_require_columns_passes_when_all_present():
    df = pd.DataFrame({"지역": ["전국"], "연령별": ["0세"]})
    require_columns(df, ["지역", "연령별"], source_name="테스트 시트")


def test_require_columns_raises_with_missing_column_names():
    df = pd.DataFrame({"지역": ["전국"]})
    with pytest.raises(KeyError, match="연령별"):
        require_columns(df, ["지역", "연령별"], source_name="테스트 시트")


def test_require_sheets_raises_with_missing_sheet_names():
    with pytest.raises(KeyError, match="계산 시트"):
        require_sheets(["결과 시트"], ["계산 시트", "결과 시트"], source_name="테스트 파일")


def test_to_numeric_strict_treats_allowed_token_as_missing():
    series = pd.Series(["10", "-", "20"])
    result = to_numeric_strict(series, allowed_missing_tokens=frozenset({"-"}))
    assert result.tolist()[0] == 10
    assert pd.isna(result.tolist()[1])
    assert result.tolist()[2] == 20


def test_to_numeric_strict_raises_on_unexpected_non_numeric_value():
    series = pd.Series(["10", "오류값", "20"])
    with pytest.raises(ValueError, match="오류값"):
        to_numeric_strict(series)


def test_to_numeric_strict_keeps_native_nan_as_missing_without_error():
    series = pd.Series([10.0, None, 20.0])
    result = to_numeric_strict(series)
    assert result.tolist()[0] == 10.0
    assert pd.isna(result.tolist()[1])


def test_compare_region_year_matrices_passes_when_identical():
    left = pd.DataFrame({2016: [1.0, 2.0], 2017: [3.0, 4.0]}, index=["전국", "서울"])
    right = left.copy()

    result = compare_region_year_matrices(
        left,
        right,
        expected_regions=["전국", "서울"],
        expected_years=[2016, 2017],
        label="테스트",
    )

    assert bool(result) is True
    assert result.comparison_count == 4
    assert result.max_abs_diff == 0.0
    assert result.mismatch_count == 0
    assert result.missing_combinations.empty


def test_compare_region_year_matrices_detects_missing_region_instead_of_reporting_zero_error():
    # 서울이 right에서 통째로 빠진 경우 - reindex 후 max()만 봤다면 놓쳤을 상황
    left = pd.DataFrame({2016: [1.0, 2.0]}, index=["전국", "서울"])
    right = pd.DataFrame({2016: [1.0]}, index=["전국"])

    result = compare_region_year_matrices(
        left,
        right,
        expected_regions=["전국", "서울"],
        expected_years=[2016],
        label="테스트",
    )

    assert bool(result) is False
    assert len(result.missing_combinations) == 1
    assert result.missing_combinations.iloc[0]["지역"] == "서울"
    assert not result.missing_combinations.iloc[0]["right_있음"]
    # 실제로 비교 가능했던 값(전국)은 일치했으므로 그 부분 오차는 0
    assert result.max_abs_diff == 0.0


def test_compare_region_year_matrices_reports_max_diff_location():
    left = pd.DataFrame({2016: [1.0, 2.0], 2017: [3.0, 100.0]}, index=["전국", "서울"])
    right = pd.DataFrame({2016: [1.0, 2.0], 2017: [3.0, 4.0]}, index=["전국", "서울"])

    result = compare_region_year_matrices(
        left,
        right,
        expected_regions=["전국", "서울"],
        expected_years=[2016, 2017],
        label="테스트",
    )

    assert bool(result) is False
    assert result.mismatch_count == 1
    assert result.max_diff_location == ("서울", 2017)
    assert result.max_abs_diff == pytest.approx(96.0)
    assert len(result.detail) == 1


def test_compare_region_year_matrices_raises_on_duplicate_region_index():
    left = pd.DataFrame({2016: [1.0, 2.0]}, index=["전국", "전국"])
    right = pd.DataFrame({2016: [1.0]}, index=["전국"])

    with pytest.raises(ValueError, match="중복"):
        compare_region_year_matrices(
            left,
            right,
            expected_regions=["전국"],
            expected_years=[2016],
            label="테스트",
        )


def test_compare_region_year_matrices_respects_tolerance():
    left = pd.DataFrame({2016: [1.0001]}, index=["전국"])
    right = pd.DataFrame({2016: [1.0]}, index=["전국"])

    strict = compare_region_year_matrices(
        left, right, expected_regions=["전국"], expected_years=[2016], label="strict"
    )
    lenient = compare_region_year_matrices(
        left,
        right,
        expected_regions=["전국"],
        expected_years=[2016],
        tolerance=0.001,
        label="lenient",
    )

    assert bool(strict) is False
    assert bool(lenient) is True


def test_to_verification_record_summarizes_comparison_result():
    left = pd.DataFrame({2016: [1.0]}, index=["전국"])
    right = pd.DataFrame({2016: [1.0]}, index=["전국"])
    result = compare_region_year_matrices(
        left, right, expected_regions=["전국"], expected_years=[2016], label="테스트"
    )

    record = to_verification_record(
        result, indicator_id="youth_employment_rate", stage="결과 시트 대조"
    )

    assert record["지표ID"] == "youth_employment_rate"
    assert record["검증단계"] == "결과 시트 대조"
    assert record["비교건수"] == 1
    assert record["불일치건수"] == 0
    assert record["판정"] == "정상"


def test_weighted_response_mean_divides_by_actual_category_sum_not_100():
    # 응답 비율 합이 반올림 때문에 100이 아닌 경우(99.9)
    df = pd.DataFrame(
        {
            "매우만족": [1.1],
            "약간만족": [12.1],
            "보통": [41.2],
            "약간불만족": [36.1],
            "매우불만족": [9.4],
        },
        index=["전국"],
    )
    scores = {"매우만족": 5, "약간만족": 4, "보통": 3, "약간불만족": 2, "매우불만족": 1}

    result = weighted_response_mean(df, scores=scores, expected_regions=["전국"])

    # (1.1*5+12.1*4+41.2*3+36.1*2+9.4*1) / 99.9
    assert result.loc["전국"] == pytest.approx(2.593594, abs=1e-6)


def test_weighted_response_mean_raises_on_missing_region():
    df = pd.DataFrame({"만족": [1.0], "불만족": [1.0]}, index=["전국"])
    with pytest.raises(ValueError, match="서울"):
        weighted_response_mean(
            df, scores={"만족": 1, "불만족": 0}, expected_regions=["전국", "서울"]
        )
