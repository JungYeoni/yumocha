import pandas as pd
import pytest

from scripts.build_subarea_fiscal_index_tfr_regression_sample import (
    INDEX_COLUMN,
    LAG1_COLUMN,
    LAG2_COLUMN,
    build_index_lags,
    build_sample,
)


def test_build_index_lags_creates_one_and_two_year_lags():
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": year, "세부영역": "A", INDEX_COLUMN: float(year)}
            for year in range(2016, 2020)
        ]
    )
    result = build_index_lags(panel)
    assert result[LAG1_COLUMN].isna().sum() == 1
    assert result[LAG2_COLUMN].isna().sum() == 2
    assert result.loc[result["연도"].eq(2018), LAG2_COLUMN].iloc[0] == 2016


def test_build_index_lags_rejects_year_gap():
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": year, "세부영역": "A", INDEX_COLUMN: 1.0}
            for year in [2016, 2018]
        ]
    )
    with pytest.raises(ValueError, match="연도 공백"):
        build_index_lags(panel)


def test_build_sample_preserves_tfr_and_lags():
    index = pd.DataFrame(
        [
            {"지역": "서울", "연도": year, "세부영역": "A", INDEX_COLUMN: float(year)}
            for year in range(2016, 2019)
        ]
    )
    fiscal = pd.DataFrame(
        [
            {"지역": "서울", "연도": year, "세부영역": "A", "합계출산율": 1.0}
            for year in range(2016, 2019)
        ]
    )
    result = build_sample(index, fiscal)
    assert len(result) == 3
    assert result["합계출산율"].eq(1.0).all()
    assert result.loc[result["연도"].eq(2018), LAG1_COLUMN].iloc[0] == 2017


def test_build_sample_rejects_fiscal_rows_missing_from_index_panel():
    index = pd.DataFrame(
        [
            {"지역": "서울", "연도": year, "세부영역": "A", INDEX_COLUMN: float(year)}
            for year in range(2016, 2019)
        ]
    )
    fiscal = pd.DataFrame(
        [
            {"지역": "서울", "연도": year, "세부영역": "A", "합계출산율": 1.0}
            for year in range(2016, 2019)
        ]
        + [{"지역": "부산", "연도": 2018, "세부영역": "A", "합계출산율": 1.0}]
    )

    with pytest.raises(ValueError, match="재정대응지수 패널에 없는"):
        build_sample(index, fiscal)
