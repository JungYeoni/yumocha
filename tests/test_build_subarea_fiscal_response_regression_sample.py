import pandas as pd

import pytest

from scripts.build_subarea_fiscal_response_regression_sample import (
    FISCAL_TO_STRUCTURAL_SUBCATEGORY,
    build_lagged_fiscal_response,
    build_lagged_structural_index,
    build_regression_sample,
)


def _structural_scores() -> pd.DataFrame:
    rows = []
    for subcategory in FISCAL_TO_STRUCTURAL_SUBCATEGORY.values():
        for year, score in ((2016, 10.0), (2017, 20.0), (2018, 30.0)):
            rows.append(
                {
                    "region": "A",
                    "year": year,
                    "category": "cat",
                    "subcategory": subcategory,
                    "subcategory_score": score,
                }
            )
    return pd.DataFrame(rows)


def test_build_lagged_structural_index_renames_and_shifts_by_one_year():
    result = build_lagged_structural_index(_structural_scores())

    row_2017 = result.loc[
        result["지역"].eq("A") & result["연도"].eq(2017) & result["세부영역"].eq("1-1. 고용여건")
    ].iloc[0]
    row_2016 = result.loc[
        result["지역"].eq("A") & result["연도"].eq(2016) & result["세부영역"].eq("1-1. 고용여건")
    ].iloc[0]
    assert row_2017["구조환경지수_전년도"] == pytest.approx(10.0)
    assert pd.isna(row_2016["구조환경지수_전년도"])


def test_build_lagged_structural_index_rejects_unexpected_subcategory_labels():
    scores = _structural_scores()
    scores.loc[0, "subcategory"] = "존재하지않는영역"

    with pytest.raises(ValueError, match="매핑표와 다릅니다"):
        build_lagged_structural_index(scores)


def _fiscal_response_single_subarea(subarea: str = "1-1. 고용여건") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "지역": ["A", "A", "A"],
            "연도": [2016, 2017, 2018],
            "세부영역": [subarea] * 3,
            "인구1인당_실질예산_원": [100.0, 200.0, 300.0],
        }
    )


def test_build_lagged_fiscal_response_shifts_by_one_and_two_years():
    result = build_lagged_fiscal_response(_fiscal_response_single_subarea())

    row_2018 = result.loc[result["연도"].eq(2018)].iloc[0]
    row_2017 = result.loc[result["연도"].eq(2017)].iloc[0]
    row_2016 = result.loc[result["연도"].eq(2016)].iloc[0]
    assert row_2018["인구1인당_실질예산_전년도"] == pytest.approx(200.0)
    assert row_2018["인구1인당_실질예산_전전년도"] == pytest.approx(100.0)
    assert row_2017["인구1인당_실질예산_전년도"] == pytest.approx(100.0)
    assert pd.isna(row_2017["인구1인당_실질예산_전전년도"])
    assert pd.isna(row_2016["인구1인당_실질예산_전년도"])


def test_build_lagged_fiscal_response_excludes_out_of_taxonomy():
    fiscal_response = pd.concat(
        [
            _fiscal_response_single_subarea(),
            _fiscal_response_single_subarea("지표체계 외"),
        ],
        ignore_index=True,
    )

    result = build_lagged_fiscal_response(fiscal_response)

    assert set(result["세부영역"]) == {"1-1. 고용여건"}


def test_build_lagged_fiscal_response_rejects_year_gaps():
    fiscal_response = _fiscal_response_single_subarea()
    fiscal_response = fiscal_response.loc[fiscal_response["연도"].ne(2017)]

    with pytest.raises(ValueError, match="연도가 연속하지 않는"):
        build_lagged_fiscal_response(fiscal_response)


def test_build_regression_sample_excludes_out_of_taxonomy_and_joins_on_key():
    fiscal_response = pd.DataFrame(
        {
            "지역": ["A", "A"],
            "연도": [2017, 2017],
            "세부영역": ["1-1. 고용여건", "지표체계 외"],
            "인구1인당_실질예산_원": [100.0, 999.0],
        }
    )
    fertility_lagged = pd.DataFrame(
        {
            "지역": ["A"],
            "연도": [2017],
            "직전1년_출산율하락도": [0.05],
            "합계출산율": [0.8],
        }
    )
    structural_lagged = pd.DataFrame(
        {
            "지역": ["A"],
            "연도": [2017],
            "세부영역": ["1-1. 고용여건"],
            "구조환경지수_전년도": [42.0],
        }
    )
    fiscal_lagged = pd.DataFrame(
        {
            "지역": ["A"],
            "연도": [2017],
            "세부영역": ["1-1. 고용여건"],
            "인구1인당_실질예산_전년도": [90.0],
            "인구1인당_실질예산_전전년도": [80.0],
        }
    )

    result = build_regression_sample(
        fiscal_response, fertility_lagged, structural_lagged, fiscal_lagged
    )

    assert len(result) == 1
    assert result.iloc[0]["세부영역"] == "1-1. 고용여건"
    assert result.iloc[0]["구조환경지수_전년도"] == pytest.approx(42.0)
    assert result.iloc[0]["직전1년_출산율하락도"] == pytest.approx(0.05)
    assert result.iloc[0]["인구1인당_실질예산_전년도"] == pytest.approx(90.0)
    assert result.iloc[0]["인구1인당_실질예산_전전년도"] == pytest.approx(80.0)
