import pandas as pd

import pytest

from scripts.build_subarea_fiscal_response_regression_sample import (
    FISCAL_TO_STRUCTURAL_SUBCATEGORY,
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

    result = build_regression_sample(fiscal_response, fertility_lagged, structural_lagged)

    assert len(result) == 1
    assert result.iloc[0]["세부영역"] == "1-1. 고용여건"
    assert result.iloc[0]["구조환경지수_전년도"] == pytest.approx(42.0)
    assert result.iloc[0]["직전1년_출산율하락도"] == pytest.approx(0.05)
