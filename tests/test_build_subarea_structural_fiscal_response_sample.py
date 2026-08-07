import numpy as np
import pandas as pd
import pytest

from scripts.build_subarea_fiscal_response_regression_sample import (
    FISCAL_TO_STRUCTURAL_SUBCATEGORY,
)
from scripts.build_subarea_structural_fiscal_response_sample import (
    FISCAL_OUTCOME,
    LOG_FISCAL_OUTCOME,
    STRUCTURAL_PREDICTOR,
    build_structural_fiscal_response_sample,
    fiscal_outcome_columns,
)


def _structural_scores() -> pd.DataFrame:
    rows = []
    for subcategory in FISCAL_TO_STRUCTURAL_SUBCATEGORY.values():
        for year in (2016, 2017, 2018):
            rows.append(
                {
                    "region": "A",
                    "year": year,
                    "subcategory": subcategory,
                    "subcategory_score": float(year - 2000),
                }
            )
    return pd.DataFrame(rows)


def _fiscal_panel() -> pd.DataFrame:
    rows = []
    for subarea in FISCAL_TO_STRUCTURAL_SUBCATEGORY:
        for year in (2018, 2019, 2020):
            rows.append(
                {
                    "지역": "A",
                    "연도": year,
                    "세부영역": subarea,
                    "인구1인당_실질예산_원": float(year * 10),
                }
            )
    rows.append(
        {
            "지역": "A",
            "연도": 2018,
            "세부영역": "지표체계 외",
            "인구1인당_실질예산_원": 999.0,
        }
    )
    return pd.DataFrame(rows)


def test_build_sample_aligns_structure_with_budget_two_years_later():
    result = build_structural_fiscal_response_sample(_fiscal_panel(), _structural_scores())

    row = result.loc[result["구조환경연도"].eq(2016) & result["세부영역"].eq("1-1. 고용여건")].iloc[
        0
    ]
    assert row["예산연도"] == 2018
    assert row[STRUCTURAL_PREDICTOR] == pytest.approx(16.0)
    assert row[FISCAL_OUTCOME] == pytest.approx(20180.0)
    assert row[LOG_FISCAL_OUTCOME] == pytest.approx(np.log1p(20180.0))
    assert "지표체계 외" not in set(result["세부영역"])


def test_build_sample_aligns_structure_with_budget_one_year_later():
    outcome, log_outcome = fiscal_outcome_columns(1)
    result = build_structural_fiscal_response_sample(
        _fiscal_panel(), _structural_scores(), lag_years=1
    )

    row = result.loc[result["구조환경연도"].eq(2017) & result["세부영역"].eq("1-1. 고용여건")].iloc[
        0
    ]
    assert row["예산연도"] == 2018
    assert row[outcome] == pytest.approx(20180.0)
    assert row[log_outcome] == pytest.approx(np.log1p(20180.0))


def test_build_sample_rejects_missing_two_year_budget_match():
    fiscal = _fiscal_panel()
    fiscal = fiscal.loc[~(fiscal["세부영역"].eq("1-1. 고용여건") & fiscal["연도"].eq(2018))]

    with pytest.raises(ValueError, match="필수값이 누락"):
        build_structural_fiscal_response_sample(fiscal, _structural_scores())


def test_build_sample_rejects_negative_budget():
    fiscal = _fiscal_panel()
    fiscal.loc[fiscal.index[0], "인구1인당_실질예산_원"] = -1.0

    with pytest.raises(ValueError, match="음수"):
        build_structural_fiscal_response_sample(fiscal, _structural_scores())
