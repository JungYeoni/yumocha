import numpy as np
import pandas as pd
import pytest

from scripts.build_subarea_structural_fiscal_response_sample import (
    LOG_FISCAL_OUTCOME,
    STRUCTURAL_PREDICTOR,
)
from scripts.run_region_heterogeneous_structural_fiscal_response import (
    fit_region_interaction_model,
    run_common_response_models,
)


def _synthetic_sample(*, collinear_slopes: bool = False) -> pd.DataFrame:
    rows = []
    regions = ["서울", "부산", "대구", "광주"]
    slopes = {"서울": 0.5, "부산": -0.2, "대구": 0.8, "광주": 0.1}
    for region_index, region in enumerate(regions):
        for year_index, budget_year in enumerate(range(2018, 2025)):
            if collinear_slopes:
                predictor = float(year_index + region_index)
            else:
                predictor = float(
                    (year_index - 3) * (region_index + 1) + ((region_index + year_index) % 3) * 0.3
                )
            outcome = (
                slopes[region] * predictor
                + region_index * 0.7
                + year_index * 0.4
                + ((region_index + year_index) % 2) * 0.01
            )
            rows.append(
                {
                    "지역": region,
                    "구조환경연도": budget_year - 2,
                    "예산연도": budget_year,
                    "세부영역": "1-1. 고용여건",
                    STRUCTURAL_PREDICTOR: predictor,
                    LOG_FISCAL_OUTCOME: outcome,
                }
            )
    return pd.DataFrame(rows)


def test_common_model_uses_budget_year_fixed_effects():
    result = run_common_response_models(_synthetic_sample())

    assert len(result) == 1
    assert result.iloc[0]["관측치"] == 28
    assert result.iloc[0]["지역수"] == 4
    assert result.iloc[0]["연도수"] == 7
    assert result.iloc[0]["설명변수"] == STRUCTURAL_PREDICTOR
    values = result.loc[0, ["계수", "군집표준오차", "p값", "FDR_q값"]].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_common_model_accepts_t_plus_1_outcome():
    outcome = "log1p_인구1인당_실질예산_t+1"
    sample = _synthetic_sample().rename(columns={LOG_FISCAL_OUTCOME: outcome})

    result = run_common_response_models(sample, outcome=outcome)

    assert result.iloc[0]["종속변수"] == outcome
    assert np.isfinite(result.iloc[0]["계수"])


def test_region_interaction_model_returns_each_region_slope():
    result = fit_region_interaction_model(_synthetic_sample())

    assert len(result) == 4
    assert result["추정가능"].all()
    expected = {"서울": 0.5, "부산": -0.2, "대구": 0.8, "광주": 0.1}
    for region, coefficient in expected.items():
        estimated = result.loc[result["지역"].eq(region), "지역별_반응계수"].iloc[0]
        assert estimated == pytest.approx(coefficient, abs=0.03)


def test_region_interaction_model_marks_rank_deficiency_without_forcing_coefficients():
    result = fit_region_interaction_model(_synthetic_sample(collinear_slopes=True))

    assert not result["추정가능"].any()
    assert result["지역별_반응계수"].isna().all()
    assert result["추정제외사유"].str.contains("선형종속").all()
    assert (result["설계행렬_계수"] < result["설계행렬_모수수"]).all()
