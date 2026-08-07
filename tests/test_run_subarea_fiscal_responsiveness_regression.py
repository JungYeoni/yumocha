import numpy as np
import pandas as pd

import pytest

from scripts.run_subarea_fiscal_responsiveness_regression import run_subarea_models


def _synthetic_sample(true_coefficient: float, subarea: str) -> list[dict[str, object]]:
    rows = []
    for region_index, region in enumerate(["서울", "부산", "대구", "광주"]):
        for year_index, year in enumerate(range(2018, 2024)):
            predictor = (region_index - 1.5) * (year_index - 2.5) / 10
            noise = ((region_index + year_index) % 3 - 1) * 0.01
            log1p_target = (
                true_coefficient * predictor + region_index * 0.4 + year_index * 0.2 + noise
            )
            rows.append(
                {
                    "지역": region,
                    "연도": year,
                    "세부영역": subarea,
                    "직전1년_출산율하락도": predictor,
                    "인구1인당_실질예산_원": np.expm1(log1p_target),
                }
            )
    return rows


def test_run_subarea_models_estimates_each_subarea_independently():
    sample = pd.DataFrame(
        _synthetic_sample(2.0, "1-1. 고용여건") + _synthetic_sample(-1.5, "2-1. 돌봄 여건")
    )

    result = run_subarea_models(sample)

    assert len(result) == 2
    row_x = result.loc[result["모형"].eq("1-1. 고용여건")].iloc[0]
    row_y = result.loc[result["모형"].eq("2-1. 돌봄 여건")].iloc[0]
    assert row_x["계수"] == pytest.approx(2.0, abs=0.03)
    assert row_y["계수"] == pytest.approx(-1.5, abs=0.03)
    assert row_x["관측치"] == 24
    assert row_x["지역수"] == 4
    assert row_x["연도수"] == 6


def test_run_subarea_models_drops_rows_with_missing_predictor():
    rows = _synthetic_sample(2.0, "1-1. 고용여건")
    rows[0]["직전1년_출산율하락도"] = None
    sample = pd.DataFrame(rows)

    result = run_subarea_models(sample)

    assert result.iloc[0]["관측치"] == 23
