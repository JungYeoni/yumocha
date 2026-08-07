import numpy as np
import pandas as pd

import pytest

from scripts.run_subarea_fiscal_tfr_regression import (
    LAG1_COLUMN,
    LAG2_COLUMN,
    STRUCTURAL_CONTROL_COLUMN,
    run_subarea_models,
)


def _synthetic_sample(
    true_coefficient: float,
    subarea: str,
    *,
    lag_column: str = LAG1_COLUMN,
    control_coefficient: float | None = None,
) -> list[dict[str, object]]:
    rows = []
    for region_index, region in enumerate(["서울", "부산", "대구", "광주"]):
        for year_index, year in enumerate(range(2018, 2024)):
            log_predictor = (region_index - 1.5) * (year_index - 2.5) / 10
            control_value = (region_index % 2) * (year_index % 2) * 0.3
            noise = ((region_index + year_index) % 3 - 1) * 0.001
            outcome = (
                true_coefficient * log_predictor + region_index * 0.4 + year_index * 0.2 + noise
            )
            if control_coefficient is not None:
                outcome += control_coefficient * control_value
            row = {
                "지역": region,
                "연도": year,
                "세부영역": subarea,
                "합계출산율": outcome,
                lag_column: np.expm1(log_predictor),
            }
            if control_coefficient is not None:
                row[STRUCTURAL_CONTROL_COLUMN] = control_value
            rows.append(row)
    return rows


def test_run_subarea_models_estimates_each_subarea_independently_with_lag1():
    sample = pd.DataFrame(
        _synthetic_sample(2.0, "1-1. 고용여건") + _synthetic_sample(-1.5, "2-1. 돌봄 여건")
    )

    result = run_subarea_models(sample, lag_column=LAG1_COLUMN)

    assert len(result) == 2
    row_x = result.loc[result["모형"].eq("1-1. 고용여건")].iloc[0]
    row_y = result.loc[result["모형"].eq("2-1. 돌봄 여건")].iloc[0]
    assert row_x["계수"] == pytest.approx(2.0, abs=0.03)
    assert row_y["계수"] == pytest.approx(-1.5, abs=0.03)
    assert row_x["설명변수"] == f"log1p_{LAG1_COLUMN}"
    assert row_x["관측치"] == 24


def test_run_subarea_models_supports_two_year_lag_column():
    sample = pd.DataFrame(_synthetic_sample(1.2, "1-1. 고용여건", lag_column=LAG2_COLUMN))

    result = run_subarea_models(sample, lag_column=LAG2_COLUMN)

    assert result.iloc[0]["계수"] == pytest.approx(1.2, abs=0.03)
    assert result.iloc[0]["설명변수"] == f"log1p_{LAG2_COLUMN}"


def test_run_subarea_models_supports_standardized_index_without_log_transform():
    rows = _synthetic_sample(1.2, "1-1. 고용여건")
    sample = pd.DataFrame(rows)
    sample[LAG1_COLUMN] = np.log1p(sample[LAG1_COLUMN])

    result = run_subarea_models(sample, lag_column=LAG1_COLUMN, transform="identity")

    assert result.iloc[0]["계수"] == pytest.approx(1.2, abs=0.03)
    assert result.iloc[0]["설명변수"] == LAG1_COLUMN


def test_run_subarea_models_drops_rows_with_missing_lag_predictor():
    rows = _synthetic_sample(2.0, "1-1. 고용여건")
    rows[0][LAG1_COLUMN] = None
    sample = pd.DataFrame(rows)

    result = run_subarea_models(sample, lag_column=LAG1_COLUMN)

    assert result.iloc[0]["관측치"] == 23


def test_run_subarea_models_with_structural_control_recovers_both_coefficients():
    sample = pd.DataFrame(_synthetic_sample(2.0, "1-1. 고용여건", control_coefficient=0.7))

    result = run_subarea_models(
        sample, lag_column=LAG1_COLUMN, controls=[STRUCTURAL_CONTROL_COLUMN]
    )

    assert result.iloc[0]["계수"] == pytest.approx(2.0, abs=0.03)
    assert result.iloc[0]["관측치"] == 24


def test_run_subarea_models_with_structural_control_drops_rows_missing_control():
    rows = _synthetic_sample(2.0, "1-1. 고용여건", control_coefficient=0.7)
    rows[0][STRUCTURAL_CONTROL_COLUMN] = None
    sample = pd.DataFrame(rows)

    result = run_subarea_models(
        sample, lag_column=LAG1_COLUMN, controls=[STRUCTURAL_CONTROL_COLUMN]
    )

    assert result.iloc[0]["관측치"] == 23
