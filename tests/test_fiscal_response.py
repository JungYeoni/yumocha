"""재정반응성 고정효과 모형 유틸 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.fiscal_response import (
    fit_two_way_fixed_effects,
    run_fiscal_response_models,
)


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    for region_index, region in enumerate(["서울", "부산", "대구", "광주"]):
        for year_index, year in enumerate(range(2018, 2024)):
            predictor = (region_index - 1.5) * (year_index - 2.5) / 10
            noise = ((region_index + year_index) % 3 - 1) * 0.01
            outcome = 2.0 * predictor + region_index * 0.4 + year_index * 0.2 + noise
            rows.append(
                {
                    "지역": region,
                    "연도": year,
                    "직전1년_출산율하락도": predictor,
                    "재정대응지수_z": outcome,
                    "총액기준_재정대응지수_z": outcome * 0.5,
                    "원자료_누락주의": "누락" if region == "서울" and year == 2019 else None,
                }
            )
    return pd.DataFrame(rows)


def test_fit_two_way_fixed_effects_recovers_predictor_coefficient():
    panel = _synthetic_panel()

    model, sample = fit_two_way_fixed_effects(
        panel,
        outcome="재정대응지수_z",
        predictor="직전1년_출산율하락도",
    )

    predictor_index = model.model.exog_names.index("직전1년_출산율하락도")
    assert model.params.iloc[predictor_index] == pytest.approx(2.0, abs=0.03)
    assert model.cov_type == "cluster"
    assert model.use_t
    assert len(sample) == 24


def test_run_fiscal_response_models_builds_outcome_and_sensitivity_rows():
    panel = _synthetic_panel()

    summary, models = run_fiscal_response_models(
        panel,
        outcomes={
            "인구당지수": "재정대응지수_z",
            "총액지수": "총액기준_재정대응지수_z",
        },
    )

    assert len(summary) == 4
    assert len(models) == 4
    assert set(summary["관측치"]) == {23, 24}
    assert summary["지역수"].eq(4).all()
    assert summary["연도수"].eq(6).all()
    assert summary["지역군집표준오차"].all()
    assert np.isfinite(summary[["계수", "군집표준오차", "p값"]].to_numpy()).all()


def test_fit_two_way_fixed_effects_with_controls_recovers_both_coefficients():
    panel = _synthetic_panel()
    rng = np.random.default_rng(42)
    panel["통제변수"] = rng.normal(size=len(panel))
    region_effect = panel["지역"].map({"서울": 0, "부산": 0.4, "대구": 0.8, "광주": 1.2})
    year_effect = (panel["연도"] - 2018) * 0.2
    panel["재정대응지수_z"] = (
        2.0 * panel["직전1년_출산율하락도"] + 1.5 * panel["통제변수"] + region_effect + year_effect
    )

    model, sample = fit_two_way_fixed_effects(
        panel,
        outcome="재정대응지수_z",
        predictor="직전1년_출산율하락도",
        controls=["통제변수"],
    )

    predictor_index = model.model.exog_names.index("직전1년_출산율하락도")
    control_index = model.model.exog_names.index("통제변수")
    assert model.params.iloc[predictor_index] == pytest.approx(2.0, abs=0.03)
    assert model.params.iloc[control_index] == pytest.approx(1.5, abs=0.03)
    assert len(sample) == 24


def test_fit_two_way_fixed_effects_rejects_missing_control_values():
    panel = _synthetic_panel()
    panel["통제변수"] = 1.0
    panel.loc[0, "통제변수"] = np.nan

    with pytest.raises(ValueError, match="결측 또는 비수치"):
        fit_two_way_fixed_effects(
            panel,
            outcome="재정대응지수_z",
            predictor="직전1년_출산율하락도",
            controls=["통제변수"],
        )


def test_fit_two_way_fixed_effects_rejects_duplicate_and_missing_observations():
    panel = _synthetic_panel()
    duplicated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="지역×연도 키 중복"):
        fit_two_way_fixed_effects(
            duplicated,
            outcome="재정대응지수_z",
            predictor="직전1년_출산율하락도",
        )

    panel.loc[0, "재정대응지수_z"] = np.nan
    with pytest.raises(ValueError, match="결측 또는 비수치"):
        fit_two_way_fixed_effects(
            panel,
            outcome="재정대응지수_z",
            predictor="직전1년_출산율하락도",
        )
