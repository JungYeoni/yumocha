import numpy as np
import pandas as pd
import pytest

from scripts.run_cluster_fiscal_tfr_regression import (
    _coefficient_contrast,
    add_bh,
    merge_clusters,
    run_interaction_models,
    run_subgroup_models,
)


class _StubModel:
    def __init__(self, names, params, covariance):
        self.model = type("M", (), {"exog_names": names})()
        self.params = pd.Series(params, index=names)
        self._covariance = np.asarray(covariance)
        self.df_resid = 100

    def cov_params(self):
        return self._covariance


def test_merge_clusters_requires_complete_region_mapping():
    sample = pd.DataFrame({"지역": ["서울", "부산"], "연도": [2020, 2020]})
    clusters = pd.DataFrame(
        {
            "region": ["서울", *[f"지역{i}" for i in range(16)]],
            "군집_2개": [1] * 17,
            "군집_3개": [1] * 17,
        }
    )
    with pytest.raises(ValueError, match="매칭되지 않은"):
        merge_clusters(sample, clusters)


def test_add_bh_corrects_within_each_lag_and_cluster():
    table = pd.DataFrame(
        {
            "시차": ["t+1"] * 4,
            "군집": [1, 1, 2, 2],
            "p값": [0.001, 0.2, 0.04, 0.5],
        }
    )
    result = add_bh(table, ["시차", "군집"])
    assert result.loc[0, "FDR_q값"] == pytest.approx(0.002)
    assert result.loc[2, "FDR_q값"] == pytest.approx(0.08)


def test_add_bh_preserves_unavailable_nan_rows():
    table = pd.DataFrame(
        {
            "시차": ["t+1"] * 3,
            "군집": [1, 1, 1],
            "p값": [0.01, float("nan"), 0.2],
        }
    )
    result = add_bh(table, ["시차", "군집"])
    assert result.loc[0, "FDR_q값"] == pytest.approx(0.02)
    assert pd.isna(result.loc[1, "FDR_q값"])


def test_coefficient_contrast_returns_nan_p_value_for_zero_standard_error():
    model = _StubModel(["x"], [0.5], [[0.0]])
    coefficient, standard_error, p_value, lower, upper = _coefficient_contrast(model, {"x": 1.0})
    assert coefficient == pytest.approx(0.5)
    assert standard_error == 0.0
    assert np.isnan(p_value)
    assert lower == upper == pytest.approx(0.5)


def test_unavailable_subgroup_uses_actual_subarea_observation_counts():
    sample = pd.DataFrame(
        {
            "지역": ["가", "가", "나"],
            "연도": [2020, 2021, 2020],
            "세부영역": ["A", "A", "B"],
            "군집_3개": [2, 2, 2],
            "합계출산율_t+1": [1.0, 0.9, 1.1],
            "합계출산율_t+2": [0.9, 0.8, 1.0],
            "인구1인당_실질예산_3개년평균": [10.0, 20.0, 30.0],
        }
    )
    result = run_subgroup_models(sample, cluster_count=3)
    t1_cluster2 = result.loc[result["시차"].eq("t+1") & result["군집"].eq(2)]
    counts = t1_cluster2.set_index("모형")["관측치"].to_dict()
    assert counts == {"A": 2, "B": 1}


def test_two_region_subgroup_returns_coefficients_without_inference():
    rows = []
    for region, offset in (("대전", 0.0), ("세종", 0.1)):
        for year in range(2018, 2024):
            budget = float((year - 2017) * 10 + offset)
            rows.append(
                {
                    "지역": region,
                    "연도": year,
                    "세부영역": "A",
                    "군집_3개": 2,
                    "합계출산율_t+1": 0.8 + 0.01 * budget,
                    "합계출산율_t+2": 0.7 + 0.01 * budget,
                    "인구1인당_실질예산_3개년평균": budget,
                }
            )
    result = run_subgroup_models(pd.DataFrame(rows), cluster_count=3)
    exploratory = result.loc[result["군집"].eq(2)]
    assert exploratory["계수"].notna().all()
    assert exploratory["p값"].isna().all()
    assert exploratory["추정가능"].all()
    assert exploratory["추론가능"].eq(False).all()  # noqa: E712
    assert exploratory["분석구분"].eq("2개 시도 계수만 탐색").all()


def test_interaction_model_records_missing_cluster_diagnostic():
    sample = pd.DataFrame(
        {
            "지역": ["가", "나", "가", "나"],
            "연도": [2020, 2020, 2020, 2020],
            "세부영역": ["A", "A", "B", "B"],
            "군집_2개": [1, 1, 1, 1],
            "합계출산율_t+1": [1.0, 1.1, 1.0, 1.1],
            "합계출산율_t+2": [0.9, 1.0, 0.9, 1.0],
            "인구1인당_실질예산_3개년평균": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = run_interaction_models(sample)
    assert len(result) == 4
    assert result["시차"].value_counts().to_dict() == {"t+1": 2, "t+2": 2}
    assert (~result["추정가능"]).all()
    assert result["추정불가사유"].str.contains("2개 군집").all()
