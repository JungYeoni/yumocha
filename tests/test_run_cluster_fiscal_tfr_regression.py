import pandas as pd
import pytest

from scripts.run_cluster_fiscal_tfr_regression import add_bh, merge_clusters


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
