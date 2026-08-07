import pandas as pd
import pytest

from scripts.build_structural_environment_clusters import (
    EXPECTED_CATEGORIES,
    build_elbow_diagnostics,
    build_region_profiles,
    fit_cluster_solution,
)


def _panel() -> pd.DataFrame:
    rows = []
    for region_index in range(17):
        for year in range(2016, 2025):
            for category_index, category in enumerate(EXPECTED_CATEGORIES):
                rows.append(
                    {
                        "region": f"지역{region_index:02d}",
                        "year": year,
                        "category": category,
                        "category_score": region_index * 3 + category_index + (year - 2016) * 0.1,
                    }
                )
    return pd.DataFrame(rows)


def test_build_region_profiles_validates_complete_panel():
    profiles = build_region_profiles(_panel())
    assert profiles.shape == (17, 4)
    assert list(profiles.columns) == EXPECTED_CATEGORIES


def test_build_region_profiles_rejects_duplicate_key():
    data = _panel()
    with pytest.raises(ValueError, match="키 중복"):
        build_region_profiles(pd.concat([data, data.iloc[[0]]], ignore_index=True))


def test_fit_cluster_solution_is_deterministic_and_numbered_from_one():
    profiles = build_region_profiles(_panel())
    first, metrics = fit_cluster_solution(profiles, 2)
    second, _ = fit_cluster_solution(profiles, 2)
    assert first["군집_2개"].tolist() == second["군집_2개"].tolist()
    assert set(first["군집_2개"]) == {1, 2}
    assert metrics["군집수"] == 2


def test_build_elbow_diagnostics_includes_wcss_and_silhouette():
    diagnostics = build_elbow_diagnostics(build_region_profiles(_panel()), max_clusters=4)
    assert diagnostics["군집수"].tolist() == [1, 2, 3, 4]
    assert diagnostics["WCSS"].is_monotonic_decreasing
    assert pd.isna(diagnostics.loc[0, "실루엣점수"])
    assert diagnostics.loc[1:, "실루엣점수"].notna().all()
