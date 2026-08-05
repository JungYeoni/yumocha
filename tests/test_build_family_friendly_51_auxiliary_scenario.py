import numpy as np
import pandas as pd

from scripts.build_family_friendly_51_auxiliary_scenario import (
    COMPOSITION_METHOD,
    RAKING_METHOD,
    build_composition_candidates,
    build_raking_candidates,
)
from scripts.build_family_friendly_national_candidates import NATIONAL_CUMULATIVE_TOTALS


def _sample_regions() -> list[str]:
    return ["서울", "부산", "대구"]


def _sample_denominators(regions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            2016: [1_000, 2_000, 1_500],
            2017: [1_050, 2_050, 1_550],
            2019: [1_100, 2_100, 1_600],
            2020: [1_150, 2_150, 1_650],
        },
        index=regions,
    )


def test_build_raking_candidates_sums_to_national_total():
    regions = _sample_regions()
    count_2018 = pd.Series([10.0, 20.0, 5.0], index=regions)
    count_2020 = pd.Series([12.0, 24.0, 7.0], index=regions)
    denominators = _sample_denominators(regions)
    qa: list[dict[str, object]] = []

    # patch REGION_ORDER-dependent module constant via monkeypatch-free approach:
    # build_raking_candidates uses the module-level REGION_ORDER for reindexing,
    # so restrict comparison to the 3 sample regions actually present.
    import scripts.build_family_friendly_51_auxiliary_scenario as mod

    original_region_order = mod.REGION_ORDER
    mod.REGION_ORDER = regions
    try:
        result = build_raking_candidates(count_2018, count_2020, denominators, qa)
    finally:
        mod.REGION_ORDER = original_region_order

    assert len(result) == 3
    assert result["방법"].eq(RAKING_METHOD).all()
    assert np.isclose(result["추정_분자"].sum(), NATIONAL_CUMULATIVE_TOTALS[2019], atol=1e-6)
    # interpolated 서울 = (10+12)/2 = 11, before raking
    interpolated_seoul = (10.0 + 12.0) / 2
    raking_factor = NATIONAL_CUMULATIVE_TOTALS[2019] / ((10 + 12) / 2 + (20 + 24) / 2 + (5 + 7) / 2)
    expected_seoul_numerator = interpolated_seoul * raking_factor
    seoul_row = result.loc[result["지역"].eq("서울")].iloc[0]
    assert np.isclose(seoul_row["추정_분자"], expected_seoul_numerator, rtol=1e-9)
    assert np.isclose(seoul_row["추정_비율"], expected_seoul_numerator / 1_100 * 100, rtol=1e-9)
    assert pd.DataFrame(qa)["판정"].eq("PASS").all()


def test_build_composition_candidates_preserves_2018_share():
    regions = _sample_regions()
    count_2018 = pd.Series([30.0, 60.0, 10.0], index=regions)  # 30%,60%,10% share
    denominators = _sample_denominators(regions)
    qa: list[dict[str, object]] = []

    import scripts.build_family_friendly_51_auxiliary_scenario as mod

    original_region_order = mod.REGION_ORDER
    mod.REGION_ORDER = regions
    try:
        result = build_composition_candidates(count_2018, denominators, (2016, 2017), qa)
    finally:
        mod.REGION_ORDER = original_region_order

    assert len(result) == 6
    assert result["방법"].eq(COMPOSITION_METHOD).all()
    for year in (2016, 2017):
        year_rows = result.loc[result["연도"].eq(year)]
        assert np.isclose(year_rows["추정_분자"].sum(), NATIONAL_CUMULATIVE_TOTALS[year], atol=1e-6)
        seoul = year_rows.loc[year_rows["지역"].eq("서울")].iloc[0]
        # 서울 share = 30/100 = 0.3
        assert np.isclose(seoul["추정_분자"], 0.3 * NATIONAL_CUMULATIVE_TOTALS[year], rtol=1e-9)
    assert pd.DataFrame(qa)["판정"].eq("PASS").all()
