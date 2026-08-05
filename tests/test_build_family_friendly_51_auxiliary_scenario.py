import numpy as np
import pandas as pd

from scripts.build_family_friendly_51_auxiliary_scenario import (
    RAKING_METHOD,
    build_raking_candidates,
)
from scripts.build_family_friendly_national_candidates import NATIONAL_CUMULATIVE_TOTALS


def _sample_regions() -> list[str]:
    return ["서울", "부산", "대구"]


def _sample_denominators(regions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            2017: [1_050, 2_050, 1_550],
            2019: [1_100, 2_100, 1_600],
        },
        index=regions,
    )


def test_build_raking_candidates_sums_to_national_total_for_2019():
    regions = _sample_regions()
    count_2018 = pd.Series([10.0, 20.0, 5.0], index=regions)
    count_2020 = pd.Series([12.0, 24.0, 7.0], index=regions)
    denominators = _sample_denominators(regions)
    qa: list[dict[str, object]] = []

    import scripts.build_family_friendly_51_auxiliary_scenario as mod

    original_region_order = mod.REGION_ORDER
    mod.REGION_ORDER = regions
    try:
        result = build_raking_candidates(
            2019, count_2018, count_2020, denominators, qa, before_year=2018, after_year=2020
        )
    finally:
        mod.REGION_ORDER = original_region_order

    assert len(result) == 3
    assert result["연도"].eq(2019).all()
    assert result["방법"].eq(RAKING_METHOD).all()
    assert np.isclose(result["추정_분자"].sum(), NATIONAL_CUMULATIVE_TOTALS[2019], atol=1e-6)

    interpolated_seoul = (10.0 + 12.0) / 2
    raking_factor = NATIONAL_CUMULATIVE_TOTALS[2019] / ((10 + 12) / 2 + (20 + 24) / 2 + (5 + 7) / 2)
    expected_seoul_numerator = interpolated_seoul * raking_factor
    seoul_row = result.loc[result["지역"].eq("서울")].iloc[0]
    assert np.isclose(seoul_row["추정_분자"], expected_seoul_numerator, rtol=1e-9)
    assert np.isclose(seoul_row["추정_비율"], expected_seoul_numerator / 1_100 * 100, rtol=1e-9)
    assert pd.DataFrame(qa)["판정"].eq("PASS").all()


def test_build_raking_candidates_sums_to_national_total_for_2017():
    regions = _sample_regions()
    count_2016 = pd.Series([9.0, 18.0, 4.0], index=regions)
    count_2018 = pd.Series([10.0, 20.0, 5.0], index=regions)
    denominators = _sample_denominators(regions)
    qa: list[dict[str, object]] = []

    import scripts.build_family_friendly_51_auxiliary_scenario as mod

    original_region_order = mod.REGION_ORDER
    mod.REGION_ORDER = regions
    try:
        result = build_raking_candidates(
            2017, count_2016, count_2018, denominators, qa, before_year=2016, after_year=2018
        )
    finally:
        mod.REGION_ORDER = original_region_order

    assert len(result) == 3
    assert result["연도"].eq(2017).all()
    assert result["방법"].eq(RAKING_METHOD).all()
    assert np.isclose(result["추정_분자"].sum(), NATIONAL_CUMULATIVE_TOTALS[2017], atol=1e-6)
    assert pd.DataFrame(qa)["판정"].eq("PASS").all()
