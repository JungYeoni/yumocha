import pandas as pd

import pytest

from scripts.verify_housework_gender_equality_interpolation_safety import (
    NEUTRAL_MIDPOINT,
    OBSERVED_YEARS,
    check_interpolation_safety,
)

REGIONS = ["전국", "서울", "부산"]


def _means_below_midpoint() -> pd.DataFrame:
    # every value stays below 3 across all observed years, like the real data
    return pd.DataFrame(
        {
            year: [2.5 + 0.02 * i, 2.6 + 0.02 * i, 2.4 + 0.02 * i]
            for i, year in enumerate(OBSERVED_YEARS)
        },
        index=REGIONS,
    )


def _official_from_means(means: pd.DataFrame) -> pd.DataFrame:
    return 1 - (means - NEUTRAL_MIDPOINT).abs() / 2


def test_passes_when_no_value_crosses_midpoint():
    means = _means_below_midpoint()
    official = _official_from_means(means)

    qa = check_interpolation_safety(means, official, REGIONS)

    assert qa["판정"].eq("PASS").all()


def test_detects_midpoint_crossing():
    means = _means_below_midpoint()
    means.loc["서울", OBSERVED_YEARS[-1]] = 3.2  # crosses the midpoint
    official = _official_from_means(means)

    with pytest.raises(ValueError, match="교차 위험"):
        check_interpolation_safety(means, official, REGIONS)


def test_detects_mismatch_against_panel_values():
    means = _means_below_midpoint()
    official = _official_from_means(means)
    official.loc["부산", OBSERVED_YEARS[0]] += 0.01  # panel disagrees with reproduction

    with pytest.raises(ValueError, match="최대오차"):
        check_interpolation_safety(means, official, REGIONS)
