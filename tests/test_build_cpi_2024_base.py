import pandas as pd

import pytest

from scripts.build_cpi_2024_base import rebase_cpi


def _sample_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "연도": [2020, 2021, 2024],
            "소비자물가지수": [100.000, 102.590, 114.240],
            "전년대비상승률_pct": [0.5, 2.6, 2.3],
            "기준연도": ["2020=100", "2020=100", "2020=100"],
        }
    )


def test_rebase_cpi_sets_target_year_to_100():
    rebased = rebase_cpi(_sample_source(), target_base_year=2024)
    target_row = rebased.loc[rebased["연도"].eq(2024)].iloc[0]
    assert target_row["소비자물가지수"] == 100.0
    assert target_row["기준연도"] == "2024=100"


def test_rebase_cpi_preserves_relative_ratios():
    rebased = rebase_cpi(_sample_source(), target_base_year=2024)
    original = _sample_source()
    original_ratio = (
        original.loc[original["연도"].eq(2021), "소비자물가지수"].iloc[0]
        / original.loc[original["연도"].eq(2020), "소비자물가지수"].iloc[0]
    )
    rebased_ratio = (
        rebased.loc[rebased["연도"].eq(2021), "소비자물가지수"].iloc[0]
        / rebased.loc[rebased["연도"].eq(2020), "소비자물가지수"].iloc[0]
    )
    # rebase_cpi rounds to 6 decimals for output readability, so the ratio is
    # only approximately preserved, not exact to floating-point precision.
    assert rebased_ratio == pytest.approx(original_ratio, rel=1e-6)


def test_rebase_cpi_rejects_wrong_source_unit():
    source = _sample_source()
    source["기준연도"] = "2015=100"
    with pytest.raises(ValueError, match="기준연도 표기"):
        rebase_cpi(source, target_base_year=2024)


def test_rebase_cpi_rejects_missing_target_year():
    source = _sample_source()
    source = source.loc[source["연도"].ne(2024)]
    with pytest.raises(ValueError, match="목표 기준연도"):
        rebase_cpi(source, target_base_year=2024)
