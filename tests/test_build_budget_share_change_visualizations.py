import pandas as pd
import pytest

from scripts.build_budget_share_change_visualizations import (
    build_budget_share_panel,
    build_national_change_summary,
    build_regional_change_table,
    validate_budget_panel,
)


def _panel() -> pd.DataFrame:
    rows = []
    for region_index in range(17):
        region = f"지역{region_index}"
        for year in range(2016, 2025):
            for category_index in range(12):
                rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "세부영역": f"영역{category_index}",
                        "당해계획예산_백만원_provisional": float(
                            (category_index + 1) * (2 if year == 2024 else 1)
                        ),
                        "사업수": 1,
                        "예산결측_사업수": 0,
                    }
                )
    return pd.DataFrame(rows)


def test_validate_budget_panel_rejects_missing_category() -> None:
    panel = _panel().iloc[:-1]
    with pytest.raises(ValueError, match="12개 세부영역"):
        validate_budget_panel(panel)


def test_budget_shares_sum_to_100_by_region_year() -> None:
    shares = build_budget_share_panel(_panel())
    totals = shares.groupby(["지역", "연도"])["계획예산비중_pct"].sum()
    assert totals.eq(100).all()


def test_change_tables_keep_all_categories_and_regions() -> None:
    shares = build_budget_share_panel(_panel())
    national = build_national_change_summary(shares)
    regional = build_regional_change_table(shares)
    assert len(national) == 12
    assert len(regional) == 17 * 12
    assert national["2016→2024_비중변화_pp"].abs().max() < 1e-10
