import pandas as pd

import pytest

from scripts.verify_provisional_area_shares import (
    check_out_of_range_is_only_known_cause,
    check_share_ranges,
    compute_shares,
)


def test_compute_shares_sums_to_one_for_normal_budgets():
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "대영역": "A", "당해계획예산_백만원_provisional": 60.0},
            {"지역": "서울", "연도": 2021, "대영역": "B", "당해계획예산_백만원_provisional": 40.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    assert shares["구성비"].tolist() == [0.6, 0.4]
    assert shares["구성비"].sum() == pytest.approx(1.0)


def test_check_share_ranges_passes_for_normal_shares():
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "대영역": "A", "당해계획예산_백만원_provisional": 60.0},
            {"지역": "서울", "연도": 2021, "대영역": "B", "당해계획예산_백만원_provisional": 40.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    qa = check_share_ranges(shares, category_column="대영역")
    hard_failures = qa.loc[qa["판정"].eq("FAIL")]
    assert hard_failures.empty


def test_check_share_ranges_flags_share_sum_mismatch():
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "대영역": "A", "당해계획예산_백만원_provisional": 60.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    # 인위적으로 합계가 1이 되도록 만든 구성비를 깨서 불일치를 만든다.
    shares["구성비"] = 0.5
    qa = check_share_ranges(shares, category_column="대영역")
    row = qa.loc[qa["검사항목"].str.contains("합계 != 1")].iloc[0]
    assert row["판정"] == "FAIL"
    assert row["실제값"] == 1


def test_check_out_of_range_is_only_known_cause_passes_for_gyeongbuk_2018():
    panel = pd.DataFrame(
        [
            {"지역": "경북", "연도": 2018, "대영역": "A", "당해계획예산_백만원_provisional": -20.0},
            {"지역": "경북", "연도": 2018, "대영역": "B", "당해계획예산_백만원_provisional": 120.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    check_out_of_range_is_only_known_cause(shares)  # 예외 없이 통과해야 한다


def test_check_out_of_range_raises_for_unexpected_region_year():
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "대영역": "A", "당해계획예산_백만원_provisional": -20.0},
            {"지역": "서울", "연도": 2021, "대영역": "B", "당해계획예산_백만원_provisional": 120.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    with pytest.raises(ValueError, match="알려진 원인"):
        check_out_of_range_is_only_known_cause(shares)
