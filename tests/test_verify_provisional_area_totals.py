import pandas as pd

import pytest

from scripts.verify_provisional_area_totals import compare_major_and_sub_totals


def test_compare_major_and_sub_totals_passes_when_sums_match():
    major = pd.DataFrame(
        [
            {
                "지역": "서울",
                "연도": 2021,
                "대영역": "1. 경제·고용·주거",
                "당해계획예산_백만원_provisional": 60.0,
            },
            {
                "지역": "서울",
                "연도": 2021,
                "대영역": "2. 가족·생활",
                "당해계획예산_백만원_provisional": 40.0,
            },
        ]
    )
    sub = pd.DataFrame(
        [
            {
                "지역": "서울",
                "연도": 2021,
                "세부영역": "1-1. 고용여건",
                "당해계획예산_백만원_provisional": 60.0,
            },
            {
                "지역": "서울",
                "연도": 2021,
                "세부영역": "2-1. 돌봄 여건",
                "당해계획예산_백만원_provisional": 40.0,
            },
        ]
    )
    result = compare_major_and_sub_totals(major, sub)
    row = result.iloc[0]
    assert row["대영역_합계_백만원"] == 100.0
    assert row["세부영역_합계_백만원"] == 100.0
    assert row["일치"]


def test_compare_major_and_sub_totals_flags_mismatch():
    major = pd.DataFrame([{"지역": "서울", "연도": 2021, "당해계획예산_백만원_provisional": 100.0}])
    sub = pd.DataFrame([{"지역": "서울", "연도": 2021, "당해계획예산_백만원_provisional": 90.0}])
    result = compare_major_and_sub_totals(major, sub)
    row = result.iloc[0]
    assert not row["일치"]
    assert row["차이_백만원"] == pytest.approx(10.0)


def test_compare_major_and_sub_totals_rejects_mismatched_keys():
    major = pd.DataFrame([{"지역": "서울", "연도": 2021, "당해계획예산_백만원_provisional": 100.0}])
    sub = pd.DataFrame([{"지역": "부산", "연도": 2021, "당해계획예산_백만원_provisional": 90.0}])
    with pytest.raises(ValueError, match="키가 서로 다릅니다"):
        compare_major_and_sub_totals(major, sub)
