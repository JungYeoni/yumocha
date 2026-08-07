import pandas as pd

import pytest

from scripts.verify_provisional_area_shares import check_share_ranges, compute_shares


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


def test_check_share_ranges_flags_out_of_range_share():
    """음수 예산 셀이 있으면(더 이상 알려진 예외 없음) 그 자체로 실패해야 한다.

    경북 2018 "여성지도자 육성" -20은 원본 PDF 오탈자로 확인돼 20으로
    보정했으므로(src/provisional/loader.py), 이제 범위 이탈이 있으면 예외
    없이 바로 실패로 잡아야 한다.
    """
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "대영역": "A", "당해계획예산_백만원_provisional": -20.0},
            {"지역": "서울", "연도": 2021, "대영역": "B", "당해계획예산_백만원_provisional": 120.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    qa = check_share_ranges(shares, category_column="대영역")
    row = qa.loc[qa["검사항목"].str.contains("범위 이탈")].iloc[0]
    assert row["판정"] == "FAIL"
    # 총액이 100인데 -20(구성비 -0.2)과 120(구성비 1.2) 둘 다 0~1 범위를 벗어난다.
    assert row["실제값"] == 2


def test_check_share_ranges_flags_non_finite_share_for_zero_total():
    """총액이 0인 지역·연도는 0/0 나눗셈으로 비유한값(NaN)이 나온다.

    src/provisional/aggregator.py의 _fill_empty_combinations은 예산 0인 행을
    명시적으로 만들 수 있으므로(어떤 지역이 특정 연도에 실제로 예산을 하나도
    안 썼다면), 지역·연도 총액 자체가 0인 상황이 실제로 발생할 수 있다.
    """
    panel = pd.DataFrame(
        [
            {"지역": "서울", "연도": 2021, "대영역": "A", "당해계획예산_백만원_provisional": 0.0},
            {"지역": "서울", "연도": 2021, "대영역": "B", "당해계획예산_백만원_provisional": 0.0},
        ]
    )
    shares = compute_shares(panel, category_column="대영역")
    qa = check_share_ranges(shares, category_column="대영역")
    row = qa.loc[qa["검사항목"].str.contains("비유한값")].iloc[0]
    assert row["판정"] == "FAIL"
