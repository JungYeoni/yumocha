"""2021-2024 검토완료 워크북 확정 라벨 추출 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.consolidate_2021_2024_confirmed_labels import (
    TRAIN_COLUMNS,
    build_qa,
    load_confirmed_labels,
)
from scripts.consolidate_2021_area_labels import REGION_ORDER


def _review_frame(**overrides: list) -> pd.DataFrame:
    base = {
        "연도": [2021, 2021, 2022, 2016],
        "지역": ["서울", "부산", "서울", "서울"],
        "원본행": [1, 2, 3, 4],
        "세부사업명": ["청년 취업 지원", "아이 돌봄", "노인 일자리", "미검토 사업"],
        "주요내용_정제": ["고용 지원", "돌봄 서비스", "일자리 연계", pd.NA],
        "검토상태": ["확정", "수정", "확정", "미검토"],
        "검토_대영역": ["1. 경제·고용·주거", "2. 가족·생활", "1. 경제·고용·주거", None],
        "검토_세부영역": ["1-1. 고용여건", "2-1. 돌봄 여건", "1-1. 고용여건", None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_keeps_only_confirmed_and_revised_rows():
    result = load_confirmed_labels(_review_frame())

    assert len(result) == 3
    assert "미검토 사업" not in result["세부사업명"].tolist()
    assert list(result.columns) == TRAIN_COLUMNS


def test_excludes_confirmed_rows_outside_training_years():
    # 검토 워크북 작업 중 2016-2020년에도 스필오버로 확정되는 행이 있는데,
    # 이 행이 예측 대상(2016-2020) 취합본에도 포함될 수 있어 학습에서는
    # 제외해야 학습·예측 대상이 겹치지 않는다.
    frame = _review_frame(
        검토상태=["확정", "수정", "확정", "확정"],
        검토_대영역=["1. 경제·고용·주거", "2. 가족·생활", "1. 경제·고용·주거", "2. 가족·생활"],
        검토_세부영역=["1-1. 고용여건", "2-1. 돌봄 여건", "1-1. 고용여건", "2-1. 돌봄 여건"],
    )  # 마지막 행은 2016년, 확정 상태

    result = load_confirmed_labels(frame)

    assert len(result) == 3
    assert "미검토 사업" not in result["세부사업명"].tolist()
    assert set(result["연도"]) == {2021, 2022}


def test_renames_review_columns_to_training_schema():
    result = load_confirmed_labels(_review_frame())

    row = result.loc[result["세부사업명"].eq("청년 취업 지원")].iloc[0]
    assert row["대영역"] == "1. 경제·고용·주거"
    assert row["세부영역"] == "1-1. 고용여건"


def test_sorted_by_year_region_then_source_row():
    result = load_confirmed_labels(_review_frame())
    ordering = list(result[["연도", "지역", "원본행"]].itertuples(index=False, name=None))
    region_rank = {region: rank for rank, region in enumerate(REGION_ORDER)}

    assert ordering == sorted(ordering, key=lambda item: (item[0], region_rank[item[1]], item[2]))


def test_missing_required_column_raises():
    frame = _review_frame().drop(columns="검토_대영역")
    with pytest.raises(ValueError, match="필수 열 누락"):
        load_confirmed_labels(frame)


def test_empty_frame_raises():
    empty = pd.DataFrame(
        columns=[
            "연도",
            "지역",
            "원본행",
            "세부사업명",
            "주요내용_정제",
            "검토상태",
            "검토_대영역",
            "검토_세부영역",
        ]
    )
    with pytest.raises(ValueError, match="비어 있습니다"):
        load_confirmed_labels(empty)


def test_no_confirmed_rows_raises():
    frame = _review_frame(검토상태=["미검토", "미검토", "보류", "미검토"])
    with pytest.raises(ValueError, match="확정·수정 상태인 행이 없습니다"):
        load_confirmed_labels(frame)


def test_unknown_region_raises():
    frame = _review_frame(지역=["평양", "부산", "서울", "서울"])
    with pytest.raises(ValueError, match="REGION_ORDER에 없는"):
        load_confirmed_labels(frame)


def test_missing_label_on_confirmed_row_raises():
    frame = _review_frame(
        검토상태=["확정", "확정", "확정", "미검토"],
        검토_대영역=["1. 경제·고용·주거", None, "1. 경제·고용·주거", None],
    )
    with pytest.raises(ValueError, match="대영역·세부영역 결측"):
        load_confirmed_labels(frame)


def test_unknown_subcategory_raises():
    frame = _review_frame(
        검토_세부영역=["1-1. 고용여건", "존재하지않는영역", "1-1. 고용여건", None]
    )
    with pytest.raises(ValueError, match="정의되지 않은 세부영역"):
        load_confirmed_labels(frame)


def test_major_category_taxonomy_mismatch_raises():
    frame = _review_frame(
        검토_대영역=["1. 경제·고용·주거", "4. 사회·문화", "1. 경제·고용·주거", None]
    )
    with pytest.raises(ValueError, match="taxonomy와 어긋납니다"):
        load_confirmed_labels(frame)


def test_duplicate_key_raises():
    frame = _review_frame(원본행=[1, 1, 3, 4], 지역=["서울", "서울", "서울", "서울"])
    with pytest.raises(ValueError, match="키 중복"):
        load_confirmed_labels(frame)


def test_blank_business_name_raises():
    frame = _review_frame(세부사업명=["  ", "아이 돌봄", "노인 일자리", "미검토 사업"])
    with pytest.raises(ValueError, match="세부사업명이 비어 있는"):
        load_confirmed_labels(frame)


def test_build_qa_reports_status_distribution_and_yearly_counts():
    review = _review_frame()
    confirmed = load_confirmed_labels(review)

    qa = build_qa(review, confirmed)

    status_rows = qa.loc[qa["구분"].eq("전체_검토상태분포")]
    assert status_rows["건수"].sum() == len(review)
    year_rows = qa.loc[qa["구분"].eq("확정라벨_연도별건수")]
    assert dict(zip(year_rows["검토상태"], year_rows["건수"], strict=True)) == {2021: 2, 2022: 1}
