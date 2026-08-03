"""세부사업명·주요내용 유사도 그룹핑 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from src.modeling.similarity_grouping import (
    CONTENT_GROUP_COLUMN,
    NAME_GROUP_COLUMN,
    NAME_GROUP_SCORE_COLUMN,
    UNGROUPED_SIMILARITY_SCORE,
    assign_similarity_groups,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "연도": [2016, 2018, 2020, 2019],
            "지역": ["부산", "서울", "서울", "서울"],
            "원본행": [1, 1, 2, 3],
            "세부사업명": [
                "청소년 방과후 아카데미",
                "청년창업지원사업",
                "청년창업지원금",
                "노인일자리사업",
            ],
            "주요내용_정제": [
                "청소년 대상 방과후 프로그램",
                "청년 대상 창업자금 지원",
                "청년 대상 창업 자금 지원사업",
                "노인 대상 일자리 연계",
            ],
        }
    )


def test_similar_names_are_grouped_together_and_dissimilar_names_are_not():
    result = assign_similarity_groups(_frame())
    seoul = result.loc[result["지역"].eq("서울")].set_index("세부사업명")

    assert (
        seoul.loc["청년창업지원사업", NAME_GROUP_COLUMN]
        == seoul.loc["청년창업지원금", NAME_GROUP_COLUMN]
    )
    assert (
        seoul.loc["청년창업지원사업", NAME_GROUP_COLUMN]
        != seoul.loc["노인일자리사업", NAME_GROUP_COLUMN]
    )
    assert seoul.loc["노인일자리사업", NAME_GROUP_SCORE_COLUMN] == UNGROUPED_SIMILARITY_SCORE


def test_content_subgroup_is_scoped_within_name_group():
    result = assign_similarity_groups(_frame())
    seoul = result.loc[result["지역"].eq("서울")].set_index("세부사업명")

    grouped_content = seoul.loc[["청년창업지원사업", "청년창업지원금"], CONTENT_GROUP_COLUMN]
    assert grouped_content.nunique() == 1
    assert grouped_content.iloc[0].startswith(str(seoul.loc["청년창업지원사업", NAME_GROUP_COLUMN]))


def test_higher_similarity_group_is_ordered_before_singletons_within_region():
    result = assign_similarity_groups(_frame())
    seoul_names = result.loc[result["지역"].eq("서울"), "세부사업명"].tolist()

    assert seoul_names.index("청년창업지원사업") < seoul_names.index("노인일자리사업")
    assert seoul_names.index("청년창업지원금") < seoul_names.index("노인일자리사업")


def test_similar_groups_are_ordered_before_singletons_across_regions():
    result = assign_similarity_groups(_frame())
    assert result["세부사업명"].tolist()[:2] == ["청년창업지원사업", "청년창업지원금"]


def test_nationwide_group_is_then_sorted_by_region_year_and_source_row():
    frame = pd.DataFrame(
        {
            "연도": [2020, 2018, 2019, 2017],
            "지역": ["부산", "서울", "서울", "부산"],
            "원본행": [2, 3, 1, 1],
            "세부사업명": ["청년창업지원사업"] * 4,
            "주요내용_정제": ["청년 창업 지원"] * 4,
        }
    )

    result = assign_similarity_groups(frame)

    assert result[NAME_GROUP_COLUMN].nunique() == 1
    assert list(result[["지역", "연도", "원본행"]].itertuples(index=False, name=None)) == [
        ("서울", 2018, 3),
        ("서울", 2019, 1),
        ("부산", 2017, 1),
        ("부산", 2020, 2),
    ]


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="필수 열 누락"):
        assign_similarity_groups(pd.DataFrame({"지역": ["서울"]}))


def test_empty_frame_raises():
    empty = pd.DataFrame(columns=["지역", "세부사업명", "주요내용_정제"])
    with pytest.raises(ValueError, match="비어 있습니다"):
        assign_similarity_groups(empty)


def test_blank_business_name_raises():
    frame = _frame()
    frame.loc[0, "세부사업명"] = "   "
    with pytest.raises(ValueError, match="세부사업명이 비어 있는"):
        assign_similarity_groups(frame)


def test_unknown_region_raises():
    frame = _frame()
    frame.loc[0, "지역"] = "평양"
    with pytest.raises(ValueError, match="REGION_ORDER에 없는"):
        assign_similarity_groups(frame)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match="사업명 유사도 임계값"):
        assign_similarity_groups(_frame(), name_similarity_threshold=1.5)


def test_blank_contents_in_same_name_group_remain_separate_subgroups():
    frame = pd.DataFrame(
        {
            "연도": [2019, 2020],
            "지역": ["서울", "부산"],
            "원본행": [1, 1],
            "세부사업명": ["동일 사업", "동일 사업"],
            "주요내용_정제": ["", pd.NA],
        }
    )

    result = assign_similarity_groups(frame)

    assert result[NAME_GROUP_COLUMN].nunique() == 1
    assert result[CONTENT_GROUP_COLUMN].nunique() == 2
