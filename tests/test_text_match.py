import pandas as pd
import pytest

from src.features.text_match import (
    check_pattern,
    check_pattern_broad,
    dedup_label,
    extract_via_regex,
    find_near_duplicates,
    normalize_for_match,
)


def test_normalize_for_match_removes_spacing_and_punctuation():
    assert normalize_for_match("노인 복지관-운영(지원)") == "노인복지관운영지원"
    assert normalize_for_match(None) == ""


def test_find_near_duplicates_returns_only_high_similarity_pairs():
    source = pd.DataFrame(
        {
            "지역": ["서울", "서울", "서울", "부산"],
            "중분류": ["돌봄", "돌봄", "돌봄", "돌봄"],
            "세부사업명": [
                "노인복지관 운영 지원",
                "노인복지관 운영지원 확대",
                "청년 월세 지원",
                "노인복지관 운영지원 확대",
            ],
            "당해예산": [10, 20, 30, 40],
            "주요내용": ["A", "B", "C", "D"],
        }
    )

    result = find_near_duplicates(source, threshold=80)

    assert len(result) == 1
    assert result.loc[0, "지역"] == "서울"
    assert result.loc[0, "세부사업명1"] == "노인복지관 운영 지원"
    assert result.loc[0, "세부사업명2"] == "노인복지관 운영지원 확대"


def test_find_near_duplicates_excludes_spacing_only_difference():
    source = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "중분류": ["돌봄", "돌봄"],
            "세부사업명": ["노인 복지관 운영", "노인복지관운영"],
            "당해예산": [10, 20],
            "주요내용": ["A", "B"],
        }
    )

    result = find_near_duplicates(source)

    assert result.empty
    assert result.columns.tolist() == [
        "지역",
        "중분류",
        "세부사업명1",
        "당해예산1",
        "주요내용1",
        "세부사업명2",
        "당해예산2",
        "주요내용2",
        "유사도",
    ]


def test_find_near_duplicates_validates_input():
    with pytest.raises(KeyError, match="필요한 컬럼"):
        find_near_duplicates(pd.DataFrame({"지역": ["서울"]}))

    with pytest.raises(ValueError, match="0 이상 100 이하"):
        find_near_duplicates(pd.DataFrame(), threshold=101)


def test_dedup_label_normalizes_parentheses_bullets_and_duplicates():
    text = "• 지원대상: 지원대상： 서울시민 (지원내용) 돌봄 제공"

    result = dedup_label(text)

    assert result == "지원대상 : 서울시민 지원내용 : 돌봄 제공"
    assert pd.isna(dedup_label(pd.NA))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, "결측"),
        ("지원대상: 서울시민 지원내용: 돌봄 제공", "일치"),
        ("사업대상: 서울시민 사업내용: 돌봄 제공", "불일치"),
    ],
)
def test_check_pattern(text, expected):
    assert check_pattern(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, "결측"),
        ("지원대상: 서울시민 지원내용: 돌봄 제공", "일치"),
        ("사업대상 서울시민 사업내용 돌봄 제공", "일치"),
        ("대상: 서울시민 내용: 돌봄 제공", "일치"),
        ("서울시민에게 돌봄 제공", "불일치"),
    ],
)
def test_check_pattern_broad(text, expected):
    assert check_pattern_broad(text) == expected


def test_extract_via_regex_extracts_labeled_content():
    assert extract_via_regex("사업대상: 서울시민 사업내용: 돌봄 제공") == {
        "지원대상": "서울시민",
        "지원내용": "돌봄 제공",
    }
    assert extract_via_regex("라벨 없는 문장") == {
        "지원대상": None,
        "지원내용": None,
    }
    assert extract_via_regex(None) == {
        "지원대상": None,
        "지원내용": None,
    }
