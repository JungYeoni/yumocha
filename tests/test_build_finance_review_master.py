import pandas as pd

from scripts.build_finance_review_master import (
    add_similarity_groups,
    changed_cleaned_flags,
    normalize_business_name,
)


def test_normalize_business_name_removes_spacing_and_punctuation():
    assert normalize_business_name("출산 장려금(지원)") == "출산장려금지원"


def test_add_similarity_groups_places_equivalent_names_in_same_group():
    frame = pd.DataFrame(
        {
            "연도": [2020, 2021, 2022],
            "지역": ["서울", "부산", "대구"],
            "원본행": ["1", "2", "3"],
            "세부사업명": ["출산장려금 지원", "출산 장려금 지원사업", "노인 교통비 지원"],
        }
    )

    result = add_similarity_groups(frame, threshold=0.7)
    births = result[result["세부사업명"].str.contains("출산")]

    assert births["유사사업그룹ID"].nunique() == 1
    assert births["유사검토대상"].all()


def test_changed_cleaned_flags_excludes_new_rows_and_normalizes_blanks():
    result = changed_cleaned_flags(
        pd.Series(["기존", None, None]),
        pd.Series(["최신", "", "신규"]),
        pd.Series([False, False, True]),
    )

    assert result.tolist() == [True, False, False]
