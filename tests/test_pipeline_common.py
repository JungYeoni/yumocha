import re

import pandas as pd
import pytest

from src.features.pipeline_common import (
    UNIT_NOTATION_PATTERN,
    assign_labels,
    build_subtotal_qa,
    calculate_budget_changes,
    classify_row,
    clean_text,
    get_sido_dir,
    normalize_budget_type,
    show_table1_around,
    to_numeric_budget,
)


@pytest.mark.parametrize(
    ("detail_name", "expected"),
    [
        (None, "헤더반복"),
        ("세부사업명", "헤더반복"),
        ("붙임(서울)", "헤더반복"),
        ("Ⅰ. 공통사업", "대분류_소계"),
        ("1. 함께 돌보는 사회(공통)", "중분류_소계"),
        ("아이돌봄 지원", "세부사업"),
    ],
)
def test_classify_row_classifies_standard_rows(detail_name, expected):
    assert classify_row(detail_name) == expected


def test_classify_row_accepts_extra_header_pattern():
    assert (
        classify_row("2023년 시행계획", extra_header_patterns=[re.compile(r"시행계획$")])
        == "헤더반복"
    )


@pytest.mark.parametrize(
    ("detail_name", "expected"),
    [
        ("(단위 : 백만원)", "헤더반복"),
        ("(단위：백만원)", "헤더반복"),
        ("(단위:백만원)", "헤더반복"),
        # "단위"를 포함하지만 단위표기 행이 아닌 정상 세부사업명은 오분류되면 안 된다
        ("직장단위 결혼장려 만남의 장", "세부사업"),
        ("면단위 공영목욕장 운영 지원", "세부사업"),
    ],
)
def test_classify_row_filters_unit_notation_via_extra_header_pattern(detail_name, expected):
    assert classify_row(detail_name, extra_header_patterns=[UNIT_NOTATION_PATTERN]) == expected


def test_assign_labels_propagates_hierarchy_in_original_row_order():
    source = pd.DataFrame(
        {
            "원본행": [3, 1, 2],
            "사업행구분": ["세부사업", "대분류_소계", "중분류_소계"],
            "세부사업명": ["아이돌봄", "Ⅰ. 공통사업", "1. 돌봄(공통)"],
        }
    )

    result = assign_labels(source)

    assert result["원본행"].tolist() == [1, 2, 3]
    assert result.loc[result["원본행"].eq(3), "대분류"].item() == "Ⅰ. 공통사업"
    assert result.loc[result["원본행"].eq(3), "중분류"].item() == "1. 돌봄(공통)"


def test_assign_labels_rejects_missing_columns():
    with pytest.raises(KeyError, match="필요한 컬럼"):
        assign_labels(pd.DataFrame({"세부사업명": ["사업"]}))


def test_clean_text_normalizes_whitespace_pua_and_leading_bullet():
    source = pd.Series(["  ㅇ지원\n 대상  ", "노인 복지관", "\ue000문장", pd.NA])

    result = clean_text(source, strip_leading_bullet=True)

    assert result.iloc[0] == "지원 대상"
    assert result.iloc[1] == "노인 복지관"
    assert result.iloc[2] == "문장"
    assert pd.isna(result.iloc[3])


def test_normalize_budget_type_removes_all_whitespace():
    result = normalize_budget_type(pd.Series([" 공 통 ", "자\n체", pd.NA]))

    assert result.iloc[0] == "공통"
    assert result.iloc[1] == "자체"
    assert pd.isna(result.iloc[2])


def test_to_numeric_budget_handles_commas_zero_tokens_and_missing_values():
    source = pd.Series(["1,234", "-", "비예산", pd.NA])

    result = to_numeric_budget(source, zero_tokens=("-", "비예산"))

    assert result.iloc[0] == 1234
    assert result.iloc[1] == 0
    assert result.iloc[2] == 0
    assert pd.isna(result.iloc[3])


def test_get_sido_dir_creates_and_returns_directory(tmp_path):
    result = get_sido_dir(tmp_path, "서울")

    assert result == tmp_path / "서울"
    assert result.is_dir()


def test_show_table1_around_returns_requested_window_and_columns():
    source = pd.DataFrame(
        {
            "사업": ["A", "B", "C", "D", "E"],
            "빈칸": [None] * 5,
            "구분": ["공통"] * 5,
            "예산": [1, 2, 3, 4, 5],
        }
    )

    result = show_table1_around(source, 3, window=1)

    assert result["세부사업명"].tolist() == ["B", "C", "D"]
    assert result.columns.tolist() == ["세부사업명", "공통/자체", "예산"]


def test_show_table1_around_validates_arguments():
    source = pd.DataFrame([[1, 2, 3, 4]])

    with pytest.raises(ValueError, match="1 이상"):
        show_table1_around(source, 0)
    with pytest.raises(ValueError, match="0 이상"):
        show_table1_around(source, 1, window=-1)
    with pytest.raises(ValueError, match="길이가 같아야"):
        show_table1_around(source, 1, column_indices=(0,), column_names=("A", "B"))


def test_calculate_budget_changes_handles_increase_decrease_and_no_change():
    current = pd.Series([120.0, 80.0, 100.0])
    previous = pd.Series([100.0, 100.0, 100.0])

    result = calculate_budget_changes(current, previous)

    assert result["증감액"].tolist() == [20.0, -20.0, 0.0]
    assert result["증감율"].tolist() == [20.0, -20.0, 0.0]


def test_calculate_budget_changes_keeps_zero_and_missing_rates_missing():
    current = pd.Series([100.0, None, 50.0])
    previous = pd.Series([0.0, 100.0, None])

    result = calculate_budget_changes(current, previous)

    assert result["증감액"].iloc[0] == 100.0
    assert pd.isna(result["증감율"].iloc[0])
    assert pd.isna(result["증감액"].iloc[1])
    assert pd.isna(result["증감율"].iloc[1])
    assert pd.isna(result["증감액"].iloc[2])
    assert pd.isna(result["증감율"].iloc[2])


def test_calculate_budget_changes_rounds_rate():
    current = pd.Series([2.0])
    previous = pd.Series([3.0])

    result = calculate_budget_changes(current, previous, rate_digits=2)

    assert result.loc[0, "증감액"] == -1.0
    assert result.loc[0, "증감율"] == -33.33


def test_calculate_budget_changes_preserves_index():
    index = pd.Index(["사업A", "사업B"])
    current = pd.Series([120, 80], index=index)
    previous = pd.Series([100, 100], index=index)

    result = calculate_budget_changes(current, previous)

    assert result.index.equals(index)


def test_calculate_budget_changes_rejects_different_indices():
    current = pd.Series([100], index=[0])
    previous = pd.Series([100], index=[1])

    with pytest.raises(ValueError, match="인덱스가 같아야"):
        calculate_budget_changes(current, previous)


def test_calculate_budget_changes_rejects_negative_digits():
    with pytest.raises(ValueError, match="0 이상"):
        calculate_budget_changes(
            pd.Series([100]),
            pd.Series([100]),
            rate_digits=-1,
        )


def make_qa_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "지역": ["서울", "서울", "서울", "서울", "서울", "서울"],
            "대분류": ["공통", "공통", "공통", "공통", "공통", "공통"],
            "중분류": ["돌봄", "돌봄", "돌봄", "고령", "고령", "청년"],
            "사업행구분": [
                "중분류_소계",
                "세부사업",
                "세부사업",
                "중분류_소계",
                "세부사업",
                "중분류_소계",
            ],
            "예산_num": [100.0, 40.0, 60.0, 51.0, 50.0, 30.0],
        }
    )


def test_build_subtotal_qa_compares_subtotal_and_leaf_sum():
    result = build_subtotal_qa(make_qa_source(), budget_col="예산_num")

    result_by_category = result.set_index("중분류")
    assert result_by_category.loc["돌봄", "leaf_합계"] == 100.0
    assert result_by_category.loc["돌봄", "차이"] == 0.0
    assert result_by_category.loc["돌봄", "오차율(%)"] == 0.0
    assert result_by_category.loc["돌봄", "결과"] == "일치"
    assert result_by_category.loc["고령", "차이"] == -1.0
    assert result_by_category.loc["고령", "오차율(%)"] == -1.96
    assert result_by_category.loc["고령", "결과"] == "불일치"
    assert result_by_category.loc["청년", "QA_병합상태"] == "원본소계만"
    assert pd.isna(result_by_category.loc["청년", "오차율(%)"])
    assert result_by_category.loc["청년", "결과"] == "불일치"


def test_build_subtotal_qa_applies_tolerance():
    result = build_subtotal_qa(
        make_qa_source(),
        budget_col="예산_num",
        tolerance=1,
    ).set_index("중분류")

    assert result.loc["고령", "결과"] == "일치"


def test_build_subtotal_qa_marks_leaf_only_group_as_mismatch():
    source = make_qa_source()
    leaf_only = pd.DataFrame(
        {
            "지역": ["서울"],
            "대분류": ["공통"],
            "중분류": ["주거"],
            "사업행구분": ["세부사업"],
            "예산_num": [20.0],
        }
    )
    source = pd.concat([source, leaf_only], ignore_index=True)

    result = build_subtotal_qa(source, budget_col="예산_num").set_index("중분류")

    assert result.loc["주거", "QA_병합상태"] == "leaf합계만"
    assert result.loc["주거", "결과"] == "불일치"


def test_build_subtotal_qa_keeps_error_rate_missing_for_zero_subtotal():
    source = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "대분류": ["공통", "공통"],
            "중분류": ["돌봄", "돌봄"],
            "사업행구분": ["중분류_소계", "세부사업"],
            "예산_num": [0.0, 10.0],
        }
    )

    result = build_subtotal_qa(source, budget_col="예산_num")

    assert result.loc[0, "차이"] == 10.0
    assert pd.isna(result.loc[0, "오차율(%)"])
    assert result.loc[0, "결과"] == "불일치"
    assert result.loc[0, "허용기준결과"] == "판정불가"


@pytest.mark.parametrize(
    ("leaf_budget", "expected_rate", "expected_result"),
    [
        (110.0, 10.0, "허용"),
        (90.0, -10.0, "허용"),
        (110.01, 10.01, "초과"),
        (89.99, -10.01, "초과"),
    ],
)
def test_build_subtotal_qa_applies_absolute_rate_tolerance(
    leaf_budget,
    expected_rate,
    expected_result,
):
    source = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "대분류": ["공통", "공통"],
            "중분류": ["돌봄", "돌봄"],
            "사업행구분": ["중분류_소계", "세부사업"],
            "예산_num": [100.0, leaf_budget],
        }
    )

    result = build_subtotal_qa(source, budget_col="예산_num")

    assert result.loc[0, "오차율(%)"] == expected_rate
    assert result.loc[0, "허용기준결과"] == expected_result


def test_build_subtotal_qa_supports_custom_rate_tolerance():
    source = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "대분류": ["공통", "공통"],
            "중분류": ["돌봄", "돌봄"],
            "사업행구분": ["중분류_소계", "세부사업"],
            "예산_num": [100.0, 106.0],
        }
    )

    result = build_subtotal_qa(
        source,
        budget_col="예산_num",
        rate_tolerance=5.0,
    )

    assert result.loc[0, "오차율(%)"] == 6.0
    assert result.loc[0, "허용기준결과"] == "초과"


def test_build_subtotal_qa_rejects_duplicate_subtotals():
    source = make_qa_source()
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="중분류 소계가 여러 행"):
        build_subtotal_qa(source, budget_col="예산_num")


def test_build_subtotal_qa_validates_arguments():
    source = make_qa_source()

    with pytest.raises(ValueError, match="0 이상"):
        build_subtotal_qa(source, budget_col="예산_num", tolerance=-1)

    with pytest.raises(ValueError, match="rate_tolerance는 0 이상"):
        build_subtotal_qa(source, budget_col="예산_num", rate_tolerance=-1)

    with pytest.raises(ValueError, match="하나 이상의 컬럼"):
        build_subtotal_qa(source, budget_col="예산_num", group_cols=())

    with pytest.raises(KeyError, match="필요한 컬럼"):
        build_subtotal_qa(source.drop(columns="중분류"), budget_col="예산_num")

    string_budget = source.assign(예산_num=source["예산_num"].astype("string"))
    with pytest.raises(TypeError, match="숫자형"):
        build_subtotal_qa(string_budget, budget_col="예산_num")
