import pandas as pd
import pytest

from src.features.pipeline_common import build_subtotal_qa, calculate_budget_changes


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
    assert result_by_category.loc["돌봄", "결과"] == "일치"
    assert result_by_category.loc["고령", "차이"] == -1.0
    assert result_by_category.loc["고령", "결과"] == "불일치"
    assert result_by_category.loc["청년", "QA_병합상태"] == "원본소계만"
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


def test_build_subtotal_qa_rejects_duplicate_subtotals():
    source = make_qa_source()
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="중분류 소계가 여러 행"):
        build_subtotal_qa(source, budget_col="예산_num")


def test_build_subtotal_qa_validates_arguments():
    source = make_qa_source()

    with pytest.raises(ValueError, match="0 이상"):
        build_subtotal_qa(source, budget_col="예산_num", tolerance=-1)

    with pytest.raises(ValueError, match="하나 이상의 컬럼"):
        build_subtotal_qa(source, budget_col="예산_num", group_cols=())

    with pytest.raises(KeyError, match="필요한 컬럼"):
        build_subtotal_qa(source.drop(columns="중분류"), budget_col="예산_num")

    string_budget = source.assign(예산_num=source["예산_num"].astype("string"))
    with pytest.raises(TypeError, match="숫자형"):
        build_subtotal_qa(string_budget, budget_col="예산_num")
