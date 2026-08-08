import pandas as pd
import pytest
from kiwipiepy import Kiwi

from src.features.keyword_tfidf import (
    compare_rankings,
    extract_nouns,
    prepare_text_column,
    rank_group_keywords,
)


def test_extract_nouns_filters_budget_stopwords() -> None:
    nouns = extract_nouns("청년 주거 지원 사업 운영", kiwi=Kiwi())
    assert "청년" in nouns
    assert "주거" in nouns
    assert "지원" not in nouns
    assert "사업" not in nouns


def test_prepare_text_column_falls_back_and_counts_missing() -> None:
    frame = pd.DataFrame({"정제": ["돌봄 교실", None, None], "원문": ["x", "청년 고용", None]})
    text, stats = prepare_text_column(frame, preferred_column="정제", fallback_column="원문")
    assert text.tolist() == ["돌봄 교실", "청년 고용", ""]
    assert stats == {
        "전체행": 3,
        "우선텍스트사용": 1,
        "원문대체": 1,
        "분석제외": 1,
        "원문열존재": True,
    }


def test_rank_group_keywords_is_deterministic_and_supports_deduplication() -> None:
    frame = pd.DataFrame(
        {
            "세부영역": ["고용", "고용", "고용", "돌봄", "돌봄"],
            "텍스트": [
                "청년 일자리 취업 고용",
                "청년 일자리 취업 고용",
                "청년 취업 교육",
                "아동 보육 돌봄 교실",
                "아동 보육 돌봄 시설",
            ],
        }
    )
    baseline = rank_group_keywords(frame, text_column="텍스트", min_df=1, max_df=1.0, top_n=5)
    sensitivity = rank_group_keywords(
        frame, text_column="텍스트", min_df=1, max_df=1.0, top_n=5, deduplicate=True
    )
    assert set(baseline["세부영역"]) == {"고용", "돌봄"}
    assert baseline.groupby("세부영역")["순위"].min().eq(1).all()
    assert sensitivity.loc[sensitivity["세부영역"].eq("고용"), "분석문서수"].eq(2).all()
    assert "지원" not in set(baseline["단어"])
    comparison = compare_rankings(baseline, sensitivity)
    assert set(comparison["세부영역"]) == {"고용", "돌봄"}


def test_rank_group_keywords_rejects_empty_documents() -> None:
    with pytest.raises(ValueError, match="텍스트가 없습니다"):
        rank_group_keywords(
            pd.DataFrame({"세부영역": ["고용"], "텍스트": [None]}), text_column="텍스트"
        )
