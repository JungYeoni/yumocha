"""세부사업 텍스트의 한국어 명사 TF-IDF 핵심어를 집계한다."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

NOUN_TAGS = frozenset({"NNG", "NNP"})
DEFAULT_STOPWORDS = frozenset(
    {
        "지원",
        "사업",
        "운영",
        "확대",
        "추진",
        "제공",
        "관리",
        "서비스",
        "프로그램",
        "활성화",
        "강화",
        "조성",
        "개선",
        "실시",
        "대상",
        "관련",
        "지역",
        "센터",
        "내용",
        "이상",
        "이하",
        "미만",
        "초과",
        "기준",
        "경우",
    }
)
SPACE_PATTERN = re.compile(r"\s+")


def normalize_keyword_text(value: object) -> str:
    """텍스트를 NFKC와 단일 공백으로 정규화한다."""
    if value is None or pd.isna(value):
        return ""
    return SPACE_PATTERN.sub(" ", unicodedata.normalize("NFKC", str(value))).strip()


def extract_nouns(
    text: object,
    *,
    kiwi: Kiwi,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    min_length: int = 2,
) -> list[str]:
    """Kiwi 분석 결과에서 두 글자 이상의 일반·고유명사를 추출한다."""
    normalized = normalize_keyword_text(text)
    if not normalized:
        return []
    excluded = set(stopwords)
    return [
        token.form
        for token in kiwi.tokenize(normalized)
        if token.tag in NOUN_TAGS and len(token.form) >= min_length and token.form not in excluded
    ]


def prepare_text_column(
    frame: pd.DataFrame,
    *,
    preferred_column: str,
    fallback_column: str | None = None,
) -> tuple[pd.Series, dict[str, int | bool]]:
    """우선 텍스트를 사용하고 결측일 때만 선택적 원문으로 대체한다."""
    if preferred_column not in frame.columns:
        raise ValueError(f"텍스트 열 누락: {preferred_column}")
    preferred = frame[preferred_column].map(normalize_keyword_text)
    fallback = pd.Series("", index=frame.index, dtype="string")
    if fallback_column and fallback_column in frame.columns:
        fallback = frame[fallback_column].map(normalize_keyword_text).astype("string")
    fallback_mask = preferred.eq("") & fallback.ne("")
    output = preferred.mask(fallback_mask, fallback)
    stats = {
        "전체행": len(frame),
        "우선텍스트사용": int(preferred.ne("").sum()),
        "원문대체": int(fallback_mask.sum()),
        "분석제외": int(output.eq("").sum()),
        "원문열존재": bool(fallback_column and fallback_column in frame.columns),
    }
    return output.astype("string"), stats


def rank_group_keywords(
    frame: pd.DataFrame,
    *,
    text_column: str,
    group_column: str = "세부영역",
    top_n: int = 20,
    min_df: int = 2,
    max_df: float = 0.95,
    sublinear_tf: bool = True,
    deduplicate: bool = False,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
    kiwi: Kiwi | None = None,
) -> pd.DataFrame:
    """사업별 TF-IDF를 계산한 뒤 세부영역별 평균 점수 상위어를 반환한다."""
    missing = [column for column in (group_column, text_column) if column not in frame.columns]
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")
    if top_n < 1 or min_df < 1:
        raise ValueError("top_n과 min_df는 1 이상이어야 합니다.")

    working = frame[[group_column, text_column]].copy()
    working[text_column] = working[text_column].map(normalize_keyword_text)
    working = working.loc[working[group_column].notna() & working[text_column].ne("")]
    if deduplicate:
        working = working.drop_duplicates([group_column, text_column])
    if working.empty:
        raise ValueError("TF-IDF를 계산할 텍스트가 없습니다.")

    analyzer = kiwi or Kiwi()
    tokenized = [
        extract_nouns(value, kiwi=analyzer, stopwords=stopwords) for value in working[text_column]
    ]
    valid_mask = np.fromiter((bool(tokens) for tokens in tokenized), dtype=bool)
    working = working.loc[valid_mask].reset_index(drop=True)
    tokenized = [tokens for tokens, valid in zip(tokenized, valid_mask, strict=True) if valid]
    if not tokenized:
        raise ValueError("명사 추출 후 남은 문서가 없습니다.")

    vectorizer = TfidfVectorizer(
        analyzer=lambda document: document,
        lowercase=False,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(tokenized)
    terms = vectorizer.get_feature_names_out()
    rows: list[dict[str, object]] = []
    for group in sorted(working[group_column].unique()):
        mask = working[group_column].eq(group).to_numpy()
        scores = np.asarray(matrix[mask].mean(axis=0)).ravel()
        document_frequency = np.asarray((matrix[mask] > 0).sum(axis=0)).ravel()
        order = np.lexsort((terms, -scores))[:top_n]
        for rank, position in enumerate(order, start=1):
            if scores[position] <= 0:
                continue
            rows.append(
                {
                    group_column: group,
                    "순위": rank,
                    "단어": terms[position],
                    "평균_TFIDF": float(scores[position]),
                    "등장문서수": int(document_frequency[position]),
                    "분석문서수": int(mask.sum()),
                    "중복제거": deduplicate,
                }
            )
    return pd.DataFrame(rows)


def compare_rankings(baseline: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    """기본 결과와 중복 제거 결과의 상위어 중첩 및 순위 상관을 요약한다."""
    records: list[dict[str, object]] = []
    for group in sorted(set(baseline["세부영역"]) | set(sensitivity["세부영역"])):
        left = baseline.loc[baseline["세부영역"].eq(group), ["단어", "순위"]]
        right = sensitivity.loc[sensitivity["세부영역"].eq(group), ["단어", "순위"]]
        merged = left.merge(right, on="단어", suffixes=("_기본", "_중복제거"))
        denominator = max(len(left), len(right), 1)
        correlation = merged["순위_기본"].corr(merged["순위_중복제거"], method="spearman")
        records.append(
            {
                "세부영역": group,
                "공통단어수": len(merged),
                "상위어_중첩률": len(merged) / denominator,
                "공통단어_순위상관": None if pd.isna(correlation) else float(correlation),
            }
        )
    return pd.DataFrame(records)
