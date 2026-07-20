"""세부사업명 유사도 비교와 주요내용 라벨 추출을 위한 공통 함수."""

from __future__ import annotations

import re
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz


MATCH_NORMALIZE_PATTERN = re.compile(r"[\s,./\-()]")
LABEL_PATTERN = r"지원대상|지원내용|사업대상|사업내용|주요내용|전달체계|목적|대상|내용"
PAREN_LABEL_PATTERN = re.compile(rf"\(\s*({LABEL_PATTERN})\s*\)")
BULLET_BEFORE_LABEL_PATTERN = re.compile(rf"[•·]\s*(?=({LABEL_PATTERN})\s*[:：])")
STRICT_SUPPORT_PATTERN = re.compile(
    r"^지원대상\s*[:：]\s*(.*?)\s*지원내용\s*[:：]\s*(.*)$",
    re.DOTALL,
)
BROAD_SUPPORT_PATTERN = re.compile(
    r"^(지원대상|사업대상|대상)\s*[:：]?\s*(.*?)\s*"
    r"(지원내용|사업내용|내용)\s*[:：]?\s*(.*)$",
    re.DOTALL,
)

NEAR_DUPLICATE_COLUMNS = [
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


def normalize_for_match(name: object) -> str:
    """사업명 비교를 위해 공백과 일부 문장부호를 제거한다."""
    if pd.isna(name):
        return ""
    return MATCH_NORMALIZE_PATTERN.sub("", str(name))


def find_near_duplicates(
    df: pd.DataFrame,
    threshold: int = 90,
) -> pd.DataFrame:
    """같은 지역·중분류 안에서 유사도가 높은 사업명 쌍을 찾는다.

    완전히 같은 사업명과 공백·문장부호만 다른 사업명은 제외한다. 반환값은
    검토 후보이며 동일 사업이라는 자동 판정으로 사용하지 않는다.
    """
    if not 0 <= threshold <= 100:
        raise ValueError("threshold는 0 이상 100 이하여야 합니다.")

    required_cols = {"지역", "중분류", "세부사업명", "당해예산", "주요내용"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"유사 사업명 비교에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

    candidates: list[dict[str, object]] = []

    for (sido, medium_category), group in df.groupby(
        ["지역", "중분류"],
        dropna=False,
    ):
        rows = group[["세부사업명", "당해예산", "주요내용"]].itertuples(
            index=False,
            name=None,
        )

        for row_a, row_b in combinations(rows, 2):
            name_a, budget_a, content_a = row_a
            name_b, budget_b, content_b = row_b

            if pd.isna(name_a) or pd.isna(name_b) or name_a == name_b:
                continue

            if normalize_for_match(name_a) == normalize_for_match(name_b):
                continue

            score = fuzz.ratio(str(name_a), str(name_b))
            if score < threshold:
                continue

            candidates.append(
                {
                    "지역": sido,
                    "중분류": medium_category,
                    "세부사업명1": name_a,
                    "당해예산1": budget_a,
                    "주요내용1": content_a,
                    "세부사업명2": name_b,
                    "당해예산2": budget_b,
                    "주요내용2": content_b,
                    "유사도": score,
                }
            )

    result = pd.DataFrame(candidates, columns=NEAR_DUPLICATE_COLUMNS)
    return result.sort_values("유사도", ascending=False, ignore_index=True)


def dedup_label(text: object) -> object:
    """괄호형 라벨과 연속으로 중복된 라벨을 최소한으로 정리한다."""
    if pd.isna(text):
        return text

    value = PAREN_LABEL_PATTERN.sub(r"\1 : ", str(text))
    value = BULLET_BEFORE_LABEL_PATTERN.sub("", value)

    for label in ("지원대상", "지원내용", "사업대상", "사업내용"):
        value = re.sub(
            rf"({label}\s*[:：]\s*)+",
            f"{label} : ",
            value,
        )

    return value.strip()


def check_pattern(text: object) -> str:
    """엄격한 지원대상·지원내용 라벨 패턴의 일치 여부를 반환한다."""
    if pd.isna(text):
        return "결측"
    return "일치" if STRICT_SUPPORT_PATTERN.match(str(text)) else "불일치"


def check_pattern_broad(text: object) -> str:
    """확장된 대상·내용 라벨 패턴의 일치 여부를 반환한다."""
    if pd.isna(text):
        return "결측"
    return "일치" if BROAD_SUPPORT_PATTERN.match(str(text)) else "불일치"


def extract_via_regex(text: object) -> dict[str, str | None]:
    """확장 라벨 패턴이 일치하면 지원대상과 지원내용을 분리한다."""
    empty_result = {"지원대상": None, "지원내용": None}
    if pd.isna(text):
        return empty_result

    match = BROAD_SUPPORT_PATTERN.match(str(text))
    if match is None:
        return empty_result

    return {
        "지원대상": match.group(2).strip(),
        "지원내용": match.group(4).strip(),
    }


__all__ = [
    "check_pattern",
    "check_pattern_broad",
    "dedup_label",
    "extract_via_regex",
    "find_near_duplicates",
    "normalize_for_match",
]
