"""텍스트 정제 모듈에서 공통으로 사용하는 정규식 상수."""

from __future__ import annotations

import re


PUA_PATTERN = re.compile(r"[\uE000-\uF8FF]")

CONTENT_LABELS = (
    "지원대상",
    "지원내용",
    "사업대상",
    "사업내용",
    "주요내용",
    "전달체계",
    "목적",
    "대상",
    "내용",
)
CONTENT_LABEL_PATTERN = "|".join(CONTENT_LABELS)
PAREN_LABEL_PATTERN = re.compile(rf"\(\s*({CONTENT_LABEL_PATTERN})\s*\)")


__all__ = [
    "CONTENT_LABEL_PATTERN",
    "CONTENT_LABELS",
    "PAREN_LABEL_PATTERN",
    "PUA_PATTERN",
]
