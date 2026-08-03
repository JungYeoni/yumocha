"""재정팀 검토 산출물에서 공통으로 사용하는 행 키 정규화."""

from __future__ import annotations

import pandas as pd

KEY_COLUMNS = ["연도", "지역", "원본행"]


def normalize_source_row(value: object) -> str:
    """정수형 행 번호의 문자열 표현에서 불필요한 ``.0``을 제거한다."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_review_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """연도·지역·원본행을 모든 검토 스크립트에서 같은 형식으로 맞춘다."""
    output = frame.copy()
    output["연도"] = pd.to_numeric(output["연도"], errors="raise").astype("int64")
    output["지역"] = output["지역"].astype("string").str.strip()
    output["원본행"] = output["원본행"].map(normalize_source_row).astype("string")
    return output
