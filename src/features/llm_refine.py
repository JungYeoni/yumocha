"""LLM 보존형 텍스트 교정과 결과 검증을 위한 공통 함수."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)

PUA_PATTERN = re.compile(r"[\uE000-\uF8FF]")
PAREN_LABEL_PATTERN = re.compile(
    r"\((지원대상|지원내용|사업대상|사업내용|주요내용|전달체계|목적|대상|내용)\)"
)
NUMBER_PATTERN = re.compile(r"\d+")


def call_llm_once(
    name: str,
    content: str,
    *,
    client: Any,
    llm_config: Mapping[str, Any],
) -> str | None:
    """LLM API를 한 번 호출하고 파싱된 정제 문장을 반환한다.

    응답이 JSON이 아니거나 ``정제문장`` 키가 없으면 ``None``을 반환한다.
    네트워크 오류 등 API 호출 자체의 예외는 재시도 여부를 결정할 수 있도록
    호출자에게 전달한다.
    """
    prompt = llm_config["prompt"]["template"].format(
        name=name,
        content=content,
    )
    response_format = {
        "type": "json_schema",
        "json_schema": llm_config["response_schema"],
    }

    response = client.chat.completions.create(
        model=llm_config["upstage"]["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=llm_config["upstage"]["temperature"],
        response_format=response_format,
    )
    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
        cleaned = parsed["정제문장"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    return cleaned if isinstance(cleaned, str) else None


def clean_sentence(
    name: str,
    content: object,
    *,
    call_once: Callable[[str, str], str | None],
    max_attempts: int = 2,
) -> str | None:
    """주요내용을 교정하고 반복 실패 시 원문을 반환한다.

    결측값은 호출하지 않고 ``None``으로 유지한다. ``max_attempts=2``는 기존
    노트북의 최초 호출 1회와 재시도 1회 동작에 해당한다.
    """
    if pd.isna(content):
        return None

    if max_attempts < 1:
        raise ValueError("max_attempts는 1 이상이어야 합니다.")

    original = str(content)

    for attempt in range(1, max_attempts + 1):
        try:
            result = call_once(name, original)
        except Exception as error:
            LOGGER.warning(
                "LLM API 호출 실패(%s회차): %s -> %s",
                attempt,
                name,
                error,
            )
            result = None

        if result is not None:
            return result

    LOGGER.warning("LLM 정제 실패, 원문 유지: %s", name)
    return original


def extract_numbers(text: object) -> list[str]:
    """텍스트에서 숫자 시퀀스를 등장 순서대로 추출한다."""
    if pd.isna(text):
        return []
    return NUMBER_PATTERN.findall(str(text))


def numbers_preserved(original: object, cleaned: object) -> bool:
    """교정 전후의 숫자 시퀀스와 순서가 동일한지 확인한다."""
    return extract_numbers(original) == extract_numbers(cleaned)


def needs_llm_rerun(text: object) -> bool:
    """PUA 문자나 괄호형 라벨이 남은 텍스트인지 확인한다."""
    if pd.isna(text):
        return False

    value = str(text)
    return bool(PUA_PATTERN.search(value) or PAREN_LABEL_PATTERN.search(value))


__all__ = [
    "call_llm_once",
    "clean_sentence",
    "extract_numbers",
    "needs_llm_rerun",
    "numbers_preserved",
]
