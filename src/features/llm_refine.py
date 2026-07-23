"""LLM 보존형 텍스트 교정과 결과 검증을 위한 공통 함수."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.text_patterns import PAREN_LABEL_PATTERN, PUA_PATTERN

LOGGER = logging.getLogger(__name__)

NUMBER_PATTERN = re.compile(r"\d+")
QUANTITY_PATTERN = re.compile(
    r"[+-]?\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:~|∼|～|-|–|—)\s*[+-]?\d[\d,]*(?:\.\d+)?)?"
    r"\s*(?:%|퍼센트|원|천원|만원|백만원|억원|조원|명|가구|세대|개소|개|회|개월|년|월|일|세)?"
)
QUOTED_TERM_PATTERN = re.compile(r"[「『\"“‘]([^」』\"”’]{2,})[」』\"”’]")
PROPER_NAME_TOKEN_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9·]+(?:특별자치도|특별자치시|특별시|광역시|도청|시청|군청|구청|"
    r"위원회|복지관|보건소|대학교|대학|학교|병원|의원|재단|공단|공사|협회|센터|연구원)"
)


@dataclass(frozen=True)
class RefinementResult:
    """LLM 교정 결과와 재개·감사를 위한 최소 상태."""

    cleaned_text: str | None
    status: str
    attempts: int
    error_type: str
    violations: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointRunSummary:
    """체크포인트 기반 LLM 실행 집계."""

    total_rows: int
    llm_target_rows: int
    reused_rows: int
    new_called_rows: int
    new_success_rows: int
    failed_rows: int
    held_rows: int
    rerun_rows: int


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
    validator: Callable[[str, str], tuple[str, ...]] | None = None,
) -> str | None:
    """주요내용을 교정하고 반복 실패 시 원문을 반환한다.

    결측값은 호출하지 않고 ``None``으로 유지한다. ``max_attempts=2``는 기존
    노트북의 최초 호출 1회와 재시도 1회 동작에 해당한다.
    """
    return refine_sentence(
        name,
        content,
        call_once=call_once,
        max_attempts=max_attempts,
        validator=validator,
    ).cleaned_text


def extract_numbers(text: object) -> list[str]:
    """텍스트에서 숫자 시퀀스를 등장 순서대로 추출한다."""
    if pd.isna(text):
        return []
    return NUMBER_PATTERN.findall(str(text))


def numbers_preserved(original: object, cleaned: object) -> bool:
    """교정 전후의 숫자 시퀀스와 순서가 동일한지 확인한다."""
    return extract_numbers(original) == extract_numbers(cleaned)


def extract_quantities(text: object) -> list[str]:
    """금액·퍼센트·범위를 포함한 수량 표현을 공백 정규화 후 추출한다."""
    if pd.isna(text):
        return []
    return [re.sub(r"\s+", "", token) for token in QUANTITY_PATTERN.findall(str(text))]


def extract_proper_names(
    text: object,
    *,
    context_terms: tuple[str, ...] = (),
) -> list[str]:
    """따옴표 표기와 기관명 접미사를 이용해 보존할 고유명사 후보를 추출한다."""
    if pd.isna(text):
        return []

    value = str(text)
    candidates = [*QUOTED_TERM_PATTERN.findall(value), *PROPER_NAME_TOKEN_PATTERN.findall(value)]
    candidates.extend(term for term in context_terms if term and term in value)
    return list(dict.fromkeys(candidates))


def preservation_violations(
    original: object,
    cleaned: object,
    *,
    context_terms: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """빈 결과와 숫자·수량·고유명사 보존 위반 사유를 반환한다."""
    if pd.isna(original):
        return () if pd.isna(cleaned) else ("결측 원문 변경",)
    if pd.isna(cleaned) or not str(cleaned).strip():
        return ("빈 결과",)

    violations: list[str] = []
    if not numbers_preserved(original, cleaned):
        violations.append("숫자 불일치")
    if extract_quantities(original) != extract_quantities(cleaned):
        violations.append("금액·퍼센트·범위 불일치")

    cleaned_value = str(cleaned)
    missing_names = [
        term
        for term in extract_proper_names(original, context_terms=context_terms)
        if term not in cleaned_value
    ]
    if missing_names:
        violations.append("고유명사 불일치")

    return tuple(violations)


def refine_sentence(
    name: str,
    content: object,
    *,
    call_once: Callable[[str, str], str | None],
    max_attempts: int = 2,
    validator: Callable[[str, str], tuple[str, ...]] | None = None,
) -> RefinementResult:
    """LLM을 재시도하고 성공·결측·보존위반·실패 상태를 구조화해 반환한다."""
    if pd.isna(content):
        return RefinementResult(None, "결측", 0, "", ())
    if max_attempts < 1:
        raise ValueError("max_attempts는 1 이상이어야 합니다.")

    original = str(content)
    last_error_type = ""
    last_violations: tuple[str, ...] = ()

    for attempt in range(1, max_attempts + 1):
        try:
            result = call_once(name, original)
        except Exception as error:
            last_error_type = type(error).__name__
            LOGGER.warning(
                "LLM API 호출 실패(%s회차): %s -> %s",
                attempt,
                name,
                last_error_type,
            )
            continue

        if result is None:
            last_error_type = "InvalidResponse"
            continue

        last_violations = validator(original, result) if validator else ()
        if not last_violations:
            return RefinementResult(result, "성공", attempt, "", ())

        last_error_type = "PreservationViolation"
        LOGGER.warning(
            "LLM 보존 검사 실패(%s회차): %s -> %s",
            attempt,
            name,
            ", ".join(last_violations),
        )

    if last_violations:
        LOGGER.warning("LLM 보존 위반, 원문 유지: %s", name)
        return RefinementResult(
            original,
            "보존위반",
            max_attempts,
            last_error_type,
            last_violations,
        )

    LOGGER.warning("LLM 정제 실패, 원문 유지: %s", name)
    return RefinementResult(original, "실패", max_attempts, last_error_type, ())


def run_checkpointed_refinement(
    df: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    call_once: Callable[[str, str], str | None],
    identity_cols: tuple[str, ...] = ("지역", "원본행", "세부사업명"),
    name_col: str = "세부사업명",
    content_col: str = "주요내용",
    cleaned_col: str = "주요내용_정제",
    max_attempts: int = 2,
    max_workers: int = 6,
    chunk_size: int = 50,
    min_call_interval_seconds: float = 0,
) -> tuple[pd.DataFrame, CheckpointRunSummary]:
    """완료 행을 재사용하고 대상만 병렬 호출하며 청크별 체크포인트를 저장한다."""
    if max_workers < 1 or chunk_size < 1:
        raise ValueError("max_workers와 chunk_size는 1 이상이어야 합니다.")
    if min_call_interval_seconds < 0:
        raise ValueError("min_call_interval_seconds는 0 이상이어야 합니다.")

    required_cols = {*identity_cols, name_col, content_col}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"LLM 정제에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")
    if df.index.has_duplicates:
        raise ValueError("LLM 정제 입력 인덱스에 중복이 있습니다.")

    path = Path(checkpoint_path)
    status_col = "LLM_상태"
    attempts_col = "LLM_시도횟수"
    error_col = "LLM_오류유형"
    violations_col = "LLM_보존위반"
    result_cols = [
        *identity_cols,
        cleaned_col,
        status_col,
        attempts_col,
        error_col,
        violations_col,
    ]
    rerun_rows = 0

    if path.exists():
        checkpoint_df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
        required_checkpoint_cols = {*identity_cols, cleaned_col}
        missing_checkpoint_cols = required_checkpoint_cols.difference(checkpoint_df.columns)
        if missing_checkpoint_cols:
            raise KeyError(
                "체크포인트 신원·결과 컬럼이 없습니다: "
                f"{sorted(missing_checkpoint_cols)}"
            )
        checkpoint_df.index = pd.to_numeric(checkpoint_df.index, errors="raise").astype(int)
        checkpoint_df = checkpoint_df.loc[~checkpoint_df.index.duplicated(keep="last")]
        checkpoint_df = checkpoint_df.loc[checkpoint_df.index.intersection(df.index)].copy()

        for column, default in (
            (status_col, "기존완료"),
            (attempts_col, 0),
            (error_col, ""),
            (violations_col, ""),
        ):
            if column not in checkpoint_df:
                checkpoint_df[column] = default

        current_identity = df.loc[checkpoint_df.index, list(identity_cols)].astype("string")
        saved_identity = checkpoint_df.loc[:, list(identity_cols)].astype("string")
        identity_matches = current_identity.fillna("<NA>").eq(
            saved_identity.fillna("<NA>")
        ).all(axis=1)
        invalid_index = identity_matches.index[~identity_matches]
        rerun_index = checkpoint_df.index[
            checkpoint_df[cleaned_col].apply(needs_llm_rerun)
            | checkpoint_df[status_col].eq("실패")
        ]
        dropped_index = invalid_index.union(rerun_index)
        rerun_rows = len(dropped_index)
        checkpoint_df = checkpoint_df.drop(index=dropped_index, errors="ignore")
        checkpoint_df = checkpoint_df.reindex(columns=result_cols)
    else:
        checkpoint_df = pd.DataFrame(columns=result_cols)

    reused_rows = len(checkpoint_df)
    targets = df.index.difference(checkpoint_df.index, sort=False).tolist()
    call_lock = threading.Lock()
    next_call_at = 0.0

    def throttled_call_once(name: str, content: str) -> str | None:
        nonlocal next_call_at
        if min_call_interval_seconds:
            with call_lock:
                now = time.monotonic()
                wait_seconds = next_call_at - now
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                next_call_at = time.monotonic() + min_call_interval_seconds
        return call_once(name, content)

    def refine_index(index: int) -> dict[str, object]:
        row = df.loc[index]
        result = refine_sentence(
            str(row[name_col]),
            row[content_col],
            call_once=throttled_call_once,
            max_attempts=max_attempts,
            validator=lambda original, cleaned: preservation_violations(
                original,
                cleaned,
                context_terms=(str(row[name_col]),),
            ),
        )
        return {
            "_index": index,
            **{column: row[column] for column in identity_cols},
            cleaned_col: result.cleaned_text,
            status_col: result.status,
            attempts_col: result.attempts,
            error_col: result.error_type,
            violations_col: " | ".join(result.violations),
        }

    for start in range(0, len(targets), chunk_size):
        chunk = targets[start : start + chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(refine_index, chunk))
        partial_df = pd.DataFrame(rows).set_index("_index")
        checkpoint_df = pd.concat([checkpoint_df, partial_df])
        checkpoint_df = checkpoint_df.loc[
            ~checkpoint_df.index.duplicated(keep="last")
        ].sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_df.to_csv(path, encoding="utf-8-sig")
        LOGGER.info("LLM 체크포인트 저장: %s/%s", min(start + chunk_size, len(targets)), len(targets))

        called_rows = partial_df.loc[partial_df[attempts_col].gt(0)]
        if start == 0 and len(called_rows) and called_rows[status_col].eq("실패").all():
            error_counts = called_rows[error_col].value_counts().to_dict()
            raise RuntimeError(f"초기 LLM 청크 전체 실패: {error_counts}")

    missing_index = df.index.difference(checkpoint_df.index, sort=False)
    if len(missing_index):
        raise ValueError(f"체크포인트 누락 인덱스: {missing_index.tolist()}")

    checkpoint_df = checkpoint_df.loc[df.index]
    target_status = checkpoint_df.loc[targets, status_col] if targets else pd.Series(dtype="string")
    summary = CheckpointRunSummary(
        total_rows=len(df),
        llm_target_rows=int(df[content_col].notna().sum()),
        reused_rows=reused_rows,
        new_called_rows=int(df.loc[targets, content_col].notna().sum()),
        new_success_rows=int(target_status.eq("성공").sum()),
        failed_rows=int(target_status.eq("실패").sum()),
        held_rows=int(target_status.isin(["결측", "보존위반"]).sum()),
        rerun_rows=rerun_rows,
    )
    return checkpoint_df, summary


def needs_llm_rerun(text: object) -> bool:
    """PUA 문자나 괄호형 라벨이 남은 텍스트인지 확인한다."""
    if pd.isna(text):
        return False

    value = str(text)
    return bool(PUA_PATTERN.search(value) or PAREN_LABEL_PATTERN.search(value))


__all__ = [
    "call_llm_once",
    "CheckpointRunSummary",
    "clean_sentence",
    "extract_proper_names",
    "extract_quantities",
    "extract_numbers",
    "needs_llm_rerun",
    "numbers_preserved",
    "preservation_violations",
    "refine_sentence",
    "RefinementResult",
    "run_checkpointed_refinement",
]
