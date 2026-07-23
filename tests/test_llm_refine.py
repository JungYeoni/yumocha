import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.features.llm_refine import (
    call_llm_once,
    clean_sentence,
    extract_proper_names,
    extract_quantities,
    extract_numbers,
    needs_llm_rerun,
    numbers_preserved,
    preservation_violations,
    refine_sentence,
    run_checkpointed_refinement,
)


def make_client_response(content):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    create = Mock(return_value=response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


@pytest.fixture
def llm_config():
    return {
        "upstage": {
            "model": "solar-pro3",
            "temperature": 0,
        },
        "prompt": {
            "template": "사업명: {name}\n주요내용: {content}",
        },
        "response_schema": {
            "name": "refined_text",
            "schema": {"type": "object"},
        },
    }


def test_call_llm_once_returns_cleaned_sentence(llm_config):
    raw = json.dumps({"정제문장": "공백을 정리한 문장"}, ensure_ascii=False)
    client, create = make_client_response(raw)

    result = call_llm_once(
        "사업 A",
        "공백을  정리한 문장",
        client=client,
        llm_config=llm_config,
    )

    assert result == "공백을 정리한 문장"
    create.assert_called_once_with(
        model="solar-pro3",
        messages=[
            {
                "role": "user",
                "content": "사업명: 사업 A\n주요내용: 공백을  정리한 문장",
            }
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": llm_config["response_schema"],
        },
    )


@pytest.mark.parametrize(
    "raw",
    [
        "JSON 아님",
        json.dumps({"다른키": "값"}, ensure_ascii=False),
        json.dumps({"정제문장": 123}, ensure_ascii=False),
        None,
    ],
)
def test_call_llm_once_returns_none_for_invalid_response(llm_config, raw):
    client, _ = make_client_response(raw)

    result = call_llm_once(
        "사업 A",
        "원문",
        client=client,
        llm_config=llm_config,
    )

    assert result is None


def test_clean_sentence_keeps_missing_value_without_calling_llm():
    call_once = Mock()

    result = clean_sentence("사업 A", pd.NA, call_once=call_once)

    assert result is None
    call_once.assert_not_called()


def test_clean_sentence_retries_once_and_returns_result():
    call_once = Mock(side_effect=[None, "정제 결과"])

    result = clean_sentence("사업 A", "원문", call_once=call_once)

    assert result == "정제 결과"
    assert call_once.call_count == 2


def test_clean_sentence_returns_original_after_failures():
    call_once = Mock(side_effect=RuntimeError("API 오류"))

    result = clean_sentence("사업 A", "원문", call_once=call_once)

    assert result == "원문"
    assert call_once.call_count == 2


def test_clean_sentence_rejects_invalid_attempt_count():
    with pytest.raises(ValueError, match="1 이상"):
        clean_sentence("사업 A", "원문", call_once=Mock(), max_attempts=0)


def test_refine_sentence_reports_api_failure_without_exposing_error_message():
    call_once = Mock(side_effect=RuntimeError("secret-bearing API detail"))

    result = refine_sentence("사업 A", "원문", call_once=call_once)

    assert result.cleaned_text == "원문"
    assert result.status == "실패"
    assert result.attempts == 2
    assert result.error_type == "RuntimeError"
    assert "secret-bearing" not in result.error_type


def test_refine_sentence_retries_preservation_violation_and_holds_original():
    call_once = Mock(return_value="월 300만원 지원")

    result = refine_sentence(
        "사업 A",
        "월 30만원 지원",
        call_once=call_once,
        validator=lambda original, cleaned: preservation_violations(original, cleaned),
    )

    assert result.cleaned_text == "월 30만원 지원"
    assert result.status == "보존위반"
    assert result.violations == ("숫자 불일치", "금액·퍼센트·범위 불일치")
    assert call_once.call_count == 2


def test_run_checkpointed_refinement_reuses_completed_and_reruns_marked_rows(tmp_path):
    source = pd.DataFrame(
        {
            "지역": ["서울", "서울", "서울"],
            "원본행": [10, 11, 12],
            "세부사업명": ["사업 A", "사업 B", "사업 C"],
            "주요내용": ["원문 A", "원문 B", pd.NA],
        },
        index=[100, 101, 102],
    )
    checkpoint_path = tmp_path / "checkpoint.csv"
    first_call = Mock(side_effect=lambda name, content: content.replace("원문", "정제"))

    first_checkpoint, first_summary = run_checkpointed_refinement(
        source,
        checkpoint_path=checkpoint_path,
        call_once=first_call,
        max_workers=2,
        chunk_size=2,
    )

    assert first_summary.total_rows == 3
    assert first_summary.llm_target_rows == 2
    assert first_summary.new_success_rows == 2
    assert first_summary.held_rows == 1
    assert first_call.call_count == 2

    first_checkpoint.loc[100, "주요내용_정제"] = "\uf09f재실행 대상"
    first_checkpoint.to_csv(checkpoint_path, encoding="utf-8-sig")
    second_call = Mock(return_value="정제 A")

    second_checkpoint, second_summary = run_checkpointed_refinement(
        source,
        checkpoint_path=checkpoint_path,
        call_once=second_call,
        max_workers=1,
        chunk_size=1,
    )

    assert second_summary.reused_rows == 2
    assert second_summary.rerun_rows == 1
    assert second_summary.new_success_rows == 1
    assert second_call.call_count == 1
    assert second_checkpoint.loc[100, "주요내용_정제"] == "정제 A"


def test_extract_numbers_preserves_order():
    assert extract_numbers("만 0~1세, 월 30만원씩 3개월") == ["0", "1", "30", "3"]
    assert extract_numbers(None) == []


def test_numbers_preserved_compares_sequences():
    assert numbers_preserved("월 30만원, 3개월", "월 30만 원, 3개월")
    assert not numbers_preserved("월 30만원, 3개월", "월 300만원, 3개월")


def test_extract_quantities_preserves_amount_percent_and_range_tokens():
    text = "월 30만원, 5%, 만 0~1세, 2020-2022년"

    assert extract_quantities(text) == ["30만원", "5%", "0~1세", "2020-2022년"]


def test_extract_proper_names_uses_quotes_institutions_and_present_context_only():
    text = "「아이행복」 사업을 서울복지관에서 운영"

    assert extract_proper_names(
        text,
        context_terms=("서울복지관에서", "없는 사업명"),
    ) == ["아이행복", "서울복지관", "서울복지관에서"]


def test_preservation_violations_detects_proper_name_and_empty_result():
    original = "「아이행복」 대상에게 서울복지관에서 월 30만원 지원"

    assert preservation_violations(original, "") == ("빈 결과",)
    assert preservation_violations(
        original,
        "대상에게 복지관에서 월 30만원 지원",
    ) == ("고유명사 불일치",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (None, False),
        ("정상 문장", False),
        ("\uf09f지원대상: 서울시민", True),
        ("(지원대상) 서울시민", True),
        ("(사업내용) 돌봄서비스 제공", True),
    ],
)
def test_needs_llm_rerun(text, expected):
    assert needs_llm_rerun(text) is expected
