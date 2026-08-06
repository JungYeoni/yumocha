import pandas as pd

import pytest

from scripts.build_provisional_area_labels import (
    REQUIRED_COLUMNS,
    build_outputs,
    build_uncertainty_summary,
    select_labels,
    validate_schema_and_keys,
)


def _sample_review(rows: list[dict[str, object]]) -> pd.DataFrame:
    base = {column: None for column in REQUIRED_COLUMNS}
    frame = pd.DataFrame([{**base, **row} for row in rows])
    frame["연도"] = frame["연도"].astype(int)
    frame["원본행"] = frame["원본행"].astype(int)
    return frame[REQUIRED_COLUMNS]


def _confirmed_row(연도: int, 지역: str, 원본행: int, 세부영역: str = "1-1. 고용여건") -> dict:
    return {
        "연도": 연도,
        "지역": 지역,
        "원본행": 원본행,
        "세부사업명": "테스트 사업",
        "예측_대영역": "1. 경제·고용·주거",
        "예측_세부영역": "1-1. 고용여건",
        "예측_신뢰도": 0.9,
        "저신뢰_검토대상": False,
        "검토_대영역": "1. 경제·고용·주거",
        "검토_세부영역": 세부영역,
        "검토상태": "확정",
        "검토메모": None,
        "명칭_내용_불일치_복합대응": None,
        "당해예산": 100.0,
    }


def test_validate_schema_and_keys_passes_for_clean_input():
    review = _sample_review([_confirmed_row(2021, "서울", 1)])
    qa_records: list[dict[str, object]] = []
    validate_schema_and_keys(review, qa_records)
    qa = pd.DataFrame(qa_records)
    assert qa.loc[qa["판정"].eq("FAIL")].empty


def test_validate_schema_and_keys_flags_duplicate_keys():
    review = _sample_review([_confirmed_row(2021, "서울", 1), _confirmed_row(2021, "서울", 1)])
    qa_records: list[dict[str, object]] = []
    validate_schema_and_keys(review, qa_records)
    qa = pd.DataFrame(qa_records)
    row = qa.loc[qa["검사항목"].eq("키 중복(연도·지역·원본행)")].iloc[0]
    assert row["판정"] == "FAIL"
    assert row["실제값"] == 1


def test_validate_schema_and_keys_flags_unknown_region_and_taxonomy():
    row = _confirmed_row(2021, "부산", 1)
    row["지역"] = "가상지역"
    row["검토_세부영역"] = "9-9. 존재하지않는영역"
    review = _sample_review([row])
    qa_records: list[dict[str, object]] = []
    validate_schema_and_keys(review, qa_records)
    qa = pd.DataFrame(qa_records)
    assert qa.loc[qa["검사항목"].eq("REGION_ORDER 외 지역"), "판정"].iloc[0] == "FAIL"
    assert qa.loc[qa["검사항목"].eq("검토_세부영역 taxonomy 외 값"), "판정"].iloc[0] == "FAIL"


def test_select_labels_prefers_confirmed_over_predicted():
    review = _sample_review([_confirmed_row(2021, "서울", 1, "2-1. 돌봄 여건")])
    labeled = select_labels(review)
    row = labeled.iloc[0]
    assert row["세부영역"] == "2-1. 돌봄 여건"
    assert row["대영역"] == "2. 가족·생활"
    assert row["라벨출처"] == "검토_확정수정"


def test_select_labels_falls_back_to_prediction_when_not_confirmed():
    row = _confirmed_row(2021, "서울", 1)
    row["검토상태"] = "미검토"
    row["검토_세부영역"] = None
    row["검토_대영역"] = None
    review = _sample_review([row])
    labeled = select_labels(review)
    result = labeled.iloc[0]
    assert result["세부영역"] == "1-1. 고용여건"
    assert result["라벨출처"] == "TFIDF_예측"


def test_select_labels_treats_blank_status_with_note_as_confirmed():
    """검토상태가 비어도 S열(명칭_내용_불일치_복합대응)에 값이 있으면 이미 검토가
    끝난 것으로 본다 — 재정팀이 소계성 항목은 검토상태 대신 이 비고 열에
    사유를 남기는 관행이 있음을 실제 데이터에서 확인했다(24건 전부 S열=
    "소계 표기").
    """
    row = _confirmed_row(2021, "서울", 1, "지표체계 외")
    row["검토상태"] = None
    row["명칭_내용_불일치_복합대응"] = "소계 표기"
    review = _sample_review([row])
    labeled = select_labels(review)
    result = labeled.iloc[0]
    assert result["세부영역"] == "지표체계 외"
    assert result["라벨출처"] == "검토_비고기반확정"


def test_select_labels_falls_back_to_prediction_when_status_blank_and_no_note():
    row = _confirmed_row(2021, "서울", 1)
    row["검토상태"] = None
    row["검토_세부영역"] = None
    row["검토_대영역"] = None
    review = _sample_review([row])
    labeled = select_labels(review)
    result = labeled.iloc[0]
    assert result["세부영역"] == "1-1. 고용여건"
    assert result["라벨출처"] == "TFIDF_예측"


def test_select_labels_leaves_unassigned_when_neither_available():
    row = _confirmed_row(2021, "서울", 1)
    row["검토상태"] = "보류"
    row["검토_세부영역"] = None
    row["검토_대영역"] = None
    row["예측_세부영역"] = None
    review = _sample_review([row])
    labeled = select_labels(review)
    result = labeled.iloc[0]
    assert pd.isna(result["세부영역"])
    assert pd.isna(result["대영역"])
    assert result["라벨출처"] == "미배정"


def test_build_uncertainty_summary_aggregates_budget_and_counts():
    review = _sample_review(
        [
            _confirmed_row(2021, "서울", 1, "1-1. 고용여건"),
            _confirmed_row(2021, "서울", 2, "1-1. 고용여건"),
        ]
    )
    labeled = select_labels(review)
    summary = build_uncertainty_summary(labeled)
    row = summary.iloc[0]
    assert row["건수"] == 2
    assert row["예산액_백만원"] == 200.0
    assert row["전체예산_백만원"] == 200.0
    assert row["예산비중"] == 1.0


def test_build_outputs_excludes_unassigned_rows_from_label_csv():
    resolved_row = _confirmed_row(2021, "서울", 1)
    unresolved_row = _confirmed_row(2021, "서울", 2)
    unresolved_row["검토상태"] = "보류"
    unresolved_row["검토_세부영역"] = None
    unresolved_row["검토_대영역"] = None
    unresolved_row["예측_세부영역"] = None
    review = _sample_review([resolved_row, unresolved_row])

    label_csv, uncertainty_summary, unresolved_audit, qa_records = build_outputs(review)

    assert len(label_csv) == 1
    assert list(label_csv.columns) == ["지역", "연도", "원본행", "대영역", "세부영역"]
    assert len(unresolved_audit) == 1
    assert unresolved_audit.iloc[0]["원본행"] == 2
    assert not uncertainty_summary.empty
    qa = pd.DataFrame(qa_records)
    assert qa.loc[qa["판정"].eq("FAIL")].empty


def test_build_outputs_raises_on_hard_qa_failure():
    review = _sample_review([_confirmed_row(2021, "서울", 1), _confirmed_row(2021, "서울", 1)])
    with pytest.raises(ValueError, match="§A 입력검증 실패"):
        build_outputs(review)
