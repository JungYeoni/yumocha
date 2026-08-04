"""2016-2020 예측과 2021-2024 확정 라벨 통합 검토 워크북 조립 테스트."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.build_2016_2024_review_workbook import (
    CONFIRMED_LABEL,
    DISPLAY_COLUMNS,
    DISPLAY_NOTE_COLUMN,
    MASTER_NOTE_COLUMN,
    NEW_PREDICTION_LABEL,
    SOURCE_LABEL_COLUMN,
    build_combined_frame,
    load_2016_2020_rows,
    load_2021_2024_confirmed_rows,
)


def _write_prediction_and_budget(tmp_path: Path) -> tuple[Path, Path]:
    prediction_path = tmp_path / "prediction.csv"
    budget_path = tmp_path / "budget.csv"
    pd.DataFrame(
        {
            "연도": [2018, 2019],
            "지역": ["서울", "부산"],
            "원본행": [1, 2],
            "세부사업명": ["청년 취업 지원", "아이 돌봄"],
            "주요내용_정제": ["고용 지원", "돌봄 서비스"],
            "예측_대영역": ["1. 경제·고용·주거", "2. 가족·생활"],
            "예측_세부영역": ["1-1. 고용여건", "2-1. 돌봄 여건"],
            "예측_신뢰도": [0.7, 0.4],
            "저신뢰_검토대상": [False, True],
        }
    ).to_csv(prediction_path, index=False)
    pd.DataFrame(
        {
            "연도": [2018, 2019],
            "지역": ["서울", "부산"],
            "원본행": [1, 2],
            "당해예산": [120.0, 80.0],
            "전년도예산": [100.0, 90.0],
        }
    ).to_csv(budget_path, index=False)
    return prediction_path, budget_path


def _review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "연도": [2021, 2022, 2023, 2016],
            "지역": ["서울", "부산", "대구", "서울"],
            "원본행": [10, 20, 30, 99],
            "세부사업명": ["청년 창업 지원", "노인 일자리", "협의 대상 사업", "미검토 사업"],
            "주요내용_정제": ["창업자금 지원", "일자리 연계", "논의 필요 내용", "내용"],
            "예측_대영역": ["1. 경제·고용·주거", "1. 경제·고용·주거", None, "지표체계 외"],
            "예측_세부영역": ["1-1. 고용여건", "1-1. 고용여건", None, "지표체계 외"],
            "예측_신뢰도": [0.9, 0.6, 0.3, 0.1],
            "저신뢰_검토대상": [False, False, True, True],
            "검토상태": ["확정", "수정", "논의필요", "미검토"],
            "검토_대영역": ["1. 경제·고용·주거", "1. 경제·고용·주거", None, None],
            "검토_세부영역": ["1-1. 고용여건", "1-3. 경제적 여건", None, None],
            "검토메모": ["확인완료", None, "재정팀 협의 필요", None],
            MASTER_NOTE_COLUMN: ["복합대응", None, None, None],
            "당해예산(백만원)": [50.0, 30.0, 20.0, 0.0],
            "전년도예산(백만원)": [40.0, 25.0, 18.0, 0.0],
        }
    )


def test_load_2016_2020_rows_merges_budget_and_blanks_review_state(tmp_path):
    prediction_path, budget_path = _write_prediction_and_budget(tmp_path)

    result = load_2016_2020_rows(prediction_path, budget_path)

    assert list(result.columns) == DISPLAY_COLUMNS
    assert result["당해예산"].tolist() == [120.0, 80.0]
    assert result["검토_세부영역"].isna().all()
    assert (result[SOURCE_LABEL_COLUMN] == NEW_PREDICTION_LABEL).all()


def test_load_2016_2020_rows_allows_legitimate_missing_budget_value(tmp_path):
    # "(신규)"·"(추가)" 사업처럼 전년도예산이 원래 없는 행은 매칭은
    # 됐지만 값이 NaN인 정상 케이스라 에러가 나면 안 된다.
    prediction_path, budget_path = _write_prediction_and_budget(tmp_path)
    budget = pd.read_csv(budget_path)
    budget.loc[0, "전년도예산"] = pd.NA
    budget.to_csv(budget_path, index=False)

    result = load_2016_2020_rows(prediction_path, budget_path)

    assert result["전년도예산"].isna().sum() == 1


def test_load_2016_2020_rows_rejects_unmatched_budget(tmp_path):
    prediction_path, budget_path = _write_prediction_and_budget(tmp_path)
    budget = pd.read_csv(budget_path)
    budget.loc[0, "연도"] = 2099  # 예측 대상과 매칭되지 않게 만듦
    budget.to_csv(budget_path, index=False)

    with pytest.raises(ValueError, match="키를 찾지 못한"):
        load_2016_2020_rows(prediction_path, budget_path)


def test_load_2021_2024_confirmed_rows_filters_by_year_only():
    result = load_2021_2024_confirmed_rows(_review_frame())

    assert len(result) == 3
    assert set(result["연도"]) == {2021, 2022, 2023}
    assert "미검토 사업" not in result["세부사업명"].tolist()
    row = result.loc[result["세부사업명"].eq("청년 창업 지원")].iloc[0]
    assert row["당해예산"] == 50.0
    assert row[DISPLAY_NOTE_COLUMN] == "복합대응"
    assert (result[SOURCE_LABEL_COLUMN] == CONFIRMED_LABEL).all()


def test_load_2021_2024_confirmed_rows_keeps_unresolved_statuses():
    # 논의필요·보류·다문화처럼 아직 미해결인 상태도 확정·수정만 걸러내면
    # 조용히 빠졌었다. 실제 검토상태 그대로 나와야 재정팀이 놓치지 않는다.
    result = load_2021_2024_confirmed_rows(_review_frame())

    row = result.loc[result["세부사업명"].eq("협의 대상 사업")].iloc[0]
    assert row["검토상태"] == "논의필요"
    assert pd.isna(row["검토_대영역"])
    assert row["검토메모"] == "재정팀 협의 필요"


def test_load_2021_2024_confirmed_rows_fills_missing_prediction_from_review():
    # 2021년 원본 라벨처럼 예측_* 열이 통째로 결측인 행은 검토(확정)값으로
    # 채우고 신뢰도 1.0·저신뢰 아님으로 표시해야 한다.
    frame = _review_frame()
    frame["저신뢰_검토대상"] = frame["저신뢰_검토대상"].astype("object")
    frame.loc[0, "예측_대영역"] = None
    frame.loc[0, "예측_세부영역"] = None
    frame.loc[0, "예측_신뢰도"] = None
    frame.loc[0, "저신뢰_검토대상"] = None

    result = load_2021_2024_confirmed_rows(frame)

    row = result.loc[result["세부사업명"].eq("청년 창업 지원")].iloc[0]
    assert row["예측_세부영역"] == "1-1. 고용여건"
    assert row["예측_신뢰도"] == 1.0
    assert row["저신뢰_검토대상"] is False


def test_build_combined_frame_concatenates_both_periods(tmp_path):
    prediction_path, budget_path = _write_prediction_and_budget(tmp_path)
    review_path = tmp_path / "review.xlsx"
    _review_frame().to_excel(review_path, sheet_name="작업용_유사사업순", index=False)

    combined = build_combined_frame(prediction_path, budget_path, review_path)

    assert len(combined) == 5  # 2016-2020 신규 2건 + 2021-2024 전체 3건(미해결 1건 포함)
    assert set(combined["연도"]) == {2018, 2019, 2021, 2022, 2023}
    assert set(combined[SOURCE_LABEL_COLUMN]) == {NEW_PREDICTION_LABEL, CONFIRMED_LABEL}


def test_build_combined_frame_rejects_duplicate_keys(tmp_path):
    prediction_path, budget_path = _write_prediction_and_budget(tmp_path)
    review_path = tmp_path / "review.xlsx"
    duplicate_review = _review_frame()
    # 부산·2022·20 확정 행의 키를 서울·2021·10(이미 확정 상태인 다른 행)과
    # 겹치게 만들어, 두 확정 행이 같은 키를 갖는 상황을 재현한다.
    duplicate_review.loc[1, ["연도", "지역", "원본행"]] = [2021, "서울", 10]
    duplicate_review.to_excel(review_path, sheet_name="작업용_유사사업순", index=False)

    with pytest.raises(ValueError, match="키 중복"):
        build_combined_frame(prediction_path, budget_path, review_path)
