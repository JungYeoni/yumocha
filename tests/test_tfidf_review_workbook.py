"""TF-IDF 영역분류 수작업 검토 Excel 테스트."""

from __future__ import annotations

import pandas as pd
import pytest
from openpyxl import load_workbook

from src.modeling.tfidf_review_workbook import create_review_workbook


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "연도": [2020, 2019],
            "지역": ["서울", "부산"],
            "원본행": [1, 2],
            "세부사업명": ["청년 취업", "아이 돌봄"],
            "주요내용_정제": ["고용 지원", pd.NA],
            "예측_대영역": ["1. 경제·고용·주거", "2. 가족·생활"],
            "예측_세부영역": ["1-1. 고용여건", "2-1. 돌봄 여건"],
            "예측_신뢰도": [0.9, 0.2],
            "저신뢰_검토대상": [False, True],
        }
    )


def test_create_review_workbook_adds_dropdowns_and_formulas(tmp_path):
    path = tmp_path / "review.xlsx"

    create_review_workbook(_prediction_frame(), path)

    workbook = load_workbook(path, data_only=False)
    sheet = workbook["영역분류검토"]
    assert sheet["D2"].value == "아이 돌봄"
    assert sheet["E2"].value is None
    assert sheet["K2"].value == "2-1. 돌봄 여건"
    assert sheet["L2"].value == "미검토"
    assert sheet["J2"].value.startswith("=IFERROR(VLOOKUP(")
    validations = {
        str(validation.sqref): validation.formula1
        for validation in sheet.data_validations.dataValidation
    }
    assert validations["K2:K3"] == "'라벨목록'!$A$2:$A$13"
    assert validations["L2:L3"] == "'라벨목록'!$A$16:$A$19"
    assert workbook["라벨목록"].sheet_state == "hidden"


def test_create_review_workbook_rejects_missing_prediction_columns(tmp_path):
    with pytest.raises(ValueError, match="필수 열 누락"):
        create_review_workbook(pd.DataFrame({"연도": [2020]}), tmp_path / "review.xlsx")
