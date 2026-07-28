"""TF-IDF 영역분류 결과를 수작업 검토용 Excel로 내보낸다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from scripts.consolidate_2021_area_labels import MAJOR_BY_SUBCATEGORY

REVIEW_STATUSES = ["미검토", "확정", "수정", "보류"]
REVIEW_COLUMNS = [
    "연도",
    "지역",
    "원본행",
    "세부사업명",
    "주요내용_정제",
    "예측_대영역",
    "예측_세부영역",
    "예측_신뢰도",
    "저신뢰_검토대상",
    "검토_대영역",
    "검토_세부영역",
    "검토상태",
    "검토메모",
]


def validate_review_source(frame: pd.DataFrame) -> None:
    """검토용 Excel 생성에 필요한 예측 열을 확인한다."""
    required = [
        "연도",
        "지역",
        "원본행",
        "세부사업명",
        "주요내용_정제",
        "예측_대영역",
        "예측_세부영역",
        "예측_신뢰도",
        "저신뢰_검토대상",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"검토용 Excel 필수 열 누락: {missing}")
    if frame.empty:
        raise ValueError("검토할 예측 데이터가 비어 있습니다.")
    if frame[["연도", "지역", "원본행", "예측_세부영역"]].isna().any().any():
        raise ValueError("검토 키 또는 예측 라벨에 결측이 있습니다.")


def create_review_workbook(frame: pd.DataFrame, output_path: Path) -> Path:
    """예측값을 초기 검토값으로 채운 드롭다운 Excel을 생성한다."""
    validate_review_source(frame)
    ordered = frame.sort_values(
        ["저신뢰_검토대상", "예측_신뢰도", "연도", "지역", "원본행"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)

    workbook = Workbook()
    workbook.calculation.calcMode = "auto"
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    review_sheet = workbook.active
    review_sheet.title = "영역분류검토"
    label_sheet = workbook.create_sheet("라벨목록")

    label_sheet.append(["세부영역", "대영역"])
    for subcategory, major in MAJOR_BY_SUBCATEGORY.items():
        label_sheet.append([subcategory, major])
    label_sheet.append([])
    label_sheet.append(["검토상태"])
    for status in REVIEW_STATUSES:
        label_sheet.append([status])

    review_sheet.append(REVIEW_COLUMNS)
    for row in ordered.itertuples(index=False):
        review_sheet.append(
            [
                row.연도,
                row.지역,
                row.원본행,
                row.세부사업명,
                None if pd.isna(row.주요내용_정제) else row.주요내용_정제,
                row.예측_대영역,
                row.예측_세부영역,
                float(row.예측_신뢰도),
                bool(row.저신뢰_검토대상),
                None,
                row.예측_세부영역,
                "미검토",
                None,
            ]
        )

    last_row = review_sheet.max_row
    subcategory_count = len(MAJOR_BY_SUBCATEGORY)
    status_start = subcategory_count + 4
    status_end = status_start + len(REVIEW_STATUSES) - 1

    subcategory_validation = DataValidation(
        type="list",
        formula1=f"'라벨목록'!$A$2:$A${subcategory_count + 1}",
        allow_blank=False,
    )
    status_validation = DataValidation(
        type="list",
        formula1=f"'라벨목록'!$A${status_start}:$A${status_end}",
        allow_blank=False,
    )
    review_sheet.add_data_validation(subcategory_validation)
    review_sheet.add_data_validation(status_validation)
    subcategory_validation.add(f"K2:K{last_row}")
    status_validation.add(f"L2:L{last_row}")

    for row_number in range(2, last_row + 1):
        review_sheet.cell(row=row_number, column=10).value = (
            f"=IFERROR(VLOOKUP(K{row_number},'라벨목록'!$A$2:$B$"
            f'{subcategory_count + 1},2,FALSE),"")'
        )

    header_fill = PatternFill("solid", fgColor="1F4E78")
    editable_fill = PatternFill("solid", fgColor="FFF2CC")
    low_confidence_fill = PatternFill("solid", fgColor="FCE4D6")
    for cell in review_sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for column in ("J", "K", "L", "M"):
        for cell in review_sheet[column][1:]:
            cell.fill = editable_fill

    review_sheet.conditional_formatting.add(
        f"A2:M{last_row}",
        FormulaRule(formula=["$I2=TRUE"], fill=low_confidence_fill),
    )
    review_sheet.conditional_formatting.add(
        f"A2:M{last_row}",
        FormulaRule(
            formula=['$L2="수정"'],
            fill=PatternFill("solid", fgColor="E2F0D9"),
        ),
    )

    review_sheet.freeze_panes = "D2"
    review_sheet.auto_filter.ref = f"A1:M{last_row}"
    review_sheet.sheet_view.showGridLines = False
    review_sheet.row_dimensions[1].height = 28

    widths = {
        "A": 10,
        "B": 9,
        "C": 11,
        "D": 32,
        "E": 55,
        "F": 22,
        "G": 27,
        "H": 13,
        "I": 17,
        "J": 22,
        "K": 27,
        "L": 12,
        "M": 35,
    }
    for column, width in widths.items():
        review_sheet.column_dimensions[column].width = width
    for row in review_sheet.iter_rows(min_row=2, max_row=last_row):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=False)
        row[7].number_format = "0.000"
    for column in ("D", "E", "M"):
        for cell in review_sheet[column][1:]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    for cell in label_sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
    label_sheet.column_dimensions["A"].width = 30
    label_sheet.column_dimensions["B"].width = 24
    label_sheet.sheet_state = "hidden"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    return output_path
