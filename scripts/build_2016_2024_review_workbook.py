"""2016-2020년 신규 예측과 2021-2024년 확정 라벨을 한 검토 워크북으로 합친다.

`train_tfidf_area_classifier.py`가 만드는 검토 워크북은 2016-2020년
예측만 담고 있어, "이 사업이 2021-2024년엔 뭐였는지" 비교하며 검토할
수 없었다. 이 스크립트는 두 기간을 하나로 합치고, 예산·기존 검토
메모(명칭불일치·복합대응 여부)도 함께 넣어 `#65`의 그룹핑을
2016-2024년 전체에 대해 다시 계산한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.consolidate_2021_2024_confirmed_labels import (
    CONFIRMED_STATUSES,
    REVIEW_SHEET_NAME,
    TRAINING_YEARS,
)
from scripts.consolidate_2021_area_labels import normalize_text
from src.modeling.similarity_grouping import assign_similarity_groups
from src.modeling.tfidf_review_workbook import create_review_workbook

MASTER_NOTE_COLUMN = "명칭-내용 불일치 / 원본PDF 동일 여부\n & 복합대응사업 여부"
DISPLAY_NOTE_COLUMN = "명칭_내용_불일치_복합대응"
SOURCE_LABEL_COLUMN = "자료구분"
NEW_PREDICTION_LABEL = "신규 예측(2016-2020)"
CONFIRMED_LABEL = "기존 확정(2021-2024)"

DISPLAY_COLUMNS = [
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
    DISPLAY_NOTE_COLUMN,
    "당해예산",
    "전년도예산",
    SOURCE_LABEL_COLUMN,
]


def load_2016_2020_rows(prediction_path: Path, budget_path: Path) -> pd.DataFrame:
    """신규 예측에 예산을 붙이고, 아직 검토 전이므로 검토_* 열은 비워 둔다."""
    predicted = pd.read_csv(prediction_path)
    budget = pd.read_csv(budget_path, usecols=["연도", "지역", "원본행", "당해예산", "전년도예산"])
    merged = predicted.merge(
        budget, on=["연도", "지역", "원본행"], how="left", validate="one_to_one"
    )

    output = merged.copy()
    for column in ("검토_대영역", "검토_세부영역", "검토상태", "검토메모", DISPLAY_NOTE_COLUMN):
        output[column] = pd.NA
    output[SOURCE_LABEL_COLUMN] = NEW_PREDICTION_LABEL
    return output[DISPLAY_COLUMNS]


def load_2021_2024_confirmed_rows(review: pd.DataFrame) -> pd.DataFrame:
    """검토완료 워크북에서 확정 라벨을 예측·검토·예산·메모까지 보존해 그대로 가져온다."""
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
        "검토상태",
        "검토_대영역",
        "검토_세부영역",
        "검토메모",
        MASTER_NOTE_COLUMN,
        "당해예산(백만원)",
        "전년도예산(백만원)",
    ]
    missing = [column for column in required if column not in review.columns]
    if missing:
        raise ValueError(f"검토 워크북 필수 열 누락: {missing}")

    frame = review[required].copy()
    frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype("int64")
    frame["지역"] = frame["지역"].map(normalize_text)
    confirmed = frame.loc[
        frame["연도"].isin(TRAINING_YEARS) & frame["검토상태"].isin(CONFIRMED_STATUSES)
    ].copy()
    if confirmed.empty:
        raise ValueError("확정·수정 상태인 2021-2024년 행이 없습니다.")

    # 2021년 원본 라벨은 TF-IDF가 아니라 사람이 직접 붙인 정답이라
    # 예측_* 열 자체가 없다(마스터 파일에서 결측으로 비워 둠). 검토용
    # 워크북 표시를 위해 확정값을 예측값 자리에 채우고 신뢰도는 1.0,
    # 저신뢰 플래그는 False로 둔다(모델 추측이 아니라 원 정답이라는 뜻).
    no_prediction = confirmed["예측_세부영역"].isna()
    confirmed.loc[no_prediction, "예측_대영역"] = confirmed.loc[no_prediction, "검토_대영역"]
    confirmed.loc[no_prediction, "예측_세부영역"] = confirmed.loc[no_prediction, "검토_세부영역"]
    confirmed.loc[no_prediction, "예측_신뢰도"] = 1.0
    confirmed.loc[no_prediction, "저신뢰_검토대상"] = False

    confirmed = confirmed.rename(
        columns={
            MASTER_NOTE_COLUMN: DISPLAY_NOTE_COLUMN,
            "당해예산(백만원)": "당해예산",
            "전년도예산(백만원)": "전년도예산",
        }
    )
    confirmed[SOURCE_LABEL_COLUMN] = CONFIRMED_LABEL
    return confirmed[DISPLAY_COLUMNS]


def build_combined_frame(
    prediction_path: Path,
    budget_path: Path,
    review_workbook_path: Path,
    review_sheet_name: str = REVIEW_SHEET_NAME,
) -> pd.DataFrame:
    """2016-2020 신규 예측과 2021-2024 확정 라벨을 하나의 검토용 프레임으로 합친다."""
    new_rows = load_2016_2020_rows(prediction_path, budget_path)
    review = pd.read_excel(review_workbook_path, sheet_name=review_sheet_name, engine="openpyxl")
    confirmed_rows = load_2021_2024_confirmed_rows(review)

    combined = pd.concat([new_rows, confirmed_rows], ignore_index=True)
    if combined.duplicated(["연도", "지역", "원본행"]).any():
        duplicate = combined.loc[
            combined.duplicated(["연도", "지역", "원본행"], keep=False),
            ["연도", "지역", "원본행"],
        ]
        raise ValueError(
            f"연도·지역·원본행 키 중복: {duplicate.drop_duplicates().to_dict('records')}"
        )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=Path(
            "data/processed/영역분류_라벨링/TFIDF_예측_2021_2024재학습/TFIDF_영역분류_예측.csv"
        ),
    )
    parser.add_argument(
        "--budget-path",
        type=Path,
        default=Path(
            "data/interim/영역분류_라벨링/TFIDF_분류대상/2016_2020_TFIDF_분류대상_통합.csv"
        ),
    )
    parser.add_argument(
        "--review-workbook",
        type=Path,
        default=Path(
            "data/interim/영역분류_라벨링/재정팀_검토본/현재검토본/"
            "2021포함_전체57979건_유사사업별_시도연도순_예산포함.xlsx"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "data/processed/영역분류_라벨링/TFIDF_예측_2021_2024재학습/"
            "TFIDF_영역분류_검토_2016_2024_통합.xlsx"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = build_combined_frame(args.prediction_path, args.budget_path, args.review_workbook)
    grouped = assign_similarity_groups(combined)
    create_review_workbook(grouped, args.output_path, preserve_order=True)

    print(f"통합 행: {len(combined):,}개")
    print(
        f"  신규 예측(2016-2020): {int((combined[SOURCE_LABEL_COLUMN] == NEW_PREDICTION_LABEL).sum()):,}개"
    )
    print(
        f"  기존 확정(2021-2024): {int((combined[SOURCE_LABEL_COLUMN] == CONFIRMED_LABEL).sum()):,}개"
    )
    print(f"저장: {args.output_path}")


if __name__ == "__main__":
    main()
