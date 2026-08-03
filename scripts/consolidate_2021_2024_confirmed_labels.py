"""검토완료 워크북에서 확정 라벨을 뽑아 TF-IDF 재학습용 데이터로 만든다.

기존 2021년 라벨(`consolidate_2021_area_labels.py`)은 재정팀이 지역별로
나눠 준 XLSX가 입력이었지만, 이번 입력은 2021~2024년 전체를 한 시트에
모아 사람이 검토한 워크북(`검토상태`·`검토_대영역`·`검토_세부영역` 열
포함) 하나다. 그래서 별도 스크립트로 분리했다. taxonomy 정의
(`MAJOR_BY_SUBCATEGORY`, `REGION_ORDER`)는 기존 모듈을 그대로 재사용한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.consolidate_2021_area_labels import (
    MAJOR_BY_SUBCATEGORY,
    REGION_ORDER,
    normalize_text,
)

REVIEW_SHEET_NAME = "작업용_유사사업순"
CONFIRMED_STATUSES = ("확정", "수정")
TRAINING_YEARS = {2021, 2022, 2023, 2024}
REQUIRED_COLUMNS = [
    "연도",
    "지역",
    "원본행",
    "세부사업명",
    "주요내용_정제",
    "검토상태",
    "검토_대영역",
    "검토_세부영역",
]
TRAIN_COLUMNS = [
    "연도",
    "지역",
    "원본행",
    "세부사업명",
    "주요내용_정제",
    "대영역",
    "세부영역",
]


def load_confirmed_labels(review: pd.DataFrame) -> pd.DataFrame:
    """검토완료 워크북 프레임에서 확정·수정 라벨만 골라 학습 스키마로 정리한다."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in review.columns]
    if missing_columns:
        raise ValueError(f"검토 워크북 필수 열 누락: {missing_columns}")
    if review.empty:
        raise ValueError("검토 워크북이 비어 있습니다.")

    frame = review[REQUIRED_COLUMNS].copy()
    frame["지역"] = frame["지역"].map(normalize_text)
    frame["연도"] = pd.to_numeric(frame["연도"], errors="raise").astype("int64")
    frame["원본행"] = pd.to_numeric(frame["원본행"], errors="raise").astype("int64")

    unknown_regions = sorted(set(frame["지역"]) - set(REGION_ORDER))
    if unknown_regions:
        raise ValueError(f"REGION_ORDER에 없는 지역이 있습니다: {unknown_regions}")

    confirmed = frame.loc[
        frame["연도"].isin(TRAINING_YEARS) & frame["검토상태"].isin(CONFIRMED_STATUSES)
    ].drop(columns="검토상태")
    if confirmed.empty:
        raise ValueError("확정·수정 상태인 행이 없습니다.")

    if confirmed[["검토_대영역", "검토_세부영역"]].isna().any().any():
        missing = confirmed.loc[
            confirmed[["검토_대영역", "검토_세부영역"]].isna().any(axis=1),
            ["연도", "지역", "원본행"],
        ]
        raise ValueError(
            f"확정 라벨에 대영역·세부영역 결측이 있습니다: {missing.to_dict('records')}"
        )

    unknown_subcategory = sorted(set(confirmed["검토_세부영역"]) - set(MAJOR_BY_SUBCATEGORY))
    if unknown_subcategory:
        raise ValueError(f"정의되지 않은 세부영역: {unknown_subcategory}")

    canonical_major = confirmed["검토_세부영역"].map(MAJOR_BY_SUBCATEGORY)
    mismatched = confirmed.loc[confirmed["검토_대영역"].ne(canonical_major)]
    if not mismatched.empty:
        keys = mismatched[["연도", "지역", "원본행"]].to_dict("records")
        raise ValueError(f"검토_대영역이 세부영역 taxonomy와 어긋납니다: {keys}")

    confirmed = confirmed.rename(columns={"검토_대영역": "대영역", "검토_세부영역": "세부영역"})

    if confirmed.duplicated(["연도", "지역", "원본행"]).any():
        duplicate = confirmed.loc[
            confirmed.duplicated(["연도", "지역", "원본행"], keep=False),
            ["연도", "지역", "원본행"],
        ]
        raise ValueError(
            f"연도·지역·원본행 키 중복: {duplicate.drop_duplicates().to_dict('records')}"
        )

    if confirmed["세부사업명"].map(normalize_text).eq("").any():
        raise ValueError("세부사업명이 비어 있는 확정 행이 있습니다.")

    confirmed["지역"] = pd.Categorical(confirmed["지역"], categories=REGION_ORDER, ordered=True)
    confirmed = confirmed.sort_values(["연도", "지역", "원본행"]).reset_index(drop=True)
    confirmed["지역"] = confirmed["지역"].astype("string")
    return confirmed[TRAIN_COLUMNS]


def build_qa(review: pd.DataFrame, confirmed: pd.DataFrame) -> pd.DataFrame:
    """검토상태 분포와 연도별 확정 건수를 요약한다."""
    status_counts = (
        review["검토상태"]
        .value_counts(dropna=False)
        .rename_axis("검토상태")
        .reset_index(name="건수")
    )
    status_counts.insert(0, "구분", "전체_검토상태분포")

    year_counts = confirmed.groupby("연도").size().rename("건수").reset_index()
    year_counts.insert(0, "구분", "확정라벨_연도별건수")
    year_counts = year_counts.rename(columns={"연도": "검토상태"})

    return pd.concat([status_counts, year_counts], ignore_index=True)


def save_outputs(confirmed: pd.DataFrame, qa: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """확정 라벨 통합본과 QA를 UTF-8-SIG CSV로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "확정라벨": output_dir / "2021_2024_확정라벨_통합.csv",
        "QA": output_dir / "2021_2024_확정라벨_QA.csv",
    }
    confirmed.to_csv(paths["확정라벨"], index=False, encoding="utf-8-sig")
    qa.to_csv(paths["QA"], index=False, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-workbook",
        type=Path,
        default=Path(
            "data/interim/영역분류_라벨링/재정팀_검토본/현재검토본/"
            "2021포함_전체57979건_유사사업별_시도연도순_예산포함.xlsx"
        ),
    )
    parser.add_argument("--sheet-name", default=REVIEW_SHEET_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/2021_2024/통합"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = pd.read_excel(args.review_workbook, sheet_name=args.sheet_name, engine="openpyxl")
    confirmed = load_confirmed_labels(review)
    qa = build_qa(review, confirmed)
    paths = save_outputs(confirmed, qa, args.output_dir)

    print(f"확정·수정 라벨: {len(confirmed):,}행")
    print(f"연도별: {confirmed.groupby('연도').size().to_dict()}")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
