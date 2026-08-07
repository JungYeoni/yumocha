"""통합 영역분류 검토 워크북을 이슈 #81 파이프라인이 쓸 라벨 CSV로 변환한다.

`scripts/run_provisional_pipeline.py --label-files`는 이미 `지역·연도·원본행·
대영역·세부영역` 5개 열짜리 CSV를 기대하지만, 이 CSV를 만드는 단계가 없었다.
이 스크립트가 그 빠진 앞단이다 — 파이프라인 코드(`src/provisional/`,
`run_provisional_pipeline.py`)는 건드리지 않는다.

라벨 선택 규칙(이슈 #81 본문)을 그대로 따른다:
1. `검토상태`가 `확정`/`수정`이고 `검토_세부영역`이 taxonomy와 일치하면 그 라벨을 쓴다.
2. 확정 라벨이 없고 `예측_세부영역`(TF-IDF)이 유효하면 그 라벨을 쓴다.
3. 둘 다 없으면 라벨을 만들지 않고 `미배정`으로 별도 감사 목록에 남긴다
   (`_load_labels`가 라벨 누락 행이 있으면 즉시 실패하도록 설계돼 있으므로,
   이 스크립트는 미배정 행을 라벨 CSV에서 제외하고 별도 파일로만 보고한다 —
   그 행들을 파이프라인 입력에서 어떻게 처리할지는 팀 결정이 필요하다).
5. 대영역은 세부영역 taxonomy(`MAJOR_BY_SUBCATEGORY`)에서 재생성한다 — 원본
   `검토_대영역`/`예측_대영역` 열은 교차검증에만 쓰고 그대로 출력하지 않는다.

taxonomy(`REGION_ORDER`, `MAJOR_BY_SUBCATEGORY`)는 `consolidate_2021_area_labels.py`
것을 그대로 재사용한다(재발명 금지) — 지금 워크북의 검토_세부영역/예측_세부영역
값 12종이 이 taxonomy와 정확히 일치함을 확인했다.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKBOOK = (
    REPO_ROOT
    / "data"
    / "processed"
    / "영역분류_라벨링"
    / "TFIDF_예측_2021_2024재학습"
    / "TFIDF_영역분류_검토_2016_2024_통합.xlsx"
)
SHEET_NAME = "영역분류검토"
OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "영역분류_라벨링" / "2016_2024_통합"
QA_REPORT_PATH = REPO_ROOT / "reports" / "20260807_영역분류_2016_2024_라벨선택_QA.csv"

EXPECTED_YEARS = tuple(range(2016, 2025))
EXPECTED_TOTAL_ROWS = 57_979
CONFIRMED_STATUSES = ("확정", "수정")

REQUIRED_COLUMNS = [
    "연도",
    "지역",
    "원본행",
    "세부사업명",
    "예측_대영역",
    "예측_세부영역",
    "예측_신뢰도",
    "저신뢰_검토대상",
    "검토_대영역",
    "검토_세부영역",
    "검토상태",
    "검토메모",
    "명칭_내용_불일치_복합대응",
    "당해예산",
]
LABEL_OUTPUT_COLUMNS = ["지역", "연도", "원본행", "대영역", "세부영역"]
AUDIT_COLUMNS = [
    "지역",
    "연도",
    "원본행",
    "세부사업명",
    "검토상태",
    "검토메모",
    "예측_세부영역",
    "예측_신뢰도",
    "명칭_내용_불일치_복합대응",
]


def _add_qa(
    qa_records: list[dict[str, object]],
    section: str,
    item: str,
    expected: object,
    actual: object,
    *,
    informational: bool = False,
) -> None:
    """`informational=True`는 통과·실패를 가르지 않고 보고만 하는 항목이다."""
    qa_records.append(
        {
            "구분": section,
            "검사항목": item,
            "기대값": expected,
            "실제값": actual,
            "판정": "INFO" if informational else ("PASS" if expected == actual else "FAIL"),
        }
    )


def load_review_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    """통합 검토 워크북을 그대로 읽는다(시트 고정, 최신 파일 자동 선택 없음)."""
    if not path.is_file():
        raise FileNotFoundError(f"검토 워크북이 없습니다: {path}")
    return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")


def validate_schema_and_keys(review: pd.DataFrame, qa_records: list[dict[str, object]]) -> None:
    """§A: 필수 열·행수·키 중복·지역/연도 범위·taxonomy 정합성을 검증한다."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in review.columns]
    if missing_columns:
        raise ValueError(f"검토 워크북 필수 열 누락: {missing_columns}")

    budget_numeric = pd.to_numeric(review["당해예산"], errors="coerce")
    invalid_budget = review["당해예산"].notna() & budget_numeric.isna()
    if invalid_budget.any():
        samples = review.loc[invalid_budget, "당해예산"].astype(str).unique()[:5].tolist()
        raise ValueError(f"검토 워크북 당해예산에 숫자로 변환할 수 없는 값이 있습니다: {samples}")
    review["당해예산"] = budget_numeric

    _add_qa(
        qa_records,
        "§A 입력검증",
        f"전체 행 수(참고: 확인 시점 기준값 {EXPECTED_TOTAL_ROWS:,})",
        None,
        len(review),
        informational=True,
    )

    keys = ["연도", "지역", "원본행"]
    _add_qa(
        qa_records,
        "§A 입력검증",
        "키 중복(연도·지역·원본행)",
        0,
        int(review.duplicated(keys).sum()),
    )

    unknown_regions = sorted(set(review["지역"].map(normalize_text)) - set(REGION_ORDER))
    _add_qa(qa_records, "§A 입력검증", "REGION_ORDER 외 지역", [], unknown_regions)

    unknown_years = sorted(set(review["연도"].astype(int)) - set(EXPECTED_YEARS))
    _add_qa(qa_records, "§A 입력검증", "2016-2024 외 연도", [], unknown_years)

    taxonomy_subcategories = set(MAJOR_BY_SUBCATEGORY)
    for column in ("검토_세부영역", "예측_세부영역"):
        observed = set(review[column].dropna().unique())
        unknown = sorted(observed - taxonomy_subcategories)
        _add_qa(qa_records, "§A 입력검증", f"{column} taxonomy 외 값", [], unknown)

    # 대영역은 이 스크립트가 세부영역에서 재생성하지만(규칙 5), 원본 검토_대영역이
    # 이미 taxonomy와 어긋나 있는지도 참고용으로 기록한다.
    has_both = review["검토_세부영역"].notna() & review["검토_대영역"].notna()
    derived_major = review.loc[has_both, "검토_세부영역"].map(MAJOR_BY_SUBCATEGORY)
    mismatched = int(review.loc[has_both, "검토_대영역"].ne(derived_major).sum())
    _add_qa(
        qa_records,
        "§A 입력검증",
        "검토_대영역이 세부영역 taxonomy와 불일치",
        0,
        mismatched,
        informational=True,
    )

    _add_qa(
        qa_records,
        "§A 입력검증",
        "당해예산 음수 건수(원자료 그대로 보존)",
        None,
        int(review["당해예산"].lt(0).sum()),
        informational=True,
    )
    _add_qa(
        qa_records,
        "§A 입력검증",
        "당해예산 결측 건수(원자료 그대로 보존)",
        None,
        int(review["당해예산"].isna().sum()),
        informational=True,
    )


NOTE_COLUMN = "명칭_내용_불일치_복합대응"


def select_labels(review: pd.DataFrame) -> pd.DataFrame:
    """라벨 선택 규칙(확정·수정 우선 → TF-IDF 예측 → 미배정)을 적용한다.

    검토상태가 비어 있어도 명칭_내용_불일치_복합대응(S열)에 값이 있으면(예:
    "소계 표기") 검토_세부영역을 무시하지 않는다 — 재정팀이 이미 "소계성
    항목이라 실제 사업 카테고리로 억지 분류하지 않는다"고 판단해 그 비고
    열에 사유를 남기고 세부영역엔 `지표체계 외`를 넣어둔 것이므로, 검토상태
    드롭다운을 안 눌렀다고 TF-IDF 예측으로 넘기면 이미 끝난 검토를 덮어쓰게
    된다.
    """
    frame = review.copy()
    frame["지역"] = frame["지역"].map(normalize_text)

    reviewed_without_status = frame["검토상태"].isna() & frame[NOTE_COLUMN].notna()
    confirmed_mask = (frame["검토상태"].isin(CONFIRMED_STATUSES) | reviewed_without_status) & frame[
        "검토_세부영역"
    ].notna()
    predicted_mask = ~confirmed_mask & frame["예측_세부영역"].notna()

    frame["세부영역"] = pd.NA
    frame.loc[confirmed_mask, "세부영역"] = frame.loc[confirmed_mask, "검토_세부영역"]
    frame.loc[predicted_mask, "세부영역"] = frame.loc[predicted_mask, "예측_세부영역"]
    frame["대영역"] = frame["세부영역"].map(MAJOR_BY_SUBCATEGORY)

    frame["라벨출처"] = "미배정"
    frame.loc[confirmed_mask, "라벨출처"] = "검토_확정수정"
    frame.loc[confirmed_mask & reviewed_without_status, "라벨출처"] = "검토_비고기반확정"
    frame.loc[predicted_mask, "라벨출처"] = "TFIDF_예측"
    return frame


def build_uncertainty_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    """§D: 지역·연도별 라벨출처 건수·예산액·전체예산 대비 비중을 집계한다."""
    frame = labeled.copy()
    frame["당해예산_유효"] = frame["당해예산"].fillna(0.0)

    counts = frame.groupby(["지역", "연도", "라벨출처"], as_index=False).agg(
        건수=("원본행", "size"), 예산액_백만원=("당해예산_유효", "sum")
    )
    totals = frame.groupby(["지역", "연도"], as_index=False)["당해예산_유효"].sum()
    totals = totals.rename(columns={"당해예산_유효": "전체예산_백만원"})
    summary = counts.merge(totals, on=["지역", "연도"], how="left")
    summary["예산비중"] = summary["예산액_백만원"] / summary["전체예산_백만원"].replace(0, pd.NA)
    return summary.sort_values(["연도", "지역", "라벨출처"]).reset_index(drop=True)


def build_outputs(
    review: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    qa_records: list[dict[str, object]] = []
    validate_schema_and_keys(review, qa_records)
    hard_failures = [record for record in qa_records if record["판정"] == "FAIL"]
    if hard_failures:
        raise ValueError(f"§A 입력검증 실패: {hard_failures}")

    labeled = select_labels(review)
    resolved = labeled.loc[labeled["세부영역"].notna()]
    unresolved = labeled.loc[labeled["세부영역"].isna()]

    _add_qa(qa_records, "§A 라벨선택", "라벨 확보 행 수", None, len(resolved), informational=True)
    _add_qa(qa_records, "§A 라벨선택", "미배정 행 수", None, len(unresolved), informational=True)
    _add_qa(
        qa_records,
        "§A 라벨선택",
        "확보+미배정 = 전체",
        len(labeled),
        len(resolved) + len(unresolved),
    )

    label_csv = resolved[LABEL_OUTPUT_COLUMNS].reset_index(drop=True)
    if label_csv.duplicated(["지역", "연도", "원본행"]).any():
        raise ValueError("라벨 CSV 키(지역·연도·원본행)가 중복됩니다.")

    uncertainty_summary = build_uncertainty_summary(labeled)
    unresolved_audit = unresolved[AUDIT_COLUMNS].reset_index(drop=True)

    qa = pd.DataFrame(qa_records)
    failures = qa.loc[qa["판정"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(f"§A 입력검증 실패: {failures.to_dict('records')}")

    return label_csv, uncertainty_summary, unresolved_audit, qa_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet-name", default=SHEET_NAME)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--qa-output", type=Path, default=QA_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review = load_review_workbook(args.workbook, args.sheet_name)
    label_csv, uncertainty_summary, unresolved_audit, qa_records = build_outputs(review)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)

    label_path = args.output_dir / "2016_2024_영역라벨.csv"
    uncertainty_path = args.output_dir / "2016_2024_라벨불확실성_요약.csv"
    unresolved_path = args.output_dir / "2016_2024_미배정_감사.csv"

    label_csv.to_csv(label_path, index=False, encoding="utf-8-sig")
    uncertainty_summary.to_csv(uncertainty_path, index=False, encoding="utf-8-sig")
    unresolved_audit.to_csv(unresolved_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(qa_records).to_csv(args.qa_output, index=False, encoding="utf-8-sig")

    print(
        f"라벨 CSV(run_provisional_pipeline.py --label-files 입력용): {label_path} ({len(label_csv):,}행)"
    )
    print(f"라벨 불확실성 요약: {uncertainty_path}")
    print(f"미배정 감사 목록: {unresolved_path} ({len(unresolved_audit):,}행)")
    print(f"QA: {args.qa_output}")


if __name__ == "__main__":
    main()
