"""의미보존 오류로 정제값이 바뀐 행의 라벨·예산 영향표를 생성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.apply_2021_semantic_corrections import CORRECTIONS

KEY_COLUMNS = ["연도", "지역", "원본행"]
BUDGET_COLUMNS = ["사업분류재정구분", "당해예산", "전년도예산", "증감액", "증감율"]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        keep_default_na=False,
        dtype={"원본행": "string"},
    )


def _standardize(
    frame: pd.DataFrame,
    *,
    before_column: str,
    after_column: str,
    result_column: str,
    reason_column: str,
) -> pd.DataFrame:
    output = frame[
        [
            "연도",
            "지역",
            "원본행",
            "세부사업명",
            "주요내용",
            before_column,
            after_column,
            result_column,
            reason_column,
        ]
    ].copy()
    return output.rename(
        columns={
            before_column: "수정전_주요내용_정제",
            after_column: "수정후_주요내용_정제",
            result_column: "변경유형",
            reason_column: "판단근거",
        }
    )


def collect_changed_rows(reports_root: Path) -> pd.DataFrame:
    """연도별 검토표에서 실제 정제값 변경 행 79건을 통합한다."""
    standard_frames: list[pd.DataFrame] = []

    for year in (2020, 2023, 2024):
        frame = _read_csv(reports_root / "yearly" / str(year) / f"{year}_LLM_의미보존_전수검토.csv")
        frame = frame.loc[frame["TFIDF_영향"].str.contains("예측 입력 변경")].copy()
        standard_frames.append(
            _standardize(
                frame,
                before_column="수정전_주요내용_정제",
                after_column="수정후_주요내용_정제",
                result_column="사람검토결과",
                reason_column="조치내용",
            )
        )

    blank_2023 = _read_csv(reports_root / "yearly/2023/2023_LLM_빈원문_환각_복구검토.csv")
    standard_frames.append(
        _standardize(
            blank_2023,
            before_column="복구전_주요내용_정제",
            after_column="복구후_주요내용_정제",
            result_column="검토결과",
            reason_column="판단근거",
        )
    )

    rows_2021 = pd.DataFrame(
        [
            {
                "연도": 2021,
                "지역": correction.region,
                "원본행": correction.source_row,
                "세부사업명": correction.name,
                "주요내용": correction.expected_original,
                "수정전_주요내용_정제": correction.expected_cleaned,
                "수정후_주요내용_정제": correction.corrected_cleaned,
                "변경유형": "의미보존 오류 수정",
                "판단근거": correction.reason,
            }
            for correction in CORRECTIONS
        ]
    )
    rows_2021["원본행"] = rows_2021["원본행"].astype("string")
    standard_frames.append(rows_2021)

    combined = pd.concat(standard_frames, ignore_index=True)
    combined["연도"] = pd.to_numeric(combined["연도"], errors="raise").astype("int64")
    combined["원본행"] = combined["원본행"].astype("string")
    if len(combined) != 79:
        raise ValueError(f"정제값 변경 행은 79건이어야 합니다: {len(combined)}")
    if combined.duplicated(KEY_COLUMNS).any():
        duplicate = combined.loc[combined.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS]
        raise ValueError(f"정제값 변경 키 중복: {duplicate.to_dict('records')}")
    return combined.sort_values(KEY_COLUMNS).reset_index(drop=True)


def attach_latest_budget(changed: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """최신 wide를 키로 연결하고 정제문·예산의 일대일 정합성을 검증한다."""
    source_frames: list[pd.DataFrame] = []
    for (year, region), _ in changed.groupby(["연도", "지역"], sort=True):
        path = data_root / str(region) / f"{year}_{region}_세부사업_정제.csv"
        source = _read_csv(path)
        source["연도"] = pd.to_numeric(source["연도"], errors="raise").astype("int64")
        source_frames.append(source[KEY_COLUMNS + ["세부사업명", "주요내용_정제", *BUDGET_COLUMNS]])

    latest = pd.concat(source_frames, ignore_index=True)
    if latest.duplicated(KEY_COLUMNS).any():
        raise ValueError("최신 wide에 연도·지역·원본행 중복이 있습니다.")
    merged = changed.merge(
        latest,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_최신"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        missing = merged.loc[merged["_merge"].ne("both"), KEY_COLUMNS]
        raise ValueError(f"최신 wide 연결 누락: {missing.to_dict('records')}")
    if not merged["세부사업명"].eq(merged["세부사업명_최신"]).all():
        raise ValueError("최신 wide 세부사업명이 변경 행 검토표와 다릅니다.")
    if not merged["수정후_주요내용_정제"].eq(merged["주요내용_정제"]).all():
        raise ValueError("최신 wide 정제문이 수정후 정제문과 다릅니다.")

    merged["예산연결상태"] = "키 일치·예산 유지"
    return merged.drop(columns=["_merge", "세부사업명_최신", "주요내용_정제"])


def attach_2021_labels(impact: pd.DataFrame, reports_root: Path) -> pd.DataFrame:
    """2021년 수정 4건의 기존 라벨 유지 판정을 함께 보존한다."""
    label_impact = _read_csv(
        reports_root / "yearly/2021/2021_LLM_정제변경_라벨영향검토.csv"
    ).rename(
        columns={
            "대영역": "기존_대영역",
            "세부영역": "기존_세부영역",
            "라벨검토결과": "라벨영향검토결과",
            "판단근거": "라벨판단근거",
        }
    )
    label_impact["연도"] = 2021
    columns = [
        *KEY_COLUMNS,
        "기존_대영역",
        "기존_세부영역",
        "라벨영향검토결과",
        "라벨판단근거",
    ]
    if label_impact.duplicated(KEY_COLUMNS).any():
        raise ValueError("2021년 라벨 영향표에 키 중복이 있습니다.")
    output = impact.merge(
        label_impact[columns],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    rows_2021 = output["연도"].eq(2021)
    if not output.loc[rows_2021, "라벨영향검토결과"].eq("기존 라벨 유지").all():
        raise ValueError("2021년 변경 4건의 라벨 영향 검토가 완료되지 않았습니다.")
    output.loc[~rows_2021, "라벨영향검토결과"] = "재예측 후 변경 시 검토"
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/20260731_LLM_정제변경_라벨예산_영향검토.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changed = collect_changed_rows(args.reports_root)
    impact = attach_latest_budget(changed, args.data_root)
    impact = attach_2021_labels(impact, args.reports_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    impact.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"정제값 변경 영향표: {len(impact):,}건")
    print("최신 wide 키 연결·정제문 일치·2021 기존 라벨 유지 확인")


if __name__ == "__main__":
    main()
