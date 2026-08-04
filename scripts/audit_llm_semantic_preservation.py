"""연도별 LLM 정제 의미보존 감사 후보와 요약을 생성한다."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.features.llm_refine import (
    extract_quantities,
    numbers_preserved,
    semantic_preservation_flags,
)

DEFAULT_YEARS = tuple(range(2016, 2025))
REGIONS = (
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
)
REQUIRED_COLUMNS = {
    "연도",
    "지역",
    "원본행",
    "세부사업명",
    "주요내용",
    "주요내용_정제",
}


def _compact_length(value: object) -> int:
    if pd.isna(value):
        return 0
    return len(re.sub(r"\s+", "", str(value)))


def _length_ratio(original: object, cleaned: object) -> float | None:
    original_length = _compact_length(original)
    if not original_length:
        return None
    return round(_compact_length(cleaned) / original_length, 4)


def load_year_wide_files(
    year: int,
    *,
    data_root: Path,
) -> tuple[pd.DataFrame, list[str]]:
    """17개 지역 wide 파일을 읽고 누락 지역을 함께 반환한다."""
    frames: list[pd.DataFrame] = []
    missing_regions: list[str] = []

    for region in REGIONS:
        path = data_root / region / f"{year}_{region}_세부사업_정제.csv"
        if not path.exists():
            missing_regions.append(region)
            continue
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"원본행": "string"})
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise KeyError(f"{path}: 필수 컬럼 누락 {sorted(missing_columns)}")
        wrong_scope = frame.loc[
            frame["연도"].ne(year) | frame["지역"].ne(region),
            ["연도", "지역", "원본행"],
        ]
        if not wrong_scope.empty:
            raise ValueError(
                f"{path}: 경로와 파일 내용의 연도·지역 불일치 "
                f"{wrong_scope.head(10).to_dict('records')}"
            )
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"{year}년 wide 정제 파일을 찾지 못했습니다: {data_root}")

    combined = pd.concat(frames, ignore_index=True)
    return combined, missing_regions


def build_audit_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """wide 정제 데이터에서 의미보존 감사 후보만 추출한다."""
    key_columns = ["연도", "지역", "원본행"]
    duplicate_mask = frame.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_keys = frame.loc[duplicate_mask, key_columns].to_dict(orient="records")
        raise ValueError(f"연도·지역·원본행 키 중복: {duplicate_keys[:10]}")

    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        flags = semantic_preservation_flags(
            row["주요내용"],
            row["주요내용_정제"],
            context_terms=(str(row["세부사업명"]),),
        )
        if not flags:
            continue
        rows.append(
            {
                "연도": row["연도"],
                "지역": row["지역"],
                "원본행": row["원본행"],
                "세부사업명": row["세부사업명"],
                "주요내용": row["주요내용"],
                "주요내용_정제": row["주요내용_정제"],
                "자동검출사유": " | ".join(flags),
                "길이비": _length_ratio(row["주요내용"], row["주요내용_정제"]),
                "숫자보존": numbers_preserved(row["주요내용"], row["주요내용_정제"]),
                "수량보존": extract_quantities(row["주요내용"])
                == extract_quantities(row["주요내용_정제"]),
                "사람검토결과": "",
                "조치내용": "",
            }
        )

    columns = [
        "연도",
        "지역",
        "원본행",
        "세부사업명",
        "주요내용",
        "주요내용_정제",
        "자동검출사유",
        "길이비",
        "숫자보존",
        "수량보존",
        "사람검토결과",
        "조치내용",
    ]
    return pd.DataFrame(rows, columns=columns)


def audit_year(
    year: int,
    *,
    data_root: Path,
    reports_root: Path,
) -> dict[str, object]:
    """한 연도의 감사 후보 CSV를 저장하고 요약을 반환한다."""
    frame, missing_regions = load_year_wide_files(year, data_root=data_root)
    candidates = build_audit_candidates(frame)
    output_path = reports_root / str(year) / f"{year}_LLM_의미보존_감사.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False, encoding="utf-8-sig")

    reason_counts: dict[str, int] = {}
    for reasons in candidates["자동검출사유"]:
        for reason in str(reasons).split(" | "):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "year": year,
        "rows": len(frame),
        "candidate_rows": len(candidates),
        "candidate_rate": round(len(candidates) / len(frame), 6),
        "missing_regions": missing_regions,
        "reason_counts": reason_counts,
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports/yearly"))
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/20260731_LLM_의미보존_감사_요약.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [
        audit_year(year, data_root=args.data_root, reports_root=args.reports_root)
        for year in args.years
    ]
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
