"""Build the national (전국) direct-observation candidates for the family-friendly indicator.

The 17-province panel already has 34 directly-restored 2020/2021 observations and 2024/2018/2022/2023
reference values, but the panel's separate "전국" rows for 2016·2017·2019·2020·2021 remained
`pending_review` — the province-level restoration never touched them
(see `reports/20260804_가족친화_2020_2021_직접복원_QA.md`, "전국 가족친화 2020·2021은 이번 34건 직접
복원 범위에 포함되지 않는다").

This script fills those 5 national rows from two independently verifiable official sources:

- Numerator: the cumulative-by-year table published on the certification body's own status page
  (`https://www.ffsb.kr/ffm/ffmCertStatus.do`), which reports year-end cumulative valid
  certification counts. Its 2018/2020/2021/2022/2023/2024 values match the province-level
  official rosters already used in the pipeline exactly, which is the evidence this table uses
  the same "연말 기준 유효 인증기업 수" definition as those rosters (not a raw registration count).
- Denominator: the same KOSIS 사업체수 workbooks already used for the province-level ratios
  (`DENOMINATOR_SOURCE`), plus a 2016-only workbook that was not previously wired into this
  pipeline.

It never touches the panel or the 37-row confirmed/candidate tables produced by
`build_family_friendly_candidates.py`; it only emits a new, separately-scoped artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_family_friendly_candidates import (
    DENOMINATOR_SOURCE,
    INDICATOR_ID,
    OFFICIAL_ROSTERS,
    RAW_DIR,
    add_qa,
    load_denominator_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260805_가족친화_전국_직접복원_후보.csv"
QA_PATH = REPO_ROOT / "reports" / "20260805_가족친화_전국_직접복원_QA.csv"
REPORT_PATH = REPO_ROOT / "reports" / "methodology" / "20260805_가족친화_전국_직접복원_QA.md"

DENOMINATOR_2016_SOURCE = {
    "file_name": (
        "2016_시도별__산업별__규모별__사업체수_및_종사자수_성별__20260715194020.csv.xlsx"
    ),
    "file_size": 11_156,
    "sha256": "fe69a2b1c4903ad408904a8ba185fa7f1bc51fa74f289492f6be071a181e0acb",
}

NATIONAL_CUMULATIVE_SOURCE = {
    "provider": "가족친화지원사업(한국건강가정진흥원)",
    "url": "https://www.ffsb.kr/ffm/ffmCertStatus.do",
    "table_caption": "가족친화인증기업(관) 현황 — 연도별 누계",
    "accessed_at": "2026-08-05T02:36:26+09:00",
}
NATIONAL_CUMULATIVE_TOTALS = {2016: 1_828, 2017: 2_802, 2019: 3_833, 2020: 4_340, 2021: 4_918}
TARGET_YEARS = (2016, 2017, 2019, 2020, 2021)


def build_national_denominators(
    raw_dir: Path, qa_records: list[dict[str, object]]
) -> dict[int, int]:
    """전국 사업체수 분모를 연도별로 모은다. 2016은 별도 워크북, 나머지는 기존 분모 소스다."""
    denom_2016 = load_denominator_source(
        raw_dir, DENOMINATOR_2016_SOURCE, (2016,), qa_records, "2016 분모(전국)"
    )
    denom_rest = load_denominator_source(
        raw_dir, DENOMINATOR_SOURCE, (2017, 2019, 2020, 2021), qa_records, "2017-2021 분모(전국)"
    )
    denominators = {2016: int(denom_2016.loc["전국", 2016])}
    for year in (2017, 2019, 2020, 2021):
        denominators[year] = int(denom_rest.loc["전국", year])
    return denominators


def build_national_candidates(
    denominators: dict[int, int], qa_records: list[dict[str, object]]
) -> pd.DataFrame:
    rows = []
    for year in TARGET_YEARS:
        numerator = NATIONAL_CUMULATIVE_TOTALS[year]
        denominator = denominators[year]
        rows.append(
            {
                "지역": "전국",
                "지표_id": INDICATOR_ID,
                "연도": year,
                "공식_분자": numerator,
                "사업체수_분모": denominator,
                "측정값": numerator / denominator * 100,
                "분자_출처": (
                    f"{NATIONAL_CUMULATIVE_SOURCE['url']} "
                    f"({NATIONAL_CUMULATIVE_SOURCE['table_caption']})"
                ),
                "분모_출처": (
                    f"{DENOMINATOR_2016_SOURCE['file_name'] if year == 2016 else DENOMINATOR_SOURCE['file_name']}"
                    f"::열={year};산업분류별=전체;규모별=전규모;시도별(17개)=전국"
                ),
                "반영유형": "공식 누계 통계 직접 복원",
                "관측상태": "관측",
                "QA_상태": "PASS",
            }
        )
    candidates = pd.DataFrame(rows)

    section = "전국 5개 후보"
    add_qa(qa_records, section, "행 수", 5, len(candidates))
    add_qa(qa_records, section, "연도", list(TARGET_YEARS), candidates["연도"].tolist())
    add_qa(qa_records, section, "지역", ["전국"] * 5, candidates["지역"].tolist())
    formula = candidates["공식_분자"] / candidates["사업체수_분모"] * 100
    add_qa(
        qa_records,
        section,
        "비율 산식 일치",
        5,
        int(np.isclose(candidates["측정값"], formula, rtol=0, atol=1e-15).sum()),
    )

    # Cross-check: for years already resolved at province level (2020, 2021), the cumulative
    # table's national total must equal the same official roster's already-confirmed total.
    for year in (2020, 2021):
        add_qa(
            qa_records,
            "전국-지역합 교차검증",
            f"{year} 누계표 대 공식 명단 합계",
            OFFICIAL_ROSTERS[year]["total"],
            NATIONAL_CUMULATIVE_TOTALS[year],
        )

    return candidates


def render_report(candidates: pd.DataFrame, qa: pd.DataFrame) -> str:
    lines = [
        "# 가족친화 인증기업 비율 전국 직접 복원 QA",
        "",
        "## 범위",
        "",
        "- 17개 시도 34건 직접 복원과 별개로 결측정책에서 `pending_review`로 남아 있던 "
        "'전국' 5개 행(2016·2017·2019·2020·2021)을 채운다.",
        "- 산식: `(연도 말 유효 인증기업·기관 수 ÷ 같은 연도 전산업·전규모 사업체 수) × 100`",
        "",
        "## 공식 원본",
        "",
        f"- 분자: [{NATIONAL_CUMULATIVE_SOURCE['table_caption']}]"
        f"({NATIONAL_CUMULATIVE_SOURCE['url']}), 접속 {NATIONAL_CUMULATIVE_SOURCE['accessed_at']}",
        f"- 2016 분모: `{DENOMINATOR_2016_SOURCE['file_name']}`, "
        f"{DENOMINATOR_2016_SOURCE['file_size']:,} bytes, SHA-256 `{DENOMINATOR_2016_SOURCE['sha256']}`",
        f"- 2017·2019·2020·2021 분모: `{DENOMINATOR_SOURCE['file_name']}`, "
        f"SHA-256 `{DENOMINATOR_SOURCE['sha256']}`",
        "",
        "## 복원값",
        "",
        "| 연도 | 공식 분자(누계) | 전국 사업체수 분모 | 비율(%) |",
        "|---:|---:|---:|---:|",
    ]
    for _, row in candidates.iterrows():
        lines.append(
            f"| {int(row['연도'])} | {int(row['공식_분자']):,} | {int(row['사업체수_분모']):,} | "
            f"{row['측정값']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 교차검증",
            "",
            "- 2020·2021 누계표 값은 이미 확정된 공식 명단 전국 합계(4,340 / 4,918)와 정확히 일치한다. "
            "이는 이 누계표가 province-level 공식 명단과 같은 '연말 기준 유효 인증기업 수' 정의를 "
            "쓴다는 근거다.",
            "- 2018·2022·2023·2024 값도 기존 REFERENCE_ROSTERS/OFFICIAL_ROSTERS 합계와 별도로 "
            "정확히 일치했다(본 스크립트 범위 밖, 대화 중 수기 대조).",
            "",
            "## QA 요약",
            "",
            f"- QA 항목: {len(qa)}개",
            f"- PASS: {int(qa['판정'].eq('PASS').sum())}개",
            f"- FAIL: {int(qa['판정'].eq('FAIL').sum())}개",
            "",
            "## 남은 범위",
            "",
            "- 17개 시도 × 2016·2017·2019 = 51건은 이 스크립트로 풀리지 않는다. "
            "이 표는 전국 합계 1개 값이라 지역별 분해가 없다.",
            "- 이 후보 CSV는 결측정책 매핑이나 통합 패널에 아직 병합되지 않았다. "
            "병합은 별도 검토 후 진행한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_records: list[dict[str, object]] = []

    denominators = build_national_denominators(args.raw_dir, qa_records)
    candidates = build_national_candidates(denominators, qa_records)

    qa = pd.DataFrame(qa_records)
    if not qa["판정"].eq("PASS").all():
        failed = qa.loc[qa["판정"].eq("FAIL")]
        raise ValueError(f"QA 실패:\n{failed.to_string(index=False)}")

    for output in (args.candidate_output, args.qa_output, args.report_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        args.candidate_output, index=False, encoding="utf-8-sig", float_format="%.16f"
    )
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    args.report_output.write_text(render_report(candidates, qa), encoding="utf-8")

    print(f"전국 후보 저장: {args.candidate_output} ({len(candidates)}행)")
    print(f"QA 저장: {args.qa_output} ({len(qa)}개 PASS)")
    print(f"보고서 저장: {args.report_output}")


if __name__ == "__main__":
    main()
