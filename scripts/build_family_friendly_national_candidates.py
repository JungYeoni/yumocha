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

It never touches the 37-row confirmed/candidate tables produced by
`build_family_friendly_candidates.py`. Merging these 5 values into the integrated panel and the
missing-policy mapping is optional and only runs when `--panel`/`--mapping` are passed and exist
(as of 2026-08-05 neither file exists on every machine — the panel is a local, gitignored,
notebook-generated artifact) — see `apply_national_observations_to_panel` and
`remove_resolved_rows_from_mapping`.
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
    """전국 사업체수 분모를 연도별로 모은다. 2016은 별도 워크북, 나머지는 기존 분모 소스다.

    같은 워크북(`DENOMINATOR_SOURCE`) 안에서 2019(2,146,156) → 2020(1,865,536) 전국
    사업체수가 약 13% 줄어든다(2021엔 1,995,751로 다시 늘어남). 통계청 KOSIS 원자료를
    그대로 읽은 값이고 열 정렬·처리 오류는 아님을 확인했지만(같은 함수로 2020·2021
    시도별 합계가 이미 검증된 공식 명단 합계와 정확히 일치, `build_national_candidates`의
    "전국-지역합 교차검증" 참고), 조사 범위·분류 기준이 실제로 바뀐 것인지는 KOSIS
    통계설명자료로 확인하지 못했다 — 이 사업체수 시계열을 그대로 비교 지표로 쓸 때는
    이 단절 가능성을 감안해야 한다.
    """
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


def apply_national_observations_to_panel(
    panel: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the 5 verified national observations to the 28-indicator long panel.

    `panel` must be the long-format `[지역, 지표_id, 연도, 측정값, ...]` table consumed by
    `build_structural_missing_policy.py` (`REQUIRED_PANEL_COLUMNS`), not the 21-indicator wide
    table `apply_confirmed_observations` in `build_family_friendly_candidates.py` operates on.
    Only currently-missing cells may be filled; an existing non-missing value that disagrees
    with the candidate raises, mirroring the discipline of that province-level function.
    """
    required = {"지역", "지표_id", "연도", "측정값"}
    missing_columns = required - set(panel.columns)
    if missing_columns:
        raise ValueError(f"패널 필수 열 누락: {sorted(missing_columns, key=str)}")
    keys = ["지역", "지표_id", "연도"]
    if len(candidates) != 5 or candidates.duplicated(keys).any():
        raise ValueError("전국 후보표는 중복 없는 5개 키여야 합니다.")
    if not candidates["지역"].eq("전국").all():
        raise ValueError("전국 후보표에 전국 이외 지역이 포함됐습니다.")
    if not candidates["QA_상태"].eq("PASS").all():
        raise ValueError("QA를 통과하지 않은 후보값은 반영할 수 없습니다.")

    result = panel.copy()
    audit_rows = []
    for row in candidates.to_dict("records"):
        mask = (
            result["지역"].eq(row["지역"])
            & result["지표_id"].eq(row["지표_id"])
            & result["연도"].eq(row["연도"])
        )
        matched = int(mask.sum())
        if matched != 1:
            raise ValueError(
                f"전국 후보 반영 키가 1개가 아닙니다({matched}개): {row['지역']}, {row['연도']}"
            )
        before = result.loc[mask, "측정값"].iloc[0]
        acceptable = pd.isna(before) or np.isclose(
            float(before), float(row["측정값"]), rtol=0, atol=1e-15
        )
        if not acceptable:
            raise ValueError(f"전국 직접 복원 대상의 기존 패널값이 예상과 다릅니다: {row['연도']}")
        result.loc[mask, "측정값"] = float(row["측정값"])
        if "원본행존재" in result.columns:
            result.loc[mask, "원본행존재"] = True
        if "관측상태" in result.columns:
            result.loc[mask, "관측상태"] = row["관측상태"]
        audit_rows.append(
            {
                "지역": row["지역"],
                "지표_id": row["지표_id"],
                "연도": row["연도"],
                "반영전값": before,
                "반영후값": float(row["측정값"]),
                "반영유형": row["반영유형"],
                "관측상태": row["관측상태"],
                "QA_상태": "PASS",
            }
        )
    return result, pd.DataFrame(audit_rows)


def remove_resolved_rows_from_mapping(
    mapping: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop the 5 now-observed national keys from the missing-policy mapping.

    Observed cells are not represented in the mapping at all (the 34 already-restored
    province-level 2020/2021 rows follow the same pattern), so once the panel has real values
    for these 5 keys, the mapping's matching rows must disappear rather than keep a stale
    `pending_review` policy. Raises if the mapping doesn't have exactly the 5 expected rows
    already blocked, so this cannot silently drop something unexpected.
    """
    keys = ["지역", "지표_id", "연도"]
    candidate_keys = candidates[keys]
    mask = (
        mapping["지역"].eq("전국")
        & mapping["지표_id"].eq(INDICATOR_ID)
        & mapping["연도"].isin(candidates["연도"])
    )
    matched = mapping.loc[mask]
    if len(matched) != 5:
        raise ValueError(f"매핑에서 제거 대상 전국 5건과 일치하지 않습니다: {len(matched)}건")
    if not matched.merge(candidate_keys, on=keys, how="inner").shape[0] == 5:
        raise ValueError("매핑의 전국 행 키가 후보표 키와 정확히 대응하지 않습니다.")
    if not (matched["block_imputation"] & matched["imputation_policy"].eq("pending_review")).all():
        raise ValueError("제거 대상 행이 예상한 pending_review/block_imputation 상태가 아닙니다.")

    return mapping.loc[~mask].reset_index(drop=True), matched.reset_index(drop=True)


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
            "- `--panel`/`--mapping`을 실제 파일 경로로 넘기면 "
            "`apply_national_observations_to_panel`/`remove_resolved_rows_from_mapping`으로 "
            "이 5건을 실제 패널·매핑에 반영한다. 인자를 생략하면 후보 CSV만 생성하고 "
            "패널·매핑은 건드리지 않는다(2026-08-05 기준 패널 파일은 이 저장소의 모든 "
            "환경에 있지 않은 gitignored 로컬 산출물이다).",
        ]
    )
    return "\n".join(lines) + "\n"


PANEL_AUDIT_PATH = REPO_ROOT / "reports" / "20260805_가족친화_전국_패널반영_QA.csv"
MAPPING_REMOVED_PATH = REPO_ROOT / "reports" / "20260805_가족친화_전국_매핑제거_QA.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="28개 지표 통합 패널 CSV. 존재할 때만 전국 5건을 반영한다(기본: 반영 안 함).",
    )
    parser.add_argument(
        "--panel-output",
        type=Path,
        default=None,
        help="갱신된 패널 저장 경로. 생략하면 --panel 경로를 덮어쓴다.",
    )
    parser.add_argument("--panel-audit-output", type=Path, default=PANEL_AUDIT_PATH)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="결측정책 전수매핑 CSV. 존재할 때만 해소된 전국 5행을 제거한다(기본: 반영 안 함).",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="갱신된 매핑 저장 경로. 생략하면 --mapping 경로를 덮어쓴다.",
    )
    parser.add_argument("--mapping-removed-output", type=Path, default=MAPPING_REMOVED_PATH)
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

    if args.panel is not None and not args.panel.exists():
        raise FileNotFoundError(f"--panel 경로가 존재하지 않습니다: {args.panel}")
    if args.panel is not None and args.panel.exists():
        panel = pd.read_csv(args.panel)
        updated_panel, panel_audit = apply_national_observations_to_panel(panel, candidates)
        panel_output = args.panel_output or args.panel
        args.panel_audit_output.parent.mkdir(parents=True, exist_ok=True)
        updated_panel.to_csv(panel_output, index=False, encoding="utf-8-sig")
        panel_audit.to_csv(args.panel_audit_output, index=False, encoding="utf-8-sig")
        print(f"패널 반영 완료: {panel_output} ({len(panel_audit)}행 갱신)")
        print(f"패널 반영 감사표: {args.panel_audit_output}")
    else:
        print("패널 인자 없음/미존재 — 패널 반영 생략")

    if args.mapping is not None and not args.mapping.exists():
        raise FileNotFoundError(f"--mapping 경로가 존재하지 않습니다: {args.mapping}")
    if args.mapping is not None and args.mapping.exists():
        mapping = pd.read_csv(args.mapping)
        updated_mapping, removed = remove_resolved_rows_from_mapping(mapping, candidates)
        mapping_output = args.mapping_output or args.mapping
        args.mapping_removed_output.parent.mkdir(parents=True, exist_ok=True)
        updated_mapping.to_csv(mapping_output, index=False, encoding="utf-8-sig")
        removed.to_csv(args.mapping_removed_output, index=False, encoding="utf-8-sig")
        print(f"매핑 갱신 완료: {mapping_output} ({len(removed)}행 제거)")
        print(f"매핑 제거 감사표: {args.mapping_removed_output}")
    else:
        print("매핑 인자 없음/미존재 — 매핑 갱신 생략")


if __name__ == "__main__":
    main()
