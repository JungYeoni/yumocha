"""가족친화 인증기업 비율 2016년 17개 시도 실측값을 원자료에서 직접 계산한다.

`data/raw/.../가족친화인증기업_기관 명단-1_828개-2016년 말 기준-지역(시도_시군구) 포함.xlsx`는
2016년 말 기준 전체 유효 인증기업·기관 명단(시도·시군구 포함)이다. 총계 1,828개가 전국
실측 합계와 정확히 일치해 51건(17개 시도 × 2016·2017·2019) 중 2016년치(17건)를 raking/
구성비 추정이 아니라 **실측**으로 대체할 수 있다.

`OFFICIAL_ROSTERS[2016]`(build_family_friendly_candidates.py)에 이미 등록해 뒀으므로
`load_official_roster`를 그대로 재사용한다. `build_candidate_table`은 `TARGET_YEARS=(2020,
2021)`을 순회하도록 하드코딩돼 있어 2016년에는 그대로 못 쓰고, 이 모듈에 대응 버전을 둔다.

2017·2019년은 이 파일과 달리 "그 해 신규·연장·재인증 처리 건수" 공고문뿐이라(연말 기준
전체 명단 아님) 실측으로 못 바꾼다 — `20260806_가족친화_2017_2019_선정명단_사용불가_확인.md`
참고. 51건 중 2016년(17건)만 이 스크립트로 해소되고 나머지 34건은 기존 추정치를 유지한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.build_family_friendly_candidates import (
    INDICATOR_ID,
    OFFICIAL_ROSTERS,
    RAW_DIR,
    REGION_ORDER,
    load_denominator_source,
    load_official_roster,
)
from scripts.build_family_friendly_national_candidates import DENOMINATOR_2016_SOURCE

REPO_ROOT = Path(__file__).resolve().parents[1]
YEAR = 2016
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260806_가족친화_2016_지역별_직접복원_후보.csv"
QA_PATH = REPO_ROOT / "reports" / "20260806_가족친화_2016_지역별_직접복원_QA.csv"
REPORT_PATH = REPO_ROOT / "reports" / "methodology" / "20260806_가족친화_2016_지역별_직접복원_QA.md"


def build_2016_candidates(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    qa_records: list[dict[str, object]] = []
    _, region_counts = load_official_roster(raw_dir, YEAR, qa_records)
    denominators = load_denominator_source(
        raw_dir, DENOMINATOR_2016_SOURCE, (YEAR,), qa_records, "분모 2016(지역, 2016년 실측 반영)"
    )

    spec = OFFICIAL_ROSTERS[YEAR]
    rows = []
    for region in REGION_ORDER:
        numerator = int(region_counts.loc[region])
        denominator = int(denominators.loc[region, YEAR])
        rows.append(
            {
                "지역": region,
                "지표_id": INDICATOR_ID,
                "연도": YEAR,
                "공식_분자": numerator,
                "사업체수_분모": denominator,
                "측정값": numerator / denominator * 100,
                "분자_출처": spec["file_name"],
                "분모_출처": f"{DENOMINATOR_2016_SOURCE['file_name']}::열={YEAR};산업분류별=전체;규모별=전규모",
                "반영유형": "공식 원자료 직접 복원",
                "관측상태": "관측",
                "QA_상태": "PASS",
            }
        )
    candidates = pd.DataFrame(rows)

    def check(item: str, expected: object, actual: object) -> None:
        qa_records.append(
            {
                "구분": "2016 지역별 후보",
                "검증항목": item,
                "기대값": expected,
                "실제값": actual,
                "판정": "PASS" if expected == actual else "FAIL",
            }
        )

    check("행 수", 17, len(candidates))
    check("지역 집합", set(REGION_ORDER), set(candidates["지역"]))
    check("분자 합계", spec["total"], int(candidates["공식_분자"].sum()))
    check("분자 결측", 0, int(candidates["공식_분자"].isna().sum()))
    check("분모 결측", 0, int(candidates["사업체수_분모"].isna().sum()))
    check(
        "산식 일치",
        True,
        bool(
            (candidates["측정값"] - candidates["공식_분자"] / candidates["사업체수_분모"] * 100)
            .abs()
            .lt(1e-12)
            .all()
        ),
    )

    qa = pd.DataFrame(qa_records)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"2016 지역별 직접복원 QA 실패: {failures.to_dict('records')}")
    return candidates, qa


def apply_2016_candidates_to_panel(
    panel: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2016년 17개 시도 실측값을 28개 지표 롱 패널에 반영한다.

    기존 값이 결측이 아니면 예외를 내는 원칙은 `apply_national_observations_to_panel`/
    `apply_regional_confirmed_observations_to_panel`과 같지만, 이 배치는 17행(2016년
    지역만)이라 그 함수들의 하드코딩된 행 수(5, 37)와 안 맞아 대응 버전을 새로 둔다.
    """
    required = {"지역", "지표_id", "연도", "측정값"}
    missing_columns = required - set(panel.columns)
    if missing_columns:
        raise ValueError(f"패널 필수 열 누락: {sorted(missing_columns, key=str)}")

    keys = ["지역", "지표_id", "연도"]
    if len(candidates) != 17 or candidates.duplicated(keys).any():
        raise ValueError("2016 지역별 후보표는 중복 없는 17개 키여야 합니다.")
    if not candidates["연도"].eq(YEAR).all():
        raise ValueError(f"2016 지역별 후보표에 {YEAR}년 이외 연도가 포함됐습니다.")
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
            raise ValueError(f"2016 후보 반영 키가 1개가 아닙니다({matched}개): {row['지역']}")
        before = result.loc[mask, "측정값"].iloc[0]
        if pd.notna(before):
            raise ValueError(f"2016 직접복원 대상의 기존 패널값이 이미 존재합니다: {row['지역']}")
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
    """이제 실측이 생긴 2016년 17개 지역 키를 결측정책 매핑에서 제거한다."""
    keys = ["지역", "지표_id", "연도"]
    mask = (
        mapping["지역"].isin(candidates["지역"])
        & mapping["지표_id"].eq(INDICATOR_ID)
        & mapping["연도"].eq(YEAR)
    )
    matched = mapping.loc[mask]
    if len(matched) != 17:
        raise ValueError(f"매핑에서 제거 대상 2016년 17건과 일치하지 않습니다: {len(matched)}건")
    if not matched.merge(candidates[keys], on=keys, how="inner").shape[0] == 17:
        raise ValueError("매핑의 2016년 행 키가 후보표 키와 정확히 대응하지 않습니다.")
    if not (matched["block_imputation"] & matched["imputation_policy"].eq("pending_review")).all():
        raise ValueError("제거 대상 행이 예상한 pending_review/block_imputation 상태가 아닙니다.")

    return mapping.loc[~mask].reset_index(drop=True), matched.reset_index(drop=True)


def render_report(candidates: pd.DataFrame, qa: pd.DataFrame) -> str:
    lines = [
        "# 가족친화 인증기업 비율 2016년 지역별 직접 복원 QA",
        "",
        "## 범위",
        "",
        "- 원본: `가족친화인증기업_기관 명단-1_828개-2016년 말 기준-지역(시도_시군구) 포함.xlsx`"
        " (2016년 말 기준 전체 유효 명단, 총 1,828개)",
        "- 51건(17개 시도 × 2016·2017·2019) 중 2016년치 17건을 이 원자료로 실측 대체",
        "- 2017·2019년은 `20260806_가족친화_2017_2019_선정명단_사용불가_확인.md` 참고 —"
        " 그 해 처리 건수 공고뿐이라 실측 대체 불가, 기존 추정치 유지",
        "",
        "## 알려진 원자료 이상",
        "",
        "- 연번 549 (주)대흥에코: 시도=서울, 시군구=경기도 양주시로 원자료 자체가 불일치."
        " 집계용 전용 시도 열(서울)을 적용.",
        "- 연번 1794 충청남도 공주시청: 시군구 필드가 '충정남도'로 오탈자. 시도 필드(충남)는"
        " 정상이라 집계에 영향 없음.",
        "",
        "## 결과",
        "",
        "| 지역 | 공식 분자 | 사업체수 분모 | 측정값(%) |",
        "|---|---:|---:|---:|",
    ]
    for row in candidates.sort_values("지역").itertuples():
        lines.append(
            f"| {row.지역} | {row.공식_분자:,} | {row.사업체수_분모:,} | {row.측정값:.4f} |"
        )
    lines.extend(
        [
            "",
            "## QA",
            "",
            f"- QA 항목 {len(qa):,}개 모두 PASS",
        ]
    )
    return "\n".join(lines) + "\n"


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
        help="28개 지표 통합 패널 CSV. 존재할 때만 2016년 17건을 반영한다(기본: 반영 안 함).",
    )
    parser.add_argument("--panel-output", type=Path, default=None)
    parser.add_argument(
        "--panel-audit-output",
        type=Path,
        default=Path("reports/20260806_가족친화_2016_패널반영_감사.csv"),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="결측정책 전수매핑 CSV. 존재할 때만 해소된 2016년 17행을 제거한다(기본: 반영 안 함).",
    )
    parser.add_argument("--mapping-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, qa = build_2016_candidates(args.raw_dir)

    for path in (args.candidate_output, args.qa_output, args.report_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidate_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    args.report_output.write_text(render_report(candidates, qa), encoding="utf-8")

    print(f"2016 지역별 후보: {len(candidates)}건 ({args.candidate_output})")
    print(f"QA {len(qa)}개 항목 모두 PASS")

    if args.panel is not None and not args.panel.exists():
        raise FileNotFoundError(f"--panel 경로가 존재하지 않습니다: {args.panel}")
    if args.panel is not None and args.panel.exists():
        panel = pd.read_csv(args.panel, encoding="utf-8-sig")
        updated_panel, panel_audit = apply_2016_candidates_to_panel(panel, candidates)
        panel_output = args.panel_output or args.panel
        args.panel_audit_output.parent.mkdir(parents=True, exist_ok=True)
        updated_panel.to_csv(panel_output, index=False, encoding="utf-8-sig")
        panel_audit.to_csv(args.panel_audit_output, index=False, encoding="utf-8-sig")
        print(f"패널 반영 완료: {panel_output} (감사: {args.panel_audit_output})")

    if args.mapping is not None and not args.mapping.exists():
        raise FileNotFoundError(f"--mapping 경로가 존재하지 않습니다: {args.mapping}")
    if args.mapping is not None and args.mapping.exists():
        mapping = pd.read_csv(args.mapping, encoding="utf-8-sig")
        updated_mapping, removed = remove_resolved_rows_from_mapping(mapping, candidates)
        mapping_output = args.mapping_output or args.mapping
        updated_mapping.to_csv(mapping_output, index=False, encoding="utf-8-sig")
        print(f"매핑 갱신 완료: {mapping_output} ({len(removed)}행 제거)")


if __name__ == "__main__":
    main()
