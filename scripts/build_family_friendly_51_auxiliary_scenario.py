"""가족친화 인증기업 비율 17개 시도 34건(2017·2019)의 raking 추정치를 계산하고 본계열에 반영한다.

`reports/methodology/20260805_가족친화_지역별_51건_보간_방법_제안.md`에서 제안한 방법을
실제로 계산한다. 이 34건은 공식 명단을 확보하지 못해 `pending_review`/
`block_imputation=True`로 본계열에서 차단돼 있었으나, 이슈 #70 팀 논의 결과 raking
추정치를 본계열로 채택하기로 했다(2026-08-06). `apply_raking_candidates_to_panel`/
`remove_resolved_rows_from_mapping`이 `--panel`/`--mapping` 인자로 실제 패널·매핑에
반영한다 — 실측이 아니라 추정치라는 점을 남기기 위해 반영 후에도 `관측상태`는 "추정"으로
기록하고(2016년 실측 반영 시 "관측"으로 기록한 것과 구분), `반영유형`에 raking임을
명시한다.

**2026-08-06 갱신**: 원래 51건(2016·2017·2019) 중 2016년치는 "2016년 말 기준" 전체
유효 명단을 확보해 실측으로 대체했다(`build_family_friendly_2016_regional_candidates.py`).
그 결과 2017년도 더 이상 "앞이 텅 빈 선행 결측"이 아니라 2016(실측)·2018(실측) 사이에 낀
"중간 결측"이 되어, 2019년과 같은 raking 방식(2016·2018 선형보간 후 전국 실측 합계로
비례 보정)으로 다시 계산했다 — 2018년 구성비를 그대로 빌려쓰던 이전 방식(`composition_ratio_2018`)
보다 실제 2016→2018 추세를 반영하므로 더 정확하다. 이전 51건(2016 포함) 산출물은
`reports/20260805_가족친화_지역별_51건_보조시나리오_후보.csv`에 비교용으로 남겨뒀다.

- 2017(중간 결측, 신규): 2016·2018 지역별 분자(공식 인증기업 수)로 선형보간한 뒤, 보간된
  17개 지역 분자의 합을 전국 실측 분자 합계(2,802)에 맞춰 비례 보정(raking)한다.
- 2019(중간 결측, 기존과 동일): 2018·2020 지역별 분자로 선형보간한 뒤 전국 실측 분자
  합계(3,833)로 raking한다.

두 경우 모두 분모(사업체수)는 해당 연도의 실제 지역별 사업체수를 그대로 쓰고, 분자만
추정한 뒤 최종 비율 = 추정 분자 / 지역별 분모 × 100으로 계산한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.build_family_friendly_2016_regional_candidates import (
    CANDIDATE_PATH as CANDIDATE_2016_PATH,
)
from scripts.build_family_friendly_candidates import (
    CONFIRMED_PATH,
    INDICATOR_ID,
    RAW_DIR,
    REGION_ORDER,
    add_qa,
    load_denominators,
    load_reference_counts,
)
from scripts.build_family_friendly_national_candidates import NATIONAL_CUMULATIVE_TOTALS

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = REPO_ROOT / "reports" / "20260806_가족친화_지역별_34건_보조시나리오_후보.csv"
QA_PATH = REPO_ROOT / "reports" / "20260806_가족친화_지역별_34건_보조시나리오_QA.csv"

RAKING_YEARS = (2017, 2019)
RAKING_METHOD = "raking"


def load_2016_confirmed_counts(candidate_2016_path: Path) -> pd.Series:
    """2016년 지역별 실측 복원 후보표에서 지역별 분자(공식 원자료 직접 복원)를 뽑는다."""
    confirmed = pd.read_csv(candidate_2016_path, encoding="utf-8-sig")
    if len(confirmed) != 17 or confirmed["지역"].duplicated().any():
        raise ValueError(f"2016년 지역별 분자가 17개 시도와 일치하지 않습니다: {len(confirmed)}건")
    return confirmed.set_index("지역")["공식_분자"].astype(int).reindex(REGION_ORDER)


def load_2018_reference_counts(raw_dir: Path, qa_records: list[dict[str, object]]) -> pd.Series:
    """2018년 지역별 공식 명단 재집계 분자(인증기업 수)를 불러온다."""
    counts = load_reference_counts(raw_dir, 2018)
    add_qa(qa_records, "2018 분자", "지역 수", 17, len(counts))
    add_qa(qa_records, "2018 분자", "합계", 3_328, int(counts.sum()))
    return counts.reindex(REGION_ORDER)


def load_2020_confirmed_counts(confirmed_path: Path) -> pd.Series:
    """37건 공식 관측 반영표에서 2020년 지역별 분자(공식 원자료 직접 복원)만 뽑는다."""
    confirmed = pd.read_csv(confirmed_path, encoding="utf-8-sig")
    selected = confirmed.loc[
        confirmed["지표_id"].eq(INDICATOR_ID)
        & confirmed["연도"].eq(2020)
        & confirmed["반영유형"].eq("공식 원자료 직접 복원")
    ]
    if len(selected) != 17 or selected["지역"].duplicated().any():
        raise ValueError(f"2020년 지역별 분자가 17개 시도와 일치하지 않습니다: {len(selected)}건")
    return selected.set_index("지역")["공식_분자"].astype(int).reindex(REGION_ORDER)


def build_raking_candidates(
    year: int,
    count_before: pd.Series,
    count_after: pd.Series,
    denominators: pd.DataFrame,
    qa_records: list[dict[str, object]],
    *,
    before_year: int,
    after_year: int,
) -> pd.DataFrame:
    """지역별 분자를 앞뒤 실측 연도로 선형보간한 뒤 전국 실측 합계로 raking한다."""
    interpolated = (count_before + count_after) / 2.0
    target_total = NATIONAL_CUMULATIVE_TOTALS[year]
    raked = interpolated * (target_total / interpolated.sum())
    denom = denominators[year]
    ratio = raked / denom * 100

    section = f"{year} raking"
    add_qa(qa_records, section, "raked 분자 합계", target_total, round(float(raked.sum()), 6))
    add_qa(qa_records, section, "지역 수", len(REGION_ORDER), len(raked))
    add_qa(qa_records, section, "음수·NA 없음", 0, int((raked < 0).sum() + raked.isna().sum()))

    return pd.DataFrame(
        {
            "지역": REGION_ORDER,
            "지표_id": INDICATOR_ID,
            "연도": year,
            "추정_분자": raked.reindex(REGION_ORDER).to_numpy(),
            "사업체수_분모": denom.reindex(REGION_ORDER).to_numpy(),
            "추정_비율": ratio.reindex(REGION_ORDER).to_numpy(),
            "방법": RAKING_METHOD,
            "근거": (
                f"{before_year}·{after_year} 지역 분자 선형보간 후 전국 실측 분자 합계"
                f"({target_total:,})로 raking"
            ),
            "반영유형": "raking 추정치 반영(본계열, 이슈 #70 팀 결정 2026-08-06)",
            "관측상태": "추정",
            "QA_상태": "PASS",
        }
    )


def build_candidates(
    raw_dir: Path, confirmed_path: Path, candidate_2016_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qa_records: list[dict[str, object]] = []
    count_2016 = load_2016_confirmed_counts(candidate_2016_path)
    count_2018 = load_2018_reference_counts(raw_dir, qa_records)
    count_2020 = load_2020_confirmed_counts(confirmed_path)
    denominators = load_denominators(raw_dir, (2017, 2019), qa_records).reindex(REGION_ORDER)

    raking_2017 = build_raking_candidates(
        2017, count_2016, count_2018, denominators, qa_records, before_year=2016, after_year=2018
    )
    raking_2019 = build_raking_candidates(
        2019, count_2018, count_2020, denominators, qa_records, before_year=2018, after_year=2020
    )
    candidates = pd.concat([raking_2017, raking_2019], ignore_index=True).sort_values(
        ["연도", "지역"]
    )

    add_qa(qa_records, "전체", "총 행 수", 34, len(candidates))
    add_qa(
        qa_records, "전체", "고유 지역×연도", 34, len(candidates.drop_duplicates(["지역", "연도"]))
    )
    add_qa(
        qa_records,
        "전체",
        "음수·NA 비율 없음",
        0,
        int((candidates["추정_비율"] < 0).sum() + candidates["추정_비율"].isna().sum()),
    )

    qa = pd.DataFrame(qa_records)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"보조 시나리오 QA 실패: {failures.to_dict('records')}")
    return candidates.reset_index(drop=True), qa


def apply_raking_candidates_to_panel(
    panel: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """34건(2017·2019) raking 추정치를 28개 지표 롱 패널에 반영한다.

    2016년 실측 반영(`apply_2016_candidates_to_panel`)과 구조는 같지만(기존 값이
    결측이 아니면 예외), 이 값은 실측이 아니라 raking 추정치이므로 `관측상태`를
    "추정"으로 남겨 실측과 구분한다.
    """
    required = {"지역", "지표_id", "연도", "측정값"}
    missing_columns = required - set(panel.columns)
    if missing_columns:
        raise ValueError(f"패널 필수 열 누락: {sorted(missing_columns, key=str)}")

    keys = ["지역", "지표_id", "연도"]
    if len(candidates) != 34 or candidates.duplicated(keys).any():
        raise ValueError("raking 후보표는 중복 없는 34개 키여야 합니다.")
    if not candidates["연도"].isin(RAKING_YEARS).all():
        raise ValueError(f"raking 후보표에 {RAKING_YEARS} 이외 연도가 포함됐습니다.")
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
                f"raking 후보 반영 키가 1개가 아닙니다({matched}개): {row['지역']}, {row['연도']}"
            )
        before = result.loc[mask, "측정값"].iloc[0]
        if pd.notna(before):
            raise ValueError(
                f"raking 반영 대상의 기존 패널값이 이미 존재합니다: {row['지역']}, {row['연도']}"
            )
        result.loc[mask, "측정값"] = float(row["추정_비율"])
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
                "반영후값": float(row["추정_비율"]),
                "방법": row["방법"],
                "근거": row["근거"],
                "반영유형": row["반영유형"],
                "관측상태": row["관측상태"],
                "QA_상태": "PASS",
            }
        )
    return result, pd.DataFrame(audit_rows)


def remove_resolved_rows_from_mapping(
    mapping: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """이제 raking 추정치로 채워진 34개(2017·2019) 키를 결측정책 매핑에서 제거한다."""
    keys = ["지역", "지표_id", "연도"]
    mask = (
        mapping["지역"].isin(candidates["지역"])
        & mapping["지표_id"].eq(INDICATOR_ID)
        & mapping["연도"].isin(RAKING_YEARS)
    )
    matched = mapping.loc[mask]
    if len(matched) != 34:
        raise ValueError(f"매핑에서 제거 대상 34건과 일치하지 않습니다: {len(matched)}건")
    if not matched.merge(candidates[keys], on=keys, how="inner").shape[0] == 34:
        raise ValueError("매핑의 raking 대상 행 키가 후보표 키와 정확히 대응하지 않습니다.")
    if not (
        matched["block_imputation"]
        & matched["imputation_policy"].eq("pending_review")
        & matched["auxiliary_scenario_policy"].eq("auxiliary_raking_composition_ratio")
    ).all():
        raise ValueError(
            "제거 대상 행이 예상한 pending_review/raking 보조 시나리오 상태가 아닙니다."
        )

    return mapping.loc[~mask].reset_index(drop=True), matched.reset_index(drop=True)


def render_report(candidates: pd.DataFrame, qa: pd.DataFrame, panel_audit: pd.DataFrame) -> str:
    if "반영전값" in panel_audit.columns:
        violation_count = int(panel_audit["반영전값"].notna().sum())
    else:
        violation_count = 0

    lines = [
        "# 가족친화 인증기업 비율 2017·2019년 raking 추정치 본계열 반영 QA",
        "",
        "## 결정 경위",
        "",
        "- 2016년 실측(`20260806_가족친화_2016_지역별_직접복원_QA.md`) 확보로 2017년이 선행",
        "  결측에서 2016·2018 사이 중간 결측으로 재분류돼, 2019와 같은 raking 방식으로",
        "  재계산했다(`20260805_가족친화_지역별_51건_보간_방법_제안.md`).",
        "- 이슈 #70 팀 논의 결과 이 34건(2017·2019)의 raking 추정치를 **본계열에 반영**하기로",
        "  결정했다(2026-08-06). 실측이 아니라는 점은 패널의 `관측상태=추정`으로 남긴다.",
        "",
        "## QA 요약",
        "",
        f"- 후보 계산 QA: {len(qa)}개 항목 모두 PASS",
        f"- 패널 반영 행 수: {len(panel_audit)}건 (기대 34건, `--panel` 미지정 시 0건)",
        f"- 반영 전 기존 값 존재(위반) 건수: {violation_count} (기대 0)",
        "",
        "## 한계",
        "",
        "- 여전히 **추정치**다. 지역별 실제 인증기업 증감을 정확히 반영하지 못할 수 있다.",
        "- 최종 지수·회귀 결과에는 이 34건을 포함한 버전과 제외한 버전(민감도 비교)을",
        "  함께 제시해야 한다.",
    ]
    return "\n".join(lines) + "\n"


PANEL_AUDIT_PATH = REPO_ROOT / "reports" / "20260806_가족친화_2017_2019_raking_패널반영_감사.csv"
REPORT_PATH = (
    REPO_ROOT / "reports" / "methodology" / "20260806_가족친화_2017_2019_raking_본계열반영_QA.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--confirmed", type=Path, default=CONFIRMED_PATH)
    parser.add_argument("--candidate-2016", type=Path, default=CANDIDATE_2016_PATH)
    parser.add_argument("--candidate-output", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="28개 지표 통합 패널 CSV. 존재할 때만 34건을 본계열에 반영한다(기본: 반영 안 함).",
    )
    parser.add_argument("--panel-output", type=Path, default=None)
    parser.add_argument("--panel-audit-output", type=Path, default=PANEL_AUDIT_PATH)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="결측정책 전수매핑 CSV. 존재할 때만 해소된 34행을 제거한다(기본: 반영 안 함).",
    )
    parser.add_argument("--mapping-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, qa = build_candidates(args.raw_dir, args.confirmed, args.candidate_2016)

    for path in (args.candidate_output, args.qa_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.candidate_output, index=False, encoding="utf-8-sig")
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")

    print(f"보조 시나리오 후보: {len(candidates)}건 ({args.candidate_output})")
    print(f"QA: {len(qa)}개 항목 모두 PASS")

    panel_audit = pd.DataFrame()
    if args.panel is not None and not args.panel.exists():
        raise FileNotFoundError(f"--panel 경로가 존재하지 않습니다: {args.panel}")
    if args.panel is not None and args.panel.exists():
        panel = pd.read_csv(args.panel, encoding="utf-8-sig")
        updated_panel, panel_audit = apply_raking_candidates_to_panel(panel, candidates)
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
        mapping = pd.read_csv(args.mapping, encoding="utf-8-sig")
        updated_mapping, removed = remove_resolved_rows_from_mapping(mapping, candidates)
        mapping_output = args.mapping_output or args.mapping
        updated_mapping.to_csv(mapping_output, index=False, encoding="utf-8-sig")
        print(f"매핑 갱신 완료: {mapping_output} ({len(removed)}행 제거)")
    else:
        print("매핑 인자 없음/미존재 — 매핑 갱신 생략")

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(render_report(candidates, qa, panel_audit), encoding="utf-8")
    print(f"보고서 저장: {args.report_output}")


if __name__ == "__main__":
    main()
