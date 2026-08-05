"""#70 기준패널에 결측 정책 매핑을 실제로 적용해 처리 완료 패널을 만든다.

`src/features/structural_missing.py`의 `process_structural_indicator_panel()`은 이미
중간 결측 선형보간, 경계 결측 hold, 구조적 결측 보존 로직을 갖고 있다(재발명 금지). 이
스크립트는 결측정책 전수 매핑의 `block_imputation` 컬럼을 그 모듈의 구조적 결측 플래그로
그대로 넘겨서 본계열(경계 결측 hold=boundary_carry)과 민감도 비교용 대안본(경계 결측
제외=관측구간 제한)을 함께 산출한다. 차단된(pending_review·keep_missing) 셀은 두 버전
모두에서 NA로 보존된다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.structural_missing import (
    ExtrapolationStrategy,
    process_structural_indicator_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_PATH = REPO_ROOT / "data" / "processed" / "구조환경지표_28개_보간전_기준패널.csv"
DEFAULT_MAPPING_PATH = REPO_ROOT / "reports" / "20260805_구조환경지표_결측정책_전수매핑_최신화.csv"
DEFAULT_MAIN_OUTPUT_PATH = (
    REPO_ROOT / "data" / "processed" / "구조환경지표_28개_결측처리후_본계열패널.csv"
)
DEFAULT_SENSITIVITY_OUTPUT_PATH = (
    REPO_ROOT / "data" / "processed" / "구조환경지표_28개_결측처리_민감도비교.csv"
)
DEFAULT_QA_REPORT_PATH = (
    REPO_ROOT / "reports" / "methodology" / "20260805_구조환경지표_결측처리_실제적용_QA.md"
)

KEY_COLUMNS = ["지역", "지표_id", "연도"]
EXPECTED_YEARS = list(range(2016, 2025))


def attach_block_imputation_flag(panel: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """매핑의 `block_imputation`을 패널에 결합한다. 매핑에 없는 행(실측)은 False로 채운다."""
    merged = panel.merge(
        mapping[[*KEY_COLUMNS, "block_imputation"]],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if (merged.loc[panel["측정값"].notna(), "block_imputation"].notna()).any():
        raise ValueError(
            "실측 행이 결측정책 매핑에 존재합니다. 매핑이 패널과 어긋났을 수 있습니다."
        )
    merged["block_imputation"] = merged["block_imputation"].fillna(False).astype(bool)
    return merged


def build_processed_panels(
    panel_with_flag: pd.DataFrame,
    *,
    expected_years: list[int] | None = EXPECTED_YEARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """본계열(경계=hold)과 대안본(경계=제외)을 각각 생성한다."""
    main_panel = process_structural_indicator_panel(
        panel_with_flag,
        region_col="지역",
        indicator_col="지표_id",
        year_col="연도",
        value_col="측정값",
        structural_missing_col="block_imputation",
        expected_years=expected_years,
        leading_strategy=ExtrapolationStrategy.HOLD,
        trailing_strategy=ExtrapolationStrategy.HOLD,
    )
    alternative_panel = process_structural_indicator_panel(
        panel_with_flag,
        region_col="지역",
        indicator_col="지표_id",
        year_col="연도",
        value_col="측정값",
        structural_missing_col="block_imputation",
        expected_years=expected_years,
        leading_strategy=ExtrapolationStrategy.EXCLUDE,
        trailing_strategy=ExtrapolationStrategy.EXCLUDE,
    )
    return main_panel, alternative_panel


def build_sensitivity_comparison(
    main_panel: pd.DataFrame,
    alternative_panel: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """`boundary_carry`(경계 결측) 대상만 뽑아 본계열 값과 대안 값을 나란히 둔다."""
    boundary_keys = mapping.loc[
        mapping["imputation_policy"].eq("boundary_carry"), [*KEY_COLUMNS, "policy_risk_level"]
    ]
    main_slim = main_panel[[*KEY_COLUMNS, "지표명", "processed_value", "missing_type"]].rename(
        columns={"processed_value": "processed_value_본계열_carry"}
    )
    alt_slim = alternative_panel[[*KEY_COLUMNS, "processed_value"]].rename(
        columns={"processed_value": "processed_value_대안_관측구간제한"}
    )
    comparison = boundary_keys.merge(main_slim, on=KEY_COLUMNS, how="left", validate="one_to_one")
    comparison = comparison.merge(alt_slim, on=KEY_COLUMNS, how="left", validate="one_to_one")
    return comparison.sort_values(KEY_COLUMNS).reset_index(drop=True)


def _count_dict(series: pd.Series) -> dict[object, int]:
    return {key: int(value) for key, value in series.value_counts(dropna=False).items()}


def validate_outputs(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
    main_panel: pd.DataFrame,
    alternative_panel: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """원본 관측값 보존과 정책별 처리 건수 일치를 검증한다."""
    checks: list[dict[str, object]] = []

    def check(item: str, expected: object, actual: object) -> None:
        checks.append(
            {
                "검사항목": item,
                "기대값": expected,
                "실제값": actual,
                "판정": "PASS" if actual == expected else "FAIL",
            }
        )

    # process_structural_indicator_panel()은 groupby 순서로 재정렬해 반환하므로(원본 행
    # 순서·인덱스 보존 보장 없음) 위치 기반 비교가 아니라 키(KEY_COLUMNS) 기준으로 대조한다.
    original_observed = panel.loc[panel["측정값"].notna(), [*KEY_COLUMNS, "측정값"]]
    joined_main = original_observed.merge(
        main_panel[[*KEY_COLUMNS, "측정값", "processed_value", "is_observed"]],
        on=KEY_COLUMNS,
        how="left",
        suffixes=("_원본", "_본계열"),
        validate="one_to_one",
    )
    check("행 수 변화 없음(본계열)", len(panel), len(main_panel))
    check("행 수 변화 없음(대안본)", len(panel), len(alternative_panel))
    check("실측 행 is_observed=True(본계열)", True, bool(joined_main["is_observed"].all()))
    value_diff = joined_main["processed_value"].to_numpy() - joined_main["측정값_원본"].to_numpy()
    check("실측값 변경 없음(본계열)", 0, int((value_diff != 0).sum()))

    blocked_keys = mapping.loc[mapping["block_imputation"], KEY_COLUMNS]
    blocked_index = pd.MultiIndex.from_frame(blocked_keys)
    main_index = pd.MultiIndex.from_frame(main_panel[KEY_COLUMNS])
    blocked_mask = main_index.isin(blocked_index)
    check(
        "차단된 셀 NA 보존(본계열)",
        0,
        int(main_panel.loc[blocked_mask, "processed_value"].notna().sum()),
    )
    check(
        "차단된 셀 NA 보존(대안본)",
        0,
        int(alternative_panel.loc[blocked_mask, "processed_value"].notna().sum()),
    )

    expected_boundary = int(mapping["imputation_policy"].eq("boundary_carry").sum())
    check("민감도 비교 대상 건수", expected_boundary, len(comparison))
    check(
        "대안본 경계 결측 값 전부 NA",
        expected_boundary,
        int(comparison["processed_value_대안_관측구간제한"].isna().sum()),
    )
    check(
        "본계열 경계 결측 값 전부 채워짐",
        expected_boundary,
        int(comparison["processed_value_본계열_carry"].notna().sum()),
    )

    qa = pd.DataFrame(checks)
    failures = qa.loc[qa["판정"].ne("PASS")]
    if not failures.empty:
        raise ValueError(f"결측 처리 적용 QA 실패: {failures.to_dict('records')}")
    return qa


def render_report(
    panel: pd.DataFrame,
    mapping: pd.DataFrame,
    main_panel: pd.DataFrame,
    comparison: pd.DataFrame,
    qa: pd.DataFrame,
) -> str:
    strategy_counts = _count_dict(main_panel["processing_strategy"])
    lines = [
        "# 구조환경지표 결측 처리 실제 적용 QA",
        "",
        "## 범위",
        "",
        f"- 입력 패널: `{DEFAULT_PANEL_PATH.relative_to(REPO_ROOT)}` ({len(panel):,}행)",
        f"- 입력 정책 매핑: `{DEFAULT_MAPPING_PATH.relative_to(REPO_ROOT)}` ({len(mapping):,}행)",
        f"- 차단(block_imputation=True) {int(mapping['block_imputation'].sum()):,}건은 두 버전 모두 NA 보존",
        "",
        "## 처리 전략 분포(본계열)",
        "",
        "| processing_strategy | 건수 |",
        "|---|---:|",
    ]
    for strategy, count in sorted(strategy_counts.items(), key=lambda item: str(item[0])):
        lines.append(f"| {strategy} | {count:,} |")
    lines.extend(
        [
            "",
            "## 민감도 비교(boundary_carry)",
            "",
            f"- 대상: {len(comparison):,}건",
            "- 본계열: 경계 결측을 최근접 관측값으로 유지(hold)",
            "- 대안: 관측구간 밖을 제외(NA 유지)",
            "",
            "## QA",
            "",
            f"- QA 항목 {len(qa):,}개 모두 PASS",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--main-output", type=Path, default=DEFAULT_MAIN_OUTPUT_PATH)
    parser.add_argument("--sensitivity-output", type=Path, default=DEFAULT_SENSITIVITY_OUTPUT_PATH)
    parser.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.panel, encoding="utf-8-sig")
    mapping = pd.read_csv(args.mapping, encoding="utf-8-sig")

    panel_with_flag = attach_block_imputation_flag(panel, mapping)
    main_panel, alternative_panel = build_processed_panels(panel_with_flag)
    comparison = build_sensitivity_comparison(main_panel, alternative_panel, mapping)
    qa = validate_outputs(panel, mapping, main_panel, alternative_panel, comparison)
    report = render_report(panel, mapping, main_panel, comparison, qa)

    for path in (args.main_output, args.sensitivity_output, args.qa_report):
        path.parent.mkdir(parents=True, exist_ok=True)
    main_panel.to_csv(args.main_output, index=False, encoding="utf-8-sig")
    comparison.to_csv(args.sensitivity_output, index=False, encoding="utf-8-sig")
    args.qa_report.write_text(report, encoding="utf-8")

    print(f"본계열 패널: {args.main_output} ({len(main_panel):,}행)")
    print(f"민감도 비교: {args.sensitivity_output} ({len(comparison):,}행)")
    print(f"QA: {len(qa):,}개 항목 모두 PASS")


if __name__ == "__main__":
    main()
