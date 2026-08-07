"""이슈 #81 필수 QA: 대영역 합계와 세부영역 합계가 지역·연도별로 일치하는지 검증한다.

`run_provisional_pipeline.py`의 `_compare_area_totals`가 대영역·세부영역 각각을
총예산 패널과 대조하지만, 대영역 대 세부영역을 직접 맞대는 별도 산출물은 없어서
이 스크립트로 분리했다. 파이프라인 코드는 건드리지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAJOR_PATH = (
    REPO_ROOT / "data" / "interim" / "provisional" / "provisional_major_area_panel.csv"
)
DEFAULT_SUB_PATH = REPO_ROOT / "data" / "interim" / "provisional" / "provisional_sub_area_panel.csv"
QA_OUTPUT_PATH = REPO_ROOT / "reports" / "20260807_잠정재정패널_대영역_세부영역_합계대조_QA.csv"
BUDGET_COLUMN = "당해계획예산_백만원_provisional"
TOLERANCE = 1e-6


def compare_major_and_sub_totals(major: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """지역·연도별로 대영역 합계와 세부영역 합계를 대조한다."""
    required = {"지역", "연도", BUDGET_COLUMN}
    for name, frame in (("대영역 패널", major), ("세부영역 패널", sub)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} 필수 열 누락: {sorted(missing)}")

    major_totals = (
        major.groupby(["지역", "연도"], as_index=False)[BUDGET_COLUMN]
        .sum()
        .rename(columns={BUDGET_COLUMN: "대영역_합계_백만원"})
    )
    sub_totals = (
        sub.groupby(["지역", "연도"], as_index=False)[BUDGET_COLUMN]
        .sum()
        .rename(columns={BUDGET_COLUMN: "세부영역_합계_백만원"})
    )
    merged = major_totals.merge(sub_totals, on=["지역", "연도"], how="outer", validate="one_to_one")
    if merged[["대영역_합계_백만원", "세부영역_합계_백만원"]].isna().any().any():
        missing_keys = merged.loc[
            merged["대영역_합계_백만원"].isna() | merged["세부영역_합계_백만원"].isna(),
            ["지역", "연도"],
        ]
        raise ValueError(
            f"대영역·세부영역 지역·연도 키가 서로 다릅니다: {missing_keys.to_dict('records')}"
        )

    merged["차이_백만원"] = merged["대영역_합계_백만원"] - merged["세부영역_합계_백만원"]
    merged["일치"] = merged["차이_백만원"].abs().le(TOLERANCE)
    return merged.sort_values(["연도", "지역"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--major-panel", type=Path, default=DEFAULT_MAJOR_PATH)
    parser.add_argument("--sub-panel", type=Path, default=DEFAULT_SUB_PATH)
    parser.add_argument("--qa-output", type=Path, default=QA_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    major = pd.read_csv(args.major_panel)
    sub = pd.read_csv(args.sub_panel)
    comparison = compare_major_and_sub_totals(major, sub)

    mismatches = comparison.loc[~comparison["일치"]]
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.qa_output, index=False, encoding="utf-8-sig")

    print(f"지역·연도 조합: {len(comparison)}건")
    print(f"불일치(허용오차 {TOLERANCE} 초과): {len(mismatches)}건")
    print(f"최대 차이: {comparison['차이_백만원'].abs().max()}")
    print(f"QA 저장: {args.qa_output}")
    if not mismatches.empty:
        raise ValueError(f"대영역·세부영역 합계 불일치: {mismatches.to_dict('records')}")


if __name__ == "__main__":
    main()
