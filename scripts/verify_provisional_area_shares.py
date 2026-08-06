"""이슈 #81 필수 QA: 대영역·세부영역 구성비가 유한값이며 정상 범위인지 검증한다.

구성비 = 카테고리 예산 / 지역·연도 총예산. 총예산이 0이면 나눗셈이 정의되지
않고(NaN/inf), 음수 예산 셀이 섞이면 구성비가 0~1 범위를 벗어날 수 있다 —
경북 2018년 "여성지도자 육성"의 -20백만원(기존 방침대로 원자료 그대로 유지,
#52/#81 이슈 코멘트 참고)이 정확히 이런 사례를 만든다. 이 스크립트는 그런
셀을 임의 보정하지 않고 있는 그대로 찾아서 보고한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KNOWN_NEGATIVE_BUDGET_REGION_YEARS = {("경북", 2018)}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAJOR_PATH = (
    REPO_ROOT / "data" / "interim" / "provisional" / "provisional_major_area_panel.csv"
)
DEFAULT_SUB_PATH = REPO_ROOT / "data" / "interim" / "provisional" / "provisional_sub_area_panel.csv"
QA_OUTPUT_PATH = REPO_ROOT / "reports" / "20260807_잠정재정패널_구성비_범위_QA.csv"
BUDGET_COLUMN = "당해계획예산_백만원_provisional"
SHARE_SUM_TOLERANCE = 1e-6


def compute_shares(panel: pd.DataFrame, *, category_column: str) -> pd.DataFrame:
    """지역·연도별 총예산 대비 카테고리 구성비를 계산한다(원자료 값 그대로, 보정 없음)."""
    required = {"지역", "연도", category_column, BUDGET_COLUMN}
    missing = required - set(panel.columns)
    if missing:
        raise KeyError(f"필수 열 누락: {sorted(missing)}")

    frame = panel.copy()
    totals = frame.groupby(["지역", "연도"])[BUDGET_COLUMN].transform("sum")
    frame["지역연도_총예산_백만원"] = totals
    frame["구성비"] = frame[BUDGET_COLUMN] / totals
    return frame


def check_share_ranges(shares: pd.DataFrame, *, category_column: str) -> pd.DataFrame:
    """구성비가 유한값인지, 0~1 범위인지, 지역·연도별 합이 1인지 검사한다."""
    checks: list[dict[str, object]] = []

    def add(
        item: str, expected: object, actual: object, note: str = "", *, informational: bool = False
    ) -> None:
        checks.append(
            {
                "검사항목": item,
                "기대값": expected,
                "실제값": actual,
                "판정": "INFO" if informational else ("PASS" if expected == actual else "FAIL"),
                "비고": note,
            }
        )

    non_finite = ~np.isfinite(shares["구성비"])
    add("구성비 비유한값(NaN·inf) 건수", 0, int(non_finite.sum()))

    finite_shares = shares.loc[~non_finite, "구성비"]
    out_of_range = finite_shares.lt(0) | finite_shares.gt(1)
    add(
        "구성비 0~1 범위 이탈 건수(음수 예산 셀 영향, 원자료 그대로 유지 — #52/#81 결정)",
        None,
        int(out_of_range.sum()),
        note="정보용 항목 — 알려진 음수 예산 원자료(경북 2018) 때문이면 실패로 보지 않음",
        informational=True,
    )

    sums = shares.groupby(["지역", "연도"])["구성비"].sum()
    sum_mismatch = (sums - 1.0).abs().gt(SHARE_SUM_TOLERANCE)
    add(f"{category_column} 구성비 합계 != 1 인 지역·연도 수", 0, int(sum_mismatch.sum()))

    return pd.DataFrame(checks)


def check_out_of_range_is_only_known_cause(shares: pd.DataFrame) -> None:
    """범위 이탈이 알려진 음수 예산 지역·연도 밖에서도 발생하면 실패시킨다."""
    out_of_range = shares.loc[
        ~np.isfinite(shares["구성비"]) | shares["구성비"].lt(0) | shares["구성비"].gt(1)
    ]
    if out_of_range.empty:
        return
    unexpected = out_of_range.loc[
        ~out_of_range.set_index(["지역", "연도"]).index.isin(KNOWN_NEGATIVE_BUDGET_REGION_YEARS)
    ]
    if not unexpected.empty:
        raise ValueError(
            "알려진 원인(경북 2018 음수 예산) 밖에서 구성비 범위 이탈이 발생했습니다: "
            f"{unexpected[['지역', '연도']].drop_duplicates().to_dict('records')}"
        )


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

    major_shares = compute_shares(major, category_column="대영역")
    sub_shares = compute_shares(sub, category_column="세부영역")

    major_qa = check_share_ranges(major_shares, category_column="대영역")
    major_qa.insert(0, "구분", "대영역")
    sub_qa = check_share_ranges(sub_shares, category_column="세부영역")
    sub_qa.insert(0, "구분", "세부영역")
    qa = pd.concat([major_qa, sub_qa], ignore_index=True)

    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    qa.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    print(qa.to_string(index=False))
    print(f"QA 저장: {args.qa_output}")

    hard_failures = qa.loc[qa["판정"].eq("FAIL")]
    if not hard_failures.empty:
        raise ValueError(f"구성비 검증 실패: {hard_failures.to_dict('records')}")

    check_out_of_range_is_only_known_cause(major_shares)
    check_out_of_range_is_only_known_cause(sub_shares)

    out_of_range_major = major_shares.loc[
        ~np.isfinite(major_shares["구성비"])
        | major_shares["구성비"].lt(0)
        | major_shares["구성비"].gt(1)
    ]
    if not out_of_range_major.empty:
        print("\n대영역 구성비 범위 이탈(음수 예산 원자료 영향, 참고용):")
        print(
            out_of_range_major[["지역", "연도", "대영역", BUDGET_COLUMN, "구성비"]].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
