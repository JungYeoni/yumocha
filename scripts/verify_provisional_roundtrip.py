"""이슈 #81 필수 QA: 저장된 CSV가 파이프라인이 계산한 메모리 값과 완전히 같은지 검증한다.

`run_provisional_pipeline.py`의 계산 함수를 그대로 다시 호출해 얻은 메모리상
DataFrame과, 그 실행이 디스크에 저장한 CSV를 다시 읽은 결과를 셀 단위로
대조한다. CSV는 텍스트 포맷이라 저장·재읽기 과정에서 소수점 오차, 자료형
변경(정수→실수, NaN 유입 시 흔함), 결측 표현 방식 차이가 생길 수 있어 이
대조가 필요하다. 파이프라인 함수는 호출만 하고 수정하지 않는다.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

from scripts.consolidate_2021_area_labels import MAJOR_BY_SUBCATEGORY
from scripts.run_provisional_pipeline import _load_labels
from src.features.analysis_panel import (
    build_current_budget_panel,
    validate_budget_totals_against_detail,
)
from src.provisional.adjust import apply_cpi_adjustment, read_cpi
from src.provisional.aggregator import aggregate_labels_to_panels
from src.provisional.loader import EXPECTED_YEARS, STANDARD_REGIONS, read_raw_file_list

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "provisional"
QA_OUTPUT_PATH = REPO_ROOT / "reports" / "20260807_잠정재정패널_저장전후_대조_QA.csv"

# run_provisional_pipeline.py의 --major-labels/--sub-labels는 호출자가 명시적으로
# 넘겨야 한다(기본값이 없다 — 라벨 개수가 taxonomy와 다르면 즉시 실패시키기 위해).
# 이 QA 스크립트가 재계산에 쓸 기본값은 그 taxonomy의 단일 출처인
# MAJOR_BY_SUBCATEGORY에서 뽑는다 — 별도로 하드코딩하면 taxonomy가 바뀔 때 한쪽만
# 갱신되고 다른 쪽이 낡은 채로 남아 QA가 오탐(정상 저장을 FAIL)을 낼 수 있다.
SUB_LABELS = list(MAJOR_BY_SUBCATEGORY)
MAJOR_LABELS = list(dict.fromkeys(MAJOR_BY_SUBCATEGORY.values()))


def compare_frame_to_saved_csv(memory: pd.DataFrame, csv_path: Path) -> dict[str, object]:
    """메모리상 DataFrame과, 그걸 저장한 뒤 다시 읽은 CSV를 셀 단위로 대조한다."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"저장된 산출물이 없습니다: {csv_path}")
    saved = pd.read_csv(csv_path)

    shape_match = memory.shape == saved.shape
    column_match = list(memory.columns) == list(saved.columns)
    dtype_mismatches: list[str] = []
    value_mismatches = 0
    if shape_match and column_match:
        # CSV는 저장 시 항상 문자열을 거치므로, 메모리쪽도 같은 왕복을 한 번
        # 거쳐 비교해야 "저장 자체가 값을 바꿨는가"만 순수하게 잡아낼 수 있다.
        reencoded = pd.read_csv(io.StringIO(memory.to_csv(index=False)))
        for column in memory.columns:
            if reencoded[column].dtype != saved[column].dtype:
                dtype_mismatches.append(
                    f"{column}: {reencoded[column].dtype} -> {saved[column].dtype}"
                )
            equal = (reencoded[column] == saved[column]) | (
                reencoded[column].isna() & saved[column].isna()
            )
            value_mismatches += int((~equal).sum())

    return {
        "파일": csv_path.name,
        "메모리_shape": memory.shape,
        "저장본_shape": saved.shape,
        "shape_일치": shape_match,
        "열_순서_일치": column_match,
        "자료형_불일치": dtype_mismatches,
        "값_불일치_셀_수": value_mismatches,
        "판정": "PASS"
        if shape_match and column_match and not dtype_mismatches and value_mismatches == 0
        else "FAIL",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", type=Path, nargs="+", required=True)
    parser.add_argument("--cpi-file", type=Path, required=True)
    parser.add_argument("--cpi-encoding", default="utf-8-sig")
    parser.add_argument("--cpi-year-column", default="연도")
    parser.add_argument("--cpi-index-column", default="소비자물가지수")
    parser.add_argument("--cpi-unit", default="2024=100")
    parser.add_argument("--cpi-base-year", type=int, default=2024)
    parser.add_argument("--label-files", type=Path, nargs="*", default=[])
    parser.add_argument("--major-labels", nargs="*", default=MAJOR_LABELS)
    parser.add_argument("--sub-labels", nargs="*", default=SUB_LABELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--qa-output", type=Path, default=QA_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    detail = read_raw_file_list(
        args.input_files, expected_regions=STANDARD_REGIONS, expected_years=EXPECTED_YEARS
    )
    budget_panel = build_current_budget_panel(
        detail, expected_regions=STANDARD_REGIONS, expected_years=EXPECTED_YEARS
    )
    validate_budget_totals_against_detail(budget_panel, detail)

    cpi_data = read_cpi(
        args.cpi_file,
        encoding=args.cpi_encoding,
        year_column=args.cpi_year_column,
        index_column=args.cpi_index_column,
        unit=args.cpi_unit,
        base_year=args.cpi_base_year,
        expected_years=EXPECTED_YEARS,
    )
    adjusted_panel = apply_cpi_adjustment(
        budget_panel, cpi_data.series, base_year=args.cpi_base_year
    )

    labeled_detail, _ = _load_labels(args.label_files, detail)
    major_panel = sub_panel = None
    if not labeled_detail.empty:
        major_panel, sub_panel = aggregate_labels_to_panels(
            labeled_detail,
            expected_regions=STANDARD_REGIONS,
            expected_years=EXPECTED_YEARS,
            expected_major_labels=args.major_labels,
            expected_sub_labels=args.sub_labels,
        )

    comparisons = [
        compare_frame_to_saved_csv(
            detail, args.output_dir / "provisional_detail_current_budget.csv"
        ),
        compare_frame_to_saved_csv(budget_panel, args.output_dir / "provisional_budget_panel.csv"),
        compare_frame_to_saved_csv(
            adjusted_panel, args.output_dir / "provisional_budget_panel_cpi.csv"
        ),
    ]
    if major_panel is not None and sub_panel is not None:
        comparisons.append(
            compare_frame_to_saved_csv(
                major_panel, args.output_dir / "provisional_major_area_panel.csv"
            )
        )
        comparisons.append(
            compare_frame_to_saved_csv(
                sub_panel, args.output_dir / "provisional_sub_area_panel.csv"
            )
        )

    result = pd.DataFrame(comparisons)
    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.qa_output, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"QA 저장: {args.qa_output}")

    failures = result.loc[result["판정"].eq("FAIL")]
    if not failures.empty:
        raise ValueError(f"저장 전후 대조 실패: {failures.to_dict('records')}")


if __name__ == "__main__":
    main()
