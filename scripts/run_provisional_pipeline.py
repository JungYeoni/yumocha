"""Build the #81 provisional budget panel from nine explicit annual workbooks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.analysis_panel import (  # noqa: E402
    build_current_budget_panel,
    validate_budget_totals_against_detail,
)
from src.provisional.adjust import (  # noqa: E402
    REAL_BUDGET_COLUMN,
    REAL_YOY_COLUMN,
    apply_cpi_adjustment,
    read_cpi,
)
from src.provisional.aggregator import aggregate_labels_to_panels  # noqa: E402
from src.provisional.loader import (  # noqa: E402
    BUDGET_UNIT,
    EXPECTED_YEARS,
    STANDARD_REGIONS,
    read_raw_file_list,
    validate_input_files,
)
from src.provisional.manifest import (  # noqa: E402
    build_manifest,
    describe_file,
    save_manifest,
)


DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "interim" / "provisional"
YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        required=True,
        help="Exactly nine explicit 2016-2024 normalized workbook paths.",
    )
    parser.add_argument("--cpi-file", type=Path, required=True)
    parser.add_argument("--cpi-encoding", default="utf-8-sig")
    parser.add_argument("--cpi-year-column", default="연도")
    parser.add_argument("--cpi-index-column", default="소비자물가지수")
    parser.add_argument("--cpi-unit", default="2024=100")
    parser.add_argument("--cpi-base-year", type=int, default=2024)
    parser.add_argument(
        "--qa-files",
        type=Path,
        nargs="*",
        default=[],
        help="Optional explicit annual QA CSV references; no discovery is performed.",
    )
    parser.add_argument("--label-files", type=Path, nargs="*", default=[])
    parser.add_argument("--major-labels", nargs="*", default=[])
    parser.add_argument("--sub-labels", nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def _validate_output_dir(path: Path) -> Path:
    allowed = DEFAULT_OUTPUT_DIR.resolve()
    resolved = path.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"잠정 산출물 경로는 {allowed} 아래여야 합니다: {resolved}")
    return resolved


def _year_from_reference(path: Path) -> int:
    years = {int(value) for value in YEAR_PATTERN.findall(path.name)}
    if len(years) != 1:
        raise ValueError(f"QA 파일명에서 연도를 하나만 확인해야 합니다: {path.name}")
    return years.pop()


def load_qa_reference(
    paths: Sequence[Path],
    *,
    expected_years: Sequence[int] = EXPECTED_YEARS,
    expected_regions: Sequence[str] = STANDARD_REGIONS,
) -> tuple[pd.DataFrame, list[dict]]:
    if not paths:
        return pd.DataFrame(columns=["지역", "연도", "기존_QA_leaf합계_백만원"]), []
    if len(paths) != len(expected_years):
        raise ValueError(f"연도별 QA 파일 수 불일치: 기대={len(expected_years)}, 실제={len(paths)}")

    frames: list[pd.DataFrame] = []
    metadata: list[dict] = []
    seen_years: set[int] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"QA 참조 파일이 없습니다: {path}")
        year = _year_from_reference(path)
        if year in seen_years:
            raise ValueError(f"QA 참조 연도 중복: {year}")
        seen_years.add(year)
        frame = pd.read_csv(path)
        required = {"지역", "leaf_합계"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: QA 참조 컬럼 누락={missing}")
        frame["지역"] = frame["지역"].astype("string").str.strip()
        frame["leaf_합계"] = pd.to_numeric(frame["leaf_합계"], errors="coerce")
        totals = (
            frame.groupby("지역", as_index=False)["leaf_합계"]
            .sum(min_count=1)
            .rename(columns={"leaf_합계": "기존_QA_leaf합계_백만원"})
        )
        totals["연도"] = year
        frames.append(totals)
        metadata.append(
            describe_file(
                path,
                role="yearly_budget_qa_reference",
                schema=list(frame.columns),
                unit=BUDGET_UNIT,
                year=year,
                encoding="utf-8",
            )
        )

    expected_year_set = {int(year) for year in expected_years}
    if seen_years != expected_year_set:
        raise ValueError(
            f"QA 참조 연도 불일치: 누락={sorted(expected_year_set - seen_years)}, "
            f"예상외={sorted(seen_years - expected_year_set)}"
        )
    reference = pd.concat(frames, ignore_index=True)
    actual_regions = set(reference["지역"])
    expected_region_set = set(expected_regions)
    if actual_regions != expected_region_set:
        raise ValueError(
            f"QA 참조 지역 불일치: 누락={sorted(expected_region_set - actual_regions)}, "
            f"예상외={sorted(actual_regions - expected_region_set)}"
        )
    if reference.duplicated(["지역", "연도"]).any():
        raise ValueError("QA 참조 지역×연도 키가 중복됩니다.")
    return reference.sort_values(["지역", "연도"]).reset_index(drop=True), metadata


def _load_labels(paths: Sequence[Path], detail: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    if not paths:
        return pd.DataFrame(), []
    frames: list[pd.DataFrame] = []
    metadata: list[dict] = []
    required = {"지역", "연도", "원본행", "대영역", "세부영역"}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"라벨 파일이 없습니다: {path}")
        frame = pd.read_csv(path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: 라벨 컬럼 누락={missing}")
        frames.append(frame[list(required)].copy())
        metadata.append(
            describe_file(
                path,
                role="area_label_input",
                schema=list(frame.columns),
                unit="행 단위 영역 라벨",
                encoding="utf-8",
            )
        )
    labels = pd.concat(frames, ignore_index=True)
    key = ["지역", "연도", "원본행"]
    if labels.duplicated(key).any():
        raise ValueError("라벨 지역×연도×원본행 키가 중복됩니다.")
    labeled = detail.merge(labels, on=key, how="left", validate="one_to_one", indicator=True)
    unmatched = labeled["_merge"].ne("both") | labeled[["대영역", "세부영역"]].isna().any(axis=1)
    if unmatched.any():
        raise ValueError(f"세부사업 라벨 누락: {int(unmatched.sum())}행")
    return labeled.drop(columns="_merge"), metadata


def _compare_area_totals(area_panel: pd.DataFrame, budget_panel: pd.DataFrame, column: str) -> None:
    totals = (
        area_panel.groupby(["지역", "연도"], as_index=False)[column]
        .sum(min_count=1)
        .rename(columns={column: "영역합계_백만원"})
    )
    comparison = budget_panel[["지역", "연도", "당해계획예산_백만원"]].merge(
        totals, on=["지역", "연도"], how="outer", validate="one_to_one"
    )
    difference = comparison["당해계획예산_백만원"] - comparison["영역합계_백만원"]
    if difference.isna().any() or difference.abs().gt(1e-6).any():
        raise ValueError("영역 패널 합계와 총예산 패널 합계가 일치하지 않습니다.")


def run_pipeline(args: argparse.Namespace) -> dict:
    output_dir = _validate_output_dir(args.out_dir)
    # Validate before any output directory or artifact is created.
    validate_input_files(args.input_files, expected_years=EXPECTED_YEARS)
    detail = read_raw_file_list(
        args.input_files,
        expected_regions=STANDARD_REGIONS,
        expected_years=EXPECTED_YEARS,
    )
    budget_panel = build_current_budget_panel(
        detail,
        expected_regions=STANDARD_REGIONS,
        expected_years=EXPECTED_YEARS,
    )
    direct_comparison = validate_budget_totals_against_detail(budget_panel, detail)

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
        budget_panel,
        cpi_data.series,
        base_year=args.cpi_base_year,
    )

    qa_reference, qa_inputs = load_qa_reference(args.qa_files)
    qa = budget_panel.merge(
        direct_comparison[["지역", "연도", "원본당해예산합계_백만원", "집계차이_백만원"]],
        on=["지역", "연도"],
        how="left",
        validate="one_to_one",
    )
    if not qa_reference.empty:
        qa = qa.merge(qa_reference, on=["지역", "연도"], how="left", validate="one_to_one")
        qa["기존_QA_차이_백만원"] = qa["당해계획예산_백만원"] - qa["기존_QA_leaf합계_백만원"]
        qa["기존_QA_일치"] = qa["기존_QA_차이_백만원"].abs().le(1e-6)
    else:
        qa["기존_QA_leaf합계_백만원"] = pd.NA
        qa["기존_QA_차이_백만원"] = pd.NA
        qa["기존_QA_일치"] = pd.NA

    labeled_detail, label_inputs = _load_labels(args.label_files, detail)
    major_panel: pd.DataFrame | None = None
    sub_panel: pd.DataFrame | None = None
    if not labeled_detail.empty:
        if len(args.major_labels) != 5 or len(args.sub_labels) != 12:
            raise ValueError("라벨 산출 시 대영역 5개와 세부영역 12개를 명시해야 합니다.")
        major_panel, sub_panel = aggregate_labels_to_panels(
            labeled_detail,
            expected_regions=STANDARD_REGIONS,
            expected_years=EXPECTED_YEARS,
            expected_major_labels=args.major_labels,
            expected_sub_labels=args.sub_labels,
        )
        _compare_area_totals(major_panel, budget_panel, "당해계획예산_백만원_provisional")
        _compare_area_totals(sub_panel, budget_panel, "당해계획예산_백만원_provisional")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[tuple[Path, pd.DataFrame, str, str]] = [
        (
            output_dir / "provisional_detail_current_budget.csv",
            detail,
            "provisional_current_budget_detail",
            BUDGET_UNIT,
        ),
        (
            output_dir / "provisional_budget_panel.csv",
            budget_panel,
            "provisional_nominal_budget_panel",
            BUDGET_UNIT,
        ),
        (
            output_dir / "provisional_budget_panel_cpi.csv",
            adjusted_panel,
            "provisional_real_budget_panel",
            f"{args.cpi_base_year}년 가격 백만원; 증감률은 %",
        ),
        (
            output_dir / "provisional_budget_qa.csv",
            qa,
            "provisional_budget_qa",
            BUDGET_UNIT,
        ),
    ]
    if major_panel is not None and sub_panel is not None:
        artifacts.extend(
            [
                (
                    output_dir / "provisional_major_area_panel.csv",
                    major_panel,
                    "provisional_major_area_panel",
                    BUDGET_UNIT,
                ),
                (
                    output_dir / "provisional_sub_area_panel.csv",
                    sub_panel,
                    "provisional_sub_area_panel",
                    BUDGET_UNIT,
                ),
            ]
        )
    for path, frame, _, _ in artifacts:
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    workbook_inputs: list[dict] = []
    for metadata in detail.attrs["workbooks"]:
        metadata = dict(metadata)
        path = metadata.pop("path")
        schema = metadata.pop("schema")
        unit = metadata.pop("unit")
        workbook_inputs.append(
            describe_file(
                path,
                role="annual_budget_workbook",
                schema=schema,
                unit=unit,
                **metadata,
            )
        )
    cpi_input = {"role": "cpi_input", **cpi_data.metadata}
    output_descriptions = [
        describe_file(path, role=role, schema=list(frame.columns), unit=unit)
        for path, frame, role, unit in artifacts
    ]
    reference_mismatches = (
        int(qa["기존_QA_일치"].eq(False).sum()) if not qa_reference.empty else None
    )
    manifest = build_manifest(
        inputs=[*workbook_inputs, cpi_input, *qa_inputs, *label_inputs],
        outputs=output_descriptions,
        pipeline={
            "issue": 81,
            "status": "provisional",
            "expected_years": list(EXPECTED_YEARS),
            "expected_regions": list(STANDARD_REGIONS),
            "expected_region_year_rows": 153,
            "budget_unit": BUDGET_UNIT,
            "cpi_formula": cpi_data.metadata["formula"],
            "first_year_real_yoy": "missing per region",
            "labels_status": "applied"
            if label_inputs
            else "not supplied; area outputs not generated",
            "reference_qa_mismatches": reference_mismatches,
        },
    )
    manifest_path = save_manifest(manifest, output_dir / "provisional_manifest.json")

    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "panel_rows": len(budget_panel),
        "detail_rows": len(detail),
        "years": [int(budget_panel["연도"].min()), int(budget_panel["연도"].max())],
        "regions": int(budget_panel["지역"].nunique()),
        "nominal_total": float(budget_panel["당해계획예산_백만원"].sum()),
        "detail_total": float(direct_comparison["원본당해예산합계_백만원"].sum()),
        "missing_budget_rows": int(budget_panel["예산결측_사업수"].sum()),
        "non_numeric_budget_rows": int(budget_panel["예산비수치_사업수"].sum()),
        "negative_budget_rows": int(budget_panel["음수예산_사업수"].sum()),
        "reference_qa_mismatches": reference_mismatches,
        "area_outputs_generated": major_panel is not None,
        "real_budget_column": REAL_BUDGET_COLUMN,
        "real_yoy_column": REAL_YOY_COLUMN,
    }


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_pipeline(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
