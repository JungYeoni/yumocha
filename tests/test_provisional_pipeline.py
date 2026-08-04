from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_provisional_pipeline import parse_args
from src.features.analysis_panel import (
    build_current_budget_panel,
    validate_budget_totals_against_detail,
)
from src.provisional.adjust import (
    REAL_BUDGET_COLUMN,
    REAL_YOY_COLUMN,
    apply_cpi_adjustment,
    read_cpi,
)
from src.provisional.aggregator import aggregate_labels_to_panels, validate_full_grid
from src.provisional.loader import (
    EXPECTED_YEARS,
    STANDARD_REGIONS,
    read_raw_file_list,
    validate_input_files,
)
from src.provisional.manifest import build_manifest, describe_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def _detail_grid() -> pd.DataFrame:
    rows = []
    for region in STANDARD_REGIONS:
        for year in EXPECTED_YEARS:
            rows.append(
                {
                    "지역": region,
                    "연도": year,
                    "세부사업명": f"{region}-{year}",
                    "예산구분": "당해예산",
                    "예산액": 10.0,
                    "예산_비수치": False,
                }
            )
    return pd.DataFrame(rows)


def _normalized_columns(year: int) -> list[str]:
    return [
        "지역",
        "세부사업명",
        "사업분류재정구분",
        f"{year}년 예산",
        f"{year - 1}년 예산",
        "증감액",
        "비율",
        "주요내용",
        "원본표구간",
        "머리글행",
        "원본행",
        "행유형",
    ]


def _validation_columns(year: int) -> list[str]:
    return [
        "원본표구간",
        "머리글행",
        "블록끝행",
        "사업명열",
        "구분열",
        f"{year}년 예산열",
        f"{year - 1}년 예산열",
        "증감액열",
        "비율열",
        "내용열",
        "추출행수",
        "점검내용",
    ]


def _write_reduced_workbook(path: Path, *, year: int = 2024) -> None:
    rows = [
        ["서울", "Ⅰ. 공통사업", "공통", None, None, None, None, None, "T1", 1, 1, "본문"],
        ["서울", "1. 생활지원(공통)", "공통", None, None, None, None, None, "T1", 1, 2, "본문"],
        ["서울", "정상 예산 사업", "공통", "1,000", 800, None, None, None, "T1", 1, 3, "본문"],
        ["서울", "결측 예산 사업", "공통", None, 10, None, None, None, "T1", 1, 4, "본문"],
        ["서울", "비수치 예산 사업", "공통", "미정", 10, None, None, None, "T1", 1, 5, "본문"],
        ["서울", "음수 예산 사업", "공통", -20, 10, None, None, None, "T1", 1, 6, "본문"],
    ]
    normalized = pd.DataFrame(rows, columns=_normalized_columns(year))
    validation = pd.DataFrame(
        [["T1", 1, 6, "A", "B", "C", "D", "E", "F", "G", 6, "정상"]],
        columns=_validation_columns(year),
    )
    raw = pd.DataFrame([["(단위: 백만원)"], ["원문 대조용"]])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="Table 1", index=False, header=False)
        normalized.to_excel(writer, sheet_name="정리본_자동", index=False)
        validation.to_excel(writer, sheet_name="검증_자동", index=False)


def test_runner_import_and_cli_help() -> None:
    assert parse_args is not None
    completed = subprocess.run(
        [sys.executable, "-B", "scripts/run_provisional_pipeline.py", "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--input-files" in completed.stdout
    assert "--cpi-file" in completed.stdout


def test_explicit_input_list_requires_one_file_per_expected_year(tmp_path: Path) -> None:
    paths = []
    for year in EXPECTED_YEARS:
        path = tmp_path / f"budget_{year}.xlsx"
        path.write_bytes(b"fixture")
        paths.append(path)

    validated = validate_input_files(paths[::-1])
    assert [year for year, _ in validated] == list(EXPECTED_YEARS)
    with pytest.raises(ValueError, match="입력 Excel 수 불일치"):
        validate_input_files(paths[:-1])
    with pytest.raises(ValueError, match="연도가 중복"):
        validate_input_files([*paths[:-1], paths[0]])


def test_actual_workbook_shape_fixture_uses_named_authoritative_sheet(tmp_path: Path) -> None:
    workbook = tmp_path / "reduced_2024.xlsx"
    _write_reduced_workbook(workbook)

    detail = read_raw_file_list([workbook], expected_regions=["서울"], expected_years=[2024])

    assert len(detail) == 4
    assert detail["예산액"].sum() == 980
    assert detail["예산_결측"].sum() == 2
    assert detail["예산_비수치"].sum() == 1
    assert detail["예산_음수"].sum() == 1
    metadata = detail.attrs["workbooks"][0]
    assert metadata["authoritative_sheet"] == "정리본_자동"
    assert metadata["header_row"] == 1
    assert metadata["unit"] == "백만원"
    assert metadata["sheet_roles"]["Table 1"] == "원문 대조용 원본 시트"


def test_17_by_9_complete_grid_and_detail_total_match() -> None:
    detail = _detail_grid()
    panel = build_current_budget_panel(
        detail,
        expected_regions=STANDARD_REGIONS,
        expected_years=EXPECTED_YEARS,
    )

    assert len(panel) == 153
    assert panel["지역"].nunique() == 17
    assert panel["연도"].nunique() == 9
    comparison = validate_budget_totals_against_detail(panel, detail)
    assert comparison["집계차이_백만원"].eq(0).all()
    assert panel["당해계획예산_백만원"].sum() == detail["예산액"].sum()


@pytest.mark.parametrize(
    ("column", "value"),
    [("지역", "서울"), ("연도", 2016)],
)
def test_expected_region_or_year_omission_is_detected(column: str, value: object) -> None:
    detail = _detail_grid()
    incomplete = detail.loc[~detail[column].eq(value)].copy()
    with pytest.raises(ValueError, match="조합 불일치"):
        build_current_budget_panel(
            incomplete,
            expected_regions=STANDARD_REGIONS,
            expected_years=EXPECTED_YEARS,
        )


def test_duplicate_missing_negative_and_non_numeric_budgets_are_detected() -> None:
    detail = pd.DataFrame(
        {
            "지역": ["서울", "서울", "서울"],
            "연도": [2024, 2024, 2024],
            "세부사업명": ["정상", "결측", "음수"],
            "예산구분": ["당해예산"] * 3,
            "예산액": [100.0, None, -20.0],
            "예산_비수치": [False, True, False],
        }
    )
    panel = build_current_budget_panel(detail, expected_regions=["서울"], expected_years=[2024])
    assert panel.loc[0, "예산결측_사업수"] == 1
    assert panel.loc[0, "예산비수치_사업수"] == 1
    assert panel.loc[0, "음수예산_사업수"] == 1

    invalid = detail.astype({"예산액": "object"}).copy()
    invalid.loc[0, "예산액"] = "not-a-number"
    with pytest.raises(ValueError, match="숫자 변환 실패"):
        build_current_budget_panel(invalid, expected_regions=["서울"], expected_years=[2024])

    duplicated_panel = pd.DataFrame({"지역": ["서울", "서울"], "연도": [2024, 2024]})
    with pytest.raises(ValueError, match="키 중복"):
        validate_full_grid(duplicated_panel, ["서울"], [2024])


def test_aggregate_before_after_mismatch_is_rejected() -> None:
    detail = _detail_grid()
    panel = build_current_budget_panel(
        detail, expected_regions=STANDARD_REGIONS, expected_years=EXPECTED_YEARS
    )
    panel.loc[0, "당해계획예산_백만원"] += 1
    with pytest.raises(ValueError, match="합계 불일치"):
        validate_budget_totals_against_detail(panel, detail)


def test_cpi_real_budget_and_group_sorted_yoy_are_numerically_checked(
    tmp_path: Path,
) -> None:
    cpi_path = tmp_path / "cpi.csv"
    pd.DataFrame(
        {
            "연도": [2019, 2020],
            "소비자물가지수": [50.0, 100.0],
            "기준연도": ["2020=100", "2020=100"],
        }
    ).to_csv(cpi_path, index=False, encoding="utf-8")
    cpi_data = read_cpi(
        cpi_path,
        encoding="utf-8",
        year_column="연도",
        index_column="소비자물가지수",
        unit="2020=100",
        base_year=2020,
        expected_years=[2019, 2020],
    )
    panel = pd.DataFrame(
        {
            "지역": ["부산", "서울", "부산", "서울"],
            "연도": [2020, 2020, 2019, 2019],
            "당해계획예산_백만원": [100.0, 100.0, 100.0, 100.0],
        }
    )
    adjusted = apply_cpi_adjustment(panel, cpi_data.series, base_year=2020)

    for _, group in adjusted.groupby("지역"):
        assert group["연도"].tolist() == [2019, 2020]
        assert group[REAL_BUDGET_COLUMN].tolist() == pytest.approx([200.0, 100.0])
        assert pd.isna(group.iloc[0][REAL_YOY_COLUMN])
        assert group.iloc[1][REAL_YOY_COLUMN] == pytest.approx(-50.0)
    assert cpi_data.metadata["encoding"] == "utf-8"
    assert cpi_data.metadata["unit"] == "2020=100"
    assert len(cpi_data.metadata["sha256"]) == 64


def test_label_aggregation_requires_exact_5_and_12_category_grids() -> None:
    major_labels = [f"대영역{i}" for i in range(5)]
    sub_labels = [f"세부영역{i}" for i in range(12)]
    rows = []
    for region in STANDARD_REGIONS:
        for year in EXPECTED_YEARS:
            for index, sub_label in enumerate(sub_labels):
                rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "세부사업명": f"사업-{index}",
                        "대영역": major_labels[index % len(major_labels)],
                        "세부영역": sub_label,
                        "예산액": 1.0,
                    }
                )
    labeled = pd.DataFrame(rows)

    major, sub = aggregate_labels_to_panels(
        labeled,
        expected_regions=STANDARD_REGIONS,
        expected_years=EXPECTED_YEARS,
        expected_major_labels=major_labels,
        expected_sub_labels=sub_labels,
    )

    assert len(major) == 17 * 9 * 5
    assert len(sub) == 17 * 9 * 12
    assert major["당해계획예산_백만원_provisional"].sum() == len(labeled)
    assert sub["당해계획예산_백만원_provisional"].sum() == len(labeled)


def test_manifest_separates_input_lineage_from_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.xlsx"
    output = tmp_path / "output.csv"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    input_description = describe_file(
        source, role="annual_budget_workbook", schema=["지역", "예산"], unit="백만원"
    )
    output_description = describe_file(
        output, role="provisional_panel", schema=["지역", "예산"], unit="백만원"
    )

    manifest = build_manifest(
        inputs=[input_description],
        outputs=[output_description],
        pipeline={"issue": 81},
    )

    assert manifest["inputs"][0]["path"] == str(source.resolve())
    assert manifest["outputs"][0]["path"] == str(output.resolve())
    assert "generated_from" not in manifest
    for section in ("inputs", "outputs"):
        record = manifest[section][0]
        assert {"path", "filename", "size", "sha256", "schema", "unit"} <= set(record)
