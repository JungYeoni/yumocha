"""지역×연도 계획예산·합계출산율 패널 유틸 테스트."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.analysis_panel import (
    add_fiscal_response_features,
    build_budget_fertility_panel,
    load_current_budget_panel,
    load_fertility_panel,
    validate_budget_totals_against_sources,
)


REGIONS = ["서울", "부산"]
YEARS = [2020, 2021]


def _write_budget_file(
    base: Path,
    *,
    region: str,
    year: int,
    current_values: list[float | None],
    previous_values: list[float],
) -> None:
    region_dir = base / region
    region_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (current, previous) in enumerate(zip(current_values, previous_values, strict=True)):
        rows.extend(
            [
                {
                    "지역": region,
                    "연도": year,
                    "세부사업명": f"사업{index}",
                    "예산구분": "당해예산",
                    "예산액": current,
                },
                {
                    "지역": region,
                    "연도": year - 1,
                    "세부사업명": f"사업{index}",
                    "예산구분": "전년도예산",
                    "예산액": previous,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(
        region_dir / f"{year}_{region}_세부사업_정제_long.csv",
        index=False,
    )


def test_load_current_budget_panel_uses_only_current_and_tracks_missing(tmp_path):
    for year in YEARS:
        _write_budget_file(
            tmp_path,
            region="서울",
            year=year,
            current_values=[10.0, None],
            previous_values=[1000.0, 2000.0],
        )
        _write_budget_file(
            tmp_path,
            region="부산",
            year=year,
            current_values=[20.0, -1.0],
            previous_values=[3000.0, 4000.0],
        )

    panel = load_current_budget_panel(
        tmp_path,
        expected_regions=REGIONS,
        expected_years=YEARS,
    )

    seoul_2020 = panel.query("지역 == '서울' and 연도 == 2020").iloc[0]
    busan_2020 = panel.query("지역 == '부산' and 연도 == 2020").iloc[0]
    assert seoul_2020["당해계획예산_백만원"] == 10.0
    assert seoul_2020["예산결측_사업수"] == 1
    assert busan_2020["당해계획예산_백만원"] == 19.0
    assert busan_2020["음수예산_사업수"] == 1


def test_load_current_budget_panel_rejects_wrong_current_year(tmp_path):
    for region in REGIONS:
        _write_budget_file(
            tmp_path,
            region=region,
            year=2020,
            current_values=[10.0],
            previous_values=[9.0],
        )
    path = tmp_path / "서울" / "2020_서울_세부사업_정제_long.csv"
    df = pd.read_csv(path)
    df.loc[df["예산구분"].eq("당해예산"), "연도"] = 2019
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="당해예산 연도 불일치"):
        load_current_budget_panel(
            tmp_path,
            expected_regions=REGIONS,
            expected_years=[2020],
        )


def test_load_fertility_panel_maps_regions_and_excludes_nationwide(tmp_path):
    fertility_path = tmp_path / "fertility.csv"
    mapping_path = tmp_path / "mapping.csv"
    columns = pd.MultiIndex.from_tuples(
        [
            ("시군구별", "시군구별"),
            ("합계출산율", "2020"),
            ("합계출산율", "2021"),
            ("합계출산율", "2025 p)"),
        ]
    )
    pd.DataFrame(
        [
            ["전국", 0.84, 0.81, 0.75],
            ["서울특별시", 0.64, 0.63, 0.60],
            ["부산광역시", 0.75, 0.73, 0.70],
        ],
        columns=columns,
    ).to_csv(fertility_path, index=False, encoding="cp949")
    pd.DataFrame(
        {
            "지역": REGIONS,
            "지역명_전체": ["서울특별시", "부산광역시"],
        }
    ).to_csv(mapping_path, index=False)

    panel, nationwide = load_fertility_panel(
        fertility_path,
        mapping_path,
        expected_years=YEARS,
    )

    assert len(panel) == 4
    assert set(panel["지역"]) == set(REGIONS)
    assert set(panel["연도"]) == set(YEARS)
    assert nationwide["합계출산율"].tolist() == [0.84, 0.81]


def test_load_fertility_panel_rejects_out_of_range_value(tmp_path):
    fertility_path = tmp_path / "fertility.csv"
    mapping_path = tmp_path / "mapping.csv"
    columns = pd.MultiIndex.from_tuples(
        [
            ("시군구별", "시군구별"),
            ("합계출산율", "2020"),
            ("합계출산율", "2021"),
        ]
    )
    pd.DataFrame(
        [
            ["전국", 0.84, 0.81],
            ["서울특별시", 0.64, 5.1],
            ["부산광역시", 0.75, 0.73],
        ],
        columns=columns,
    ).to_csv(fertility_path, index=False, encoding="cp949")
    pd.DataFrame(
        {
            "지역": REGIONS,
            "지역명_전체": ["서울특별시", "부산광역시"],
        }
    ).to_csv(mapping_path, index=False)

    with pytest.raises(ValueError, match=r"허용범위\(0~5\) 이탈"):
        load_fertility_panel(
            fertility_path,
            mapping_path,
            expected_years=YEARS,
        )


def test_validate_budget_totals_against_sources(tmp_path):
    source_paths = []
    for region, values in {"서울": [10.0, None], "부산": [20.0, -1.0]}.items():
        _write_budget_file(
            tmp_path,
            region=region,
            year=2020,
            current_values=values,
            previous_values=[100.0, 200.0],
        )
        source_paths.append(tmp_path / region / f"2020_{region}_세부사업_정제_long.csv")

    panel = load_current_budget_panel(
        tmp_path,
        expected_regions=REGIONS,
        expected_years=[2020],
    )
    comparison = validate_budget_totals_against_sources(panel, source_paths)

    assert comparison["집계차이_백만원"].eq(0).all()

    panel.loc[panel["지역"].eq("서울"), "당해계획예산_백만원"] += 1
    with pytest.raises(ValueError, match="당해예산 합계 불일치"):
        validate_budget_totals_against_sources(panel, source_paths)


def test_build_panel_requires_one_to_one_complete_match():
    budget = pd.DataFrame(
        {
            "지역": ["서울", "부산"],
            "연도": [2020, 2020],
            "당해계획예산_백만원": [10.0, 20.0],
        }
    )
    fertility = pd.DataFrame(
        {
            "지역": ["서울"],
            "연도": [2020],
            "합계출산율": [0.64],
        }
    )

    with pytest.raises(ValueError, match="예산·출산율 미매칭"):
        build_budget_fertility_panel(
            budget,
            fertility,
            expected_regions=REGIONS,
            expected_years=[2020],
        )


def test_add_fiscal_response_features_uses_only_prior_tfr():
    panel = pd.DataFrame(
        {
            "지역": ["서울"] * 3,
            "연도": [2020, 2021, 2022],
            "합계출산율": [0.70, 0.60, 0.55],
        }
    )

    result = add_fiscal_response_features(panel)

    row_2022 = result.loc[result["연도"].eq(2022)].iloc[0]
    assert row_2022["전년도_합계출산율"] == 0.60
    assert row_2022["전전년도_합계출산율"] == 0.70
    assert row_2022["직전1년_출산율하락도"] == pytest.approx(0.10)
