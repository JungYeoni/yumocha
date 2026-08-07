from pathlib import Path

import pandas as pd

import pytest

from scripts.build_subarea_fiscal_response_variable import (
    build_subarea_response_variable,
    load_prime_age_population,
    load_total_population,
)


def _write_population_workbook(path: Path, *, prime_age_value: int = 1_000) -> None:
    rows = []
    regions = {
        "전국": None,
        "서울특별시": "서울",
        "부산광역시": "부산",
    }
    ages = [("계", 1_000_000)] + [(f"{age}세", prime_age_value) for age in range(20, 40)]
    for full_name in regions:
        for age, value in ages:
            for item in ("총인구수[명]", "여자인구수[명]"):
                row = {"행정구역(시군구)별": full_name, "연령별": age, "항목": item}
                for year in range(2016, 2025):
                    row[f"{year} 년"] = value
                rows.append(row)
    pd.DataFrame(rows).to_excel(path, index=False)


def _write_mapping(path: Path) -> None:
    pd.DataFrame(
        {
            "지역": ["서울", "부산"],
            "지역명_전체": ["서울특별시", "부산광역시"],
        }
    ).to_csv(path, index=False)


def test_load_total_population_extracts_total_and_maps_regions(tmp_path: Path):
    population_path = tmp_path / "population.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    _write_population_workbook(population_path)
    _write_mapping(mapping_path)

    result = load_total_population(population_path, mapping_path)

    assert len(result) == 2 * 9
    assert set(result["지역"]) == {"서울", "부산"}
    assert result["전체인구_명"].eq(1_000_000).all()
    assert "전국" not in result["지역"].tolist()


def test_load_total_population_rejects_unmapped_region(tmp_path: Path):
    population_path = tmp_path / "population.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    _write_population_workbook(population_path)
    pd.DataFrame({"지역": ["서울"], "지역명_전체": ["서울특별시"]}).to_csv(
        mapping_path, index=False
    )

    with pytest.raises(ValueError, match="지역명 매핑 실패"):
        load_total_population(population_path, mapping_path)


def test_load_prime_age_population_sums_single_ages_and_maps_regions(tmp_path: Path):
    population_path = tmp_path / "population.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    _write_population_workbook(population_path, prime_age_value=1_000)
    _write_mapping(mapping_path)

    result = load_prime_age_population(population_path, mapping_path)

    assert len(result) == 2 * 9
    assert set(result["지역"]) == {"서울", "부산"}
    assert result["20_39세인구_명"].eq(1_000 * 20).all()
    assert "전국" not in result["지역"].tolist()


def test_load_prime_age_population_rejects_missing_age_label(tmp_path: Path):
    population_path = tmp_path / "population.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    _write_population_workbook(population_path, prime_age_value=1_000)
    _write_mapping(mapping_path)

    raw = pd.read_excel(population_path)
    raw = raw.loc[~((raw["연령별"] == "39세") & (raw["항목"] == "총인구수[명]"))]
    raw.to_excel(population_path, index=False)

    with pytest.raises(ValueError, match="20~39세 연령 라벨 불일치"):
        load_prime_age_population(population_path, mapping_path)


def test_load_prime_age_population_rejects_age_missing_for_single_region_only(
    tmp_path: Path,
):
    """파일 전체엔 "39세"가 있어도(부산엔 있음) 서울만 빠지면 잡아야 한다."""
    population_path = tmp_path / "population.xlsx"
    mapping_path = tmp_path / "mapping.csv"
    _write_population_workbook(population_path, prime_age_value=1_000)
    _write_mapping(mapping_path)

    raw = pd.read_excel(population_path)
    drop_seoul_39 = (
        raw["행정구역(시군구)별"].eq("서울특별시")
        & raw["연령별"].eq("39세")
        & raw["항목"].eq("총인구수[명]")
    )
    raw = raw.loc[~drop_seoul_39]
    raw.to_excel(population_path, index=False)

    with pytest.raises(ValueError, match="일부 지역에서 불완전"):
        load_prime_age_population(population_path, mapping_path)


def test_build_subarea_response_variable_computes_expected_per_capita_value():
    sub_area_panel = pd.DataFrame(
        {
            "지역": ["A", "A"],
            "연도": [2016, 2017],
            "세부영역": ["X", "X"],
            "당해계획예산_백만원_provisional": [100.0, 110.0],
        }
    )
    cpi = pd.DataFrame(
        {
            "연도": [2016, 2017, 2024],
            "소비자물가지수": [80.0, 90.0, 100.0],
            "기준연도": ["2024=100"] * 3,
        }
    )
    population = pd.DataFrame(
        {
            "지역": ["A", "A"],
            "연도": [2016, 2017],
            "전체인구_명": [1_000_000, 1_000_000],
        }
    )

    result = build_subarea_response_variable(sub_area_panel, cpi, population)

    row_2016 = result.loc[result["연도"].eq(2016)].iloc[0]
    row_2017 = result.loc[result["연도"].eq(2017)].iloc[0]
    assert row_2016["인구1인당_실질예산_원"] == pytest.approx(125.0)
    assert row_2017["인구1인당_실질예산_원"] == pytest.approx(122.222222, rel=1e-6)


def test_build_subarea_response_variable_accepts_alternate_population_column():
    sub_area_panel = pd.DataFrame(
        {
            "지역": ["A"],
            "연도": [2024],
            "세부영역": ["X"],
            "당해계획예산_백만원_provisional": [100.0],
        }
    )
    cpi = pd.DataFrame({"연도": [2024], "소비자물가지수": [100.0], "기준연도": ["2024=100"]})
    population = pd.DataFrame({"지역": ["A"], "연도": [2024], "20_39세인구_명": [500_000]})

    result = build_subarea_response_variable(
        sub_area_panel, cpi, population, population_column="20_39세인구_명"
    )

    assert result.iloc[0]["인구1인당_실질예산_원"] == pytest.approx(200.0)


def test_build_subarea_response_variable_rejects_unmatched_population():
    sub_area_panel = pd.DataFrame(
        {
            "지역": ["A"],
            "연도": [2016],
            "세부영역": ["X"],
            "당해계획예산_백만원_provisional": [100.0],
        }
    )
    cpi = pd.DataFrame(
        {
            "연도": [2016, 2024],
            "소비자물가지수": [80.0, 100.0],
            "기준연도": ["2024=100", "2024=100"],
        }
    )
    population = pd.DataFrame({"지역": ["B"], "연도": [2016], "전체인구_명": [1_000_000]})

    with pytest.raises(ValueError, match="인구 결합 실패"):
        build_subarea_response_variable(sub_area_panel, cpi, population)
