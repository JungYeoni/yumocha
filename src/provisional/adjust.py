"""CPI input validation and real-budget calculations for #81."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


REAL_BUDGET_COLUMN = "당해계획예산_실질_백만원_provisional"
REAL_YOY_COLUMN = "전년대비_실질증감률_pct_provisional"


@dataclass(frozen=True)
class CPIData:
    series: pd.Series
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cpi(
    cpi_path: str | Path,
    *,
    encoding: str,
    year_column: str,
    index_column: str,
    unit: str,
    base_year: int,
    expected_years: list[int] | tuple[int, ...],
) -> CPIData:
    """Read one explicitly configured CPI CSV without encoding fallback."""

    path = Path(cpi_path)
    if not path.is_file():
        raise FileNotFoundError(f"CPI 파일이 없습니다: {path}")
    frame = pd.read_csv(path, encoding=encoding)
    missing_columns = {year_column, index_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CPI 필수 컬럼 누락: {sorted(missing_columns)}")

    years = pd.to_numeric(frame[year_column], errors="raise").astype(int)
    if years.duplicated().any():
        raise ValueError(f"CPI 연도 중복: {sorted(years.loc[years.duplicated()].unique())}")
    indices = pd.to_numeric(frame[index_column], errors="raise")
    if indices.isna().any() or indices.le(0).any():
        raise ValueError("CPI 지수는 결측 없이 0보다 커야 합니다.")

    expected = {int(year) for year in expected_years}
    actual = set(years)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"CPI 연도 불일치: 누락={missing}, 예상외={unexpected}")
    if int(base_year) not in actual:
        raise ValueError(f"CPI 기준연도 {base_year}가 없습니다.")

    expected_unit = f"{int(base_year)}=100"
    if unit.replace(" ", "") != expected_unit:
        raise ValueError(f"CPI 단위와 기준연도 불일치: unit={unit}, 기대={expected_unit}")
    if "기준연도" in frame.columns:
        source_units = set(frame["기준연도"].dropna().astype(str).str.replace(" ", ""))
        if source_units != {expected_unit}:
            raise ValueError(f"CPI 파일 기준연도 표기 불일치: {sorted(source_units)}")

    series = pd.Series(indices.to_numpy(dtype=float), index=years, name=index_column).sort_index()
    if float(series.loc[int(base_year)]) != 100.0:
        raise ValueError(f"CPI 기준연도 지수는 100이어야 합니다: {series.loc[int(base_year)]}")
    metadata = {
        "path": str(path.resolve()),
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
        "encoding": encoding,
        "year_column": year_column,
        "index_column": index_column,
        "unit": unit,
        "base_year": int(base_year),
        "schema": list(frame.columns),
        "formula": "real_budget = nominal_budget * (CPI_base_year / CPI_year)",
    }
    return CPIData(series=series, metadata=metadata)


def apply_cpi_adjustment(
    panel: pd.DataFrame,
    cpi: pd.Series,
    *,
    base_year: int = 2020,
    budget_column: str = "당해계획예산_백만원",
    group_columns: tuple[str, ...] = ("지역",),
) -> pd.DataFrame:
    """Convert nominal budgets to base-year prices and calculate real YoY percent.

    Formula: ``nominal * CPI(base_year) / CPI(year)``. The first year in each
    group has no prior observation, so its YoY value is explicitly missing.
    """

    required = {"연도", budget_column, *group_columns}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"panel 필수 컬럼 누락: {missing}")
    if panel.duplicated([*group_columns, "연도"]).any():
        raise ValueError("CPI 적용 패널의 그룹×연도 키가 중복됩니다.")
    if int(base_year) not in cpi.index:
        raise ValueError(f"CPI 기준연도 {base_year}가 없습니다.")

    result = panel.copy()
    result["연도"] = pd.to_numeric(result["연도"], errors="raise").astype(int)
    nominal = pd.to_numeric(result[budget_column], errors="coerce")
    invalid = result[budget_column].notna() & nominal.isna()
    if invalid.any():
        raise ValueError("명목예산에 숫자로 변환할 수 없는 값이 있습니다.")
    missing_cpi_years = sorted(set(result["연도"]) - set(map(int, cpi.index)))
    if missing_cpi_years:
        raise ValueError(f"패널 연도에 대응하는 CPI가 없습니다: {missing_cpi_years}")

    base_index = float(cpi.loc[int(base_year)])
    result["CPI_지수"] = result["연도"].map(cpi.astype(float))
    result["CPI_기준연도"] = int(base_year)
    result[REAL_BUDGET_COLUMN] = nominal * (base_index / result["CPI_지수"])
    result = result.sort_values([*group_columns, "연도"]).reset_index(drop=True)
    result[REAL_YOY_COLUMN] = (
        result.groupby(list(group_columns), sort=False)[REAL_BUDGET_COLUMN]
        .pct_change(fill_method=None)
        .mul(100)
    )
    first_in_group = result.groupby(list(group_columns), sort=False).cumcount().eq(0)
    if result.loc[first_in_group, REAL_YOY_COLUMN].notna().any():
        raise AssertionError("그룹별 첫 연도의 실질 증감률은 결측이어야 합니다.")
    return result
