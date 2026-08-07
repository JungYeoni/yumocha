"""지역×연도 계획예산·합계출산율 기초패널 생성 유틸.

시행계획 long 파일에는 당해예산과 전년도예산이 함께 있으므로 지역 총액은
각 연도 문서의 당해예산만 사용한다. 세부사업 예산 결측은 0으로 추정하지
않고 합계에서 제외하되, 지역×연도별 결측 건수를 품질정보로 보존한다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PANEL_KEY = ["지역", "연도"]
BUDGET_REQUIRED_COLUMNS = {
    "지역",
    "연도",
    "세부사업명",
    "예산구분",
    "예산액",
}
QA_REQUIRED_COLUMNS = {"지역", "결과", "허용기준결과"}


def _require_columns(df: pd.DataFrame, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} 필수 컬럼 누락: {missing}")


def _validate_unique_panel(
    df: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
    label: str,
) -> None:
    duplicated = df.duplicated(PANEL_KEY, keep=False)
    if duplicated.any():
        duplicate_keys = df.loc[duplicated, PANEL_KEY].drop_duplicates()
        raise ValueError(f"{label} 지역×연도 중복: {duplicate_keys.to_dict(orient='records')}")

    actual = set(map(tuple, df[PANEL_KEY].itertuples(index=False, name=None)))
    expected = {(region, year) for region in expected_regions for year in expected_years}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} 지역×연도 조합 불일치: 누락={missing[:10]}, 예상외={unexpected[:10]}"
        )


def _validate_contiguous_region_years(df: pd.DataFrame, *, label: str) -> None:
    """지역별 연도가 중복 없이 1년 간격으로 이어지는지 검증한다."""

    _require_columns(df, set(PANEL_KEY), label=label)
    duplicated = df.duplicated(PANEL_KEY, keep=False)
    if duplicated.any():
        duplicate_keys = df.loc[duplicated, PANEL_KEY].drop_duplicates()
        raise ValueError(f"{label} 지역×연도 중복: {duplicate_keys.to_dict(orient='records')}")

    ordered = df.sort_values(PANEL_KEY)
    year_gaps = ordered.groupby("지역", sort=False)["연도"].diff()
    invalid = year_gaps.notna() & year_gaps.ne(1)
    if invalid.any():
        samples = ordered.loc[invalid, PANEL_KEY].to_dict(orient="records")
        raise ValueError(f"{label} 지역별 연도가 연속적이지 않습니다: {samples[:10]}")


def _require_finite_numeric(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    """지정 열이 결측 없는 유한 숫자인지 검증한다."""

    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if invalid.any():
            samples = df.loc[invalid, PANEL_KEY + [column]].head(10).to_dict(orient="records")
            raise ValueError(f"{label} {column}에 결측 또는 비수치 값이 있습니다: {samples}")


def _pooled_z_score(series: pd.Series, *, label: str) -> pd.Series:
    """전체 관측치 평균과 모집단 표준편차(ddof=0)로 z-score를 계산한다."""

    mean = series.mean()
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 0:
        raise ValueError(f"{label} pooled z-score를 계산할 분산이 없습니다.")
    return (series - mean) / std


def _normalize_expected_axes(
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
) -> tuple[list[str], list[int]]:
    regions = [str(region).strip() for region in expected_regions]
    years = [int(year) for year in expected_years]
    if not regions or not years:
        raise ValueError("expected_regions와 expected_years는 비어 있을 수 없습니다.")
    if len(regions) != len(set(regions)):
        raise ValueError("expected_regions에 중복이 있습니다.")
    if len(years) != len(set(years)):
        raise ValueError("expected_years에 중복이 있습니다.")
    return regions, years


def build_current_budget_panel(
    detail: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
) -> pd.DataFrame:
    """정제된 세부사업 long 데이터에서 당해 계획예산 패널을 만든다.

    파일 탐색과 Excel 구조 해석은 호출자 책임이다. 이 함수가 지역×연도
    집계와 완전격자 검증의 단일 구현이며, 파일 기반 로더와 #81 잠정
    workbook 로더가 함께 재사용한다.
    """

    _require_columns(detail, BUDGET_REQUIRED_COLUMNS, label="detail")
    regions, years = _normalize_expected_axes(expected_regions, expected_years)

    current = detail.loc[detail["예산구분"].eq("당해예산")].copy()
    if current.empty:
        raise ValueError("detail에 당해예산 행이 없습니다.")

    current["연도"] = pd.to_numeric(current["연도"], errors="raise").astype(int)
    current["지역"] = current["지역"].astype("string").str.strip()
    numeric = pd.to_numeric(current["예산액"], errors="coerce")
    invalid = current["예산액"].notna() & numeric.isna()
    if invalid.any():
        samples = current.loc[invalid, "예산액"].astype(str).unique()[:5].tolist()
        raise ValueError(f"detail 예산액 숫자 변환 실패: {samples}")
    current["예산액_숫자"] = numeric

    if "예산_비수치" in current.columns:
        current["예산_비수치"] = current["예산_비수치"].fillna(False).astype(bool)
    else:
        current["예산_비수치"] = False

    panel = (
        current.groupby(PANEL_KEY, as_index=False)
        .agg(
            당해계획예산_백만원=(
                "예산액_숫자",
                lambda values: values.sum(min_count=1),
            ),
            세부사업수=("세부사업명", "size"),
            예산금액_존재_사업수=("예산액_숫자", "count"),
            예산결측_사업수=("예산액_숫자", lambda values: int(values.isna().sum())),
            예산비수치_사업수=("예산_비수치", "sum"),
            음수예산_사업수=("예산액_숫자", lambda values: int((values < 0).sum())),
        )
        .sort_values(PANEL_KEY)
        .reset_index(drop=True)
    )
    panel["연도"] = panel["연도"].astype(int)
    panel["예산비수치_사업수"] = panel["예산비수치_사업수"].astype(int)
    _validate_unique_panel(
        panel,
        expected_regions=regions,
        expected_years=years,
        label="계획예산 패널",
    )
    return panel


def load_current_budget_panel(
    interim_dir: str | Path,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
) -> pd.DataFrame:
    """연도별 시도 long 파일에서 당해예산만 합산한다.

    반환 합계는 결측 예산을 0으로 대체하지 않는다. 일부 세부사업이 결측이면
    숫자가 있는 행만 합산하고 ``예산결측_사업수``에 누락 규모를 기록한다.
    지역×연도의 모든 세부사업 예산이 결측이면 합계도 결측으로 유지한다.
    """

    regions, years = _normalize_expected_axes(expected_regions, expected_years)
    interim_dir = Path(interim_dir)
    frames: list[pd.DataFrame] = []

    for year in years:
        files = sorted(interim_dir.glob(f"*/{year}_*_세부사업_정제_long.csv"))
        if len(files) != len(regions):
            raise ValueError(
                f"{year}년 long 파일 수 불일치: 기대={len(regions)}, 실제={len(files)}"
            )

        file_regions: set[str] = set()
        for path in files:
            df = pd.read_csv(path)
            _require_columns(df, BUDGET_REQUIRED_COLUMNS, label=str(path))

            current = df.loc[df["예산구분"].eq("당해예산")].copy()
            if current.empty:
                raise ValueError(f"{path}에 당해예산 행이 없습니다.")

            row_years = set(pd.to_numeric(current["연도"], errors="raise").astype(int))
            if row_years != {year}:
                raise ValueError(
                    f"{path} 당해예산 연도 불일치: 파일연도={year}, 행연도={sorted(row_years)}"
                )

            row_regions = set(current["지역"].dropna().astype(str).str.strip())
            if len(row_regions) != 1:
                raise ValueError(f"{path} 지역값이 하나가 아닙니다: {sorted(row_regions)}")
            region = next(iter(row_regions))
            file_regions.add(region)

            frames.append(current)

        missing_regions = sorted(set(regions) - file_regions)
        unexpected_regions = sorted(file_regions - set(regions))
        if missing_regions or unexpected_regions:
            raise ValueError(
                f"{year}년 파일 지역 불일치: 누락={missing_regions}, 예상외={unexpected_regions}"
            )

    detail = pd.concat(frames, ignore_index=True)
    return build_current_budget_panel(
        detail,
        expected_regions=regions,
        expected_years=years,
    )


def load_fertility_panel(
    fertility_path: str | Path,
    region_mapping_path: str | Path,
    *,
    expected_years: Sequence[int],
    encoding: str = "cp949",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KOSIS 2행 헤더 CSV를 시도×연도 합계출산율 long 패널로 변환한다.

    Returns:
        ``(시도 패널, 전국 추세)``. 시도 패널의 지역명은 기존 lookup의
        ``지역`` 축약명으로 통일한다.
    """

    raw = pd.read_csv(fertility_path, encoding=encoding, header=[0, 1])
    if "합계출산율" not in raw.columns.get_level_values(0):
        raise ValueError("합계출산율 상위 헤더를 찾을 수 없습니다.")

    region_source = raw.iloc[:, 0].astype("string").str.strip()
    tfr = raw["합계출산율"].copy()

    year_columns: dict[str, int] = {}
    year_sources: dict[int, list[str]] = {}
    for column in tfr.columns:
        match = re.search(r"\d{4}", str(column))
        if match:
            year = int(match.group())
            year_columns[column] = year
            year_sources.setdefault(year, []).append(str(column))

    conflicting_years = {
        year: sources for year, sources in year_sources.items() if len(sources) > 1
    }
    if conflicting_years:
        raise ValueError(f"합계출산율 연도 컬럼 중복 매핑: {conflicting_years}")

    tfr = tfr.rename(columns=year_columns)

    missing_years = sorted(set(expected_years) - set(tfr.columns))
    if missing_years:
        raise ValueError(f"합계출산율 연도 컬럼 누락: {missing_years}")

    tfr = tfr[list(expected_years)].apply(pd.to_numeric, errors="raise")
    invalid_tfr = tfr.isna() | tfr.lt(0) | tfr.gt(5)
    if invalid_tfr.any(axis=None):
        invalid_rows, invalid_columns = invalid_tfr.to_numpy().nonzero()
        samples = [
            {
                "지역명_전체": region_source.iloc[row],
                "연도": int(tfr.columns[column]),
                "합계출산율": tfr.iloc[row, column],
            }
            for row, column in zip(invalid_rows[:10], invalid_columns[:10], strict=True)
        ]
        raise ValueError(f"합계출산율 결측 또는 허용범위(0~5) 이탈: {samples}")

    tfr.insert(0, "지역명_전체", region_source)
    long = tfr.melt(
        id_vars="지역명_전체",
        var_name="연도",
        value_name="합계출산율",
    )
    long["연도"] = long["연도"].astype(int)

    nationwide = (
        long.loc[long["지역명_전체"].eq("전국"), ["연도", "합계출산율"]]
        .sort_values("연도")
        .reset_index(drop=True)
    )

    mapping = pd.read_csv(region_mapping_path)
    _require_columns(mapping, {"지역", "지역명_전체"}, label=str(region_mapping_path))
    if mapping["지역명_전체"].duplicated().any():
        raise ValueError("지역 매핑의 지역명_전체가 중복됩니다.")

    panel = long.loc[~long["지역명_전체"].eq("전국")].merge(
        mapping[["지역", "지역명_전체"]],
        on="지역명_전체",
        how="left",
        validate="many_to_one",
    )
    unmapped = panel.loc[panel["지역"].isna(), "지역명_전체"].drop_duplicates().tolist()
    if unmapped:
        raise ValueError(f"합계출산율 지역명 미매핑: {unmapped}")

    panel = panel[["지역", "연도", "합계출산율"]].sort_values(PANEL_KEY).reset_index(drop=True)
    _validate_unique_panel(
        panel,
        expected_regions=mapping["지역"].tolist(),
        expected_years=expected_years,
        label="합계출산율 패널",
    )
    return panel, nationwide


def _compare_budget_totals(
    budget_panel: pd.DataFrame,
    source_totals: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(
        budget_panel,
        {"지역", "연도", "당해계획예산_백만원"},
        label="budget_panel",
    )
    comparison = budget_panel[PANEL_KEY + ["당해계획예산_백만원"]].merge(
        source_totals,
        on=PANEL_KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    comparison["집계차이_백만원"] = (
        comparison["당해계획예산_백만원"] - comparison["원본당해예산합계_백만원"]
    )

    unmatched = comparison.loc[~comparison["_merge"].eq("both")]
    if not unmatched.empty:
        raise ValueError(
            f"패널·원본 예산 지역×연도 미매칭: "
            f"{unmatched[PANEL_KEY + ['_merge']].to_dict(orient='records')}"
        )
    comparison = comparison.drop(columns="_merge").sort_values(PANEL_KEY).reset_index(drop=True)

    floating_point_tolerance = 1e-6
    within_tolerance = comparison["집계차이_백만원"].abs().le(floating_point_tolerance)
    comparison.loc[within_tolerance, "집계차이_백만원"] = 0.0
    mismatched = comparison["집계차이_백만원"].isna() | comparison["집계차이_백만원"].abs().gt(
        floating_point_tolerance
    )
    if mismatched.any():
        raise ValueError(
            "패널·원본 당해예산 합계 불일치: "
            f"{comparison.loc[mismatched].to_dict(orient='records')[:10]}"
        )
    return comparison


def validate_budget_totals_against_detail(
    budget_panel: pd.DataFrame,
    detail: pd.DataFrame,
) -> pd.DataFrame:
    """패널 총액을 메모리상의 세부사업 당해예산 합계와 역대조한다."""

    _require_columns(detail, BUDGET_REQUIRED_COLUMNS, label="detail")
    current = detail.loc[detail["예산구분"].eq("당해예산"), PANEL_KEY + ["예산액"]].copy()
    if current.empty:
        raise ValueError("역대조할 당해예산 세부사업 행이 없습니다.")
    current["연도"] = pd.to_numeric(current["연도"], errors="raise").astype(int)
    numeric = pd.to_numeric(current["예산액"], errors="coerce")
    invalid = current["예산액"].notna() & numeric.isna()
    if invalid.any():
        samples = current.loc[invalid, "예산액"].astype(str).unique()[:5].tolist()
        raise ValueError(f"역대조 예산액 숫자 변환 실패: {samples}")
    current["예산액"] = numeric
    source_totals = (
        current.groupby(PANEL_KEY, as_index=False)["예산액"]
        .sum(min_count=1)
        .rename(columns={"예산액": "원본당해예산합계_백만원"})
    )
    return _compare_budget_totals(budget_panel, source_totals)


def validate_budget_totals_against_sources(
    budget_panel: pd.DataFrame,
    source_paths: Sequence[str | Path],
) -> pd.DataFrame:
    """패널 총액을 원본 long 파일의 당해예산 직접 합계와 역대조한다."""

    source_frames: list[pd.DataFrame] = []
    for source_path in source_paths:
        path = Path(source_path)
        source = pd.read_csv(path)
        _require_columns(source, BUDGET_REQUIRED_COLUMNS, label=str(path))
        source_frames.append(source)

    if not source_frames:
        raise ValueError("역대조할 예산 원본 파일이 없습니다.")

    return validate_budget_totals_against_detail(
        budget_panel,
        pd.concat(source_frames, ignore_index=True),
    )


def load_budget_qa_panel(
    reports_dir: str | Path,
    *,
    expected_years: Sequence[int],
) -> pd.DataFrame:
    """연도별 전국 QA 결과를 지역×연도 품질정보로 집계한다."""

    reports_dir = Path(reports_dir)
    frames: list[pd.DataFrame] = []
    for year in expected_years:
        path = reports_dir / "yearly" / str(year) / f"{year}_전국_QA_검증결과.csv"
        df = pd.read_csv(path)
        _require_columns(df, QA_REQUIRED_COLUMNS, label=str(path))
        df = df.copy()
        df["연도"] = year

        if "절대오차율(%)" in df.columns:
            df["QA_절대오차율"] = pd.to_numeric(df["절대오차율(%)"], errors="coerce")
        elif "오차율(%)" in df.columns:
            df["QA_절대오차율"] = pd.to_numeric(df["오차율(%)"], errors="coerce").abs()
        else:
            df["QA_절대오차율"] = pd.Series(float("nan"), index=df.index, dtype="float64")
        frames.append(df)

    qa = pd.concat(frames, ignore_index=True)
    panel = (
        qa.groupby(PANEL_KEY, as_index=False)
        .agg(
            예산_QA_그룹수=("결과", "size"),
            예산_QA_일치건수=("결과", lambda values: int(values.eq("일치").sum())),
            예산_QA_불일치건수=("결과", lambda values: int(values.eq("불일치").sum())),
            예산_QA_판정불가건수=(
                "결과",
                lambda values: int(values.eq("판정불가").sum()),
            ),
            예산_QA_허용초과건수=(
                "허용기준결과",
                lambda values: int(values.eq("초과").sum()),
            ),
            예산_QA_최대절대오차율=("QA_절대오차율", "max"),
        )
        .sort_values(PANEL_KEY)
        .reset_index(drop=True)
    )
    panel["연도"] = panel["연도"].astype(int)
    return panel


def build_budget_fertility_panel(
    budget_panel: pd.DataFrame,
    fertility_panel: pd.DataFrame,
    *,
    expected_regions: Sequence[str],
    expected_years: Sequence[int],
    qa_panel: pd.DataFrame | None = None,
    quality_notes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """계획예산·합계출산율·선택적 QA 정보를 1:1 결합한다."""

    panel = budget_panel.merge(
        fertility_panel,
        on=PANEL_KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = panel.loc[~panel["_merge"].eq("both"), PANEL_KEY + ["_merge"]]
    if not unmatched.empty:
        raise ValueError(f"예산·출산율 미매칭: {unmatched.to_dict(orient='records')}")
    panel = panel.drop(columns="_merge")

    if qa_panel is not None:
        panel = panel.merge(qa_panel, on=PANEL_KEY, how="left", validate="one_to_one")

    if quality_notes is not None:
        _require_columns(
            quality_notes,
            {"지역", "연도", "원자료_누락주의"},
            label="quality_notes",
        )
        panel = panel.merge(
            quality_notes[PANEL_KEY + ["원자료_누락주의"]],
            on=PANEL_KEY,
            how="left",
            validate="one_to_one",
        )

    if "원자료_누락주의" not in panel:
        panel["원자료_누락주의"] = pd.NA

    panel = panel.sort_values(PANEL_KEY).reset_index(drop=True)
    _validate_unique_panel(
        panel,
        expected_regions=expected_regions,
        expected_years=expected_years,
        label="기초패널",
    )
    return panel


def add_fiscal_index_features(
    panel: pd.DataFrame,
    *,
    budget_col: str = "당해계획예산_백만원",
    cpi_col: str = "소비자물가지수",
    population_col: str = "20_39세_인구_명",
    cpi_base_value: float = 100.0,
    real_price_label: str = "2020년가격",
) -> pd.DataFrame:
    """계획예산을 실질화하고 재정대응지수와 전년 변화를 생성한다.

    예산 단위는 백만원, 인구 단위는 명을 전제로 한다. 주 지수는 20~39세
    1인당 실질예산을 ``log1p`` 변환한 뒤 전체 지역×연도 관측치 기준으로
    표준화한다. 총액 기준 지수와 표시용 0~100 점수도 함께 반환한다.

    지역별 전년 변화가 실제 t-1 비교가 되도록 지역×연도 키 중복과 연도
    연속성을 검증한다. CPI와 인구는 양수, 예산은 0 이상이어야 한다.

    ``real_price_label``은 산출 컬럼명(``실질계획예산_{label}_백만원``)에 그대로
    들어간다. 호출자가 넘긴 CPI가 실제로 어느 연도 가격 기준인지와 반드시
    일치해야 한다 — 이 함수는 CPI 파일의 실제 기준연도를 검증하지 않으므로
    (그건 호출자가 CPI를 로드할 때 이미 확인했다고 가정), 라벨과 CPI가 어긋나면
    계산은 맞아도 컬럼명이 거짓 정보를 표시하게 된다.
    """

    required = {*PANEL_KEY, budget_col, cpi_col, population_col}
    _require_columns(panel, required, label="재정대응지수 패널")
    _validate_contiguous_region_years(panel, label="재정대응지수 패널")
    _require_finite_numeric(
        panel,
        [budget_col, cpi_col, population_col],
        label="재정대응지수 패널",
    )

    if not np.isfinite(cpi_base_value) or cpi_base_value <= 0:
        raise ValueError("CPI 기준값은 0보다 큰 유한 숫자여야 합니다.")

    result = panel.sort_values(PANEL_KEY).reset_index(drop=True).copy()
    budget = pd.to_numeric(result[budget_col])
    cpi = pd.to_numeric(result[cpi_col])
    population = pd.to_numeric(result[population_col])

    if budget.lt(0).any():
        samples = result.loc[budget.lt(0), PANEL_KEY + [budget_col]].head(10)
        raise ValueError(f"계획예산은 0 이상이어야 합니다: {samples.to_dict(orient='records')}")
    if cpi.le(0).any():
        samples = result.loc[cpi.le(0), PANEL_KEY + [cpi_col]].head(10)
        raise ValueError(f"소비자물가지수는 0보다 커야 합니다: {samples.to_dict(orient='records')}")
    if population.le(0).any():
        samples = result.loc[population.le(0), PANEL_KEY + [population_col]].head(10)
        raise ValueError(f"대상인구는 0보다 커야 합니다: {samples.to_dict(orient='records')}")

    real_budget_col = f"실질계획예산_{real_price_label}_백만원"
    result[real_budget_col] = budget * cpi_base_value / cpi
    result["log1p_실질계획예산"] = np.log1p(result[real_budget_col])
    result["20_39세_1인당_실질예산_원"] = result[real_budget_col] * 1_000_000 / population
    result["log1p_20_39세_1인당_실질예산"] = np.log1p(result["20_39세_1인당_실질예산_원"])
    result["총액기준_재정대응지수_z"] = _pooled_z_score(
        result["log1p_실질계획예산"],
        label="총액 기준 재정대응지수",
    )
    result["재정대응지수_z"] = _pooled_z_score(
        result["log1p_20_39세_1인당_실질예산"],
        label="인구당 재정대응지수",
    )

    score_range = result["재정대응지수_z"].max() - result["재정대응지수_z"].min()
    if not np.isfinite(score_range) or score_range <= 0:
        raise ValueError("재정대응점수 0~100 변환을 계산할 범위가 없습니다.")
    result["재정대응점수_0_100"] = (
        100 * (result["재정대응지수_z"] - result["재정대응지수_z"].min()) / score_range
    )

    grouped = result.groupby("지역", sort=False)
    result["실질예산_전년증감액_백만원"] = grouped[real_budget_col].diff()
    result["실질예산_전년증감률"] = grouped[real_budget_col].pct_change(fill_method=None)
    result["log실질예산_전년변화"] = grouped["log1p_실질계획예산"].diff()
    result["log1인당실질예산_전년변화"] = grouped["log1p_20_39세_1인당_실질예산"].diff()

    return result


def add_subarea_fiscal_index_features(
    panel: pd.DataFrame,
    *,
    group_column: str = "세부영역",
    per_capita_budget_col: str = "인구1인당_실질예산_원",
) -> pd.DataFrame:
    """세부영역별로 log1p 후 pooled z-score·0~100 점수를 계산한다(제주도 방식).

    구조환경지수(standardize_structural_indicators)와 같은 원리로, 표준화를
    세부영역 그룹 안에서만 수행한다 — 세부영역마다 예산 규모 자릿수가 크게
    달라서(예: 돌봄여건 vs 가사수행 격차) 전체를 한번에 표준화하면 규모가 큰
    영역이 지배해버린다.

    2026-08-07 팀 결정: 회귀분석에는 이 지수가 아니라 ``per_capita_budget_col``
    원본값을 그대로 쓴다. 이 함수가 만드는 z-score·0~100 점수는 구조환경지수와
    비교하거나 지역 간 상대적 위치를 보여주는 표시·시각화용이다.
    """

    required = {*PANEL_KEY, group_column, per_capita_budget_col}
    _require_columns(panel, required, label="세부영역 재정대응지수")

    per_capita = pd.to_numeric(panel[per_capita_budget_col])
    if per_capita.lt(0).any():
        samples = panel.loc[per_capita.lt(0), PANEL_KEY + [group_column, per_capita_budget_col]]
        raise ValueError(
            f"인구1인당 예산은 0 이상이어야 합니다: {samples.head(10).to_dict(orient='records')}"
        )

    result = panel.copy()
    result["log1p_인구1인당_실질예산"] = np.log1p(per_capita)

    z_score_col = "세부영역_재정대응지수_z"
    score_col = "세부영역_재정대응점수_0_100"
    result[z_score_col] = np.nan
    result[score_col] = np.nan

    for group_key, group in result.groupby(group_column, sort=False):
        z = _pooled_z_score(group["log1p_인구1인당_실질예산"], label=f"{group_key} 재정대응지수")
        z_range = z.max() - z.min()
        if not np.isfinite(z_range) or z_range <= 0:
            raise ValueError(f"{group_key} 재정대응점수 0~100 변환을 계산할 범위가 없습니다.")
        result.loc[group.index, z_score_col] = z
        result.loc[group.index, score_col] = 100 * (z - z.min()) / z_range

    return result


def add_total_expenditure_ratio(
    panel: pd.DataFrame,
    denominator: pd.DataFrame,
    *,
    budget_col: str = "당해계획예산_백만원",
    denominator_col: str = "세출예산순계액_백만원",
) -> pd.DataFrame:
    """재정대응 패널에 총세출 분모와 계획예산 비율을 결합한다.

    주 분모는 지역통합(광역본청+산하 시군구), 3개 회계
    (일반회계+기타특별회계+공기업특별회계)의 당초예산 순계다. 분자와
    분모는 모두 백만원 단위여야 하며 두 입력의 지역×연도 키가 정확히
    일치해야 한다.
    """

    denominator_columns = {
        *PANEL_KEY,
        "예산단계",
        "포함회계",
        "자치단체수",
        "세출예산순계액_원",
        denominator_col,
        "출처",
    }
    _require_columns(panel, {*PANEL_KEY, budget_col}, label="재정대응지수 패널")
    _require_columns(denominator, denominator_columns, label="총세출 분모 패널")

    for frame, label in ((panel, "재정대응지수 패널"), (denominator, "총세출 분모 패널")):
        duplicated = frame.duplicated(PANEL_KEY, keep=False)
        if duplicated.any():
            samples = frame.loc[duplicated, PANEL_KEY].drop_duplicates().head(10)
            raise ValueError(f"{label} 지역×연도 중복: {samples.to_dict(orient='records')}")

    _require_finite_numeric(panel, [budget_col], label="재정대응지수 패널")
    _require_finite_numeric(denominator, [denominator_col], label="총세출 분모 패널")

    budget = pd.to_numeric(panel[budget_col])
    if budget.lt(0).any():
        samples = panel.loc[budget.lt(0), PANEL_KEY + [budget_col]].head(10)
        raise ValueError(f"계획예산은 0 이상이어야 합니다: {samples.to_dict(orient='records')}")

    expenditure = pd.to_numeric(denominator[denominator_col])
    if expenditure.le(0).any():
        samples = denominator.loc[expenditure.le(0), PANEL_KEY + [denominator_col]].head(10)
        raise ValueError(f"총세출 분모는 0보다 커야 합니다: {samples.to_dict(orient='records')}")

    metadata_columns = sorted(denominator_columns - set(PANEL_KEY) - {denominator_col})
    missing_metadata = denominator[metadata_columns].isna().any(axis=1)
    if missing_metadata.any():
        samples = denominator.loc[missing_metadata, PANEL_KEY].head(10)
        raise ValueError(
            f"총세출 분모 메타데이터가 누락되었습니다: {samples.to_dict(orient='records')}"
        )

    denominator_subset = denominator[PANEL_KEY + sorted(denominator_columns - set(PANEL_KEY))]
    result = panel.merge(
        denominator_subset,
        on=PANEL_KEY,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = result["_merge"].ne("both")
    if unmatched.any():
        samples = result.loc[unmatched, PANEL_KEY + ["_merge"]].head(10)
        raise ValueError(f"재정대응·총세출 키 불일치: {samples.to_dict(orient='records')}")

    result = result.drop(columns="_merge").sort_values(PANEL_KEY).reset_index(drop=True)
    result["계획예산_총세출비율_pct"] = (
        pd.to_numeric(result[budget_col]) / pd.to_numeric(result[denominator_col]) * 100
    )
    result["분모대안"] = "지역통합_3개회계_순계"
    result["분자분모_단위"] = "백만원"
    return result


def add_fiscal_response_features(
    panel: pd.DataFrame,
    *,
    tfr_col: str = "합계출산율",
) -> pd.DataFrame:
    """재정반응성 기초분석용 선행 출산율 변수를 생성한다.

    지역별 연도가 빠짐없이 연속됐는지 검증해 ``shift(1)``과 ``shift(2)``가
    실제 t-1, t-2를 뜻하도록 보장한다.
    """

    _require_columns(panel, {*PANEL_KEY, tfr_col}, label="재정반응성 패널")
    _validate_contiguous_region_years(panel, label="재정반응성 패널")
    _require_finite_numeric(panel, [tfr_col], label="재정반응성 패널")

    result = panel.sort_values(PANEL_KEY).reset_index(drop=True).copy()
    grouped = result.groupby("지역", sort=False)[tfr_col]
    result["전년도_합계출산율"] = grouped.shift(1)
    result["전전년도_합계출산율"] = grouped.shift(2)
    result["직전1년_출산율하락도"] = result["전전년도_합계출산율"] - result["전년도_합계출산율"]
    return result
