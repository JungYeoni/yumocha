"""지역×연도 계획예산·합계출산율 기초패널 생성 유틸.

시행계획 long 파일에는 당해예산과 전년도예산이 함께 있으므로 지역 총액은
각 연도 문서의 당해예산만 사용한다. 세부사업 예산 결측은 0으로 추정하지
않고 합계에서 제외하되, 지역×연도별 결측 건수를 품질정보로 보존한다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

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

    interim_dir = Path(interim_dir)
    frames: list[pd.DataFrame] = []

    for year in expected_years:
        files = sorted(interim_dir.glob(f"*/{year}_*_세부사업_정제_long.csv"))
        if len(files) != len(expected_regions):
            raise ValueError(
                f"{year}년 long 파일 수 불일치: 기대={len(expected_regions)}, 실제={len(files)}"
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

            regions = set(current["지역"].dropna().astype(str).str.strip())
            if len(regions) != 1:
                raise ValueError(f"{path} 지역값이 하나가 아닙니다: {sorted(regions)}")
            region = next(iter(regions))
            file_regions.add(region)

            numeric = pd.to_numeric(current["예산액"], errors="coerce")
            invalid = current["예산액"].notna() & numeric.isna()
            if invalid.any():
                samples = current.loc[invalid, "예산액"].astype(str).unique()[:5].tolist()
                raise ValueError(f"{path} 예산액 숫자 변환 실패: {samples}")

            current["예산액_숫자"] = numeric
            frames.append(current)

        missing_regions = sorted(set(expected_regions) - file_regions)
        unexpected_regions = sorted(file_regions - set(expected_regions))
        if missing_regions or unexpected_regions:
            raise ValueError(
                f"{year}년 파일 지역 불일치: 누락={missing_regions}, 예상외={unexpected_regions}"
            )

    detail = pd.concat(frames, ignore_index=True)
    panel = (
        detail.groupby(PANEL_KEY, as_index=False)
        .agg(
            당해계획예산_백만원=(
                "예산액_숫자",
                lambda values: values.sum(min_count=1),
            ),
            세부사업수=("세부사업명", "size"),
            예산금액_존재_사업수=("예산액_숫자", "count"),
            예산결측_사업수=("예산액_숫자", lambda values: int(values.isna().sum())),
            음수예산_사업수=("예산액_숫자", lambda values: int((values < 0).sum())),
        )
        .sort_values(PANEL_KEY)
        .reset_index(drop=True)
    )
    panel["연도"] = panel["연도"].astype(int)
    _validate_unique_panel(
        panel,
        expected_regions=expected_regions,
        expected_years=expected_years,
        label="계획예산 패널",
    )
    return panel


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
    for column in tfr.columns:
        match = re.search(r"\d{4}", str(column))
        if match:
            year_columns[column] = int(match.group())
    tfr = tfr.rename(columns=year_columns)

    missing_years = sorted(set(expected_years) - set(tfr.columns))
    if missing_years:
        raise ValueError(f"합계출산율 연도 컬럼 누락: {missing_years}")

    tfr = tfr[list(expected_years)].apply(pd.to_numeric, errors="raise")
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
            df["QA_절대오차율"] = pd.NA
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


def add_fiscal_response_features(
    panel: pd.DataFrame,
    *,
    tfr_col: str = "합계출산율",
) -> pd.DataFrame:
    """재정반응성 기초분석용 선행 출산율 변수를 생성한다."""

    result = panel.sort_values(PANEL_KEY).copy()
    grouped = result.groupby("지역", sort=False)[tfr_col]
    result["전년도_합계출산율"] = grouped.shift(1)
    result["전전년도_합계출산율"] = grouped.shift(2)
    result["직전1년_출산율하락도"] = result["전전년도_합계출산율"] - result["전년도_합계출산율"]
    return result
