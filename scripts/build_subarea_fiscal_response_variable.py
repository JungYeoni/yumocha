"""#62/#98 회귀분석의 핵심 변수 — 세부영역별 인구1인당 실질예산액을 만든다.

2026-08-07 팀 결정(이슈 #62 코멘트):
- 재정대응지수도 구조환경지수처럼 세부영역별로 산출한다.
- 회귀분석에는 재정대응지수(z-score)가 아니라 **인구1인당 예산액 자체**를 쓴다.
- 인구 분모는 전체인구 1인당이 주 기준(제주도 방식). 20~39세는 참고용 추가.

입력:
- #81 `provisional_sub_area_panel.csv`(지역×연도×세부영역 명목예산, 세부영역 12개 —
  "지표체계 외" 포함. 회귀에는 실질 11개 세부영역만 쓰겠지만 완전격자 계산엔 12개
  그대로 둔다)
- CPI 2024=100
- 전체인구·20~39세인구(주민등록인구 원자료, 같은 파일에서 연령 필터만 다르게)

파이프라인 코드(src/provisional/, src/features/analysis_panel.py)는 호출만 하고
수정하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUB_AREA_PANEL = (
    REPO_ROOT / "data" / "interim" / "provisional" / "provisional_sub_area_panel.csv"
)
DEFAULT_CPI_FILE = REPO_ROOT / "data" / "lookup" / "연도별_소비자물가지수_2024기준.csv"
DEFAULT_MAPPING = REPO_ROOT / "data" / "lookup" / "시도_지역코드_매핑.csv"
DEFAULT_POPULATION_PATH = (
    REPO_ROOT
    / "data"
    / "raw"
    / "지표별_원데이터"
    / "1-1.1. 청년고용률 원자료"
    / "2016-2025_행정구역(시도)별_1세별_주민등록인구_20260715.xlsx"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis"

YEARS = list(range(2016, 2025))
CPI_BASE_YEAR = 2024


def load_total_population(population_path: Path, mapping_path: Path) -> pd.DataFrame:
    """지역×연도 전체인구를 만든다(연령별='계', 항목='총인구수[명]', 전국 제외)."""
    raw = pd.read_excel(population_path)
    required = {"행정구역(시군구)별", "연령별", "항목"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"인구 원자료 필수 컬럼 누락: {sorted(missing)}")

    total = raw.loc[raw["연령별"].eq("계") & raw["항목"].eq("총인구수[명]")].copy()
    year_columns = {
        column: int(str(column).split()[0])
        for column in total.columns
        if str(column).split()[0].isdigit() and int(str(column).split()[0]) in YEARS
    }
    if len(year_columns) != len(YEARS):
        raise ValueError(f"인구 원자료 연도 컬럼 불일치: {sorted(year_columns.values())}")

    mapping = pd.read_csv(mapping_path)
    total = total.rename(columns={"행정구역(시군구)별": "지역명_전체"})
    total = total.loc[total["지역명_전체"].ne("전국"), ["지역명_전체", *year_columns]]

    population = (
        total.rename(columns=year_columns)
        .melt(id_vars="지역명_전체", var_name="연도", value_name="전체인구_명")
        .merge(
            mapping[["지역", "지역명_전체"]],
            on="지역명_전체",
            how="left",
            validate="many_to_one",
        )[["지역", "연도", "전체인구_명"]]
    )
    if population["지역"].isna().any():
        unmapped = sorted(
            total.loc[~total["지역명_전체"].isin(mapping["지역명_전체"]), "지역명_전체"].unique()
        )
        raise ValueError(f"지역명 매핑 실패: {unmapped}")
    expected_rows = mapping["지역"].nunique() * len(YEARS)
    if len(population) != expected_rows:
        raise ValueError(f"인구 패널 행 수 불일치: 기대={expected_rows}, 실제={len(population)}")
    if population.duplicated(["지역", "연도"]).any():
        raise ValueError("인구 패널 지역·연도 키 중복")
    if population["전체인구_명"].le(0).any():
        raise ValueError("인구 패널에 0 이하 값이 있습니다.")
    return population.sort_values(["지역", "연도"]).reset_index(drop=True)


def build_subarea_response_variable(
    sub_area_panel: pd.DataFrame,
    cpi: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    """세부영역별 실질예산·인구1인당 실질예산액을 계산한다."""
    from src.provisional.adjust import REAL_BUDGET_COLUMN, apply_cpi_adjustment

    adjusted = apply_cpi_adjustment(
        sub_area_panel,
        cpi.set_index("연도")["소비자물가지수"],
        base_year=CPI_BASE_YEAR,
        budget_column="당해계획예산_백만원_provisional",
        group_columns=("지역", "세부영역"),
    )

    merged = adjusted.merge(population, on=["지역", "연도"], how="left", validate="many_to_one")
    if merged["전체인구_명"].isna().any():
        raise ValueError("인구 결합 실패: 매칭 안 된 지역·연도가 있습니다.")

    merged["인구1인당_실질예산_원"] = merged[REAL_BUDGET_COLUMN] * 1_000_000 / merged["전체인구_명"]
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub-area-panel", type=Path, default=DEFAULT_SUB_AREA_PANEL)
    parser.add_argument("--cpi-file", type=Path, default=DEFAULT_CPI_FILE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sub_area_panel.is_file():
        raise FileNotFoundError(f"세부영역 예산 패널이 없습니다: {args.sub_area_panel}")

    sub_area_panel = pd.read_csv(args.sub_area_panel)
    cpi = pd.read_csv(args.cpi_file, encoding="utf-8-sig")
    population = load_total_population(args.population, args.mapping)

    result = build_subarea_response_variable(sub_area_panel, cpi, population)

    expected_rows = 17 * len(YEARS) * 12
    if len(result) != expected_rows:
        raise ValueError(f"결과 행 수 불일치: 기대={expected_rows}, 실제={len(result)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "2016-2024_세부영역별_인구1인당_실질예산액.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"저장: {output_path} ({len(result)}행)")
    print(
        result.groupby("세부영역", as_index=False)["인구1인당_실질예산_원"]
        .mean()
        .round(0)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
