"""#62 민감도 분석 — 세부영역 재정반응성 회귀(모형 A)의 인구 분모를 전체인구 vs
20~39세인구로 바꿔가며 계수·p값이 얼마나 달라지는지 비교한다.

2026-08-07 팀 결정 3번: 주 기준은 전체인구 1인당(제주도 방식)이고, 20~39세는
참고용 대안이다. 이 스크립트는 그 대안이 결과를 흔드는지 확인하는 용도이며,
build_subarea_fiscal_response_variable.py/build_subarea_fiscal_response_regression_sample.py
/run_subarea_fiscal_responsiveness_regression.py의 기존 함수를 그대로 재사용하고
수정하지 않는다 — 인구 컬럼만 바꿔 같은 파이프라인을 두 번 돌린다.

ΔTFR·구조환경지수 시차 변수는 인구 분모와 무관하므로 한 번만 계산해서 두 분모
버전에 공통으로 결합한다.
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
DEFAULT_FERTILITY_PATH = (
    REPO_ROOT / "data" / "raw" / "출산동향" / "2016-2025_시도별_출생아수_합계출산율_20260703.csv"
)
DEFAULT_STRUCTURAL_SUBCATEGORY = (
    REPO_ROOT
    / "data"
    / "processed"
    / "structural_index"
    / "structural_index_pooled_subcategory_scores.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis"

YEARS = list(range(2016, 2025))
DENOMINATOR_LABELS = {"전체인구": "전체인구_명", "20_39세인구": "20_39세인구_명"}


def build_regression_sample_for_denominator(
    *,
    denominator: str,
    sub_area_panel: pd.DataFrame,
    cpi: pd.DataFrame,
    population_path: Path,
    mapping_path: Path,
    fertility_lagged: pd.DataFrame,
    structural_lagged: pd.DataFrame,
) -> pd.DataFrame:
    """분모 하나에 대해 재정대응 변수→회귀표본까지 기존 파이프라인을 그대로 돌린다."""
    from scripts.build_subarea_fiscal_response_regression_sample import build_regression_sample
    from scripts.build_subarea_fiscal_response_variable import (
        build_subarea_response_variable,
        load_prime_age_population,
        load_total_population,
    )

    if denominator == "전체인구":
        population = load_total_population(population_path, mapping_path)
    elif denominator == "20_39세인구":
        population = load_prime_age_population(population_path, mapping_path)
    else:
        raise ValueError(f"알 수 없는 denominator: {denominator}")

    fiscal_response = build_subarea_response_variable(
        sub_area_panel,
        cpi,
        population,
        population_column=DENOMINATOR_LABELS[denominator],
    )
    return build_regression_sample(fiscal_response, fertility_lagged, structural_lagged)


def build_comparison_table(results_by_denominator: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """분모별 결과를 세부영역 기준으로 옆으로 붙여 계수·p값을 나란히 비교한다."""
    columns = ["모형", "계수", "p값"]
    wide = None
    for denominator, results in results_by_denominator.items():
        renamed = results[columns].rename(
            columns={"계수": f"계수_{denominator}", "p값": f"p값_{denominator}"}
        )
        wide = renamed if wide is None else wide.merge(renamed, on="모형", how="outer")

    labels = list(results_by_denominator)
    wide["부호_일치"] = wide[f"계수_{labels[0]}"].mul(wide[f"계수_{labels[1]}"]).ge(0)
    wide["유의성_5%_일치"] = (wide[f"p값_{labels[0]}"] < 0.05) == (wide[f"p값_{labels[1]}"] < 0.05)
    return wide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub-area-panel", type=Path, default=DEFAULT_SUB_AREA_PANEL)
    parser.add_argument("--cpi-file", type=Path, default=DEFAULT_CPI_FILE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION_PATH)
    parser.add_argument("--fertility", type=Path, default=DEFAULT_FERTILITY_PATH)
    parser.add_argument(
        "--structural-subcategory-scores", type=Path, default=DEFAULT_STRUCTURAL_SUBCATEGORY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    from scripts.build_subarea_fiscal_response_regression_sample import (
        build_lagged_structural_index,
    )
    from scripts.run_subarea_fiscal_responsiveness_regression import run_subarea_models
    from src.features.analysis_panel import add_fiscal_response_features, load_fertility_panel

    args = parse_args()
    for path in (
        args.sub_area_panel,
        args.cpi_file,
        args.mapping,
        args.population,
        args.fertility,
        args.structural_subcategory_scores,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    sub_area_panel = pd.read_csv(args.sub_area_panel)
    cpi = pd.read_csv(args.cpi_file, encoding="utf-8-sig")

    fertility_panel, _ = load_fertility_panel(args.fertility, args.mapping, expected_years=YEARS)
    fertility_lagged = add_fiscal_response_features(fertility_panel)
    structural_scores = pd.read_csv(args.structural_subcategory_scores)
    structural_lagged = build_lagged_structural_index(structural_scores)

    results_by_denominator: dict[str, pd.DataFrame] = {}
    for denominator in DENOMINATOR_LABELS:
        regression_sample = build_regression_sample_for_denominator(
            denominator=denominator,
            sub_area_panel=sub_area_panel,
            cpi=cpi,
            population_path=args.population,
            mapping_path=args.mapping,
            fertility_lagged=fertility_lagged,
            structural_lagged=structural_lagged,
        )
        results_by_denominator[denominator] = run_subarea_models(regression_sample)

    comparison = build_comparison_table(results_by_denominator)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "2016-2024_세부영역별_재정반응성_인구분모_민감도비교.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(comparison.round(4).to_string(index=False))
    print(f"저장: {output_path}")
    print(
        f"부호 일치: {comparison['부호_일치'].sum()}/{len(comparison)}개 세부영역, "
        f"5% 유의성 판정 일치: {comparison['유의성_5%_일치'].sum()}/{len(comparison)}개 세부영역"
    )


if __name__ == "__main__":
    main()
