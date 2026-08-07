"""#62 모형 A(재정반응성) — 세부영역별 지역·연도 고정효과 회귀를 추정한다.

2026-08-07 팀 결정: 주 분석은 C(#98, TFR ~ F_i,t-1)이고, A는 보완 분석이다.

식(방법론 메모 §3.1-A, 기본모형 — 통제변수 없음):

    log1p(F_it) = φ·ΔTFR_i,t-1 + α_i + λ_t + ε_it

- F_it: 세부영역별 인구1인당 실질예산액(2016-2024_세부영역별_재정반응성_회귀표본.csv)
- ΔTFR_i,t-1: 직전1년 출산율 하락도(전전년도 - 전년도 합계출산율)
- α_i, λ_t: 지역·연도 고정효과(더미), 지역 군집표준오차

F_it 자체는 오른쪽으로 크게 치우쳐 있어(세부영역별 평균이 22원~35만원까지
1만 배 차이) log1p로 변환한 뒤 회귀에 넣는다 — 팀 결정문("예산액이 쓰인다")은
수준(level) 대 지수(z-score)를 구분한 것이지 로그변환 여부를 명시하지 않았으므로,
이 스크립트가 내린 모형화 판단임을 결과 보고서에 남긴다.

계산 로직(src/modeling/fiscal_response.py의 fit_two_way_fixed_effects,
summarize_fixed_effects)은 이전 WIP 커밋에서 이미 만들어져 테스트된 유틸을
그대로 재사용하고 수정하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGRESSION_SAMPLE = (
    REPO_ROOT / "data" / "processed" / "analysis" / "2016-2024_세부영역별_재정반응성_회귀표본.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis"
PREDICTOR = "직전1년_출산율하락도"
OUTCOME = "log1p_인구1인당_실질예산"


def run_subarea_models(sample: pd.DataFrame) -> pd.DataFrame:
    """세부영역마다 별도로 지역·연도 고정효과 회귀를 추정한다."""
    from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects

    summaries: list[dict[str, object]] = []
    for subarea, group in sample.groupby("세부영역", sort=False):
        usable = group.dropna(subset=[PREDICTOR]).copy()
        usable[OUTCOME] = np.log1p(usable["인구1인당_실질예산_원"])

        model, fitted_sample = fit_two_way_fixed_effects(
            usable,
            outcome=OUTCOME,
            predictor=PREDICTOR,
        )
        summaries.append(
            summarize_fixed_effects(
                model,
                fitted_sample,
                model_name=subarea,
                outcome=OUTCOME,
                predictor=PREDICTOR,
                excluded_quality_rows=False,
            )
        )
    return pd.DataFrame(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-sample", type=Path, default=DEFAULT_REGRESSION_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.regression_sample.is_file():
        raise FileNotFoundError(f"회귀표본이 없습니다: {args.regression_sample}")

    sample = pd.read_csv(args.regression_sample)
    results = run_subarea_models(sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "2016-2024_세부영역별_재정반응성_고정효과_결과.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_columns = ["모형", "계수", "군집표준오차", "p값", "관측치", "지역수", "연도수"]
    print(results[display_columns].round(4).to_string(index=False))
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
