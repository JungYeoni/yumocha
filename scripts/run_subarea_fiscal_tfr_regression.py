"""#98 모형 C(주분석) — 세부영역별 재정대응(F_i,t-1)과 합계출산율(TFR)의 관련성을 추정한다.

2026-08-07 팀 결정(이슈 #98)에 따른 모형 구성:

    합계출산율_i,t = φ·log1p(F_i,t-1) + α_i + λ_t + ε_it   (기본모형)

- 종속변수: 합계출산율(TFR_i,t), 변환 없음
- 설명변수: 세부영역별 인구1인당 실질예산액의 1년 전 값(F_i,t-1)
- α_i, λ_t: 지역·연도 고정효과(더미), 표준오차는 지역 군집(17개)으로 보정
- 세부영역마다 별도 추정, 기본모형은 통제변수 없음(팀 결정)
- pooled + 세부영역 더미×F 상호작용 방식은 채택하지 않음(표본 대비 모수 과다,
  2026-08-07 팀 결정)

F_i,t-1은 모형 A(#62)와 동일한 이유로 log1p 변환한다 — 세부영역별 예산 스케일이
1만 배 이상 차이 나는 우측왜도 때문이며, 팀 결정문이 아니라 이 스크립트의
모형화 판단이다. 계수는 F_i,t-1 1% 변화당 TFR의 수준(절대) 변화로 해석한다.

강건성체크 2종(팀 결정, 기본모형과 분리 제시):
1. F_i,t-2(2년 시차) — 재정멘토가 2년 시차를 권장
2. S_i,t-1(구조환경지수 전년도)을 통제변수로 추가 — 기본모형과 계수·신뢰구간·부호
   변화를 나란히 기록한다. 결과가 불리하거나 부호가 바뀌더라도 임의로 제외하지
   않는다(탐색적 강건성 체크로만 취급)

계산 로직(src/modeling/fiscal_response.py의 fit_two_way_fixed_effects,
summarize_fixed_effects)은 모형 A에서 이미 만들어져 테스트된 유틸을 그대로
재사용한다. controls 옵션만 이번에 추가했다(#98 전용, 기존 호출부는 영향 없음).
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

OUTCOME = "합계출산율"
LAG1_COLUMN = "인구1인당_실질예산_전년도"
LAG2_COLUMN = "인구1인당_실질예산_전전년도"
STRUCTURAL_CONTROL_COLUMN = "구조환경지수_전년도"


def run_subarea_models(
    sample: pd.DataFrame,
    *,
    lag_column: str,
    controls: list[str] | None = None,
    group_col: str = "세부영역",
) -> pd.DataFrame:
    """그룹(기본: 세부영역)마다 별도로 TFR ~ log1p(F_i,t-k) [+ 통제변수] 고정효과 회귀를 추정한다.

    group_col은 세부영역 대신 대영역 등 다른 그룹 단위로 동일한 회귀를 돌릴 때
    쓴다(예: scripts/run_major_category_fiscal_tfr_regression.py).
    """
    from src.modeling.fiscal_response import fit_two_way_fixed_effects, summarize_fixed_effects

    predictor = f"log1p_{lag_column}"
    dropna_columns = [lag_column, *(controls or [])]

    summaries: list[dict[str, object]] = []
    for group_label, group in sample.groupby(group_col, sort=False):
        usable = group.dropna(subset=dropna_columns).copy()
        usable[predictor] = np.log1p(usable[lag_column])

        model, fitted_sample = fit_two_way_fixed_effects(
            usable,
            outcome=OUTCOME,
            predictor=predictor,
            controls=controls,
        )
        summaries.append(
            summarize_fixed_effects(
                model,
                fitted_sample,
                model_name=group_label,
                outcome=OUTCOME,
                predictor=predictor,
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

    variants = {
        "기본모형(F_t-1)": run_subarea_models(sample, lag_column=LAG1_COLUMN),
        "강건성체크_2년시차(F_t-2)": run_subarea_models(sample, lag_column=LAG2_COLUMN),
        "강건성체크_S통제(F_t-1+S_t-1)": run_subarea_models(
            sample, lag_column=LAG1_COLUMN, controls=[STRUCTURAL_CONTROL_COLUMN]
        ),
    }

    results = []
    for label, table in variants.items():
        table = table.copy()
        table.insert(0, "모형버전", label)
        results.append(table)
    combined = pd.concat(results, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "2016-2024_세부영역별_재정_TFR_모형C_고정효과_결과.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    display_columns = [
        "모형버전",
        "모형",
        "계수",
        "군집표준오차",
        "p값",
        "관측치",
        "지역수",
        "연도수",
    ]
    print(combined[display_columns].round(4).to_string(index=False))
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
