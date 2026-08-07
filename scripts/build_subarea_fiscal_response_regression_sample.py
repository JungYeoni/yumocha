"""#62/#98 회귀분석용 표본 — 세부영역별 재정대응(F_it)에 시차 TFR·구조환경(S_i,t-1)을 결합한다.

2026-08-07 팀 결정(이슈 #62 코멘트)에 따른 변수 구성:
- F_it: 세부영역별 인구1인당 실질예산액(z-score 아님, 원본값)
- ΔTFR_i,t-1: 직전1년 출산율 하락도(전전년도 - 전년도 합계출산율,
  기존 add_fiscal_response_features 재사용)
- S_i,t-1: 같은 세부영역의 전년도 구조환경지수(#82/#96 pooled 산출물, 1년 시차)
- F_i,t-1, F_i,t-2: 같은 세부영역의 1년/2년 전 인구1인당 실질예산액
  (#98 모형 C 기본 설명변수 및 2년 시차 강건성체크용)

파이프라인 코드(src/features/analysis_panel.py)는 호출만 하고 수정하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FISCAL_RESPONSE = (
    REPO_ROOT / "data" / "processed" / "analysis" / "2016-2024_세부영역별_인구1인당_실질예산액.csv"
)
DEFAULT_FERTILITY_PATH = (
    REPO_ROOT / "data" / "raw" / "출산동향" / "2016-2025_시도별_출생아수_합계출산율_20260703.csv"
)
DEFAULT_MAPPING = REPO_ROOT / "data" / "lookup" / "시도_지역코드_매핑.csv"
DEFAULT_STRUCTURAL_SUBCATEGORY = (
    REPO_ROOT
    / "data"
    / "processed"
    / "structural_index"
    / "structural_index_pooled_subcategory_scores.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis"

YEARS = list(range(2016, 2025))

# 구조환경지수(#82/#96)의 subcategory 표기는 재정 라벨 taxonomy(#81, "N-M. " 접두어)와
# 다르다 — 번호 접두어가 없고, 2개는 띄어쓰기도 다르다("돌봄여건" vs "돌봄 여건",
# "주거안정성" vs "주거 안정성"). 자동 접두어 제거만으로는 이 두 건을 못 잡으므로
# 11개 전체를 명시적으로 매핑한다(재정 세부영역 -> 구조환경 subcategory).
FISCAL_TO_STRUCTURAL_SUBCATEGORY = {
    "1-1. 고용여건": "고용여건",
    "1-2. 주거안정성": "주거 안정성",
    "1-3. 경제적 여건": "경제적 여건",
    "2-1. 돌봄 여건": "돌봄여건",
    "2-2. 여가 인프라": "여가 인프라",
    "2-3. 가사수행 격차": "가사수행 격차",
    "3-1. 의료서비스 여건": "의료서비스 여건",
    "3-2. 산후조리 여건": "산후조리 여건",
    "3-3. 아동안전 수준": "아동안전 수준",
    "4-1. 일·가정 양립 여건": "일·가정 양립 여건",
    "4-2. 사회적 가치관": "사회적 가치관",
}


def build_lagged_structural_index(subcategory_scores: pd.DataFrame) -> pd.DataFrame:
    """세부영역별 구조환경지수를 1년 시차(S_i,t-1)로 만든다."""
    observed = set(subcategory_scores["subcategory"])
    expected = set(FISCAL_TO_STRUCTURAL_SUBCATEGORY.values())
    if observed != expected:
        raise ValueError(
            f"구조환경지수 subcategory 표기가 매핑표와 다릅니다: "
            f"구조환경에만={sorted(observed - expected)}, 매핑표에만={sorted(expected - observed)}"
        )
    structural_to_fiscal = {value: key for key, value in FISCAL_TO_STRUCTURAL_SUBCATEGORY.items()}

    panel = subcategory_scores.rename(columns={"region": "지역", "year": "연도"}).copy()
    panel["세부영역"] = panel["subcategory"].map(structural_to_fiscal)
    panel = panel[["지역", "연도", "세부영역", "subcategory_score"]]

    if panel.duplicated(["지역", "연도", "세부영역"]).any():
        raise ValueError("구조환경지수 패널 지역·연도·세부영역 키 중복")

    # shift(1)은 연도가 연속이라는 전제로 동작한다 — 중간 연도가 빠지면 2년 이상
    # 전 값을 "전년도"로 잘못 참조하므로, 시프트 전에 그룹별 연도 연속성을 검증한다.
    # (절대 범위가 YEARS와 같아야 하는 건 아니고, 그룹 내부에 빈 연도만 없으면 된다.)
    def _has_gap(values: pd.Series) -> bool:
        return set(values) != set(range(int(values.min()), int(values.max()) + 1))

    incomplete = panel.groupby(["지역", "세부영역"])["연도"].apply(_has_gap)
    if incomplete.any():
        missing_groups = incomplete.loc[incomplete].index.tolist()
        raise ValueError(f"구조환경지수 패널 연도가 연속하지 않는 지역·세부영역: {missing_groups}")

    panel = panel.sort_values(["지역", "세부영역", "연도"])
    panel["구조환경지수_전년도"] = panel.groupby(["지역", "세부영역"], sort=False)[
        "subcategory_score"
    ].shift(1)
    return panel[["지역", "연도", "세부영역", "구조환경지수_전년도"]]


def build_lagged_fiscal_response(fiscal_response: pd.DataFrame) -> pd.DataFrame:
    """세부영역별 인구1인당 실질예산액을 1년/2년 시차(F_i,t-1, F_i,t-2)로 만든다.

    #98 모형 C(TFR_i,t ~ F_i,t-1)의 설명변수 및 F_i,t-2 강건성체크용.
    """
    panel = fiscal_response.loc[fiscal_response["세부영역"].ne("지표체계 외")].copy()
    panel = panel[["지역", "연도", "세부영역", "인구1인당_실질예산_원"]]

    if panel.duplicated(["지역", "연도", "세부영역"]).any():
        raise ValueError("재정반응 패널 지역·연도·세부영역 키 중복")

    # build_lagged_structural_index와 동일한 이유로 shift 전에 연도 연속성을 검증한다.
    def _has_gap(values: pd.Series) -> bool:
        return set(values) != set(range(int(values.min()), int(values.max()) + 1))

    incomplete = panel.groupby(["지역", "세부영역"])["연도"].apply(_has_gap)
    if incomplete.any():
        missing_groups = incomplete.loc[incomplete].index.tolist()
        raise ValueError(f"재정반응 패널 연도가 연속하지 않는 지역·세부영역: {missing_groups}")

    panel = panel.sort_values(["지역", "세부영역", "연도"])
    grouped = panel.groupby(["지역", "세부영역"], sort=False)["인구1인당_실질예산_원"]
    panel["인구1인당_실질예산_전년도"] = grouped.shift(1)
    panel["인구1인당_실질예산_전전년도"] = grouped.shift(2)
    return panel[
        ["지역", "연도", "세부영역", "인구1인당_실질예산_전년도", "인구1인당_실질예산_전전년도"]
    ]


def build_regression_sample(
    fiscal_response: pd.DataFrame,
    fertility_lagged: pd.DataFrame,
    structural_lagged: pd.DataFrame,
    fiscal_lagged: pd.DataFrame,
) -> pd.DataFrame:
    """F_it·ΔTFR_i,t-1·S_i,t-1·F_i,t-1·F_i,t-2를 지역·연도·세부영역 키로 결합한다."""
    fiscal = fiscal_response.loc[fiscal_response["세부영역"].ne("지표체계 외")].copy()

    merged = (
        fiscal.merge(
            fertility_lagged[["지역", "연도", "직전1년_출산율하락도", "합계출산율"]],
            on=["지역", "연도"],
            how="left",
            validate="many_to_one",
        )
        .merge(
            structural_lagged,
            on=["지역", "연도", "세부영역"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            fiscal_lagged,
            on=["지역", "연도", "세부영역"],
            how="left",
            validate="one_to_one",
        )
    )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fiscal-response", type=Path, default=DEFAULT_FISCAL_RESPONSE)
    parser.add_argument("--fertility", type=Path, default=DEFAULT_FERTILITY_PATH)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument(
        "--structural-subcategory-scores", type=Path, default=DEFAULT_STRUCTURAL_SUBCATEGORY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    from src.features.analysis_panel import add_fiscal_response_features, load_fertility_panel

    args = parse_args()
    for path in (
        args.fiscal_response,
        args.fertility,
        args.mapping,
        args.structural_subcategory_scores,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    fiscal_response = pd.read_csv(args.fiscal_response)
    fertility_panel, _ = load_fertility_panel(args.fertility, args.mapping, expected_years=YEARS)
    fertility_lagged = add_fiscal_response_features(fertility_panel)

    structural_scores = pd.read_csv(args.structural_subcategory_scores)
    structural_lagged = build_lagged_structural_index(structural_scores)
    fiscal_lagged = build_lagged_fiscal_response(fiscal_response)

    result = build_regression_sample(
        fiscal_response, fertility_lagged, structural_lagged, fiscal_lagged
    )

    expected_rows = 17 * len(YEARS) * 11
    if len(result) != expected_rows:
        raise ValueError(f"결과 행 수 불일치: 기대={expected_rows}, 실제={len(result)}")
    if result[["지역", "연도", "세부영역"]].duplicated().any():
        raise ValueError("회귀표본 지역·연도·세부영역 키 중복")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "2016-2024_세부영역별_재정반응성_회귀표본.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")

    usable = result.dropna(subset=["직전1년_출산율하락도", "구조환경지수_전년도"])
    usable_model_c = result.dropna(subset=["인구1인당_실질예산_전년도"])
    print(
        f"저장: {output_path} ({len(result)}행, "
        f"모형A 시차변수(ΔTFR+S) 완비 {len(usable)}행, "
        f"모형C 시차변수(F_i,t-1) 완비 {len(usable_model_c)}행)"
    )
    print(
        result[
            [
                "직전1년_출산율하락도",
                "구조환경지수_전년도",
                "인구1인당_실질예산_원",
                "인구1인당_실질예산_전년도",
                "인구1인당_실질예산_전전년도",
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
