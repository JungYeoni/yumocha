"""#62 재정대응지수 분석의 재정 기초패널을 #81 최종 산출물 기준으로 재생성한다.

기존 산출물(``2016-2024_시도별_계획예산_합계출산율_기초패널.csv``)은 #52
노트북(``20260726_EDA_..._생성.ipynb``)의 ``load_current_budget_panel()``로
만들어졌는데, 이 함수가 읽는 ``data/interim/{지역}/{연도}_{지역}_세부사업_정제_long.csv``
안의 소계 행("소 계" 등)을 걸러내지 않아 전남 2017년 예산이 195,657백만원(14%)
과다계상돼 있었다(전남 2017 원본행 9505 "소 계" 행과 정확히 일치 — #73 감사 당시
이미 발견된 행). #81의 ``provisional_budget_panel.csv``는 재정팀 검토 라벨 기반으로
소계 행을 명시적으로 leaf 추출에서 제외하므로(``src/provisional/loader.py``의
``_classify_rows``), 이걸 예산 원천으로 쓴다.

합계출산율·QA 교차검증·원자료 누락 주석은 #52 노트북과 동일한 기존 유틸
(``load_fertility_panel``, ``load_budget_qa_panel``, ``build_budget_fertility_panel``)을
그대로 재사용한다 — 이 로직은 예산 원천과 무관하게 독립적으로 유효하다.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET_PANEL = (
    REPO_ROOT / "data" / "interim" / "provisional" / "provisional_budget_panel.csv"
)
DEFAULT_MAPPING = REPO_ROOT / "data" / "lookup" / "시도_지역코드_매핑.csv"
DEFAULT_FERTILITY = (
    REPO_ROOT / "data" / "raw" / "출산동향" / "2016-2025_시도별_출생아수_합계출산율_20260703.csv"
)
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis"

YEARS = list(range(2016, 2025))

# #52 노트북(셀 13)의 원자료 누락 주석을 그대로 승계한다 — #81 예산 원천을 바꿔도
# 원본 PDF 자체의 세부내역 누락(강원 2018, 전남 2018)은 여전히 유효한 사실이다.
QUALITY_NOTES = pd.DataFrame(
    [
        {
            "지역": "강원",
            "연도": 2018,
            "원자료_누락주의": "원자료에 자체사업 상세가 없어 세부사업 합계가 과소대표될 수 있음",
        },
        {
            "지역": "전남",
            "연도": 2018,
            "원자료_누락주의": (
                "원자료에 자체사업 일부 세부내역이 없어 세부사업 합계가 과소대표될 수 있음"
            ),
        },
    ]
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-panel", type=Path, default=DEFAULT_BUDGET_PANEL)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--fertility", type=Path, default=DEFAULT_FERTILITY)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from src.features.analysis_panel import (
        add_fiscal_response_features,
        build_budget_fertility_panel,
        load_budget_qa_panel,
        load_fertility_panel,
    )

    args = parse_args(argv)
    if not args.budget_panel.is_file():
        raise FileNotFoundError(f"#81 예산 패널이 없습니다: {args.budget_panel}")

    region_mapping = pd.read_csv(args.mapping)
    regions = region_mapping["지역"].tolist()

    budget_panel = pd.read_csv(args.budget_panel)

    fertility_panel, nationwide_fertility = load_fertility_panel(
        args.fertility, args.mapping, expected_years=YEARS
    )
    qa_panel = load_budget_qa_panel(args.reports_dir, expected_years=YEARS)

    base_panel = build_budget_fertility_panel(
        budget_panel,
        fertility_panel,
        expected_regions=regions,
        expected_years=YEARS,
        qa_panel=qa_panel,
        quality_notes=QUALITY_NOTES,
    )

    if len(base_panel) != 17 * 9:
        raise ValueError(f"기초패널 행 수 불일치: {len(base_panel)}")
    if base_panel[["지역", "연도"]].duplicated().any():
        raise ValueError("기초패널 지역·연도 키 중복")
    if base_panel["합계출산율"].isna().any():
        raise ValueError("합계출산율 결측 존재")
    if base_panel["당해계획예산_백만원"].isna().any():
        raise ValueError("당해계획예산 결측 존재")

    panel_qa_summary = pd.DataFrame(
        {
            "검증항목": [
                "최종 행 수",
                "지역 수",
                "연도 수",
                "지역연도 중복",
                "계획예산 합계 결측",
                "합계출산율 결측",
                "예산 결측 세부사업",
                "음수 예산 세부사업",
                "원자료 누락주의 지역연도",
            ],
            "값": [
                len(base_panel),
                base_panel["지역"].nunique(),
                base_panel["연도"].nunique(),
                int(base_panel[["지역", "연도"]].duplicated().sum()),
                int(base_panel["당해계획예산_백만원"].isna().sum()),
                int(base_panel["합계출산율"].isna().sum()),
                int(base_panel["예산결측_사업수"].sum()),
                int(base_panel["음수예산_사업수"].sum()),
                int(base_panel["원자료_누락주의"].notna().sum()),
            ],
        }
    )

    fiscal_response_panel = add_fiscal_response_features(base_panel)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_output = args.output_dir / "2016-2024_시도별_계획예산_합계출산율_기초패널.csv"
    response_output = args.output_dir / "2016-2024_재정반응성_기초분석패널.csv"
    nationwide_output = args.output_dir / "2016-2024_전국_합계출산율_추세.csv"
    qa_output = args.output_dir / "2016-2024_기초패널_QA_요약.csv"

    base_panel.to_csv(base_output, index=False, encoding="utf-8-sig")
    fiscal_response_panel.to_csv(response_output, index=False, encoding="utf-8-sig")
    nationwide_fertility.to_csv(nationwide_output, index=False, encoding="utf-8-sig")
    panel_qa_summary.to_csv(qa_output, index=False, encoding="utf-8-sig")

    print(panel_qa_summary.to_string(index=False))
    print(f"저장: {base_output}")
    print(f"저장: {response_output}")
    print(f"저장: {nationwide_output}")
    print(f"저장: {qa_output}")


if __name__ == "__main__":
    main()
