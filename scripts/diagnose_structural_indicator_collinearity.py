"""#62 재정반응성 회귀 전 구조환경 대영역·세부영역 지수 간 다중공선성을 진단한다.

방법론 메모(``reports/20260803_재정대응지수_구조환경지수_회귀설계_방법론_정리.md``
§3.3)의 권장 진단 순서를 그대로 실행한다: pooled Pearson·Spearman, within(지역
평균 제거) 상관, 지역·연도 고정효과 잔차 상관, VIF·조건수. #82/#96의 pooled
구조환경지수 산출물(``data/processed/structural_index/``)을 입력으로 쓰며, 지수
계산 로직 자체는 건드리지 않는다.

이 스크립트는 진단 결과만 만든다 — 어떤 변수를 최종 회귀에 넣을지는 이슈 #62의
미해결 쟁점(팀 결정 필요)이므로 자동으로 판단하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRUCTURAL_INDEX_DIR = REPO_ROOT / "data" / "processed" / "structural_index"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "methodology"
CORRELATION_THRESHOLD = 0.7
VIF_WARNING_THRESHOLD = 5.0
VIF_SEVERE_THRESHOLD = 10.0


def _build_report(
    *,
    level_name: str,
    columns: list[str],
    pooled_pearson: pd.DataFrame,
    pooled_spearman: pd.DataFrame,
    within_pearson: pd.DataFrame,
    fe_residual_pearson: pd.DataFrame,
    vif_table: pd.DataFrame,
    condition_number: float,
    high_corr_pairs: dict[str, pd.DataFrame],
) -> str:
    vif_display = vif_table.copy()
    vif_display["판정"] = vif_display["VIF"].apply(
        lambda value: (
            "심함"
            if value >= VIF_SEVERE_THRESHOLD
            else "주의"
            if value >= VIF_WARNING_THRESHOLD
            else "정상"
        )
    )

    lines = [
        f"## {level_name} 다중공선성 진단",
        "",
        f"- 변수: {', '.join(columns)} ({len(columns)}개)",
        f"- 조건수(상관행렬 고유값 기준): {condition_number:.2f}",
        f"- VIF 판정 기준(방법론 메모 §3.3): 정상 < {VIF_WARNING_THRESHOLD:g} <= 주의 < "
        f"{VIF_SEVERE_THRESHOLD:g} <= 심함 (기계적 탈락 기준이 아니라 참고 신호)",
        "",
        "### VIF",
        "",
        "```",
        vif_display.round(3).to_string(index=False),
        "```",
        "",
        f"### |r| >= {CORRELATION_THRESHOLD} 변수쌍",
        "",
    ]
    for stage, pairs in high_corr_pairs.items():
        if pairs.empty:
            lines.append(f"- {stage}: 없음")
        else:
            lines.append(f"- {stage}: {len(pairs)}쌍")
            lines.append("```")
            lines.append(pairs.round(3).to_string(index=False))
            lines.append("```")
    lines.append("")
    return "\n".join(lines)


def diagnose_level(
    long_scores: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
    level_name: str,
) -> tuple[str, dict[str, pd.DataFrame]]:
    from src.modeling.collinearity import (
        compute_pooled_correlation,
        compute_two_way_fe_residuals,
        compute_vif,
        compute_within_region_correlation,
        flag_high_correlation_pairs,
        pivot_scores_to_wide,
    )

    wide = pivot_scores_to_wide(long_scores, group_col=group_col, value_col=value_col)
    columns = sorted(set(long_scores[group_col]))

    pooled_pearson, pooled_counts = compute_pooled_correlation(
        wide, columns=columns, method="pearson"
    )
    pooled_spearman, _ = compute_pooled_correlation(wide, columns=columns, method="spearman")
    within_pearson = compute_within_region_correlation(wide, columns=columns)
    fe_residuals = compute_two_way_fe_residuals(wide, columns=columns)
    fe_residual_pearson = fe_residuals.corr(method="pearson")
    vif_table, condition_number = compute_vif(wide, columns=columns)

    high_corr_pairs = {
        "pooled Pearson": flag_high_correlation_pairs(
            pooled_pearson, threshold=CORRELATION_THRESHOLD
        ),
        "pooled Spearman": flag_high_correlation_pairs(
            pooled_spearman, threshold=CORRELATION_THRESHOLD
        ),
        "within(지역 평균 제거)": flag_high_correlation_pairs(
            within_pearson, threshold=CORRELATION_THRESHOLD
        ),
        "지역·연도 고정효과 잔차": flag_high_correlation_pairs(
            fe_residual_pearson, threshold=CORRELATION_THRESHOLD
        ),
    }

    report = _build_report(
        level_name=level_name,
        columns=columns,
        pooled_pearson=pooled_pearson,
        pooled_spearman=pooled_spearman,
        within_pearson=within_pearson,
        fe_residual_pearson=fe_residual_pearson,
        vif_table=vif_table,
        condition_number=condition_number,
        high_corr_pairs=high_corr_pairs,
    )

    tables = {
        "pooled_pearson": pooled_pearson,
        "pooled_spearman": pooled_spearman,
        "pooled_observation_counts": pooled_counts,
        "within_region_pearson": within_pearson,
        "fixed_effects_residual_pearson": fe_residual_pearson,
        "vif": vif_table,
    }
    return report, tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structural-index-dir", type=Path, default=DEFAULT_STRUCTURAL_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rename_key = {"region": "지역", "year": "연도"}
    category_scores = pd.read_csv(
        args.structural_index_dir / "structural_index_pooled_category_scores.csv"
    ).rename(columns=rename_key)
    subcategory_scores = pd.read_csv(
        args.structural_index_dir / "structural_index_pooled_subcategory_scores.csv"
    ).rename(columns=rename_key)

    reports = []
    all_tables: dict[str, pd.DataFrame] = {}
    for level_name, long_scores, group_col, value_col in (
        ("대영역(4개)", category_scores, "category", "category_score"),
        ("세부영역(11개)", subcategory_scores, "subcategory", "subcategory_score"),
    ):
        report, tables = diagnose_level(
            long_scores, group_col=group_col, value_col=value_col, level_name=level_name
        )
        reports.append(report)
        prefix = "category" if "대영역" in level_name else "subcategory"
        for name, table in tables.items():
            all_tables[f"{prefix}_{name}"] = table

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "20260807_구조환경지수_다중공선성_진단_QA.md"
    terminology_note = (
        "## 용어 설명\n\n"
        "판넬데이터에는 변동이 두 종류 있다 — 지역들끼리 원래 수준이 다른 **between(지역 간) "
        "변동**과, 같은 지역 안에서 연도가 바뀌며 생기는 **within(지역 내) 변동**이다. "
        "아래 세 단계는 이 중 어디까지 제거했는지가 다르다(제거한 것 기준이 아니라 "
        '남은 변동 기준으로 이름이 붙는다 — "within"은 between을 없애 within만 남겼다는 뜻).\n\n'
        "| 단계 | 지역 효과 제거 | 연도 효과 제거 | 계산 |\n"
        "|---|:---:|:---:|---|\n"
        "| pooled | 안 함 | 안 함 | 원본 값 그대로 |\n"
        "| within(지역 평균 제거) | 함 | 안 함 | 값 − 지역평균 |\n"
        "| 지역·연도 고정효과 잔차 | 함 | 함 | 값 − 지역평균 − 연도평균 + 전체평균"
        "(균형패널에서 지역+연도 더미를 넣은 회귀의 잔차와 수학적으로 동일) |\n\n"
        'pooled에서 within으로 갈 때 상관이 커지면 "지역 수준 차이를 없애니 오히려 상관이 '
        '드러났다"는 뜻이고, within에서 고정효과 잔차로 갈 때 상관이 줄면 "남아있던 전국 공통 '
        '연도 추세 때문에 상관이 부풀려져 있었다"는 뜻이다.\n\n'
    )

    interpretation_note = (
        '## 해석: 가장 끈질기게 남은 상관 — "가사수행 격차" × "사회적 가치관"\n\n'
        "세부영역 진단에서 확인된 |r|>=0.7 쌍 10개는 전부 고정효과 잔차에서 0.7 밑으로 "
        '떨어졌지만(대부분 부호까지 반전), 유일하게 "가사수행 격차"×"사회적 가치관"만 '
        "pooled 0.45 → within 0.49 → 고정효과 잔차 0.53으로 **오히려 커졌다**(세부영역 "
        "전체에서 고정효과 잔차 최댓값). 지표 구성을 `configs/structural_indicators_"
        "verification.yaml`에서 직접 확인한 결과:\n\n"
        '- **가사수행 격차**(2개 지표): "가사노동 공동 참여도"·"돌봄노동 공동 참여도" — '
        "부부가 실제로 얼마나 공평하게 분담하는지를 재는 **행동** 지표\n"
        '- **사회적 가치관**(4개 지표 중 하나): "가사분담에 대한 성평등 인식" — 공평하게 '
        "분담해야 한다는 데 얼마나 동의하는지를 재는 **태도** 지표\n\n"
        "즉 두 세부영역이 부분적으로 같은 주제(가사분담 성평등)를 태도·행동 양쪽에서 재고 "
        "있어서, 공통 연도 추세를 걷어내도 지역·연도별로 같이 움직일 이유가 있는 것으로 "
        "보인다 — 다른 9쌍(예: 산후조리 여건×돌봄여건)이 서로 무관한 주제인데 공통 연도 "
        "추세로만 엮였던 것과는 성격이 다르다.\n\n"
        '**한계**: "가사수행 격차"의 두 지표는 가족실태조사(비정기, 2020·2023년만 실측) '
        '기반이라 나머지 7개 연도는 #70 결측처리 정책으로 채운 값이다. "가사분담에 대한 '
        '성평등 인식"은 별도의 사회조사(2016·18·20·22·24, 5개 시점 실측)라 같은 결측보간을 '
        '공유하지는 않지만, "가사수행 격차" 쪽 시간 변동의 상당 부분이 보간값이라는 점은 '
        "이 상관의 확실성을 다소 낮추는 요인이다.\n\n"
    )

    report_text = (
        "# 구조환경지수 대영역·세부영역 다중공선성 진단 — 2026-08-07\n\n"
        "이슈 #62 재정반응성 회귀에 구조환경지수를 독립변수로 투입하기 전, "
        "방법론 메모(`20260803_..._정리.md` §3.3) 권장 진단 순서를 실제 데이터로 실행한 결과다. "
        "어떤 변수를 최종 회귀에 넣을지는 이 문서가 결정하지 않는다 — 이슈 #62 코멘트의 "
        "미해결 쟁점 2번(대영역별 분리 vs 전체 합산)에 참고자료로 쓴다.\n\n"
        + terminology_note
        + "\n\n".join(reports)
        + "\n\n"
        + interpretation_note
    )
    report_path.write_text(report_text, encoding="utf-8")

    csv_dir = args.output_dir.parent / "20260807_구조환경지수_다중공선성_진단"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for name, table in all_tables.items():
        table.to_csv(csv_dir / f"{name}.csv", encoding="utf-8-sig")

    print(f"보고서 저장: {report_path}")
    print(f"CSV 저장: {csv_dir}")


if __name__ == "__main__":
    main()
