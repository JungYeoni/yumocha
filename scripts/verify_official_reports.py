"""Verify data-backed claims in the issue #111 official synthesis reports."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = REPO_ROOT / "reports" / "official"


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _record(checks: list[Check], name: str, condition: bool, detail: str) -> None:
    """검사 결과 하나를 표준 형식으로 추가한다."""
    checks.append(Check(name=name, passed=bool(condition), detail=detail))


def _read_csv(relative_path: str) -> pd.DataFrame:
    """저장소 기준 상대 경로의 CSV를 읽는다."""
    return pd.read_csv(REPO_ROOT / relative_path)


def _read_text(relative_path: str) -> str:
    """저장소 기준 상대 경로의 UTF-8 문서를 읽는다."""
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _contains_all(text: str, snippets: tuple[str, ...]) -> bool:
    """문서가 기대 문자열을 모두 포함하는지 확인한다."""
    return all(snippet in text for snippet in snippets)


def _row_by_label(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    """라벨과 정확히 일치하는 단일 행을 반환한다."""
    rows = frame.loc[frame[column].eq(label)]
    if len(rows) != 1:
        raise ValueError(f"{column}={label!r} 행이 정확히 1개가 아닙니다: {len(rows)}")
    return rows.iloc[0]


def _check_markdown_links(checks: list[Check]) -> None:
    """공식 문서의 저장소 내부 링크가 실제 파일을 가리키는지 검사한다."""
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for path in sorted(OFFICIAL_DIR.glob("*.md")):
        missing = []
        for target in link_pattern.findall(path.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            if not (path.parent / unquote(target)).resolve().exists():
                missing.append(target)
        _record(
            checks,
            f"링크:{path.name}",
            not missing,
            "누락 0건" if not missing else f"누락: {missing}",
        )


def _check_structural_report(checks: list[Check]) -> None:
    """구조환경 기준 산출물과 공식 문서 본문의 핵심 주장을 함께 대조한다."""
    official = _read_text("reports/official/구조환경지표_구조환경지수_공식_종합.md")
    panel_qa = _read_text("reports/20260804_구조환경지표_28개_보간전_통합패널_QA.md")
    _record(
        checks,
        "구조환경 패널 행",
        "| 최종 패널 행 수 | 4,536 | 4,536 | PASS |" in panel_qa and "**4,536행**" in official,
        "추적 QA에서 4,536행 확인",
    )
    _record(
        checks,
        "구조환경 패널 차원",
        "4,536행, 28개 지표, 18개 지역, 2016–2024년" in panel_qa
        and "18개 지역(전국+17개 시도) × 9개 연도 × 28개 지표" in official,
        "추적 QA에서 지표 28, 지역 18, 연도 9 확인",
    )

    index_qa = _read_text("reports/methodology/20260806_구조환경지수_실제패널_산출_QA.md")
    _record(
        checks,
        "17개 시도 처리값 완비",
        _contains_all(
            index_qa,
            ("17개 시도 × 2016–2024년 × 28개 지표 = 4,284행", "최종지수 153개"),
        )
        and "**4,284행**" in official,
        "추적 QA에서 4,284행과 최종지수 153개 확인",
    )

    missing_qa = _read_text("reports/methodology/20260805_구조환경지표_결측처리_실제적용_QA.md")
    expected_strategies = (
        "| hold first observed | 586 |",
        "| hold last observed | 90 |",
        "| linear interpolation | 486 |",
        "| none | 3,374 |",
    )
    _record(
        checks,
        "결측 처리 전략 건수",
        _contains_all(missing_qa, expected_strategies)
        and _contains_all(
            official,
            (
                "| 선형보간 | 486 |",
                "| 최초 관측값 과거 유지 | 586 |",
                "| 최종 관측값 이후 유지 | 90 |",
            ),
        ),
        "추적 QA에서 none 3,374, 최초 586, 선형 486, 최종 90 확인",
    )

    family_qa = _read_text(
        "reports/methodology/20260806_가족친화_2017_2019_raking_본계열반영_QA.md"
    )
    _record(
        checks,
        "가족친화 2017·2019 추정 상태",
        "이 34건(2017·2019)의 raking 추정치를 **본계열에 반영**" in family_qa
        and "34개 추정값은 `관측상태=추정`으로 보존" in official,
        "추적 QA에서 2017·2019 합계 34건 확인",
    )

    weights = yaml.safe_load(
        (REPO_ROOT / "configs/structural_index_weights.yaml").read_text(encoding="utf-8")
    )["weights"]
    weight_sum = sum(float(value["weight"]) for value in weights.values())
    _record(
        checks,
        "AHP 가중치",
        len(weights) == 28
        and abs(weight_sum - 1.0) < 1e-12
        and "28개 합이 1이 되도록 정규화" in official,
        f"지표 {len(weights)}, 합 {weight_sum:.15f}",
    )

    expected_rows = {
        "pooled": "| pooled | 4,284 | 153 | 0 | 28.7401–69.8601 |",
        "yearly": "| yearly | 4,284 | 153 | 0 | 29.6371–65.1633 |",
    }
    for method, expected_row in expected_rows.items():
        _record(
            checks,
            f"구조환경 {method} 최종지수",
            expected_row in index_qa and expected_row.split(" | ")[-2] in official,
            expected_row.strip("| "),
        )


def _check_fiscal_report(checks: list[Check]) -> None:
    """재정 분석 산출물과 공식 문서 본문의 핵심 주장을 함께 대조한다."""
    official = _read_text("reports/official/재정대응지수_재정반응성_공식_종합.md")
    regression = _read_csv("data/processed/analysis/2016-2024_세부영역별_재정반응성_회귀표본.csv")
    fiscal_qa = _read_text(
        "reports/methodology/20260807_세부영역별_재정대응지수_구축_및_재정반응성_모형A_결과보고서.md"
    )
    _record(
        checks,
        "세부영역 예산 패널",
        "1,836행 = 17×9×12" in fiscal_qa and "1,836행(17×9×12)" in official,
        "추적 QA에서 1,836행 확인",
    )
    _record(
        checks,
        "11개 영역 회귀 패널",
        len(regression) == 1_683 and "1,683행(17×9×11)" in official,
        f"실제 {len(regression):,}행",
    )

    model_a = _read_csv("data/processed/analysis/2016-2024_세부영역별_재정반응성_고정효과_결과.csv")
    care_a = _row_by_label(model_a, "모형", "2-1. 돌봄 여건")
    _record(
        checks,
        "모형 A 돌봄 경계 결과",
        len(model_a) == 11
        and int(care_a["관측치"]) == 119
        and abs(float(care_a["계수"]) - 1.588) < 1e-3
        and abs(float(care_a["p값"]) - 0.051) < 1e-3
        and "`p=0.051`" in official,
        f"계수 {care_a['계수']:.4f}, p={care_a['p값']:.4f}, n={int(care_a['관측치'])}",
    )

    model_c_all = _read_csv(
        "data/processed/analysis/2016-2024_세부영역별_재정_TFR_모형C_고정효과_결과.csv"
    )
    model_c = model_c_all.loc[model_c_all["모형버전"].eq("기본모형(F_t-1)")]
    ci_contains_zero = (model_c["95%신뢰구간_하한"] <= 0) & (model_c["95%신뢰구간_상한"] >= 0)
    _record(
        checks,
        "모형 C 기본결과",
        len(model_c) == 11
        and set(model_c["관측치"]) == {136}
        and ci_contains_zero.all()
        and "11개 영역 모두 95% 신뢰구간이 0을 포함" in official,
        f"영역 {len(model_c)}, 0 포함 신뢰구간 {int(ci_contains_zero.sum())}/11",
    )

    moving = _read_csv(
        "data/processed/analysis/2016-2024_세부영역별_3개년평균예산_TFR_고정효과_결과.csv"
    )
    moving_t1 = moving.loc[moving["모형버전"].eq("3개년평균_t+1")]
    moving_t2 = moving.loc[moving["모형버전"].eq("3개년평균_t+2")]
    care_t2 = _row_by_label(moving_t2, "모형", "2-1. 돌봄 여건")
    significant_t1 = int(moving_t1["FDR_0.05_유의"].sum())
    significant_t2 = moving_t2.loc[moving_t2["FDR_0.05_유의"], "모형"].tolist()
    _record(
        checks,
        "3개년 평균 t+1 다중검정",
        len(moving_t1) == 11 and significant_t1 == 0 and set(moving_t1["관측치"]) == {102},
        f"유의 {significant_t1}개, 영역당 n=102",
    )
    _record(
        checks,
        "3개년 평균 t+2 돌봄 결과",
        len(moving_t2) == 11
        and significant_t2 == ["2-1. 돌봄 여건"]
        and int(care_t2["관측치"]) == 85
        and abs(float(care_t2["계수"]) - 0.0454) < 5e-5
        and abs(float(care_t2["FDR_q값"]) - 0.000103) < 5e-7
        and _contains_all(official, ("`β=0.0454`", "`q=0.000103`")),
        f"계수 {care_t2['계수']:.4f}, q={care_t2['FDR_q값']:.6f}, n={int(care_t2['관측치'])}",
    )

    structural_response = _read_csv(
        "data/processed/analysis/2016-2024_세부영역별_구조환경_재정대응_공통반응계수.csv"
    )
    _record(
        checks,
        "구조환경 수준→t+2 예산",
        len(structural_response) == 11
        and not structural_response["FDR_0.05_유의"].any()
        and set(structural_response["관측치"]) == {119},
        f"BH 유의 {int(structural_response['FDR_0.05_유의'].sum())}개",
    )

    change_response = _read_csv(
        "data/processed/analysis/2016-2024_세부영역별_구조환경변화_예산비중대응_결과.csv"
    )
    _record(
        checks,
        "구조환경 변화→예산비중 변화",
        len(change_response) == 11
        and not change_response["FDR_0.05_유의"].any()
        and set(change_response["관측치"]) == {102},
        f"BH 유의 {int(change_response['FDR_0.05_유의'].sum())}개",
    )

    decline = _read_csv(
        "data/processed/analysis/2016-2024_세부영역별_구조환경하락_후행예산비중증가_요약.csv"
    )
    declines = int(decline["구조환경_하락사례수"].sum())
    increases = int(decline["후행예산비중_증가건수"].sum())
    rate = increases / declines * 100
    _record(
        checks,
        "구조환경 하락 대응 방향",
        declines == 295
        and increases == 147
        and abs(rate - 49.8) < 0.05
        and "295건 중 후행 예산비중 증가가 147건(49.8%)" in official,
        f"{increases}/{declines}={rate:.1f}%",
    )

    shares = _read_csv("data/processed/analysis/2016-2024_세부영역별_계획예산비중_변화요약.csv")
    economy = _row_by_label(shares, "세부영역", "1-3. 경제적 여건")
    care_share = _row_by_label(shares, "세부영역", "2-1. 돌봄 여건")
    _record(
        checks,
        "예산 구성비 최대 증감",
        abs(float(economy["2016→2024_비중변화_pp"]) - 13.6367) < 5e-4
        and abs(float(care_share["2016→2024_비중변화_pp"]) + 14.28) < 5e-3,
        f"경제 {economy['2016→2024_비중변화_pp']:+.4f}%p, 돌봄 {care_share['2016→2024_비중변화_pp']:+.4f}%p",
    )

    clusters = _read_csv("data/processed/analysis/2016-2024_구조환경군집_예산상호작용_TFR_결과.csv")
    cluster_t2 = clusters.loc[clusters["시차"].eq("t+2")]
    significant_cluster = cluster_t2.loc[cluster_t2["FDR_0.05_유의"]]
    gap = _row_by_label(cluster_t2, "세부영역", "2-3. 가사수행 격차")
    _record(
        checks,
        "군집 간 t+2 탐색 신호",
        significant_cluster["세부영역"].tolist() == ["2-3. 가사수행 격차"]
        and abs(float(gap["군집간_계수차"]) - 0.0381) < 5e-5
        and abs(float(gap["FDR_q값"]) - 0.0091) < 5e-5,
        f"차이 {gap['군집간_계수차']:.4f}, q={gap['FDR_q값']:.4f}",
    )


def run_checks() -> list[Check]:
    """공식 종합 문서 전체 검사를 실행한다."""
    checks: list[Check] = []
    _check_markdown_links(checks)
    _check_structural_report(checks)
    _check_fiscal_report(checks)
    return checks


def main() -> int:
    """검사 결과를 출력하고 실패 여부를 종료 코드로 반환한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="성공 상세 출력을 생략합니다.")
    args = parser.parse_args()

    checks = run_checks()
    failed = [check for check in checks if not check.passed]
    if not args.quiet:
        for check in checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
    print(f"검증 결과: {len(checks) - len(failed)}/{len(checks)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
