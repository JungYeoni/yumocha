"""이슈 #116 핵심어 표·워드클라우드 산출물의 자동 완료 기준을 검증한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features.keyword_tfidf import DEFAULT_STOPWORDS

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis" / "keyword_tfidf"
FIGURE_DIR = REPO_ROOT / "reports" / "figures" / "keyword_tfidf"
EXPECTED_GROUPS = 12
MIN_TERMS_PER_GROUP = 15


@dataclass(frozen=True)
class Check:
    """단일 자동검증 결과다."""

    name: str
    passed: bool
    detail: str


def run_checks() -> list[Check]:
    """핵심어 CSV와 시각화의 완전성·상투어 제외 기준을 검사한다."""
    checks: list[Check] = []
    for field in ("세부사업명", "주요내용"):
        ranking_path = OUTPUT_DIR / f"2016-2024_세부영역별_{field}_TFIDF_상위단어.csv"
        sensitivity_path = OUTPUT_DIR / f"2016-2024_세부영역별_{field}_TFIDF_중복민감도.csv"
        ranking = pd.read_csv(ranking_path)
        sensitivity = pd.read_csv(sensitivity_path)
        counts = ranking.groupby("세부영역").size()
        minimum_terms = int(counts.min()) if not counts.empty else 0
        generic = sorted(set(ranking["단어"]) & set(DEFAULT_STOPWORDS))
        checks.extend(
            [
                Check(
                    f"{field}:12개 영역",
                    ranking["세부영역"].nunique() == EXPECTED_GROUPS,
                    f"실제 {ranking['세부영역'].nunique()}개",
                ),
                Check(
                    f"{field}:영역당 상위어",
                    len(counts) == EXPECTED_GROUPS and counts.ge(MIN_TERMS_PER_GROUP).all(),
                    f"최소 {minimum_terms}개",
                ),
                Check(f"{field}:불용어 제외", not generic, f"잔존 {generic}"),
                Check(
                    f"{field}:중복 민감도",
                    len(sensitivity) == EXPECTED_GROUPS
                    and sensitivity["상위어_중첩률"].between(0, 1).all(),
                    f"영역 {len(sensitivity)}개",
                ),
            ]
        )
        for kind in ("워드클라우드", "상위단어_막대그래프"):
            figure_path = FIGURE_DIR / f"2016-2024_세부영역별_{field}_{kind}.png"
            checks.append(
                Check(
                    f"{field}:{kind}",
                    figure_path.is_file() and figure_path.stat().st_size > 10_000,
                    str(figure_path.relative_to(REPO_ROOT)),
                )
            )
    return checks


def main() -> int:
    """검증 결과를 출력하고 실패가 있으면 종료 코드 1을 반환한다."""
    checks = run_checks()
    for check in checks:
        print(f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    failed = [check for check in checks if not check.passed]
    print(f"검증 결과: {len(checks) - len(failed)}/{len(checks)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
