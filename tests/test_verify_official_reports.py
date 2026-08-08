from scripts.verify_official_reports import run_checks


EXPECTED_CHECK_NAMES = {
    "링크:20260808_공식_종합_자동검증_결과.md",
    "링크:20260808_사람검토_체크리스트.md",
    "링크:20260808_주제분류_및_대상문서_확정.md",
    "링크:구조환경지표_구조환경지수_공식_종합.md",
    "링크:재정대응지수_재정반응성_공식_종합.md",
    "구조환경 패널 행",
    "구조환경 패널 차원",
    "17개 시도 처리값 완비",
    "결측 처리 전략 건수",
    "가족친화 2017·2019 추정 상태",
    "AHP 가중치",
    "구조환경 pooled 최종지수",
    "구조환경 yearly 최종지수",
    "세부영역 예산 패널",
    "11개 영역 회귀 패널",
    "모형 A 돌봄 경계 결과",
    "모형 C 기본결과",
    "3개년 평균 t+1 다중검정",
    "3개년 평균 t+2 돌봄 결과",
    "구조환경 수준→t+2 예산",
    "구조환경 변화→예산비중 변화",
    "구조환경 하락 대응 방향",
    "예산 구성비 최대 증감",
    "군집 간 t+2 탐색 신호",
}


def test_official_report_claims_match_tracked_artifacts() -> None:
    checks = run_checks()
    failed = [check for check in checks if not check.passed]

    assert {check.name for check in checks} == EXPECTED_CHECK_NAMES
    assert failed == []
