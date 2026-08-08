from scripts.verify_subarea_keyword_outputs import run_checks


EXPECTED_CHECK_NAMES = {
    f"{field}:{check}"
    for field in ("세부사업명", "주요내용")
    for check in (
        "12개 영역",
        "영역당 상위어",
        "불용어 제외",
        "중복 민감도",
        "워드클라우드",
        "상위단어_막대그래프",
    )
}


def test_keyword_outputs_meet_completion_criteria() -> None:
    checks = run_checks()
    assert {check.name for check in checks} == EXPECTED_CHECK_NAMES
    assert [check for check in checks if not check.passed] == []
