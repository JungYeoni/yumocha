from scripts.verify_official_reports import run_checks


def test_official_report_claims_match_tracked_artifacts() -> None:
    checks = run_checks()
    failed = [check for check in checks if not check.passed]

    assert len(checks) >= 20
    assert failed == []
