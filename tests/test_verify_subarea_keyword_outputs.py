import pandas as pd

import scripts.verify_subarea_keyword_outputs as verifier
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


def test_empty_ranking_returns_failed_check_instead_of_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(verifier, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(verifier, "FIGURE_DIR", tmp_path)
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    for field in ("세부사업명", "주요내용"):
        pd.DataFrame(columns=["세부영역", "단어"]).to_csv(
            tmp_path / f"2016-2024_세부영역별_{field}_TFIDF_상위단어.csv", index=False
        )
        pd.DataFrame(columns=["상위어_중첩률"]).to_csv(
            tmp_path / f"2016-2024_세부영역별_{field}_TFIDF_중복민감도.csv", index=False
        )

    checks = verifier.run_checks()

    term_checks = [check for check in checks if check.name.endswith(":영역당 상위어")]
    assert all(not check.passed and check.detail == "최소 0개" for check in term_checks)
