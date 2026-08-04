from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_family_friendly_national_candidates import (
    NATIONAL_CUMULATIVE_TOTALS,
    TARGET_YEARS,
    build_national_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "reports" / "20260805_가족친화_전국_직접복원_후보.csv"
QA_PATH = ROOT / "reports" / "20260805_가족친화_전국_직접복원_QA.csv"


def test_build_national_candidates_computes_expected_ratios():
    denominators = {2016: 1_000, 2017: 2_000, 2019: 3_000, 2020: 4_000, 2021: 5_000}
    qa_records: list[dict[str, object]] = []

    candidates = build_national_candidates(denominators, qa_records)

    assert len(candidates) == 5
    assert set(candidates["연도"]) == set(TARGET_YEARS)
    assert candidates["지역"].eq("전국").all()
    assert candidates["관측상태"].eq("관측").all()
    for year in TARGET_YEARS:
        row = candidates.loc[candidates["연도"].eq(year)].iloc[0]
        expected = NATIONAL_CUMULATIVE_TOTALS[year] / denominators[year] * 100
        assert np.isclose(row["측정값"], expected, rtol=0, atol=1e-15)
    assert pd.DataFrame(qa_records)["판정"].eq("PASS").all()


def test_committed_national_candidate_artifacts_are_complete():
    candidates = pd.read_csv(CANDIDATE_PATH)
    qa = pd.read_csv(QA_PATH)

    assert len(candidates) == 5
    assert set(candidates["연도"]) == set(TARGET_YEARS)
    assert candidates["지역"].eq("전국").all()
    assert candidates["QA_상태"].eq("PASS").all()
    formula = candidates["공식_분자"] / candidates["사업체수_분모"] * 100
    assert np.allclose(candidates["측정값"], formula, rtol=0, atol=1e-15)
    assert candidates.set_index("연도")["공식_분자"].to_dict() == NATIONAL_CUMULATIVE_TOTALS
    assert qa["판정"].eq("PASS").all()

    # 2020/2021 cross-check against the already-confirmed province-level national totals.
    assert candidates.set_index("연도").loc[2020, "공식_분자"] == 4_340
    assert candidates.set_index("연도").loc[2021, "공식_분자"] == 4_918
