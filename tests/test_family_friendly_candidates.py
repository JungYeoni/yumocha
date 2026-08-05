from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_family_friendly_candidates import (
    INDICATOR_ID,
    REGION_ORDER,
    TARGET_YEARS,
    UNRESOLVED_YEARS,
    apply_confirmed_observations,
    build_candidate_table,
    normalize_region,
    render_report,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_후보.csv"
METADATA_PATH = ROOT / "reports" / "20260804_가족친화_공식원본_메타데이터.csv"
QA_PATH = ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_QA.csv"
CONFIRMED_PATH = ROOT / "reports" / "20260804_가족친화_공식관측_반영값.csv"
FALSE_MATCH_PATH = ROOT / "reports" / "20260804_가족친화_2024_부분일치_중복행_QA.csv"
REGION_COUNTS_2024_PATH = ROOT / "reports" / "20260804_가족친화_2024_공식시도별_집계_QA.csv"
APPLICATION_QA_PATH = ROOT / "reports" / "20260804_가족친화_공식관측_패널반영_QA.csv"
MAPPING_PATH = ROOT / "reports" / "20260804_구조환경지표_결측정책_전수매핑.csv"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("서울", "서울"),
        ("서울특별시 강남구", "서울"),
        ("강원특별자치도 원주시", "강원"),
        ("전라북도 전주시", "전북"),
        ("전북특별자치도 전주시", "전북"),
        ("제주특별자치도 제주시", "제주"),
        (" 세종특별자치시 ", "세종"),
        ("부산광역시 해운대구", "부산"),
        ("경기도 광주시", "경기"),
        (None, None),
        ("", None),
        ("확인불가", None),
    ],
)
def test_normalize_region(raw, expected):
    assert normalize_region(raw) == expected


def test_build_candidate_table_uses_same_year_formula():
    numerator_counts = {
        year: pd.Series({region: index + year for index, region in enumerate(REGION_ORDER)})
        for year in TARGET_YEARS
    }
    denominators = pd.DataFrame(
        {
            year: {region: 100_000 + index for index, region in enumerate(REGION_ORDER)}
            for year in TARGET_YEARS
        }
    )

    candidates = build_candidate_table(numerator_counts, denominators)

    assert len(candidates) == 34
    assert not candidates.duplicated(["지역", "지표_id", "연도"]).any()
    expected = candidates["공식_분자"] / candidates["사업체수_분모"] * 100
    assert np.allclose(candidates["계산_비율"], expected, rtol=0, atol=1e-15)


def test_committed_candidate_artifacts_are_complete():
    candidates = pd.read_csv(CANDIDATE_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    qa = pd.read_csv(QA_PATH)

    assert len(candidates) == 34
    assert set(candidates["지역"]) == set(REGION_ORDER)
    assert set(candidates["연도"]) == set(TARGET_YEARS)
    assert candidates["지표_id"].eq(INDICATOR_ID).all()
    assert candidates["QA_상태"].eq("PASS").all()
    assert not candidates.duplicated(["지역", "지표_id", "연도"]).any()
    assert candidates.groupby("연도")["공식_분자"].sum().to_dict() == {
        2020: 4_340,
        2021: 4_918,
    }
    formula = candidates["공식_분자"] / candidates["사업체수_분모"] * 100
    assert np.allclose(candidates["계산_비율"], formula, rtol=0, atol=1e-14)

    assert len(metadata) == 7
    assert metadata["연도"].tolist() == [2020, 2021, 2024, 2025, 2018, 2022, 2023]
    assert metadata["파일크기_bytes"].tolist() == [
        211_042,
        242_583,
        242_330,
        278_535,
        120_569,
        189_351,
        205_120,
    ]
    assert metadata["SHA-256"].tolist() == [
        "e3bd6762f24a5b66588c242f3e4598f69a328950d8985e11a39e257438d43f64",
        "af79a43769a4b7db59fa2446f832f181cff0f93e8c7978517a350c21dff14b1f",
        "6026921d157c499156d6ddbce820833c6f96a6709b3814e940dae5a47f3d4696",
        "187c81031cfdc84032d9419aa1b26cce0f8482a133093c9364b3942a61ff3275",
        "dc5eab3fd5cc00ace2aea8b1ea7ec3af416662368c2ea5a0ff5239c809ec477e",
        "71ae5370af8a5d12942cdafca891129127d68892c7dfc6807b1b71ea4d1cbce7",
        "3a2de0e483edb98c275d1e5b61171438945977114a11d42a3fb20b45f7a73571",
    ]
    # 2020·2021·2024·2025는 다운로드 경위·출처가 확인·검증됐고(PASS로 시작),
    # 2018·2022·2023은 파일 자체(SHA-256)는 있지만 정확한 다운로드 출처가 아직
    # 미확인 상태임을 "미검증"으로 정직하게 표시한다.
    assert metadata["원본_검증상태"].str.startswith("PASS").sum() == 4
    assert metadata["원본_검증상태"].str.startswith("미검증").sum() == 3
    assert qa["판정"].eq("PASS").all()


def test_confirmed_observations_and_false_match_evidence_are_exact():
    confirmed = pd.read_csv(CONFIRMED_PATH)
    false_matches = pd.read_csv(FALSE_MATCH_PATH)
    region_counts_2024 = pd.read_csv(REGION_COUNTS_2024_PATH)
    application_qa = pd.read_csv(APPLICATION_QA_PATH)

    assert len(confirmed) == 37
    assert not confirmed.duplicated(["지역", "지표_id", "연도"]).any()
    assert confirmed["반영유형"].value_counts().to_dict() == {
        "공식 원자료 직접 복원": 34,
        "공식 집계 오류 정정": 3,
    }
    assert confirmed["관측상태"].eq("관측").all()
    formula = confirmed["공식_분자"] / confirmed["사업체수_분모"] * 100
    assert np.allclose(confirmed["측정값"], formula, rtol=0, atol=1e-15)
    corrected = confirmed.loc[confirmed["연도"].eq(2024)].set_index("지역")
    assert corrected["공식_분자"].to_dict() == {"전국": 6_502, "대구": 218, "광주": 140}

    assert len(false_matches) == 79
    assert false_matches.groupby(["공식_지역", "부분일치_오배정_지역"]).size().to_dict() == {
        ("부산", "대구"): 59,
        ("경기", "광주"): 20,
    }
    assert false_matches["정상화_검증"].eq(false_matches["공식_지역"]).all()

    assert len(region_counts_2024) == 17
    assert set(region_counts_2024["지역"]) == set(REGION_ORDER)
    assert region_counts_2024["공식_분자"].sum() == 6_502
    assert region_counts_2024["패널_값_변경"].sum() == 2
    formula_2024 = region_counts_2024["공식_분자"] / region_counts_2024["사업체수_분모"] * 100
    assert np.allclose(region_counts_2024["계산_비율"], formula_2024, rtol=0, atol=1e-15)

    assert len(application_qa) == 37
    assert application_qa["QA_상태"].eq("PASS").all()
    direct = application_qa["반영유형"].eq("공식 원자료 직접 복원")
    correction = application_qa["반영유형"].eq("공식 집계 오류 정정")
    assert direct.sum() == 34
    assert application_qa.loc[direct, "반영전값"].isna().all()
    assert correction.sum() == 3
    assert application_qa.loc[correction, "지역"].tolist() == ["전국", "대구", "광주"]
    applied = application_qa.merge(
        confirmed[["지역", "지표_id", "연도", "측정값"]],
        on=["지역", "지표_id", "연도"],
        validate="one_to_one",
    )
    assert np.allclose(applied["반영후값"], applied["측정값"], rtol=0, atol=1e-15)


def test_apply_confirmed_observations_changes_only_reviewed_keys():
    confirmed = pd.read_csv(CONFIRMED_PATH)
    regions = ["전국", *REGION_ORDER]
    frame = pd.DataFrame(
        {
            "지역": regions,
            "세부지표": ["가족친화인증기업 비율"] * len(regions),
            **{year: [float(year)] * len(regions) for year in range(2016, 2025)},
        }
    )
    for year in TARGET_YEARS:
        frame.loc[frame["지역"].isin(REGION_ORDER), year] = np.nan
    before = frame.copy()

    applied, audit = apply_confirmed_observations(frame, confirmed)

    assert len(audit) == 37
    assert audit["QA_상태"].eq("PASS").all()
    for row in confirmed.to_dict("records"):
        actual = applied.loc[applied["지역"].eq(row["지역"]), row["연도"]].iloc[0]
        assert np.isclose(actual, row["측정값"], rtol=0, atol=1e-15)
    unchanged_years = [2016, 2017, 2018, 2019, 2022, 2023]
    assert applied[unchanged_years].equals(before[unchanged_years])


def test_render_report_handles_missing_panel_cross_checks():
    """render_report must not crash when main() ran in bootstrap mode (no panel yet).

    Regression test for the #70 circular-dependency fix: on a clean checkout the panel and
    mapping files do not exist yet, so main() now passes an empty ``reference_matches`` dict
    instead of skipping the report. render_report used to index it directly
    (``reference_matches[2018]``), which raised KeyError in that mode.
    """
    candidates = pd.read_csv(CANDIDATE_PATH)
    confirmed = pd.read_csv(CONFIRMED_PATH)
    metadata = pd.read_csv(METADATA_PATH)
    qa = pd.read_csv(QA_PATH)
    false_matches = pd.read_csv(FALSE_MATCH_PATH)
    region_counts_2024 = pd.read_csv(REGION_COUNTS_2024_PATH)

    numerator_counts = {
        year: candidates.loc[candidates["연도"].eq(year)].set_index("지역")["공식_분자"]
        for year in TARGET_YEARS
    }
    numerator_counts[2024] = region_counts_2024.set_index("지역")["공식_분자"]

    report = render_report(metadata, numerator_counts, candidates, confirmed, false_matches, {}, qa)

    assert "N/A(패널 없음)" in report
    assert "패널·정책 매핑 파일이 없어" in report


def test_unresolved_51_keys_remain_blocked_and_have_no_candidates():
    candidates = pd.read_csv(CANDIDATE_PATH)
    mapping = pd.read_csv(MAPPING_PATH)
    unresolved = mapping.loc[
        mapping["지표_id"].eq(INDICATOR_ID)
        & mapping["지역"].isin(REGION_ORDER)
        & mapping["연도"].isin(UNRESOLVED_YEARS)
    ]

    assert len(unresolved) == 51
    assert unresolved["imputation_policy"].eq("pending_review").all()
    assert unresolved["block_imputation"].all()
    assert not candidates["연도"].isin(UNRESOLVED_YEARS).any()
    assert candidates.merge(
        unresolved[["지역", "지표_id", "연도"]],
        on=["지역", "지표_id", "연도"],
        how="inner",
    ).empty
