from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_family_friendly_candidates import (
    INDICATOR_ID,
    REGION_ORDER,
    TARGET_YEARS,
    UNRESOLVED_YEARS,
    build_candidate_table,
    normalize_region,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_후보.csv"
METADATA_PATH = ROOT / "reports" / "20260804_가족친화_공식원본_메타데이터.csv"
QA_PATH = ROOT / "reports" / "20260804_가족친화_2020_2021_직접복원_QA.csv"
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

    assert len(metadata) == 2
    assert metadata["연도"].tolist() == [2020, 2021]
    assert metadata["파일크기_bytes"].tolist() == [211_042, 242_583]
    assert metadata["SHA-256"].tolist() == [
        "e3bd6762f24a5b66588c242f3e4598f69a328950d8985e11a39e257438d43f64",
        "af79a43769a4b7db59fa2446f832f181cff0f93e8c7978517a350c21dff14b1f",
    ]
    assert metadata["원본_검증상태"].eq("PASS").all()
    assert qa["판정"].eq("PASS").all()


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
