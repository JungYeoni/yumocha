import numpy as np
import pandas as pd

import pytest

from scripts.apply_confirmed_observations_to_panel import (
    CORRECTION_TYPE,
    RESTORATION_TYPE,
    apply_regional_confirmed_observations_to_panel,
)
from scripts.build_family_friendly_candidates import INDICATOR_ID


def _sample_panel() -> pd.DataFrame:
    rows = [
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2020,
            "측정값": float("nan"),
            "원본행존재": False,
        },
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2021,
            "측정값": float("nan"),
            "원본행존재": False,
        },
        {
            "지역": "전국",
            "지표_id": INDICATOR_ID,
            "연도": 2024,
            "측정값": 999.0,
            "원본행존재": True,
        },
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2018,
            "측정값": 1.23,
            "원본행존재": True,
        },
        {
            "지역": "서울",
            "지표_id": "youth_employment_rate",
            "연도": 2020,
            "측정값": 42.0,
            "원본행존재": True,
        },
    ]
    return pd.DataFrame(rows)


def _sample_confirmed() -> pd.DataFrame:
    rows = [
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2020,
            "측정값": 0.294037,
            "반영유형": RESTORATION_TYPE,
            "관측상태": "관측",
            "QA_상태": "PASS",
        },
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2021,
            "측정값": 0.3,
            "반영유형": RESTORATION_TYPE,
            "관측상태": "관측",
            "QA_상태": "PASS",
        },
        {
            "지역": "전국",
            "지표_id": INDICATOR_ID,
            "연도": 2024,
            "측정값": 0.3088699644148738,
            "반영유형": CORRECTION_TYPE,
            "관측상태": "관측",
            "QA_상태": "PASS",
        },
    ]
    # pad up to the expected 37-row contract with additional restoration rows on
    # otherwise-unused indicator/year/region combinations.
    for i in range(34):
        rows.append(
            {
                "지역": f"지역{i}",
                "지표_id": INDICATOR_ID,
                "연도": 2020,
                "측정값": float(i),
                "반영유형": RESTORATION_TYPE,
                "관측상태": "관측",
                "QA_상태": "PASS",
            }
        )
    return pd.DataFrame(rows[:37]) if len(rows) >= 37 else pd.DataFrame(rows)


def test_fills_missing_restoration_cell_and_overwrites_correction_cell():
    panel = pd.concat(
        [
            _sample_panel(),
            pd.DataFrame(
                [
                    {
                        "지역": f"지역{i}",
                        "지표_id": INDICATOR_ID,
                        "연도": 2020,
                        "측정값": float("nan"),
                        "원본행존재": False,
                    }
                    for i in range(34)
                ]
            ),
        ],
        ignore_index=True,
    )
    confirmed = _sample_confirmed()
    assert len(confirmed) == 37

    updated, audit = apply_regional_confirmed_observations_to_panel(panel, confirmed)

    assert len(audit) == 37
    assert audit["QA_상태"].eq("PASS").all()

    seoul_2020 = updated.loc[
        updated["지역"].eq("서울") & updated["연도"].eq(2020) & updated["지표_id"].eq(INDICATOR_ID)
    ].iloc[0]
    assert np.isclose(seoul_2020["측정값"], 0.294037, rtol=0, atol=1e-12)
    assert bool(seoul_2020["원본행존재"]) is True

    national_2024 = updated.loc[
        updated["지역"].eq("전국") & updated["연도"].eq(2024) & updated["지표_id"].eq(INDICATOR_ID)
    ].iloc[0]
    assert np.isclose(national_2024["측정값"], 0.3088699644148738, rtol=0, atol=1e-12)

    correction_audit = audit.loc[audit["반영유형"].eq(CORRECTION_TYPE)].iloc[0]
    assert correction_audit["반영전값"] == 999.0

    # unrelated rows untouched
    unrelated = updated.loc[updated["연도"].eq(2018)].iloc[0]
    assert unrelated["측정값"] == 1.23
    other_indicator = updated.loc[updated["지표_id"].eq("youth_employment_rate")].iloc[0]
    assert other_indicator["측정값"] == 42.0


def test_rejects_conflicting_existing_value_for_restoration_type():
    panel = _sample_panel()
    panel.loc[panel["연도"].eq(2020) & panel["지역"].eq("서울"), "측정값"] = 555.0
    confirmed = _sample_confirmed()

    with pytest.raises(ValueError, match="기존 패널값이 예상과 다릅니다"):
        apply_regional_confirmed_observations_to_panel(panel, confirmed)


def test_rejects_wrong_row_count():
    panel = _sample_panel()
    confirmed = _sample_confirmed().iloc[:5]

    with pytest.raises(ValueError, match="37개 키"):
        apply_regional_confirmed_observations_to_panel(panel, confirmed)


def test_rejects_unknown_application_type():
    panel = _sample_panel()
    confirmed = _sample_confirmed().copy()
    confirmed.loc[confirmed.index[0], "반영유형"] = "알수없음"

    with pytest.raises(ValueError, match="알 수 없는 반영유형"):
        apply_regional_confirmed_observations_to_panel(panel, confirmed)
