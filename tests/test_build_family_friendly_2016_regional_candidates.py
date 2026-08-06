import pandas as pd

import pytest

from scripts.build_family_friendly_2016_regional_candidates import (
    YEAR,
    apply_2016_candidates_to_panel,
    remove_resolved_rows_from_mapping,
)
from scripts.build_family_friendly_candidates import INDICATOR_ID, REGION_ORDER


def _sample_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "지역": region,
                "지표_id": INDICATOR_ID,
                "연도": YEAR,
                "측정값": 10.0 + i,
                "QA_상태": "PASS",
                "반영유형": "공식 원자료 직접 복원",
                "관측상태": "관측",
            }
            for i, region in enumerate(REGION_ORDER)
        ]
    )


def _sample_panel() -> pd.DataFrame:
    rows = [
        {
            "지역": region,
            "지표_id": INDICATOR_ID,
            "연도": YEAR,
            "측정값": float("nan"),
            "원본행존재": False,
        }
        for region in REGION_ORDER
    ]
    rows.append(
        {"지역": "서울", "지표_id": INDICATOR_ID, "연도": 2018, "측정값": 1.23, "원본행존재": True}
    )
    return pd.DataFrame(rows)


def test_apply_2016_candidates_to_panel_fills_only_the_seventeen_cells():
    panel = _sample_panel()
    candidates = _sample_candidates()

    updated, audit = apply_2016_candidates_to_panel(panel, candidates)

    assert len(audit) == 17
    assert audit["QA_상태"].eq("PASS").all()
    assert audit["반영전값"].isna().all()
    for i, region in enumerate(REGION_ORDER):
        row = updated.loc[updated["지역"].eq(region) & updated["연도"].eq(YEAR)].iloc[0]
        assert row["측정값"] == 10.0 + i
        assert bool(row["원본행존재"]) is True

    unrelated = updated.loc[updated["연도"].eq(2018)].iloc[0]
    assert unrelated["측정값"] == 1.23


def test_apply_2016_candidates_to_panel_rejects_existing_value():
    panel = _sample_panel()
    panel.loc[panel["지역"].eq("서울") & panel["연도"].eq(YEAR), "측정값"] = 999.0
    candidates = _sample_candidates()

    with pytest.raises(ValueError, match="이미 존재"):
        apply_2016_candidates_to_panel(panel, candidates)


def _sample_mapping() -> pd.DataFrame:
    rows = [
        {
            "지역": region,
            "지표_id": INDICATOR_ID,
            "연도": YEAR,
            "block_imputation": True,
            "imputation_policy": "pending_review",
        }
        for region in REGION_ORDER
    ]
    rows.append(
        {
            "지역": "서울",
            "지표_id": INDICATOR_ID,
            "연도": 2017,
            "block_imputation": True,
            "imputation_policy": "pending_review",
        }
    )
    return pd.DataFrame(rows)


def test_remove_resolved_rows_from_mapping_drops_only_2016():
    mapping = _sample_mapping()
    candidates = _sample_candidates()

    updated, removed = remove_resolved_rows_from_mapping(mapping, candidates)

    assert len(removed) == 17
    assert len(updated) == 1
    assert updated.iloc[0]["연도"] == 2017
