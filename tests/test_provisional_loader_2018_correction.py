import pandas as pd

import pytest

from src.provisional.loader import _prepare_funding_rows


def _sample_2018_frame(*, 원본행: float = 8200.0, 예산: object = -20) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "지역": "경북",
                "세부사업명": "여성지도자 육성",
                "사업분류재정구분": "계",
                "재원구분": pd.NA,
                "머리글행": 1.0,
                "원본행": 원본행,
                "2018년 예산": 예산,
                "2017년 예산": 45,
            }
        ]
    )


def test_gyeongbuk_2018_negative_budget_typo_is_corrected_to_positive():
    frame = _sample_2018_frame()
    result = _prepare_funding_rows(
        frame, year=2018, current_column="2018년 예산", previous_column="2017년 예산"
    )
    row = result.loc[result["원본행"].eq(8200.0)].iloc[0]
    assert row["2018년 예산"] == 20


def test_gyeongbuk_2018_correction_raises_if_row_missing():
    frame = _sample_2018_frame(원본행=9999.0)
    with pytest.raises(ValueError, match="보정 대상 행이 1개가 아닙니다"):
        _prepare_funding_rows(
            frame, year=2018, current_column="2018년 예산", previous_column="2017년 예산"
        )


def test_gyeongbuk_2018_correction_raises_if_value_already_changed():
    frame = _sample_2018_frame(예산=-999)
    with pytest.raises(ValueError, match="원본값이 예상과 다릅니다"):
        _prepare_funding_rows(
            frame, year=2018, current_column="2018년 예산", previous_column="2017년 예산"
        )
