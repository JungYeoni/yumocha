import pandas as pd
import pytest

from scripts.run_subarea_fiscal_tfr_moving_average_regression import (
    MA3,
    add_bh_correction,
    build_moving_average_sample,
)


def test_build_moving_average_sample_aligns_future_tfr():
    sample = pd.DataFrame(
        {
            "지역": ["서울"] * 5,
            "세부영역": ["돌봄"] * 5,
            "연도": range(2016, 2021),
            "인구1인당_실질예산_원": [10, 20, 30, 40, 50],
            "합계출산율": [1.0, 1.1, 1.2, 1.3, 1.4],
        }
    )
    result = build_moving_average_sample(sample)
    row = result.loc[result["연도"].eq(2018)].iloc[0]
    assert row[MA3] == pytest.approx(20)
    assert row["합계출산율_t+1"] == pytest.approx(1.3)
    assert row["합계출산율_t+2"] == pytest.approx(1.4)


def test_add_bh_correction_adds_adjusted_results():
    result = add_bh_correction(pd.DataFrame({"p값": [0.001, 0.2, 0.8]}))
    assert list(result.columns[-2:]) == ["FDR_q값", "FDR_0.05_유의"]
    assert bool(result.loc[0, "FDR_0.05_유의"])
