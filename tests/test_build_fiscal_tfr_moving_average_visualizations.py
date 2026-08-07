import numpy as np
import pandas as pd

from scripts.build_fiscal_tfr_moving_average_visualizations import (
    BUDGET,
    OUTCOME,
    PREDICTOR,
    prepare_care_sample,
    residualize_two_way,
)


def test_prepare_care_sample_filters_and_transforms_budget():
    data = pd.DataFrame(
        {
            "지역": ["서울", "서울"],
            "연도": [2018, 2018],
            "세부영역": ["2-1. 돌봄 여건", "1-1. 고용여건"],
            BUDGET: [99.0, 10.0],
            OUTCOME: [0.8, 0.8],
        }
    )
    result = prepare_care_sample(data)
    assert len(result) == 1
    assert result.iloc[0][PREDICTOR] == np.log1p(99.0)


def test_residualize_two_way_removes_region_and_year_means():
    rows = []
    for region_index, region in enumerate(["서울", "부산", "대구"]):
        for year_index, year in enumerate([2018, 2019, 2020]):
            rows.append({"지역": region, "연도": year, "값": region_index + year_index})
    sample = pd.DataFrame(rows)
    residuals = residualize_two_way(sample, "값")
    assert np.max(np.abs(residuals)) < 1e-10
