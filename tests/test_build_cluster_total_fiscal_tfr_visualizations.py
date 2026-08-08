import pandas as pd
import pytest

from scripts.build_cluster_total_fiscal_tfr_visualizations import (
    build_cluster_trends,
    validate_inputs,
)


def _sample() -> pd.DataFrame:
    rows = []
    for region, k2, k3 in (("가", 1, 1), ("나", 2, 2), ("다", 2, 3)):
        for year in (2018, 2019):
            rows.append(
                {
                    "지역": region,
                    "연도": year,
                    "전체_인구1인당_실질예산_3개년평균": 100.0 + year,
                    "합계출산율_t+1": 1.0,
                    "합계출산율_t+2": 0.9,
                    "군집_2개": k2,
                    "군집_3개": k3,
                }
            )
    return pd.DataFrame(rows)


def _coefficients() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "군집수": k,
                "시차": lag,
                "군집": cluster,
                "계수": 0.1,
                "95%신뢰구간_하한": -0.1,
                "95%신뢰구간_상한": 0.3,
            }
            for k in (2, 3)
            for lag in ("t+1", "t+2")
            for cluster in range(1, k + 1)
        ]
    )


def test_validate_inputs_requires_all_cluster_lag_coefficients() -> None:
    validate_inputs(_sample(), _coefficients())
    with pytest.raises(ValueError, match="모두 필요"):
        validate_inputs(_sample(), _coefficients().iloc[:-1])


def test_build_cluster_trends_keeps_k2_and_k3() -> None:
    result = build_cluster_trends(_sample())
    assert set(result["군집수"]) == {2, 3}
    assert result.loc[result["군집수"].eq(2), "지역수"].max() == 2
