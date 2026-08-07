import pandas as pd

from scripts.compare_subarea_population_denominator_sensitivity import build_comparison_table


def test_build_comparison_table_flags_sign_and_significance_agreement():
    results_by_denominator = {
        "전체인구": pd.DataFrame(
            {
                "모형": ["1-1. 고용여건", "2-1. 돌봄 여건"],
                "계수": [2.0, -1.5],
                "p값": [0.01, 0.20],
            }
        ),
        "20_39세인구": pd.DataFrame(
            {
                "모형": ["1-1. 고용여건", "2-1. 돌봄 여건"],
                "계수": [1.8, 1.0],
                "p값": [0.03, 0.04],
            }
        ),
    }

    comparison = build_comparison_table(results_by_denominator)

    row_agree = comparison.loc[comparison["모형"].eq("1-1. 고용여건")].iloc[0]
    row_disagree = comparison.loc[comparison["모형"].eq("2-1. 돌봄 여건")].iloc[0]
    assert bool(row_agree["부호_일치"]) is True
    assert bool(row_agree["유의성_5%_일치"]) is True
    assert bool(row_disagree["부호_일치"]) is False
    assert bool(row_disagree["유의성_5%_일치"]) is False
