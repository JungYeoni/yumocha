import pandas as pd
import pytest

from scripts.build_structural_decline_budget_share_response_visualizations import (
    build_decline_events,
    summarize_by_region,
    summarize_by_subarea,
    summarize_region_subarea,
)


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "지역": ["가", "가", "나", "나"],
            "기준연도": [2016, 2017, 2016, 2017],
            "세부영역": ["A", "A", "A", "B"],
            "구조환경지수_변화_t_t1": [-1.0, -2.0, 1.0, -1.0],
            "계획예산비중_변화_t2_t3_pp": [2.0, 0.0, 3.0, -1.0],
            "예산누락주의_두연도": [False, True, False, False],
        }
    )


def test_build_decline_events_uses_strict_decline_and_strict_increase() -> None:
    result = build_decline_events(_sample())
    assert len(result) == 3
    assert result["후행예산비중_증가"].tolist() == [True, False, False]


def test_subarea_summary_reports_numerator_denominator_and_clean_rate() -> None:
    result = summarize_by_subarea(build_decline_events(_sample())).set_index("세부영역")
    assert result.loc["A", "구조환경_하락사례수"] == 2
    assert result.loc["A", "후행예산비중_증가건수"] == 1
    assert result.loc["A", "후행예산비중_증가비율_pct"] == pytest.approx(50)
    assert result.loc["A", "누락주의제외_증가비율_pct"] == pytest.approx(100)


def test_region_summary_ranks_higher_response_rate_first() -> None:
    result = summarize_by_region(build_decline_events(_sample())).set_index("지역")
    assert result.loc["가", "후행예산비중_증가비율_pct"] == pytest.approx(50)
    assert result.loc["나", "후행예산비중_증가비율_pct"] == pytest.approx(0)
    assert result.loc["가", "지역순위"] == 1
    assert result.loc["나", "지역순위"] == 2


def test_region_subarea_summary_keeps_all_regions_and_subareas() -> None:
    result = summarize_region_subarea(build_decline_events(_sample()))
    assert len(result) == 4
    assert set(result["지역"]) == {"가", "나"}
    empty = result.loc[result["지역"].eq("가") & result["세부영역"].eq("B")].iloc[0]
    assert empty["구조환경_하락사례수"] == 0
    assert pd.isna(empty["후행예산비중_증가비율_pct"])


def test_region_with_no_decline_cases_is_preserved() -> None:
    sample = pd.concat(
        [
            _sample(),
            pd.DataFrame(
                {
                    "지역": ["다", "다"],
                    "기준연도": [2016, 2017],
                    "세부영역": ["A", "B"],
                    "구조환경지수_변화_t_t1": [1.0, 2.0],
                    "계획예산비중_변화_t2_t3_pp": [1.0, 1.0],
                    "예산누락주의_두연도": [False, False],
                }
            ),
        ],
        ignore_index=True,
    )
    result = summarize_region_subarea(build_decline_events(sample))
    no_decline = result.loc[result["지역"].eq("다")]
    assert len(no_decline) == 2
    assert no_decline["구조환경_하락사례수"].eq(0).all()
    assert no_decline["후행예산비중_증가비율_pct"].isna().all()
