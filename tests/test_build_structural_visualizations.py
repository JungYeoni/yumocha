import pandas as pd

from scripts.build_structural_visualizations import (
    EXPECTED_YEARS,
    load_indicator_names,
    prepare_visualization_input,
)
from src.features.trend_eda import STRUCTURAL_INDICATOR_DIRECTIONS


def test_manifest_and_direction_map_cover_all_28_panel_indicators():
    names = load_indicator_names()

    assert len(names) == 28
    assert set(names).issubset(STRUCTURAL_INDICATOR_DIRECTIONS)


def test_prepare_visualization_input_pivots_long_panel_without_status_duplicates():
    rows = []
    for region in ["서울", "부산", "전국"]:
        for year in EXPECTED_YEARS:
            rows.append(
                {
                    "지역": region,
                    "지표_id": "indicator",
                    "지표명": "지표",
                    "연도": year,
                    "측정값": float(year),
                    "대분류": "경제·고용·주거",
                    "세부영역": "고용여건",
                    "방향": "positive",
                    "관측상태": "observed" if year % 2 else "missing",
                }
            )

    wide = prepare_visualization_input(
        pd.DataFrame(rows),
        expected_regions=("서울", "부산"),
    )

    assert len(wide) == 3
    assert set(wide["지역"]) == {"서울", "부산", "전국"}
    assert wide["검증상태"].eq("검증 원자료 기준").all()
