from pathlib import Path

import pandas as pd

from scripts.build_subarea_keyword_visualizations import (
    display_input_path,
    plot_bar_grid,
    plot_wordcloud_grid,
)
from src.features.keyword_tfidf import prepare_text_column


def _ranking() -> pd.DataFrame:
    rows = []
    for group_index in range(12):
        for rank in range(1, 4):
            rows.append(
                {
                    "세부영역": f"영역 {group_index + 1}",
                    "순위": rank,
                    "단어": f"단어{group_index + 1}_{rank}",
                    "평균_TFIDF": 1 / rank,
                }
            )
    return pd.DataFrame(rows)


def test_visualization_builders_create_files(tmp_path: Path) -> None:
    ranking = _ranking()
    font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if not font_path.exists():
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    cloud_path = tmp_path / "cloud.png"
    bar_path = tmp_path / "bar.png"
    plot_wordcloud_grid(ranking, title="테스트", output_path=cloud_path, font_path=font_path)
    plot_bar_grid(ranking, title="테스트", output_path=bar_path)
    assert cloud_path.stat().st_size > 0
    assert bar_path.stat().st_size > 0


def test_display_input_path_accepts_external_path(tmp_path: Path) -> None:
    external = (tmp_path / "input.xlsx").resolve()
    assert display_input_path(external) == str(external)


def test_prepare_text_column_records_fallback_column_presence() -> None:
    with_fallback = pd.DataFrame({"정제": ["", "정제"], "원문": ["원문", "원문2"]})
    _, present_stats = prepare_text_column(
        with_fallback, preferred_column="정제", fallback_column="원문"
    )
    _, absent_stats = prepare_text_column(
        with_fallback.drop(columns="원문"), preferred_column="정제", fallback_column="원문"
    )
    assert present_stats["원문열존재"] is True
    assert present_stats["원문대체"] == 1
    assert absent_stats["원문열존재"] is False
