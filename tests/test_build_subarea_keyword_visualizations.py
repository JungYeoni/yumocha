from pathlib import Path

import pandas as pd

from scripts.build_subarea_keyword_visualizations import plot_bar_grid, plot_wordcloud_grid


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
