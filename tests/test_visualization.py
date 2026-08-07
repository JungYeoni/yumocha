"""시각화 설정 단위 테스트."""

import matplotlib.pyplot as plt
import pandas as pd


def test_korean_font_is_not_overwritten_by_seaborn_style():
    """Seaborn 스타일 적용 후에도 한글 폰트가 유지되어야 한다."""
    from src.visualization import plots

    assert plots.plt.rcParams["font.family"] == [plots.KOREAN_FONT]
    assert plots.plt.rcParams["axes.unicode_minus"] is False


def test_static_palette_matches_yumocha_web_light_theme():
    from src.visualization import plots

    assert plots.PALETTE[0] == "#246BEB"
    assert plots.PALETTE[1] == "#d1d5db"
    assert plots.PALETTE[2] == "#f3f4f6"
    assert plots.PALETTE[3] == "#c0392b"


def test_structural_overview_uses_regional_median_when_nationwide_is_missing():
    from src.visualization.trends import plot_structural_indicator_overview

    df = pd.DataFrame(
        {
            "지역": ["서울", "서울", "부산", "부산"],
            "연도": [2020, 2021, 2020, 2021],
            "세부지표": ["근로시간"] * 4,
            "측정값": [160.0, 165.0, 170.0, 168.0],
            "실측여부": [True] * 4,
            "급등락후보": [False] * 4,
        }
    )

    fig = plot_structural_indicator_overview(df, indicator="근로시간")

    assert len(fig.axes) == 4
    assert "전국 미공표" in fig.axes[0].get_title()
    plt.close(fig)
