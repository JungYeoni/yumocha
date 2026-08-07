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


def _fiscal_response_sample() -> pd.DataFrame:
    rows = []
    for region in ("서울", "부산"):
        for year in (2016, 2017):
            for subarea, base in (("1-1. 고용여건", 100.0), ("2-1. 돌봄 여건", 5_000.0)):
                rows.append(
                    {
                        "지역": region,
                        "연도": year,
                        "세부영역": subarea,
                        "인구1인당_실질예산_원": base + year,
                    }
                )
    return pd.DataFrame(rows)


def test_fiscal_response_overview_uses_log_axes_for_scale_gap():
    from src.visualization.trends import plot_fiscal_response_overview

    fig = plot_fiscal_response_overview(_fiscal_response_sample())

    assert len(fig.axes) == 4
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[2].get_xscale() == "log"
    plt.close(fig)


def test_fiscal_response_subarea_small_multiples_creates_one_panel_per_subarea():
    from src.visualization.trends import plot_fiscal_response_subarea_small_multiples

    fig = plot_fiscal_response_subarea_small_multiples(
        _fiscal_response_sample(), subarea_order=["1-1. 고용여건", "2-1. 돌봄 여건"]
    )

    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == 2
    assert visible_axes[0].get_yscale() == "log"
    plt.close(fig)
