"""시각화 설정 단위 테스트."""


def test_korean_font_is_not_overwritten_by_seaborn_style():
    """Seaborn 스타일 적용 후에도 한글 폰트가 유지되어야 한다."""
    from src.visualization import plots

    assert plots.plt.rcParams["font.family"] == ["AppleGothic"]
    assert plots.plt.rcParams["axes.unicode_minus"] is False
