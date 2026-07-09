"""피처 엔지니어링 단위 테스트."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_time_features, compute_psi

# ── 픽스처 ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ts() -> pd.DataFrame:
    """일별 시계열 샘플 데이터."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    return pd.DataFrame({"value": np.random.randn(100).cumsum()}, index=dates)


@pytest.fixture
def sample_tabular() -> pd.DataFrame:
    """정형 테이블 샘플 데이터."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "feature_a": np.random.randn(n),
            "feature_b": np.random.randn(n) * 2 + 1,
            "feature_c": np.random.randn(n),
        }
    )


# ── 시계열 피처 테스트 ─────────────────────────────────────────────────────────


class TestBuildTimeFeatures:
    def test_lag_columns_created(self, sample_ts):
        result = build_time_features(sample_ts, "value", lags=[1, 7], windows=[])
        assert "value_lag_1" in result.columns
        assert "value_lag_7" in result.columns

    def test_rolling_columns_created(self, sample_ts):
        result = build_time_features(sample_ts, "value", lags=[], windows=[7])
        assert "value_roll_mean_7d" in result.columns
        assert "value_roll_std_7d" in result.columns

    def test_no_leakage_lag1(self, sample_ts):
        """lag_1은 현재 시점 값과 같으면 안 됨 (누수 검증)."""
        result = build_time_features(sample_ts, "value", lags=[1], windows=[])
        # lag_1은 t-1 값이어야 하므로 현재 값과 일치하면 안 됨
        assert not (result["value"] == result["value_lag_1"]).all()

    def test_seasonality_columns_created(self, sample_ts):
        result = build_time_features(sample_ts, "value", lags=[], windows=[])
        assert "day_sin" in result.columns
        assert "month_cos" in result.columns

    def test_original_unchanged(self, sample_ts):
        """원본 데이터프레임이 수정되지 않아야 함."""
        original_cols = list(sample_ts.columns)
        _ = build_time_features(sample_ts, "value", lags=[1], windows=[7])
        assert list(sample_ts.columns) == original_cols


# ── PSI 테스트 ─────────────────────────────────────────────────────────────────


class TestComputePsi:
    def test_identical_distributions(self, sample_tabular):
        """동일 분포는 PSI ≈ 0."""
        psi = compute_psi(sample_tabular["feature_a"], sample_tabular["feature_a"])
        assert psi < 0.01

    def test_different_distributions(self):
        """분포가 다를수록 PSI가 높아야 함."""
        np.random.seed(42)
        expected = pd.Series(np.random.randn(1000))
        actual = pd.Series(np.random.randn(1000) + 5)  # 평균이 5 이동
        psi = compute_psi(expected, actual)
        assert psi > 0.2  # 심각한 드리프트

    def test_returns_float(self, sample_tabular):
        psi = compute_psi(sample_tabular["feature_a"], sample_tabular["feature_b"])
        assert isinstance(psi, float)
