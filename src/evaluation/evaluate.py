"""
모델 평가 지표 계산 및 잔차 진단.

보고 원칙:
- 점 추정치 단독이 아닌 신뢰구간 또는 CV 표준편차와 함께 제시
- 통계적 유의성과 실질적 유의성(effect size)을 구분
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true, y_pred) -> dict:
    """회귀 평가 지표 계산."""
    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "MAPE": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    """분류 평가 지표 계산."""
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "f1_weighted": round(report["weighted avg"]["f1-score"], 4),
    }
    if y_prob is not None:
        try:
            metrics["auc_roc"] = round(roc_auc_score(y_true, y_prob, multi_class="ovr"), 4)
        except ValueError:
            pass  # 단일 클래스 등 예외 상황
    return metrics


def residual_summary(y_true, y_pred) -> pd.DataFrame:
    """잔차 기술통계 — 회귀 진단용."""
    residuals = np.array(y_true) - np.array(y_pred)
    return pd.DataFrame(
        {
            "mean": [residuals.mean()],
            "std": [residuals.std()],
            "min": [residuals.min()],
            "max": [residuals.max()],
            "skewness": [pd.Series(residuals).skew()],
            "kurtosis": [pd.Series(residuals).kurtosis()],
        }
    ).round(4)
