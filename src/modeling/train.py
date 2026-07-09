"""
모델 학습 및 하이퍼파라미터 튜닝 유틸리티.

원칙:
- 전처리는 Pipeline 내에서만 수행 (leakage 방지)
- 테스트셋은 최종 평가에서만 1회 사용
- 모든 실험에 random_state=42 적용
"""

from pathlib import Path

import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def evaluate_models(
    models: dict,
    preprocessor,
    X_train,
    y_train,
    scoring: str = "f1_macro",
    cv_folds: int = 5,
) -> dict:
    """여러 모델을 교차검증으로 비교.

    Args:
        models: {"모델명": estimator} 딕셔너리
        preprocessor: ColumnTransformer (fit되지 않은 상태)
        scoring: sklearn scoring string
        cv_folds: K-Fold 수

    Returns:
        dict: {"모델명": {"mean": float, "std": float}}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {"mean": round(scores.mean(), 4), "std": round(scores.std(), 4)}
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


def save_model(model, path: str | Path) -> None:
    """학습된 모델을 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"모델 저장: {path}")


def load_model(path: str | Path):
    """저장된 모델 로드."""
    return joblib.load(path)
