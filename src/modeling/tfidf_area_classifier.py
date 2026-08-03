"""TF-IDF 기반 세부사업 영역분류 학습·평가·추론 유틸리티."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from scripts.consolidate_2021_area_labels import MAJOR_BY_SUBCATEGORY, normalize_text

RANDOM_STATE = 42
TEXT_VARIANTS = {
    "사업명": ("세부사업명",),
    "사업명_주요내용": ("세부사업명", "주요내용_정제"),
}

np.random.seed(RANDOM_STATE)


@dataclass(frozen=True)
class EvaluationResult:
    """입력 방식 하나의 교차검증 결과."""

    text_variant: str
    metrics: dict[str, float]
    class_report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    predictions: pd.DataFrame


def build_text(frame: pd.DataFrame, text_variant: str) -> pd.Series:
    """입력 방식에 맞는 분류 텍스트를 생성한다."""
    if text_variant not in TEXT_VARIANTS:
        raise ValueError(f"지원하지 않는 입력 방식: {text_variant}")

    columns = TEXT_VARIANTS[text_variant]
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"텍스트 열 누락: {missing_columns}")

    normalized = [frame[column].map(normalize_text) for column in columns]
    text = normalized[0]
    for values in normalized[1:]:
        text = text + " " + values
    text = text.str.strip()

    if normalized[0].eq("").any():
        raise ValueError("세부사업명이 비어 있는 행이 있습니다.")
    if text.eq("").any():
        raise ValueError("분류 텍스트를 만들 수 없는 행이 있습니다.")
    return text


def validate_training_data(frame: pd.DataFrame) -> None:
    """학습 데이터의 필수 열·키·taxonomy를 검증한다.

    2021년 단일 연도 학습본에는 ``연도`` 열이 없어 지역·원본행만으로도
    키가 유일했다. 2021~2024년을 합친 학습본은 같은 지역·원본행 번호가
    연도마다 반복되므로, ``연도`` 열이 있으면 연도·지역·원본행을 키로
    쓰고 없으면 기존과 동일하게 지역·원본행만 쓴다.
    """
    required = ["지역", "원본행", "세부사업명", "주요내용_정제", "대영역", "세부영역"]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"학습 필수 열 누락: {missing_columns}")
    if frame.empty:
        raise ValueError("학습 데이터가 비어 있습니다.")
    if frame[["지역", "원본행", "대영역", "세부영역"]].isna().any().any():
        raise ValueError("학습 키 또는 라벨에 결측이 있습니다.")
    key_columns = ["연도", "지역", "원본행"] if "연도" in frame.columns else ["지역", "원본행"]
    if frame.duplicated(key_columns).any():
        raise ValueError(f"학습 데이터의 {'·'.join(key_columns)} 키가 중복되었습니다.")

    canonical_major = frame["세부영역"].map(MAJOR_BY_SUBCATEGORY)
    if canonical_major.isna().any():
        unknown = sorted(frame.loc[canonical_major.isna(), "세부영역"].unique())
        raise ValueError(f"정의되지 않은 세부영역: {unknown}")
    if not frame["대영역"].eq(canonical_major).all():
        raise ValueError("대영역과 세부영역 taxonomy가 일치하지 않습니다.")

    build_text(frame, "사업명")
    build_text(frame, "사업명_주요내용")


def build_pipeline() -> Pipeline:
    """재현 가능한 문자 단위 TF-IDF 분류 파이프라인을 반환한다."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_text_variant(
    frame: pd.DataFrame,
    text_variant: str,
    *,
    n_splits: int = 5,
) -> EvaluationResult:
    """지역 단위 교차검증으로 입력 방식 하나를 평가한다."""
    validate_training_data(frame)
    if frame["지역"].nunique() < n_splits:
        raise ValueError("교차검증 fold 수보다 지역 수가 적습니다.")

    labels = frame["세부영역"].astype("string")
    groups = frame["지역"].astype("string")
    # 분할 기준: 지역을 그룹으로 완전히 분리하고 세부영역 라벨을 stratify한다.
    # n_splits는 지역 그룹을 나눌 교차검증 fold 수를 결정한다.
    # 피처(텍스트) 생성보다 분할을 먼저 확정하기 위해 fold 인덱스를 미리
    # 계산한다. StratifiedGroupKFold.split은 X의 값이 아니라 길이만
    # 사용하므로 자리표시자로도 동일한 분할이 나온다.
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_indices = list(splitter.split(np.zeros(len(frame)), labels, groups))

    text = build_text(frame, text_variant)
    probabilities = cross_val_predict(
        build_pipeline(),
        text,
        labels,
        cv=fold_indices,
        method="predict_proba",
        n_jobs=1,
    )
    classes = np.sort(labels.unique())
    predicted = classes[np.argmax(probabilities, axis=1)]
    confidence = probabilities.max(axis=1)

    report_dict = classification_report(
        labels,
        predicted,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(report_dict["accuracy"]),
        "f1_macro": float(report_dict["macro avg"]["f1-score"]),
        "f1_weighted": float(report_dict["weighted avg"]["f1-score"]),
    }
    class_report = pd.DataFrame(report_dict).transpose().rename_axis("라벨").reset_index()
    matrix = pd.DataFrame(
        confusion_matrix(labels, predicted, labels=classes),
        index=classes,
        columns=classes,
    )
    matrix.index.name = "실제"
    matrix.columns.name = "예측"

    predictions = frame[["지역", "원본행", "세부사업명", "대영역", "세부영역"]].copy()
    predictions["입력방식"] = text_variant
    predictions["예측_세부영역"] = predicted
    predictions["예측_대영역"] = pd.Series(predicted, index=predictions.index).map(
        MAJOR_BY_SUBCATEGORY
    )
    predictions["예측_신뢰도"] = confidence
    predictions["정답일치"] = predictions["세부영역"].eq(predictions["예측_세부영역"])
    return EvaluationResult(text_variant, metrics, class_report, matrix, predictions)


def choose_best_result(results: list[EvaluationResult]) -> EvaluationResult:
    """macro-F1, weighted-F1, 입력 방식 이름 순으로 최종 결과를 선택한다."""
    if not results:
        raise ValueError("비교할 평가 결과가 없습니다.")
    return max(
        results,
        key=lambda result: (
            result.metrics["f1_macro"],
            result.metrics["f1_weighted"],
            result.text_variant,
        ),
    )


def fit_model(frame: pd.DataFrame, text_variant: str) -> Pipeline:
    """전체 학습 데이터로 최종 모델을 적합한다."""
    validate_training_data(frame)
    model = build_pipeline()
    model.fit(build_text(frame, text_variant), frame["세부영역"].astype("string"))
    return model


def predict(
    model: Pipeline,
    frame: pd.DataFrame,
    text_variant: str,
    *,
    low_confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """분류 대상에 세부·대영역 예측과 신뢰도 플래그를 추가한다."""
    if not 0 < low_confidence_threshold < 1:
        raise ValueError("저신뢰 임계값은 0과 1 사이여야 합니다.")

    output = frame.copy()
    probabilities = model.predict_proba(build_text(output, text_variant))
    classes = model.named_steps["classifier"].classes_
    predicted = classes[np.argmax(probabilities, axis=1)]
    output["예측_세부영역"] = predicted
    output["예측_대영역"] = pd.Series(predicted, index=output.index).map(MAJOR_BY_SUBCATEGORY)
    output["예측_신뢰도"] = probabilities.max(axis=1)
    output["저신뢰_검토대상"] = output["예측_신뢰도"].lt(low_confidence_threshold)
    return output


def save_model_bundle(
    path: Path,
    *,
    model: Pipeline,
    text_variant: str,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> None:
    """모델 번들과 무결성 검증용 SHA-256 sidecar를 함께 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "text_variant": text_variant,
        "text_columns": TEXT_VARIANTS[text_variant],
        "metrics": metrics,
        "metadata": metadata,
    }
    joblib.dump(bundle, path)
    model_checksum_path(path).write_text(_file_sha256(path) + "\n", encoding="ascii")


def load_model_bundle(path: Path) -> dict[str, Any]:
    """신뢰된 출처의 모델 번들을 체크섬 검증 후 불러온다.

    ``joblib``은 pickle 역직렬화를 사용하므로 외부에서 받은 신뢰할 수 없는
    파일에는 사용하지 않는다. sidecar 체크섬은 파일 손상·잘못된 아티팩트
    사용을 탐지하지만 악의적인 파일과 체크섬의 동시 변조를 막는 서명은 아니다.
    """
    checksum_path = model_checksum_path(path)
    if not checksum_path.exists():
        raise ValueError(f"모델 체크섬 파일이 없습니다: {checksum_path}")
    expected_checksum = checksum_path.read_text(encoding="ascii").strip()
    actual_checksum = _file_sha256(path)
    if not expected_checksum or actual_checksum != expected_checksum:
        raise ValueError("모델 번들의 SHA-256 체크섬이 일치하지 않습니다.")

    bundle = joblib.load(path)
    required = {"model", "text_variant", "text_columns", "metrics", "metadata"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("올바른 TF-IDF 모델 번들이 아닙니다.")
    return bundle


def model_checksum_path(path: Path) -> Path:
    """모델 파일에 대응하는 SHA-256 sidecar 경로를 반환한다."""
    return path.with_suffix(path.suffix + ".sha256")


def _file_sha256(path: Path) -> str:
    """파일의 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
