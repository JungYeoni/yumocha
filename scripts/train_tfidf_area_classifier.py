"""2021년 라벨로 TF-IDF 영역분류기를 학습하고 8개년을 예측한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src.modeling.tfidf_area_classifier import (
    RANDOM_STATE,
    TEXT_VARIANTS,
    choose_best_result,
    evaluate_text_variant,
    fit_model,
    model_checksum_path,
    predict,
    save_model_bundle,
)
from src.modeling.tfidf_review_workbook import create_review_workbook
from src.modeling.similarity_grouping import (
    DEFAULT_CONTENT_SIMILARITY_THRESHOLD,
    DEFAULT_NAME_SIMILARITY_THRESHOLD,
    assign_similarity_groups,
)

np.random.seed(RANDOM_STATE)


def file_sha256(path: Path) -> str:
    """입력 파일 변경 여부 확인용 SHA-256을 계산한다."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reusable_summary(
    summary_path: Path,
    *,
    model_path: Path,
    prediction_output_path: Path,
    review_workbook_path: Path,
    training_sha256: str,
    prediction_sha256: str,
    low_confidence_threshold: float,
    n_splits: int,
    name_similarity_threshold: float = DEFAULT_NAME_SIMILARITY_THRESHOLD,
    content_similarity_threshold: float = DEFAULT_CONTENT_SIMILARITY_THRESHOLD,
    additional_output_paths: tuple[Path, ...] = (),
) -> dict[str, object] | None:
    """입력·설정이 같은 기존 실행 결과가 완전하면 요약을 반환한다."""
    required_outputs = [
        summary_path,
        model_path,
        model_checksum_path(model_path),
        prediction_output_path,
        review_workbook_path,
        *additional_output_paths,
    ]
    if not all(path.exists() for path in required_outputs):
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    expected = {
        "training_sha256": training_sha256,
        "prediction_sha256": prediction_sha256,
        "low_confidence_threshold": low_confidence_threshold,
        "n_splits": n_splits,
        "name_similarity_threshold": name_similarity_threshold,
        "content_similarity_threshold": content_similarity_threshold,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        return None

    summary["reused_existing"] = True
    return summary


def validate_prediction_targets(frame: pd.DataFrame) -> None:
    """QA·검토 Excel에 필요한 예측 대상 키 열을 학습 전에 검증한다."""
    required = ["연도", "지역", "원본행"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"예측 대상 필수 열 누락: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("예측 대상의 연도·지역·원본행에 결측이 있습니다.")


def run_training(
    training_path: Path,
    prediction_path: Path,
    output_dir: Path,
    model_path: Path,
    *,
    low_confidence_threshold: float = 0.5,
    n_splits: int = 5,
    force: bool = False,
) -> dict[str, object]:
    """모델 비교·최종 적합·예측·산출물 저장을 한 번에 수행한다.

    산출 파일명에는 대상 연도 범위를 넣지 않는다. 학습·예측 대상 연도
    범위는 실행마다 바뀔 수 있으므로(예: 2021년만 -> 2021~2024년,
    2016~2020+2022~2024 -> 2016~2020), 그 범위 정보는 파일명이 아니라
    ``output_dir`` 자체로 구분한다(예: `TFIDF_예측_2021_2024재학습/`).
    파일명을 연도에 고정하면 범위가 바뀔 때마다 이 함수도 같이 고쳐야
    한다.
    """
    summary_path = output_dir / "TFIDF_학습_예측_요약.json"
    prediction_output_path = output_dir / "TFIDF_영역분류_예측.csv"
    similarity_output_path = output_dir / "TFIDF_영역분류_유사사업순.csv"
    review_workbook_path = output_dir / "TFIDF_영역분류_검토.xlsx"
    low_confidence_output_path = output_dir / "TFIDF_저신뢰_검토대상.csv"
    prediction_qa_path = output_dir / "TFIDF_예측_QA.csv"
    evaluation_output_paths = (
        output_dir / "TFIDF_모델_비교.csv",
        low_confidence_output_path,
        prediction_qa_path,
        similarity_output_path,
        *(
            output_dir / f"TFIDF_{text_variant}_{suffix}.csv"
            for text_variant in TEXT_VARIANTS
            for suffix in ("클래스별_평가", "혼동행렬", "교차검증_예측")
        ),
    )
    training_sha256 = file_sha256(training_path)
    prediction_sha256 = file_sha256(prediction_path)

    if not force:
        reusable = load_reusable_summary(
            summary_path,
            model_path=model_path,
            prediction_output_path=prediction_output_path,
            review_workbook_path=review_workbook_path,
            training_sha256=training_sha256,
            prediction_sha256=prediction_sha256,
            low_confidence_threshold=low_confidence_threshold,
            n_splits=n_splits,
            additional_output_paths=evaluation_output_paths,
        )
        if reusable is not None:
            return reusable

    output_dir.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(training_path)
    targets = pd.read_csv(prediction_path)
    validate_prediction_targets(targets)

    evaluations = [
        evaluate_text_variant(training, text_variant, n_splits=n_splits)
        for text_variant in TEXT_VARIANTS
    ]
    best = choose_best_result(evaluations)
    model = fit_model(training, best.text_variant)
    predicted = predict(
        model,
        targets,
        best.text_variant,
        low_confidence_threshold=low_confidence_threshold,
    )

    metrics = pd.DataFrame(
        [{"입력방식": result.text_variant, **result.metrics} for result in evaluations]
    ).sort_values("f1_macro", ascending=False)
    metrics.to_csv(output_dir / "TFIDF_모델_비교.csv", index=False, encoding="utf-8-sig")

    for result in evaluations:
        result.class_report.to_csv(
            output_dir / f"TFIDF_{result.text_variant}_클래스별_평가.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result.confusion_matrix.to_csv(
            output_dir / f"TFIDF_{result.text_variant}_혼동행렬.csv",
            encoding="utf-8-sig",
        )
        result.predictions.to_csv(
            output_dir / f"TFIDF_{result.text_variant}_교차검증_예측.csv",
            index=False,
            encoding="utf-8-sig",
        )

    predicted.to_csv(prediction_output_path, index=False, encoding="utf-8-sig")
    predicted.loc[predicted["저신뢰_검토대상"]].sort_values("예측_신뢰도").to_csv(
        low_confidence_output_path,
        index=False,
        encoding="utf-8-sig",
    )
    qa = (
        predicted.groupby(["연도", "지역"], sort=False)
        .agg(
            행수=("원본행", "size"),
            세부영역_종류=("예측_세부영역", "nunique"),
            평균_신뢰도=("예측_신뢰도", "mean"),
            저신뢰_건수=("저신뢰_검토대상", "sum"),
        )
        .reset_index()
    )
    qa["저신뢰_비율"] = qa["저신뢰_건수"] / qa["행수"]
    qa.to_csv(
        prediction_qa_path,
        index=False,
        encoding="utf-8-sig",
    )
    grouped = assign_similarity_groups(predicted)
    grouped.to_csv(similarity_output_path, index=False, encoding="utf-8-sig")
    create_review_workbook(grouped, review_workbook_path, preserve_order=True)

    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "random_state": RANDOM_STATE,
        "training_file": str(training_path),
        "training_rows": len(training),
        "training_regions": int(training["지역"].nunique()),
        "label_count": int(training["세부영역"].nunique()),
        "prediction_file": str(prediction_path),
        "prediction_rows": len(targets),
        "cv_strategy": f"StratifiedGroupKFold(n_splits={n_splits}, groups=지역)",
        "low_confidence_threshold": low_confidence_threshold,
        "name_similarity_threshold": DEFAULT_NAME_SIMILARITY_THRESHOLD,
        "content_similarity_threshold": DEFAULT_CONTENT_SIMILARITY_THRESHOLD,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    save_model_bundle(
        model_path,
        model=model,
        text_variant=best.text_variant,
        metrics=best.metrics,
        metadata=metadata,
    )

    summary = {
        "selected_text_variant": best.text_variant,
        "selected_metrics": best.metrics,
        "prediction_rows": len(predicted),
        "low_confidence_rows": int(predicted["저신뢰_검토대상"].sum()),
        "low_confidence_threshold": low_confidence_threshold,
        "n_splits": n_splits,
        "name_similarity_threshold": DEFAULT_NAME_SIMILARITY_THRESHOLD,
        "content_similarity_threshold": DEFAULT_CONTENT_SIMILARITY_THRESHOLD,
        "training_sha256": training_sha256,
        "prediction_sha256": prediction_sha256,
        "model_sha256": model_checksum_path(model_path).read_text(encoding="ascii").strip(),
        "model_path": str(model_path),
        "review_workbook_path": str(review_workbook_path),
        "similarity_output_path": str(similarity_output_path),
        "reused_existing": False,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-path",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/2021/통합/2021_17개시도_TFIDF_학습데이터.csv"),
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=Path(
            "data/interim/영역분류_라벨링/TFIDF_분류대상/2016_2020_2022_2024_TFIDF_분류대상.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/영역분류_라벨링/TFIDF_예측"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/tfidf_area_classifier.joblib"),
    )
    parser.add_argument("--low-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="동일한 기존 산출물이 있어도 모델을 다시 학습하고 예측합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_training(
        args.training_path,
        args.prediction_path,
        args.output_dir,
        args.model_path,
        low_confidence_threshold=args.low_confidence_threshold,
        n_splits=args.n_splits,
        force=args.force,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
