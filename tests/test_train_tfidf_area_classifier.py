"""TF-IDF 학습 스크립트의 기존 결과 재사용 테스트."""

from __future__ import annotations

import json

from scripts.train_tfidf_area_classifier import file_sha256, load_reusable_summary


def test_load_reusable_summary_reuses_only_matching_complete_outputs(tmp_path):
    training = tmp_path / "training.csv"
    prediction = tmp_path / "prediction.csv"
    model = tmp_path / "model.joblib"
    prediction_output = tmp_path / "predicted.csv"
    workbook = tmp_path / "review.xlsx"
    summary_path = tmp_path / "summary.json"
    training.write_text("training", encoding="utf-8")
    prediction.write_text("prediction", encoding="utf-8")
    model.touch()
    prediction_output.touch()
    workbook.touch()
    summary = {
        "training_sha256": file_sha256(training),
        "prediction_sha256": file_sha256(prediction),
        "low_confidence_threshold": 0.5,
        "n_splits": 5,
        "reused_existing": False,
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    reused = load_reusable_summary(
        summary_path,
        model_path=model,
        prediction_output_path=prediction_output,
        review_workbook_path=workbook,
        training_sha256=file_sha256(training),
        prediction_sha256=file_sha256(prediction),
        low_confidence_threshold=0.5,
        n_splits=5,
    )

    assert reused is not None
    assert reused["reused_existing"] is True


def test_load_reusable_summary_rejects_changed_input(tmp_path):
    paths = [tmp_path / name for name in ["model", "prediction", "workbook"]]
    for path in paths:
        path.touch()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "training_sha256": "old",
                "prediction_sha256": "same",
                "low_confidence_threshold": 0.5,
                "n_splits": 5,
            }
        ),
        encoding="utf-8",
    )

    reused = load_reusable_summary(
        summary_path,
        model_path=paths[0],
        prediction_output_path=paths[1],
        review_workbook_path=paths[2],
        training_sha256="new",
        prediction_sha256="same",
        low_confidence_threshold=0.5,
        n_splits=5,
    )

    assert reused is None
