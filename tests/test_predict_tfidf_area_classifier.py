"""저장 모델 기반 TF-IDF 추론 스크립트 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.predict_tfidf_area_classifier as inference


def test_run_inference_rejects_text_column_contract_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        inference,
        "load_model_bundle",
        lambda path: {
            "model": object(),
            "text_variant": "사업명_주요내용",
            "text_columns": ("세부사업명",),
        },
    )

    def fail_if_read(path):
        raise AssertionError("입력 CSV를 읽기 전에 번들 계약을 검증해야 합니다.")

    monkeypatch.setattr(pd, "read_csv", fail_if_read)

    with pytest.raises(ValueError, match="텍스트 열 구성"):
        inference.run_inference(
            tmp_path / "model.joblib",
            tmp_path / "input.csv",
            tmp_path / "output.csv",
        )
