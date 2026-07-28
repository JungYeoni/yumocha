"""저장된 TF-IDF 모델 번들로 신규 시행계획 데이터를 분류한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.tfidf_area_classifier import (
    RANDOM_STATE,
    TEXT_VARIANTS,
    load_model_bundle,
    predict,
)

np.random.seed(RANDOM_STATE)


def run_inference(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    low_confidence_threshold: float = 0.5,
) -> pd.DataFrame:
    """모델 번들을 불러와 신규 CSV에 영역 예측을 추가한다."""
    bundle = load_model_bundle(model_path)
    expected_columns = TEXT_VARIANTS.get(bundle["text_variant"])
    if expected_columns is None or tuple(bundle["text_columns"]) != expected_columns:
        raise ValueError(
            "번들의 텍스트 열 구성이 현재 TEXT_VARIANTS 정의와 다릅니다. "
            "현재 코드로 모델을 다시 학습하세요."
        )
    frame = pd.read_csv(input_path)
    predicted = predict(
        bundle["model"],
        frame,
        bundle["text_variant"],
        low_confidence_threshold=low_confidence_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(output_path, index=False, encoding="utf-8-sig")
    return predicted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted = run_inference(
        args.model_path,
        args.input_path,
        args.output_path,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    print(f"예측 행: {len(predicted):,}개")
    print(f"저신뢰 검토 대상: {int(predicted['저신뢰_검토대상'].sum()):,}개")
    print(f"저장: {args.output_path}")


if __name__ == "__main__":
    main()
