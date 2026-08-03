"""TF-IDF 영역분류 학습·추론 유틸리티 테스트."""

from __future__ import annotations

import hashlib

import joblib
import pandas as pd
import pytest

from src.modeling.tfidf_area_classifier import (
    build_text,
    choose_best_result,
    evaluate_text_variant,
    fit_model,
    load_model_bundle,
    model_checksum_path,
    predict,
    save_model_bundle,
    validate_training_data,
)


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "지역": ["서울", "서울", "부산", "부산"],
            "원본행": [1, 2, 1, 2],
            "세부사업명": ["청년 취업", "아이 돌봄", "일자리 지원", "공동육아"],
            "주요내용_정제": ["고용 지원", pd.NA, "취업 알선", "돌봄 서비스"],
            "대영역": ["1. 경제·고용·주거", "2. 가족·생활"] * 2,
            "세부영역": ["1-1. 고용여건", "2-1. 돌봄 여건"] * 2,
        }
    )


def test_build_text_uses_project_name_when_content_is_missing():
    frame = _training_frame()

    text = build_text(frame, "사업명_주요내용")

    assert text.iloc[1] == "아이 돌봄"


def test_validate_training_data_rejects_taxonomy_mismatch():
    frame = _training_frame()
    frame.loc[0, "대영역"] = "2. 가족·생활"

    with pytest.raises(ValueError, match="taxonomy"):
        validate_training_data(frame)


def test_build_text_rejects_blank_project_name():
    frame = _training_frame()
    frame.loc[0, "세부사업명"] = " "

    with pytest.raises(ValueError, match="세부사업명"):
        build_text(frame, "사업명")


def test_validate_training_data_rejects_duplicate_key_without_year():
    frame = _training_frame()
    frame.loc[1, ["지역", "원본행"]] = ["서울", 1]

    with pytest.raises(ValueError, match="지역·원본행 키가 중복"):
        validate_training_data(frame)


def test_validate_training_data_allows_same_region_row_across_different_years():
    frame = _training_frame()
    frame.insert(0, "연도", [2021, 2021, 2022, 2022])
    frame.loc[2, ["지역", "원본행"]] = ["서울", 1]

    validate_training_data(frame)


def test_validate_training_data_rejects_missing_year_when_year_column_present():
    frame = _training_frame()
    frame.insert(0, "연도", [2021, 2021, None, 2022])

    with pytest.raises(ValueError, match="결측"):
        validate_training_data(frame)


def test_validate_training_data_rejects_duplicate_key_with_year():
    frame = _training_frame()
    frame.insert(0, "연도", [2021, 2021, 2021, 2021])
    frame.loc[1, ["지역", "원본행"]] = ["서울", 1]

    with pytest.raises(ValueError, match="연도·지역·원본행 키가 중복"):
        validate_training_data(frame)


def test_choose_best_result_rejects_empty_results():
    with pytest.raises(ValueError, match="비교할 평가 결과"):
        choose_best_result([])


def test_fit_predict_and_model_bundle_round_trip(tmp_path):
    training = _training_frame()
    model = fit_model(training, "사업명_주요내용")
    targets = training[["지역", "원본행", "세부사업명", "주요내용_정제"]].copy()

    predicted = predict(model, targets, "사업명_주요내용", low_confidence_threshold=0.6)

    assert predicted["예측_세부영역"].notna().all()
    assert predicted["예측_대영역"].notna().all()
    assert predicted["예측_신뢰도"].between(0, 1).all()
    assert predicted["저신뢰_검토대상"].dtype == bool

    path = tmp_path / "model.joblib"
    save_model_bundle(
        path,
        model=model,
        text_variant="사업명_주요내용",
        metrics={"f1_macro": 0.5},
        metadata={"training_rows": len(training)},
    )
    loaded = load_model_bundle(path)

    assert loaded["text_variant"] == "사업명_주요내용"
    assert loaded["metadata"]["training_rows"] == 4
    assert model_checksum_path(path).exists()


@pytest.mark.parametrize("threshold", [0, 1, -0.1, 1.1])
def test_predict_rejects_threshold_outside_open_unit_interval(threshold):
    with pytest.raises(ValueError, match="0과 1 사이"):
        predict(object(), _training_frame(), "사업명", low_confidence_threshold=threshold)


def test_load_model_bundle_rejects_missing_required_key(tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump({"model": "missing other keys"}, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    model_checksum_path(path).write_text(checksum + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="올바른 TF-IDF 모델 번들"):
        load_model_bundle(path)


def test_load_model_bundle_rejects_checksum_mismatch(tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump({"model": "untrusted"}, path)
    model_checksum_path(path).write_text("0" * 64 + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="체크섬"):
        load_model_bundle(path)


def test_evaluate_text_variant_returns_grouped_cv_predictions():
    rows = []
    for index, region in enumerate(["서울", "부산", "대구", "인천", "광주", "대전"]):
        rows.extend(
            [
                {
                    "지역": region,
                    "원본행": index * 2 + 1,
                    "세부사업명": f"{region} 청년 취업 지원",
                    "주요내용_정제": "일자리 고용 서비스",
                    "대영역": "1. 경제·고용·주거",
                    "세부영역": "1-1. 고용여건",
                },
                {
                    "지역": region,
                    "원본행": index * 2 + 2,
                    "세부사업명": f"{region} 아이 돌봄 지원",
                    "주요내용_정제": "공동육아 돌봄 서비스",
                    "대영역": "2. 가족·생활",
                    "세부영역": "2-1. 돌봄 여건",
                },
            ]
        )
    frame = pd.DataFrame(rows)

    result = evaluate_text_variant(frame, "사업명_주요내용", n_splits=3)

    assert len(result.predictions) == len(frame)
    assert set(result.metrics) == {"accuracy", "f1_macro", "f1_weighted"}
    assert result.predictions["예측_신뢰도"].between(0, 1).all()
