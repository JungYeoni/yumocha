"""2016-2020년 TF-IDF 예측 대상 데이터를 취합한다.

2021-2024년이 확정 라벨(학습 데이터)로 편입되면서 예측 대상은
2016-2020년만 남는다. 취합·검증 로직은 8개년용
`consolidate_tfidf_prediction_targets.py`의 `consolidate_prediction_targets`를
그대로 재사용하고, 연도 범위와 출력 파일명만 다르게 한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.consolidate_2021_area_labels import REGION_ORDER
from scripts.consolidate_tfidf_prediction_targets import (
    PREDICTION_COLUMNS,
    consolidate_prediction_targets,
)

PREDICTION_YEARS = [2016, 2017, 2018, 2019, 2020]


def save_outputs(combined, qa, output_dir: Path) -> dict[str, Path]:
    """통합본·분류 핵심본·QA를 UTF-8-SIG CSV로 저장한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "통합본": output_dir / "2016_2020_TFIDF_분류대상_통합.csv",
        "분류본": output_dir / "2016_2020_TFIDF_분류대상.csv",
        "QA": output_dir / "2016_2020_TFIDF_분류대상_QA.csv",
    }
    combined.to_csv(paths["통합본"], index=False, encoding="utf-8-sig")
    combined[PREDICTION_COLUMNS].to_csv(paths["분류본"], index=False, encoding="utf-8-sig")
    qa.to_csv(paths["QA"], index=False, encoding="utf-8-sig")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/interim"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/영역분류_라벨링/TFIDF_분류대상"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined, qa = consolidate_prediction_targets(
        args.source_dir, years=PREDICTION_YEARS, regions=REGION_ORDER
    )
    paths = save_outputs(combined, qa, args.output_dir)

    print(f"통합 행: {len(combined):,}개")
    print(f"연도·지역 조합: {combined[['연도', '지역']].drop_duplicates().shape[0]}개")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
