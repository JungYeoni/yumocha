"""연도별 전국 QA 검증결과 CSV에 `판정불가_사유` 컬럼을 재적용한다.

`reports/yearly/{연도}/{연도}_전국_QA_검증결과.csv`는 이미 `build_subtotal_qa()`가
만드는 `QA_병합상태`, `원본_소계값`(또는 `QA_비교_소계값`), `leaf_합계` 컬럼을
갖고 있어 노트북을 재실행하지 않고도 `assign_qa_undetermined_reason()`으로
사유를 계산해 붙일 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline_common import assign_qa_undetermined_reason  # noqa: E402

DEFAULT_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)


def annotate_subtotal_qa_reason(
    reports_dir: Path,
    *,
    years: tuple[int, ...] = DEFAULT_YEARS,
) -> dict[int, pd.Series]:
    """대상 QA CSV에 `판정불가_사유`를 채우고 연도별 사유 집계를 반환한다."""
    counts_by_year: dict[int, pd.Series] = {}

    for year in years:
        path = reports_dir / "yearly" / str(year) / f"{year}_전국_QA_검증결과.csv"
        if not path.exists():
            raise FileNotFoundError(f"QA 결과 파일이 없습니다: {path}")

        qa = pd.read_csv(path, encoding="utf-8-sig")
        required_cols = {"QA_병합상태", "leaf_합계", "결과", "허용기준결과"}
        missing_cols = required_cols.difference(qa.columns)
        if missing_cols:
            raise KeyError(f"{path}에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

        comparison_col = "QA_비교_소계값" if "QA_비교_소계값" in qa.columns else "원본_소계값"
        qa["판정불가_사유"] = assign_qa_undetermined_reason(qa, comparison_col)

        undetermined = qa["결과"].eq("판정불가") | qa["허용기준결과"].eq("판정불가")
        has_reason = qa["판정불가_사유"].notna()
        if not undetermined.equals(has_reason):
            mismatched = qa.loc[undetermined != has_reason, ["지역", "대분류", "중분류"]]
            raise ValueError(
                f"{path}: 판정불가 여부와 사유 존재 여부가 일치하지 않습니다: "
                f"{mismatched.to_dict(orient='records')}"
            )

        columns = [column for column in qa.columns if column != "판정불가_사유"]
        insert_at = columns.index("허용기준결과") + 1
        columns.insert(insert_at, "판정불가_사유")
        qa = qa[columns]

        qa.to_csv(path, index=False, encoding="utf-8-sig")
        counts_by_year[year] = qa["판정불가_사유"].value_counts(dropna=True)

    return counts_by_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
    )
    args = parser.parse_args()

    counts_by_year = annotate_subtotal_qa_reason(
        args.reports_dir.resolve(),
        years=tuple(args.years),
    )
    for year, counts in counts_by_year.items():
        print(f"--- {year} ---")
        if counts.empty:
            print("  판정불가 없음")
        for reason, count in counts.items():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
