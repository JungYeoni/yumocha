"""이슈 #116 세부영역별 사업명·주요내용 핵심어와 워드클라우드를 생성한다."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from wordcloud import WordCloud

from scripts.build_provisional_area_labels import DEFAULT_WORKBOOK, SHEET_NAME, select_labels
from src.features.keyword_tfidf import (
    DEFAULT_STOPWORDS,
    compare_rankings,
    prepare_text_column,
    rank_group_keywords,
)
from src.visualization.plots import KOREAN_FONT

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "analysis" / "keyword_tfidf"
FIGURE_DIR = REPO_ROOT / "reports" / "figures" / "keyword_tfidf"
REPORT_PATH = REPO_ROOT / "reports" / "methodology" / "20260809_세부영역별_핵심단어_시각화_결과.md"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
)

np.random.seed(42)


def file_sha256(path: Path) -> str:
    """파일 SHA-256을 반환한다."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_input_path(path: Path) -> str:
    """저장소 내부 경로는 상대경로로, 외부 경로는 절대경로로 기록한다."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_korean_font(explicit: Path | None = None) -> Path:
    """워드클라우드에 사용할 한국어 글꼴 파일을 찾는다."""
    candidates = ((explicit,) if explicit else ()) + FONT_CANDIDATES
    for path in candidates:
        if path and path.is_file():
            return path
    raise FileNotFoundError("한국어 워드클라우드 글꼴을 찾지 못했습니다. --font-path를 지정하세요.")


def _frequency_by_group(ranking: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        group: dict(zip(part["단어"], part["평균_TFIDF"], strict=True))
        for group, part in ranking.groupby("세부영역", sort=True)
    }


def plot_wordcloud_grid(
    ranking: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    font_path: Path,
) -> None:
    """12개 세부영역 워드클라우드를 하나의 격자 그림으로 저장한다."""
    frequencies = _frequency_by_group(ranking)
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    for axis, group in zip(axes.flat, sorted(frequencies), strict=True):
        cloud = WordCloud(
            width=900,
            height=520,
            background_color="white",
            colormap="viridis",
            font_path=str(font_path),
            prefer_horizontal=0.9,
            random_state=42,
            collocations=False,
        ).generate_from_frequencies(frequencies[group])
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_title(group, fontsize=14, fontweight="bold")
        axis.axis("off")
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.995)
    fig.text(0.5, 0.006, "단어 크기는 세부영역 내 평균 TF-IDF에 비례함", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_bar_grid(ranking: pd.DataFrame, *, title: str, output_path: Path) -> None:
    """정확한 점수 비교를 위한 영역별 상위 10개 가로 막대그래프를 저장한다."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))
    for axis, (group, part) in zip(axes.flat, ranking.groupby("세부영역", sort=True), strict=True):
        shown = part.nsmallest(10, "순위").sort_values("평균_TFIDF")
        axis.barh(shown["단어"], shown["평균_TFIDF"], color=plt.cm.viridis(0.65))
        axis.set_title(group, fontsize=13, fontweight="bold")
        axis.set_xlabel("평균 TF-IDF")
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_report(metadata: dict[str, object]) -> None:
    fallback_note = (
        f"{metadata['주요내용_원문대체']:,}행"
        if metadata["주요내용_원문열존재"]
        else f"{metadata['주요내용_원문대체']:,}행(입력 워크북에 원문 열이 없어 대체 미적용)"
    )
    report = f"""# 세부영역별 세부사업명·주요내용 핵심 단어 시각화 결과

- 기준일: 2026-08-09
- 관련 이슈: #116
- 입력: `{metadata["입력파일"]}`
- 입력 SHA-256: `{metadata["입력_SHA256"]}`
- 대상: 2016–2024년 17개 시도, {metadata["전체행"]:,}행, 세부영역 12종
- 라벨: 검토 확정·수정 {metadata["검토_확정수정"]:,}행, 비고 기반 확정 {metadata["검토_비고기반확정"]:,}행

## 방법

각 세부사업 행을 문서 하나로 보고 `kiwipiepy`의 일반명사·고유명사 중 두 글자 이상을 추출했다. 세부사업명과 주요내용은 서로 섞지 않고 별도의 말뭉치로 TF-IDF를 계산했다. 설정은 `min_df=2`, `max_df=0.95`, `sublinear_tf=True`, L2 정규화다. 사업별 TF-IDF를 세부영역별 평균으로 집계하고 상위 20개 단어를 제시했다.

이 작업은 2016–2024년 전체 적격 말뭉치의 특징을 요약하는 기술 분석이며 예측 실험이 아니다. 따라서 train/validation/test 분할, 예측 평가지표 및 교차검증(CV) fold는 적용하지 않았다.

정적 불용어는 예산 문서 전반에 반복되는 {metadata["불용어수"]}개 단어만 사용했다. 목록은 다음과 같다.

`{", ".join(metadata["불용어"])}`

## 입력 및 결측

- 세부사업명: 전체 {metadata["전체행"]:,}행 분석
- 주요내용_정제 사용: {metadata["주요내용_우선텍스트사용"]:,}행
- 원문 대체: {fallback_note}
- 주요내용 분석 제외: {metadata["주요내용_분석제외"]:,}행

## 산출물

- [세부사업명 워드클라우드](../figures/keyword_tfidf/2016-2024_세부영역별_세부사업명_워드클라우드.png)
- [주요내용 워드클라우드](../figures/keyword_tfidf/2016-2024_세부영역별_주요내용_워드클라우드.png)
- 정확한 점수 비교용 가로 막대그래프 2종과 영역별 상위 20개 CSV 2종
- 중복 텍스트 제거 민감도 CSV 2종과 영역별 순위 중첩·상관 요약

## 자동 품질 기준

- 12개 세부영역 모두 사업명·주요내용 상위 단어가 존재한다.
- 각 영역 결과에 최소 15개 단어가 존재한다.
- `지원`, `사업`, `운영`, `확대`는 상위 결과에 0건이다.
- 기본 결과와 중복 제거 결과의 상위어 중첩률·순위상관을 저장한다.

## 한계

TF-IDF는 특정 영역에서 상대적으로 두드러지는 단어를 보여줄 뿐 정책 중요도나 효과를 측정하지 않는다. 세부영역별 표본 수, 반복 사업, 형태소 분석의 복합명사 분리와 기존 라벨 오류가 결과에 영향을 줄 수 있다. 중복 제거 결과는 이런 반복 기록 민감도를 확인하는 보조 결과다.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


def build_outputs(workbook_path: Path, *, font_path: Path) -> dict[str, object]:
    """입력 워크북에서 모든 표·그림·보고서를 생성한다."""
    review = pd.read_excel(workbook_path, sheet_name=SHEET_NAME, engine="openpyxl")
    labeled = select_labels(review)
    labeled["세부사업명_분석"] = prepare_text_column(labeled, preferred_column="세부사업명")[0]
    content, content_stats = prepare_text_column(
        labeled,
        preferred_column="주요내용_정제",
        fallback_column="주요내용",
    )
    labeled["주요내용_분석"] = content

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = KOREAN_FONT
    kiwi = Kiwi()
    summaries = []
    # 예측이 아닌 2016–2024년 기술적 TF-IDF 요약이므로 분할 없이 전체 적격 말뭉치를 쓴다.
    # 반복 사업과 영역별 표본 수 차이는 아래 중복 제거 민감도 및 보고서 한계에 기록한다.
    for field, column, label in (
        ("세부사업명", "세부사업명_분석", "세부사업명"),
        ("주요내용", "주요내용_분석", "주요내용_정제"),
    ):
        baseline = rank_group_keywords(labeled, text_column=column, kiwi=kiwi)
        sensitivity = rank_group_keywords(labeled, text_column=column, kiwi=kiwi, deduplicate=True)
        comparison = compare_rankings(baseline, sensitivity)
        baseline.to_csv(
            OUTPUT_DIR / f"2016-2024_세부영역별_{field}_TFIDF_상위단어.csv",
            index=False,
            encoding="utf-8-sig",
        )
        sensitivity.to_csv(
            OUTPUT_DIR / f"2016-2024_세부영역별_{field}_TFIDF_중복제거_상위단어.csv",
            index=False,
            encoding="utf-8-sig",
        )
        comparison.to_csv(
            OUTPUT_DIR / f"2016-2024_세부영역별_{field}_TFIDF_중복민감도.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_wordcloud_grid(
            baseline,
            title=f"세부영역별 {label} 핵심 단어",
            output_path=FIGURE_DIR / f"2016-2024_세부영역별_{field}_워드클라우드.png",
            font_path=font_path,
        )
        plot_bar_grid(
            baseline,
            title=f"세부영역별 {label} 상위 단어",
            output_path=FIGURE_DIR / f"2016-2024_세부영역별_{field}_상위단어_막대그래프.png",
        )
        summaries.append(comparison.assign(텍스트구분=field))

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(
        OUTPUT_DIR / "2016-2024_세부영역별_TFIDF_중복민감도_요약.csv",
        index=False,
        encoding="utf-8-sig",
    )
    label_counts = labeled["라벨출처"].value_counts()
    metadata = {
        "입력파일": display_input_path(workbook_path),
        "입력_SHA256": file_sha256(workbook_path),
        "전체행": len(labeled),
        "검토_확정수정": int(label_counts.get("검토_확정수정", 0)),
        "검토_비고기반확정": int(label_counts.get("검토_비고기반확정", 0)),
        "주요내용_우선텍스트사용": content_stats["우선텍스트사용"],
        "주요내용_원문대체": content_stats["원문대체"],
        "주요내용_원문열존재": content_stats["원문열존재"],
        "주요내용_분석제외": content_stats["분석제외"],
        "불용어수": len(DEFAULT_STOPWORDS),
        "불용어": sorted(DEFAULT_STOPWORDS),
    }
    (OUTPUT_DIR / "2016-2024_세부영역별_TFIDF_실행메타데이터.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(metadata)
    return metadata


def main() -> int:
    """명령행 인수를 처리하고 #116 산출물을 생성한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--font-path", type=Path)
    args = parser.parse_args()
    metadata = build_outputs(args.workbook.resolve(), font_path=resolve_korean_font(args.font_path))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
