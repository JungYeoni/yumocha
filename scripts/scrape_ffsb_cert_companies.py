# 2026-08-06
"""가족친화지원사업(ffsb.kr) 인증기업 정보 검색 페이지에서 연도별 신규인증기업 목록을 수집한다."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, NavigableString

SEARCH_URL = "https://www.ffsb.kr/ffm/ffmCertCompSearch.do"
MENU_SEQ = "1152"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}
REQUEST_INTERVAL_SEC = 0.5  # 서버 부하 방지용 페이지 간 대기 시간
REQUEST_TIMEOUT_SEC = 30
COLUMNS = ["신규인증년도", "지역", "기업(관)명", "구분"]


def fetch_page(year: str, page_no: int, session: requests.Session) -> str:
    """지정 연도·페이지 번호의 검색 결과 HTML을 가져온다."""
    payload = {
        "pageNo": str(page_no),
        "keyWord1": year,
        "keyWord2": "",
        "keyWord3": "",
        "keyWord4": "",
        "keyWord5": "",
        "keyKind": "",
        "awardTypeCd": "",
    }
    response = session.post(
        f"{SEARCH_URL}?menuSeq={MENU_SEQ}",
        data=payload,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.text


def parse_total_pages(html: str) -> int:
    """페이지네이션 영역의 '끝으로' 링크에서 마지막 페이지 번호를 추출한다."""
    soup = BeautifulSoup(html, "html.parser")
    last_link = soup.select_one("p.pagings1 a.last")
    if last_link is None:
        return 1
    match = re.search(r"fnGoPage\('(\d+)'", last_link.get("href", ""))
    return int(match.group(1)) if match else 1


def _cell_value(td) -> str:
    """<span> 라벨 뒤에 붙은 실제 값 텍스트만 추출한다 (예: '지역: 서울' -> '서울')."""
    text_nodes = [node for node in td.contents if isinstance(node, NavigableString)]
    return str(text_nodes[-1]).strip() if text_nodes else ""


def parse_rows(html: str) -> list[dict]:
    """검색 결과 테이블에서 신규인증년도·지역·기업(관)명·구분을 추출한다."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    for tr in soup.select(".list_table table tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        rows.append(dict(zip(COLUMNS, (_cell_value(td) for td in cells[:4]))))
    return rows


def crawl_year(year: str, session: requests.Session) -> pd.DataFrame:
    """지정 연도의 신규인증기업 목록을 전체 페이지에 걸쳐 수집한다."""
    first_html = fetch_page(year, 1, session)
    total_pages = parse_total_pages(first_html)
    all_rows = parse_rows(first_html)

    for page_no in range(2, total_pages + 1):
        time.sleep(REQUEST_INTERVAL_SEC)
        html = fetch_page(year, page_no, session)
        all_rows.extend(parse_rows(html))

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    unexpected_years = set(df["신규인증년도"]) - {year}
    if unexpected_years:
        raise ValueError(f"{year}년 조회 결과에 다른 연도 혼입: {unexpected_years}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="ffsb.kr 인증기업 정보 연도별 크롤링")
    parser.add_argument("--years", nargs="+", default=["2017", "2019"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/지표별_원데이터/4-1.3. 가족친화인증기업 비율"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for year in args.years:
        df = crawl_year(year, session)
        output_path = args.output_dir / f"가족친화인증기업_ffsb크롤링_{year}.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"{year}년: {len(df)}건 저장 -> {output_path}")


if __name__ == "__main__":
    main()
