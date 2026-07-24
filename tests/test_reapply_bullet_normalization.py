from pathlib import Path
import unicodedata

import pandas as pd

from scripts.reapply_bullet_normalization import reapply_bullet_normalization


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def test_reapply_bullet_normalization_dry_run_and_write_with_backup(tmp_path):
    region_dir = tmp_path / "서울"
    wide_path = region_dir / unicodedata.normalize("NFD", "2022_서울_세부사업_정제.csv")
    long_path = region_dir / "2022_서울_세부사업_정제_long.csv"
    row = {
        "원본행": 10,
        "세부사업명": "사업 A",
        "주요내용": "지원대상 : 시민 • 지원내용 : 상담 3회",
        "주요내용_정제": "시민 상담 • 3회",
    }
    _write_csv(wide_path, [row])
    _write_csv(long_path, [row, row])

    dry_run = reapply_bullet_normalization(tmp_path, years=(2022,))

    assert dry_run["검사파일"] == 2
    assert dry_run["변경파일"] == 2
    assert dry_run["변경셀"] == 6
    assert "•" in pd.read_csv(wide_path, encoding="utf-8-sig").loc[0, "주요내용"]

    written = reapply_bullet_normalization(tmp_path, years=(2022,), write=True)

    assert written["변경파일"] == 2
    assert written["백업경로"] is not None
    normalized = pd.read_csv(wide_path, encoding="utf-8-sig")
    assert normalized.loc[0, "주요내용"] == "지원대상 : 시민 지원내용 : 상담 3회"
    assert normalized.loc[0, "주요내용_정제"] == "시민 상담 3회"

    backup_dir = Path(str(written["백업경로"]))
    backup = pd.read_csv(backup_dir / "서울" / wide_path.name, encoding="utf-8-sig")
    assert "•" in backup.loc[0, "주요내용"]
