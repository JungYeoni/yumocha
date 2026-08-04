# yumocha — 출생환경지표 분석 프로젝트

전국 17개 시도의 **저출생 대응 시행계획 세부사업 예산**을 정제·분류하고, **구조환경지표**·**합계출산율**과 연결해 분석하는 저장소입니다.

원본 예산 자료는 연도마다 형식이 다른 단일 대형 시트로 제공되기 때문에, 이 저장소의 작업 대부분은 다음 두 가지에 집중되어 있습니다.

1. 연도·시도별로 흩어진 원본을 **검증 가능한 형태로 정제**하는 일
2. 정제된 세부사업을 **출생환경 영역별로 분류**해 지표와 대응시키는 일

분석 절차, 산출물 위치, 방법론 판단 근거는 모두 `reports/`에 문서로 남기는 것을 원칙으로 합니다.

---

## 데이터 출처와 범위

| 데이터 | 범위 | 위치 | 비고 |
|--------|------|------|------|
| 시행계획 세부사업 예산현황 | 2016~2024년 9개년 × 17개 시도 | `data/raw/칼럼정렬/{연도}_칼럼정렬.xlsx` | 데이터수집팀이 컬럼을 정렬한 `정리본_자동` 시트를 표준 입력으로 사용 |
| 원본 시행계획 (교차검증용) | 동일 | `data/raw/` | 시트가 `Table 1` 하나뿐이며 17개 시도가 이어 붙은 롱 테이블 |
| 재정팀 영역분류 라벨·검토본 | 2021~2024년 | `data/interim/영역분류_라벨링/` | 사람이 검토·확정한 라벨. 모델 학습 데이터의 근거 |
| 구조환경지표 21종 | 지역 × 연도 | `data/interim/` | 재정팀 산출값을 원자료로 재현해 전수 검증 |
| 지표 산출용 조사 원자료 | 연도별 | `data/raw/` | 지역별고용조사, 가족실태조사, 주거실태조사 등 |
| 보조 매핑 테이블 | — | `data/lookup/` | 시도-지역코드 매핑, 시도·연도별 시군구 수, 소비자물가지수 (git 추적 대상) |

**데이터 취급 규칙**

- `data/raw/`, `data/interim/`, `data/processed/`는 `.gitignore`에서 제외되어 **git으로 추적하지 않습니다.** 디렉터리 유지용 `.gitkeep`만 예외입니다. 원본과 중간 산출물은 Google Drive로 공유합니다.
- `data/lookup/`의 소형 참조 CSV만 저장소에 포함됩니다.
- 원본 파일은 수정하지 않고, 처리 결과만 `data/interim/` 또는 `data/processed/`에 씁니다.

**범위상 주의할 점**

- **2016~2019년**은 세부사업 한 건이 `계 / 국비 / 지방비` 3행으로 나뉘어 있어 `계` 행만 예산으로 사용합니다.
- **2022년**은 `data/raw/` 최상위에 표준과 다른 별도 가공 시트가 섞여 있어 입력 경로를 반드시 확인해야 합니다.
- **2025년**은 원본 미수집 상태로 아직 범위 밖입니다.

---

## 분석 파이프라인

작업은 재정팀 로드맵의 **Level 1(시도별 분리·정제)** → **Level 2(세부사업 영역분류)** 순서로 진행되며, Level 2는 Level 1 산출물을 입력으로 받습니다.

### 1단계 — 시도별 분리·정제 (Level 1)

단일 롱 테이블을 시도별로 분리하고, 대분류/중분류/세부사업(leaf) 계층을 판별한 뒤, leaf 예산 합계와 소계값을 대조하는 QA를 수행합니다.

- 연도별 노트북: `notebooks/{날짜}_EDA_{연도}_전국_시도분리정제.ipynb` (2016~2024)
- 공통 유틸: `src/features/pipeline_common.py` (`classify_row`, `flatten_header` 등)
- 노트북 템플릿: `notebooks/templates/시행계획_시도분리정제_공통유틸_템플릿.ipynb`
- 계획 문서: `reports/planning/20260712_시도별_예산현황_분리정제_작업계획.md`

### 2단계 — 텍스트 정규화와 LLM 보존형 정제

세부사업명·주요내용의 불릿과 줄바꿈을 정규화하고, LLM 교정 결과가 **원문 의미를 훼손하지 않았는지** 별도 감사(audit) 절차로 검증합니다.

- 모듈: `src/features/bullet_normalization.py`, `src/features/llm_refine.py`, `src/features/text_patterns.py`
- 스크립트: `scripts/apply_{연도}_semantic_corrections.py`, `scripts/audit_llm_semantic_preservation.py`, `scripts/reapply_bullet_normalization.py`

### 3단계 — 검토 워크북 생성과 사람 검토

전국 세부사업을 명칭·내용 유사도로 묶어 재정팀이 라벨링하기 좋은 Excel 워크북으로 내보냅니다.

- 모듈: `src/modeling/similarity_grouping.py`, `src/modeling/tfidf_review_workbook.py`, `src/features/text_match.py`, `src/features/review_keys.py`
- 스크립트: `scripts/build_finance_review_master.py`, `scripts/build_grouped_finance_review_workbook.py`, `scripts/export_finance_review_workbook.py`

### 4단계 — 세부사업 영역분류 (Level 2)

확정 라벨을 학습 데이터로 모아 TF-IDF 분류기를 학습하고, 라벨이 없는 연도를 예측합니다.

```bash
uv run python scripts/consolidate_2021_2024_confirmed_labels.py   # 확정 라벨 취합
uv run python scripts/consolidate_2016_2020_prediction_targets.py # 예측 대상 취합
uv run python scripts/train_tfidf_area_classifier.py              # 학습·평가
uv run python scripts/predict_tfidf_area_classifier.py            # 추론
```

- 모듈: `src/modeling/tfidf_area_classifier.py`
- 계획 문서: `reports/planning/20260715_세부사업_영역분류_작업계획.md`
- 최신 결과: `reports/20260803_2021_2024_통합_라벨_TFIDF_재학습_결과.md`

계획 범위와 실제 처리 범위가 다릅니다.

- **계획 범위**: 제4차 기본계획 기간인 2021~2025년
- **실제 처리 범위**: 2016~2024년 — 2021~2024년 확정 라벨로 학습하고 2016~2020년을 예측했습니다.
- **2025년은 제외**되어 있습니다. 원본 자료가 아직 수집되지 않아 정제·분류 어느 단계에도 들어가지 않았습니다.

### 5단계 — 구조환경지표 검증

재정팀이 산출한 지표값을 원자료로 재현해 지역 × 연도 단위로 전수 대조합니다. 결측·중복이 조용히 넘어가지 않도록 검증 유틸을 별도로 두었습니다.

- 모듈: `src/evaluation/structural_validation.py`
- 노트북: `notebooks/20260724_EDA_구조환경지표_21개_원자료기반_전수검증.ipynb` 외
- 결과: `reports/20260724_구조환경지표_21개_검증_진행상황.md`, `reports/20260727_구조환경지표별_17개시도_상세결과.md`

### 6단계 — 기초패널 생성과 추세 EDA

지역 × 연도 계획예산과 합계출산율을 결합한 기초패널을 만들고 추세를 확인합니다. 예산 결측은 0으로 추정하지 않고 합계에서 제외하되, 결측 건수를 품질정보로 함께 보존합니다.

- 모듈: `src/features/analysis_panel.py`, `src/features/trend_eda.py`, `src/visualization/trends.py`
- 노트북: `notebooks/20260726_EDA_2016_2024_계획예산_합계출산율_기초패널_생성.ipynb`

---

## 저장소 구조

```text
yumocha/
├── configs/                      # 공통·환경별 설정 (경로, seed, 분할 기준)
│
├── data/
│   ├── raw/                      # 원본 데이터, git 추적 제외
│   ├── interim/                  # 중간 처리 데이터, git 추적 제외
│   ├── processed/                # 분석 입력용 최종 데이터, git 추적 제외
│   └── lookup/                   # 소형 참조 테이블 (git 추적)
│
├── notebooks/                    # 연도별 EDA·모델링 노트북
│   ├── templates/                # 연도 확장용 노트북 템플릿
│   └── archive/                  # 대체된 과거 노트북
│
├── reports/                      # 작업 결과·계획·방법론 문서 (reports/README.md 참고)
│   ├── yearly/{연도}/            # 연도별 EDA 보고서와 QA·LLM 검토 CSV
│   ├── methodology/              # 연도에 매이지 않는 공통 방법론
│   ├── planning/                 # 착수 전 작업계획
│   └── archive/                  # 보존용 과거 자료
│
├── scripts/                      # 실행 단위 파이프라인 스크립트
│
├── src/
│   ├── features/                 # 정제·텍스트 처리·패널 생성 유틸
│   ├── modeling/                 # 영역분류 모델과 검토 워크북 생성
│   ├── evaluation/               # 지표 검증과 평가
│   └── visualization/            # 시각화
│
├── tests/                        # 단위 테스트
├── .github/                      # 이슈·PR 템플릿, CI 워크플로우
└── .claude/                      # Claude Code 규칙·에이전트·커맨드 설정
```

---

## 개발 환경

Python 3.11 이상과 `uv` 기반 의존성 관리를 사용합니다.

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

`pip`만 사용할 수 있는 환경:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest tests/ -v
```

의존성은 `pyproject.toml`에 정의하고 `uv.lock`으로 잠급니다. `requirements.txt`는 외부 배포·호환용 보조 파일입니다.

```bash
uv add pandas          # 런타임 의존성 추가
uv add --dev pytest    # 개발 의존성 추가
```

---

## 품질 확인

커밋하거나 PR을 열기 전에 실행합니다.

```bash
uv run --extra dev ruff check src/ tests/
uv run --extra dev ruff format --check src/ tests/
uv run --extra dev pytest tests/ -v
```

---

## 작업 원칙

### 재현성

- 모든 분석에 random seed를 고정하고, 데이터 분할 기준을 코드나 문서에 남깁니다.
- 결과 문서에는 입력 파일 경로와 행 수, 가능하면 해시를 함께 기록합니다.
- 스크립트로 재현 가능한 작업은 노트북이 아니라 `scripts/`에 둡니다.

### 데이터 누수 방지

- train/validation/test 분리 후 전처리 기준을 학습 데이터에서만 계산합니다.
- 인코더·스케일러·imputing 파라미터는 학습 데이터에만 fit합니다.
- 시계열 rolling/lag 피처는 `shift(1)` 이후 계산합니다.

### 재발명 금지

- 새 정제·분류 로직을 만들기 전에 `src/`와 기존 노트북, `reports/`의 계획 문서에 같은 목적의 함수가 있는지 먼저 확인합니다.
- 특히 세부사업명 정제(`classify_row`, `flatten_header` 등)는 여러 작업에서 공통으로 재사용하는 대상입니다.
- 기존 로직을 그대로 쓸 수 없다면 그 이유를 주석이나 문서에 남깁니다.

### 설명 가능한 결과

- 성능 수치뿐 아니라 데이터 가정, 한계, 실패 사례를 함께 기록합니다.
- 복잡한 모델보다 단순한 baseline을 먼저 만듭니다.
- QA 불일치는 숨기지 않고 원인을 규명해 문서에 남깁니다.

---

## 작업 흐름

1. GitHub Issue에 목표, 대상 데이터, 성공 기준을 적습니다.
2. 브랜치를 만듭니다. (`git checkout -b feat/{날짜}_#{이슈번호}_{요약}`)
3. 탐색·검증은 `notebooks/`, 반복 실행할 코드는 `scripts/`와 `src/`로 옮깁니다.
4. 중요한 로직에는 `tests/`에 테스트를 추가합니다.
5. 결과와 판단 근거를 `reports/`에 문서로 남깁니다.
6. PR을 열고 체크리스트를 확인합니다.

PR 제목은 형식을 강제하지 않지만 작업 성격이 드러나게 씁니다.

| 예시 | 사용 시점 |
|------|----------|
| `feat: 시계열 lag 피처 추가` | 기능 또는 분석 함수 추가 |
| `fix: PSI 계산의 0 나눗셈 처리` | 버그 수정 |
| `docs: 데이터 수집 절차 정리` | 문서 변경 |
| `refactor: 학습 파이프라인 함수 분리` | 동작 변경 없는 구조 개선 |
| `chore: 개발 의존성 업데이트` | 설정, 의존성, 자동화 변경 |

---

## GitHub 자동화

| 워크플로우 | 트리거 | 내용 |
|-----------|--------|------|
| `ci.yml` | `main` push / PR | ruff lint, ruff format check, pytest |
| `changelog.yml` | `main` push | `CHANGELOG.md` 자동 생성 |
| `issue-helper-private-repo.yml` | 이슈 생성 시, 또는 이슈 **제목**이 수정될 때 | 브랜치명·커밋 메시지 제안 코멘트 작성 (본문만 수정한 경우에는 동작하지 않음) |
| `sync-labels.yml` | `.github/labels.yml`이 변경된 `main` push / 수동 실행 | `labels.yml` 기준으로 라벨 동기화. `skip-delete: false`이므로 **정의에 없는 기존 라벨은 삭제**됨 |

변경 이력은 README에 넣지 않고 [`CHANGELOG.md`](CHANGELOG.md)로 관리합니다.

---

## Claude Code 연동

`CLAUDE.md`와 `.claude/`는 Claude Code가 프로젝트 맥락과 작업 규칙을 자동으로 읽도록 만든 설정입니다. 분석 원칙, 역할별 서브에이전트, `/timeseries`·`/tabular`·`/gis`·`/regression`·`/ml`·`/visualization` 슬래시 커맨드가 포함되어 있습니다.

`scripts/sync_template.sh`는 **스크립트가 있는 저장소를 원본으로 삼아** `.claude/`의 `rules`·`agents`·`commands`·`skills`와 `.github/workflows/`를 `scripts/projects.txt`에 절대 경로로 등록된 프로젝트로 복사합니다. 현재 `projects.txt`에 이 저장소는 포함되어 있지 않습니다.

`rsync --delete`를 사용하므로 **위 대상 디렉터리 안에서 원본에 없는 파일은 삭제됩니다.** 대상 프로젝트에만 있던 규칙·커맨드·워크플로우가 사라질 수 있으니, 실행 전 `projects.txt`의 경로 목록을 확인하세요.

Claude Code를 쓰지 않아도 프로젝트 실행에는 문제가 없습니다.

---

## 라이선스

MIT
