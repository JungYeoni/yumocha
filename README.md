# da-template

[![Use this template](https://img.shields.io/badge/Use%20this%20template-2ea44f?style=for-the-badge&logo=github)](https://github.com/JungYeoni/da-template/generate)

데이터 분석과 머신러닝 프로젝트를 빠르게 시작하기 위한 Python 프로젝트 템플릿입니다.

반복적으로 필요한 디렉터리 구조, 설정 파일, 테스트, GitHub Actions, 이슈/PR 템플릿, Claude Code용 분석 지침을 미리 갖추고 있습니다. 의존성 관리는 기본적으로 `uv`와 `pyproject.toml`/`uv.lock`을 사용합니다. 새 프로젝트를 만들 때 위 **Use this template** 버튼을 누르면 같은 구조로 저장소를 시작할 수 있습니다.

---

## 언제 사용하나요?

- 정형 데이터 EDA, 전처리, 피처 엔지니어링
- 분류/회귀 모델링과 모델 평가
- 시계열 분석과 예측
- 회귀분석, 인과추론, 패널 데이터 분석
- GIS 데이터 결합 분석
- 보고서용 시각화와 간단한 대시보드 제작

개인 프로젝트뿐 아니라 소규모 팀에서 분석 흐름과 산출물 위치를 맞추고 싶을 때도 사용할 수 있습니다.

---

## 빠른 시작

### 1. 템플릿으로 저장소 만들기

GitHub 상단의 **Use this template** 버튼을 눌러 새 저장소를 만듭니다.

또는 이 저장소를 직접 클론해 실험할 수 있습니다.

```bash
git clone https://github.com/JungYeoni/da-template.git my-project
cd my-project
```

### 2. 개발 환경 준비

이 프로젝트는 Python 3.11 이상과 `uv` 기반 의존성 관리를 기본으로 합니다.

권장 방식:

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

`pip`만 사용할 수 있는 환경에서는 아래 방식도 가능합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

Windows PowerShell에서는 가상환경 활성화 명령이 다릅니다.

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 프로젝트 이름과 설정 바꾸기

새 프로젝트로 복사한 뒤에는 아래 항목을 먼저 바꾸는 것을 권장합니다.

- `pyproject.toml`의 `name`, `description`, 의존성
- `README.md`의 프로젝트 설명
- `configs/base.yaml`의 경로, seed, train/validation/test 분할 기준
- `.github/CODEOWNERS`와 GitHub 이슈/PR 템플릿
- `cliff.toml`의 GitHub 저장소 URL

---

## 의존성 관리

이 템플릿은 `uv`를 기본 패키지 매니저로 사용합니다.

- 런타임/개발 의존성은 `pyproject.toml`에 정의합니다.
- 잠금 파일은 `uv.lock`으로 관리합니다.
- 새 환경을 만들 때는 `uv sync --extra dev`를 사용합니다.
- 명령 실행은 `uv run <command>` 형식을 권장합니다.
- `requirements.txt`는 호환성이나 외부 배포가 필요한 경우를 위한 보조 파일입니다.

자주 쓰는 명령:

```bash
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

의존성을 추가할 때:

```bash
uv add pandas
uv add --dev pytest
```

---

## 디렉터리 구조

```text
da-template/
├── README.md                     # 프로젝트 설명과 사용 방법
├── CHANGELOG.md                  # 변경 이력
├── CLAUDE.md                     # Claude Code용 프로젝트 지침
├── pyproject.toml                # 패키지 메타데이터와 도구 설정
├── uv.lock                       # uv 잠금 파일
├── requirements.txt              # 핵심 의존성 목록
├── cliff.toml                    # git-cliff 변경 이력 설정
│
├── configs/
│   ├── base.yaml                 # 공통 설정
│   ├── dev.yaml                  # 개발 환경 설정
│   └── prod.yaml                 # 제출/운영 환경 설정
│
├── data/
│   ├── raw/                      # 원본 데이터, git 추적 제외
│   ├── interim/                  # 중간 처리 데이터
│   └── processed/                # 모델 입력용 최종 데이터
│
├── notebooks/                    # 탐색 분석과 실험 노트북
├── reports/                      # 보고서, 그림, 표, 대시보드 산출물
│
├── src/
│   ├── features/                 # 피처 생성 코드
│   ├── modeling/                 # 모델 학습 코드
│   ├── evaluation/               # 평가 지표와 검증 코드
│   └── visualization/            # 시각화 코드
│
├── tests/                        # 단위 테스트
│
├── .github/
│   ├── ISSUE_TEMPLATE/           # 이슈 템플릿
│   ├── pull_request_template.md  # PR 체크리스트
│   └── workflows/                # CI, 노트북 검사, changelog 자동화
│
└── .claude/                      # Claude Code 명령, 규칙, 에이전트 설정
```

---

## 기본 작업 흐름

### 새 분석을 시작할 때

1. GitHub Issue를 만들고 목표, 데이터, 성공 기준을 적습니다.
2. 브랜치를 만듭니다.

```bash
git checkout -b experiment/short-description
```

3. 원본 데이터는 `data/raw/`에 둡니다.
4. 탐색 분석은 `notebooks/`에서 진행합니다.
5. 재사용할 코드는 `src/` 아래로 옮깁니다.
6. 중요한 로직에는 `tests/`에 테스트를 추가합니다.
7. PR을 열고 체크리스트를 확인합니다.

### 노트북과 소스 코드의 역할

노트북은 탐색과 의사결정을 기록하는 공간입니다. 반복해서 쓰는 전처리, 피처 생성, 평가, 시각화 코드는 `src/`로 옮겨 테스트 가능한 함수로 관리하는 것을 권장합니다.

예를 들어:

- `notebooks/01_eda.ipynb`: 데이터 확인, 결측치/분포/이상치 탐색
- `src/features/build_features.py`: 실험에 반복 사용되는 피처 함수
- `src/modeling/train.py`: 모델 학습 함수
- `src/evaluation/evaluate.py`: 평가 지표 계산
- `tests/test_features.py`: 피처 함수 검증

---

## 품질 확인

커밋하거나 PR을 열기 전에 아래 명령을 실행하세요.

```bash
uv run --extra dev ruff check src/ tests/
uv run --extra dev ruff format --check src/ tests/
uv run --extra dev pytest tests/ -v
```

`pip` 환경에서는 가상환경을 활성화한 뒤 아래처럼 실행하면 됩니다.

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest tests/ -v
```

---

## 분석 원칙

### 재현성

- random seed를 명시합니다.
- 데이터 분할 기준을 코드나 문서에 남깁니다.
- 원본 데이터는 수정하지 않고, 처리 결과는 `data/interim/` 또는 `data/processed/`에 둡니다.

### 데이터 누수 방지

- train/validation/test 분리 후 전처리 기준을 학습 데이터에서만 계산합니다.
- 시계열 rolling/lag 피처는 미래 값을 참조하지 않도록 `shift(1)` 이후 계산합니다.
- 인코더, 스케일러, imputing 파라미터는 학습 데이터에만 fit합니다.

### 설명 가능한 결과

- 모델 성능뿐 아니라 데이터 가정, 한계, 실패 사례를 함께 기록합니다.
- 복잡한 모델을 쓰기 전에 단순한 baseline을 먼저 만듭니다.
- 중요한 판단은 노트북, 이슈, PR 설명 중 한 곳에 남깁니다.

---

## GitHub 자동화

| 워크플로우 | 트리거 | 내용 |
|-----------|--------|------|
| `ci.yml` | push/PR to `main` | ruff lint, ruff format check, pytest |
| `changelog.yml` | `main` push | `CHANGELOG.md` 자동 생성 |

변경 이력은 README에 직접 삽입하지 않고, 별도 [`CHANGELOG.md`](CHANGELOG.md) 파일로 관리합니다.

---

## PR 제목 예시

PR 제목 형식은 강제하지 않지만, 아래처럼 작업 성격이 드러나게 쓰는 것을 권장합니다.

| 예시 | 사용 시점 |
|------|----------|
| `experiment: baseline 모델 비교` | 새 분석 실험 |
| `feat: 시계열 lag 피처 추가` | 기능 또는 분석 함수 추가 |
| `fix: PSI 계산의 0 나눗셈 처리` | 버그 수정 |
| `docs: 데이터 수집 절차 정리` | 문서 변경 |
| `refactor: 학습 파이프라인 함수 분리` | 동작 변경 없는 구조 개선 |
| `chore: 개발 의존성 업데이트` | 설정, 의존성, 자동화 변경 |

---

## Claude Code 연동

`CLAUDE.md`와 `.claude/` 폴더는 Claude Code에서 프로젝트 맥락을 자동으로 읽을 수 있도록 만든 설정입니다.

포함된 내용:

- 데이터 분석 프로젝트의 기본 원칙
- 역할별 서브에이전트 설정
- `/timeseries`, `/tabular`, `/gis`, `/regression`, `/ml`, `/visualization` 같은 분석용 슬래시 커맨드
- 민감 파일 접근 제한과 작업 규칙

Claude Code를 사용하지 않아도 프로젝트 실행에는 문제가 없습니다. 다른 에디터나 코딩 에이전트를 쓰는 경우에도 `CLAUDE.md`를 분석 가이드 문서로 참고할 수 있습니다.

---

## 변경 이력

변경 이력은 [`CHANGELOG.md`](CHANGELOG.md)에서 확인할 수 있습니다.

---

## 라이선스

MIT
