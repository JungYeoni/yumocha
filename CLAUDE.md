# CLAUDE.md — 데이터분석·ML 프로젝트 템플릿

이 저장소는 정형 데이터 분석, 시계열, 회귀/인과추론, 머신러닝, GIS 결합형 분석 프로젝트에 공통으로 사용하는 템플릿이다.

## 이 프로젝트에서 Claude가 따를 원칙

### Git 작업 안전
- 사용자가 명시적으로 요청한 경우에만 커밋, 푸시, PR 생성을 진행한다.
- 코드 수정, 테스트, 검증은 수행할 수 있지만, 배포성 Git 작업은 자동으로 진행하지 않는다.
- 커밋이나 푸시가 필요해 보이는 상황에서도 먼저 변경 내용과 검증 결과를 보고하고 사용자의 요청을 기다린다.

### 재현성 우선
- 모든 분석에 `np.random.seed(42)` 고정
- 데이터 분할 기준(날짜, 비율, stratify 여부)을 코드 주석으로 명시
- 전처리 → 모델링 → 평가 절차를 `sklearn.pipeline.Pipeline`으로 통합

### 데이터 누수 방지
- 피처 생성 전에 train/val/test 분리를 먼저 확정
- 인코더·스케일러는 학습 데이터에서만 fit, 나머지에는 transform만 적용
- 시계열에서 rolling/lag 계산 시 `shift(1)` 선행 필수

### 역할 분리
- 데이터 로딩/검증: `src/features/`
- 피처 엔지니어링: `src/features/`
- 모델 학습: `src/modeling/`
- 평가 지표 계산: `src/evaluation/`
- 시각화 산출물: `src/visualization/`

### 방법론적 타당성 우선
- 단순 성능 비교보다 통계적 가정 검증, 계수 해석, 한계점 명시를 우선
- 복잡한 모델보다 해석 가능하고 재현 가능한 방법을 선택

### 실험 결과 보고 기준
모든 실험 결과에는 아래 항목이 포함되어야 한다:
- 데이터셋 버전 및 기간
- 분할 전략 (train/val/test 기준)
- 평가 지표와 CV fold 수
- 핵심 가정 및 방법론적 제약
- 한계점 및 잠재 편향

## 서브에이전트 사용 가이드

| 에이전트 | 사용 시점 |
|----------|-----------|
| `data-scientist` | 시계열 분석(ARIMA/VAR), 회귀/인과추론(OLS/DiD/패널), sklearn 정형 데이터 분류·회귀 |
| `feature-engineer` | 시계열 lag/rolling/계절성 피처, 테이블 인코딩/스케일링/교호작용, GIS 행정구역 집계·공간조인 |
| `data-visualization` | 논문·보고서용 정적 이미지(dpi=300), Plotly/Streamlit 인터랙티브 대시보드 |

에이전트 정의 파일: `.claude/agents/`

## 기술 스택

- **언어**: Python 3.11+
- **데이터**: pandas, polars
- **통계**: statsmodels, scipy
- **ML**: scikit-learn, xgboost, lightgbm
- **시각화**: matplotlib, seaborn, plotly, streamlit
- **GIS**: geopandas, folium, shapely
- **환경**: pyproject.toml 기반 의존성 관리
