# Skill: /tabular — 테이블 데이터 EDA

## 트리거
사용자가 `/tabular` 또는 EDA, 탐색, 데이터 파악, 기술통계 요청 시 활성화

## 분석 단계

### 1. 데이터 로드 & 기본 정보
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")  # 또는 사용자 데이터프레임

print("=== 기본 정보 ===")
print(f"Shape: {df.shape}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nNull 비율:\n{df.isnull().mean().round(3) * 100}")
print(f"\n중복 행: {df.duplicated().sum()}")
```

### 2. 기술 통계
```python
print("=== 수치형 컬럼 ===")
display(df.describe().round(2))

print("\n=== 범주형 컬럼 ===")
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    print(f"\n{col}: {df[col].nunique()}개 고유값")
    print(df[col].value_counts().head(10))
```

### 3. 결측치 히트맵
```python
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("결측치 패턴")
plt.tight_layout()
plt.show()
```

### 4. 수치형 분포
```python
num_cols = df.select_dtypes(include=np.number).columns
n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
    axes[i].set_title(col)
    axes[i].set_xlabel("값")
    axes[i].set_ylabel("빈도")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()
```

### 5. 상관관계 히트맵
```python
plt.figure(figsize=(12, 10))
corr = df.select_dtypes(include=np.number).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True)
plt.title("상관관계 히트맵")
plt.tight_layout()
plt.show()
```

### 6. 이상치 탐지 (IQR)
```python
def detect_outliers_iqr(df, cols):
    outlier_report = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_report[col] = {
            "count": n_outliers,
            "pct": round(n_outliers / len(df) * 100, 2),
            "range": f"[{lower:.2f}, {upper:.2f}]"
        }
    return pd.DataFrame(outlier_report).T

outliers = detect_outliers_iqr(df, num_cols)
print(outliers[outliers['count'] > 0])
```

## 자동 체크리스트
- [ ] shape, dtypes 확인
- [ ] null 비율 확인 (>30% 컬럼 플래그)
- [ ] 중복 행 제거 여부 결정
- [ ] 수치형 분포 확인 (정규성 여부)
- [ ] 범주형 고유값 수 확인 (cardinality)
- [ ] 상관관계 확인 (>0.8 강상관 플래그)
- [ ] 이상치 탐지 및 처리 방향 결정

## 출력 형식
```
## EDA 결과 요약

**데이터 규모:** N행 × M열
**결측치:** X개 컬럼에 존재 (최대 XX%)
**이상치:** X개 컬럼에서 탐지
**강상관 쌍:** (col_a, col_b): 0.92
**권장 전처리:** ...
```
