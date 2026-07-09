# Skill: /regression — 회귀 분석

## 트리거
사용자가 `/regression` 또는 회귀, 예측, 선형, 상관, OLS, 계수 요청 시 활성화

## 분석 단계

### 1. 데이터 준비
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 특성/타겟 분리
X = df.drop(columns=['target'])
y = df['target']

# 학습/검증 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 스케일링 (선형 회귀 시)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. 선형 회귀 (OLS) — 통계적 해석
```python
import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train_scaled)
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())

# 주요 확인: R², p-value, 계수, VIF
```

### 3. VIF (다중공선성 확인)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_df = pd.DataFrame({
    "Feature": X.columns,
    "VIF": [variance_inflation_factor(X_train_scaled, i)
            for i in range(X_train_scaled.shape[1])]
}).sort_values("VIF", ascending=False)

print(vif_df)
print("\nVIF > 10: 다중공선성 주의")
```

### 4. 잔차 진단
```python
y_pred = model.predict(sm.add_constant(X_test_scaled))
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 잔차 vs 예측값
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title("잔차 vs 예측값")
axes[0].set_xlabel("예측값")
axes[0].set_ylabel("잔차")

# 잔차 분포
axes[1].hist(residuals, bins=30, edgecolor='black')
axes[1].set_title("잔차 분포")

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[2])
axes[2].set_title("Q-Q Plot")

plt.tight_layout()
plt.show()
```

### 5. 성능 지표
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

metrics = {
    "R²": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MAPE": np.mean(np.abs((y_test - y_pred) / y_test)) * 100
}

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
```

### 6. 규제 회귀 (Ridge / Lasso)
```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, m in models.items():
    scores = cross_val_score(m, X_train_scaled, y_train,
                             cv=5, scoring='r2')
    print(f"{name}: R² = {scores.mean():.4f} (±{scores.std():.4f})")
```

### 7. 계수 시각화
```python
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.params[1:]  # 상수항 제외
}).sort_values('coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
colors = ['steelblue' if c > 0 else 'salmon' for c in coef_df['coefficient']]
plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("회귀 계수 (표준화)")
plt.xlabel("계수값")
plt.tight_layout()
plt.show()
```

## 자동 체크리스트
- [ ] 타겟 변수 분포 확인 (정규성 여부)
- [ ] 다중공선성 확인 (VIF)
- [ ] 잔차 정규성, 등분산성 확인
- [ ] 영향력 점 확인 (Cook's distance)
- [ ] 교차검증으로 과적합 확인
- [ ] 적절한 규제 방법 선택

## 출력 형식
```
## 회귀 분석 결과

**모델:** OLS / Ridge / Lasso
**R²:** train=0.XX, test=0.XX
**RMSE:** X.XXX
**주요 변수:** col1 (β=0.XX), col2 (β=-0.XX)
**가정 검토:** 잔차 정규성 O/X, 등분산성 O/X, 다중공선성 O/X
```
