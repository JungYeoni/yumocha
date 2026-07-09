머신러닝 파이프라인을 구성하고 모델을 학습/평가한다. 사용자가 데이터와 목적을 제공하면 아래 단계를 순서대로 실행한다.

## 분석 단계

### 1. 문제 유형 파악
```python
# 분석 날짜: YYYY-MM-DD
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한국어 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

CLASSIFICATION_THRESHOLD = 20  # 고유값 수 기준

print(f"타겟 타입: {y.dtype}")
print(f"고유값 수: {y.nunique()}")

if y.dtype == 'object' or y.nunique() < CLASSIFICATION_THRESHOLD:
    problem_type = "classification"
else:
    problem_type = "regression"
print(f"문제 유형: {problem_type}")

# 클래스 불균형 확인 (분류)
if problem_type == "classification":
    print(y.value_counts(normalize=True).round(3))
```

### 2. 전처리 파이프라인
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "classification" else None
)
```

### 3. 모델 비교
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"{name}: {scores.mean():.4f} (±{scores.std():.4f})")
```

### 4. 하이퍼파라미터 튜닝
```python
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

best_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

search = RandomizedSearchCV(
    best_pipe, param_grid, n_iter=20, cv=5,
    scoring='f1_macro', random_state=42, n_jobs=-1
)
search.fit(X_train, y_train)
print(f"최적 파라미터: {search.best_params_}")
print(f"최적 CV 점수: {search.best_score_:.4f}")
```

### 5. 평가 (분류)
```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
```

### 6. 특성 중요도
```python
rf_model = best_model.named_steps['model']
feature_names = (
    num_cols +
    best_model.named_steps['prep']
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(cat_cols).tolist()
)

importance_df = (
    pd.DataFrame({'feature': feature_names, 'importance': rf_model.feature_importances_})
    .sort_values('importance', ascending=False)
    .head(20)
)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
plt.title("Feature Importance (Top 20)")
plt.tight_layout()
plt.show()
```

### 7. SHAP 해석
```python
import shap

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(
    best_model.named_steps['prep'].transform(X_test)
)
shap.summary_plot(shap_values, feature_names=feature_names)
```

### 8. 모델 저장
```python
import joblib

MODEL_PATH = Path("results/model.pkl")
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"모델 저장: {MODEL_PATH}")
```

## 출력 형식
```
## ML 분석 결과

**문제 유형:** 분류/회귀
**최적 모델:** XGBoost
**성능:** F1=0.XX, AUC=0.XX (test set)
**주요 특성:** feature1 (중요도 0.XX)
**클래스 불균형:** O/X (처리 방법: ...)
```
