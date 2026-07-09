# Skill: /ml — 머신러닝 파이프라인

## 트리거
사용자가 `/ml` 또는 분류, 클러스터링, 모델, 예측, XGBoost, 머신러닝 요청 시 활성화

## 분석 단계

### 1. 문제 유형 파악
```python
# 타겟 변수 확인
print(f"타겟 타입: {y.dtype}")
print(f"고유값 수: {y.nunique()}")

# 분류 vs 회귀 판단
if y.dtype == 'object' or y.nunique() < 20:
    problem_type = "classification"
else:
    problem_type = "regression"
print(f"문제 유형: {problem_type}")

# 클래스 불균형 확인 (분류)
print(y.value_counts(normalize=True).round(3))
```

### 2. 전처리 파이프라인
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

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
```

### 3. 모델 비교
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_macro')
    results[name] = scores
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
# Random Forest / XGBoost 계열
rf_model = best_model.named_steps['model']
feature_names = (
    num_cols +
    best_model.named_steps['prep']
    .named_transformers_['cat']
    .named_steps['encoder']
    .get_feature_names_out(cat_cols).tolist()
)

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

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

## 자동 체크리스트
- [ ] 문제 유형 확인 (분류/회귀/클러스터링)
- [ ] 클래스 불균형 확인 및 대응 (SMOTE, class_weight)
- [ ] 누수(leakage) 방지: 전처리를 파이프라인 내에서만
- [ ] 교차검증으로 성능 평가
- [ ] 테스트셋은 최종 평가에만 사용
- [ ] 특성 중요도 또는 SHAP으로 해석성 확보
- [ ] 모델 저장 (joblib)

## 출력 형식
```
## ML 분석 결과

**문제 유형:** 분류/회귀
**최적 모델:** XGBoost
**성능:** F1=0.XX, AUC=0.XX (test set)
**주요 특성:** feature1 (중요도 0.XX)
**클래스 불균형:** O/X (처리 방법: ...)
```
