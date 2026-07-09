시계열 분석을 수행한다. 사용자가 데이터나 분석 요청을 제공하면 아래 단계를 순서대로 안내하거나 직접 실행한다.

## 분석 단계

### 1. 기본 검사
```python
# 분석 날짜: YYYY-MM-DD
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 한국어 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 날짜 컬럼 파싱 및 인덱스 설정
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

print(f"기간: {df.index.min()} ~ {df.index.max()}")
print(f"데이터 포인트: {len(df)}")
print(f"결측치:\n{df.isnull().sum()}")
```

### 2. 추세 & 계절성 분해
```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df['value'], model='additive', period=12)
decomp.plot()
plt.tight_layout()
plt.show()
```

### 3. 정상성 검정 (ADF Test)
```python
def adf_test(series: pd.Series) -> None:
    """ADF 정상성 검정 결과 출력"""
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("정상성:", "O (p < 0.05)" if result[1] < 0.05 else "X (차분 필요)")

adf_test(df['value'])
```

### 4. ACF / PACF 플롯
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(df['value'].dropna(), ax=axes[0], lags=40)
plot_pacf(df['value'].dropna(), ax=axes[1], lags=40)
plt.tight_layout()
plt.show()
```

### 5. ARIMA 모델링
```python
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)
model = ARIMA(df['value'], order=(1, 1, 1))
result = model.fit()
print(result.summary())

forecast = result.forecast(steps=12)
```

### 6. Prophet 예측
```python
from prophet import Prophet

np.random.seed(42)
df_prophet = df.reset_index().rename(columns={'date': 'ds', 'value': 'y'})
m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
m.fit(df_prophet)

future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)
m.plot(forecast)
plt.show()
```

## 체크리스트
- 날짜 인덱스 설정 및 정렬
- 결측치 처리 (보간 또는 제거)
- 이상치 탐지 (IQR 또는 Z-score)
- 정상성 확인 (ADF test)
- 모델 선택 (ARIMA / Prophet / LSTM)
- 예측 구간(confidence interval) 포함

## 출력 형식
```
## 시계열 분석 결과

**기간:** YYYY-MM-DD ~ YYYY-MM-DD
**빈도:** 일/주/월별
**정상성:** O/X (ADF p-value: 0.xxx)
**주요 패턴:** 추세 상승/하락, 계절성 존재 여부
**권장 모델:** ARIMA(p,d,q) / Prophet
```
