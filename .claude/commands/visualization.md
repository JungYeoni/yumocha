시각화를 생성한다. 사용자가 데이터와 목적을 제공하면 적절한 차트를 선택하여 실행한다.

## 기본 설정
```python
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# 한국어 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 6)

# 색약 친화 팔레트
PALETTE = sns.color_palette("colorblind")
SEQUENTIAL = "YlOrRd"
DIVERGING = "RdBu_r"

sns.set_style("whitegrid")
```

## 차트 유형별 코드

### 1. 분포 비교 (히스토그램 + KDE)
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data, bins=30, color=PALETTE[0], edgecolor='white', alpha=0.8)
axes[0].set_title("분포")
axes[0].set_xlabel("값")
axes[0].set_ylabel("빈도")

sns.kdeplot(data=df, x='value', hue='group', ax=axes[1], palette=PALETTE)
axes[1].set_title("그룹별 밀도")

plt.tight_layout()
plt.show()
```

### 2. 카테고리별 비교 (Bar + Box)
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

order = df.groupby('category')['value'].median().sort_values(ascending=False).index
sns.barplot(data=df, x='category', y='value', order=order,
            palette=PALETTE, ax=axes[0], errorbar='sd')
axes[0].set_title("카테고리별 평균 (±SD)")
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(data=df, x='category', y='value', order=order,
            palette=PALETTE, ax=axes[1])
axes[1].set_title("카테고리별 분포")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### 3. 추세 시각화 (Line + Confidence Interval)
```python
fig, ax = plt.subplots(figsize=(14, 6))

for i, group in enumerate(df['group'].unique()):
    subset = df[df['group'] == group].sort_values('date')
    ax.plot(subset['date'], subset['value'],
            label=group, color=PALETTE[i], linewidth=2)
    ax.fill_between(subset['date'],
                    subset['value'] - subset['std'],
                    subset['value'] + subset['std'],
                    alpha=0.15, color=PALETTE[i])

ax.set_title("그룹별 추세")
ax.set_xlabel("날짜")
ax.set_ylabel("값")
ax.legend()
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

### 4. 산점도 + 회귀선
```python
fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(df['x'], df['y'],
                     c=df['color_var'], cmap=SEQUENTIAL,
                     alpha=0.6, s=60, edgecolor='white')

z = np.polyfit(df['x'], df['y'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['x'].min(), df['x'].max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="추세선")

plt.colorbar(scatter, ax=ax, label='색상 변수')
ax.set_title("X vs Y 산점도")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
plt.tight_layout()
plt.show()
```

### 5. 히트맵 (상관관계 / 피벗)
```python
pivot = df.pivot_table(index='row_var', columns='col_var', values='value', aggfunc='mean')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt=".1f",
            cmap=DIVERGING, center=0,
            linewidths=0.5, cbar_kws={'label': '값'})
plt.title("피벗 히트맵")
plt.tight_layout()
plt.show()
```

### 6. 인터랙티브 (Plotly)
```python
# 산점도
fig = px.scatter(df, x='x', y='y', color='group',
                 size='size_var', hover_data=['label'],
                 title="인터랙티브 산점도",
                 color_discrete_sequence=px.colors.qualitative.Safe)
fig.show()

# 라인 차트
fig = px.line(df, x='date', y='value', color='group',
              title="그룹별 추세",
              labels={'value': '값', 'date': '날짜'})
fig.update_layout(hovermode='x unified')
fig.show()
```

### 7. 대시보드 레이아웃 (Plotly Subplots)
```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("분포", "추세", "상관관계", "카테고리별"),
    specs=[[{"type": "histogram"}, {"type": "scatter"}],
           [{"type": "heatmap"}, {"type": "bar"}]]
)

fig.add_trace(go.Histogram(x=df['value'], name="분포"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['date'], y=df['value'], name="추세"), row=1, col=2)

fig.update_layout(height=800, title="분석 대시보드", showlegend=True)
fig.show()
```

## 저장
```python
OUTPUT_PATH = Path("results")
OUTPUT_PATH.mkdir(exist_ok=True)

# matplotlib
plt.savefig(OUTPUT_PATH / "output.png", dpi=150, bbox_inches='tight', facecolor='white')

# plotly
fig.write_html(str(OUTPUT_PATH / "dashboard.html"))
fig.write_image(str(OUTPUT_PATH / "output.png"), width=1200, height=800, scale=2)
```

## 차트 선택 가이드
| 목적 | 권장 차트 |
|------|-----------|
| 분포 확인 | 히스토그램, KDE, 박스플롯 |
| 비교 | 막대 그래프, 바이올린 플롯 |
| 추세 | 라인 차트, 면적 차트 |
| 관계 | 산점도, 버블 차트 |
| 구성 | 파이 차트 (≤5개), 스택 바 |
| 패턴 | 히트맵, 상관관계 매트릭스 |
