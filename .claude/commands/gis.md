지리공간 분석을 수행한다. 사용자가 GeoJSON, Shapefile, 좌표 데이터를 제공하면 아래 단계를 순서대로 실행한다.

## 분석 단계

### 1. 데이터 로드
```python
# 분석 날짜: YYYY-MM-DD
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point

# 한국어 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = Path("data.geojson")

# GeoJSON / Shapefile 로드
gdf = gpd.read_file(DATA_PATH)
print(f"CRS: {gdf.crs}")
print(f"Geometry type: {gdf.geom_type.unique()}")
print(f"Shape: {gdf.shape}")
print(f"유효한 geometry: {gdf.is_valid.all()}")

# 좌표 데이터프레임 → GeoDataFrame 변환 (lat/lon 컬럼이 있는 경우)
# geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
# gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
```

### 2. 좌표계 변환
```python
# WGS84 → 한국 표준 (EPSG:5186)
gdf_korea = gdf.to_crs("EPSG:5186")

# 거리 계산 시 meter 단위 CRS
gdf_proj = gdf.to_crs("EPSG:3857")
```

### 3. 정적 지도 (matplotlib)
```python
OUTPUT_PATH = Path("results/map_output.png")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
gdf.plot(
    column='value',
    cmap='YlOrRd',
    legend=True,
    legend_kwds={'label': '지표값', 'orientation': 'vertical'},
    ax=ax,
    edgecolor='gray',
    linewidth=0.5
)
ax.set_title("지역별 분포", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
plt.show()
```

### 4. 인터랙티브 지도 (folium)
```python
MAP_PATH = Path("results/map.html")

m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles='CartoDB positron')

folium.Choropleth(
    geo_data=gdf.to_json(),
    name='choropleth',
    data=df,
    columns=['region_code', 'value'],
    key_on='feature.properties.region_code',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.5,
    legend_name='지표값'
).add_to(m)

folium.LayerControl().add_to(m)
m.save(MAP_PATH)
m
```

### 5. 공간 조인
```python
# 포인트 → 폴리곤 조인 (어느 지역에 속하는지)
joined = gpd.sjoin(points_gdf, polygons_gdf, how='left', predicate='within')
```

### 6. 클러스터링 (DBSCAN)
```python
import numpy as np
from sklearn.cluster import DBSCAN

np.random.seed(42)

coords = gdf[['geometry']].copy()
coords['lon'] = gdf.geometry.x
coords['lat'] = gdf.geometry.y

# 반경 500m ≈ 0.005도
EPSILON = 0.005
MIN_SAMPLES = 5

db = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES,
            algorithm='ball_tree', metric='haversine')
gdf['cluster'] = db.fit_predict(np.radians(coords[['lat', 'lon']]))
print(f"클러스터 수: {gdf['cluster'].nunique() - 1} (노이즈 제외)")
```

### 7. 거리 계산
```python
from geopy.distance import geodesic

def calc_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 (km)"""
    return geodesic((lat1, lon1), (lat2, lon2)).km

# 예시: 서울-부산
dist = calc_distance_km(37.5665, 126.9780, 35.1796, 129.0756)
print(f"서울-부산: {dist:.1f} km")
```

## 출력 형식
```
## GIS 분석 결과

**데이터:** N개 피처, CRS: EPSG:XXXX
**공간 범위:** 위도 XX~XX, 경도 XX~XX
**주요 발견:** 고밀도 지역 / 클러스터 수 / 분포 특성
**산출물:** results/map.html, results/map_output.png
```
