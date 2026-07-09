# Skill: /gis — 지리공간 분석

## 트리거
사용자가 `/gis` 또는 지도, 공간, 좌표, shapefile, GeoJSON, 지역별 분석 요청 시 활성화

## 분석 단계

### 1. 데이터 로드
```python
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point

# GeoJSON / Shapefile 로드
gdf = gpd.read_file("data.geojson")
print(f"CRS: {gdf.crs}")
print(f"Geometry type: {gdf.geom_type.unique()}")
print(gdf.head())

# 좌표 데이터프레임 → GeoDataFrame 변환
# df에 lat, lon 컬럼이 있는 경우
geometry = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
```

### 2. 좌표계 변환
```python
# WGS84 → 한국 표준 (EPSG:5186)
gdf_korea = gdf.to_crs("EPSG:5186")

# 거리 계산 시 meter 단위 CRS 사용
gdf_proj = gdf.to_crs("EPSG:3857")  # Web Mercator
```

### 3. 정적 지도 (matplotlib)
```python
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
plt.savefig("map_output.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 4. 인터랙티브 지도 (folium)
```python
m = folium.Map(location=[36.5, 127.5], zoom_start=7, tiles='CartoDB positron')

# Choropleth
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
m.save("map.html")
m
```

### 5. 공간 조인
```python
# 포인트 → 폴리곤 조인 (어느 지역에 속하는지)
joined = gpd.sjoin(points_gdf, polygons_gdf, how='left', predicate='within')
```

### 6. 클러스터링 (DBSCAN)
```python
from sklearn.cluster import DBSCAN
import numpy as np

coords = gdf[['geometry']].copy()
coords['lon'] = gdf.geometry.x
coords['lat'] = gdf.geometry.y

# 반경 500m = 0.005도 (근사)
db = DBSCAN(eps=0.005, min_samples=5, algorithm='ball_tree', metric='haversine')
gdf['cluster'] = db.fit_predict(np.radians(coords[['lat', 'lon']]))
```

### 7. 거리 계산
```python
from geopy.distance import geodesic

def calc_distance_km(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 (km)"""
    return geodesic((lat1, lon1), (lat2, lon2)).km

# 예시
dist = calc_distance_km(37.5665, 126.9780, 35.1796, 129.0756)
print(f"서울-부산: {dist:.1f} km")
```

## 자동 체크리스트
- [ ] CRS 확인 및 필요시 변환 (WGS84 기준)
- [ ] geometry 유효성 검사 (`gdf.is_valid.all()`)
- [ ] 빈 geometry 제거
- [ ] 공간 인덱스 활용 (대용량 데이터)
- [ ] 지도 출력 (정적 + 인터랙티브)
- [ ] 공간 통계 요약 (면적, 밀도, 중심점 등)

## 출력 형식
```
## GIS 분석 결과

**데이터:** N개 피처, CRS: EPSG:XXXX
**공간 범위:** 위도 XX~XX, 경도 XX~XX
**주요 발견:** 고밀도 지역 / 클러스터 수 / 분포 특성
**산출물:** map.html, map_output.png
```
