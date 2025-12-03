import os


os.environ['PROJ_LIB']  = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
os.environ['PROJ_DATA'] = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
#!/usr/bin/env python
"""
어구의 최종 위치(위도, 경도)를 이용한 애니메이션 및 침전(해수면 아래) 밀집 구역 히트맵 생성:
- 주기적 시딩을 통해 지속적으로 어구(쓰레기)를 투입함.
- 각 시딩 이벤트마다 매번 랜덤한 위치에서 500개의 시딩점을 균등하게 선택.
- 각 입자에 대해 밀도에 따른 개별 terminal_velocity 부여.
- 해류, 풍력, 수심 강제자료 반영. (해류와 바람은 원 자료는 3시간 단위이나, 시뮬레이션 내부에서는 1시간 간격 보간 적용)
- 시뮬레이션 종료 후, 모델 내부의 최종 입자 위치를 활용하여 애니메이션 생성 및 
  최종 위치(위도/경도) 데이터와 해수면 아래 침전 입자 기준 히트맵 생성.
"""

import os
from datetime import timedelta, datetime
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from opendrift.models.sedimentdrift import SedimentDrift
import pandas as pd

# ------------------------- 환경 변수 및 강제자료 파일 경로 설정 -------------------------
os.environ['PROJ_LIB'] = r'C:\Users\HUFS\anaconda3\envs\opendrift-env\Library\share\proj'

# 강제자료 파일 경로 (실제 환경에 맞게 수정)
ocean_nc_file       = r'C:\Users\HUFS\opendrift\examples\HYCOM\processed_current.nc'
# 바람 강제자료 경로 업데이트 (3시간 단위 자료)
wind_nc_file        = r'C:\Users\HUFS\Desktop\침적opendrift\바람데이터_3_9.nc'
bathymetry_nc_file  = r'C:\Ocean-Solution-Association\tinppl\desktop\opendrfit_middle\rename_output.nc'

# ------------------------- STEP 0: 해안선 기반 해안 영역 정의 -------------------------
coastline_file = r"C:\Users\HUFS\Desktop\2024년 전국해안선.shp"
coast = gpd.read_file(coastline_file)
if coast.crs is None or coast.crs.to_string() != 'EPSG:4326':
    coast = coast.to_crs(epsg=4326)

    # "north": "34.9",
    # "west": "125.48",
    # "east": "128",
    # "south": "34.1"

# 남해안 후보 영역 (예: 경도 125~129, 위도 33~35)
lon_min, lon_max = 125.48, 128
lat_min, lat_max = 34.1, 34.9

# 해안선 데이터를 EPSG:3857로 투영 후 15km 버퍼 생성
coast_proj = coast.to_crs(epsg=3857)
coastal_zone = coast_proj.buffer(15000).unary_union

# ------------------------- STEP 1: 시딩점 생성 함수 정의 -------------------------
def generate_seed_points():
    """
    주어진 남해안 후보 영역 내에서 무작위로 500개 시딩점을 생성하고,
    해안선으로부터 15km 내 영역에 있는 점만 선택하여 WGS84 좌표(lon, lat)를 반환합니다.
    """
    num_candidates = 5000  # 충분한 후보점 생성
    cand_lons = np.random.uniform(lon_min, lon_max, num_candidates)
    cand_lats = np.random.uniform(lat_min, lat_max, num_candidates)
    cand_points = [Point(lon, lat) for lon, lat in zip(cand_lons, cand_lats)]
    gdf_cand = gpd.GeoDataFrame({'geometry': cand_points}, crs="EPSG:4326")
    
    # 후보점을 EPSG:3857로 변환 후, 해안선 15km 버퍼 내에 있는지 체크
    gdf_cand_proj = gdf_cand.to_crs(epsg=3857)
    mask = gdf_cand_proj['geometry'].apply(lambda geom: coastal_zone.contains(geom))
    filtered = gdf_cand_proj[mask]
    
    if len(filtered) < 500:
        raise Exception("해안선으로부터 15km 내 시딩 후보점이 부족합니다.")
    
    # 500점 무작위로 선택
    selected = filtered.sample(n=500, random_state=None)
    selected_wgs84 = selected.to_crs(epsg=4326)
    return selected_wgs84.geometry.x.values, selected_wgs84.geometry.y.values

# ------------------------- STEP 2: 모델 생성 및 강제자료 추가 -------------------------
model = SedimentDrift(loglevel=20)
model.add_readers_from_list([ocean_nc_file, wind_nc_file, bathymetry_nc_file])
model.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
model.set_config('drift:horizontal_diffusivity', 100)
model.set_config('drift:current_uncertainty', 0.2)
model.set_config('drift:wind_uncertainty', 2)
# (수직 보간은 모델 내부에서 유효한 수심 범위 내에서 자동으로 적용됩니다.)

# ------------------------- STEP 3: 주기적 시딩 및 입자 개별 특성 부여 -------------------------
# 시뮬레이션 기간 설정: 시작 시간과 종료 시간을 모두 00:00:00으로 맞춤
simulation_start = datetime(2024, 3, 1, 0, 0, 0)
simulation_end   = datetime(2024, 3, 15, 0, 0, 0)
simulation_duration = simulation_end - simulation_start
# 시딩 간격은 기존 6시간 간격 유지 (필요에 따라 변경 가능)
seeding_interval = timedelta(hours=6)
num_seeding_events = int(simulation_duration.total_seconds() // seeding_interval.total_seconds()) + 1

for i in range(num_seeding_events):
    seed_time = simulation_start + seeding_interval * i
    # 매 시딩 이벤트마다 새로운 시딩점 생성 (고르게 랜덤 분포)
    lons, lats = generate_seed_points()
    # terminal_velocity는 노말분포를 따르며 음수를 제거
    terminal_velocities = np.clip(np.random.normal(loc=0.0, scale=0.05, size=len(lons)), a_min=0, a_max=None)
    print(f"{seed_time} 시딩: 입자 수 = {len(lons)}, 평균 terminal_velocity = {np.mean(terminal_velocities):.3f} m/s")
    model.seed_elements(lon=lons, lat=lats, time=seed_time, terminal_velocity=terminal_velocities)

# ------------------------- STEP 4: 시뮬레이션 실행 및 애니메이션 생성 -------------------------
# 시뮬레이션 내부 시간 단위를 1시간(3600초)으로 설정하여 내부 보간을 통한 정밀 계산을 진행
model.run(time_step=3600, time_step_output=3600, duration=simulation_duration)
model.animation(color='z', fast=False, buffer=0.01)

# ------------------------- STEP 5: 최종 입자 위치 추출 및 히트맵 생성 -------------------------
# model.elements는 단일 SedimentElement 객체로, 내부에 lon, lat, z가 numpy 배열 형태로 저장되어 있다고 가정
final_df = pd.DataFrame({
    'lon': model.elements.lon,
    'lat': model.elements.lat,
    'z': model.elements.z
})

# 해수면 아래 (z < 0) 침전 입자만 선택
deposits = final_df[final_df['z'] < 0]
print("침전(해수면 아래)된 입자 수:", len(deposits))
print("최종 입자 위치 데이터 (일부):")
print(final_df[['lon', 'lat']].head())

# 히트맵 생성: deposits의 위도, 경도 데이터를 히스토그램으로 표현
heatmap, xedges, yedges = np.histogram2d(deposits['lon'], deposits['lat'], bins=100)
plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto')
plt.colorbar(label="입자 밀도")
plt.xlabel("경도")
plt.ylabel("위도")
plt.title("침전 입자 밀집 구역 히트맵")

# 지정한 위도/경도 범위 내 육지 면적 오버레이 (경계나 노드 없이 면만 채움)
bbox = box(lon_min, lat_min, lon_max, lat_max)
coast_clipped = gpd.clip(coast, bbox)
ax = plt.gca()
coast_clipped.plot(ax=ax, color='lightgray', zorder=10)  # 원하는 색상으로 채움

plt.show()
