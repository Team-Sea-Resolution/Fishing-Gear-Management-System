import os


os.environ['PROJ_LIB']  = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
os.environ['PROJ_DATA'] = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
#!/usr/bin/env python

from datetime import datetime, timedelta
import numpy as np
import random
from shapely.geometry import Polygon, Point, box
from opendrift.models.sedimentdrift import SedimentDrift
# from opendrift.models.oceandrift import OceanDrift
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False



# --- 파일 경로 설정 ---
ocean_nc_file = r'.\new_dataset\new_sediment_data_final_uv.nc'
wind_nc_file = r'.\new_dataset\new_wind.nc'
bathymetry_nc_file = r'.\new_dataset\new_BADA2024.nc'

# ------------------------- STEP 0: 해안선 기반 해안 영역 정의 -------------------------
coastline_file = r".\dataset\2024년 전국해안선.shp"
coast = gpd.read_file(coastline_file)
if coast.crs is None or coast.crs.to_string() != 'EPSG:4326':
    coast = coast.to_crs(epsg=4326)

    # "north": "34.9",
    # "west": "125.48",
    # "east": "128",
    # "south": "34.1"


coast_proj = coast.to_crs(epsg=3857)
coastal_zone = coast_proj.buffer(15000).unary_union


land_shp = './dataset/ne_land/ne_10m_land.shp'  # 위치에 맞게 조정
land_gdf = gpd.read_file(land_shp)
land_union = land_gdf.unary_union



# ----------------------------
# 시딩 영역 정의 (진해항)
# ----------------------------

regions_polygon_coords = {
    "고흥군": [
        (127.2287, 34.699),
        (127.1149, 34.6899),
        (126.976, 34.4676),
        (127.0914, 34.4404)
    ],
    "대광이도": [
        (128.5345, 35.1036),
        (128.6559, 35.0007),
        (128.5547, 34.9138),
        (128.4121, 34.9903)
    ],
    "진해항": [
        (128.6807, 35.1423),
        (128.8441, 35.0182),
        (128.7544, 34.9632),
        (128.604, 35.1087)
    ],
    "사천시": [
        (128.0309, 35.0233),
        (128.0334, 34.9319),
        (127.9959, 34.9163),
        (127.8976, 34.9479)
    ],
    "거제도": [
        (128.5791, 34.8106),
        (128.5307, 34.7139),
        (128.3377, 34.7663),
        (128.4586, 34.8751)
    ],
    "변산반도 국립공원": [
        (126.5378, 35.7035),
        (126.4213, 35.438),
        (126.2213, 35.5624),
        (126.3255, 35.7058)
    ]
}
SELECTED_REGION = "고흥군"
selected_coords = regions_polygon_coords[SELECTED_REGION]
polygon = Polygon(selected_coords)

polygon_lons, polygon_lats = zip(*selected_coords)
lon_min, lon_max = min(polygon_lons) - 0.06, max(polygon_lons) + 0.06
lat_min, lat_max = min(polygon_lats) - 0.06, max(polygon_lats) + 0.06


# ------------------------- STEP 1: 시딩점 생성 함수 정의 -------------------------
def generate_seed_points(num_points=100, seed=42):
    np.random.seed(seed)  # 항상 같은 난수 시퀀스 유지
    points = []
    while len(points) < num_points:
        lon = np.random.uniform(lon_min, lon_max)
        lat = np.random.uniform(lat_min, lat_max)
        pt = Point(lon, lat)
        if polygon.contains(pt) and not land_union.contains(pt):
            points.append(pt)
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    return gdf.geometry.x.values, gdf.geometry.y.values

# ------------------------- STEP 2: 모델 생성 및 강제자료 추가 -------------------------
model = SedimentDrift(loglevel=20)
# model = OceanDrift(loglevel=20)
#=============================================================================

from opendrift.readers import reader_netCDF_CF_generic
model.add_reader([reader_netCDF_CF_generic.Reader(ocean_nc_file),
                        reader_netCDF_CF_generic.Reader(wind_nc_file),
                        reader_netCDF_CF_generic.Reader(bathymetry_nc_file)])


# runs = []
seafloor_actions = ['previous', 'deactivate', 'lift_to_seafloor']

for seafloor_action in seafloor_actions:
    
    model.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')

    model.set_config('seed:wind_drift_factor', 0.02)
    model.set_config('drift:stokes_drift', True)


    model.set_config('drift:vertical_advection', True)

    model.set_config('drift:vertical_mixing', False)
    model.set_config('general:coastline_action', 'previous')

    # model.set_config('general:seafloor_action', 'deactivate')
    model.set_config('general:seafloor_action', seafloor_action)

    model.set_config('drift:horizontal_diffusivity', 100)
    model.set_config('drift:current_uncertainty', 0.2)
    model.set_config('drift:wind_uncertainty', 2)

# (수직 보간은 모델 내부에서 유효한 수심 범위 내에서 자동으로 적용됩니다.)

# ------------------------- STEP 3: 주기적 시딩 및 입자 개별 특성 부여 -------------------------
simulation_start = datetime(2024, 3, 1, 0, 0, 0)
simulation_end   = datetime(2024, 3, 10, 0, 0, 0)
seeding_end      = datetime(2024, 3, 5, 23, 59, 59)
seeding_interval = timedelta(hours=12)

total_hours = (simulation_end - simulation_start).total_seconds() / 3600
num_seeding_events = int(total_hours // 12) + 1

# 폐기물 유형별 침강속도 정의 (mean ± std)
debris_types = {
    '목재': {'mean': 0.175, 'std': 0.025},
    '의류/신발류': {'mean': 0.1, 'std': 0.03},
    '삼마 로프': {'mean': 0.15, 'std': 0.01},
    '알루미늄 캔': {'mean': 0.3, 'std': 0.05},
    '병류': {'mean': 1.25, 'std': 0.2},
    '타이어': {'mean': 0.5, 'std': 0.1},
    '와이어 로프': {'mean': 0.6, 'std': 0.15},
}



SELECTED_DEBRIS = '목재'
params = debris_types[SELECTED_DEBRIS]
mean_v = params['mean']
std_v = params['std']


for i in range(num_seeding_events):
    seed_time = simulation_start + seeding_interval * i
    if seed_time > seeding_end:
        break
    lons, lats = generate_seed_points()
    terminal_velocities = np.random.normal(loc=-mean_v, scale=std_v, size=len(lons))

    print(f"{seed_time} 시딩: 입자 수 = {len(lons)}, 평균 terminal_velocity = {np.mean(terminal_velocities):.3f} m/s")
    model.seed_elements(lon=lons, lat=lats, time=seed_time, terminal_velocity=terminal_velocities)



# ------------------------- STEP 4: 시뮬레이션 실행 및 애니메이션 생성 -------------------------
# 시뮬레이션 내부 시간 단위를 1시간(3600초)으로 설정하여 내부 보간을 통한 정밀 계산을 진행
model.run(time_step=3600, time_step_output=3600, duration=simulation_end - simulation_start)



# model.animation(color='z', fast=False, buffer=0.01)
# model.animation(color='z', fast=False, buffer=0.01,
#                 background='sea_floor_depth_below_sea_level',
#                 vmin=0, vmax=300)
model.animation(
    legend=seafloor_actions,
    background='sea_floor_depth_below_sea_level',
    vmin=0,
    vmax=300,
    fast=False,
    buffer=0.01
)






# ------------------------- STEP 5: 최종 입자 위치 추출 및 히트맵 생성 -------------------------
# 침적 상태(status == 2)만 필터링
mask = model.elements.status == 2

# 침적 입자 정보 DataFrame 생성
final_df = pd.DataFrame({
    'lon': model.elements.lon[mask],
    'lat': model.elements.lat[mask],
    'z': model.elements.z[mask]
})


# # --- 시각화 ---
# fig, ax = plt.subplots(figsize=(10, 8))

# # 해안선 시각화 (좀 더 진하게 표시)
# coast.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=0.5, zorder=1)

# # 침적 입자 및 전체 입자
# # scatter_all = ax.scatter(model.elements.lon, model.elements.lat, c='lightgray', s=10, label='전체 입자', zorder=2)
# scatter_dep = ax.scatter(final_df['lon'], final_df['lat'], c='red', s=20, label='침적된 입자', zorder=3)

# # 범례
# # ax.legend(handles=[scatter_all, scatter_dep])
# ax.legend(handles=[scatter_dep])
# # 지도 줌 → 진해항 중심
# margin = 0.05
# ax.set_xlim(min(polygon_lons)-margin, max(polygon_lons)+margin)
# ax.set_ylim(min(polygon_lats)-margin, max(polygon_lats)+margin)

# # 꾸미기
# ax.set_title('침적 입자 시뮬레이션 결과 (정확한 해안선 + 진해항 영역)')
# ax.set_xlabel('경도')
# ax.set_ylabel('위도')
# ax.grid(True)
# plt.tight_layout()
# plt.show()

# # ------------------------- STEP 6: 침적 입자 히트맵 집계 및 CSV 저장 -------------------------

# # 격자 크기 설정 (도 단위)
# grid_size = 0.01  # 예: 0.01도 간격

# # 위경도 격자 셀에 매핑
# final_df['lon_bin'] = (final_df['lon'] // grid_size) * grid_size + grid_size / 2
# final_df['lat_bin'] = (final_df['lat'] // grid_size) * grid_size + grid_size / 2

# # 그룹화하여 셀별 입자 수 집계
# heatmap_df = final_df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')

# # 컬럼 이름 정리
# heatmap_df.rename(columns={'lat_bin': 'latitude', 'lon_bin': 'longitude'}, inplace=True)

# # 파일 저장
# output_filename = f'{SELECTED_DEBRIS}.csv'
# heatmap_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

# print(f"📁 침적 히트맵 데이터 저장 완료: {output_filename}")