

import os

os.environ['PROJ_LIB']  = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
os.environ['PROJ_DATA'] = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
#!/usr/bin/env python

from datetime import datetime, timedelta
import numpy as np
import random
from shapely.geometry import Polygon, Point, box
from opendrift.models.sedimentdrift import SedimentDrift
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
# SELECTED_REGION = "고흥군"
# selected_coords = regions_polygon_coords[SELECTED_REGION]
# polygon = Polygon(selected_coords)

# polygon_lons, polygon_lats = zip(*selected_coords)
# lon_min, lon_max = min(polygon_lons) - 0.06, max(polygon_lons) + 0.06
# lat_min, lat_max = min(polygon_lats) - 0.06, max(polygon_lats) + 0.06



lat_min, lat_max = 34.1, 34.9      # 위도 범위: 34.1° ~ 34.9°
lon_min, lon_max = 125.48, 128.0   # 경도 범위: 125.48° ~ 128.0°
# ------------------------- STEP 1: 시딩점 생성 함수 정의 -------------------------
def generate_seed_points(num_points=1000, seed=42):
    np.random.seed(seed)  # 항상 같은 난수 시퀀스 유지
    points = []

    while len(points) < num_points:
        lon = np.random.uniform(lon_min, lon_max)
        lat = np.random.uniform(lat_min, lat_max)
        pt = Point(lon, lat)

        # 폴리건 조건을 제거하고, 육지(land_union) 위를 피하여 바다에서만 시딩
        if not land_union.contains(pt):
            points.append(pt)

    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    return gdf.geometry.x.values, gdf.geometry.y.values

#=============================================================================
from datetime import timedelta
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.sedimentdrift import SedimentDrift

# 1) Reader 인스턴스 생성 (한 번만!)
reader_ocean = reader_netCDF_CF_generic.Reader(ocean_nc_file)
reader_wind  = reader_netCDF_CF_generic.Reader(wind_nc_file)
reader_bathy = reader_netCDF_CF_generic.Reader(bathymetry_nc_file)


# 3) 같은 인스턴스들을 모델에 추가
model = SedimentDrift(loglevel=20)
model.add_reader([reader_ocean, reader_wind,
                   reader_bathy])


# model = SedimentDrift(loglevel=20)

# from datetime import timedelta
# from opendrift.readers import reader_netCDF_CF_generic


# model.add_reader([reader_netCDF_CF_generic.Reader(ocean_nc_file),
#                         reader_netCDF_CF_generic.Reader(wind_nc_file),
#                         reader_netCDF_CF_generic.Reader(bathymetry_nc_file)])


model.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')

model.set_config('seed:wind_drift_factor', 0.02)
model.set_config('drift:stokes_drift', True)


model.set_config('drift:vertical_advection', True)

model.set_config('drift:vertical_mixing', False)
model.set_config('general:coastline_action', 'previous')

model.set_config('general:seafloor_action', 'deactivate')

model.set_config('drift:horizontal_diffusivity', 100)
model.set_config('drift:current_uncertainty', 0.2)
model.set_config('drift:wind_uncertainty', 2)

# (수직 보간은 모델 내부에서 유효한 수심 범위 내에서 자동으로 적용됩니다.)

# ------------------------- STEP 3: 주기적 시딩 및 입자 개별 특성 부여 -------------------------
simulation_start = datetime(2024, 3, 1, 0, 0, 0)
simulation_end   = datetime(2024, 6, 1, 0, 0, 0)
seeding_end      = datetime(2024, 5, 1, 0, 0, 0)
seeding_interval = timedelta(hours=12)

total_hours = (simulation_end - simulation_start).total_seconds() / 3600
num_seeding_events = int(total_hours // 3) + 1

# 폐기물 유형별 침강속도 정의 (mean ± std)
debris_types = {
    '목재류': {'mean': 0.16, 'std': 0.01},
    '알루미늄 캔류': {'mean': 0.2925, 'std': 0.0412},
    '타이어류': {'mean': 0.49, 'std': 0.0587},
    '로프류': {'mean': 0.48, 'std': 0.2743},
    '침투성 폐품류': {'mean': 0.0991, 'std': 0.0362},
}

confidence_intervals = {
    '목재류': [0.033, 0.287],
    '알루미늄 캔류': [0.258, 0.327],
    '타이어류': [0.430, 0.539],
    '로프류': [0.044, 1.066],
    '침투성 폐품류': [0.075, 0.123]
}

# 키 목록을 미리 뽑아두면 반복문 안에서 편리합니다.
debris_keys = list(debris_types.keys())

for i in range(num_seeding_events):
    # (1) 매 시드마다 debris_types의 키 중 하나를 무작위로 선택
    SELECTED_DEBRIS = random.choice(debris_keys)
    params = debris_types[SELECTED_DEBRIS]
    mean_v = params['mean']
    std_v  = params['std']

    # (2) 시드 시각 계산
    seed_time = simulation_start + seeding_interval * i
    if seed_time > seeding_end:
        break

    # (3) 시드 위치 생성
    lons, lats = generate_seed_points()
    # (4) 음수 평균을 주어 "아래로 가라앉는 속도" 랜덤 표본 생성
    low, high = confidence_intervals[SELECTED_DEBRIS]
    terminal_velocities = -np.clip(
        np.random.normal(loc=mean_v, scale=std_v, size=len(lons)),
        low, high
    )

    # (5) 디버깅용 출력
    print(f"{seed_time} 시딩 ({SELECTED_DEBRIS}): "
          f"입자 수 = {len(lons)}, "
          f"평균 terminal_velocity = {np.mean(terminal_velocities):.3f} m/s")

    # (6) 모델에 시딩
    seed_elements(
        lon=lons,      
        lat=lats,      
        time=seed_time,
        terminal_velocity=terminal_velocities
    )



# ------------------------- STEP 4: 시뮬레이션 실행 및 애니메이션 생성 -------------------------
# 시뮬레이션 내부 시간 단위를 1시간(3600초)으로 설정하여 내부 보간을 통한 정밀 계산을 진행
model.run(time_step=3600, time_step_output=3600, duration=simulation_end - simulation_start)



#입자색변함함
model.animation(color='moving', colorbar=False, legend=['Sedimented', 'Moving'], fast=True, buffer=.01)



# # ------------------------- STEP 5: 원하는 변수들만 DataFrame으로 추출 -------------------------
# # ------------------------- STEP 5: 원하는 변수들만 DataFrame으로 추출 -------------------------
vars_to_extract = [
    'time',
    'lon',
    'lat',
    'z',
    'status',
    'age',
    'beached',
    'sedimented',
    'origin_marker'
]

# 실제로 model.result 안에 있는 변수만 남기기 (없는 변수는 자동 제외)
available_vars = [v for v in vars_to_extract if v in model.result.data_vars]

# xarray Dataset → pandas DataFrame 변환
df_result = model.result[available_vars].to_dataframe().reset_index()

# ------------------------- 추가: NaN 행 제거 -------------------------
# 'lon', 'lat', 'z', 'status', 'origin_marker' 중 하나라도 NaN이 있으면 해당 행을 drop
df_result = df_result.dropna(subset=['lon', 'lat', 'z', 'status', 'origin_marker'])

# ------------------------- STEP 6: CSV로 저장 -------------------------
output_path = fr'.\sediment_selected_vars_all.csv'  # 원하는 경로와 파일명으로 변경하세요
df_result.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"선택한 변수들을 CSV로 저장했습니다: {output_path}")



