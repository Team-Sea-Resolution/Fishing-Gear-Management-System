import os

os.environ['PROJ_LIB']  = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
os.environ['PROJ_DATA'] = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"

from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from opendrift.models.sedimentdrift import SedimentDrift

# 기본 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 좌표 범위
lat_min, lat_max = 34.1, 34.9
lon_min, lon_max = 125.48, 128.0

# 시딩 좌표
manual_seed_coords = [
    (126.1, 34.3),  # 진도 서쪽 바다
    (126.7, 34.4),  # 해남 남서쪽 해역
    (127.2, 34.5),  # 여수 앞바다
    (127.6, 34.6),  # 남해도 서쪽 연안
    (127.85, 34.65)  # 고흥 앞바다
]


# 파일 경로
ocean_nc_file = r'.\new_dataset\new_sediment_data_final_uv.nc'
wind_nc_file = r'.\new_dataset\new_wind.nc'
bathymetry_nc_file = r'.\new_dataset\new_BADA2024.nc'
land_shp = './dataset/ne_land/ne_10m_land.shp'  # 위치에 맞게 조정
coast = gpd.read_file(land_shp)
land_union = coast.unary_union





# 모델 초기화
from opendrift.readers import reader_netCDF_CF_generic
reader_ocean = reader_netCDF_CF_generic.Reader(ocean_nc_file)
reader_wind  = reader_netCDF_CF_generic.Reader(wind_nc_file)
reader_bathy = reader_netCDF_CF_generic.Reader(bathymetry_nc_file)


model = SedimentDrift(loglevel=20)
model.add_reader([reader_ocean, reader_wind,
                   reader_bathy])
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


# 시뮬레이션 설정
simulation_start = datetime(2024, 3, 1)
simulation_end = datetime(2024, 3, 10)

# 입자 시딩
seed_lons, seed_lats = zip(*manual_seed_coords)

debris_types = {
    '자망': {'mean': 0.5, 'std': 0.05}
}
SELECTED_DEBRIS = '자망'
params = debris_types[SELECTED_DEBRIS]
mean_v = params['mean']
std_v = params['std']
terminal_velocity = np.random.normal(loc=-mean_v, scale=std_v, size=len(seed_lons))
model.seed_elements(lon=seed_lons, lat=seed_lats, time=simulation_start, terminal_velocity=terminal_velocity)

# 시뮬레이션 실행
model.run(time_step=1800, time_step_output=10800, duration=simulation_end - simulation_start)
model.animation(color='moving', colorbar=False, legend=['Sedimented', 'Moving'], fast=True, buffer=.01)

# 최종 위치 추출 (history 사용)
final_df = pd.DataFrame({
    'id': model.elements.ID,
    'lon': model.elements.lon,
    'lat': model.elements.lat,
    'z': model.elements.z,
    'status': model.elements.status
})


# 입자 이동 결과 출력
print("입자 이동 결과 요약:")
print("───────────────────────────────────────────────")
for i, row in final_df.iterrows():
    start_lon, start_lat = manual_seed_coords[i]
    end_lon, end_lat = row['lon'], row['lat']
    z = row['z']
    status = row['status']
    print(f"[입자 {row['id']}]")
    print(f"  • 시작 위치  : 위도 {start_lat:.4f}, 경도 {start_lon:.4f}")
    print(f"  • 종료 위치  : 위도 {end_lat:.4f}, 경도 {end_lon:.4f}")
    print(f"  • 수직 위치 z: {z:.2f} m")
    print(f"  • 상태 코드  : {status} (2: 침강됨, 1: 표류 중, 0: 비활성)")
    print("───────────────────────────────────────────────")

# 시각화
plt.figure(figsize=(8, 6))
plt.scatter(seed_lons, seed_lats, color='blue', label='시딩 위치')
plt.scatter(final_df['lon'], final_df['lat'], color='red', label='최종 위치')
plt.xlim(lon_min, lon_max)
plt.ylim(lat_min, lat_max)
plt.xlabel("경도")
plt.ylabel("위도")
plt.title("입자 시작/종료 위치 비교")
bbox = box(lon_min-3, lat_min-3, lon_max+3, lat_max+3)
coast_clipped = gpd.clip(coast, bbox)
coast_clipped.plot(ax=plt.gca(), color='lightgray', zorder=0)
plt.legend()
plt.grid()
plt.show()
