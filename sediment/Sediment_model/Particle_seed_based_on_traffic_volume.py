#!/usr/bin/env python
import os


os.environ['PROJ_LIB']  = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
os.environ['PROJ_DATA'] = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"

from datetime import timedelta, datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box, Polygon
from opendrift.models.sedimentdrift import SedimentDrift


plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# --- 파일 경로 설정 ---
# ocean_nc_file = r'C:\Users\HUFS\opendrift\examples\HYCOM\processed_current.nc'
ocean_nc_file = r'C:\Users\HUFS\Desktop\침적opendrift\HYCOM_0301_0603.nc'
wind_nc_file  = r'C:\Users\HUFS\Desktop\침적opendrift\wind.nc'
# wind_nc_file  = r'C:\Users\HUFS\Desktop\침적opendrift\바람데이터_3_9.nc'
bathymetry_nc_file = r'C:\Ocean-Solution-Association\tinppl\desktop\opendrfit_middle\rename_output.nc'
coastline_file = r"C:\Users\HUFS\Desktop\2024년 전국해안선.shp"
traffic_file = r".\교통량_202505192156.xlsx"

# --- 시딩 지역 설정 ---
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
selected_polygon = Polygon(selected_coords)
polygon_lons, polygon_lats = zip(*selected_coords)
lon_min, lon_max = min(polygon_lons)-0.03, max(polygon_lons)+0.03
lat_min, lat_max = min(polygon_lats)-0.03, max(polygon_lats)+0.03

# --- 해안선 데이터 ---
coast = gpd.read_file(coastline_file)
if coast.crs is None or coast.crs.to_string() != 'EPSG:4326':
    coast = coast.to_crs(epsg=4326)

# --- 교통량 데이터에서 시딩 좌표 생성 ---
df = pd.read_excel(traffic_file)
df[['lat', 'lon']] = df['위경도'].str.split(',', expand=True).astype(float)
df = df[(df['lon'] >= lon_min) & (df['lon'] <= lon_max) & (df['lat'] >= lat_min) & (df['lat'] <= lat_max)].copy()
df['normalized_weight'] = df['교통량(척)'] / df['교통량(척)'].max()
MAX_PARTICLES = 50
df['seed_count'] = (df['normalized_weight'] * MAX_PARTICLES).astype(int).clip(lower=1)

delta_lon = 0.0125  # ≈ 2.8km
delta_lat = 0.01    # ≈ 2.3km

def generate_seed_points_from_traffic():
    all_lons, all_lats = [], []
    for _, row in df.iterrows():
        lons = np.random.uniform(row['lon'] - delta_lon, row['lon'] + delta_lon, row['seed_count'])
        lats = np.random.uniform(row['lat'] - delta_lat, row['lat'] + delta_lat, row['seed_count'])
        all_lons.extend(lons)
        all_lats.extend(lats)
    return np.array(all_lons), np.array(all_lats)

# --- 모델 초기화 ---
model = SedimentDrift(loglevel=20)
model.add_readers_from_list([ocean_nc_file, wind_nc_file, bathymetry_nc_file])
model.set_config('vertical_mixing:diffusivitymodel', 'windspeed_Large1994')
model.set_config('drift:horizontal_diffusivity', 50)
model.set_config('drift:current_uncertainty', 0.2)
model.set_config('drift:wind_uncertainty', 1.5)
model.set_config('general:seafloor_action', 'deactivate')


# --- 시뮬레이션 설정 ---
simulation_start = datetime(2024, 3, 1)
simulation_end   = datetime(2024, 3, 10)
seeding_interval = timedelta(hours=12)
num_events = int((simulation_end - simulation_start) / seeding_interval) + 1

for i in range(num_events):
    seed_time = simulation_start + i * seeding_interval
    lons, lats = generate_seed_points_from_traffic()
    terminal_velocity = np.clip(np.random.normal(0.05, 0.02, len(lons)), 0, None)
    model.seed_elements(lon=lons, lat=lats, time=seed_time, terminal_velocity=terminal_velocity)

# --- 시뮬레이션 실행 ---
model.run(time_step=1800, time_step_output=10800, duration=simulation_end - simulation_start)
model.animation(color='z', fast=False, buffer=0.01)

# --- 결과 분석 및 히트맵 ---
final_df = pd.DataFrame({
    'lon': model.elements.lon,
    'lat': model.elements.lat,
    'z': model.elements.z
})


# deposits = final_df[final_df['z'] < 0]
deposits = final_df[model.elements.status == 2]



from matplotlib.colors import LogNorm

heatmap, xedges, yedges = np.histogram2d(deposits['lon'], deposits['lat'], bins=200)

# 0을 작은 값으로 바꿔서 로그 오류 방지
heatmap[heatmap == 0] = 1e-3

plt.figure(figsize=(8, 6))
plt.imshow(heatmap.T, origin='lower',
           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           aspect='auto',
           norm=LogNorm())
plt.colorbar(label="입자 밀도 (로그 스케일)")




# heatmap, xedges, yedges = np.histogram2d(deposits['lon'], deposits['lat'], bins=200)
# plt.figure(figsize=(8, 6))
# plt.imshow(heatmap.T, origin='lower',
#            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
#            aspect='auto', norm=plt.matplotlib.colors.LogNorm())
# plt.colorbar(label="입자 밀도 (로그 스케일)")



plt.xlabel("경도")
plt.ylabel("위도")
plt.title("침전 입자 밀집 구역 히트맵")
bbox = box(lon_min, lat_min, lon_max, lat_max)
coast_clipped = gpd.clip(coast, bbox)
coast_clipped.plot(ax=plt.gca(), color='lightgray', zorder=10)
plt.show()
