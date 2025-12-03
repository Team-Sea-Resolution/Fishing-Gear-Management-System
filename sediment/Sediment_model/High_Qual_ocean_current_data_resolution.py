import os



os.environ['PROJ_LIB']  = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"
os.environ['PROJ_DATA'] = r"C:\Users\HUFS\anaconda3\envs\opendrift_env_sediment\Library\share\proj"

import numpy as np
import xarray as xr
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from pyproj import Transformer


# 파일 경로 (Windows 기준)
SURF_FILE  = r'.\KHOA_nc_data\sediment_data_uv.nc'
HYCOM_FILE = r'.\HYCOM_0301_0603.nc'
BATHY_FILE = r'.\bathymetry_xyz.nc'
COAST_SHP  = r"C:\Users\HUFS\Desktop\sediment_final\dataset\ne_land\ne_10m_land.shp"
OUT_FILE   = r'C:\Users\HUFS\Desktop\sediment_opendrift\output\22222222222222222222.nc'
START_DATE = '2024-03-01'
END_DATE   = '2024-06-01'

SURF_U_VAR  = 'eastward_sea_water_velocity'
SURF_V_VAR  = 'northward_sea_water_velocity'
HYCOM_U_VAR = 'water_u'
HYCOM_V_VAR = 'water_v'
DEPTH_DIM   = 'depth'

# 수심 기반 감쇠 비율 함수
def attenuation_ratio(depth):
    if np.isnan(depth) or depth <= 0:
        return 1.0
    elif depth < 10:
        return 1.0
    elif depth < 30:
        return 0.9
    elif depth < 60:
        return 0.8
    elif depth < 100:
        return 0.7
    else:
        return 0.6

# 데이터 불러오기
ds_surf  = xr.open_dataset(SURF_FILE)
ds_hycom = xr.open_dataset(HYCOM_FILE)
ds_bathy = xr.open_dataset(BATHY_FILE)
ds_hycom = ds_hycom.sel(time=~ds_hycom.get_index("time").duplicated())

# 시간 필터링
hycom_times = ds_hycom['time'].sel(time=slice(START_DATE, END_DATE))
U0 = ds_surf[SURF_U_VAR].interp(time=hycom_times)
V0 = ds_surf[SURF_V_VAR].interp(time=hycom_times)
U0, V0 = U0.isel(time=np.unique(U0['time'], return_index=True)[1]), V0.isel(time=np.unique(V0['time'], return_index=True)[1])
Hy_u = ds_hycom[HYCOM_U_VAR].sel(time=U0['time'], method='nearest')
Hy_v = ds_hycom[HYCOM_V_VAR].sel(time=U0['time'], method='nearest')

# 비율 함수 계산
R_u = Hy_u / Hy_u.isel({DEPTH_DIM: 0})
R_v = Hy_v / Hy_v.isel({DEPTH_DIM: 0})
R_u = R_u.sel(time=U0.time)
R_v = R_v.sel(time=U0.time)

# 선형 + 최근접 보간
coords_hr = {'lat': U0['lat'], 'lon': U0['lon']}
R_u_filled = R_u.interp(coords_hr, method='linear').combine_first(R_u.interp(coords_hr, method='nearest'))
R_v_filled = R_v.interp(coords_hr, method='linear').combine_first(R_v.interp(coords_hr, method='nearest'))

# 수심 감쇠 계산
depth_bathy = ds_bathy["sea_floor_depth_below_sea_level"]
attenuation_grid = np.vectorize(attenuation_ratio)(depth_bathy.values)
attenuation_da = xr.DataArray(attenuation_grid, dims=("lat", "lon"), coords={"lat": ds_bathy["lat"], "lon": ds_bathy["lon"]})

# EPSG:5179 → EPSG:4326 위경도 변환
transformer = Transformer.from_crs("epsg:5179", "epsg:4326", always_xy=True)
x, y = ds_bathy['lon'].values, ds_bathy['lat'].values
xx, yy = np.meshgrid(x, y)
lon_wgs84, lat_wgs84 = transformer.transform(xx, yy)
points = np.column_stack([lon_wgs84.ravel(), lat_wgs84.ravel()])
values = attenuation_da.values.ravel()

# 표층 격자에 보간
lon2d, lat2d = np.meshgrid(U0['lon'].values, U0['lat'].values)
grid_interp = griddata(points, values, (lon2d, lat2d), method='nearest')
attenuation_on_surfgrid = xr.DataArray(grid_interp, dims=('lat', 'lon'), coords={'lat': U0['lat'], 'lon': U0['lon']})

# NaN 채우기
R_u_filled = R_u_filled.fillna(attenuation_on_surfgrid)
R_v_filled = R_v_filled.fillna(attenuation_on_surfgrid)

# 수심별 해류 재구성
U_rec = R_u_filled * U0
V_rec = R_v_filled * V0

# 해안선 거리 기반 마스크 생성
coast = gpd.read_file(COAST_SHP)
coast_points = []
for geom in coast.geometry:
    if geom.geom_type in ['Polygon', 'LineString']:
        coast_points.extend(np.array(geom.exterior.coords if geom.geom_type == 'Polygon' else geom.coords))
    elif geom.geom_type in ['MultiPolygon', 'MultiLineString']:
        for part in geom.geoms:
            coast_points.extend(np.array(part.exterior.coords if part.geom_type == 'Polygon' else part.coords))
coast_points = np.array(coast_points)

lon2d, lat2d = np.meshgrid(ds_surf['lon'].values, ds_surf['lat'].values)
grid_points = np.column_stack([lon2d.ravel(), lat2d.ravel()])
tree = cKDTree(coast_points)
_, idx = tree.query(grid_points, k=1)
nearest = coast_points[idx]

def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

dist2d = haversine_np(grid_points[:, 1], grid_points[:, 0], nearest[:, 1], nearest[:, 0]).reshape(lat2d.shape)
mask2d = np.minimum(dist2d / 500.0, 1.0)
mask3d = np.broadcast_to(mask2d[np.newaxis, np.newaxis, :, :], (len(U0['time']), len(ds_hycom[DEPTH_DIM]), len(U0['lat']), len(U0['lon'])))

# 마스크 적용 및 저장
U_masked = U_rec * mask3d
V_masked = V_rec * mask3d
ds_final = xr.Dataset(
    {
        'eastward_current':  (('time', DEPTH_DIM, 'lat', 'lon'), U_masked.data),
        'northward_current': (('time', DEPTH_DIM, 'lat', 'lon'), V_masked.data)
    },
    coords={
        'time':  U0['time'],
        'depth': ds_hycom[DEPTH_DIM],
        'lat':   U0['lat'],
        'lon':   U0['lon']
    }
)

ds_final.to_netcdf(OUT_FILE)
print("✅ Saved to", OUT_FILE)
