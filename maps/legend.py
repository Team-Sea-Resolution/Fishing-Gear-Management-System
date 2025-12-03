import os
import io
import base64
import numpy as np
import pandas as pd
import json
import csv
import math
import tempfile
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic
from math import radians, sin, cos, sqrt, asin
from tqdm import tqdm
from datetime import datetime, timedelta
# from sqlalchemy import create_engine
import xarray as xr
import cdsapi
import requests
from scipy.interpolate import griddata
from geopy.distance import geodesic
from maps.models import SystemData
from urllib.parse import urlencode



try:
    from django.conf import settings
    BASE_DIR = settings.BASE_DIR
except ImportError:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_output_path(filename, subfolder):
    output_dir = os.path.join(BASE_DIR, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

def load_db_to_df(report2_id):
    qs = SystemData.objects.using('gpsdb').filter(report2_id=report2_id).order_by('time_stamp')
    if not qs.exists():
        raise ValueError(f"❌ report2_id={report2_id} 데이터 없음")
    df = pd.DataFrame.from_records(qs.values(
        'report2_id', 'buyer_id', 'time_stamp', 'lat', 'lon', 'sog', 'cog', 'press'
    ))
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce')
    return df

# ─────────────────────────────────────────────────────
# 상수 정의
# ─────────────────────────────────────────────────────
TARGET_SEQ       = [3,0,3,1,3]                 # 시퀀스 탐색용
ISSUE_SEQ        = [3,0,3,2,3]

# ─────────────────────────────────────────────────────
# 1) 시퀀스 탐색용 헬퍼
# ─────────────────────────────────────────────────────
def find_sequence_groups(behaviors, target=TARGET_SEQ):
    # 연속 중복 제거 후 그룹 시퀀스에서 부분열 위치 탐색
    grp = [behaviors[0]]
    for b in behaviors[1:]:
        if b != grp[-1]:
            grp.append(b)
    n,m = len(grp), len(target)
    for i in range(n-m+1):
        if grp[i:i+m] == target:
            return i, i+m
    return None

def locate_sequence(df, target):
    raw = df['press'].tolist()
    if len(raw) < len(target): 
        return None
    # 그룹별 인덱스 매핑
    grp = [raw[0]]; starts=[0]; prev=raw[0]
    for i,b in enumerate(raw[1:], start=1):
        if b != prev:
            grp.append(b)
            starts.append(i)
            prev = b
    loc = find_sequence_groups(grp, target)
    if not loc:
        return None
    i0,i1 = loc
    start_idx = starts[i0]
    end_idx   = (starts[i1]-1) if i1<len(starts) else len(raw)-1
    return start_idx, end_idx

# ─────────────────────────────────────────────
# 거리 함수
# ─────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    r = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return r * c

def calculate_simulation_error(yang_start, yang_end, sim_start, sim_end):
    candidates_sim = [
        (yang_start, sim_start),
        (yang_start, sim_end),
        (yang_end, sim_start),
        (yang_end, sim_end)
    ]
    yang_sim_dis, yang_spot, sim_spot = min(
        [(haversine_distance(ya[0], ya[1], si[0], si[1]), ya, si) for ya, si in candidates_sim],
        key=lambda x: x[0]
    )
    return {
        'yang_sim_dis': yang_sim_dis,
        'yang_spot': yang_spot,
        'sim_spot': sim_spot
    }

# ─────────────────────────────────────────────
# 2. 가장 가까운 관측소 선택 함수
# ─────────────────────────────────────────────
def get_sorted_stations(station_df, lat, lon):
    station_df = station_df.copy()
    station_df['distance'] = station_df.apply(
        lambda row: geodesic((row['lat'], row['lon']), (lat, lon)).km, axis=1
    )
    return station_df.sort_values('distance')

# ─────────────────────────────────────────────
# 3. KHOA 해류 API 호출 및 NetCDF 저장
# ─────────────────────────────────────────────
def fetch_khoa_tidal_uv(time_list, lon_min, lon_max, lat_min, lat_max, lon_grid, lat_grid, service_key):
    all_data = []
    base_url = "http://www.khoa.go.kr/api/oceangrid/tidalCurrentAreaGeoJson/search.do"
    for t in time_list:
        params = {
            "DataType": "tidalCurrentAreaGeoJson",
            "ServiceKey": service_key,
            "Date": t.strftime("%Y%m%d"),
            "Hour": t.strftime("%H"),
            "Minute": "00",
            "MinX": lon_min, "MaxX": lon_max,
            "MinY": lat_min, "MaxY": lat_max,
            "Scale": 1000000
        }
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200 or not resp.text.startswith('{'):
            print(f"❌ API 실패({resp.status_code}) at {t}")
            continue

        for feat in resp.json().get('features', []):
            p = feat['properties']
            lat, lon = p.get('lat'), p.get('lon')
            spd_raw, direction = p.get('current_speed'), p.get('current_direct')
            if None in (lat, lon, spd_raw, direction):
                continue
            spd = spd_raw / 100.0
            radian = math.radians(direction)
            u = spd * math.sin(radian)   # 동쪽 성분 (x축)
            v = spd * math.cos(radian)   # 북쪽 성분 (y축)
            all_data.append({
                'time': t,
                'lat': lat,
                'lon': lon,
                'u': u,
                'v': v
            })
    df_all = pd.DataFrame(all_data)
    df_all['time'] = pd.to_datetime(df_all['time']).dt.tz_localize(None)
    times = np.array(sorted(df_all['time'].unique()), dtype='datetime64[ns]')
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    u_interp, v_interp = [], []
    for t in times:
        t = pd.Timestamp(t).replace(tzinfo=None) 
        sub = df_all[df_all['time'] == t]
        pts = sub[['lon', 'lat']].values
        u_grid = griddata(pts, sub['u'], (lon_mesh, lat_mesh), method='linear')
        v_grid = griddata(pts, sub['v'], (lon_mesh, lat_mesh), method='linear')
        u_interp.append(u_grid)
        v_interp.append(v_grid)

    ds = xr.Dataset(
        {
            'x_sea_water_velocity': (['time', 'lat', 'lon'], np.array(u_interp)),
            'y_sea_water_velocity': (['time', 'lat', 'lon'], np.array(v_interp))
        },
        coords={
            'time': times,
            'lat': lat_grid,
            'lon': lon_grid
        },  
        attrs={
            'title': "KHOA 해류 예측 데이터 (ub/vb 복정 적용)",
            'source': "tidalCurrentAreaGeoJson API"
        }
    )
    ds['x_sea_water_velocity'].attrs.update(standard_name="x_sea_water_velocity", units="m s-1")
    ds['y_sea_water_velocity'].attrs.update(standard_name="y_sea_water_velocity", units="m s-1")

    return ds

def fetch_temperature(time_list, lat, lon, service_key):
    # 조위관측소 데이터
    tide_data = [
        ["DT_0063", "가덕도", 35.024, 128.81],
        ["DT_0031", "거문도", 34.028, 127.308],
        ["DT_0029", "거제도", 34.801, 128.699],
        ["DT_0026", "고흥발포", 34.481, 127.342],
        ["DT_0018", "군산", 35.975, 126.563],
        ["DT_0017", "대산", 37.007, 126.352],
        ["DT_0062", "마산", 35.197, 128.576],
        ["DT_0023", "모슬포", 33.214, 126.251],
        ["DT_0007", "목포", 34.779, 126.375],
        ["DT_0006", "묵호", 37.55, 129.116],
        ["DT_0025", "보령", 36.406, 126.486],
        ["DT_0005", "부산", 35.096, 129.035],
        ["DT_0061", "삼천포", 34.924, 128.069],
        ["DT_0094", "서거차도", 34.251, 125.915],
        ["DT_0010", "서귀포", 33.24, 126.561],
        ["DT_0022", "성산포", 33.474, 126.927],
        ["DT_0012", "속초", 38.207, 128.594],
        ["IE_0061", "신안가거초", 33.941, 124.592],
        ["DT_0008", "안산", 37.192, 126.647],
        ["DT_0067", "안흥", 36.674, 126.129],
        ["DT_0037", "어청도", 36.117, 125.984],
        ["DT_0016", "여수", 34.747, 127.765],
        ["IE_0062", "옹진소청초", 37.423, 124.738],
        ["DT_0027", "완도", 34.315, 126.759],
        ["DT_0013", "울릉도", 37.491, 130.913],
        ["DT_0020", "울산", 35.501, 129.387],
        ["IE_0060", "이어도", 32.122, 125.182],
        ["DT_0001", "인천", 37.451, 126.592],
        ["DT_0004", "제주", 33.527, 126.543],
        ["DT_0028", "진도", 34.377, 126.308],
        ["DT_0021", "추자도", 33.961, 126.3],
        ["DT_0014", "통영", 34.827, 128.434],
        ["DT_0002", "평택", 36.966, 126.822],
        ["DT_0091", "포항", 36.051, 129.376],
        ["DT_0011", "후포", 36.677, 129.453],
        ["DT_0035", "흑산도", 34.684, 125.435],
    ]

    df_tide = pd.DataFrame(tide_data, columns=["obs_code", "name", "lat", "lon"])


    # 해상관측부이 데이터
    buoy_data = [
        ["TW_0088", "감천항", 35.052, 129.003],
        ["TW_0077", "경인항", 37.523, 126.592],
        ["TW_0089", "경포대해수욕장", 37.808, 128.931],
        ["TW_0095", "고래불해수욕장", 36.58, 129.454],
        ["TW_0074", "광양항", 34.859, 127.792],
        ["TW_0072", "군산항", 35.984, 126.508],
        ["TW_0091", "낙산해수욕장", 38.122, 128.65],
        ["KG_0025", "남해동부", 34.222, 128.419],
        ["TW_0069", "대천해수욕장", 36.274, 126.457],
        ["TW_0085", "마산항", 35.103, 128.631],
        ["TW_0094", "망상해수욕장", 37.616, 129.103],
        ["TW_0086", "부산항신항", 35.043, 128.761],
        ["TW_0079", "상왕등도", 35.652, 126.194],
        ["TW_0081", "생일도", 34.258, 126.96],
        ["TW_0093", "속초해수욕장", 38.198, 128.631],
        ["TW_0083", "여수항", 34.794, 127.808],
        ["TW_0078", "완도항", 34.325, 126.763],
        ["TW_0080", "우이도", 34.543, 125.802],
        ["KG_0101", "울릉도북동", 38.007, 131.552],
        ["KG_0102", "울릉도북서", 37.742, 130.601],
        ["TW_0076", "인천항", 37.389, 126.533],
        ["KG_0021", "제주남부", 32.09, 126.965],
        ["KG_0028", "제주해협", 33.7, 126.59],
        ["TW_0075", "중문해수욕장", 33.234, 126.409],
        ["TW_0082", "태안항", 37.006, 126.27],
        ["TW_0084", "통영항", 34.773, 128.46],
        ["TW_0070", "평택당진항", 37.136, 126.54],
        ["HB_0002", "한수원_고리", 35.318, 129.314],
        ["HB_0001", "한수원_기장", 35.182, 129.235],
        ["HB_0009", "한수원_나곡", 37.119, 129.395],
        ["HB_0008", "한수원_덕천", 37.1, 129.404],
        ["HB_0007", "한수원_온양", 37.019, 129.425],
        ["HB_0003", "한수원_진하", 35.384, 129.368],
    ]

    df_buoy = pd.DataFrame(buoy_data, columns=["obs_code", "name", "lat", "lon"])

    total_stations = pd.concat([df_tide, df_buoy], ignore_index=True)
    sorted_stations = get_sorted_stations(total_stations, lat, lon)

    temp_records = []
    for _, row in sorted_stations.iterrows():
        obs_code = row['obs_code']
        data_type = "tideObsTemp" if obs_code.startswith("DT") or obs_code.startswith("IE") else "tidalBuTemp"
        url_with_key = f"http://www.khoa.go.kr/api/oceangrid/{data_type}/search.do?ServiceKey={service_key}"

        temp_records.clear()
        for date_str in sorted(set(t.strftime('%Y%m%d') for t in time_list)):
            params = {
                "ObsCode": obs_code,
                "Date": date_str,
                "ResultType": "json"
            }
            r = requests.get(url_with_key, params=params)
            if not r.ok:
                continue
            result_json = r.json()
            if result_json.get("result", {}).get("error") == "No search data":
                continue

            for rec in result_json.get("result", {}).get("data", []):
                try:
                    temp_records.append({
                        "time": pd.to_datetime(rec["record_time"]),
                        "sea_water_temperature": float(rec["water_temp"])
                    })
                except:
                    continue

        if temp_records:
            print(f"✅ 수온 데이터 사용 관측소: {row['name']} ({obs_code})")
            break  # ✔️ 한 관측소에서 데이터 수집되면 종료

    if not temp_records:
        print("⚠️ 수온 데이터 없음 (모든 관측소 시도 실패)")

    return pd.DataFrame(temp_records)



def fetch_salinity(time_list, lat, lon, service_key):
    stations = [
    ["DT_0063", "가덕도", 35.024, 128.81],
    ["DT_0031", "거문도", 34.028, 127.308],
    ["DT_0029", "거제도", 34.801, 128.699],
    ["DT_0026", "고흥발포", 34.481, 127.342],
    ["DT_0018", "군산", 35.975, 126.563],
    ["DT_0062", "마산", 35.197, 128.576],
    ["DT_0023", "모슬포", 33.214, 126.251],
    ["DT_0007", "목포", 34.779, 126.375],
    ["DT_0006", "묵호", 37.55, 129.116],
    ["DT_0025", "보령", 36.406, 126.486],
    ["DT_0005", "부산", 35.096, 129.035],
    ["DT_0061", "삼천포", 34.924, 128.069],
    ["DT_0094", "서거차도", 34.251, 125.915],
    ["DT_0010", "서귀포", 33.24, 126.561],
    ["DT_0022", "성산포", 33.474, 126.927],
    ["DT_0012", "속초", 38.207, 128.594],
    ["IE_0061", "신안가거초", 33.941, 124.592],
    ["DT_0008", "안산", 37.192, 126.647],
    ["DT_0067", "안흥", 36.674, 126.129],
    ["DT_0037", "어청도", 36.117, 125.984],
    ["DT_0016", "여수", 34.747, 127.765],
    ["IE_0062", "옹진소청초", 37.423, 124.738],
    ["DT_0027", "완도", 34.315, 126.759],
    ["DT_0013", "울릉도", 37.491, 130.913],
    ["DT_0020", "울산", 35.501, 129.387],
    ["IE_0060", "이어도", 32.122, 125.182],
    ["DT_0001", "인천", 37.451, 126.592],
    ["DT_0004", "제주", 33.527, 126.543],
    ["DT_0028", "진도", 34.377, 126.308],
    ["DT_0021", "추자도", 33.961, 126.3],
    ["DT_0014", "통영", 34.827, 128.434],
    ["DT_0091", "포항", 36.051, 129.376],
    ["DT_0011", "후포", 36.677, 129.453],
    ["DT_0035", "흑산도", 34.684, 125.435]
    ]
    station_df = pd.DataFrame(stations, columns=['obs_code', 'name', 'lat', 'lon'])
    sorted_stations = get_sorted_stations(station_df, lat, lon)

    url = "http://www.khoa.go.kr/api/oceangrid/tideObsSalt/search.do"
    url_with_key = f"{url}?ServiceKey={service_key}"
    sal_records = []

    for _, row in sorted_stations.iterrows():
        obs_code = row['obs_code']
        sal_records.clear()

        for date_str in sorted(set(t.strftime('%Y%m%d') for t in time_list)):
            params = {
                "ObsCode": obs_code,
                "Date": date_str,
                "ResultType": "json"
            }
            r = requests.get(url_with_key, params=params)
            if not r.ok:
                continue

            result_json = r.json()
            if result_json.get("result", {}).get("error") == "No search data":
                continue

            for d in result_json.get("result", {}).get("data", []):
                try:
                    sal_records.append({
                        "time": pd.to_datetime(d['record_time']),
                        "sea_water_salinity": float(d['salinity'])
                    })
                except:
                    continue

        if sal_records:
            print(f"✅ 염분 데이터 사용 관측소: {row['name']} ({obs_code})")
            break

    if not sal_records:
        print("⚠️ 염분 데이터 없음 (모든 관측소 시도 실패)")

    return pd.DataFrame(sal_records)


def fetch_all_khoa(time_list, lon_min, lon_max, lat_min, lat_max,
                   lon_grid, lat_grid, service_key):

    # 1. 해류 데이터 (기존 방식 유지)
    ds = fetch_khoa_tidal_uv(time_list, lon_min, lon_max, lat_min, lat_max,
                       lon_grid, lat_grid, service_key)

    # 2. 기준 시간 및 공간 정보
    ds_time = pd.to_datetime(ds['time'].values).tz_localize(None).to_numpy(dtype='datetime64[ns]')
    lat_vals = ds['lat'].values
    lon_vals = ds['lon'].values

    # 3. 중심 좌표 및 날짜 리스트 생성
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    day_time_list = pd.to_datetime(sorted(set(t.normalize() for t in time_list))).tz_localize(None)

    # 4. 수온/염분 요청 (분 단위 데이터 → 하루 단위 요청)
    temp_df = fetch_temperature(day_time_list, center_lat, center_lon, service_key)
    sal_df  = fetch_salinity(day_time_list, center_lat, center_lon, service_key)

    # 5. 시간 기준 최근접 보간 (공간 동일값으로 확장)
    def expand_to_grid(df, var_name):

        df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
        ts = df.set_index('time')[var_name]
        if ts.index.has_duplicates:
            dup_idx = ts.index[ts.index.duplicated()].unique()
            print(f"⚠️ {var_name} 중복 인덱스 발견! 개수: {len(dup_idx)}")
            print(dup_idx)
            print(df[df['time'].isin(dup_idx)])
            ts = ts[~ts.index.duplicated(keep='first')]
        ts_interp = ts.reindex(ds_time, method='nearest', tolerance=pd.Timedelta('6H'))
        ts_interp = ts_interp.ffill().bfill()

        # (time, lat, lon) → 전체 공간에 동일한 값
        var_3d = np.broadcast_to(ts_interp.values[:, np.newaxis, np.newaxis],
                                 (len(ds_time), len(lat_vals), len(lon_vals)))
        return var_3d

    temp_grid = expand_to_grid(temp_df, 'sea_water_temperature')
    sal_grid  = expand_to_grid(sal_df, 'sea_water_salinity')

    # 6. dataset에 삽입
    ds['sea_water_temperature'] = (('time', 'lat', 'lon'), temp_grid)
    ds['sea_water_salinity'] = (('time', 'lat', 'lon'), sal_grid)

    ds['sea_water_temperature'].attrs.update(standard_name="sea_water_temperature", units="degree_Celsius")
    ds['sea_water_salinity'].attrs.update(standard_name="sea_water_salinity", units="psu")

    if not np.issubdtype(ds['time'].dtype, np.datetime64):
        ds = ds.assign_coords(time=pd.to_datetime(ds['time'].values).to_numpy(dtype='datetime64[ns]'))

    return ds



# fetch_hycom_uv 함수는 HYCOM 서버에서 해양 데이터를 다운로드하고 결합합니다.
def fetch_hycom_uv(time_list, lon_min, lon_max, lat_min, lat_max):
    """
    3시간 간격의 HYCOM uv3z 데이터와 1시간 간격의 sur 데이터를 결합하여
    1시간 간격의 보간된 uv 데이터를 생성합니다.
    
    Args:
        time_list (list): datetime 객체 리스트.
        lon_min, lon_max (float): 경도 범위.
        lat_min, lat_max (float): 위도 범위.
        
    Returns:
        xarray.Dataset: 1시간 간격의 보간된 데이터를 포함하는 xarray Dataset 객체.
        
    Raises:
        Exception: 데이터 다운로드에 실패하거나 xarray Dataset을 열지 못할 경우.
    """
    
    # 1년 전으로 시간 조정 (사용자의 요청에 따라 추가)
    time_list = [t - timedelta(days=365*2) for t in time_list]

    try:
        data_year = time_list[0].year
    except (IndexError, AttributeError):
        print("time_list가 비어 있거나 올바른 형식이 아닙니다. 현재 연도를 사용합니다.")
        data_year = datetime.now().year
    
    # HYCOM 데이터가 제공되는 연도 범위를 확인합니다.
    # GLBy0.08/expt_93.0 아카이브는 2018년 12월 4일 ~ 2024년 9월 4일 데이터만 제공합니다.
    supported_start_year = 2018
    supported_end_year = 2024
    if not (supported_start_year <= data_year <= supported_end_year):
        print(f"오류: {data_year}년 데이터는 HYCOM 서버에서 지원하지 않습니다.")
        print(f"지원되는 연도 범위: {supported_start_year}년 ~ {supported_end_year}년")
        # return xr.Dataset()

    # 시간 목록을 datetime64로 변환하고 고유한 값으로 정렬
    ds_time = pd.to_datetime(time_list).to_numpy(dtype='datetime64[ns]')

    # 요청에 따라 시작 시간은 내림, 끝 시간은 올림 처리
    time_start_dt = pd.to_datetime(ds_time.min()).replace(minute=0, second=0, microsecond=0)
    time_end_dt = pd.to_datetime(ds_time.max()).replace(minute=0, second=0, microsecond=0)
    if time_end_dt < pd.to_datetime(ds_time.max()):
        time_end_dt += timedelta(hours=1)

    time_start_str = time_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    time_end_str = time_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # 다운로드 폴더 지정
    output_dir = "hycom_nc_files"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. uv3z (3시간 간격) 데이터 다운로드
    uv3z_base_url = f"https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0/uv3z/{data_year}"
    uv3z_filename = "uv3z.nc"
    uv3z_filepath = os.path.join(output_dir, uv3z_filename)
    
    uv3z_params = {
        "var": ["water_u", "water_v"],
        "north": lat_max, "south": lat_min, "west": lon_min, "east": lon_max,
        "time_start": time_start_str, "time_end": time_end_str,
        "timeStride": 1, "vertStride": 0, "accept": "netcdf4"
    }

    # URL 생성 (요청된 변경사항)
    uv3z_request_url = f"{uv3z_base_url}?{urlencode(uv3z_params, doseq=True)}"

    try:
        print(f"uv3z 데이터 다운로드 중...: {uv3z_request_url}")
        response_uv3z = requests.get(uv3z_request_url, stream=True)
        response_uv3z.raise_for_status()
        with open(uv3z_filepath, "wb") as f:
            for chunk in response_uv3z.iter_content(chunk_size=8192):
                f.write(chunk)
        # with 문으로 xarray 데이터셋을 열어 파일 핸들을 자동으로 관리
        with xr.open_dataset(uv3z_filepath, decode_times=True) as ds_uv3z:
            
            # 2. sur (1시간 간격) 데이터 다운로드
            sur_base_url = f"https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/sur/{data_year}"
            sur_filename = "sur.nc"
            sur_filepath = os.path.join(output_dir, sur_filename)
            
            sur_params = {
                "var": ["u_barotropic_velocity", "v_barotropic_velocity"],
                "north": lat_max, "south": lat_min, "west": lon_min, "east": lon_max,
                "time_start": time_start_str, "time_end": time_end_str,
                "timeStride": 1, "accept": "netcdf4"
            }
            
            # URL 생성
            sur_request_url = f"{sur_base_url}?{urlencode(sur_params, doseq=True)}"

            try:
                print(f"sur 데이터 다운로드 중...: {sur_request_url}")
                response_sur = requests.get(sur_request_url, stream=True)
                response_sur.raise_for_status()
                with open(sur_filepath, "wb") as f:
                    for chunk in response_sur.iter_content(chunk_size=8192):
                        f.write(chunk)
                with xr.open_dataset(sur_filepath, decode_times=True) as ds_sur:

                    print("두 HYCOM 데이터셋을 결합하여 1시간 간격 데이터 생성 중...")

                    try:
                        # 3. uv3z 데이터를 sur 데이터의 시간 레벨에 맞춰 보간
                        # 깊이 0.0m의 표층 유속 데이터를 선택합니다.
                        ds_uv3z_surface = ds_uv3z.sel(depth=0.0, method='nearest').reset_coords('depth', drop=True)
                        
                        # 4. 동일 시간대에서 uv3z와 sur 데이터의 차이 계산
                        # uv3z와 sur의 시간이 일치하는 지점만 선택합니다.
                        common_times = ds_sur.time.sel(time=ds_uv3z_surface.time.values, method='nearest')
                        
                        # 표층 유속과 수송 유속의 차이를 계산 (u와 v 각각)
                        diff_u = ds_uv3z_surface['water_u'].sel(time=common_times) - ds_sur['u_barotropic_velocity'].sel(time=common_times)
                        diff_v = ds_uv3z_surface['water_v'].sel(time=common_times) - ds_sur['v_barotropic_velocity'].sel(time=common_times)

                        # 5. 차이 데이터(diff)를 1시간 간격으로 보간
                        # sur 데이터의 1시간 간격에 맞춰 차이 데이터를 선형 보간합니다.
                        diff_u_interp = diff_u.interp(time=ds_sur.time, method='linear')
                        diff_v_interp = diff_v.interp(time=ds_sur.time, method='linear')
                        
                        # 6. 보간된 차이를 sur 데이터에 더하여 최종 1시간 간격 표층 유속 추정
                        estimated_u = ds_sur['u_barotropic_velocity'] + diff_u_interp
                        estimated_v = ds_sur['v_barotropic_velocity'] + diff_v_interp

                        # 7. 최종 데이터셋 생성
                        ds_final = xr.Dataset(
                            {
                                "x_sea_water_velocity": (('time', 'lat', 'lon'), estimated_u.values),
                                "y_sea_water_velocity": (('time', 'lat', 'lon'), estimated_v.values)
                            },
                            coords={
                                "time": ds_sur.time.values,
                                "lat": ds_sur.lat.values,
                                "lon": ds_sur.lon.values
                            }
                        )
                        
                        return ds_final

                    except Exception as e:
                        print(f"데이터셋 결합 및 보간 실패: {e}")
                        return xr.Dataset()

            except requests.exceptions.RequestException as e:
                print(f"sur 데이터 다운로드 실패: {e}")
                return xr.Dataset()
        
    except requests.exceptions.RequestException as e:
        print(f"uv3z 데이터 다운로드 실패: {e}")
        return xr.Dataset()
    finally:
        # 임시 파일 정리
        if os.path.exists(uv3z_filepath):
            os.remove(uv3z_filepath)
        if os.path.exists(sur_filepath):
            os.remove(sur_filepath)


# ─────────────────────────────────────────────────────
# 4) ERA5 풍속 다운로드
# ─────────────────────────────────────────────────────

def fetch_era5(first_time, last_time, lat_min, lat_max, lon_min, lon_max):
    # 항상 다운로드하도록 변경
    year  = first_time.strftime('%Y')
    month = first_time.strftime('%m')
    num_days = (last_time.date() - first_time.date()).days + 1
    days = [(first_time + timedelta(days=i)).strftime('%d') for i in range(num_days)]
    times = [f"{h:02d}:00" for h in range(24)]
    area = [lat_max, lon_min, lat_min, lon_max]

    era5_request = {
        'product_type': ['reanalysis'],
        'variable': ['10m_u_component_of_wind','10m_v_component_of_wind'],
        'year':  [year], 'month': [month], 'day': days,
        'time': times, 'area': area, 'format': 'netcdf'
    }
    print('🌬️ ERA5 wind 요청:', era5_request)

    client = cdsapi.Client()
    file_name = 'era5_data.nc'
    # ERA5 요청
    result = client.retrieve('reanalysis-era5-single-levels', era5_request)
    result.download(file_name)  # 같은 파일에 계속 덮어쓰기

    # xarray로 읽고, 임시 파일 삭제
    ds = xr.open_dataset(file_name)

    rename_dict = {}
    if 'u10' in ds:
        rename_dict['u10'] = 'x_wind'
    if 'v10' in ds:
        rename_dict['v10'] = 'y_wind'
    ds = ds.rename(rename_dict)

    if 'x_wind' in ds:
        ds['x_wind'].attrs['standard_name'] = 'x_wind'
    if 'y_wind' in ds:
        ds['y_wind'].attrs['standard_name'] = 'y_wind'

    if 'expver' in ds.dims:
        ds = ds.isel(expver=0)
    if 'number' in ds.dims:
        ds = ds.isel(number=0)

    return ds

def run_lost_simulation(
    report2_id,
    service_key_khoa,
    time_step,
    retrieve_date   
):
    print("df 로드 중...")  
    df = load_db_to_df(report2_id)
    loc = locate_sequence(df, target=TARGET_SEQ)
    if loc is None:
        print("TARGET_SEQ 시퀀스가 포함되지 않았습니다.")
        loc = locate_sequence(df, target=ISSUE_SEQ)
    else:
        print("TARGET_SEQ 시퀀스가 포함되어 있습니다. 유실 신고 없음.")
    df = df.loc[loc[0]:loc[1]].reset_index(drop=True)
    df['prev_press'] = df['press'].shift(1)
    df0 = df[df['press'] == 0]
    if df0.empty or df[(df['press']==0)&(df['prev_press']!=0)].empty:
        raise RuntimeError("투망 구간 또는 시작 시점 데이터가 없습니다.")

    padding = 0.5
    lon_min, lon_max = df0['lon'].min() - padding, df0['lon'].max() + padding
    lat_min, lat_max = df0['lat'].min() - padding, df0['lat'].max() + padding
    
    grid_step = 0.25
    lon_grid = np.arange(
        np.floor(lon_min / grid_step) * grid_step,
        np.ceil(lon_max / grid_step) * grid_step + grid_step,
        grid_step
    )
    lat_grid = np.arange(
        np.floor(lat_min / grid_step) * grid_step,
        np.ceil(lat_max / grid_step) * grid_step + grid_step,
        grid_step
    )
    df_tumang = df[df['press'] == 0].copy().reset_index(drop=True)
    issue_time = df[df['press'] == 2][['time_stamp', 'lat', 'lon']]
    sim_start = df_tumang['time_stamp'].min().tz_localize(None)
    if retrieve_date == '':
        sim_end = sim_start + timedelta(hours=24)
    else:
        sim_end   = retrieve_date

    time_list = pd.date_range(sim_start, sim_end, freq='h')
    print("KHOA 수치조류도 데이터 가져오는 중...")
    fetch_tidal_uv = fetch_all_khoa(time_list, lon_min, lon_max, lat_min, lat_max, lon_grid, lat_grid, service_key_khoa)
    print("HYCOM 해류 데이터 가져오는 중...")
    fetch_uv = fetch_hycom_uv(time_list, lon_min, lon_max, lat_min, lat_max)
    print("ERA5 바람 데이터 가져오는 중...")
    fetch_wind = fetch_era5(sim_start, sim_end, lat_min, lat_max, lon_min, lon_max)

    class ConnectedNetDrift(OceanDrift):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ideal_distance_m = 270  # 자망 이상 거리 (m)
            self.k = 0.05  # 조정 강도 계수 (0~1 사이, 높일수록 자망 형태 강함)
            self.step = 2  # 몇 개 간격으로 연결할지 (3개 간격 연결)
            self.adjustment_loops = 2  # update 내 반복 조정 횟수

        def update(self):
            super().update()
            lon = self.elements.lon.copy()
            lat = self.elements.lat.copy()
            n = len(lon)

            for _ in range(self.adjustment_loops):
                for i in range(self.step, n):
                    prev_coord = (lat[i - self.step], lon[i - self.step])
                    curr_coord = (lat[i], lon[i])
                    dist = geodesic(prev_coord, curr_coord).meters
                    delta = dist - self.ideal_distance_m

                    if abs(delta) > 0.1:
                        dlat = lat[i] - lat[i - self.step]
                        dlon = lon[i] - lon[i - self.step]
                        scale = delta / dist * self.k

                        lat[i]              -= dlat * scale
                        lon[i]              -= dlon * scale
                        lat[i - self.step]  += dlat * scale
                        lon[i - self.step]  += dlon * scale

            self.elements.lon[:] = lon
            self.elements.lat[:] = lat

    o = ConnectedNetDrift(OceanDrift)
    # o = GradualKillDrift(kill_order=kill_order, kill_steps=kill_steps, yangmang_order=yangmang_order, loglevel=20)
    o.add_reader([reader_netCDF_CF_generic.Reader(fetch_tidal_uv),
                    reader_netCDF_CF_generic.Reader(fetch_uv),
                    reader_netCDF_CF_generic.Reader(fetch_wind)])
    o.set_config('seed:wind_drift_factor', 0.02)
    o.set_config('drift:stokes_drift', True)
    o.set_config('general:seafloor_action', 'none')
    o.set_config('drift:vertical_advection', False)
    o.set_config('drift:vertical_mixing', True)
    o.set_config('general:coastline_action', 'previous')
    for i, row in df_tumang.iterrows():
        o.seed_elements(
            lon=row['lon'],
            lat=row['lat'],
            time=pd.Timestamp(row['time_stamp']).tz_localize(None),
            z=0.0,
         ID=np.array([i], dtype=np.int32),
         origin_marker=np.array([i], dtype=np.int32)
        )

    o.run(time_step=time_step, duration=sim_end - sim_start)

    # 1. DataFrame으로 변환
    df_sim = o.result[['lat', 'lon', 'origin_marker']].to_dataframe().reset_index()
    df_sim = df_sim.rename(columns={'trajectory': 'seed_id', 'time': 'timestamp'})

    # 2. 각 origin_marker의 마지막 row만 선택
    last_df = df_sim.sort_values(['origin_marker', 'timestamp']).groupby('origin_marker').tail(1).reset_index(drop=True)

    # 3. 중앙 인덱스 추출
    center_idx = len(last_df) // 2
    center_row = last_df.iloc[center_idx]

    # 4. 결과 출력
    print(f"🧭 중심 origin_marker {center_row['origin_marker']} → lat: {center_row['lat']:.5f}, lon: {center_row['lon']:.5f}")
   
    plt.figure(figsize=(10, 7))
    for seed_id, group in df_sim.groupby('seed_id'):
        plt.plot(group['lon'], group['lat'], color='gray', alpha=0.4)
    plt.scatter(last_df['lon'], last_df['lat'], c='orange', s=10, label='비활성화 직전 위치')
    plt.plot(last_df['lon'], last_df['lat'], color='orange', linewidth=10, label='비활성화 위치 경로')
    plt.scatter(df_tumang['lon'], df_tumang['lat'], s=30, color='blue', marker='^', label='투망(0)')
    plt.scatter(issue_time['lon'], issue_time['lat'], s=30, color='red', marker='v', label='신고위치(2)')
    plt.title("시뮬레이션 궤적 및 위치 비교")
    plt.legend()
    plt.tight_layout()
   
    # --------- 메모리에 저장 (base64 변환) ---------
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # ---------------------------------------------

    return center_row['lat'], center_row['lon'], img_base64