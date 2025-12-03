import math
import os
from urllib.parse import urlencode
#–– PROJ 환경 변수 설정 (pyproj ≥3용 PROJ_DATA 포함)
os.environ['PROJ_LIB']  = r'C:\Users\HUFS\anaconda3\envs\opendrift_env\Library\share\proj'
os.environ['PROJ_DATA'] = r'C:\Users\HUFS\anaconda3\envs\opendrift_env\Library\share\proj'

import json
import glob
import time
from datetime import datetime, timedelta


import csv
from dateutil import parser
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import requests
import geopandas as gpd
from geopy.distance import geodesic
from shapely.geometry import Point, LineString

import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False



geojson_dir = r"D:\어선행적데이터\Training\02.라벨링데이터\TL_01.자망.zip (2)"
# geojson_dir = r'D:\어선행적데이터\Validation\02.라벨링데이터\VL_01.자망.zip'
geojson_files = glob.glob(os.path.join(geojson_dir, "*.geojson"))



error_log_path = os.path.join(geojson_dir, "error_log.csv")

# 에러 로그 초기화
if not os.path.exists(error_log_path):
    with open(error_log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["파일명", "오류종류", "오류메시지"])

def get_retry_session(max_retries=5, backoff_factor=1):
    """일시적인 서버/네트워크 오류에 대해 재시도 정책을 가진 requests.Session 객체를 반환합니다."""
    # 재시도할 HTTP 상태 코드와 GET 메서드 설정
    retry_strategy = Retry(
        total=max_retries, 
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        backoff_factor=backoff_factor # 1초, 2초, 4초, ... 간격으로 재시도 대기 시간 증가
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)
    return http

# 2. 전역 세션 및 타임아웃 값 설정
HTTP_SESSION = get_retry_session()
# 연결(Connect) 10초, 데이터 수신(Read) 60초 설정 (총 70초)
TIMEOUT_TUPLE = (10, 60) 

# 전체 자동 처리 루프
for input_file in geojson_files:
    # try:
        input_filename = os.path.basename(input_file)
        print(input_filename)
        if input_filename not in ['T01_DDJ134DGJ_2021-12-08 113000-114.geojson', 'T01_DDJ142HDJ_2021-06-22 150100-019.geojson', 'T01_DDJ134DGJ_2021-09-01 221500-019.geojson']:
            continue

        visibility = None            # 가시거리(m)
        distance_km = None           # 중간 투망 ↔ 양망 거리(km)
        prediction_result = "판단 불가"  # 예측 성공 여부

        print(f"\n===== 처리 시작: {input_filename} =====")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        TARGET_SEQ = [3,0,3,1,3]   # 시퀀스 탐색용

        ###############################################################################
        # PART 1: GeoJSON → DataFrame (투망 궤적 추출)
        ###############################################################################
        rows = []
        for feat in data.get("features", []):
            p = feat["properties"]
            beh = p.get("fishery_behavior")
            rows.append({
                "time_stamp": p["time_stamp"],
                "lon":         p["longitude"],
                "lat":         p["latitude"],
                "fishery_behavior": beh
            })

        df = pd.DataFrame(rows)
        if df.empty:
            print(f"⚠️ {input_filename}: DataFrame 비어있음 → 건너뜀")
            continue

        print(f"📊 {input_filename}: DataFrame 크기 = {df.shape}")

        df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce')
        df = df.sort_values('time_stamp').reset_index(drop=True)

        # 📌 시퀀스 탐색 함수
        def has_target_sequence(behaviors, target=TARGET_SEQ):
            if len(behaviors) == 0:
                print("⚠️ behaviors 비어있음")
                return False
            if len(behaviors) < len(target):
                print(f"⚠️ behaviors 길이({len(behaviors)}) < target 길이({len(target)})")
                return False

            print(f"▶ raw behaviors 길이={len(behaviors)} → {behaviors[:20]}...")  # 앞부분만 확인
            compressed = [behaviors[0]]
            for b in behaviors[1:]:
                if b != compressed[-1]:
                    compressed.append(b)

            print(f"▶ compressed behaviors 길이={len(compressed)} → {compressed}")

            # TARGET_SEQ 포함 여부 확인
            n, m = len(compressed), len(target)
            for i in range(n - m + 1):
                if compressed[i:i+m] == target:
                    print(f"✅ TARGET_SEQ {target} 발견 위치: {i}")
                    return True
            return False

        # 📌 시퀀스 없는 파일은 스킵
        if not has_target_sequence(df['fishery_behavior'].tolist(), TARGET_SEQ):
            print(f"❌ {input_filename}: 시퀀스 {TARGET_SEQ} 없음 → 건너뜀")
            continue

        # 투망 시작 지점만 필터링 (1->3 또는 0->3 변화 시점)
        df['prev_behavior'] = df['fishery_behavior'].shift(1)
        drop_points = df[
            (df['fishery_behavior'] == 3) &
            (df['prev_behavior'] != 3)
        ].copy()

        print(f"🎯 drop_points 개수: {len(drop_points)}")

        # 📌 2. 시간 리스트 (1시간 간격)
        start_time = df['time_stamp'].min().replace(minute=0, second=0)
        end_time = df['time_stamp'].max()
        print(f"🕒 start_time={start_time}, end_time={end_time}")

        time_list = []
        current_time = start_time
        while current_time <= end_time:
            time_list.append(current_time)
            current_time += timedelta(hours=1)

        print(f"🕒 time_list 길이: {len(time_list)} → {time_list[:5]}...")

        simulation_duration = end_time - start_time
        print(f"⏳ simulation_duration: {simulation_duration}")

        # 연, 월, 일 문자열로 추출
        year  = f"{start_time.strftime('%Y')}"
        month = f"{start_time.strftime('%m')}"
        day   = f"{start_time.strftime('%d')}"
        print(f"📅 날짜: {year}-{month}-{day}")

        lat_min = df['lat'].min() - 0.1
        lat_max = df['lat'].max() + 0.1
        lon_min = df['lon'].min() - 0.1
        lon_max = df['lon'].max() + 0.1

        print(f"🌍 lat 범위=({lat_min}, {lat_max}), lon 범위=({lon_min}, {lon_max})")

        lat_grid = np.arange(round(lat_min, 2), round(lat_max, 2) + 0.01, 0.01)
        lon_grid = np.arange(round(lon_min, 2), round(lon_max, 2) + 0.01, 0.01)

        print(f"🌍 lat_grid={lat_grid.shape}, lon_grid={lon_grid.shape}")
        print(f"===== 처리 완료: {input_filename} =====\n")



        # ====== NetCDF 파일 경로 미리 설정 ======
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        nc_folder = r"C:\Users\HUFS\Desktop\opendrift_middle\KHOA_nc_data"
        os.makedirs(nc_folder, exist_ok=True)


            # ====== API 호출 및 보간 수행 ======
        service_key = 'ANM8LV6zTsRNiGg6FCUMpw=='
        base_url = "http://www.khoa.go.kr/api/oceangrid/tidalCurrentAreaGeoJson/search.do"
        all_data = []


        output_path = os.path.join(nc_folder, f"{input_basename}_uv.nc")

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

        uv_ds = xr.Dataset(
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
        uv_ds['x_sea_water_velocity'].attrs.update(standard_name="x_sea_water_velocity", units="m s-1")
        uv_ds['y_sea_water_velocity'].attrs.update(standard_name="y_sea_water_velocity", units="m s-1")


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

        temp_df = pd.DataFrame(temp_records)

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

        sal_df = pd.DataFrame(sal_records)

        # 1. 해류 데이터 (기존 방식 유지)
        ds = uv_ds

        # 2. 기준 시간 및 공간 정보
        ds_time = pd.to_datetime(ds['time'].values).tz_localize(None).to_numpy(dtype='datetime64[ns]')
        lat_vals = ds['lat'].values
        lon_vals = ds['lon'].values

        # 3. 중심 좌표 및 날짜 리스트 생성
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        day_time_list = pd.to_datetime(sorted(set(t.normalize() for t in time_list))).tz_localize(None)

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
        

        nc_folder = r"C:\Users\HUFS\Desktop\opendrift_middle\KHOA_nc_data"
        os.makedirs(nc_folder, exist_ok=True)

        output_path = os.path.join(nc_folder, f"{input_basename}_uv.nc")
        ds.to_netcdf(output_path)
        print(f"✅ NetCDF 저장 완료: {output_path}")


        # ==================================
        # 🌊 2. HYCOM API 데이터 처리
        # ==================================

        def get_time_steps(start_dt, end_dt, max_hours=24):
            """요청 시간을 최대 max_hours 단위로 분할하여 (시작, 끝) 튜플 리스트를 반환합니다."""
            time_steps = []
            current_start = start_dt
            
            # end_dt는 항상 time_list.max()보다 크거나 같으므로, 루프는 end_dt에 도달할 때까지 진행됩니다.
            while current_start < end_dt:
                # 다음 스텝의 끝 시간 = 현재 시작 시간 + max_hours (단, 전체 time_end를 넘지 않도록 조정)
                current_end = min(current_start + timedelta(hours=max_hours), end_dt)
                time_steps.append((current_start, current_end))
                current_start = current_end # 다음 시작은 현재 끝 시간부터

            return time_steps

        # HYCOM 데이터 처리를 위한 폴더 설정
        hycom_output_dir = r"C:\Users\HUFS\Desktop\opendrift_middle\hycom_data"
        os.makedirs(hycom_output_dir, exist_ok=True)
        hycom_final_filename = f"{input_basename}_hycom.nc"
        hycom_final_filepath = os.path.join(hycom_output_dir, hycom_final_filename)

        # 임시 파일 경로 정의 (uv3z는 단일, sur는 리스트로 관리)
        uv3z_filepath = os.path.join(hycom_output_dir, "temp_uv3z.nc")
        temp_sur_files = [] # 분할된 sur 파일 경로를 저장할 리스트

        # ====== 파일 존재 시 다운로드 및 처리 생략 ======
        if os.path.exists(hycom_final_filepath):
            print(f"🔄 이미 HYCOM NetCDF 존재, 다운로드 생략: {hycom_final_filepath}")
        else:
            try:
                data_year = time_list[0].year
            except (IndexError, AttributeError):
                print("time_list가 비어 있거나 올바른 형식이 아닙니다. 현재 연도를 사용합니다.")
                data_year = datetime.now().year
            
            supported_start_year = 2018
            supported_end_year = 2024

            if not (supported_start_year <= data_year <= supported_end_year):
                print(f"오류: {data_year}년 데이터는 HYCOM 서버에서 지원하지 않습니다.")
                print(f"지원되는 연도 범위: {supported_start_year}년 ~ {supported_end_year}년")
            else:
                # 1. 시간 범위 설정
                ds_time = pd.to_datetime(time_list).to_numpy(dtype='datetime64[ns]')
                time_start_dt = pd.to_datetime(ds_time.min()).replace(minute=0, second=0, microsecond=0)
                time_end_dt = pd.to_datetime(ds_time.max()).replace(minute=0, second=0, microsecond=0)
                if time_end_dt < pd.to_datetime(ds_time.max()):
                    time_end_dt += timedelta(hours=1)
                    
                time_start_str = time_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                time_end_str = time_end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # 2. uv3z (3차원) 데이터 다운로드 (전체 기간 한 번에 요청)
                try:
                    print("--- 1. uv3z (3차원) 데이터 다운로드 (전체 기간) ---")
                    uv3z_base_url = f"https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0/uv3z/{data_year}"
                    uv3z_params = {
                        "var": ["water_u", "water_v"],
                        "north": round(lat_max, 2), "west": math.floor(lon_min*100) / 100, "east": round(lon_max, 2), "south": math.floor(lat_min*100) / 100, 
                        "time_start": time_start_str, "time_end": time_end_str,
                        "timeStride": 1, "vertStride": 0, "accept": "netcdf4"
                    }
                    uv3z_request_url = f"{uv3z_base_url}?{urlencode(uv3z_params, doseq=True)}"
                    
                    print(uv3z_request_url)
                    response_uv3z = requests.get(uv3z_request_url, stream=True)
                    response_uv3z.raise_for_status()
                    with open(hycom_final_filepath, "wb") as f:
                        for chunk in response_uv3z.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"uv3z 다운로드 완료: {hycom_final_filepath}")


                    # # 3. sur (표층) 데이터 다운로드 (24시간 미만으로 분할 요청)
                    # print("\n--- 2. sur (표층) 데이터 분할 다운로드 (24시간 미만) ---")
                    # time_steps = get_time_steps(time_start_dt, time_end_dt, max_hours=24) # 최대 24시간 단위로 분할

                    # for i, (start_dt, end_dt) in enumerate(time_steps):
                    #     current_start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    #     current_end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        
                    #     sur_filepath = os.path.join(hycom_output_dir, f"temp_sur_{i:03d}.nc")
                    #     temp_sur_files.append(sur_filepath) # 경로를 리스트에 추가

                    #     print(f"  > sur 구간 {i+1}/{len(time_steps)} 다운로드: {current_start_str} ~ {current_end_str}")
                        
                    #     sur_base_url = f"https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0/sur/{data_year}"
                    #     sur_params = {
                    #         "var": ["u_barotropic_velocity", "v_barotropic_velocity"],
                    #         "north": round(lat_max, 2), "west": math.floor(lon_min*100) / 100, "east": round(lon_max, 2), "south": math.floor(lat_min*100) / 100, 
                    #         "disableProjSubset": "on", "horizStride": 1, 
                    #         "time_start": current_start_str, "time_end": current_end_str, # 분할된 시간 사용
                    #         "timeStride": 1, "accept": "netcdf4"
                    #     }
                    #     sur_request_url = f"{sur_base_url}?{urlencode(sur_params, doseq=True)}"
                        
                    #     response_sur = requests.get(sur_request_url, stream=True)
                    #     response_sur.raise_for_status()
                    #     with open(sur_filepath, "wb") as f:
                    #         for chunk in response_sur.iter_content(chunk_size=8192):
                    #             f.write(chunk)
                    
                    # print(f"sur 분할 다운로드 완료. 총 {len(temp_sur_files)}개 파일.")

                    # # 4. 데이터셋 열기 및 결합
                    # print("\n--- 3. HYCOM 데이터셋 결합 및 처리 ---")
                    # # uv3z는 단일 파일로 열고, sur는 모든 분할 파일을 열어 병합 (open_mfdataset)
                    
                    # # **주의: 다운로드된 파일이 없으면 open_mfdataset이 실패하므로 확인**
                    # if not os.path.exists(uv3z_filepath) or not temp_sur_files:
                    #     raise Exception("다운로드된 파일이 불충분하여 결합을 시작할 수 없습니다.")
                        
                    # with xr.open_dataset(uv3z_filepath, decode_times=True) as ds_uv3z, \
                    #     xr.open_mfdataset(temp_sur_files, combine='by_coords', decode_times=True) as ds_sur:
                        
                    #     print("두 HYCOM 데이터셋을 결합하여 1시간 간격 데이터 생성 중...")
                        
                    #     # 4-1. uv3z (3차원) 데이터에서 표층(depth=0.0) 유속 추출
                    #     ds_uv3z_surface = ds_uv3z.sel(depth=0.0, method='nearest').reset_coords('depth', drop=True)
                        
                    #     # 4-2. 경압 성분(Baroclinic component) 계산 및 보간
                    #     # ds_sur의 1시간 간격 시간축을 최종 결과의 기준으로 사용합니다.
                        
                    #     # ds_uv3z_surface (3시간 간격)와 ds_sur (1시간 간격)의 공통 시간대(3시간 간격)를 찾습니다.
                    #     # ds_sur.time에서 ds_uv3z_surface.time과 가장 가까운 시간을 선택합니다.
                    #     # (이로써 ds_uv3z_surface와 ds_sur는 3시간 간격의 동일한 시간을 공유하게 됩니다.)
                    #     common_times = ds_sur.time.sel(time=ds_uv3z_surface.time.values, method='nearest')
                        
                    #     # 경압 성분 = 표층 유속 (uv3z) - 순압 유속 (sur)
                    #     diff_u = ds_uv3z_surface['water_u'].sel(time=common_times) - ds_sur['u_barotropic_velocity'].sel(time=common_times)
                    #     diff_v = ds_uv3z_surface['water_v'].sel(time=common_times) - ds_sur['v_barotropic_velocity'].sel(time=common_times)
                        
                    #     # 경압 성분(3시간 간격)을 sur의 1시간 간격 시간축에 선형 보간 (1시간 간격의 경압 성분 획득)
                    #     diff_u_interp = diff_u.interp(time=ds_sur.time, method='linear')
                    #     diff_v_interp = diff_v.interp(time=ds_sur.time, method='linear')
                        
                    #     # 4-3. 최종 유속 계산 (1시간 간격)
                    #     # 최종 유속 = 순압 유속 (sur, 1시간 간격) + 보간된 경압 성분 (1시간 간격)
                    #     estimated_u = ds_sur['u_barotropic_velocity'] + diff_u_interp
                    #     estimated_v = ds_sur['v_barotropic_velocity'] + diff_v_interp
                        
                    #     # 4-4. 최종 데이터셋 구성
                    #     ds_final = xr.Dataset(
                    #         {
                    #             "x_sea_water_velocity": (('time', 'lat', 'lon'), estimated_u.values),
                    #             "y_sea_water_velocity": (('time', 'lat', 'lon'), estimated_v.values)
                    #         },
                    #         coords={
                    #             "time": ds_sur.time.values, # ds_sur의 1시간 간격 시간 사용
                    #             "lat": ds_sur.lat.values,
                    #             "lon": ds_sur.lon.values
                    #         }
                    #     )

                    #     # 5. 최종 데이터셋을 지정된 형식으로 저장
                    #     ds_final.to_netcdf(hycom_final_filepath)
                    #     print(f"✅ 데이터셋이 성공적으로 저장되었습니다: {hycom_final_filepath}")

                
                except requests.exceptions.RequestException as e:
                    print(f"❌ 데이터 다운로드 실패: {e}")
                except Exception as e:
                    print(f"❌ 데이터 처리, 결합 또는 저장 실패: {e}")
                finally:
                    # 임시 파일 정리
                    print("🧹 임시 파일 정리 중...")
                    if os.path.exists(uv3z_filepath):
                        os.remove(uv3z_filepath)
                    for fpath in temp_sur_files:
                        if os.path.exists(fpath):
                            os.remove(fpath)
                    print("🧹 임시 파일 정리 완료.")

        #=======================================================
        # PART 3: ERA5 CDS API 데이터 다운로드ㅡ
        #=======================================================

        # 날짜 범위 자동 생성 (예: ['09','10','11'] 등)
        num_days = (end_time.date() - start_time.date()).days + 1
        days = [(start_time + timedelta(days=i)).strftime("%d") for i in range(num_days)]

        # 시간 리스트 (00:00 ~ 23:00)
        times = [f"{h:02d}:00" for h in range(24)]


        area   = [lat_max, lon_min, lat_min, lon_max]

        era5_request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year":  [year],
            "month": [month],
            "day":   days,
            "time":  times,
            "area":  area,
            "format": "netcdf"
        }


        # ====== 저장 경로 및 파일명 =====
        print("ERA5 요청:", era5_request)
        client = cdsapi.Client()
        wind_folder = r"C:\Users\HUFS\Desktop\opendrift_middle\wind_data"
        os.makedirs(wind_folder, exist_ok=True)
        wind_path = os.path.join(wind_folder, f"{input_basename}_wind.nc")

        # ====== 파일 존재 시 다운로드 생략 ======
        if os.path.exists(wind_path):
            print(f"🔄 이미 wind 파일 존재, 다운로드 생략: {wind_path}")
        else:
            print("🌬️ ERA5 wind 요청:", era5_request)
            client = cdsapi.Client()
            client.retrieve(
                'reanalysis-era5-single-levels',
                era5_request,
                wind_path
            )
            print(f"✅ ERA5 다운로드 완료: {wind_path}")


        ###############################################################################
        # 가시거리 데이터 가져오기
        ###############################################################################

        # 관측소 목록 (위도, 경도)
        observation_stations = {
            "SF_0001": {"name": "부산항", "latitude": 35.091, "longitude": 129.099},
            "SF_0002": {"name": "부산항(신항)", "latitude": 35.023, "longitude": 128.808},
            "SF_0009": {"name": "해운대", "latitude": 35.15909, "longitude": 129.16026},
            "SF_0010": {"name": "울산항", "latitude": 35.501, "longitude": 129.387},
            "SF_0008": {"name": "여수항", "latitude": 34.754, "longitude": 127.752},
        }

        # API 키
        service_key = 'ANM8LV6zTsRNiGg6FCUMpw=='  # 발급받은 인증키

        # JSON 파일 로드
        json_file = input_file

        # fishery_behavior가 1인 데이터 추출 (첫 번째만)
        first_fishery_behavior = None
        for feature in data['features']:
            if feature['properties']['fishery_behavior'] == 1:
                first_fishery_behavior = feature
                break  # 첫 번째 데이터만 처리

        # 가장 가까운 관측소 찾기
        def find_closest_station(lat, lon):
            closest_station = None
            min_distance = float('inf')

            # 각 관측소와의 거리 계산
            for obs_code, station in observation_stations.items():
                station_location = (station["latitude"], station["longitude"])
                current_location = (lat, lon)
                distance = geodesic(station_location, current_location).kilometers
                
                if distance < min_distance:
                    min_distance = distance
                    closest_station = obs_code
            
            return closest_station
        

        # 가장 가까운 관측소에서 가시거리 정보 가져오기
        def get_visibility_from_station(obs_code, timestamp):
            # 날짜만 추출해서 YYYYMMDD 형식으로 변환
            date_only = timestamp.split(" ")[0].replace("-", "")  # 날짜만 추출 (YYYYMMDD)

            # API 요청 URL 생성
            url = f"http://www.khoa.go.kr/api/oceangrid/seafogReal/search.do" \
                f"?DataType=seafogReal" \
                f"&ServiceKey={service_key}" \
                f"&ObsCode={obs_code}" \
                f"&Date={date_only}" \
                f"&ResultType=json"
            
            # API 요청
            response = requests.get(url)

            # 응답 데이터 확인
            if response.status_code == 200:
                data = response.json()
                
                # 응답 데이터 출력
                if 'result' in data and 'data' in data['result']:
                    closest_time_diff = float('inf')  # 가장 가까운 시간 차이
                    closest_visibility = None

                    for observation in data['result']['data']:
                        obs_time = observation['obs_time']

                        # 시간 차이 계산 (두 시간의 차이를 분 단위로 계산)
                        try:
                            timestamp_dt = parser.parse(timestamp.strip())
                            obs_time_dt = parser.parse(obs_time.strip())
                        except Exception as e:
                            print(f"시간 파싱 오류: {e}")
                            continue

                        time_diff = abs((timestamp_dt - obs_time_dt).total_seconds())  # 시간 차이 (초 단위)

                        # 가장 가까운 시간 찾기
                        if time_diff < closest_time_diff:
                            closest_time_diff = time_diff
                            if 'vis' in observation:
                                closest_visibility = observation['vis']
                    
                    if closest_visibility:
                        return closest_visibility  # 가장 가까운 가시거리 반환
            return None

        # 초기 변수
        visibility = None
        latitude = None
        longitude = None
        timestamp = None
        closest_station = None

        # fishery_behavior = 1인 데이터가 있다면 정보 저장
        if first_fishery_behavior:
            timestamp = first_fishery_behavior['properties']['time_stamp']
            latitude = first_fishery_behavior['properties']['latitude']
            longitude = first_fishery_behavior['properties']['longitude']
            closest_station = find_closest_station(latitude, longitude)
        else:
            print("fishery_behavior가 1인 데이터가 없습니다.")

        # CSV 경로 지정
        output_csv_path = r"C:\Users\HUFS\Desktop\opendrift_middle\가시거리csv\visibility_log_train_other.csv"

        # 1. CSV에서 확인
        if os.path.exists(output_csv_path):
            with open(output_csv_path, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["filename"] == input_filename:
                        visibility = row["visibility_m"]
                        print(f"🔄 기존 CSV에서 가시거리 불러옴: {visibility}")
                        break

        # 2. 없으면 API 호출
        if visibility is None  and timestamp and closest_station:
            visibility = get_visibility_from_station(closest_station, timestamp)

            if visibility:
                print(f"시간: {timestamp} / 위치: ({latitude}, {longitude})")
                print(f"가장 가까운 관측소: {observation_stations[closest_station]['name']} ({closest_station})")
                print(f"가시거리: {visibility} 미터")
            else:
                print(f"가시거리 정보를 불러올 수 없습니다.")

        # 3. CSV에 저장
        if not os.path.exists(output_csv_path):
            with open(output_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "visibility_m"])

        # 4. 중복 저장 방지 후 추가
        already_exists = False
        with open(output_csv_path, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["filename"] == input_filename:
                    already_exists = True
                    break

        if not already_exists:
            with open(output_csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([input_filename, visibility if visibility else "N/A"])



        ###############################################################################
        # (4) ERA5 wind파일 경로
        ###############################################################################
        nc_folder = r"C:\Users\HUFS\Desktop\opendrift_middle\KHOA_nc_data"
        merged_file = os.path.join(nc_folder, f"{input_basename}_uv.nc")

        hycom_folder = r"C:\Users\HUFS\Desktop\opendrift_middle\hycom_data"
        hycom_file = os.path.join(hycom_folder, f"{input_basename}_hycom.nc")

        wind_file = os.path.join(wind_folder, f"{input_basename}_wind.nc")


        # bottom_depth = r"C:\Users\HUFS\Desktop\opendrift_middle\bottom_depth.nc"
        # print("Bottom depth file:", bottom_depth)





        ###############################################################################
        # PART 3: 해안선 읽기
        ###############################################################################
        # 해안선 읽기
        coastline_file = r"C:\Users\HUFS\Downloads\해양수산부 국립해양조사원_해안선_20241231\2025년 전국 해안선.shp"
        coast = gpd.read_file(coastline_file)
        if coast.crs is None or coast.crs.to_string() != 'EPSG:4326':
            coast = coast.to_crs(epsg=4326)
        coast_proj = coast.to_crs(epsg=3857)
        coastal_zone = coast_proj.buffer(15000).union_all()
        coastal_zone_wgs84 = gpd.GeoSeries(coastal_zone, crs=3857).to_crs(epsg=4326).union_all()
        print("해안선 정보 불러오기 완료")



        ###############################################################################
        # OpenDrift 모델 설정 - ConnectedNetDrift로 변경
        ###############################################################################
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


        o = ConnectedNetDrift(loglevel=20)
        reader_uv = reader_netCDF_CF_generic.Reader(hycom_file)
        reader_tidal = reader_netCDF_CF_generic.Reader(merged_file)
        reader_wet   = reader_netCDF_CF_generic.Reader(wind_file)
        # reader_bathy = reader_netCDF_CF_generic.Reader(bottom_depth)
        o.add_reader(reader_uv)
        print("해류 nc파일 읽기 완료")
        o.add_reader(reader_tidal)
        print("수치조류도 nc파일 읽기 완료")
        o.add_reader(reader_wet)
        print("날씨 nc파일 읽기 완료")

        o.set_config('seed:wind_drift_factor', 0.02)
        o.set_config('drift:stokes_drift', True)
        o.set_config('general:seafloor_action', 'none')
        o.set_config('drift:vertical_advection', False)
        o.set_config('drift:vertical_mixing', True)
        o.set_config('general:coastline_action', 'previous')
        print("opendrift 모델 정의 완료")

        ###############################################################################
        # 입자 시딩 (자망을 순차적으로 시딩)
        df_tumang = df[df['fishery_behavior'] == 0].copy()
        df_yangmang = df[df['fishery_behavior'] == 1].copy()
        df_tumang['time_stamp'] = pd.to_datetime(df_tumang['time_stamp'])
        df_tumang = df_tumang.sort_values('time_stamp').reset_index(drop=True)
        df_yangmang = df_yangmang.sort_values('time_stamp').reset_index(drop=True)

        # 양망 구간 중간 위치 추출
        num_yangmang = len(df_yangmang)
        if num_yangmang == 0:
            # 양망 데이터가 없으면 시뮬레이션 목적에 맞지 않으므로 건너뜀
            # 이 부분은 기존 코드에 있었으므로 유지
            continue 

        # ======================== ✨ 추가된 확인 코드 블록 시작 ✨ ========================

        # 투망 데이터(시딩할 입자)가 있는지 확인
        if len(df_tumang) == 0:
            print("⚠️ 경고: df_tumang DataFrame에 투망(0) 행동 데이터가 없습니다. 입자 시딩을 건너뜁니다.")
            continue 
            
        # ======================== ✨ 추가된 확인 코드 블록 끝 ✨ ========================

        print("입자 시딩 시작...")
        for i, row in df_tumang.iterrows():
            o.seed_elements(
                lon=row['lon'],
                lat=row['lat'],
                time=row['time_stamp'],
                z=0.0,
                origin_marker=np.array([i], dtype=np.int32)
            )
            print(f"[SEED] time={row['time_stamp']}, 위치=({row['lon']}, {row['lat']})")
        ###############################################################################
        # 시뮬레이션 실행
        start_time_sim = df_tumang['time_stamp'].min()
        end_time_sim   = df[df['fishery_behavior'] == 1]['time_stamp'].max()
        # end_time_sim   = df_yangmang['time_stamp'].min()
        simulation_duration = end_time_sim - start_time_sim

        o.run(
            time_step=600,
            time_step_output=1800,
            duration=simulation_duration
        )


        # ###############################################################################
        # 초기 및 최종 입자 최종 위치 추출
        num_particles = len(df_tumang)
        if num_particles == 0:
            print("⚠️ 경고: 투망(df_tumang) 데이터가 없어 입자 최종 위치 추출을 건너뛰고 None으로 설정합니다.")
            start_lon_pred = start_lat_pred = end_lon_pred = end_lat_pred = None
        else:
            start_index = 0
            end_index = num_particles - 1
            
            # OpenDrift 결과에서 위치 배열 추출 (시뮬레이션 후 o.get_property 사용)
            lon_traj, _ = o.get_property('lon')
            lat_traj, _ = o.get_property('lat')

            # 1. 시작 입자 최종 예측 위치
            start_lon_pred = lon_traj[-1, start_index].item()
            start_lat_pred = lat_traj[-1, start_index].item()

            # 2. 끝 입자 최종 예측 위치
            end_lon_pred = lon_traj[-1, end_index].item()
            end_lat_pred = lat_traj[-1, end_index].item()

            print(f"[시작 입자 최종 예측 위치] lon: {start_lon_pred:.5f}, lat: {start_lat_pred:.5f}")
            print(f"[끝 입자 최종 예측 위치] lon: {end_lon_pred:.5f}, lat: {end_lat_pred:.5f}")

        # ###############################################################################
        # 양망 시작/끝 위치 추출
        num_yangmang = len(df_yangmang)
        if num_yangmang < 2:
            print("⚠️ 경고: 양망(df_yangmang) 데이터가 2개 미만이므로 시작/끝 위치 추출을 건너뛰고 None으로 설정합니다.")
            yang_start_lon = yang_start_lat = yang_end_lon = yang_end_lat = None
        else:
            # 양망 시작 위치 (첫 번째 행)
            start_row_yangmang = df_yangmang.iloc[0]
            yang_start_lon = start_row_yangmang['lon']
            yang_start_lat = start_row_yangmang['lat']
            yang_start_time = start_row_yangmang['time_stamp']

            # 양망 끝 위치 (마지막 행)
            end_row_yangmang = df_yangmang.iloc[-1]
            yang_end_lon = end_row_yangmang['lon']
            yang_end_lat = end_row_yangmang['lat']
            yang_end_time = end_row_yangmang['time_stamp']
            
            print(f"[양망 시작 위치] time={yang_start_time}, lon={yang_start_lon:.5f}, lat={yang_start_lat:.5f}")
            print(f"[양망 끝 위치] time={yang_end_time}, lon={yang_end_lon:.5f}, lat={yang_end_lat:.5f}")

        # ###############################################################################
        # 투망 시작/끝 시딩 위치 추출
        if num_particles > 0:
            start_row_tumang = df_tumang.iloc[start_index]
            end_row_tumang = df_tumang.iloc[end_index]
            tumang_start_lon = start_row_tumang['lon']
            tumang_start_lat = start_row_tumang['lat']
            tumang_end_lon = end_row_tumang['lon']
            tumang_end_lat = end_row_tumang['lat']
            
            print(f"[투망 시작 시딩 위치] lon={tumang_start_lon:.5f}, lat: {tumang_start_lat:.5f}")
            print(f"[투망 끝 시딩 위치] lon={tumang_end_lon:.5f}, lat: {tumang_end_lat:.5f}")
        else:
            tumang_start_lon = tumang_start_lat = tumang_end_lon = tumang_end_lat = None

        # #####################################################################
        # 4가지 거리 계산
        distance_start_pred_km = None
        distance_end_pred_km = None
        distance_tumang_start_yang_start_km = None
        distance_tumang_end_yang_end_km = None

        # 예측 위치 및 양망 위치가 모두 유효할 때만 거리 계산
        if yang_start_lon is not None and start_lon_pred is not None:
            # 양망 시작점과 끝점
            point_yang_start = (yang_start_lat, yang_start_lon)
            point_yang_end = (yang_end_lat, yang_end_lon)

            # 1. 시작 예측 위치 ↔ 양망 시작 위치
            point_start_pred = (start_lat_pred, start_lon_pred)
            distance_start_pred_km = geodesic(point_start_pred, point_yang_start).kilometers
            print(f"[1. 시작 예측 ↔ 양망 시작 거리] 약 {distance_start_pred_km:.3f} km")

            # 2. 끝 예측 위치 ↔ 양망 끝 위치
            point_end_pred = (end_lat_pred, end_lon_pred)
            distance_end_pred_km = geodesic(point_end_pred, point_yang_end).kilometers
            print(f"[2. 끝 예측 ↔ 양망 끝 거리] 약 {distance_end_pred_km:.3f} km")
            
            # 3. 투망 시작 위치 ↔ 양망 시작 위치
            point_tumang_start = (tumang_start_lat, tumang_start_lon)
            distance_tumang_start_yang_start_km = geodesic(point_tumang_start, point_yang_start).kilometers
            print(f"[3. 투망 시작 ↔ 양망 시작 거리 (기준)] 약 {distance_tumang_start_yang_start_km:.3f} km")

            # 4. 투망 끝 위치 ↔ 양망 끝 위치
            point_tumang_end = (tumang_end_lat, tumang_end_lon)
            distance_tumang_end_yang_end_km = geodesic(point_tumang_end, point_yang_end).kilometers
            print(f"[4. 투망 끝 ↔ 양망 끝 거리 (기준)] 약 {distance_tumang_end_yang_end_km:.3f} km")

        # ###############################################################################
        # 가시거리 단위 변환 및 결과 판단
        try:
            visibility_km = float(visibility) / 1000 if visibility not in (None, "N/A", "불러오기 실패") else None
        except (ValueError, TypeError):
            visibility_km = None

        # 결과 판단 (두 예측 거리 중 하나라도 가시거리 내에 들어오면 성공으로 판단)
        prediction_result = "판단 불가"
        if visibility_km is not None and distance_start_pred_km is not None and distance_end_pred_km is not None:
            # 두 예측 거리를 리스트로 묶어 None이 아닌 유효한 값만 필터링
            valid_distances = [d for d in [distance_start_pred_km, distance_end_pred_km] if d is not None]
            
            if valid_distances:
                min_distance_km = min(valid_distances)
                prediction_result = "성공" if min_distance_km < visibility_km else "실패"

        # ###############################################################################
        # CSV 저장

        result_csv_path = r"C:\Users\HUFS\Desktop\opendrift_middle\예측csv\hycom_prediction_result_train_other.csv"

        # 헤더 작성
        if not os.path.exists(result_csv_path):
            with open(result_csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "filename", 
                    "시작예측↔양망시작_km", 
                    "끝예측↔양망끝_km",   
                    "가시거리_km", 
                    "예측결과", 
                    "투망시작↔양망시작_km", 
                    "투망끝↔양망끝_km"   
                ])

        # 파일명 추출
        input_filename = os.path.basename(input_file)

        # 문자열 변환 (None 대응)
        dist_start_pred_str = f"{distance_start_pred_km:.3f}" if distance_start_pred_km is not None else "N/A"
        dist_end_pred_str = f"{distance_end_pred_km:.3f}" if distance_end_pred_km is not None else "N/A"
        visibility_str = f"{visibility_km:.3f}" if visibility_km is not None else "N/A"
        dist_tumang_start_yang_start_str = f"{distance_tumang_start_yang_start_km:.3f}" if distance_tumang_start_yang_start_km is not None else "N/A"
        dist_tumang_end_yang_end_str = f"{distance_tumang_end_yang_end_km:.3f}" if distance_tumang_end_yang_end_km is not None else "N/A"
        result_str = prediction_result if prediction_result is not None else "판단 불가"


        # 결과 쓰기
        with open(result_csv_path, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                input_filename, 
                dist_start_pred_str, 
                dist_end_pred_str, 
                visibility_str, 
                result_str, 
                dist_tumang_start_yang_start_str,
                dist_tumang_end_yang_end_str
            ])

        print("✅ CSV 저장 완료: hycom_prediction_result.csv")


        # ###############################################################################
        # 결과 시각화
        # 1. DataFrame으로 변환
        df_sim = o.result[['lat', 'lon', 'origin_marker']].to_dataframe().reset_index()
        df_sim = df_sim.rename(columns={'trajectory': 'seed_id', 'time': 'timestamp'})

        # 2. 각 origin_marker의 마지막 row만 선택 (비활성화 직전 위치)
        last_df = df_sim.sort_values(['origin_marker', 'timestamp']).groupby('origin_marker').tail(1).reset_index(drop=True)

        # 3. 중심 입자 위치 (참고용)
        center_idx = len(last_df) // 2
        center_row = last_df.iloc[center_idx]
        print(f"🧭 중심 origin_marker {center_row['origin_marker']} → lat: {center_row['lat']:.5f}, lon: {center_row['lon']:.5f}")

        # 4. 시각화
        plt.figure(figsize=(10, 7))

        # 모든 입자의 궤적
        for seed_id, group in df_sim.groupby('seed_id'):
            plt.plot(group['lon'], group['lat'], color='gray', alpha=0.4)

        # 비활성화 직전 위치 연결 선 및 점
        plt.scatter(last_df['lon'], last_df['lat'], c='orange', s=10, label='비활성화 직전 위치')
        plt.plot(last_df['lon'], last_df['lat'], color='orange', linewidth=10, alpha=0.6, label='비활성화 위치 경로')

        # 투망(시딩) 위치
        plt.scatter(df_tumang['lon'], df_tumang['lat'], s=30, color='blue', marker='^', label='투망(0)')
        plt.scatter(df_yangmang['lon'], df_yangmang['lat'], s=30, color='red', marker='^', label='양망(1)')

        # ======================== ✨ 핵심 시각화 요소 ✨ ========================

        # # 실제 양망 시작/끝 위치
        # if yang_start_lon is not None:
        #     # 양망 시작 위치 (X)
        #     plt.scatter(yang_start_lon, yang_start_lat, s=150, color='green', marker='X', linewidth=1, label='양망 시작 위치')
            
        #     # 양망 끝 위치 (P)
        #     plt.scatter(yang_end_lon, yang_end_lat, s=150, color='darkgreen', marker='P', linewidth=1, label='양망 끝 위치')

        #     # 예측 위치 시작/끝
        #     # 시작 입자 최종 예측 위치 (*)
        #     plt.scatter(start_lon_pred, start_lat_pred, s=150, color='red', marker='*', edgecolor='black', linewidth=1, label='시작 예측 최종 위치')

        #     # 끝 입자 최종 예측 위치 (s)
        #     plt.scatter(end_lon_pred, end_lat_pred, s=150, color='red', marker='s', edgecolor='black', linewidth=1, label='끝 예측 최종 위치')

        # ====================================================================

        # plt.title("시뮬레이션 궤적 및 예측 최종 위치 vs 실제 행동 위치")
        # plt.xlabel("Longitude")
        # plt.ylabel("Latitude")
        # plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()

        # === 시각화 결과 자동 저장 ===
        plot_output_dir = r"C:\Users\HUFS\Desktop\opendrift_middle\hycom_시각화결과_other"
        os.makedirs(plot_output_dir, exist_ok=True)
        plot_filename = f"{input_basename}.png"
        plot_path = os.path.join(plot_output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ 시각화 결과 저장 완료: {plot_path}")
    
    # except Exception as e:
    #     print(f"❌ 오류 발생: {input_file} - {e}")
    #     with open(error_log_path, "a", newline="", encoding="utf-8-sig") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([os.path.basename(input_file), type(e).__name__, str(e)])
    #     continue