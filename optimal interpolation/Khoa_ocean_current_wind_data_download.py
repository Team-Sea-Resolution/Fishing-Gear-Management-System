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

plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False


TARGET_SEQ = [3,0,3,1,3]   # 시퀀스 탐색용 (전역 변수로 이동)
# ─────────────────────────────────────────────────────
# 1) 시퀀스 탐색용 헬퍼
# ─────────────────────────────────────────────────────
def find_sequence_groups(behaviors, target=TARGET_SEQ):
    """연속 중복이 제거된 리스트에서 타겟 시퀀스의 위치를 찾습니다."""
    grp = [behaviors[0]]
    for b in behaviors[1:]:
        if b != grp[-1]:
            grp.append(b)
    n,m = len(grp), len(target)
    for i in range(n-m+1):
        if grp[i:i+m] == target:
            return i, i+m # 압축된 리스트에서의 시작, 끝 인덱스
    return None

def load_df(path):
    """시퀀스 스캔을 위한 최소한의 DataFrame 로더 (오류 처리 포함)"""
    geo = json.load(open(path, 'r', encoding='utf-8'))
    rows = []
    for feat in geo.get('features', []):
        p = feat['properties']
        rows.append({
            'time_stamp': p.get('time_stamp'),
            'fishery_behavior': p.get('fishery_behavior'),
        })
    df = pd.DataFrame(rows)
    # 원본 스크립트와 동일하게 errors='coerce'를 사용하여 안정성 확보
    df['time_stamp'] = pd.to_datetime(df['time_stamp'], errors='coerce') 
    return df.sort_values('time_stamp', ignore_index=True)

def locate_sequence(df):
    """DataFrame에서 TARGET_SEQ의 *원본 인덱스* 위치(시작, 끝)를 반환합니다."""
    raw = df['fishery_behavior'].tolist()
    if len(raw) < len(TARGET_SEQ): 
        return None
        
    # 그룹별 인덱스 매핑 (압축된 리스트의 인덱스가 원본 리스트의 몇 번째 인덱스에서 시작됐는지 추적)
    grp = [raw[0]]; starts=[0]; prev=raw[0]
    for i,b in enumerate(raw[1:], start=1):
        if b != prev:
            grp.append(b)
            starts.append(i) # 새로운 그룹(b)이 시작된 원본 인덱스(i)
            prev = b
            
    loc = find_sequence_groups(grp) # 압축 리스트 기준 (i0, i1)
    if not loc:
        return None
        
    i0, i1 = loc
    start_idx = starts[i0] # 시퀀스 시작 그룹의 원본 인덱스
    
    # 시퀀스 끝 그룹의 원본 *마지막* 인덱스
    # (i1은 끝나는 그룹의 다음 인덱스이므로, starts[i1]-1이 마지막 인덱스가 됨)
    end_idx = (starts[i1]-1) if i1 < len(starts) else len(raw)-1
    
    return start_idx, end_idx

def seq_times(df, loc):
    """loc (시작, 끝 인덱스)를 기반으로 실제 시작/끝 타임스탬프를 반환합니다."""
    s,e = loc
    return df.loc[s,'time_stamp'], df.loc[e,'time_stamp']

# ─────────────────────────────────────────────────────
# 2) 클러스터 스캐너
# ─────────────────────────────────────────────────────
def scan_clusters(file_list):
    """
    모든 파일의 시퀀스 시간(t0, t1)을 수집하고,
    겹치는 구간을 클러스터로 묶어 첫/마지막 파일명 리스트를 반환합니다.
    """
    intervals = []
    print(f"🔬 {len(file_list)}개 전체 파일 스캔 시작...")
    
    # 1. 모든 파일의 (t0, t1, fn) 수집
    for path in file_list:
        fn = os.path.basename(path)
        try:
            df  = load_df(path)
            loc = locate_sequence(df)
        except Exception as e:
            if isinstance(e, json.decoder.JSONDecodeError):
                print(f"  ⚠️ JSON 형식 오류, 스캔 제외: {fn}")
            else:
                print(f"  ⚠️ 스캔 중 오류, 제외: {fn} - {type(e).__name__}")
            continue
            
        if not loc: 
            # print(f"  - 시퀀스 없음, 스캔 제외: {fn}") # (로그가 너무 많아질 수 있으므로 주석 처리)
            continue
            
        t0,t1 = seq_times(df,loc)
        intervals.append((t0,t1,fn))
    
    print(f"  ✅ 시퀀스 포함 파일 {len(intervals)}개 발견.")
    
    # 2. 시작 시간 순 정렬
    intervals.sort(key=lambda x: x[0])
    
    # 3. 겹치는 구간끼리 묶기 (클러스터링)
    clusters = []
    cur, cur_end = [], None
    for iv in intervals:
        s,e,fn = iv
        if not cur: # 첫 번째 클러스터 시작
            cur = [iv]; cur_end = e
        elif s <= cur_end: # 현재 클러스터와 시간이 겹침
            cur.append(iv)
            cur_end = max(cur_end, e) # 클러스터의 끝 시간 갱신
        else: # 새로운 클러스터 시작
            clusters.append(cur)
            cur = [iv]; cur_end = e
    if cur:
        clusters.append(cur)
        
    print(f"  📊 총 {len(clusters)}개의 조업 클러스터로 그룹화 완료.")

    # 4. 클러스터별 첫/마지막 파일명 추출
    first_list = [cluster[0][2] for cluster in clusters]
    last_list  = [cluster[-1][2] for cluster in clusters]
    
    return first_list, last_list

# ==============================================================================
# 🗂️ 2. 환경 설정 및 메인 루프
# ==============================================================================

geojson_dir = r"D:\어선행적데이터\Training\02.라벨링데이터\TL_01.자망.zip"
# geojson_dir = r'D:\어선행적데이터\Validation\02.라벨링데이터\VL_01.자망.zip'
geojson_files = glob.glob(os.path.join(geojson_dir, "*.geojson"))


# ✨ (수정) 1. 먼저 모든 파일을 glob으로 찾습니다.
all_files = glob.glob(os.path.join(geojson_dir, "*.geojson"))

# ✨ (수정) 2. 스캔 함수를 호출하여 필터링합니다.
first_list, last_list = scan_clusters(all_files)

# ✨ (수정) 3. 필터링된 파일 목록(first + last)으로 geojson_files 변수를 새로 정의합니다.
# dict.fromkeys를 사용해 중복(클러스터가 파일 1개인 경우)을 제거합니다.
filtered_files_names = list(dict.fromkeys(first_list + last_list))
geojson_files = [os.path.join(geojson_dir, fn) for fn in filtered_files_names]

print(f"🎉 필터링 완료! 전체 {len(all_files)}개 파일 중 {len(geojson_files)}개 파일 처리 시작.")
print("-" * 60)


error_log_path = os.path.join(geojson_dir, "error_log.csv")

# 에러 로그 초기화
if not os.path.exists(error_log_path):
    with open(error_log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["파일명", "오류종류", "오류메시지"])

# 전체 자동 처리 루프
for input_file in geojson_files:
    try:
        input_filename = os.path.basename(input_file)
        visibility = None            # 가시거리(m)
        distance_km = None           # 중간 투망 ↔ 양망 거리(km)
        prediction_result = "판단 불가"  # 예측 성공 여부

        print(f"\n===== 처리 시작: {input_filename} =====")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)


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

        lat_min = 31.82632165
        lat_max = 35.9863583
        lon_min = 123.9609466
        lon_max = 129.2298249

        print(f"🌍 lat 범위=({lat_min}, {lat_max}), lon 범위=({lon_min}, {lon_max})")

        lat_grid = np.arange(round(lat_min, 2), round(lat_max, 2) + 0.01, 0.01)
        lon_grid = np.arange(round(lon_min, 2), round(lon_max, 2) + 0.01, 0.01)

        print(f"🌍 lat_grid={lat_grid.shape}, lon_grid={lon_grid.shape}")
        print(f"===== 처리 완료: {input_filename} =====\n")



        # ====== NetCDF 파일 경로 미리 설정 ======
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        nc_folder = r".\KHOA_nc_data"
        os.makedirs(nc_folder, exist_ok=True)


            # ====== API 호출 및 보간 수행 ======
        service_key = 'CDXK66UUPZmXiNtOX7UYBQ=='
        base_url = "http://www.khoa.go.kr/api/oceangrid/tidalCurrentAreaGeoJson/search.do"
        all_data = []


        output_path = os.path.join(nc_folder, f"{input_basename}_uv.nc")

        # ====== 파일 존재 시 생략 ======
        if os.path.exists(output_path):
            print(f"🔄 이미 NetCDF 존재, 다운로드 생략: {output_path}")
        else:

        # 📌 5. API 호출 및 데이터 저장
            for t in time_list:
                params = {
                    "DataType": "tidalCurrentAreaGeoJson",
                    "ServiceKey": service_key,
                    "Date": t.strftime("%Y%m%d"),
                    "Hour": t.strftime("%H"),
                    "Minute": "00",
                    "MinX": lon_min,
                    "MaxX": lon_max,
                    "MinY": lat_min,
                    "MaxY": lat_max,
                    "Scale": 2000000
                }

                try:
                    response = requests.get(base_url, params=params)
                    if response.status_code == 200:
                        geojson_data = response.json()
                        print(geojson_data)
                        for feature in geojson_data.get("features", []):
                            p = feature["properties"]
                            lat = p.get("lat")
                            lon = p.get("lon")
                            spd = p.get("current_speed")
                            direction = p.get("current_direct")
                            if None in (lat, lon, spd, direction):
                                continue
                            spd_m = spd / 100
                            rad = np.radians(direction)
                            u = spd_m * np.sin(rad)
                            v = spd_m * np.cos(rad)
                            all_data.append({
                                "time": t,
                                "lat": lat,
                                "lon": lon,
                                "u": u,
                                "v": v
                            })

                        else:
                            # 💡 응답이 200이지만 GeoJSON이 아닐 때 본문을 출력하여 원인 파악
                            print(f"❌ API 실패: status=200. 응답 본문이 GeoJSON이 아님.")
                            print(f"   (시간: {t.strftime('%Y%m%d %H:%M')}) 응답: {response.text[:100]}...") # 처음 100자 출력
                    else:
                        print(f"❌ API 실패: status={response.status_code}")
                except Exception as e:
                    print(f"[예외] {e}")

            # 📌 6. 정방격자 보간 및 NetCDF 생성
            df_all = pd.DataFrame(all_data)
            times = sorted(df_all["time"].unique())
            u_interp = []
            v_interp = []

            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

            for t in times:
                sub = df_all[df_all["time"] == t]
                points = np.array(sub[["lon", "lat"]])
                u_vals = sub["u"].values
                v_vals = sub["v"].values

                u_grid = griddata(points, u_vals, (lon_mesh, lat_mesh), method='linear')
                v_grid = griddata(points, v_vals, (lon_mesh, lat_mesh), method='linear')

                u_interp.append(u_grid)
                v_interp.append(v_grid)

            # 7. OpenDrift 인식 가능하도록 변수명 + 메타데이터 설정
            ds = xr.Dataset(
                {
                    "eastward_sea_water_velocity": (["time", "lat", "lon"], np.array(u_interp)),
                    "northward_sea_water_velocity": (["time", "lat", "lon"], np.array(v_interp)),
                },
                coords={
                    "time": times,
                    "lat": lat_grid,
                    "lon": lon_grid,
                },
                attrs={
                    "title": "정방격자 보간된 KHOA 해류 예측 데이터",
                    "source": "tidalCurrentAreaGeoJson API"
                }
            )

            # 변수에 CF-convention 메타데이터 추가
            ds["eastward_sea_water_velocity"].attrs["standard_name"] = "eastward_sea_water_velocity"
            ds["eastward_sea_water_velocity"].attrs["units"] = "m s-1"
            ds["northward_sea_water_velocity"].attrs["standard_name"] = "northward_sea_water_velocity"
            ds["northward_sea_water_velocity"].attrs["units"] = "m s-1"

            nc_folder = r".\KHOA_nc_data"
            os.makedirs(nc_folder, exist_ok=True)

            output_path = os.path.join(nc_folder, f"{input_basename}_uv.nc")
            ds.to_netcdf(output_path)
            print(f"✅ NetCDF 저장 완료: {output_path}")



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
        wind_folder = r".\wind_data"
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
        output_csv_path = r".\가시거리csv\visibility_log_train3.csv"

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
        nc_folder = r".\KHOA_nc_data"
        merged_file = os.path.join(nc_folder, f"{input_basename}_uv.nc")

        hycom_folder = r".\hycom_data"
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



    
    except Exception as e:
        print(f"❌ 오류 발생: {input_file} - {e}")
        with open(error_log_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(input_file), type(e).__name__, str(e)])
        continue