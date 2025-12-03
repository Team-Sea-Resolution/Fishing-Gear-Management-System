import math
import os
import json
import glob
import time
import csv
import sys # exit() 대신 사용

import cdsapi
from datetime import datetime, timedelta
from dateutil import parser
import numpy as np
import pandas as pd
import xarray as xr
import requests
import geopandas as gpd
from shapely.geometry import Point, LineString

# geopy.distance.geodesic 사용을 위한 임포트
try:
    from geopy.distance import geodesic
except ImportError:
    print("❌ geopy 라이브러리가 설치되어 있지 않습니다. 'pip install geopy' 명령으로 설치해주세요.")
    sys.exit(1)


# OpenDrift 및 관련 임포트
try:
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_netCDF_CF_generic
    from collections import OrderedDict
except ImportError:
    print("❌ OpenDrift 라이브러리가 설치되어 있지 않습니다. 설치 후 다시 실행해주세요.")
    sys.exit(1)

# 시각화 임포트
import matplotlib.pyplot as plt
import matplotlib as mpl

# ======================================================================
# ⚙️ 환경 설정 및 경로 정의
# ======================================================================

# --- PROJ 환경 변수 설정 (pyproj ≥3용 PROJ_DATA 포함)
# 사용자 환경에 맞게 경로를 수정해야 합니다.
os.environ['PROJ_LIB'] = r'C:\Users\HUFS\anaconda3\envs\opendrift_env\Library\share\proj'
os.environ['PROJ_DATA'] = r'C:\Users\HUFS\anaconda3\envs\opendrift_env\Library\share\proj'

# --- 주요 폴더 및 파일 경로
# *주의: GeoJSON 디렉토리 경로가 zip 파일로 되어 있어 glob 처리에 문제가 있을 수 있습니다.
# 압축이 해제된 폴더 경로를 사용해주세요. (예: r"D:\어선행적데이터\Training\02.라벨링데이터\TL_01.자망")
GEOJSON_DIR = r"D:\어선행적데이터\Training\02.라벨링데이터\TL_01.자망.zip" 
OI_FOLDER_PATH = r"D:\current_oi\creat_new_ocean_oi\OI_data"
WIND_FOLDER_PATH = r"C:\Users\HUFS\Desktop\opendrift_middle\wind_data"

# --- Matplotlib 한글 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# 1. 각 데이터 파일이 위치한 폴더 경로 정의
paths = {
    'buoy': r"D:\current_oi\BASE_OUTPUT_FOLDER\buoy_data",
    'hfradar': r'D:\current_oi\BASE_OUTPUT_FOLDER\hfradar_data',
    'hycom': r'C:\Users\HUFS\Desktop\opendrift_middle\fix_range_hycom_data',
    'khoa': r"D:\current_oi\khoa_down\KHOA_nc_data"
}

# ============================================================================================
# ---------------------------------------------
# ⚖️ 가중치 조합 정의 (0.0에서 1.0까지 0.1 단위)
# ---------------------------------------------
ALL_WEIGHT_PAIRS = []
for i in range(11):
    # 부동소수점 오류 방지를 위해 round 사용
    w_h = round(i * 0.1, 1)
    w_k = round(1.0 - w_h, 1)
    ALL_WEIGHT_PAIRS.append((w_h, w_k))

# ======================================================================
# 🎣 OpenDrift 사용자 정의 모델: ConnectedNetDrift
# ======================================================================
class ConnectedNetDrift(OceanDrift):
    """
    OpenDrift 모델을 상속받아 자망의 연결성을 모사하는 사용자 정의 클래스.
    일정 간격의 입자 쌍이 'ideal_distance_m'를 유지하도록 강제하는 힘을 추가합니다.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ideal_distance_m = 270 
        self.k = 0.05 
        self.step = 2 
        self.adjustment_loops = 2

    def update(self):
        """
        OpenDrift의 기본 업데이트 후 연결성 조정을 수행합니다.
        """
        super().update()
        lon = self.elements.lon.copy()
        lat = self.elements.lat.copy()
        n = len(lon)

        for _ in range(self.adjustment_loops):
            for i in range(self.step, n):
                prev_coord = (lat[i - self.step], lon[i - self.step])
                curr_coord = (lat[i], lon[i])
                
                # geopy.distance.geodesic 사용
                dist = geodesic(prev_coord, curr_coord).meters 
                delta = dist - self.ideal_distance_m

                if abs(delta) > 0.1: # 유의미한 차이가 있을 경우 조정
                    dlat = lat[i] - lat[i - self.step]
                    dlon = lon[i] - lon[i - self.step]
                    
                    # 조정 스케일 계산 (차이에 비례하고 강도 계수 k 반영)
                    scale = delta / dist * self.k

                    # 서로 밀고 당기도록 조정
                    lat[i]           -= dlat * scale
                    lon[i]           -= dlon * scale
                    lat[i - self.step] += dlat * scale
                    lon[i - self.step] += dlon * scale

        self.elements.lon[:] = lon
        self.elements.lat[:] = lat


# ======================================================================
# 🛠️ 유틸리티 함수
# ======================================================================

def load_geojson_to_dataframe(path):
    """GeoJSON 파일을 읽어 DataFrame으로 변환합니다."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = []
    for feat in data.get('features', []):
        p = feat['properties']
        records.append({
            'time_stamp': p['time_stamp'],
            'lon': p['longitude'],
            'lat': p['latitude'],
            'fishery_behavior': p['fishery_behavior']
        })
    df = pd.DataFrame(records)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    return df.sort_values('time_stamp').reset_index(drop=True)

def initialize_error_log():
    """에러 로그 파일을 초기화합니다."""
    if not os.path.exists(os.path.dirname(ERROR_LOG_PATH)):
        os.makedirs(os.path.dirname(ERROR_LOG_PATH), exist_ok=True)
    if not os.path.exists(ERROR_LOG_PATH):
        with open(ERROR_LOG_PATH, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["파일명", "오류종류", "오류메시지"])

def write_result_csv(filename, distance_km, visibility_km, prediction_result, tumang_yang_km):
    """시뮬레이션 결과를 CSV 파일에 기록합니다."""
    # 헤더 작성 (파일이 없을 경우)
    if not os.path.exists(RESULT_CSV_PATH):
        with open(RESULT_CSV_PATH, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename", 
                "예측거리_km", 
                "가시거리_km", 
                "예측결과", 
                "투망중간↔양망중간_km"
            ])

    # 문자열 변환 (None 대응)
    distance_str = f"{distance_km:.3f}" if distance_km is not None else "N/A"
    visibility_str = f"{visibility_km:.3f}" if visibility_km is not None else "N/A"
    tumang_yang_str = f"{tumang_yang_km:.3f}" if tumang_yang_km is not None else "N/A"
    result_str = prediction_result if prediction_result is not None else "판단 불가"

    # 결과 쓰기
    with open(RESULT_CSV_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            filename, 
            distance_str, 
            visibility_str, 
            result_str, 
            tumang_yang_str
        ])

def load_ocean_data_set(base_hycom_filename: str, path_dict: dict) -> dict:
    """
    주어진 HYCOM 파일명을 기준으로 관련된 모든 데이터(.nc) 파일을 찾아 로드합니다.
    (이 함수는 이전 코드와 동일합니다)
    """
    if not base_hycom_filename.endswith('_hycom.nc'):
        print(f"오류: '{base_hycom_filename}'은 유효한 HYCOM 파일명이 아닙니다.")
        return {}
    
    base_name = base_hycom_filename.replace('_hycom.nc', '')
    
    filenames_to_load = {
        'hycom': base_hycom_filename,
        'buoy': f"{base_name}_buoy.nc",
        'hfradar': f"{base_name}_hfradar.nc",
        'khoa': f"{base_name}_uv.nc"
    }

    loaded_datasets = {}
    print(f"✅ 공통 파일명 기반: {base_name}")
    for data_type, filename in filenames_to_load.items():
        full_path = os.path.join(path_dict.get(data_type, ''), filename)
        
        if os.path.exists(full_path):
            try:
                loaded_datasets[data_type] = xr.open_dataset(full_path, engine="netcdf4")
                print(f"✔️  성공: '{filename}' 로드 완료.")
            except Exception as e:
                print(f"❌ 실패: '{filename}' 로드 중 오류 발생. ({e})")
                loaded_datasets[data_type] = None
        else:
            # HYCOM 파일 외에는 파일이 없는 것이 일반적일 수 있으므로 경고 대신 간단한 메시지로 변경
            if data_type != 'hycom':
                pass # print(f"   -> 정보: '{filename}' 파일이 존재하지 않습니다.")
            else:
                 print(f"⚠️  경고: 파일을 찾을 수 없음 - {full_path}")
            loaded_datasets[data_type] = None
            
    return loaded_datasets



# --- 메인 실행 부분 ---
for w_hycom, w_khoa in ALL_WEIGHT_PAIRS: 
    # --- 출력 경로
    RESULT_CSV_PATH = rf"C:\Users\HUFS\Desktop\opendrift_middle\예측csv\oi_{w_hycom}_{w_khoa}_prediction_result.csv"
    PLOT_OUTPUT_DIR = rf"C:\Users\HUFS\Desktop\opendrift_middle\oi_{w_hycom}_{w_khoa}_시각화결과"
    ERROR_LOG_PATH = os.path.join(os.path.dirname(RESULT_CSV_PATH), "error_log.csv")

    # 최종 결과물을 저장할 기본 폴더 경로
    BASE_OUTPUT_FOLDER_PATH = rf'D:\current_oi\creat_{w_hycom}_{w_khoa}_ocean_oi'


    # 저장할 하위 폴더 경로 설정
    khoa_hycom_vector_folder_path = os.path.join(BASE_OUTPUT_FOLDER_PATH, 'khoa_hycom_vector')
    OI_FOLDER_PATH = os.path.join(BASE_OUTPUT_FOLDER_PATH, 'OI_data2')

    os.makedirs(khoa_hycom_vector_folder_path, exist_ok=True)
    os.makedirs(OI_FOLDER_PATH, exist_ok=True)

    # 1. HYCOM 데이터가 있는 폴더 경로
    hycom_folder_path = paths['hycom']

    # 2. 해당 폴더에서 '_hycom.nc'로 끝나는 모든 파일 목록을 가져오기
    try:
        hycom_files = [f for f in os.listdir(hycom_folder_path) if f.endswith('_hycom.nc')]
        if not hycom_files:
            print(f"경고: '{hycom_folder_path}' 폴더에 HYCOM 파일이 없습니다.")
    except FileNotFoundError:
        print(f"오류: '{hycom_folder_path}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        hycom_files = []

    # 3. 각 HYCOM 파일을 순회하며 작업 수행
    for hycom_filename in hycom_files:
        print(f"\n{'='*20} [{hycom_filename}] 처리 시작 {'='*20}")
        
        # 기준 HYCOM 파일에 해당하는 데이터 셋 불러오기
        ocean_data = load_ocean_data_set(hycom_filename, paths)


        # --- 파일 존재 여부 확인 ---
        ds_hycom = ocean_data.get('hycom')
        ds_khoa  = ocean_data.get('khoa')
        ds_buoy  = ocean_data.get('buoy')
        ds_radar = ocean_data.get('hfradar')

        # khoa 데이터 확인용 출력
        if ds_khoa is not None:
            print("✅ KHOA 데이터 로드 완료")
        if ds_khoa is None:
            print("⚠️ KHOA 데이터 로드 XXXXXX")


        if ds_hycom is None or ds_khoa is None:
            print(f"⚠️ 필수 데이터(HYCOM/KHOA) 없음 → 다음 파일로 넘어갑니다.")
            continue
        if ds_buoy is None or ds_radar is None:
            print(f"⚠️ 관측 데이터(Buoy/HF) 없음 → OI 생략, 벡터 합성만 진행")
        

        try:
            # --- [진단 코드 시작] ---
            print("\n--- 🚧 좌표계 진단 시작 ---")
            print(f"HYCOM 시간: {ds_hycom.time.min().values} 부터 {ds_hycom.time.max().values}")
            print(f"KHOA 시간: {ds_khoa.time.min().values} 부터 {ds_khoa.time.max().values}")
            
            print(f"HYCOM 위도: {ds_hycom.lat.min().values} ~ {ds_hycom.lat.max().values}")
            print(f"KHOA 위도: {ds_khoa.lat.min().values} ~ {ds_khoa.lat.max().values}")
            
            print(f"HYCOM 경도: {ds_hycom.lon.min().values} ~ {ds_hycom.lon.max().values}")
            print(f"KHOA 경도: {ds_khoa.lon.min().values} ~ {ds_khoa.lon.max().values}")
            print("--- 🚧 좌표계 진단 종료 ---\n")
            # --- [진단 코드 끝] ---
            # ---------------------------
            # 2. 시간과 좌표 맞추기
            # ---------------------------
            lat_khoa = ds_khoa['lat']
            lon_khoa = ds_khoa['lon']
            time_khoa = ds_khoa['time']

            # HYCOM 표층만 선택
            u_hycom = ds_hycom['x_sea_water_velocity'].isel(depth=0)
            v_hycom = ds_hycom['y_sea_water_velocity'].isel(depth=0)

            # reindex 사용
            u_hycom_reindexed = u_hycom.reindex(time=time_khoa, lat=lat_khoa, lon=lon_khoa, method='nearest')
            v_hycom_reindexed = v_hycom.reindex(time=time_khoa, lat=lat_khoa, lon=lon_khoa, method='nearest')


            # KHOA 속도
            u_khoa = ds_khoa['eastward_sea_water_velocity']
            v_khoa = ds_khoa['northward_sea_water_velocity']

            # ---------------------------
            # 3. 벡터 합성
            # ---------------------------

            # 이후 u_hycom_interp 대신 u_hycom_reindexed 사용
            u_combined = w_hycom * u_hycom_reindexed + w_khoa * u_khoa
            v_combined = w_hycom * v_hycom_reindexed + w_khoa * v_khoa
            speed_combined = np.sqrt(u_combined**2 + v_combined**2)

            # ---------------------------
            # 4. NetCDF 저장
            # ---------------------------
            ds_combined = xr.Dataset(
                {
                    "eastward_velocity": (["time", "lat", "lon"], u_combined.data),
                    "northward_velocity": (["time", "lat", "lon"], v_combined.data),
                    "speed": (["time", "lat", "lon"], speed_combined.data),
                },
                coords={
                    "time": time_khoa.data,
                    "lat": lat_khoa.data,
                    "lon": lon_khoa.data
                },
                attrs={
                    "title": "Combined HYCOM + KHOA currents",
                    "source": "HYCOM (0.5) + KHOA (0.5)",
                }
            )

            
            # --- [디버깅] NaN 값 확인 ---
            total_points = ds_combined["eastward_velocity"].size
            nan_count = np.count_nonzero(np.isnan(ds_combined["eastward_velocity"].values))
            non_nan_count = total_points - nan_count
            
            if non_nan_count == 0:
                print("❌ 치명적 오류: ds_combined의 모든 값이 NaN입니다.")
            else:
                print(f"✅ ds_combined 유효성 검사: (유효 데이터: {non_nan_count} / 전체: {total_points} | NaN: {nan_count})")




            base_with_suffix = os.path.splitext(hycom_filename)[0]
            base_name = hycom_filename.replace('_hycom.nc','')
            hycom_khoa_vector_output_filename = os.path.join(
                khoa_hycom_vector_folder_path, f"{base_name}_hycom_khoa_vector.nc"
            )

        except Exception as e:
            print(f"❌ 처리 중 오류 발생 ({hycom_filename}): {e}")
            continue  # 오류 발생 시 다음 파일로 이동

        
  
        ds_combined.to_netcdf(hycom_khoa_vector_output_filename)
        print(f"\n✅ 벡터 합성한 NetCDF 파일 생성 완료: {hycom_khoa_vector_output_filename}")
                
        # ================================================================================================================================
        # ================================================================================================================================
        # ================================================================================================================================
        # ================================================================================================================================
        # OI 최적 내삽법 적용
                

        from scipy.spatial import cKDTree
        from math import radians, sin, cos, sqrt, atan2

        # ===================================================================
        # 1. 설정 (Configuration)
        # ===================================================================

        # --- 입력 파일 경로 ---
        # 1. 배경 모델 데이터 (HYCOM + KHOA, 격자 형태)
        BACKGROUND_NC_FILE = ds_combined
        # 2. 해상 부이 관측 데이터 (이전에 생성한 파일)



        # --- 자료 동화 주요 파라미터 ---
        # 영향 반경 (단위: km): 하나의 관측값이 주변 몇 km까지 영향을 미칠지 결정하는 가장 중요한 변수
        # (실험을 통해 적절한 값을 찾아야 함, 보통 10~50km 사이에서 시작)
        INFLUENCE_RADIUS_KM = 50.0

        # ===================================================================
        # 2. 데이터 준비 및 전처리
        # ===================================================================

        print("🔄 1. 데이터 로딩 및 전처리를 시작합니다...")
        # xarray Dataset을 pandas DataFrame으로 변환 후 하나로 합치기
        ds_bg = ds_combined
        df_buoy = ds_buoy.to_dataframe().reset_index()
        df_radar = ds_radar.to_dataframe().reset_index()
        df_obs = pd.concat([df_buoy, df_radar], ignore_index=True)

        # --- 관측 데이터 형식 통일 (u, v 성분으로 변환) ---
        # 유속(cm/s) -> m/s로 변환
        df_obs['current_speed_mps'] = df_obs['current_speed'] / 100.0
        # 유향(degree) -> radian으로 변환
        df_obs['direction_rad'] = np.deg2rad(df_obs['current_direct'])

        # u(동-서), v(남-북) 성분 계산 (기상/해양학 표준)
        # u = speed * sin(direction)
        # v = speed * cos(direction)
        df_obs['u_obs'] = df_obs['current_speed_mps'] * np.sin(df_obs['direction_rad'])
        df_obs['v_obs'] = df_obs['current_speed_mps'] * np.cos(df_obs['direction_rad'])

        # 필요한 컬럼만 선택 및 결측치 제거
        df_obs = df_obs[['time', 'lat', 'lon', 'u_obs', 'v_obs']].dropna()

        # ds_buoy.time은 tz-naive datetime64 → 그냥 UTC로 가정
        print("배경 데이터 time은 timezone-naive → UTC로 처리합니다.")

        # 관측 데이터도 동일하게 UTC로 맞추기
        if df_obs['time'].dt.tz is None:
            df_obs['time'] = df_obs['time'].dt.tz_localize('UTC')
        else:
            df_obs['time'] = df_obs['time'].dt.tz_convert('UTC')



        # ===================================================================
        # 3. 자료 동화 수행 (Optimal Interpolation)
        # ===================================================================
        print(f"🚀 2. 자료 동화를 시작합니다... (영향 반경: {INFLUENCE_RADIUS_KM}km)")

        # --- 결과 저장을 위한 새로운 변수 생성 (배경장 복사) ---
        # u, v 변수명이 다를 경우 아래 'eastward_velocity', 'northward_velocity'를 실제 변수명으로 수정
        u_an = ds_bg['eastward_velocity'].copy(deep=True) 
        v_an = ds_bg['northward_velocity'].copy(deep=True)

        # --- 격자점 좌표 준비 ---
        # 경도(lon), 위도(lat) 좌표를 1차원 배열로 변환
        lons = ds_bg.lon.values
        lats = ds_bg.lat.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        grid_points = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T

        # # --- NaN 포함 행 제거 ---
        # nan_rows = np.isnan(grid_points).any(axis=1)
        # if nan_rows.any():
        #     print(f"NaN 포함 행 수: {nan_rows.sum()} 제거 후 KDTree 생성")
        # grid_points = grid_points[~nan_rows]  # NaN 있는 행 제거

        # --- 공간 검색을 위한 KD-Tree 생성 ---
        kdtree = cKDTree(grid_points)
        print(ds_bg)

        # --- 시간별 루프 실행 ---
        for t_idx, current_time in enumerate(ds_bg.time.values):
            print(f"  - 시간 처리 중: {str(current_time)}")
            
            # 현재 시간과 일치하는 관측 데이터 필터링
            obs_t = df_obs[df_obs['time'] == current_time]
            
            if len(obs_t) == 0:
                continue # 현재 시간에 관측값이 없으면 다음 시간으로
                
            # 배경장 u, v 값 (분석장 업데이트를 위해 복사)
            u_bg_t = ds_bg['eastward_velocity'][t_idx].values
            v_bg_t = ds_bg['northward_velocity'][t_idx].values
            
            u_an_t = u_an[t_idx].values
            v_an_t = v_an[t_idx].values
            
            # 각 격자점에 대한 가중치 합과 혁신(innovation) 합을 저장할 배열
            total_weights = np.zeros_like(u_bg_t)
            total_u_update = np.zeros_like(u_bg_t)
            total_v_update = np.zeros_like(u_bg_t)

            # 각 관측값에 대해 영향 계산
            for _, obs in obs_t.iterrows():
                obs_point = np.array([obs['lon'], obs['lat']])
                
                # 배경장 값을 관측 위치로 보간 (가장 가까운 격자점 값 사용)
                dist, idx = kdtree.query(obs_point)
                bg_point_flat_idx = idx
                bg_point_coords = np.unravel_index(bg_point_flat_idx, u_bg_t.shape)
                
                u_bg_at_obs = u_bg_t[bg_point_coords]
                v_bg_at_obs = v_bg_t[bg_point_coords]

                # 혁신 (Innovation) 계산: 관측값과 배경장의 차이
                u_innov = obs['u_obs'] - u_bg_at_obs
                v_innov = obs['v_obs'] - v_bg_at_obs
                
                # 영향 반경 내의 모든 격자점 찾기 (단위: degree)
                # 위도 1도는 약 111km
                radius_deg = INFLUENCE_RADIUS_KM / 111.0 
                nearby_indices = kdtree.query_ball_point(obs_point, r=radius_deg)
                
                if not nearby_indices:
                    continue
                    
                nearby_grid_points = grid_points[nearby_indices]
                
                # 관측점과 주변 격자점들 간의 거리 계산 (Haversine 공식 대신 유클리드 거리로 근사)
                distances_sq = np.sum((nearby_grid_points - obs_point)**2, axis=1)
                
                # 가중치 계산 (Gaussian weight)
                L = radius_deg  # 영향 반경을 표준편차처럼 사용
                weights = np.exp(-0.5 * distances_sq / (L**2))
                
                # 각 주변 격자점의 인덱스를 2D로 변환
                nearby_coords_2d = np.unravel_index(nearby_indices, u_bg_t.shape)
                
                # 가중치와 혁신을 누적
                total_weights[nearby_coords_2d] += weights
                total_u_update[nearby_coords_2d] += u_innov * weights
                total_v_update[nearby_coords_2d] += v_innov * weights

            # 누적된 가중치로 업데이트 값 정규화
            # 0으로 나누는 것을 방지
            mask = total_weights > 0
            u_an_t[mask] += total_u_update[mask] / total_weights[mask]
            v_an_t[mask] += total_v_update[mask] / total_weights[mask]
            
            # 최종 분석장을 업데이트
            u_an[t_idx] = u_an_t
            v_an[t_idx] = v_an_t

        # ===================================================================
        # 4. 결과 저장 (OpenDrift 호환 형식)
        # ===================================================================
        oi_file_name = os.path.join(OI_FOLDER_PATH, f"{hycom_filename}_oi.nc")
        print(f"💾 3. 동화된 결과를 OpenDrift 호환 NetCDF 파일로 저장합니다: {oi_file_name}")

        # OpenDrift가 인식할 수 있는 새로운 xarray Dataset 생성
        # 변수 이름, 좌표, 메타데이터를 지정된 형식에 맞게 구성합니다.
        ds_opendrift = xr.Dataset(
            {
                # 변수 이름을 OpenDrift 표준(e.g., eastward_sea_water_velocity)에 맞춤
                "eastward_sea_water_velocity": u_an,
                "northward_sea_water_velocity": v_an,
            },
            # 좌표는 원본 배경장(ds_buoy) 데이터셋에서 그대로 가져옴
            coords={
                "time": ds_bg.time,
                "lat": ds_bg.lat,
                "lon": ds_bg.lon,
            },
            # 파일 전체에 대한 메타데이터
            attrs={
                "title": "Data-assimilated Ocean Current Data (Optimal Interpolation)",
                "source": "Background data + Observation data",
            },
        )

        # 각 변수에 CF-convention(기후 및 예측 표준) 메타데이터 추가
        ds_opendrift["eastward_sea_water_velocity"].attrs = {
            "long_name": "Assimilated eastward sea water velocity",
            "standard_name": "eastward_sea_water_velocity",
            "units": "m s-1",  # OpenDrift는 'm/s'보다 'm s-1' 형식을 선호
        }
        ds_opendrift["northward_sea_water_velocity"].attrs = {
            "long_name": "Assimilated northward sea water velocity",
            "standard_name": "northward_sea_water_velocity",
            "units": "m s-1",
        }

        # 파일로 저장
        ds_opendrift.to_netcdf(oi_file_name)

        print("✅ 모든 작업이 완료되었습니다!")
        #                                                                    #
        # ------------------------------------------------------------------ #

        # 예시 작업: 로드된 모든 데이터셋의 정보 출력
        if all(dataset is not None for dataset in ocean_data.values()):
            print("\n[작업 예시] 모든 데이터셋이 성공적으로 로드되었습니다.")
            # hycom_data = ocean_data['hycom']
            # buoy_data = ocean_data['buoy']
            # print(f"HYCOM 시간 범위: {hycom_data.time.values[0]}")
            # print(f"Buoy 관측 지점 수: {len(buoy_data.station)}")
        else:
            print("\n[작업 예시] 일부 데이터셋이 로드되지 않아 이번 파일은 건너뜁니다.")
        
        print(f"{'='*22} [{hycom_filename}] 처리 완료 {'='*22}")
        
    initialize_error_log()
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    # 🔁 HYCOM 폴더 기준 파일 탐색 및 매칭
    try:
        oi_files = os.listdir(OI_FOLDER_PATH)
        # '_hycom.nc' 파일만 필터링하여 파일명 기준 (확장자 제거) 목록 생성
        oi_base_names = set(f.replace('_hycom.nc_oi.nc', '') for f in oi_files if '_hycom.nc_oi.nc' in f)

        if not oi_base_names:
            print(f"⚠️ {OI_FOLDER_PATH}에서 '_hycom.nc' 파일을 찾을 수 없습니다. (혹은 파일명이 '_hycom.nc'로 끝나지 않습니다)")
            continue

        print(f"총 {len(oi_base_names)}개의 HYCOM 기준 파일 처리 예정")

    except Exception as e:
        print(f"❌ 오류: HYCOM 폴더 처리 중 오류 발생 - {e}")
        continue


    # 전체 자동 처리 루프
    for input_basename in oi_base_names:
        
        # input_file은 GeoJSON 파일 (예: '20220301_1234.geojson')
        geojson_file = os.path.join(GEOJSON_DIR, f"{input_basename}.geojson")
        input_file_list = glob.glob(geojson_file)
        
        if not input_file_list:
            print(f"⚠️ GeoJSON 파일 '{input_basename}.geojson'이 {GEOJSON_DIR}에 없습니다. 건너뜁니다.")
            continue
            
        input_file = input_file_list[0]
        input_filename = os.path.basename(input_file)
        
        # 시각화 정보 초기화
        visibility = None          # 가시거리(m) (현재 코드에서는 사용자가 외부에서 제공해야 함)
        distance_km = None         # 중간 투망 ↔ 양망 거리(km)
        prediction_result = "판단 불가" 
        
        print(f"\n--- 🚀 {input_filename} 처리 시작 ---")

        df = load_geojson_to_dataframe(geojson_file)

        # 투망 시작 지점만 필터링 (1->3 또는 0->3 변화 시점)
        df['prev_behavior'] = df['fishery_behavior'].shift(1)
        drop_points = df[
            (df['fishery_behavior'] == 3) &
            (df['prev_behavior'] != 3)
        ].copy()

        df3 = df[df['fishery_behavior'] == 3].copy()
        if df3.empty or drop_points.empty:
           print("투망 구간 또는 시작 시점 데이터가 없습니다.")

        lat_min = df3['lat'].min() - 0.1
        lat_max = df3['lat'].max() + 0.1
        lon_min = df3['lon'].min() - 0.1
        lon_max = df3['lon'].max() + 0.1

        first_time = pd.to_datetime(df3['time_stamp'].min())
        last_time  = pd.to_datetime(df3['time_stamp'].max())

        # 연, 월, 일 문자열로 추출
        year  = f"{first_time.strftime('%Y')}"
        month = f"{first_time.strftime('%m')}"
        day   = f"{first_time.strftime('%d')}"

        try:
            # --- PART 1: GeoJSON → DataFrame (투망 궤적 추출) ---
            df = load_geojson_to_dataframe(input_file)
            
            # 투망 (0) 및 양망 (1) 구간 필터링
            df_tumang = df[df['fishery_behavior'] == 0].copy()
            df_yangmang = df[df['fishery_behavior'] == 1].copy()

            if df_tumang.empty or df_yangmang.empty:
                print(f"⚠️ 투망(0) 또는 양망(1) 구간 데이터가 없습니다. 건너뜁니다.")
                raise ValueError("투망 또는 양망 데이터 부족")
            
            df_tumang = df_tumang.sort_values('time_stamp').reset_index(drop=True)
            df_yangmang = df_yangmang.sort_values('time_stamp').reset_index(drop=True)
            
            start_time_sim = df_tumang['time_stamp'].min()
            end_time_sim = df_yangmang['time_stamp'].max()
            simulation_duration = end_time_sim - start_time_sim

            #=======================================================
            # PART 3: ERA5 CDS API 데이터 다운로드ㅡ
            #=======================================================

            # 날짜 범위 자동 생성 (예: ['09','10','11'] 등)
            num_days = (last_time.date() - first_time.date()).days + 1
            days = [(first_time + timedelta(days=i)).strftime("%d") for i in range(num_days)]

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
            # if os.path.exists(wind_path):
            #     print(f"🔄 이미 wind 파일 존재, 다운로드 생략: {wind_path}")
            # else:
            print("🌬️ ERA5 wind 요청:", era5_request)
            client = cdsapi.Client()
            client.retrieve(
                'reanalysis-era5-single-levels',
                era5_request,
                wind_path
            )

            # ====== 변수명 x_wind / y_wind로 변경 =====
            try:
                ds = xr.open_dataset(wind_path)

                # 기존 변수명 확인 (CDS에서 내려올 때 이름이 약간 다를 수 있음)
                rename_dict = {}
                if "u10" in ds.variables:
                    rename_dict["u10"] = "x_wind"
                elif "10m_u_component_of_wind" in ds.variables:
                    rename_dict["10m_u_component_of_wind"] = "x_wind"

                if "v10" in ds.variables:
                    rename_dict["v10"] = "y_wind"
                elif "10m_v_component_of_wind" in ds.variables:
                    rename_dict["10m_v_component_of_wind"] = "y_wind"

                if rename_dict:
                    ds = ds.rename(rename_dict)
                    ds.to_netcdf(wind_path)
                    print(f"✅ ERA5 파일 변수명 변경 완료: {rename_dict}")
                else:
                    print("⚠️ ERA5 파일에서 wind 변수명을 찾지 못했습니다.")
                ds.close()

            except Exception as e:
                print(f"❌ ERA5 wind 파일 변수명 변경 중 오류 발생: {e}")
                print(f"✅ ERA5 다운로드 완료: {wind_path}")

            # --- PART 2: OpenDrift 모델 설정 ---
            oi_file = os.path.join(OI_FOLDER_PATH, f"{input_basename}_hycom.nc_oi.nc")
            wind_file = os.path.join(WIND_FOLDER_PATH, f"{input_basename}_wind.nc")
            
            if not os.path.exists(oi_file):
                raise FileNotFoundError(f"해양 데이터 파일이 없습니다: {oi_file}")
            if not os.path.exists(wind_file):
                raise FileNotFoundError(f"바람 데이터 파일이 없습니다: {wind_file}")
                
            o = ConnectedNetDrift(loglevel=20)
            reader_uv = reader_netCDF_CF_generic.Reader(oi_file)
            reader_wind = reader_netCDF_CF_generic.Reader(wind_file)
            o.add_reader([reader_uv, reader_wind])

            o.set_config('seed:wind_drift_factor', 0.02)
            o.set_config('drift:stokes_drift', True)
            o.set_config('general:seafloor_action', 'none')
            o.set_config('drift:vertical_advection', False)
            o.set_config('drift:vertical_mixing', False)
            o.set_config('general:coastline_action', 'previous')
            
            
            # --- PART 3: 입자 시딩 ---
            for i, row in df_tumang.iterrows():
                o.seed_elements(
                    lon=row['lon'],
                    lat=row['lat'],
                    time=row['time_stamp'],
                    z=0.0,
                    origin_marker=np.array([i], dtype=np.int32)
                )
                
            print(f"[시딩] {len(df_tumang)}개 입자, 시작 시간: {start_time_sim}, 종료 시간: {end_time_sim}")


            # --- PART 4: 시뮬레이션 실행 ---
            o.run(
                time_step=600,      # 10분
                time_step_output=1800, # 30분
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

            result_csv_path = RESULT_CSV_PATH

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

            print("✅ CSV 저장 완료: oi_prediction_result.csv")


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
            plt.legend(loc='upper right')
            plt.tight_layout()

            # === 시각화 결과 자동 저장 ===
            plot_output_dir = rf"C:\Users\HUFS\Desktop\opendrift_middle\oi_{w_hycom}_{w_khoa}_시각화결과"
            os.makedirs(plot_output_dir, exist_ok=True)
            plot_filename = f"{input_basename}.png"
            plot_path = os.path.join(plot_output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"✅ 시각화 결과 저장 완료: {plot_path}")


        except Exception as e:
            print(f"❌ 오류 발생: {input_filename} - {e}")
            with open(ERROR_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([input_filename, type(e).__name__, str(e)])
            continue