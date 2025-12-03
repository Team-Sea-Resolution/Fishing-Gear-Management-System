import os
import json
import requests
import pandas as pd
import xarray as xr
from datetime import timedelta
from urllib.parse import urlencode

# ======================================================================
# 🗂 폴더 경로 설정
# ======================================================================
HYCOM_FOLDER_PATH = r'D:\current_oi\khoa_down\KHOA_nc_data'
JSON_FOLDER_PATH = r"D:\어선행적데이터\Training\02.라벨링데이터\TL_01.자망.zip"
HYCOM_OUTPUT_DIR = r"C:\Users\HUFS\Desktop\opendrift_middle\fix_range_hycom_data"
os.makedirs(HYCOM_OUTPUT_DIR, exist_ok=True)

# ======================================================================
# 🔧 전역 설정
# ======================================================================
MAX_HYCOM_HOURS = 8
HYCOM_BASE_URL = "https://ncss.hycom.org/thredds/ncss/grid/GLBy0.08/expt_93.0/uv3z"
HYCOM_VARS = ["water_u", "water_v"]
HYCOM_COMMON_PARAMS = {
    "var": HYCOM_VARS,
    "north": 35.9863583, "west": 123.9609466, "east": 129.2298249, "south": 31.82632165,
    "timeStride": 1, "vertStride": 0, "accept": "netcdf4"
}
SUPPORTED_YEARS = range(2018, 2025)

# ======================================================================
# ⏰ 시간 분할 함수
# ======================================================================
def get_time_steps(start_dt, end_dt, max_hours=8):
    time_steps = []
    current_start = start_dt
    while current_start < end_dt:
        current_end = min(current_start + timedelta(hours=max_hours), end_dt)
        time_steps.append((current_start, current_end))
        current_start = current_end + timedelta(seconds=3600)
    return time_steps

# ======================================================================
# 📄 GeoJSON → DataFrame
# ======================================================================
def load_geojson_to_dataframe(path):
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

# ======================================================================
# 🔁 HYCOM 폴더 기준 파일 탐색 및 매칭
# ======================================================================
try:
    hycom_files = os.listdir(HYCOM_FOLDER_PATH)
    hycom_base_names = set(f.rsplit('_', 1)[0] for f in hycom_files if '_uv' in f)

    if not hycom_base_names:
        print(f"⚠️ {HYCOM_FOLDER_PATH}에서 '_hycom' 파일을 찾을 수 없습니다.")
        exit()

    print(f"총 {len(hycom_base_names)}개의 HYCOM 기준 파일 처리 예정")

except Exception as e:
    print(f"❌ 오류: {e}")
    exit()

# ======================================================================
# 🔁 전체 자동 처리 루프
# ======================================================================
for base_name in hycom_base_names:
    geojson_filename = f"{base_name}.geojson"
    geojson_path = os.path.join(JSON_FOLDER_PATH, geojson_filename)

    if not os.path.exists(geojson_path):
        print(f"⚠️ {geojson_filename}이 {JSON_FOLDER_PATH}에 없음 → 건너뜀")
        continue

    print(f"\n================== 처리 시작: {geojson_filename} ==================")

    try:
        df = load_geojson_to_dataframe(geojson_path)
        if df.empty:
            print("⚠️ GeoJSON에 데이터가 없음 → 건너뜀")
            continue

        # 시간 범위 추출
        sim_start = df['time_stamp'].min().floor('h')
        sim_end   = df['time_stamp'].max().ceil('h')
        data_year = sim_start.year

        if data_year not in SUPPORTED_YEARS:
            print(f"⚠️ {data_year}년 데이터는 HYCOM 지원 범위 아님 → 건너뜀")
            continue

        hycom_final_filename = f"{base_name}_hycom.nc"
        hycom_final_filepath = os.path.join(HYCOM_OUTPUT_DIR, hycom_final_filename)

        if os.path.exists(hycom_final_filepath):
            print(f"🔄 이미 HYCOM NetCDF 존재, 다운로드 생략: {hycom_final_filepath}")
            continue

        print(f"🕒 요청 시간: {sim_start} ~ {sim_end}")

        # 분할 다운로드
        time_steps = get_time_steps(sim_start, sim_end, MAX_HYCOM_HOURS)
        temp_downloaded_files = []

        for i, (start_dt, end_dt) in enumerate(time_steps):
            current_start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            current_end_str   = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            temp_filepath = os.path.join(HYCOM_OUTPUT_DIR, f"temp_{base_name}_part{i:03d}.nc")
            temp_downloaded_files.append(temp_filepath)

            print(f"  > 다운로드 ({i+1}/{len(time_steps)}): {current_start_str} ~ {current_end_str}")

            params = HYCOM_COMMON_PARAMS.copy()
            params["time_start"] = current_start_str
            params["time_end"] = current_end_str

            request_url = f"{HYCOM_BASE_URL}/{data_year}?{urlencode(params, doseq=True)}"
            response = requests.get(request_url, stream=True, timeout=300)
            response.raise_for_status()
            with open(temp_filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        # 다운로드된 파일 병합
        with xr.open_mfdataset(temp_downloaded_files, combine='by_coords') as ds_merged:
            ds_final = ds_merged.rename({
                'water_u': 'x_sea_water_velocity',
                'water_v': 'y_sea_water_velocity'
            })[['x_sea_water_velocity', 'y_sea_water_velocity']]
            ds_final.to_netcdf(hycom_final_filepath)
            print(f"✅ 최종 HYCOM 파일 저장 완료: {hycom_final_filepath}")

        # 임시 분할 파일 유지
        print("📂 임시 분할 파일 보존됨:")
        for fpath in temp_downloaded_files:
            print(f"   └ {os.path.basename(fpath)}")

    except Exception as e:
        print(f"❌ 오류 발생: {geojson_filename} - {type(e).__name__}: {e}")

    print("========================================================\n")
