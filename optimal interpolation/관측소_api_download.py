# 해야하는 작업
# 각 날짜에 맞게 벡터 합성(khoa+hycom)파일 이름 설정하게 자동화
# OI 적용 완료 한 파일또한 이름 자동 저장하게 자동화


import os
import requests
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
import pandas as pd
import json
from datetime import datetime

# ================================================================================================================================
# ================================================================================================================================
# ================================================================================================================================


SERVICE_KEY = "ANM8LV6zTsRNiGg6FCUMpw=="



# 2️⃣ 찾고 싶은 위경도 범위 설정 (여기만 원하는 값으로 수정!)
MIN_X = 123.9609466
MAX_X = 129.22982499999998
MIN_Y = 31.82632165
MAX_Y = 35.986358300000006



data1 = [
    ["TW_0088","감천항",35.052,129.003],
    ["TW_0077","경인항",37.523,126.592],
    ["TW_0089","경포대해수욕장",37.808,128.931],
    ["TW_0095","고래불해수욕장",36.58,129.454],
    ["TW_0074","광양항",34.859,127.792],
    ["TW_0072","군산항",35.984,126.508],
    ["TW_0091","낙산해수욕장",38.122,128.65],
    ["KG_0025","남해동부",34.222,128.419],
    ["TW_0069","대천해수욕장",36.274,126.457],
    ["KG_0024","대한해협",34.919,129.121],
    ["TW_0085","마산항",35.103,128.631],
    ["TW_0094","망상해수욕장",37.616,129.103],
    ["TW_0087","부산항",35.091,129.085],
    ["TW_0086","부산항신항",35.043,128.761],
    ["TW_0079","상왕등도",35.652,126.194],
    ["TW_0081","생일도",34.258,126.96],
    ["TW_0093","속초해수욕장",38.198,128.631],
    ["TW_0090","송정해수욕장",35.164,129.219],
    ["TW_0083","여수항",34.794,127.808],
    ["TW_0078","완도항",34.325,126.763],
    ["TW_0080","우이도",34.543,125.802],
    ["KG_0101","울릉도북동",38.007,131.552],
    ["KG_0102","울릉도북서",37.742,130.601],
    ["TW_0076","인천항",37.389,126.533],
    ["TW_0092","임랑해수욕장",35.302,129.292],
    ["KG_0021","제주남부",32.09,126.965],
    ["KG_0028","제주해협",33.7,126.59],
    ["TW_0075","중문해수욕장",33.234,126.409],
    ["TW_0082","태안항",37.006,126.27],
    ["TW_0084","통영항",34.773,128.46],
    ["TW_0062","해운대해수욕장",35.148,129.17]
]
data2 = [
    ["HF_0064","광양항",34.887,127.797],
    ["HF_0076","군산항",35.99,126.358],
    ["HF_0041","대한해협",34.909,129.2],
    ["HF_0073","동해남부",36.633,130.224],
    ["HF_0075","목포항내측",34.756,126.336],
    ["HF_0074","목포항외측",34.772,126.239],
    ["HF_0040","부산항신항",35.036,128.768],
    ["HF_0065","여수광양항",34.765,127.804],
    ["HF_0039","여수해만",34.656,127.964],
    ["HF_0063","울산항",35.4,129.607],
    ["HF_0069","인천항",37.355,126.508],
    ["HF_0070","태안대산",37.073,126.292],
    ["HF_0071","포항항",36.066,129.475]
]

# 예: r"D:\my_data\fix_range_hycon_data"
HYCOM_FOLDER_PATH = r"C:\Users\HUFS\Desktop\opendrift_middle\fix_range_hycom_data" 

# 처리할 JSON 파일들이 있는 폴더 경로
JSON_FOLDER_PATH = r"D:\어선행적데이터\Training\02.라벨링데이터\TL_01.자망.zip"

# 최종 결과물을 저장할 기본 폴더 경로
BASE_OUTPUT_FOLDER_PATH = r'D:\current_oi\BASE_OUTPUT_FOLDER'

# 각 API 데이터를 저장할 하위 폴더 경로 설정
BUOY_OUTPUT_FOLDER_PATH = os.path.join(BASE_OUTPUT_FOLDER_PATH, 'buoy_data')
HFRADAR_OUTPUT_FOLDER_PATH = os.path.join(BASE_OUTPUT_FOLDER_PATH, 'hfradar_data')



# 결과물 저장 폴더들이 없으면 모두 생성
os.makedirs(BUOY_OUTPUT_FOLDER_PATH, exist_ok=True)
os.makedirs(HFRADAR_OUTPUT_FOLDER_PATH, exist_ok=True)


# ==============================================================================
# 2. JSON 파일 시간 추출 함수 (실제 파일 구조에 맞게 수정 필요)
# ==============================================================================
def extract_time_from_json(json_file_path):
    """
    JSON 파일을 읽어 시작과 끝 시간 정보만 추출하는 함수.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # [수정됨] 보내주신 JSON 구조에 맞게 'crs' 객체에서 시간 정보 추출
            crs_data = data.get('crs', {})
            start_date_str = crs_data.get('start_time') # "2022-03-12 05:09:00"
            end_date_str = crs_data.get('end_time')     # "2022-03-13 05:08:00"
            
            if not all([start_date_str, end_date_str]):
                print(f"  - ❌ 오류: JSON 파일에서 'start_time' 또는 'end_time'을 찾을 수 없습니다.")
                return None, None

            # [수정됨] 날짜 문자열 형식에 맞는 포맷으로 datetime 객체 변환
            date_format = "%Y-%m-%d %H:%M:%S"
            start_date = datetime.strptime(start_date_str, date_format)
            end_date = datetime.strptime(end_date_str, date_format)
            
            return start_date, end_date

    except json.JSONDecodeError:
        print(f"  - ❌ 오류: '{os.path.basename(json_file_path)}'는 올바른 JSON 파일이 아닙니다.")
        return None, None
    except Exception as e:
        print(f"  - ❌ '{os.path.basename(json_file_path)}' 파일 처리 중 오류: {e}")
        return None, None




# ==============================================================================
# 🔁 3. 메인 처리 로직
# ==============================================================================

print(f"HYCOM 폴더({HYCOM_FOLDER_PATH})에서 파일 목록을 읽어옵니다...")
try:
    hycom_files = os.listdir(HYCOM_FOLDER_PATH)
    # 파일 이름이 '..._hycom'으로 끝난다고 가정하고, 그 앞부분(base_name)을 추출
    # 예: 'T01_..._hycom' -> 'T01_...'
    hycom_base_names = set(f.rsplit('_', 1)[0] for f in hycom_files if '_hycom' in f)
    
    if not hycom_base_names:
        print(f"⚠️ 경고: {HYCOM_FOLDER_PATH}에서 '_hycom'으로 끝나는 파일을 찾을 수 없습니다. 작업을 중단합니다.")
        exit()
        
    print(f"총 {len(hycom_base_names)}개의 HYCOM 기준 파일을 처리합니다.")

except FileNotFoundError:
    print(f"❌ 오류: HYCOM 폴더를 찾을 수 없습니다. {HYCOM_FOLDER_PATH}")
    print("스크립트 상단의 HYCOM_FOLDER_PATH 변수를 올바르게 설정해주세요.")
    exit()
except Exception as e:
    print(f"❌ 오류: HYCOM 폴더를 읽는 중 오류 발생: {e}")
    exit()


print(f"고정 위경도 범위: LON({MIN_X}~{MAX_X}), LAT({MIN_Y}~{MAX_Y})")


# [수정됨] HYCOM 파일 목록(hycom_base_names)을 기준으로 작업 반복
for base_name in hycom_base_names:
    
    # [수정됨] 기준이 되는 base_name으로부터 geojson 파일 이름과 전체 경로 생성
    geojson_filename = f"{base_name}.geojson"
    json_file_full_path = os.path.join(JSON_FOLDER_PATH, geojson_filename)
    
    print(f"\n{'='*80}\n▶️ 작업 시작: {geojson_filename}\n{'='*80}")

    # [신규 추가] 해당 geojson 파일이 JSON_FOLDER_PATH에 실제로 존재하는지 확인
    if not os.path.exists(json_file_full_path):
        print(f"  - ⚠️ 경고: {geojson_filename} 파일이 {JSON_FOLDER_PATH}에 없습니다. 건너뜁니다.")
        continue # 다음 파일로 넘어감

    # 1. JSON에서 시간 정보 추출
    START_DATE, END_DATE = extract_time_from_json(json_file_full_path)

    if START_DATE is None:
        print(f"⚠️ 시간 정보 추출 실패로 '{geojson_filename}' 파일을 건너뜁니다.")
        continue # 다음 파일로 넘어감

    print(f"  - 추출된 기간: {START_DATE} ~ {END_DATE}")
    
    base_json_name = os.path.splitext(geojson_filename)[0]
    khoa_buoy_OUTPUT_FILENAME = os.path.join(BUOY_OUTPUT_FOLDER_PATH, f"{base_json_name}_buoy.nc")
    khoa_hfradar_output_filename = os.path.join(HFRADAR_OUTPUT_FOLDER_PATH, f"{base_json_name}_hfradar.nc")


    ########################################################
    # 해상부이, 관측소 api 끌고오기
    ########################################################


    # 데이터 필터링 및 STATIONS 딕셔너리 생성
    STATIONS1 = {}
    for item in data1:
        # item: [ID, Name, Latitude (Y), Longitude (X)]
        station_id, name, lat, lon = item[0], item[1], item[2], item[3]

        # 경도 (lon)와 위도 (lat)가 범위 내에 있는지 확인
        is_in_x_range = MIN_X <= lon <= MAX_X
        is_in_y_range = MIN_Y <= lat <= MAX_Y
        
        if is_in_x_range and is_in_y_range:
            STATIONS1[station_id] = {
                'name': name,
                'lat': lat,
                'lon': lon
            }


    # 데이터 필터링 및 STATIONS 딕셔너리 생성
    STATIONS2 = {}
    for item in data2:
        # item: [ID, Name, Latitude (Y), Longitude (X)]
        station_id, name, lat, lon = item[0], item[1], item[2], item[3]

        # 경도 (lon)와 위도 (lat)가 범위 내에 있는지 확인
        is_in_x_range = MIN_X <= lon <= MAX_X
        is_in_y_range = MIN_Y <= lat <= MAX_Y
        
        if is_in_x_range and is_in_y_range:
            STATIONS2[station_id] = {
                'name': name,
                'lat': lat,
                'lon': lon
            }


    from datetime import datetime, timedelta

    # ==============================================================================
    # 설정 부분 // 위경도에 맞게 범위 내에 있는 관측소와 해상부이 api 끌고오기
    # ==============================================================================


    # API 기본 URL
    BUOY_BASE_URL  = "http://www.khoa.go.kr/api/oceangrid/tidalBu/search.do"





    # ==============================================================================
    # 해상부위 로직
    # ==============================================================================

    if os.path.exists(khoa_buoy_OUTPUT_FILENAME):
        print(f"🔄 이미 NetCDF 파일이 존재하여 다운로드를 생략합니다: {khoa_buoy_OUTPUT_FILENAME}")
    else:
        all_records = []
        
        # API는 하루 단위로 데이터를 제공하므로, 날짜 목록 생성
        date_list = pd.date_range(start=START_DATE.date(), end=END_DATE.date()).tolist()

        # ====== API 호출 및 데이터 수집 ======
        for station_id, station_info in STATIONS1.items():
            for search_date in date_list:
                print(f"📡 {station_info['name']}({station_id}) 관측소의 {search_date.strftime('%Y-%m-%d')} 데이터 요청 중...")
                
                params = {
                    'ServiceKey': SERVICE_KEY,
                    'ObsCode': station_id,
                    'Date': search_date.strftime('%Y%m%d'),
                    'ResultType': 'json'
                }
                
                try:
                    response = requests.get(BUOY_BASE_URL , params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # 데이터가 있는지 확인
                        if 'result' in data and 'data' in data['result']:
                            for record in data['result']['data']:
                                all_records.append({
                                    'station_id': station_id,
                                    'station_name': station_info['name'],
                                    'lat': station_info['lat'],
                                    'lon': station_info['lon'],
                                    'time': pd.to_datetime(record['obs_time']),
                                    'current_speed_cm_s': pd.to_numeric(record['current_speed'], errors='coerce'),
                                    'current_direction_deg': pd.to_numeric(record['current_direct'], errors='coerce')
                                })
                        else:
                            print(f"   ⚠️ 데이터 없음: {station_info['name']} ({search_date.strftime('%Y-%m-%d')})")

                    else:
                        print(f"   ❌ API 호출 실패: {station_info['name']}, Status Code: {response.status_code}")

                except requests.exceptions.RequestException as e:
                    print(f"   ❌ 네트워크 오류 발생: {station_info['name']}, {e}")
                except Exception as e:
                    print(f"   ❌ 처리 중 예외 발생: {station_info['name']}, {e}")
        
        if not all_records:
            print("😭 수집된 데이터가 없어 NetCDF 파일을 생성할 수 없습니다.")
            continue  # 🚨 다음 JSON 파일로 바로 넘어가기
        else:
            # ====== 데이터프레임 변환 및 필터링 ======
            df = pd.DataFrame(all_records)
            df.dropna(inplace=True) # 결측값이 있는 행 제거

            # 시간 범위 및 정각 데이터 필터링
            df_hourly = df[
                (df['time'] >= START_DATE) & 
                (df['time'] <= END_DATE) & 
                (df['time'].dt.minute == 0)
            ].copy()

            print(f"\n📊 총 {len(df_hourly)}개의 시간별 관측 데이터 처리 완료.")

            # ====== NetCDF 생성 ======
            # xarray가 인식하기 좋게 station_id를 인덱스로 설정
            df_hourly.drop_duplicates(subset=['station_id', 'time'], inplace=True, keep='first')

            # 문자열 변수(station_name)를 바이트형으로 변환 (NetCDF 저장 가능하도록)
            df_hourly["station_name"] = df_hourly["station_name"].astype(str).apply(lambda x: x.encode("utf-8"))

            # 이제 중복이 제거되었으므로 에러가 발생하지 않습니다.
            df_hourly.set_index(['station_id', 'time'], inplace=True)
            ds = df_hourly.to_xarray()
            
            # 각 변수에 대한 속성(설명, 단위) 추가
            ds['station_name'].attrs = {'long_name': 'Observation station name'}
            ds['lat'].attrs = {'long_name': 'Latitude', 'units': 'degrees_north'}
            ds['lon'].attrs = {'long_name': 'Longitude', 'units': 'degrees_east'}
            ds['current_speed_cm_s'].attrs = {'long_name': 'Sea water speed', 'units': 'cm/s'}
            ds['current_direction_deg'].attrs = {'long_name': 'Sea water direction (from north)', 'units': 'degree'}

            # 파일 전체에 대한 전역 속성 추가
            ds.attrs = {
                'title': 'KHOA Oceanographic Buoy Data',
                'source': 'Korea Hydrographic and Oceanographic Agency (KHOA)',
                'api': 'tidalBu (해양관측부이)',
                'description': 'Hourly current speed and direction data from KHOA buoys.',
                'history': f'Created on {datetime.now().isoformat()}'
            }
            
            # NetCDF 파일로 저장
            ds.to_netcdf(khoa_buoy_OUTPUT_FILENAME)
            print(f"\n✅ NetCDF 파일 생성 완료: {khoa_buoy_OUTPUT_FILENAME}")
            
            
    # ==============================================================================
    # 해상 관측소 로직
    # ==============================================================================        
            
    # HF-RADAR API 기본 정보
    HFRADAR_BASE_URL = "http://www.khoa.go.kr/api/oceangrid/tidalHfRadar/search.do"

    START_DATE = START_DATE.replace(minute=0, second=0, microsecond=0)
    END_DATE = END_DATE.replace(minute=0, second=0, microsecond=0)
    # 데이터 조회 기간 설정 (이전과 동일)
    time_list = pd.date_range(start=START_DATE, end=END_DATE, freq='H')


    # ==================================
    # 2. 데이터 수집 및 처리
    # ==================================
    if os.path.exists(khoa_hfradar_output_filename):
        print(f"🔄 이미 NetCDF 파일이 존재하여 다운로드를 생략합니다: {khoa_hfradar_output_filename}")
    else:
        all_records = []

        print("🚀 HF-RADAR API 데이터 수집을 시작합니다...")
        # 모든 관측소와 시간에 대해 API 호출
        for obs_code, station_name in STATIONS2.items():
            for t in time_list:
                # Date 파라미터 형식이 YYYYMMDDHH 입니다.
                date_str = t.strftime("%Y%m%d%H")
                
                params = {
                    'ServiceKey': SERVICE_KEY,
                    'ObsCode': obs_code,
                    'Date': date_str,
                    'ResultType': 'json'
                }
                
                try:
                    response = requests.get(HFRADAR_BASE_URL, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        # 'data' 키가 있는지, 비어있지 않은지 확인
                        if 'result' in data and 'data' in data['result'] and data['result']['data']:
                            # 한 번의 호출로 여러 위치의 데이터가 들어옴
                            for record in data['result']['data']:
                                all_records.append({
                                    'time': t,
                                    'station_id': obs_code,
                                    'station_name': station_name,
                                    'lat': float(record.get('lat', float('nan'))),
                                    'lon': float(record.get('lon', float('nan'))),
                                    'current_speed': float(record.get('current_speed', float('nan'))), # cm/s
                                    'current_direct': float(record.get('current_direct', float('nan'))) # deg
                                })
                            print(f"✅ [{station_name}({obs_code})] {t.strftime('%Y-%m-%d %H:%M')} 데이터 수집 성공")
                        else:
                            print(f"⚠️ [{station_name}({obs_code})] {t.strftime('%Y-%m-%d %H:%M')} 데이터가 없습니다.")
                    else:
                        print(f"❌ [{station_name}({obs_code})] {t.strftime('%Y-%m-%d %H:%M')} API 호출 실패 (상태 코드: {response.status_code})")
                except requests.exceptions.RequestException as e:
                    print(f"❌ [{station_name}({obs_code})] {t.strftime('%Y-%m-%d %H:%M')} 요청 중 예외 발생: {e}")
                except ValueError: # JSON 디코딩 오류 처리
                    print(f"❌ [{station_name}({obs_code})] {t.strftime('%Y-%m-%d %H:%M')} JSON 파싱 실패. 응답 내용: {response.text}")


        if not all_records:
            print("😭 수집된 데이터가 없어 NetCDF 파일을 생성할 수 없습니다.")
            continue  # 🚨 다음 JSON 파일로 넘어가기
        else:
            # ==================================
            # 3. NetCDF 생성
            # ==================================
            print("\n📊 NetCDF 파일을 생성합니다...")
            
            df = pd.DataFrame(all_records)
            # 중복 데이터 제거 (안정성 확보)
            df.drop_duplicates(subset=['time', 'lat', 'lon'], inplace=True, keep='first')
            

                # ✅ 문자열을 바이트형으로 변환 (NetCDF 저장 가능하도록)
            df["station_name"] = df["station_name"].astype(str).apply(lambda x: x.encode("utf-8"))
            # 데이터프레임을 xarray Dataset으로 변환
            # HF-Radar 데이터는 각 시간이 공간 격자를 가지므로, 단순 리스트 형태로 저장
            ds = df.set_index('time').to_xarray()
            
            # 변수 속성(metadata) 추가
            ds['current_speed'].attrs = {'long_name': 'Sea Water Speed', 'units': 'cm/s'}
            ds['current_direct'].attrs = {'long_name': 'Sea Water To Direction', 'units': 'degree'}
            ds['lat'].attrs = {'units': 'degrees_north'}
            ds['lon'].attrs = {'units': 'degrees_east'}
            ds['station_id'].attrs = {'long_name': 'Observation station code'}
            
            # 전역 속성(Global attributes) 추가
            ds.attrs = {
                'title': 'KHOA HF-RADAR Ocean Current Data',
                'source': 'Korea Hydrographic and Oceanographic Agency (KHOA)',
                'api_url': 'http://www.khoa.go.kr/api/oceangrid/tidalHfRadar/search.do',
                'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            }

            # NetCDF 파일로 저장
            ds.to_netcdf(khoa_hfradar_output_filename)
            print(f"✅ NetCDF 생성 완료: {khoa_hfradar_output_filename}")
            
            
        print(f"\n🎉 {geojson_filename}에 대한 모든 작업 완료!")

print(f"\n{'='*80}\n✅ 모든 GeoJSON 파일 처리가 완료되었습니다.\n{'='*80}")
        