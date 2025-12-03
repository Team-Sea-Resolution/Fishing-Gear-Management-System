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

# ================================================================================================================================
# ================================================================================================================================
# ================================================================================================================================


SERVICE_KEY = "ANM8LV6zTsRNiGg6FCUMpw=="
# 데이터 조회 기간 설정
START_DATE = datetime(2021, 11, 1, 0, 0)
END_DATE = datetime(2021, 11, 2, 23, 0)

# 2️⃣ 찾고 싶은 위경도 범위 설정 (여기만 원하는 값으로 수정!)
MIN_X = 123.9609466
MAX_X = 129.22982499999998
MIN_Y = 31.82632165
MAX_Y = 35.986358300000006



# ================================================================================================================================
# ================================================================================================================================
# ================================================================================================================================

########################################################
# hycom, khoa 벡터데이터 합치기
########################################################


# ---------------------------
# 1. 데이터 불러오기
# ---------------------------
hycom_file = r"C:\Users\USER\Desktop\ocean_data_develop\ncfile\hycom_data.nc4" # hycom data 경로
khoa_file  = r"C:\Users\USER\Desktop\ocean_data_develop\ncfile\khoa_nc_data_uv.nc" # khoa data 경로


ds_hycom = xr.open_dataset(hycom_file)
ds_khoa  = xr.open_dataset(khoa_file)

# ---------------------------
# 2. 시간과 좌표 맞추기
# ---------------------------
lat_khoa = ds_khoa['lat']
lon_khoa = ds_khoa['lon']
time_khoa = ds_khoa['time']

# HYCOM 표층만 선택
u_hycom = ds_hycom['water_u'].isel(depth=0)
v_hycom = ds_hycom['water_v'].isel(depth=0)

# 시간 + 좌표 보간
u_hycom_interp = u_hycom.interp(time=time_khoa, lat=lat_khoa, lon=lon_khoa)
v_hycom_interp = v_hycom.interp(time=time_khoa, lat=lat_khoa, lon=lon_khoa)

# KHOA 속도
u_khoa = ds_khoa['eastward_sea_water_velocity']
v_khoa = ds_khoa['northward_sea_water_velocity']

# ---------------------------
# 3. 벡터 합성
# ---------------------------
w_hycom = 0.5
w_khoa  = 0.5

u_combined = w_hycom * u_hycom_interp + w_khoa * u_khoa
v_combined = w_hycom * v_hycom_interp + w_khoa * v_khoa
speed_combined = np.sqrt(u_combined**2 + v_combined**2)

# ---------------------------
# 4. 인터랙티브 플롯
# ---------------------------
fig, ax = plt.subplots(figsize=(10,8))
plt.subplots_adjust(bottom=0.2)

# 컬러맵 범위 고정
vmin = float(speed_combined.min())
vmax = float(speed_combined.max())
norm = Normalize(vmin=vmin, vmax=vmax)

# 초기 프레임
frame0 = 0
mesh = ax.pcolormesh(lon_khoa, lat_khoa, speed_combined.isel(time=frame0),
                     cmap='viridis', norm=norm)
q = ax.quiver(lon_khoa[::5], lat_khoa[::5],
              u_combined.isel(time=frame0)[::5, ::5],
              v_combined.isel(time=frame0)[::5, ::5],
              scale=5, color='k')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'Combined Currents at {str(time_khoa[frame0].values)}')
cbar = fig.colorbar(mesh, ax=ax, label='Current speed (m/s)')

# ---------------------------
# 5. 시간 슬라이더 추가
# ---------------------------
ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(time_khoa)-1, valinit=frame0, valstep=1)

def update(val):
    frame = int(slider.val)
    mesh.set_array(speed_combined.isel(time=frame).values.ravel())
    q.set_UVC(u_combined.isel(time=frame)[::5, ::5].values,
              v_combined.isel(time=frame)[::5, ::5].values)
    ax.set_title(f'Combined Currents at {str(time_khoa[frame].values)}')
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()


# ====================================================================================================
import xarray as xr
import numpy as np

# ---------------------------
# 1. 합성 데이터 준비
# ---------------------------
# u_combined, v_combined, speed_combined
# lat_khoa, lon_khoa, time_khoa 이미 존재

# DataArray → ndarray로 변환
u_data = u_combined.data
v_data = v_combined.data
speed_data = speed_combined.data

# 새로운 xarray.Dataset 생성
ds_combined = xr.Dataset(
    {
        "eastward_velocity": (["time", "lat", "lon"], u_data),
        "northward_velocity": (["time", "lat", "lon"], v_data),
        "speed": (["time", "lat", "lon"], speed_data)
    },
    coords={
        "time": time_khoa.data,
        "lat": lat_khoa.data,
        "lon": lon_khoa.data
    },
    attrs={
        "title": "Combined HYCOM + KHOA currents",
        "source": "HYCOM (0.5) + KHOA (0.5)",
        "description": "Eastward/northward velocities and speed",
        "Conventions": "CF-1.6"
    }
)

# ---------------------------
# 2. NetCDF 파일 저장
# ---------------------------
output_file = "combined_currents.nc"
ds_combined.to_netcdf(output_file)

print(f"✅ Combined NetCDF file saved as: {output_file}")


# ================================================================================================================================
# ================================================================================================================================
# ================================================================================================================================
# ================================================================================================================================


########################################################
# 해상부이, 관측소 api 끌고오기
########################################################

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
BASE_URL = "http://www.khoa.go.kr/api/oceangrid/tidalBu/search.do"


# 최종 저장될 NetCDF 파일 이름
khoa_buoy_OUTPUT_FILENAME = "khoa_buoy_data_20211101-20211102.nc"


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
                response = requests.get(BASE_URL, params=params, timeout=10)
                
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
BASE_URL = "http://www.khoa.go.kr/api/oceangrid/tidalHfRadar/search.do"

# 데이터 조회 기간 설정 (이전과 동일)
start_date = datetime(2021, 11, 1, 0, 0)
end_date = datetime(2021, 11, 2, 23, 0)
time_list = pd.date_range(start=start_date, end=end_date, freq='H')

# 최종 저장될 NetCDF 파일 이름
KHOA_HFRadar_output_path = f"KHOA_HFRadar_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.nc"


# ==================================
# 2. 데이터 수집 및 처리
# ==================================
if os.path.exists(KHOA_HFRadar_output_path):
    print(f"🔄 이미 NetCDF 파일이 존재하여 다운로드를 생략합니다: {KHOA_HFRadar_output_path}")
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
                response = requests.get(BASE_URL, params=params)
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
    else:
        # ==================================
        # 3. NetCDF 생성
        # ==================================
        print("\n📊 NetCDF 파일을 생성합니다...")
        
        df = pd.DataFrame(all_records)
        # 중복 데이터 제거 (안정성 확보)
        df.drop_duplicates(subset=['time', 'lat', 'lon'], inplace=True, keep='first')
        
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
        ds.to_netcdf(KHOA_HFRadar_output_path)
        print(f"✅ NetCDF 생성 완료: {KHOA_HFRadar_output_path}")
        
        
        
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
BACKGROUND_NC_FILE = r'.combined_currents.nc' 
# 2. 해상 부이 관측 데이터 (이전에 생성한 파일)
BUOY_NC_FILE = khoa_buoy_OUTPUT_FILENAME
# 3. HF-Radar 관측 데이터 (이전에 생성한 파일)
RADAR_NC_FILE = KHOA_HFRadar_output_path

# --- 출력 파일 경로 ---
OUTPUT_NC_FILE = 'assimilated_ocean_current.nc'

# --- 자료 동화 주요 파라미터 ---
# 영향 반경 (단위: km): 하나의 관측값이 주변 몇 km까지 영향을 미칠지 결정하는 가장 중요한 변수
# (실험을 통해 적절한 값을 찾아야 함, 보통 10~50km 사이에서 시작)
INFLUENCE_RADIUS_KM = 20.0

# ===================================================================
# 2. 데이터 준비 및 전처리
# ===================================================================

print("🔄 1. 데이터 로딩 및 전처리를 시작합니다...")

# --- 데이터 로딩 ---
ds_bg = xr.open_dataset(BACKGROUND_NC_FILE)
ds_buoy = xr.open_dataset(BUOY_NC_FILE)
ds_radar = xr.open_dataset(RADAR_NC_FILE)

# --- 관측 데이터 통합 ---
# xarray Dataset을 pandas DataFrame으로 변환 후 하나로 합치기
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

# 💡 해결: try-except 구문으로 시간대 정보 존재 여부를 더 안정적으로 확인
try:
    # ds_bg의 시간대 정보를 확인하려는 시도
    target_tz = ds_bg.time.dt.tz
    print(f"배경 데이터의 시간대({target_tz})에 맞춰 관측 데이터의 시간대를 통일합니다.")
    # 성공하면 (시간대 정보가 있으면) df_obs의 시간대를 통일
    if df_obs['time'].dt.tz is None:
        df_obs['time'] = df_obs['time'].dt.tz_localize('UTC').dt.tz_convert(target_tz)
    else:
        df_obs['time'] = df_obs['time'].dt.tz_convert(target_tz)
        
except AttributeError:
    # AttributeError가 발생하면 ds_bg에 시간대 정보가 없는 것이므로
    print("배경 데이터에 시간대 정보가 없어 관측 데이터의 시간대 정보도 제거합니다.")
    df_obs['time'] = df_obs['time'].dt.tz_localize(None)


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

# --- 공간 검색을 위한 KD-Tree 생성 ---
# 특정 위치에서 가까운 격자점을 빠르게 찾기 위함
kdtree = cKDTree(grid_points)

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
# 4. 결과 저장
# ===================================================================
print(f"💾 3. 동화된 결과를 NetCDF 파일로 저장합니다: {OUTPUT_NC_FILE}")

# 원본 데이터셋에 분석장을 새로운 변수로 추가
ds_assimilated = ds_bg.copy()
ds_assimilated['u_assimilated'] = u_an
ds_assimilated['v_assimilated'] = v_an
ds_assimilated['u_assimilated'].attrs = {'long_name': 'Assimilated eastward sea water velocity', 'units': 'm/s'}
ds_assimilated['v_assimilated'].attrs = {'long_name': 'Assimilated northward sea water velocity', 'units': 'm/s'}

# 파일로 저장
ds_assimilated.to_netcdf(OUTPUT_NC_FILE)

print("✅ 모든 작업이 완료되었습니다!")

        
        
        