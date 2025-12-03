import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize


# T01_DDJ024EDJ_2022-03-10 000000-000_buoy.nc
# T01_DDJ024EDJ_2022-03-10 000000-000_hf.nc
# T01_DDJ024EDJ_2022-03-10 000000-000_hycom.nc
# T01_DDJ024EDJ_2022-03-10 000000-000_khoa.nc


# # 부이
# buoy_path = r'C:\Users\USER\Desktop\ocean_data_develop\final_ocean_current_file\json_file'

# # 관측소
# hfrather_path = r'C:\Users\USER\Desktop\ocean_data_develop\final_ocean_current_file\json_file'

# # hycom
# hfrather_path = r'C:\Users\USER\Desktop\ocean_data_develop\final_ocean_current_file\json_file'

# # khoa
# hfrather_path = r'C:\Users\USER\Desktop\ocean_data_develop\final_ocean_current_file\json_file'


import os
import xarray as xr

# 1. 각 데이터 파일이 위치한 폴더 경로 정의
paths = {
    'buoy': r"D:\output_files\buoy_data",
    'hfradar': r"D:\output_files\hfradar_data",
    'hycom': r"D:\output_files\fix_range_hycom_data",
    'khoa': r"D:\output_files\KHOA_nc_data"
}

# 🗂️ [수정됨] 최종 결과물을 저장할 기본 폴더 경로

BASE_OUTPUT_FOLDER_PATH = r'D:\output_files\merged_file'


# 🗂️ [추가됨] 각 API 데이터를 저장할 하위 폴더 경로 설정
khoa_hycom_vector_folder_path = os.path.join(BASE_OUTPUT_FOLDER_PATH, 'khoa_hycom_vector')
OI_FOLDER_PATH = os.path.join(BASE_OUTPUT_FOLDER_PATH, 'OI_data')

os.makedirs(khoa_hycom_vector_folder_path, exist_ok=True)
os.makedirs(OI_FOLDER_PATH, exist_ok=True)




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
                loaded_datasets[data_type] = xr.open_dataset(full_path)
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
if __name__ == "__main__":
    # 1. HYCOM 데이터가 있는 폴더 경로
    hycom_folder_path = paths['hycom']

    # 2. '_hycom.nc'로 끝나는 모든 파일 목록 가져오기
    try:
        hycom_files = [f for f in os.listdir(hycom_folder_path) if f.endswith('_hycom.nc')]
        if not hycom_files:
            print(f"경고: '{hycom_folder_path}' 폴더에 HYCOM 파일이 없습니다.")
    except FileNotFoundError:
        print(f"오류: '{hycom_folder_path}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        hycom_files = []

    # 3. 모든 관련 파일이 존재하는 것만 remaining_files에
    remaining_files = []
    for f in hycom_files:
        base_name = f.replace('_hycom.nc', '')

        # 각 데이터 파일 경로
        expected_files = {
            'hycom': os.path.join(paths['hycom'], f),
            'buoy': os.path.join(paths['buoy'], f"{base_name}_buoy.nc"),
            'hfradar': os.path.join(paths['hfradar'], f"{base_name}_hfradar.nc"),
            'khoa': os.path.join(paths['khoa'], f"{base_name}_uv.nc"),
        }

        # 모든 파일 존재 여부 확인 (khoa_hycom_vector는 아직 생성되지 않은 파일 기준)
        all_exist = all(os.path.exists(expected_files[key]) for key in ['hycom', 'buoy', 'hfradar', 'khoa'])

    print(f"✅ 처리 대상 파일 수: {len(remaining_files)}")
    for f in remaining_files:
        print(f"   - {f}")
        
    # 4. 남은 파일에 대해 모든 데이터셋 로드
    for hycom_filename in remaining_files:
        print(f"\n🔄 '{hycom_filename}' 관련 데이터 로딩 시작...")
        ocean_data = load_ocean_data_set(hycom_filename, paths)
        
        # 로드된 결과 확인
        for key, ds in ocean_data.items():
            if ds is not None:
                print(f"   - {key} 데이터셋 로드 완료: {ds}")
            else:
                print(f"   - {key} 데이터셋 없음 또는 로드 실패")
        
        base_with_suffix = os.path.splitext(hycom_filename)[0]
        
        # 2. 기본 이름에서 '_hycom' 부분을 제거하여 순수 base_name을 얻습니다.
        # 예: 'T01_..._hycom' -> 'T01_...'
        base_name = base_with_suffix.replace('_hycom', '')
        
        
        hycom_khoa_vector_output_filename = os.path.join(khoa_hycom_vector_folder_path, f"{base_name}_hycom_khoa_vector.nc")
        
        OI_output_filename = os.path.join(OI_FOLDER_PATH, f"{base_name}_OI_data.nc")
        
        

        # ------------------------------------------------------------------ #
        #                                                                    #


        # ---------------------------
        # 1. 데이터 불러오기
        # ---------------------------


        ds_hycom = ocean_data.get('hycom')
        ds_khoa  = ocean_data.get('khoa')

        # ---------------------------
        # 2. 시간과 좌표 맞추기
        # ---------------------------
        lat_khoa = ds_khoa['lat']
        lon_khoa = ds_khoa['lon']
        time_khoa = ds_khoa['time']

        # HYCOM 표층만 선택
        u_hycom = ds_hycom['x_sea_water_velocity'].isel(depth=0)
        v_hycom = ds_hycom['y_sea_water_velocity'].isel(depth=0)

        u_hycom = u_hycom.sel(time=~u_hycom.get_index("time").duplicated())
        v_hycom = v_hycom.sel(time=~v_hycom.get_index("time").duplicated())

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
        BUOY_NC_FILE = ocean_data.get('buoy')
        # 3. HF-Radar 관측 데이터 (이전에 생성한 파일)
        RADAR_NC_FILE = ocean_data.get('HFradar')


        # --- 자료 동화 주요 파라미터 ---
        # 영향 반경 (단위: km): 하나의 관측값이 주변 몇 km까지 영향을 미칠지 결정하는 가장 중요한 변수
        # (실험을 통해 적절한 값을 찾아야 함, 보통 10~50km 사이에서 시작)
        INFLUENCE_RADIUS_KM = 50.0

        # ===================================================================
        # 2. 데이터 준비 및 전처리
        # ===================================================================

        print("🔄 1. 데이터 로딩 및 전처리를 시작합니다...")

        # --- 데이터 로딩 ---
        ds_bg = xr.open_dataset(BACKGROUND_NC_FILE, engine="netcdf4")
        ds_buoy = xr.open_dataset(BUOY_NC_FILE, engine="netcdf4")
        ds_radar = xr.open_dataset(RADAR_NC_FILE, engine="netcdf4")

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
        # 4. 결과 저장 (OpenDrift 호환 형식)
        # ===================================================================
        print(f"💾 3. 동화된 결과를 OpenDrift 호환 NetCDF 파일로 저장합니다: {OI_output_filename}")

        # OpenDrift가 인식할 수 있는 새로운 xarray Dataset 생성
        # 변수 이름, 좌표, 메타데이터를 지정된 형식에 맞게 구성합니다.
        ds_opendrift = xr.Dataset(
            {
                # 변수 이름을 OpenDrift 표준(e.g., eastward_sea_water_velocity)에 맞춤
                "eastward_sea_water_velocity": u_an,
                "northward_sea_water_velocity": v_an,
            },
            # 좌표는 원본 배경장(ds_bg) 데이터셋에서 그대로 가져옴
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
        ds_opendrift.to_netcdf(OI_output_filename)

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
        
        
        
 