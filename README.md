# 🐟 Smart Lost Fishing Gear Reporting and Retrieval System

Fishing Gear Management System is an integrated management platform designed to digitally track the deployment, movement, and retrieval status of fishing gear, aiming to reduce illegal, abandoned, and lost gear.​<br>
Based on vessel, fishing ground, and location data, it traces gear usage history and the generation of discarded gear, providing data to support marine debris reduction and resource management policy-making.<br>
It also supports the shredding and recycling linkage of collected waste nets, contributing to marine environmental protection and the realization of a circular economy.

## Core Code
<pre>
/user: 로그인 페이지와 회원가입 페이지
/lists: 구매자 DB와 신고자 DB
/maps: 유실어구 시뮬레이션 실행 페이지 및 어구 리스트 
/maps/legend: 유실어구 시뮬레이션 모델로 maps app과 연결
/rds: 임베디드 시스템에서 얻어진 GPS 데이터베이스
/schedule: 수거선 배정 페이지
/sediment: 침적 시뮬레이션 실행 페이지 및 어구 리스트/ 침적 시뮬레이션을 통한 쓰레기 밀집구역 인사이트 코드
/optimal interpolation: 자료동화기법 최적 내삽법(OI)를 통한 유실 시뮬레이션에 사용하는 해수유동데이터 제작
</pre>
---

## 1. 🎣 Fishermen report lost fishing gear with one click

Fishermen press a button on-site when they lose fishing gear during operations.

<img src="https://github.com/user-attachments/assets/965191db-2b0e-49f2-bf26-96b4e9abdd69" width="500" />

---

## 2. 📍 The reported location is automatically recorded on the web page

A web dashboard displays the gear's loss location in real-time.

<img src="https://github.com/user-attachments/assets/8577c22d-bd68-4198-937a-3d2465da6693" width="500" />

---

## 3. 🌀 Loss simulation predicts gear movement

Administrators can run a **drift simulation** from the reported location to estimate the future position of the gear.

<img src="https://github.com/user-attachments/assets/9b9f8f9d-bb42-43d0-a64f-1a41cafd39f2" width="500" />

---

## 4. 🗺️ Simulated location is visualized on the map

The predicted end location is visualized to assist in planning retrieval operations.

<img src="https://github.com/user-attachments/assets/bb3e1e80-0f49-4142-8c1d-590ef65d3fdf" width="500" />

---

## 5. 🚢 Assigning collection ships to retrieve the lost gear

Administrators assign the most suitable collection ship based on location and availability.

<img src="https://github.com/user-attachments/assets/2544f8ea-2edb-4520-9fad-6cc6609d35bf" width="500" />

---

## 6. 🪸 Sediment simulation predicts long-term deposition zones

If gear remains unretrieved, sedimentation simulations help estimate **long-term accumulation areas**.

<img src="https://github.com/user-attachments/assets/2022011b-baca-4fc6-9a83-f6eb015f68fa" width="500" />

---

## 💡 System Highlights

- ✅ Real-time reporting with embedded devices
- ✅ Ocean drift simulation using OpenDrift
- ✅ Sediment prediction for long-term risk assessment
- ✅ Centralized web dashboard for monitoring and operation
- ✅ Supports sustainable marine waste management

---
