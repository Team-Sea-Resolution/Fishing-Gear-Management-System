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
/sediment: 침적 시뮬레이션 실행 페이지 및 어구 리스트 
</pre>
---

## 1. 🎣 Fishermen report lost fishing gear with one click

Fishermen press a button on-site when they lose fishing gear during operations.

<img src="https://github.com/user-attachments/assets/9989220d-ab43-4a5d-a8ec-65f3b50b633d" width="500" />

---

## 2. 📍 The reported location is automatically recorded on the web page

A web dashboard displays the gear's loss location in real-time.

<img src="https://github.com/user-attachments/assets/d2962eb3-fa52-455e-ac77-638b615a9a09" width="500" />

---

## 3. 🌀 Loss simulation predicts gear movement

Administrators can run a **drift simulation** from the reported location to estimate the future position of the gear.

<img src="https://github.com/user-attachments/assets/21d4b089-f35b-452b-b806-481d9f14ffff" width="500" />

---

## 4. 🗺️ Simulated location is visualized on the map

The predicted end location is visualized to assist in planning retrieval operations.

<img src="https://github.com/user-attachments/assets/9a4c9f18-b0f0-4fba-90b1-14aab44392bc" width="500" />

---

## 5. 🚢 Assigning collection ships to retrieve the lost gear

Administrators assign the most suitable collection ship based on location and availability.

<img src="https://github.com/user-attachments/assets/26dac0cd-8360-4ed6-b212-f945176dd624" width="500" />

---

## 6. 🪸 Sediment simulation predicts long-term deposition zones

If gear remains unretrieved, sedimentation simulations help estimate **long-term accumulation areas**.

<img src="https://github.com/user-attachments/assets/34eb584a-47e2-420c-9222-2a40bcd9f28b" width="500" />

---

## 💡 System Highlights

- ✅ Real-time reporting with embedded devices
- ✅ Ocean drift simulation using OpenDrift
- ✅ Sediment prediction for long-term risk assessment
- ✅ Centralized web dashboard for monitoring and operation
- ✅ Supports sustainable marine waste management

---
