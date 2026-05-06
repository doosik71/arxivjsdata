# Realistic Data Generation for 6D Pose Estimation of Surgical Instruments

Juan Antonio Barragan, Jintan Zhang, Haoying Zhou, Adnan Munawar, and Peter Kazanzides (2024)

## 🧩 Problem to Solve

수술 로봇의 자동화는 환자의 안전과 수술 효율성을 높일 수 있는 잠재력이 있으나, 이를 위해서는 견고한 인지 알고리즘, 특히 수술 도구의 6D Pose Estimation(6차원 포즈 추정) 기술이 필수적이다. 6D Pose Estimation은 카메라 좌표계에 대한 객체의 위치(Translation)와 회전(Rotation)을 추정하는 작업이다.

최근 지도 학습 기반의 딥러닝 알고리즘이 우수한 성능을 보이고 있지만, 이러한 모델들은 대량의 어노테이션(Annotation)된 데이터셋을 필요로 한다. 일반적인 산업 및 가정 환경에서는 3D 컴퓨터 그래픽스 소프트웨어를 이용한 합성 데이터(Synthetic Data)로 이를 해결하지만, 수술 도메인에서는 다음과 같은 이유로 적용이 어렵다.

1. **현실적인 상호작용의 부재**: 상용 그래픽 소프트웨어는 수술 도구와 조직(Tissue) 간의 복잡한 상호작용을 사실적으로 묘사하는 도구가 부족하다.
2. **관절형 도구 지원 부족**: 수술용 로봇 도구와 같은 관절형(Articulated) 구조를 처리하는 기능이 제한적이다.

따라서 본 논문의 목표는 수술 도구의 6D Pose Estimation을 위해 대규모의 다양하고 현실적인 데이터셋을 자동으로 생성할 수 있는 개선된 시뮬레이션 환경과 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 실제 수술 환경을 모사한 물리적 기반 시뮬레이션과 텔레오퍼레이션(Teleoperation) 기록을 결합하여, 수작업 어노테이션 없이도 고품질의 학습 데이터를 대량 생산하는 자동화 파이프라인을 구축하는 것이다. 주요 기여 사항은 다음과 같다.

- **자동 데이터 생성 파이프라인**: 텔레오퍼레이션으로 기록된 궤적(Trajectory)을 재생하며 다양한 카메라 뷰와 조명 조건에서 데이터를 자동 추출하는 시스템을 제안하였다.
- **현실적인 수술 시뮬레이션 환경**: 실제 상용 수술 패드(3-Dmed)를 MRI 스캔하여 모델링함으로써, 가상 환경의 현실성을 높이고 실제 물리적 환경으로의 재현 가능성을 확보하였다.
- **수술용 바늘 데이터셋 구축**: 제안한 시스템을 통해 6D 포즈 어노테이션이 포함된 7,500장의 수술용 바늘 이미지 데이터셋을 생성하였다.
- **SOTA 모델 검증**: 생성된 데이터를 사용하여 최신 6D 포즈 추정 네트워크인 GDR-Net을 학습시키고, 폐색(Occlusion)이 존재하는 challenging한 환경에서도 유효한 성능을 보임을 입증하였다.

## 📎 Related Works

### 기존 시뮬레이션 및 데이터 생성 도구

- **Vision Blender & BlenderProc**: Blender 기반의 데이터 생성 도구로 RGB, Depth, Segmentation 맵 등을 효율적으로 생성할 수 있다. 하지만 객체의 움직임이나 객체 간 상호작용, 특히 관절형 로봇 도구를 다루는 능력에 한계가 있다.
- **NVIDIA Isaac Sim**: PhysX 엔진과 Iray 렌더링을 통해 매우 높은 물리적 정확도와 광학적 현실성을 제공한다. 하지만 본 연구에서는 다양한 입력 장치와의 통합 및 dVRK(da Vinci Research Kit)와의 호환성을 위해 AMBF를 선택하였다.

### AMBF (Asynchronous Multi-Body Framework)

- AMBF는 복잡한 폐루프(Closed-loop) 로봇 시뮬레이션에 최적화된 프레임워크이다. 기존 연구에서 AMBF 기반의 수술 시뮬레이션 환경이 제안되었으나, 딥러닝 학습에 필요한 대규모 데이터셋을 자동으로 생성하는 기능이 부족했으며 가상 환경의 모델이 지나치게 단순하여 실제 물리적 환경과의 괴리가 있었다.

## 🛠️ Methodology

### 1. 자동 데이터 생성 파이프라인

데이터 생성 과정은 크게 두 단계로 나뉜다.

- **기록 단계(Recording Step)**: 텔레오퍼레이션 장치를 사용하여 가상 로봇 조작기를 움직여 수술 작업을 수행한다. 이때 로봇의 조인트 및 카테시안 위치, 객체의 포즈 정보를 `rosbag` 파일에 저장한다.
- **처리 및 생성 단계(Processing and Generation Step)**: 저장된 궤적을 반복해서 재생한다. 이때 카메라의 시점(Viewpoint)과 조명 조건을 무작위로 변경하며, 각 프레임에서 RGB 이미지와 함께 Ground-truth(Depth map, Segmentation map, Camera intrinsics, 6D Pose)를 저장한다. 생성된 데이터는 표준 6D 포즈 추정 포맷인 **BOP (Benchmark for 6D Object Pose Estimation)** 형식을 따른다.

### 2. 가상 환경의 개선

현실성을 높이기 위해 실제 상용 수술 패드(3-Dmed)를 MRI 스캔한 뒤 3D Slicer와 Meshlab으로 전처리하여 메쉬(Mesh)를 생성하였다.

- **시각화 메쉬(Visual Mesh)**: 렌더링을 위한 고해상도 메쉬를 사용한다.
- **충돌 메쉬(Collision Mesh)**: 시뮬레이션 성능 최적화를 위해 볼록한 부분 집합(Convex subshapes)으로 구성된 단순화된 메쉬를 사용하며, 바늘이 삽입될 수 있도록 작은 통로(Corridors)를 설정하였다.

### 3. 수술용 바늘 데이터셋 생성

18.65mm 길이의 수술용 바늘을 대상으로 데이터를 수집하였다.

- 6개의 텔레오퍼레이션 기록을 각각 20번씩, 다른 카메라 위치와 각도에서 재생하였다.
- 최종적으로 바늘이 포함된 6,430장의 학습 이미지와 1,500장의 테스트 이미지를 확보하였다.
- 바늘의 가시성(Visibility)은 다음과 같이 정의하여 측정한다.
$$\text{visibility} = \frac{\text{area of visible mask}}{\text{area of projected mask}}$$
여기서 visible mask는 RGB 이미지에서 바늘에 해당하는 픽셀 집합이며, projected mask는 바늘의 CAD 모델을 Ground-truth 포즈를 이용해 이미지 평면에 투영한 픽셀 집합이다.

### 4. 6D 포즈 추정 모델: GDR-Net

본 연구에서는 fully differentiable한 네트워크인 **GDR-Net**을 사용하였다.

- **입력**: YOLOX 검출기로 추출된 객체의 2D ROI(Region of Interest) RGB 이미지.
- **중간 출력**: 가시적 객체 마스크(Visible object mask), 2D-3D 밀집 대응 맵(2D-3D dense correspondences), 표면 영역 주의 맵(Surface region attention map)의 세 가지 기하학적 특징 맵을 생성한다.
- **최종 출력**: 위 특징 맵들을 **Patch-PnP** 모듈에 입력하여 최종적인 회전(Rotation)과 이동(Translation) 값을 회귀(Regression)한다.

### 5. 평가 지표

추정된 포즈의 정확도는 다음 세 가지 지표로 평가한다.

- **Translation Error ($e^{TE}$)**: 추정된 위치와 실제 위치 사이의 유클리드 거리.
$$e^{TE} = \|\bar{t} - \hat{t}\|$$
- **Rotation Error ($e^{RE}$)**: 회전 행렬의 axis-angle 표현을 이용한 오차.
$$e^{RE} = \arccos((\text{Tr}(\bar{R}\hat{R}^{-1} - I)/2)$$
- **Maximum Symmetry-Aware Surface Distance ($e^{MSSD}$)**: 객체의 대칭성을 고려하여, 실제 포즈와 추정 포즈로 변환된 모델 정점들 간의 최대 거리 중 최소값을 측정한다.
$$e^{MSSD} = \min_{S \in S_M} \max_{x \in V_M} \|\hat{P}x - \bar{P}Sx\|_2$$

## 📊 Results

### 실험 설정

- **YOLOX**: Ranger Optimizer, batch size 16, learning rate $1e-3$, 30 epochs 학습.
- **GDR-Net**: Ranger Optimizer, batch size 48, learning rate $8e-4$, 450 epochs 학습.
- **평가 대상**: 바늘의 30% 이상이 보이는 이미지(N=1,458)만을 대상으로 평가하였다.

### 정량적 결과

평가 결과, GDR-Net은 다음과 같은 성능을 기록하였다 (Table I 참조).

| 지표 | Mean | Std | Median |
| :--- | :---: | :---: | :---: |
| $e^{RE}$ (deg) | 11.85 | 17.52 | **7.74** |
| $e^{TE}$ (mm) | 2.59 | 3.41 | **1.49** |
| $e^{MSSD}$ (mm) | 2.09 | 2.43 | **1.43** |

- **분석**: 중앙값(Median) 기준 translation error가 1.49mm로, 이는 바늘 직경의 20% 미만인 매우 정밀한 수치이다. 특히, 폐색(Occlusion)이 빈번한 데이터셋임에도 불구하고, 기존의 비폐색 바늘 추적 연구들과 비교하여 경쟁력 있는 수준의 성능을 보였다.
- **오차 원인**: 일부 샘플에서 매우 높은 오차가 발생하였는데, 이는 시각적으로 여러 포즈가 구분되지 않는 모호성(Ambiguity) 때문인 것으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 AMBF 기반의 자동화 파이프라인을 통해 수술 도구의 6D 포즈 추정을 위한 대규모 합성 데이터를 효율적으로 생성할 수 있음을 보였다. 특히 실제 수술 패드를 MRI 스캔하여 적용함으로써 시뮬레이션의 현실성을 높였으며, 이를 통해 학습된 모델이 실제 수술 환경과 유사한 폐색 상황에서도 강건하게 동작함을 입증하였다.

### 한계점 및 해결 방안

- **시각적 모호성**: 바늘의 꼬리와 끝부분이 모두 가려지거나, 바늘이 직선으로 보일 때(곡률이 보이지 않을 때) 포즈 추정의 오차가 커지는 경향이 있다. 이는 순수하게 시각 정보에만 의존하는 딥러닝 모델의 한계이며, 향후 로봇의 기구학적 정보(Kinematics)나 이전 프레임의 포즈 정보를 결합한 **Model-based Tracker**를 통해 해결할 수 있을 것으로 보인다.
- **Sim-to-Real Gap (도메인 간극)**: 현재 사용된 Blinn-Phong 쉐이딩 기법은 단순하여 실제 환경으로 전이 시 성능 저하가 우려된다. 이를 위해 저자는 Blender의 **Eevee**(실시간 래스터화) 및 **Cycles**(물리 기반 경로 추적) 렌더러를 적용하는 예비 파이프라인을 구축하였으며, 렌더링 품질 향상이 Sim-to-Real 전이에 미치는 영향을 연구할 계획이다.

## 📌 TL;DR

본 논문은 수술 로봇의 자동화를 위한 **수술 도구 6D 포즈 추정용 데이터 자동 생성 파이프라인**을 제안한다. 텔레오퍼레이션 기록 재생 방식과 MRI 기반의 정밀한 수술 패드 모델링을 통해 7,500장의 고품질 데이터셋을 구축하였으며, 이를 GDR-Net에 학습시킨 결과 폐색이 있는 환경에서도 중앙값 기준 1.49mm의 위치 오차라는 우수한 성능을 확보하였다. 이 연구는 수작업 어노테이션 비용을 획기적으로 줄이면서도 현실적인 수술 비전 알고리즘을 개발할 수 있는 표준화된 플랫폼을 제공한다는 점에서 가치가 크다.
