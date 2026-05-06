# Realistic Data Generation for 6D Pose Estimation of Surgical Instruments

Juan Antonio Barragan, Jintan Zhang, Haoying Zhou, Adnan Munawar, and Peter Kazanzides (2024)

## 🧩 Problem to Solve

외과 수술 로봇의 자동화는 환자의 안전과 수술 효율성을 높일 수 있는 잠재력을 가지고 있으나, 이를 위해서는 견고한 인지 알고리즘이 필수적이다. 특히, 시각적 피드백을 기반으로 수술 동작을 자동으로 수행하기 위해서는 수술 도구의 6D Pose Estimation(6차원 포즈 추정)이 매우 중요하다.

최근 지도 학습 기반의 딥러닝 알고리즘들이 6D Pose Estimation에서 뛰어난 성능을 보이고 있지만, 이러한 모델들의 성공은 대량의 어노테이션(annotation) 데이터 확보에 달려 있다. 일반적인 산업 환경에서는 3D 그래픽 소프트웨어를 이용한 합성 데이터(synthetic data)가 대안으로 사용되지만, 수술 도메인에서는 상용 그래픽 소프트웨어가 실제적인 도구-조직 간의 상호작용(instrument-tissue interaction)을 구현하는 데 한계가 있다. 따라서 본 논문은 수술 도구의 6D 포즈 추정을 위해 대규모의 다양하고 현실적인 데이터셋을 자동으로 생성할 수 있는 개선된 시뮬레이션 환경 및 파이프라인을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 실제 수술 환경의 물리적 특성과 동작을 반영한 시뮬레이션 환경을 구축하고, 원격 조종(teleoperation)을 통해 수집된 실제 궤적을 재활용하여 대량의 합성 데이터를 생성하는 자동화 파이프라인을 설계하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. 수술 도구의 6D Pose Estimation을 위한 자동화된 데이터 생성 파이프라인을 제안한다.
2. 상용 수술 패드(suturing pad) 모델을 기반으로 한 현실적인 수술 봉합 시뮬레이션 환경을 구축한다.
3. 시뮬레이션된 수술 바늘(surgical needle)에 대해 6D 포즈 어노테이션이 포함된 7.5k 이미지 데이터셋을 생성한다.
4. 최신 6D 포즈 추정 신경망인 GDR-Net을 사용하여 제안한 파이프라인으로 생성된 데이터의 효용성을 검증한다.

## 📎 Related Works

기존의 합성 데이터 생성 도구인 Vision Blender나 BlenderProc는 RGB, depth, segmentation 맵 등을 효율적으로 생성할 수 있으나, 객체가 장면 내에서 어떻게 움직이고 상호작용하는지에 대한 제어 능력이 부족하며 특히 관절형(articulated) 로봇 도구 지원이 미흡하다.

NVIDIA Isaac Sim과 같은 고성능 시뮬레이터는 물리적으로 정확한 환경과 포토릴리스틱한 렌더링을 제공하지만, 본 연구에서는 다양한 입력 장치 지원과 특히 da Vinci Research Kit(dVRK)과의 긴밀한 통합을 통해 실제 수술과 유사한 로봇 동작을 수집할 수 있는 Asynchronous Multi-Body Framework(AMBF)를 선택하여 사용하였다. AMBF는 복잡한 폐루프 로봇 시뮬레이션에 최적화된 프런트엔드 기술과 OpenGL 셰이더를 통한 다양한 데이터 생성 기능을 제공한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 자동 데이터 생성 파이프라인 (Data Generation Pipeline)

제안된 파이프라인은 크게 두 단계로 구성된다.

- **기록 단계 (Recording Step):** 원격 조종 장치를 사용하여 가상 로봇 조작기를 움직여 수술 작업을 수행한다. 이때 로봇의 관절 및 데카르트 위치, 그리고 시뮬레이션 내 다른 객체들의 포즈 정보를 `rosbag` 파일로 저장한다.
- **처리 및 생성 단계 (Processing and Generation Step):** 저장된 로봇 궤적을 다양한 카메라 시점과 조명 조건 하에서 여러 번 반복 재생(replay)한다. 재생 과정에서 수집 스크립트는 단안(monocular) 또는 스테레오 RGB 이미지와 함께 depth map, segmentation map, 카메라 내부 파라미터, 그리고 카메라 좌표계 기준의 객체 포즈(ground-truth)를 저장한다. 생성된 데이터는 6D 포즈 추정의 표준 포맷인 BOP (Benchmark for 6D Object Pose Estimation) 형식을 따른다.

### 2. 가상 장면 개선 (Improvements of Virtual Scene)

현실성을 높이기 위해 실제 상용 수술 패드(3-Dmed)를 시뮬레이션에 추가하였다.

- **모델링:** MRI 스캔을 통해 수술 패드의 메쉬(mesh)를 획득하고, 3D Slicer와 Meshlab을 통해 전처리하였다.
- **구현:** Blender-AMBF 애드온을 사용하여 AMBF Description File(ADF)을 생성하였다. 시각화를 위해서는 고해상도 메쉬를 사용하고, 연산 효율을 위해 충돌 계산(collision)에는 볼록한 하위 형상들로 구성된 단순화된 메쉬를 사용하였다. 특히 바늘이 실제로 삽입될 수 있도록 충돌 메쉬에 작은 통로(corridors)를 설계하였다.

### 3. 수술 바늘 데이터셋 생성 및 학습

18.65mm 길이의 수술 바늘을 대상으로 데이터를 수집하였다. 6개의 `rosbag` 기록을 각각 20번씩 서로 다른 카메라 위치와 각도에서 재생하여 총 7,930장의 이미지(학습 6,430장, 테스트 1,500장)를 확보하였다. 바늘의 가시성(visibility)은 다음과 같이 계산된다.
$$\text{visibility} = \frac{\text{area of visible mask}}{\text{area of projected mask}}$$
여기서 visible mask는 RGB 이미지에서 바늘에 해당하는 픽셀 집합이며, projected mask는 ground-truth 포즈를 이용해 바늘의 CAD 모델을 이미지 평면에 투영하여 얻은 픽셀 집합이다.

### 4. 6D 포즈 추정 모델: GDR-Net

본 연구에서는 완전 미분 가능한 구조인 GDR-Net을 사용하였다.

- **입력 및 출력:** 객체가 위치한 2D ROI(Region of Interest)를 입력받아 가시적 객체 마스크(visible object mask), 2D-3D 조밀 대응 맵(dense correspondences), 표면 영역 어텐션 맵(surface region attention map)의 세 가지 중간 기하학적 특징 맵을 출력한다.
- **포즈 도출:** 이 맵들은 Patch-PnP 모듈로 전달되어 최종적인 회전(rotation)과 평행이동(translation) 값을 회귀(regression)한다. ROI 추출을 위해서는 YOLOX 검출기를 함께 학습시켜 사용하였다.

### 5. 평가 지표 (Evaluation Metrics)

추정된 포즈의 정확도는 다음 세 가지 지표로 측정한다.

- **평행이동 오차 ($e^{TE}$):** ground-truth 평행이동 벡터 $\bar{t}$와 추정치 $\hat{t}$ 사이의 유클리드 거리이다.
$$e^{TE} = ||\bar{t} - \hat{t}||$$
- **회전 오차 ($e^{RE}$):** 회전 행렬 $\bar{R}$과 $\hat{R}$ 사이의 각도 차이를 측정한다.
$$e^{RE} = \arccos((\text{Tr}(\bar{R}\hat{R}^{-1} - 1)/2))$$
- **최대 대칭 인식 표면 거리 ($e^{MSSD}$):** 객체의 대칭성을 고려하여, 모델의 정점들이 ground-truth 포즈와 추정 포즈 사이에서 가지는 최대 거리를 측정한다.
$$e^{MSSD} = \min_{S \in S_M} \max_{x \in V_M} ||\hat{P}x - \bar{P}Sx||_2$$
(여기서 $S_M$은 객체의 대칭 변환 집합, $V_M$은 모델의 정점 집합이다.)

## 📊 Results

실험 결과, GDR-Net은 테스트 데이터셋(N=1,458)에서 다음과 같은 성능을 보였다.

- **정량적 결과:**
  - **Rotation Error ($e^{RE}$):** Median $7.74^\circ$ / Mean $11.85^\circ$
  - **Translation Error ($e^{TE}$):** Median $1.49\text{mm}$ / Mean $2.59\text{mm}$
  - **MSSD ($e^{MSSD}$):** Median $1.43\text{mm}$ / Mean $2.09\text{mm}$

바늘의 직경이 18.65mm인 점을 고려할 때, 중앙값 기준 평행이동 오차(1.49mm)는 직경의 20% 미만으로 매우 정밀한 수준이다. 특히 본 데이터셋은 바늘이 도구나 조직에 의해 부분적으로 가려지는(occlusion) 도전적인 상황을 포함하고 있음에도 불구하고, 가려짐이 없는 상황을 다룬 기존 연구들과 대등한 성능을 보였다. 다만, 일부 샘플에서는 매우 높은 오차가 발생하였는데, 이는 바늘의 팁(tip)과 꼬리(tail)가 모두 가려지거나 바늘의 곡률이 보이지 않아 직선처럼 보이는 경우와 같이 시각적 모호성이 존재하는 상황에서 기인한 것으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 AMBF 기반의 시뮬레이션 환경을 통해 수술 도구의 6D 포즈 추정에 필요한 고품질의 합성 데이터를 효율적으로 생성할 수 있음을 입증하였다. 특히 실제 수술 패드를 MRI 스캔하여 가상 환경에 구축함으로써 시뮬레이션과 실제 환경 간의 물리적 일치성을 높였다.

**강점 및 한계:**

- **강점:** 단순한 랜덤 생성이 아니라 원격 조종된 실제 궤적을 기반으로 데이터를 생성함으로써, 수술 도구의 현실적인 움직임과 상호작용을 반영한 데이터셋을 구축하였다.
- **한계 및 모호성:** 딥러닝 모델이 시각적 외형에만 의존하기 때문에, 바늘의 양끝이 가려지거나 정면에서 바라봐 곡률이 사라지는 경우 포즈 추정에 모호성이 발생한다. 저자들은 이를 해결하기 위해 로봇의 키네마틱 정보나 이전 프레임의 포즈 정보를 활용하는 모델 기반 추적기(model-based tracker)와의 결합을 해결책으로 제시한다.

**비판적 해석 및 향후 방향:**
현재의 렌더링 방식(Blinn-Phong shading)은 단순하여 실제 환경으로 전이했을 때 '도메인 갭(domain gap)' 문제가 발생할 가능성이 크다. 이를 극복하기 위해 저자들은 AMBF의 포즈 데이터를 Blender의 Eevee(실시간 래스터화) 및 Cycles(물리 기반 패스 트레이싱) 렌더러로 전달하여 고품질 이미지를 생성하는 예비 파이프라인을 구현하였다. 향후 연구에서는 이러한 렌더링 품질의 차이가 실제 Sim-to-Real 전이 성능에 미치는 영향을 분석하는 것이 핵심이 될 것이다.

## 📌 TL;DR

본 논문은 수술 로봇의 6D 포즈 추정을 위해 **실제 수술 패드 모델과 원격 조종 궤적을 활용한 자동 합성 데이터 생성 파이프라인**을 제안한다. 이를 통해 생성된 7.5k 이미지 데이터셋으로 GDR-Net을 학습시킨 결과, 부분적 가려짐이 있는 환경에서도 높은 정밀도의 포즈 추정 성능을 확인하였다. 이 연구는 수술 로봇의 시각 인지 알고리즘을 개발하고 검증하기 위한 표준화된 데이터 생성 플랫폼을 제공한다는 점에서 향후 Sim-to-Real 전이 연구에 중요한 기반이 될 것으로 보인다.
