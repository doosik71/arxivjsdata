# Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting

Shuojue Yang, Zijian Wu, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, and Yueming Jin (2025)

## 🧩 Problem to Solve

본 논문은 수술용 AI 및 자율 주행 수술 시스템의 발전에 필수적인 Real2Sim(실제 환경을 시뮬레이션으로 변환) 과정에서 발생하는 시각적 불일치 문제를 해결하고자 한다. 기존의 시뮬레이션 환경에서는 주로 CAD 메쉬 모델을 사용하는데, 이는 실제 수술 도구의 외관과 차이가 커서 Sim-to-Real 전이 학습의 성능을 저하시키는 원인이 된다.

최근 3D Gaussian Splatting (GS)과 같은 고충실도 3D 재구성 기술이 등장하였으나, 수술 도구에 적용하기에는 다음과 같은 한계가 존재한다. 첫째, 기존의 GS 기반 수술 도구 재구성 방식은 정적인 상태만을 모델링하거나 특수 데이터 수집 시스템이 필요하여, 관절 움직임에 따른 동적인 제어가 불가능하다. 둘째, 기존의 동적 장면 재구성 방식들은 변형 필드(deformation fields)가 단순히 시간(timestamp)에 의존하므로 사용자가 도구의 관절 상태를 직접 제어할 수 없으며, 수술 중 발생하는 크고 복잡한 움직임에 취약하다. 따라서 본 논문의 목표는 단안 수술 비디오와 텍스처가 없는 CAD 모델만을 이용하여, 시각적으로 매우 정교하면서도 관절 제어가 가능한 수술 도구의 3D GS 모델을 재구성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 3D Gaussian Splatting을 수술 도구의 기하학적 구조(CAD 메쉬)와 결합하여, Forward Kinematics (FK)를 통해 제어 가능한 형태로 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **Geometry Pre-training**: GS 포인트들을 파트별 메쉬 표면에 결합하여 정확한 기하학적 사전 정보를 부여함으로써, 모델의 조작 가능성과 시각적 충실도를 동시에 확보하였다.
2. **Controllable GS**: 수술 도구의 관절 상태와 포즈에 따라 GS 포인트들의 위치와 회전을 업데이트하는 FK 체계를 도입하여, 실제 도구처럼 유연하게 제어할 수 있는 3D 자산을 생성하였다.
3. **Robust Pose Tracking**: 렌더링-비교(render-and-compare) 방식에 기반하여, 세그멘테이션 마스크와 구조적 제약 조건을 활용한 포즈 추정 모듈을 제안하였다. 특히, 프레임 간 큰 움직임을 처리하기 위해 이미지 매칭과 PnP(Perspective-n-Point)를 활용한 초기화 전략을 도입하였다.

## 📎 Related Works

기존의 수술 장면 재구성 연구들은 주로 Neural Radiance Fields (NeRF)나 3D Gaussian Splatting (GS)을 활용하여 높은 시각적 품질을 구현해 왔다. 예를 들어, EndoGaussian이나 Deform3DGS와 같은 최신 기법들은 변형 가능한 조직(deformable tissue) 재구성에서 우수한 성능을 보였다.

그러나 이러한 기존 방식들은 다음과 같은 차별점이 있다. 우선, 기존의 동적 재구성 모델들은 변형 필드가 시간에 종속적이어서 사용자가 임의의 관절 각도를 입력하여 도구를 조작하는 '제어 가능성(controllability)'이 결여되어 있다. 또한, 일반적인 로봇 팔 재구성 연구들은 포즈 정보가 이미 주어져 있거나 정확하게 추정되었다고 가정하지만, 실제 수술 환경에서는 기구의 기구학(kinematics) 데이터가 없거나 노이즈가 심해 이를 그대로 적용하기 어렵다. Instrument-Splatting은 이러한 제약 없이 단안 비디오만으로 포즈 추정과 텍스처 학습을 동시에 수행한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. Forward Kinematics 및 3D GS 기초

본 연구에서는 da Vinci EndoWrist Large Needle Driver (LND)를 대상으로 한다. 도구의 관절 상태를 $q = \{\theta_1, \theta_2, \theta_3\} \in \mathbb{R}^3$로 정의하며, 이는 각각 축의 회전과 그리퍼의 개폐 각도를 나타낸다. 임의의 조인트 프레임 $j$의 점 $x_j \in \mathbb{R}^3$를 카메라 프레임으로 변환하는 함수는 다음과 같이 정의된다.

$$T_j(x_j; q, \xi) = {}^{cam}T_s(\xi) {}^{s}T_j(q)x_j$$

여기서 ${}^{s}T_j(q)$는 관절 상태 $q$에 따른 섀프트(shaft) 기준 상대 변환이며, ${}^{cam}T_s(\xi)$는 섀프트에서 카메라로의 변환 $\xi \in se(3)$를 의미한다. 3D GS는 각 포인트의 위치 $\mu$, 회전 $r$, 스케일 $s$, 불투명도 $\alpha$, 색상 $sh$를 학습 가능한 파라미터로 가진다.

### 2. Geometry Pre-training

GS 포인트들이 도구의 각 강체 파트(wrist, shaft, gripper)에 정확히 결합되도록 하기 위해 기하학적 사전 학습을 수행한다.

- 메쉬 표면에서 레이 트레이싱(ray-tracing)을 통해 포인트를 조밀하게 샘플링한다.
- Blender를 통해 렌더링된 Ground-truth 실루엣 맵과 렌더링된 불투명도 간의 $L_1$ 손실을 최소화하여 기하 구조를 학습한다.
- 각 포인트에 세만틱 라벨 $f \in \{1, 2, 3\}$을 부여하여, 관절 상태 $q$와 포즈 $\xi$가 변경될 때 다음과 같이 업데이트되도록 한다.

$$\mu'_j = T_j(\mu_j; q, \xi), \quad r'_j = R_j(r_j; q, \xi)$$

### 3. Pose Estimation and Tracking

렌더링된 결과와 실제 이미지 간의 정렬을 위해 렌더링-비교 프레임워크를 사용한다. 최적화 문제는 다음과 같이 정의된다.

$$\hat{\xi}, \hat{q} = \text{argmin } L(\xi, q), \quad L = L_{mask} + L_{tip}$$

- $L_{mask}$: 렌더링된 세만틱 실루엣과 실제 마스크 간의 $L_1$ 손실이다.
- $L_{tip} = L_{dist} + L_{struct}$: 도구 끝단(tool tip)의 정렬을 강제한다. $L_{dist}$는 렌더링된 끝단과 마스크에서 추출한 끝단 사이의 유클리드 거리 손실이며, $L_{struct}$는 왼쪽 그리퍼가 항상 오른쪽 그리퍼의 왼쪽에 위치하도록 제약하는 구조적 손실이다.
- **Pose Tracking**: 프레임 간 움직임이 클 경우를 대비해, 인접 프레임 간의 특징점 매칭을 수행하고 이를 3D로 리프트(lift)하여 PnP 솔버로 초기 포즈를 추정한 뒤 정밀 최적화를 수행한다.

### 4. Texture Learning

포즈 추정이 완료되면, 기하학적 구조를 유지하기 위해 위치 $\mu$는 고정한 채 색상 및 나머지 파라미터 $\{r, s, \alpha, sh\}$를 학습한다. 이때의 손실 함수는 다음과 같다.

$$L = L_{mask} + L_{color}$$

$L_{color}$는 RGB 픽셀 간의 $L_1$ 손실이며, $L_{mask}$는 학습 과정에서 기하학적 구조가 무너지는 것을 방지하는 정규화 역할을 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis 2017, 2018 데이터셋(각 100프레임) 및 자체 수집한 in-house 데이터셋(ex vivo 조직 및 그린 스크린 배경 비디오 4종)을 사용하였다.
- **평가 지표**: 포즈 정확도는 렌더링된 세만틱 맵과 실제 마스크 간의 Dice score로 측정하였으며, 재구성 품질은 PSNR, SSIM, LPIPS를 사용하였다.

### 주요 결과

1. **구성 요소 분석**: $L_{tip}$과 구조적 제약(Reg)을 제거했을 때, 특히 그리퍼 부분의 Dice score가 크게 하락함을 확인하였다. 또한 Geometry Pre-training (GP)을 제거할 경우 PSNR이 14.68까지 떨어져, 사전 학습된 기하 구조가 시각적 품질에 결정적인 영향을 미침을 입증하였다.
2. **재구성 품질 비교**: EndoGaussian 및 Deform3DGS와 비교한 결과, 실제 수술 비디오(EndoVis)에서는 제안 방법만이 유의미한 결과를 냈으며, 이는 기존 방법들이 큰 프레임 간 움직임을 처리하지 못하기 때문이다.
3. **정량적 성능**: In-house 데이터셋의 Wrist & Gripper 부분에서 제안 방법은 PSNR 30.44를 기록하여 Deform3DGS(28.95) 및 EndoGaussian(25.34)보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문의 강점은 수술 도구의 물리적인 기구학적 제약(FK)을 3D GS에 직접 결합함으로써, 단순한 시각적 재구성을 넘어 '제어 가능한' 디지털 트윈을 구축했다는 점이다. 특히, 렌더링-비교 방식의 고질적인 문제인 지역 최솟값(local minima) 함몰 문제를 구조적 손실 함수($L_{struct}$)와 PnP 기반 초기화 전략으로 효과적으로 해결하였다.

다만, 초기 프레임에서 로고 랜드마크를 수동으로 지정하여 PnP를 수행해야 한다는 점은 완전 자동화 측면에서 한계로 작용한다. 저자들 또한 향후 연구에서 학습 기반의 포즈 추정 모듈을 도입하여 추론 속도와 정확도를 높이겠다고 언급하고 있다. 또한, 본 연구는 da Vinci LND 모델에 최적화되어 있으나, 다른 도구의 CAD 모델만 있다면 쉽게 확장 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 단안 수술 비디오와 CAD 모델을 활용하여, 실제와 똑같이 생겼으면서도 관절 조작이 가능한 수술 도구 3D 모델을 만드는 **Instrument-Splatting** 프레임워크를 제안한다. 기하학적 사전 학습과 Forward Kinematics를 결합한 3D GS, 그리고 강건한 포즈 추적 알고리즘을 통해 기존의 정적/비제어 가능 재구성 방식의 한계를 극복하였다. 이 연구는 수술 AI 학습을 위한 고정밀 시뮬레이션 데이터 생성(Real2Sim)에 크게 기여할 가능성이 높다.
