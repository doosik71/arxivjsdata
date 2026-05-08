# Instrument-Splatting: Controllable Photorealistic Reconstruction of Surgical Instruments Using Gaussian Splatting

Shuojue Yang, Zijian Wu, Mingxuan Hong, Qian Li, Daiyun Shen, Septimiu E. Salcudean, and Yueming Jin (2025)

## 🧩 Problem to Solve

본 논문은 수술용 로봇의 자율성과 AI 발전을 위해 필수적인 **Real2Sim(실제 데이터를 시뮬레이션으로 변환)** 과정에서의 정밀한 3D 자산 생성 문제를 다룬다. 기존의 Sim2Real 전이 학습은 주로 CAD 메쉬 모델을 사용해 왔으나, 이는 시각적으로 비현실적이어서 심각한 Sim-to-Real Gap을 유발하고 결과적으로 모델의 성능을 저하시킨다.

최근 3D Gaussian Splatting (GS)과 같은 기술이 등장하며 높은 시각적 충실도를 달성할 수 있게 되었지만, 수술 도구 적용에는 다음과 같은 한계가 존재한다:

1. **조절 가능성(Controllability) 부족**: 기존의 동적 수술 장면 재구성 방식은 변형 필드가 단순히 타임스탬프에 의존하므로, 사용자가 도구의 관절 상태를 직접 제어할 수 없다.
2. **포즈 정보의 부재**: 일반적인 로봇 팔 재구성 연구는 알려진 포즈(known poses)를 가정하지만, 실제 수술 영상에서는 도구의 기구학(kinematics) 정보가 없거나 노이즈가 심해 적용하기 어렵다.
3. **복잡한 움직임**: 수술 중 발생하는 크고 복잡한 도구의 움직임은 기존 재구성 방법의 품질을 저하시킨다.

따라서 본 논문의 목표는 단안(monocular) 수술 영상과 텍스처가 없는 CAD 모델만을 이용하여, **시각적으로 매우 사실적이면서도 관절 제어가 가능한(controllable) 수술 도구의 3D GS 표현체를 재구성**하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **기하학적 사전 정보(Geometric Prior)를 가진 3D GS를 CAD 메쉬에 바인딩하고, 이를 렌더링-비교(render-and-compare) 기반의 포즈 추적 시스템과 결합**하는 것이다. 주요 기여 사항은 다음과 같다:

- **Geometry Pre-training**: GS 포인트 클라우드를 CAD 메쉬의 각 파트(shaft, wrist, gripper)에 결합하여 정확한 기하학적 구조를 가진 초기 상태를 구축한다.
- **Controllable 3D GS**: 정방향 기구학(Forward Kinematics, FK)을 도입하여, 도구의 관절 변수 및 포즈 변화에 따라 Gaussian들이 유연하게 움직이도록 설계하였다.
- **Robust Pose Tracking**: 세만틱 정보가 임베딩된 Gaussian을 활용하여, 단안 영상에서도 렌더링-비교 방식으로 프레임별 포즈와 관절 상태를 정밀하게 정제하는 추적 방법을 제안한다. 특히 큰 움직임을 처리하기 위해 이미지 매칭 기반의 PnP 초기화 전략을 사용한다.
- **Photorealistic Texture Learning**: 포즈 추적 결과를 바탕으로 도구의 외관(texture)을 학습하여 사진과 같은 수준의 렌더링 품질을 달성한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적하며 차별점을 제시한다:

- **CAD Mesh 기반 방식**: 시각적 사실감이 떨어져 Sim-to-Real Gap이 크다.
- **NeRF 및 3D GS 기반 재구성**: 높은 시각적 품질을 제공하지만, 수술 도구의 경우 관절 변화에 따른 동적 모델링이 어렵거나 데이터 수집을 위한 특수 시스템이 필요하다.
- **동적 수술 장면 재구성 (EndoGaussian, Deform3DGS 등)**: 변형 가능한 조직(deformable tissue) 재구성에는 뛰어나나, 변형 필드가 시간축에 의존하여 사용자가 직접 도구를 제어할 수 없다. 또한, 복잡하고 큰 움직임이 있는 수술 도구의 경우 재구성 품질이 급격히 떨어진다.
- **일반 로봇 팔/인체 재구성**: 알려진 포즈를 가정하므로, 기구학 정보가 없는 수술 환경에서는 실용적이지 않다.

## 🛠️ Methodology

본 논문이 제안하는 **Instrument-Splatting**의 전체 파이프라인은 크게 네 단계로 구성된다.

### 1. 정방향 기구학 (Forward Kinematics)

da Vinci EndoWrist LND를 기준으로, 샤프트(shaft) 프레임을 베이스로 하여 관절 변수 $q = \{\theta_1, \theta_2, \theta_3\}$(회전 및 그리퍼 개폐각)를 정의한다. 임의의 관절 프레임 $j$에서 카메라 프레임으로의 변환 함수 $T_j(\cdot)$는 다음과 같이 정의된다:
$$T_j(x_j; q, \xi) = {}^{cam}T_s(\xi) {}^{s}T_j(q) x_j$$
여기서 $\xi \in se(3)$는 샤프트에서 카메라로의 포즈를 나타내며, $x_j \in \mathbb{R}^3$는 관절 프레임 내의 점이다.

### 2. 기하학적 사전 학습 (Geometry Pre-training)

GS 포인트들을 도구의 각 강체 파트에 바인딩하기 위해 다음 과정을 거친다:

- CAD 모델의 메쉬 표면에서 레이 트레이싱(ray-tracing)을 통해 포인트를 조밀하게 샘플링한다.
- Blender를 사용하여 다양한 시점에서 정답 실루엣 맵(ground-truth silhouette maps)을 렌더링한다.
- 렌더링된 불투명도(opacity)와 정답 실루엣 간의 $L_1$ 손실을 최소화하여 GS의 기하학적 정보를 학습시킨다.
- 각 Gaussian에 세만틱 레이블 $f \in \{1, 2, 3\}$(wrist, shaft, gripper)을 부여하여, 이후 FK 식에 따라 각 파트별로 위치 $\mu$와 회전 $r$이 업데이트되도록 한다.

### 3. 포즈 추정 및 추적 (Pose Estimation & Tracking)

렌더링된 세만틱 실루엣과 실제 영상의 세그멘테이션 마스크를 비교하는 **render-and-compare** 프레임워크를 사용한다. 최적화 문제는 다음과 같다:
$$\hat{\xi}, \hat{q} = \text{argmin } L(\xi, q), \quad L = L_{mask} + L_{tip}$$

- $L_{mask}$: 렌더링된 실루엣과 실제 마스크 간의 $L_1$ 손실이다.
- $L_{tip}$: 툴 팁(tool tip)의 정렬을 강제하는 손실로, 거리 손실 $L_{dist} = \text{ReLU}(d-r)$와 좌우 팁의 상대적 위치를 제약하는 구조적 손실 $L_{struct} = \text{ReLU}(-(\theta_l + \theta_r))$의 합으로 구성된다.
- **추적(Tracking)**: 프레임 간 움직임이 클 경우를 대비해, 손목(wrist) 부분의 특징점 매칭과 PnP 솔버를 이용해 다음 프레임의 초기 포즈를 추정하고 이를 최적화의 시작점으로 사용한다.

### 4. 텍스처 학습 (Texture Learning)

포즈 추적이 완료된 후, 기하학적 구조를 유지하기 위해 위치 $\mu$는 고정하고 나머지 파라미터 $\{r, s, \alpha, sh\}$(회전, 스케일, 불투명도, 색상)만을 학습한다.

- 손실 함수: $L = L_{mask} + L_{color}$ (각각 실루엣과 RGB 픽셀의 $L_1$ 손실)

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis 2017/2018 (각 100프레임 추출) 및 자체 수집한 In-house 데이터(ex vivo 조직 및 green screen 배경)를 사용한다.
- **평가 지표**: 포즈 정확도는 Dice score로, 재구성 품질은 PSNR, SSIM, LPIPS로 측정한다.

### 주요 결과

- **구성 요소 분석 (Table 1)**: $L_{tip}$과 루즈 정규화(loose regularization)를 제거했을 때 특히 그리퍼(gripper) 부분의 Dice score가 크게 하락하며, 기하학적 사전 학습(GP)이 없을 경우 재구성 품질(PSNR)이 매우 낮게 나타난다. 이는 제안된 각 모듈이 성능 향상에 필수적임을 시사한다.
- **재구성 품질 비교 (Table 2)**:
  - **EndoVis 데이터**: 기존 방식인 EndoGaussian과 Deform3DGS는 큰 움직임과 긴 시간 간격으로 인해 재구성에 실패하는 반면, 본 방법은 포즈 추적 모듈 덕분에 훨씬 높은 정확도와 시각적 충실도를 보인다.
  - **In-house 데이터**: 움직임이 완만한 구간에서는 기존 방식과 PSNR이 유사하거나 높을 수 있으나, 시각적으로는 손목과 그리퍼 부분에서 심각한 아티팩트가 발생한다. 반면 Instrument-Splatting은 관절 부위에서도 사진과 같은(photorealistic) 결과를 생성한다.

## 🧠 Insights & Discussion

본 논문은 수술 도구의 **'디지털 트윈'**을 구축함으로써, 사용자가 임의의 관절 상태와 포즈를 입력했을 때 사실적인 이미지를 생성할 수 있는 제어 가능성을 확보하였다.

**강점**:

- 단순히 영상을 재구성하는 것을 넘어, 기구학적 제약 조건을 3D GS에 통합함으로써 조절 가능한(controllable) 자산을 생성했다는 점이 매우 뛰어나다.
- 단안 영상만으로도 정교한 포즈 추적을 가능하게 하여 실용성을 높였다.

**한계 및 논의**:

- **추론 속도**: 현재의 포즈 추정 방식은 최적화 기반(optimizer-based)이므로 연산 시간이 소요된다. 저자들은 향후 학습 기반(learning-based) 포즈 추정 모델을 도입하여 속도와 정확도를 개선할 필요가 있다고 언급한다.
- **가정**: 본 연구는 CAD 모델이 존재한다는 가정하에 진행된다. 모델이 없는 도구에 대한 일반화 가능성은 명시되지 않았다.

## 📌 TL;DR

이 논문은 단안 수술 영상과 CAD 모델을 이용하여, **관절 제어가 가능하고 시각적으로 매우 사실적인 수술 도구 3D Gaussian Splatting 모델을 재구성하는 Instrument-Splatting** 프레임워크를 제안한다. 기하학적 사전 학습과 정밀한 포즈 추적 모듈을 통해 기존의 비제어형 재구성 방식의 한계를 극복하였으며, 이는 향후 수술 AI 및 자율 로봇 수술을 위한 고정밀 시뮬레이션 데이터 생성(Real2Sim)에 핵심적인 역할을 할 것으로 기대된다.
