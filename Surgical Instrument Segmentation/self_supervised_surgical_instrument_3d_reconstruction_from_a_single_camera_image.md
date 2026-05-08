# Self-Supervised Surgical Instrument 3D Reconstruction from a Single Camera Image

Ange Lou, Xing Yao, Ziteng Liu, Jintong Han, Jack Noble (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 단일 카메라 이미지로부터 수술 도구의 3D 형상과 텍스처를 정확하게 복원하는 것이다. 수술 도구의 추적(Tracking)은 외과 의사에게 도구의 위치와 해부학적 구조 간의 관계에 대한 피드백을 제공하는 데 매우 중요하다. 기존의 수술 도구 추적 방법들은 주로 2D 세그멘테이션(Segmentation)이나 객체 탐지(Object Detection)에 의존하고 있으며, 이는 3D 공간에서의 자세(Pose)와 깊이(Depth)를 예측하는 데 한계가 있다.

정밀한 3D 복원은 수술 도구의 3D 포즈 및 깊이 예측을 위한 필수 전제 조건이다. 그러나 일반적인 3D 복원 방법들은 3D 속성 수준의 정답(Ground Truth) 데이터가 필요하며, 특히 수술 도구처럼 가늘고 긴(Elongated) 형태의 객체는 기존의 2D 지도 기반 복원 방법으로 구현하기 어렵다. 특히 인공와우(Cochlear Implant, CI) 수술과 같이 정밀한 삽입 경로와 깊이 제어가 필요한 경우, 실시간으로 도구의 3D 정보를 복원하는 시스템의 필요성이 매우 높다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단일 프레임의 현미경 이미지로부터 수술 도구의 3D 메쉬(Mesh)를 복원하는 엔드투엔드(End-to-end) 시스템인 **SSIR(Self-supervised Surgical Instrument Reconstruction)**을 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **Self-supervised Learning**: 3D 정답 데이터 없이, 오직 2D 실루엣 마스크(Silhouette mask) 레이블만을 사용하여 3D 복원을 수행한다.
2. **Multi-cycle-consistency 전략**: 가늘고 긴 수술 도구의 특성상 텍스처 정보를 포착하기 어렵다는 점을 해결하기 위해, 다중 사이클 일관성 학습 전략을 도입하여 텍스처 복원 성능을 향상시켰다.
3. **Dual-encoder 구조 및 랜드마크 일관성**: 두 개의 독립적인 복원 모델을 동시에 학습시키고, 랜드마크 일관성(Landmark Consistency) 손실을 통해 모델 간의 기하학적 일관성을 강제함으로써 복원 품질을 높였다.

## 📎 Related Works

기존의 3D 복원 연구들은 주로 3DMM이나 SMPL과 같은 사전 정의된 형태 모델(Morphable Model)의 파라미터를 최적화하는 방식을 사용했으나, 이는 다양한 객체에 적용하기에 비용이 많이 들고 시간이 오래 걸린다는 한계가 있다. 최근의 딥러닝 기반 방법들은 3D 정답 데이터를 직접 사용하여 오차를 최소화하는 지도 학습(Supervised Learning) 방식을 취하지만, 수술 환경에서 3D 어노테이션을 획득하는 것은 매우 어렵다.

이를 해결하기 위해 **Differentiable Rendering (DR)**을 활용한 2D 지도 기반 복원 방법들이 등장하였다. 하지만 이러한 방법들은 일반적인 자연물 객체에 최적화되어 있어, 수술 도구와 같이 극단적으로 가늘고 긴 형태의 객체를 복원할 때 정확도가 떨어지며 텍스처 정보를 제대로 학습하지 못하는 한계가 있다. SSIR은 이러한 한계를 극복하기 위해 수술 도구 특화된 자기지도 학습 전략을 제안한다.

## 🛠️ Methodology

### 전체 파이프라인

SSIR은 입력 이미지 $\mathcal{I}_i$와 실루엣 마스크 $\mathcal{M}_i$를 받아 3D 속성 $\mathcal{A}$를 예측하고, 이를 다시 2D로 렌더링하여 입력값과 비교하는 구조를 가진다.

### 1. Differentiable Rendering (DR)

3D 메쉬 모델은 정점(Vertex) $\mathcal{V} \in \mathbb{R}^{V \times 3}$와 텍스처 맵 $\mathcal{T} \in \mathbb{R}^{H \times W \times 3}$로 표현된다. 렌더링에 필요한 파라미터 $\mathcal{C}$는 카메라의 방위각(Azimuth), 고도(Elevation), 거리(Distance)로 정의되며, 조명 속성 $\mathcal{L}$은 Spherical Harmonics로 모델링된다.
렌더링 과정은 다음과 같이 정의된다:
$$\mathcal{I}_r = \text{R}(\mathcal{A}) = \text{R}([\mathcal{C}, \mathcal{L}, \mathcal{V}, \mathcal{T}])$$
여기서 $\text{R}$은 학습 가능한 파라미터가 없는 미분 가능한 렌더러(Differentiable Renderer)이다.

### 2. Reconstruction Network

네 개의 하위 인코더 $\mathcal{F} = \{\mathcal{F}_C, \mathcal{F}_L, \mathcal{F}_S, \mathcal{F}_T\}$가 각각 카메라, 조명, 형상(Shape), 텍스처를 예측한다.

- **카메라 인코더**: 방위각, 고도, 거리를 예측한다.
- **형상 인코더**: 초기 구형 메쉬 $\mathcal{V}_0$에 대한 상대적 변화량 $\Delta \mathcal{V}$를 예측하여 최종 형상 $\mathcal{V} = \Delta \mathcal{V} + \mathcal{V}_0$를 생성한다.
- **텍스처 인코더**: 2D flow map을 예측하고 공간 변환(Spatial Transformation)을 통해 UV 맵을 생성한다.
- **조명 인코더**: Spherical Harmonics 계수를 예측한다.

### 3. Multi-cycle Consistency 및 손실 함수

두 개의 독립적인 모델 $\mathcal{F}_{\theta_1}$과 $\mathcal{F}_{\theta_2}$를 동시에 학습시키며, 다음과 같은 세 가지 손실 함수를 사용한다.

**A. 2D Level Supervision ($\mathcal{L}_{2D}$)**
렌더링된 이미지 $\mathcal{I}_{i,r}$와 입력 이미지 $\mathcal{I}_i$ 사이의 L1 거리($\mathcal{L}_{img}$)와 실루엣 마스크 사이의 IoU($\mathcal{L}_{sil}$)를 측정한다.
$$\mathcal{L}_{2D}^\theta = \lambda_{img}\mathcal{L}_{img}^\theta + \lambda_{sil}\mathcal{L}_{sil}^\theta$$

**B. 3D Level Self-supervision ($\mathcal{L}_{3D}$)**
새로운 시점(Novel View)을 생성하기 위해 두 모델의 속성을 보간(Interpolation)하여 $\mathcal{A}_{ij}$를 생성한다. 이후 렌더링된 이미지를 다시 인코더에 넣어 예측된 속성이 원래의 속성과 일치하는지 확인하는 사이클 일관성을 적용한다.
$$\mathcal{L}_{3D}^\theta = \frac{1}{N} \sum \|\mathcal{F}_\theta(\text{R}(\mathcal{F}_\theta(\mathcal{I}_i))) - \mathcal{F}_\theta(\mathcal{I}_i)\|_1$$

**C. Landmark Consistency ($\mathcal{L}_{LC}$)**
VGG-19를 사용하여 특성 맵을 추출하고, 3D 정점을 2D 이미지 평면에 투영하여 각 정점이 어떤 인덱스인지 분류하는 MLP를 통해 랜드마크의 일관성을 학습한다. 이는 두 모델 $\mathcal{F}_{\theta_1}, \mathcal{F}_{\theta_2}$가 동일한 기하학적 위치를 인식하도록 강제한다.

**최종 손실 함수**:
$$\mathcal{L} = 0.5 \cdot (\lambda_{2D}\mathcal{L}_{2D}^{\theta_1} + \lambda_{3D}\mathcal{L}_{3D}^{\theta_1} + \lambda_{LC}\mathcal{L}_{LC}^{\theta_1}) + 0.5 \cdot (\lambda_{2D}\mathcal{L}_{2D}^{\theta_2} + \lambda_{3D}\mathcal{L}_{3D}^{\theta_2} + \lambda_{LC}\mathcal{L}_{LC}^{\theta_2})$$

## 📊 Results

### 실험 설정

- **데이터셋**: 인공와우(CI) 삽입 도구가 포함된 4개의 수술 비디오를 사용하였으며, MMS 알고리즘을 통해 실루엣 마스크를 생성하였다.
- **평가 지표**: 3D 복원 정확도는 **Mask IoU**로 측정하였으며, 새로운 시점(Novel View) 생성 성능은 **FID(Frechet Inception Distance)**를 사용하여 측정하였다 (FID는 낮을수록 좋음).

### 정량적 결과

| 방법론 | Mask IoU | Reconstruction FID $\downarrow$ | Rotation FID $\downarrow$ |
| :--- | :---: | :---: | :---: |
| SMR | 0.868 | 180.3 | 211.4 |
| **SSIR (Ours)** | **0.867** | **121.4** | **126.6** |

- **분석**: Mask IoU는 SMR과 유사한 수준을 보였으나, FID 지표에서 SSIR이 압도적으로 낮은 수치를 기록하였다. 이는 SSIR이 SMR보다 훨씬 더 정교한 텍스처 복원 및 새로운 시점 이미지 생성 능력을 갖추었음을 의미한다.

## 🧠 Insights & Discussion

본 논문은 3D 정답 데이터 없이 오직 2D 실루엣 마스크만을 사용하여 수술 도구를 복원할 수 있음을 입증하였다. 특히 **Multi-cycle-consistency** 전략은 가늘고 긴 형태의 도구에서 발생하는 텍스처 소실 문제를 효과적으로 해결하였으며, 이는 결과적으로 더 낮은 FID 점수로 이어졌다.

**강점**:

- 카메라 파라미터, 도구 템플릿, 사전 정의된 랜드마크 없이도 학습이 가능하다는 점이 매우 실용적이다.
- 수술 도구라는 특수한 형태(Elongated shape)에 최적화된 자기지도 학습 루프를 설계하였다.

**한계 및 논의**:

- **정량적 형상 평가 부족**: 렌더링된 이미지의 유사도(IoU, FID)는 측정하였으나, 실제 3D 모델의 정답(GT)과 정점 간의 거리 오차(Shape Error)를 직접적으로 측정하지 않았다. 이는 논문의 결론 부분에서도 향후 과제로 언급되었다.
- **데이터셋 규모**: 4개의 비디오라는 매우 적은 양의 데이터로 실험이 진행되었으므로, 더 다양한 수술 도구와 환경에서의 일반화 성능 검증이 필요하다.

## 📌 TL;DR

본 논문은 단일 이미지에서 수술 도구를 3D로 복원하는 자기지도 학습 모델 **SSIR**을 제안한다. 텍스처 복원이 어려운 가느다란 도구의 특성을 해결하기 위해 **다중 사이클 일관성(Multi-cycle-consistency)**과 **랜드마크 일관성** 손실을 도입하였으며, 실험 결과 기존 SMR 방식보다 훨씬 우수한 텍스처 복원 및 시점 생성 능력을 보여주었다. 이 연구는 향후 수술실 내 도구의 정밀한 3D 위치 추적 및 내비게이션 시스템 구축에 중요한 기초가 될 수 있다.
