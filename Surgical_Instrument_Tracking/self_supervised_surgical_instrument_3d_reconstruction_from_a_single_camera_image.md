# Self-Supervised Surgical Instrument 3D Reconstruction from a Single Camera Image

Ange Lou, Xing Yao, Ziteng Liu, Jintong Han, Jack Noble (n.d.)

## 🧩 Problem to Solve

본 논문은 단일 카메라 이미지로부터 수술 도구의 3차원 구조를 복원하는 문제를 다룬다. 수술 도구 추적(Surgical instrument tracking)은 수술 중 해부학적 구조와 도구의 상대적 위치에 대한 피드백을 제공하는 중요한 기술이다. 기존의 추적 방식은 주로 2D 세그멘테이션(Segmentation)과 객체 검출(Object detection)에 의존하고 있어, 실제 수술 환경에서 필수적인 도구의 3D 포즈(Pose)와 깊이(Depth) 정보를 정확히 예측하는 데 한계가 있다.

정확한 3D 모델을 구축하기 위해서는 3D 속성에 대한 감독 학습(Supervision)이 필요하지만, 수술 도구의 경우 3D 어노테이션(Annotation) 데이터를 확보하는 것이 매우 어렵다. 또한, 수술 도구는 일반적으로 가늘고 긴(Elongated) 형태를 띠고 있어, 자연물 객체를 대상으로 하는 기존의 단일 뷰 3D 복원 방법론들을 그대로 적용했을 때 복원 정확도가 낮고 특히 텍스처(Texture) 정보를 캡처하는 데 어려움이 있다는 문제가 있다. 따라서 본 연구의 목표는 3D 감독 데이터 없이, 오직 2D 실루엣 마스크(Silhouette mask)만을 이용하여 가늘고 긴 수술 도구의 형태와 텍스처를 복원하는 자가 지도 학습(Self-supervised learning) 시스템인 SSIR(Self-supervised Surgical Instrument Reconstruction)을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 3D 정답 데이터 없이 단일 이미지에서 수술 도구의 3D 메시(Mesh)를 복원할 수 있는 엔드 투 엔드(End-to-end) 시스템을 제안한 것이다. 특히, 가늘고 긴 수술 도구의 특성으로 인해 발생하는 텍스처 복원의 어려움을 해결하기 위해 **다중 사이클 일관성(Multi-cycle-consistency)** 전략을 도입하였다. 이 전략은 2D 수준의 감독과 3D 수준의 자가 지도 학습을 결합하여, 모델이 다양한 뷰(View)에서도 일관된 텍스처와 형태를 유지하며 학습할 수 있도록 유도한다.

## 📎 Related Works

기존의 3D 복원 연구들은 3DMM이나 SMPL과 같이 사전 정의된 형태 모델(Morphable model)의 파라미터를 맞추는 방식을 사용했으나, 이는 다양한 객체에 적용하기에 비용이 많이 들고 시간이 오래 걸린다는 단점이 있다. 최근에는 딥러닝 기반의 감독 학습 방식이 성과를 내고 있으며, 특히 미분 가능한 렌더링(Differentiable Rendering, DR)의 등장으로 3D 객체의 그래디언트를 2D 이미지 상에서 계산하여 역전파할 수 있게 되어 2D 감독 기반의 복원이 가능해졌다.

하지만 수술 도구 분야의 기존 추적 기술들은 여전히 2D 기반에 머물러 있으며, 일반적인 2D 감독 기반 3D 복원 방법들은 수술 도구 특유의 가늘고 긴 형태와 복잡한 텍스처 정보를 충분히 학습하지 못하는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 자가 지도 학습 기반의 새로운 파이프라인을 제안함으로써 기존 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. 시스템 전체 구조 및 미분 가능 렌더링

SSIR 시스템은 입력 이미지와 실루엣 마스크로부터 3D 속성을 예측하고, 이를 다시 2D로 렌더링하여 입력값과 비교하는 구조를 가진다.

- **3D 모델 표현**: 3D 메시 $\mathcal{M}(V, T)$는 정점(Vertex)의 집합 $V \in \mathbb{R}^{N \times 3}$와 텍스처 맵(UV map) $T \in \mathbb{R}^{H \times W \times 3}$로 정의된다.
- **3D 속성 $\mathcal{A}$**: 렌더링을 위해 카메라 파라미터 $\mathcal{C}$(방위각, 고도, 거리), 조명 파라미터 $\mathcal{L}$(Spherical Harmonics 계수), 형태 $V$, 텍스처 $T$를 포함하는 $\mathcal{A} = [\mathcal{C}, \mathcal{L}, V, T]$를 예측한다.
- **미분 가능 렌더러 (Differentiable Renderer)**: 학습 가능한 파라미터가 없는 렌더러 $R$은 3D 속성 $\mathcal{A}$를 입력받아 2D RGB 이미지 $I_{ren}$과 실루엣 마스크 $M_{ren}$을 생성한다.
  $$ \mathcal{X}_{ren} = R(\mathcal{A}) = [I_{ren}, M_{ren}] $$

### 2. 복원 네트워크 (Reconstruction Network)

네트워크 $\mathcal{V}_\theta$는 4개의 서브 인코더로 구성된다.

- **카메라 인코더 $\mathcal{V}_c$**: 방위각($\alpha$), 고도($\epsilon$), 거리($d$)를 예측한다.
- **조명 인코더 $\mathcal{V}_l$**: Spherical Harmonics 모델의 계수 벡터를 예측한다.
- **형태 인코더 $\mathcal{V}_s$**: 초기 구형 메시 $V_0$에 대한 상대적 변화량 $\Delta V$를 예측하여 최종 형태 $V = \Delta V + V_0$를 결정한다.
- **텍스처 인코더 $\mathcal{V}_t$**: 2D flow map을 예측하고 Spatial Transformation을 통해 UV 텍스처 맵을 생성한다.

### 3. 다중 사이클 일관성 (Multi-cycle-consistency)

가늘고 긴 도구의 텍스처를 잘 잡기 위해, 두 개의 독립적인 복원 모델 $\mathcal{V}_{\theta_1}$과 $\mathcal{V}_{\theta_2}$를 동시에 학습시킨다.

- **새로운 뷰 생성**: 서로 다른 배치에서 샘플링된 두 이미지의 3D 속성 $\mathcal{A}_i, \mathcal{A}_j$를 선형 보간하여 새로운 가상 뷰 $\mathcal{A}_{ij}$를 생성한다.
  $$ \mathcal{A}_{ij} = 0.5 \cdot [(1-\lambda_1)\mathcal{A}_{i\_1} + \lambda_1\mathcal{A}_{j\_1}] + 0.5 \cdot [(1-\lambda_2)\mathcal{A}_{i\_2} + \lambda_2\mathcal{A}_{j\_2}] $$
- **학습 루프**: 이 새로운 속성을 렌더링하여 이미지를 만들고, 그 이미지를 다시 인코더에 넣어 처음의 속성이 복원되는지를 확인하는 사이클을 통해 자가 지도 학습을 수행한다.

### 4. 손실 함수 (Loss Functions)

전체 학습은 다음 세 가지 손실 함수의 합으로 이루어진다.

- **2D 수준 감독 손실 ($\mathcal{L}_{2D}$)**: 렌더링된 이미지와 입력 이미지 사이의 $L_1$ 거리($\mathcal{L}_{img}$)와 실루엣 마스크 간의 IoU 기반 손실($\mathcal{L}_{sil}$)을 사용한다.
  $$ \mathcal{L}_{2D} = \lambda_{img}\mathcal{L}_{img} + \lambda_{sil}\mathcal{L}_{sil} $$
- **3D 수준 자가 지도 손실 ($\mathcal{L}_{3D}$)**: 렌더링된 이미지로부터 다시 예측된 3D 속성과 원래 속성 사이의 차이를 최소화한다.
  $$ \mathcal{L}_{3D} = \frac{1}{N} \sum \| \mathcal{V}_\theta(R(\mathcal{V}_\theta(\mathcal{X}_i))) - \mathcal{V}_\theta(\mathcal{X}_i) \|_1 $$
- **랜드마크 일관성 손실 ($\mathcal{L}_{LC}$)**: 사전 학습된 VGG-19를 통해 특징 맵을 추출하고, MLP를 통해 각 정점(Vertex)의 인덱스를 분류하게 함으로써 두 모델 간의 기하학적 일관성을 강제한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 4개의 와우 이식(Cochlear Implant, CI) 수술 비디오를 사용하였다. 세그멘테이션 마스크는 MMS(Min-Max Similarity) 알고리즘을 통해 생성하였다.
- **비교 대상**: 최신 자가 지도 메시 복원 방법인 SMR(Self-supervised Mesh Reconstruction)과 비교하였다.
- **평가 지표**: 복원 정확도는 Mask IoU로 측정하였으며, 새로운 뷰 생성 능력(Texture quality)은 FID(Frechet Inception Distance)를 통해 평가하였다. (FID는 낮을수록 우수함)

### 정량적 결과

| 모델 | Mask IoU | Reconstruction FID $\downarrow$ | Rotation FID $\downarrow$ |
| :--- | :---: | :---: | :---: |
| SMR | 0.868 | 180.3 | 211.4 |
| **SSIR (Ours)** | **0.867** | **121.4** | **126.6** |

실험 결과, Mask IoU는 SMR과 유사한 수준을 유지하면서도, FID 지표에서는 SSIR이 압도적으로 낮은 수치를 기록하였다. 이는 SSIR이 특히 텍스처 복원과 다양한 각도에서의 이미지 생성 능력에서 훨씬 뛰어난 성능을 보임을 의미한다.

## 🧠 Insights & Discussion

본 논문은 수술 도구라는 특수한 객체(가늘고 긴 형태)를 위해 다중 사이클 일관성이라는 전략을 도입하여 텍스처 복원 문제를 효과적으로 해결하였다. 특히, 카메라 파라미터나 도구 템플릿, 사전 정의된 랜드마크 없이 오직 실루엣 마스크만으로 3D 형태와 텍스처를 복원했다는 점에서 실용성이 높다.

**한계 및 논의사항**:

1. **데이터셋의 규모**: 4개의 비디오라는 매우 적은 양의 데이터셋으로 실험이 진행되어, 일반화 성능에 대한 검증이 더 필요하다.
2. **정밀도 측정**: 현재는 Mask IoU와 FID라는 간접 지표를 사용하고 있다. 실제 3D 모델(Ground Truth)과의 정점 간 거리(Chamfer Distance 등)를 통한 직접적인 형태 오차 측정은 이루어지지 않았다. (저자들도 이를 향후 과제로 언급함)
3. **실시간성**: 미분 가능 렌더링과 다중 인코더 구조가 실제 수술 중 실시간으로 동작할 수 있을지에 대한 분석이 누락되어 있다.

## 📌 TL;DR

본 논문은 단일 2D 이미지로부터 수술 도구의 3D 메시를 복원하는 자가 지도 학습 모델인 **SSIR**을 제안한다. 가늘고 긴 수술 도구의 텍스처 복원 어려움을 해결하기 위해 **다중 사이클 일관성(Multi-cycle-consistency)**과 **랜드마크 일관성(Landmark consistency)** 전략을 도입하였으며, 실험 결과 기존 SMR 대비 텍스처 복원 성능(FID)을 크게 향상시켰다. 이 연구는 향후 수술 중 도구의 정확한 3D 위치 및 포즈 추적 시스템의 기초가 될 가능성이 크다.
