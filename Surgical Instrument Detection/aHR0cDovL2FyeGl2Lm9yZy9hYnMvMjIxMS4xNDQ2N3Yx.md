# Self-Supervised Surgical Instrument 3D Reconstruction from a Single Camera Image

Ange Lou, Xing Yao, Ziteng Liu, Jintong Han, Jack Noble (2022)

## 🧩 Problem to Solve

본 논문은 단일 카메라 이미지로부터 수술 도구의 3D 형태와 텍스처를 복원하는 문제를 해결하고자 한다. 기존의 수술 도구 추적 방식은 주로 2D 세그멘테이션(Segmentation)이나 객체 탐지(Object Detection)에 의존하고 있으며, 이는 실제 수술 환경에서 도구의 정확한 포즈(Pose)와 깊이(Depth)를 파악하는 데 한계가 있다.

정밀한 3D 복원을 위해서는 3D 속성 수준의 감독(Supervision) 데이터가 필요하지만, 수술 도구의 3D 어노테이션을 획득하는 것은 매우 어렵다. 또한, 기존의 단일 뷰(Single-view) 3D 복원 방법들은 주로 일반적인 자연물 객체에 최적화되어 있어, 수술 도구와 같이 가늘고 긴(Elongated) 형태의 객체를 복원할 때 정확도가 떨어지며 텍스처 정보를 효과적으로 캡처하지 못하는 문제가 있다. 따라서 본 연구의 목표는 3D 정답 데이터 없이 2D 실루엣 마스크(Silhouette mask)만으로 수술 도구의 3D 메시(Mesh)를 복원하는 자기지도 학습(Self-supervised learning) 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **SSIR (Self-supervised Surgical Instrument Reconstruction)**이라는 엔드투엔드(End-to-end) 3D 복원 시스템을 제안한 것이다. SSIR의 중심적인 설계 아이디어는 다음과 같다.

1. **자기지도 학습 기반의 3D 복원**: 3D ground-truth 없이 오직 2D 실루엣 마스크만 사용하여 3D 표면 메시를 생성한다.
2. **Multi-cycle-consistency 전략**: 가늘고 긴 수술 도구의 특성상 텍스처 정보를 학습하기 어렵다는 점을 극복하기 위해, 다중 사이클 일관성 학습 전략을 도입하여 텍스처 복원 품질을 높였다.
3. **Landmark Consistency 도입**: 두 개의 독립적인 복원 네트워크를 동시에 학습시키고, VGG-19 기반의 특징 맵과 MLP를 통해 랜드마크 일관성을 강제함으로써 복원된 이미지의 품질을 향상시켰다.

## 📎 Related Works

기존의 3D 복원 방식은 3DMM이나 SMPL과 같이 사전 정의된 형태 모델(Morphable model)의 파라미터를 맞추는 방식을 사용했으나, 이는 다양한 객체에 적용하기에 비용과 시간이 너무 많이 소요된다. 딥러닝 기반의 지도 학습(Supervised learning) 방식은 높은 성능을 보이지만, 3D 어노테이션 획득의 어려움으로 인해 2D 감독 기반의 복원 방식이 주목받기 시작했다. 특히 **Differential Rendering (DR)** 기술은 3D 객체의 그래디언트를 2D 이미지로 전파할 수 있게 하여 2D 감독만으로 3D 형태를 학습 가능하게 했다.

그러나 기존의 2D 감독 기반 방식들은 수술 도구 특유의 가늘고 긴 형태를 복원하는 데 어려움이 있으며, 특히 텍스처 정보의 학습이 까다롭다. 본 논문은 이러한 한계를 극복하기 위해 단순한 2D 재구성 오차를 넘어선 사이클 일관성 및 랜드마크 일관성 제약을 추가하여 차별점을 두었다.

## 🛠️ Methodology

### 전체 시스템 구조

SSIR은 입력 이미지와 세그멘테이션 마스크를 통해 카메라, 조명, 형태, 텍스처라는 4가지 3D 속성을 예측하는 4개의 서브 인코더(Sub-encoders)로 구성된다.

- **Camera Encoder ($\mathcal{V}_c$):** 방위각(Azimuth), 고도(Elevation), 거리(Distance)를 예측한다.
- **Shape Encoder ($\mathcal{V}_s$):** 초기 구형 메시(Spherical mesh) $V_0$에 대한 상대적 변화량 $\Delta V$를 예측하여 최종 형태 $V = \Delta V + V_0$를 결정한다.
- **Texture Encoder ($\mathcal{V}_t$):** 2D flow map을 예측하고 Spatial Transformation을 통해 UV 텍스처 맵 $T$를 생성한다.
- **Light Encoder ($\mathcal{V}_l$):** Spherical Harmonics 모델의 계수를 예측하여 조명 조건 $L$을 설정한다.

### Differentiable Rendering

예측된 3D 속성 $\mathcal{A} = [C, L, V, T]$는 미분 가능한 렌더러(Differentiable Renderer) $R$을 통해 2D 이미지 $I_{rn}$과 실루엣 마스크 $M_{rn}$으로 렌더링된다.
$$I_{rn} = R(\mathcal{A}) = R([C, L, V, T])$$

### Multi-cycle-consistency 전략

텍스처 캡처 능력을 향상시키기 위해 두 개의 독립적인 네트워크 $\mathcal{V}_{\theta_1}, \mathcal{V}_{\theta_2}$를 동시에 학습시킨다. 훈련 과정에서 무작위로 선택된 두 샘플 $i, j$의 3D 속성을 보간(Interpolation)하여 새로운 뷰(New view)를 생성한다.
$$\mathcal{A}_{ij} = 0.5 \cdot [(1-\alpha_1) \cdot \mathcal{A}_{i\_1} + \alpha_1 \cdot \mathcal{A}_{j\_1}] + 0.5 \cdot [(1-\alpha_2) \cdot \mathcal{A}_{i\_2} + \alpha_2 \cdot \mathcal{A}_{j\_2}]$$
여기서 $\alpha_1, \alpha_2$는 $(0, 1)$ 사이의 균등 분포에서 샘플링된다. 이렇게 생성된 새로운 뷰 $\mathcal{A}_{ij}$는 2D 정답 데이터가 없으므로, 렌더링 후 다시 인코더에 넣었을 때 원래의 속성이 나와야 한다는 사이클 일관성 제약을 부여한다.

### 손실 함수 (Loss Functions)

전체 손실 함수는 다음과 같이 세 가지 구성 요소의 합으로 정의된다.

1. **2D Level Supervision ($\mathcal{L}_{2D}$):**
    - **이미지 손실 ($\mathcal{L}_{img}$):** 렌더링된 이미지와 입력 이미지의 $L_1$ 거리 측정.
    - **실루엣 손실 ($\mathcal{L}_{sil}$):** 입력 마스크와 렌더링된 마스크 간의 IoU 기반 손실.
    $$\mathcal{L}_{2D} = \lambda_{img}\mathcal{L}_{img} + \lambda_{sil}\mathcal{L}_{sil}$$

2. **3D Level Self-supervision ($\mathcal{L}_{3D}$):**
    - 새로운 뷰에 대해, 예측 $\rightarrow$ 렌더링 $\rightarrow$ 재예측 과정의 속성 차이를 최소화한다.
    $$\mathcal{L}_{3D} = \frac{1}{N} \sum \| \mathcal{V}_\theta(R(\mathcal{V}_\theta(I_i))) - \mathcal{V}_\theta(I_i) \|_1$$

3. **Landmark Consistency ($\mathcal{L}_{LC}$):**
    - VGG-19를 통해 특징 맵을 추출하고, 3D 정점(Vertex)을 2D로 투영하여 각 랜드마크의 인덱스를 정확히 분류할 수 있도록 학습시킨다. 이는 두 네트워크가 동일한 공간적 위치를 인식하게 함으로써 품질을 높인다.
    $$\mathcal{L}_{LC} = -\frac{1}{N} \sum \sum \mathbb{1}_{kn}^\theta y_k \log(\Psi_\phi(f_{kn}^\theta))$$

최종 손실 함수 $\mathcal{L}$은 두 네트워크 $\mathcal{V}_{\theta_1}, \mathcal{V}_{\theta_2}$에 대한 위 세 가지 손실의 평균으로 계산된다.

## 📊 Results

### 실험 설정

- **데이터셋**: 4개의 인공와우(Cochlear Implant, CI) 삽입 수술 비디오를 사용하였다.
- **마스크 생성**: 20프레임의 수동 라벨링 후, MMS(Min-Max Similarity) 알고리즘을 적용하여 나머지 프레임의 실루엣 마스크를 생성하였다.
- **평가 지표**: 렌더링된 결과와 원본의 유사도를 측정하는 **Mask IoU**와 새로운 뷰 생성 능력을 평가하는 **FID (Frechet Inception Distance)**를 사용하였다.

### 주요 결과

SMR(Self-supervised Mesh Reconstruction) 방법과 비교한 결과는 다음과 같다.

| 방법 | Mask IoU | Reconstruction FID $\downarrow$ | Rotation FID $\downarrow$ |
| :--- | :---: | :---: | :---: |
| SMR | 0.868 | 180.3 | 211.4 |
| **SSIR (Ours)** | **0.867** | **121.4** | **126.6** |

- **분석**: Mask IoU는 두 방법이 비슷하게 나타났으나, FID 지표에서 SSIR이 압도적으로 낮은 수치를 기록하였다. 이는 SSIR이 새로운 각도에서 도구를 렌더링했을 때 훨씬 더 사실적인 텍스처와 형태를 복원함을 의미한다.

## 🧠 Insights & Discussion

본 논문은 수술 도구 복원에 있어 가장 큰 걸림돌이었던 '3D 데이터 부족'과 '가늘고 긴 형태의 텍스처 학습 어려움'을 자기지도 학습과 다중 사이클 일관성 전략으로 해결하였다. 특히 카메라 파라미터, 도구 템플릿, 사전 정의된 랜드마크 없이도 학습이 가능하다는 점이 큰 강점이다.

**한계 및 비판적 해석:**

1. **데이터셋 규모**: 단 4개의 비디오 데이터셋만을 사용했다는 점이 실험 결과의 일반성을 확보하기에 부족해 보인다. 더 다양한 수술 도구와 환경에서의 검증이 필요하다.
2. **정량적 형태 평가 부족**: Mask IoU는 2D 투영 결과의 유사도일 뿐, 실제 3D 공간에서의 정점 거리(Chamfer Distance 등)를 측정한 정량적 수치가 제시되지 않았다. 저자들 또한 결론에서 이를 향후 과제로 언급하고 있다.
3. **실제 적용 가능성**: 본 연구는 복원 성능에 집중하고 있으나, 이를 통해 실제 수술실에서 실시간으로 도구의 포즈를 추적하고 좌표를 로컬라이제이션하는 단계까지는 도달하지 못했다.

## 📌 TL;DR

이 논문은 단일 이미지로부터 수술 도구를 3D로 복원하는 **SSIR** 시스템을 제안한다. 3D 정답 없이 2D 실루엣만으로 학습하며, **Multi-cycle-consistency**와 **Landmark Consistency**를 통해 수술 도구 특유의 가늘고 긴 형태와 텍스처를 효과적으로 복원한다. 특히 기존 SMR 대비 새로운 뷰 생성 능력(FID)을 크게 향상시켰으며, 이는 향후 수술 중 도구의 실시간 3D 포즈 추적 및 가이드 시스템 구축에 중요한 기반이 될 것으로 기대된다.
