# Self-supervised Object-Centric Learning for Videos

Görkay Aydemir, Weidi Xie, Fatma Güney (2023)

## 🧩 Problem to Solve

본 논문은 실제 환경의 비디오 시퀀스에서 정답 라벨 없이 여러 객체를 분할(Multi-object segmentation)하는 문제를 해결하고자 한다. 기존의 비디오 객체 중심 학습(Object-centric learning) 방식들은 주로 합성 데이터셋에서 우수한 성능을 보였으나, 실제 세계의 복잡한 시나리오에서는 성능이 크게 저하되는 한계가 있었다.

특히, 기존 연구들은 객체 분할을 돕기 위해 깊이 지도(Depth map)나 광학 흐름(Optical flow)과 같은 추가적인 모달리티(Additional modality)에 의존하는 경우가 많았다. 그러나 광학 흐름은 정적인 객체나 급격한 변형이 일어나는 객체에서 오차가 발생하며, 깊이 지도는 일반적인 비디오에서 쉽게 얻을 수 없거나 저조도 환경에서 추정이 어렵다는 문제가 있다. 따라서 본 논문의 목표는 추가적인 모달리티나 약한 감독(Weak supervision) 없이, 오직 RGB 비디오만으로 객체를 발견하고 추적하며 분할할 수 있는 완전 비지도 학습(Fully unsupervised) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **축 방향 공간-시간 슬롯 어텐션(Axial spatial-temporal slot attentions)**과 **마스킹된 특징 재구성(Masked feature reconstruction)**을 결합하는 것이다.

구체적으로는, 각 프레임 내에서 픽셀들을 공간적으로 그룹화하여 슬롯(Slot)에 바인딩하고, 이후 시간 축을 따라 이 슬롯들을 연결함으로써 시간적 문맥을 가진 객체 표현을 학습한다. 또한, 픽셀 공간이 아닌 고수준의 시맨틱 특징 공간(High-level semantic feature space)에서 마스킹된 특징을 재구성하는 목표 함수를 사용하여, 모델이 객체의 핵심적인 구조적 정보를 학습하도록 유도한다. 마지막으로, 고정된 슬롯 수로 인해 발생하는 과분할(Over-segmentation) 문제를 해결하기 위해 유사도 기반의 슬롯 병합(Slot merging) 전략을 도입하였다.

## 📎 Related Works

**1. Object-centric Learning**
최근 슬롯 어텐션(Slot Attention)을 이용해 입력 신호를 잠재 공간의 슬롯으로 분해하여 객체를 표현하는 방식이 주목받고 있다. 그러나 대부분 합성 데이터에 치중되어 있으며, 실제 데이터에 적용하기 위해서는 3D 구조 정보나 광학 흐름 같은 추가 가이드가 필요했다. 본 논문은 이러한 명시적 가이드 없이 DINO와 같은 자가 지도 학습 모델의 특징 공간에서 재구성을 수행하는 전략을 취한다.

**2. Object Localization from DINO Features**
DINOv2와 같은 모델의 특징이 객체 경계를 매우 잘 포착한다는 점이 알려져 있다. 기존에는 단순한 군집화(Clustering)나 그래프 파티셔닝을 통해 객체를 찾았으나, 본 논문은 이를 한 단계 발전시켜 학습 가능한 슬롯 기반 구조에 통합하였다.

**3. Video Object Segmentation (VOS)**
비지도 VOS 연구들은 주로 움직임(Motion) 정보에 의존하여 객체를 구분한다. 하지만 본 연구는 명시적인 움직임 정보(Optical flow 등)를 사용하지 않고, 시간적 슬롯 표현을 통해 시간적 일관성을 확보함으로써 움직임 추정 오류로 인한 성능 저하를 방지한다.

## 🛠️ Methodology

### 전체 파이프라인

모델 $\Phi$는 다음과 같은 세 가지 핵심 구성 요소로 이루어진다.
$$m_t = \Phi(V_t; \Theta) = \Phi_{\text{vis-dec}} \circ \Phi_{\text{st-bind}} \circ \Phi_{\text{vis-enc}}(V_t)$$
여기서 $V_t$는 타겟 프레임을 중심으로 한 비디오 클립이며, $\Phi_{\text{vis-enc}}$는 시각적 인코더, $\Phi_{\text{st-bind}}$는 공간-시간 바인딩 모듈, $\Phi_{\text{vis-dec}}$는 시각적 디코더이다.

### 1. Visual Encoder ($\Phi_{\text{vis-enc}}$)

- **Token Drop**: 메모리 효율성과 정규화를 위해 입력 패치 중 상당 부분을 무작위로 제거한다. 이는 MAE(Masked Autoencoder) 방식의 학습을 가능하게 하여 모델이 부분적인 관찰만으로 고수준 구조를 학습하도록 강제한다.
- **Feature Extraction**: 가중치가 고정된(Frozen) DINOv2 ViT-B/14를 사용하여 특징을 추출한다. 추출된 특징 $f_\tau \in \mathbb{R}^{N' \times D}$는 이후 슬롯 바인딩의 입력이 된다.

### 2. Spatial-Temporal Binding ($\Phi_{\text{st-bind}}$)

이 단계는 두 단계의 축 방향(Axial) 어텐션으로 구성된다.

- **Spatial Binding ($\psi_{s-bind}$)**: 각 프레임 내에서 독립적으로 수행된다. Invariant Slot Attention (ISA)을 사용하여 픽셀들을 $K$개의 슬롯으로 그룹화한다. 모든 프레임은 동일한 초기화 벡터 $Z^\tau$를 공유하여, 동일한 인덱스의 슬롯이 서로 다른 프레임에서도 동일한 객체에 바인딩될 가능성을 높인다.
- **Temporal Binding ($\psi_{t-bind}$)**: 공간 바인딩을 통해 얻은 슬롯들을 시간 축으로 연결한다. 동일한 인덱스의 슬롯들에 대해 Transformer Encoder를 적용하여, 특정 객체의 과거와 미래 정보를 참조함으로써 보다 강건한 시간적 표현 $c \in \mathbb{R}^{K \times D_{\text{slot}}}$를 생성한다.

### 3. Visual Decoder ($\Phi_{\text{vis-dec}}$)

- **Slot Merging ($\psi_{merge}$)**: 고정된 슬롯 수 $K$는 실제 객체 수와 다를 수 있어 과분할 문제가 발생한다. 이를 해결하기 위해 코사인 유사도 기반의 계층적 군집화(Agglomerative Clustering)를 사용하여 유사한 슬롯들을 병합함으로써 최적의 객체 수 $K_t$를 동적으로 결정한다.
- **Decoder ($\psi_{dec}$)**: 병합된 슬롯 $c'$를 Spatial Broadcast Decoder에 입력하여 원래의 DINOv2 특징 맵 $y$를 재구성하고, 그 과정에서 부산물로 객체별 세그멘테이션 마스크 $m$을 생성한다.

### 훈련 목표 및 손실 함수

모델은 마스킹된 입력으로부터 원래의 DINOv2 특징을 복원하는 방식으로 학습된다. 손실 함수는 원래의 특징 맵 $\phi_{DINO}(v_t)$와 재구성된 특징 맵 $y$ 사이의 $L_2$ 거리(Mean Squared Error)로 정의된다.
$$\mathcal{L} = \|\phi_{DINO}(v_t) - y\|^2$$

## 📊 Results

### 실험 설정

- **데이터셋**: MOVi-E (합성), DAVIS17 (실제), Youtube-VIS 2019 (실제)
- **지표**:
  - FG-ARI (Foreground Adjusted Rand Index): 객체 군집화 품질 측정.
  - mIoU (mean Intersection-over-Union): 분할 정확도 측정.
- **기준선(Baseline)**: DINOv2 + Clustering, OCLR (광학 흐름 기반 감독 학습 모델)

### 주요 결과

- **합성 데이터 (MOVi-E)**: 본 모델은 FG-ARI 80.8을 기록하며 기존의 모든 비지도 방식(MoTok, DINOSAUR 등)을 크게 상회하는 성능을 보였다.
- **실제 데이터 (Youtube-VIS 2019)**: mIoU 45.32, FG-ARI 29.11을 달성하였다. 특히 광학 흐름을 사용하는 OCLR과 비교했을 때, 복잡한 비디오(YTVIS19)에서 OCLR보다 훨씬 우수한 성능을 보였는데, 이는 광학 흐름 추정의 오류가 없는 본 모델의 강점 때문이다.
- **실제 데이터 (DAVIS17)**: mIoU 30.16을 기록하였다. OCLR이 mIoU 측면에서 약간 높았으나, 이는 DAVIS17의 단순한 특성상 광학 흐름이 잘 작동했기 때문이며, 객체 식별 능력(FG-ARI) 면에서는 본 모델이 경쟁력을 가졌다.

### 절제 연구 (Ablation Study)

- **구성 요소 영향**: Temporal Binding($\psi_{t-bind}$)과 Spatial Binding($\psi_{s-bind}$)을 모두 포함했을 때 성능이 가장 높았으며, 이는 공간적 그룹화와 시간적 연관성 학습이 모두 필수적임을 시사한다.
- **슬롯 수와 병합**: 슬롯 수를 늘릴 때 병합($\psi_{merge}$)을 적용하지 않으면 과분할로 인해 성능이 하락하지만, 병합을 적용하면 더 많은 슬롯을 사용하여 더 정교한 분할이 가능해진다.
- **토큰 드롭 비율**: 드롭 비율 0.5에서 최적의 성능과 메모리 효율의 균형을 보였으며, 적절한 드롭은 정규화 효과를 제공한다.

## 🧠 Insights & Discussion

**강점**
본 연구는 추가적인 모달리티(깊이, 흐름) 없이 오직 RGB 비디오만으로 실제 환경에서 다중 객체 분할이 가능함을 입증한 첫 번째 완전 비지도 모델이다. 특히 자가 지도 학습으로 사전 훈련된 DINOv2의 강력한 특징 공간을 활용함으로써, 복잡한 실제 영상에서도 객체 중심의 표현을 성공적으로 학습하였다.

**한계 및 미해결 질문**

1. **경계 정확도**: 특징 추출이 패치 단위(Patch-level)로 이루어지기 때문에 픽셀 수준의 정밀한 경계(Pixel-accurate boundary)를 얻는 데 한계가 있다.
2. **인접 객체 문제**: 매우 가까이 붙어 있는 동일 클래스의 객체들을 하나의 슬롯으로 묶어버리는 경향이 있다.
3. **미분 불가능한 병합**: 현재 사용된 계층적 군집화 기반의 슬롯 병합 방식은 미분 불가능(Non-differentiable)하므로, 향후 미분 가능한 방식으로 객체 수를 동적으로 조절하는 연구가 필요하다.

**비판적 해석**
본 모델은 '특징 재구성'이라는 간접적인 목표를 통해 객체를 분리한다. 이는 직접적인 세그멘테이션 지도 없이도 객체를 찾을 수 있게 하지만, 결국 DINOv2가 학습한 시맨틱 정보에 크게 의존한다. 따라서 사전 훈련 모델의 성능이 전체 시스템의 상한선(Upper bound)을 결정짓는 구조이며, 이를 극복하기 위해서는 비디오 특유의 시간적 역동성을 더 직접적으로 학습에 활용할 방법이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 추가적인 센서 데이터나 라벨 없이 RGB 비디오만으로 다중 객체를 분할하는 비지도 학습 모델 **SOLV**를 제안한다. DINOv2 특징 공간에서의 마스킹된 재구성 학습, 공간-시간 축 기반의 슬롯 바인딩, 그리고 과분할 방지를 위한 슬롯 병합 전략을 통해 실제 비디오 데이터셋(Youtube-VIS 2019 등)에서 기존 SOTA 모델들을 능가하는 성능을 달성하였다. 이 연구는 실제 환경의 비디오를 시맨틱 단위의 객체들로 분해하는 객체 중심 학습의 가능성을 확장했으며, 향후 자율 주행이나 로봇 제어를 위한 비지도 장면 이해 연구에 중요한 기초가 될 것으로 보인다.
