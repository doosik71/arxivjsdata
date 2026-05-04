# DeepCaps: Going Deeper with Capsule Networks

Jathushan Rajasegaran, Vinoj Jayasundara, Sandaru Jayasekara, Hirunima Jayasekara, Suranga Seneviratne, Ranga Rodrigo (2019)

## 🧩 Problem to Solve

본 논문은 기존 Capsule Network(CapsNet)가 MNIST와 같은 단순한 데이터셋에서는 우수한 성능을 보이지만, CIFAR10과 같이 복잡한 객체를 포함한 데이터셋에서는 CNN(Convolutional Neural Networks)에 비해 성능이 저하되는 문제를 해결하고자 한다.

기존 CapsNet의 구조를 단순히 깊게 쌓는(stacking) 방식은 다음과 같은 세 가지 주요 한계를 가진다:

1. **계산 복잡도**: Dynamic Routing 과정은 연산 비용이 매우 높기 때문에, 다수의 routing 레이어를 배치할 경우 훈련 및 추론 시간이 급격히 증가한다.
2. **학습 저하**: Fully-connected capsule 레이어를 단순히 쌓을 경우, 캡슐의 수가 많아지면 coupling coefficient가 너무 작아져 gradient flow가 억제되고 중간 레이어의 학습이 제대로 이루어지지 않는다.
3. **지역성(Localization) 부재**: 하위 레이어에서 서로 상관관계가 있는 유닛들은 지역적으로 집중되는 경향이 있으나, Fully-connected 구조에서는 이러한 localized routing을 구현할 수 없다.

따라서 본 연구의 목표는 계산 효율성을 높이면서도 깊은 구조를 가질 수 있는 새로운 캡슐 네트워크 아키텍처를 설계하여 복잡한 데이터셋에서의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **3D Convolution 기반의 Dynamic Routing**과 **네트워크의 심층화(Going Deeper)**를 결합하는 것이다.

1. **3D Convolution 기반 Dynamic Routing**: Fully-connected 방식의 routing 대신 3D Convolution을 도입하여 파라미터 수를 획기적으로 줄이고, 지역적(localized) routing을 가능하게 하여 연산 효율성과 학습 능력을 동시에 개선하였다.
2. **심층 캡슐 아키텍처(DeepCaps)**: Skip connection과 Convolutional capsule 레이어를 사용하여 gradient vanishing 문제를 해결하고, 더 깊은 네트워크를 통해 고수준의 특징(high-level features)을 추출할 수 있도록 설계하였다.
3. **Class-independent Decoder**: 기존의 클래스 의존적 디코더 대신 클래스 독립적 디코더를 제안하여 정규화(regularization) 효과를 강화하고, 이미지의 물리적 속성(instantiation parameters)을 제어하고 식별할 수 있는 능력을 부여하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 바탕으로 차별점을 제시한다:

- **심층 네트워크의 gradient 문제**: ResNets와 Highway Networks가 identity connection을 통해 gradient flow를 개선한 점, DenseNets가 모든 레이어를 직접 연결하여 정보 흐름을 최대화한 점을 참고하여 DeepCaps에 skip connection을 도입하였다.
- **Capsule Networks**: Sabour et al. [19]의 Dynamic Routing은 Equivariance를 달성하여 CNN의 pooling으로 인한 invariance 문제를 해결하였다. 하지만 이 모델은 단일 convolution 레이어와 단일 FC capsule 레이어로 구성되어 복잡한 데이터 처리 능력이 부족하다.
- **기타 캡슐 변형**: EM routing [8], HitNet [3], 그리고 2D convolution을 voting 절차에 사용한 SegCaps [15] 등이 있다. 특히 SegCaps는 2D convolution을 사용함으로써 깊이 방향의 캡슐 정보를 혼합시키는 한계가 있으나, 본 논문의 3D convolution 기반 routing은 depth 방향의 stride를 캡슐 차원과 일치시켜 각 캡슐이 개별적으로 voting 하도록 설계하여 차별성을 가진다.

## 🛠️ Methodology

### 1. 3D Convolution Based Dynamic Routing

기존의 fully-connected routing을 대체하기 위해 3D Convolution을 이용한 새로운 routing 메커니즘을 제안한다.

- **과정**:
  - 입력 캡슐 텐서 $\Phi^l \in \mathbb{R}^{(w^l, w^l, c^l, n^l)}$를 단일 채널 텐서 $\tilde{\Phi}^l \in \mathbb{R}^{(w^l, w^l, c^l \times n^l, 1)}$로 변형한다.
  - 이를 $(c^{l+1} \times n^{l+1})$ 개의 3D convolutional kernels $\Psi^l_t$와 컨볼루션 하여 중간 투표 값(intermediate votes) $V$를 생성한다.
  - 3D convolution 연산은 다음과 같이 정의된다:
    $$v_{i,j,k,m} = \sum_{p} \sum_{q} \sum_{r} \tilde{\Phi}^l(i-p, j-q, k-r) \cdot \Psi^l_t(p,q,r)$$
  - 여기서 depth 방향의 stride를 $n^l$로 설정하여 각 캡슐이 개별적으로 투표하도록 한다.

- **Routing 알고리즘**:
  - **Softmax3D**: coupling coefficient $k_{pqrs}$를 계산하기 위해 3D 버전의 softmax를 제안한다.
    $$k_{pqrs} = \frac{\exp(b_{pqrs})}{\sum_{x} \sum_{y} \sum_{z} \exp(b_{xyzs})}$$
  - **Squash3D**: 캡슐 벡터의 길이를 0과 1 사이로 제한하여 엔티티의 존재 확률을 나타낸다.
    $$\hat{S}_{pqr} = \text{squash}_{3D}(S_{pqr}) = \frac{\|S_{pqr}\|^2}{1 + \|S_{pqr}\|^2} \cdot \frac{S_{pqr}}{\|S_{pqr}\|}$$
  - 위 과정을 $i$번(경험적으로 3회) 반복하여 최종 출력 $\Phi^{l+1}$을 얻는다.

### 2. DeepCaps Architecture

DeepCaps는 총 16개의 convolutional capsule 레이어와 1개의 fully-connected capsule 레이어로 구성된다.

- **ConvCaps Layer**: $i=1$일 때 사용하며, 일반적인 convolution과 유사하지만 출력을 squashed 4D 텐서로 생성하여 캡슐 도메인으로 변환한다.
- **ConvCaps3D Layer**: $i > 1$일 때 사용하며, 위에서 설명한 3D convolution 기반 dynamic routing을 적용한다.
- **FlatCaps & FCcaps**: `FlatCaps`는 공간적 관계를 제거하여 텐서를 행렬로 변형하며, `FCcaps`는 학습 가능한 변환 행렬 $W$를 통해 최종 클래스 캡슐로 매핑한다.
- **Skip Connections**: 3개의 ConvCaps 레이어로 구성된 CapsCell 내에서 첫 번째 레이어의 출력을 마지막 레이어에 element-wise addition으로 연결하여 gradient vanishing을 방지하고 저수준 캡슐이 고수준 캡슐로 직접 전달되게 한다.
- **Decoder**: 기존의 FC 디코더 대신 deconvolutional decoder를 사용하여 공간적 관계를 더 잘 재구성하도록 하였다.

### 3. Loss Function

분류를 위해 Margin Loss를 사용한다:
$$L_k = T_k \max(0, m^+ - \|v_k\|)^2 + \lambda(1 - T_k) \max(0, \|v_k\| - m^-)^2$$
여기서 $T_k$는 정답 클래스 여부, $m^+ = 0.9$, $m^- = 0.1$이며, $\lambda$는 훈련 초기 단계의 gradient back propagation을 조절한다.

### 4. Class-Independent Decoder Network

기존 디코더는 모든 클래스의 활동 벡터를 입력으로 받아 클래스 정보가 암시적으로 포함되는 class-dependent 구조였다. 본 논문은 예측된 클래스의 활동 벡터 $P_t \in \mathbb{R}^{1 \times b}$만을 디코더에 입력하는 **class-independent decoder**를 제안한다.

이를 통해 모든 클래스가 동일한 $\mathbb{R}^b$ 공간 내에서 활동 벡터를 공동으로 학습하게 되며, 결과적으로 특정 instantiation parameter가 클래스에 상관없이 동일한 물리적 속성(예: 회전, 두께)을 나타내도록 강제한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: CIFAR10, SVHN, Fashion-MNIST, MNIST.
- **구현**: Keras, Tensorflow, Adam optimizer (initial LR 0.001, 20 epoch마다 절반으로 감소).
- **데이터 증강**: CIFAR10과 SVHN은 $64 \times 64 \times 3$으로 리사이징하여 처리하였다.

### 2. 분류 성능

DeepCaps는 기존의 캡슐 네트워크 모델들을 압도하는 성능을 보였다.

| 모델 | CIFAR10 | SVHN | F-MNIST | MNIST |
| :--- | :---: | :---: | :---: | :---: |
| Sabour et al. [19] | 89.40% | 95.70% | 93.60% | 99.75% |
| **DeepCaps** | **91.01%** | **97.16%** | **94.46%** | **99.72%** |
| **DeepCaps (7-ensemble)** | **92.74%** | **97.56%** | **94.73%** | - |

- **파라미터 효율성**: CIFAR10 데이터셋 기준, DeepCaps는 7.22M 개의 파라미터를 사용하는 반면, CapsNet [19]는 22.48M 개를 사용한다. 즉, 파라미터 수를 약 68% 줄이면서도 성능은 더 높였다.
- **추론 속도**: NVIDIA V100 GPU 기준, $64 \times 64$ 이미지를 처리하는 DeepCaps의 추론 시간은 1.38ms로, $32 \times 32$ 이미지를 처리하는 CapsNet(2.86ms)보다 훨씬 빠르다.

### 3. 디코더 재구성 결과

Class-independent decoder를 통해 특정 instantiation parameter가 모든 클래스에서 동일한 물리적 변형을 일으킴을 확인하였다.

- **분석**: 분산(variance)이 높은 파라미터는 회전, 수직 연장, 두께와 같은 **전역적 변형(global variations)**을 일으키고, 분산이 낮은 파라미터는 **지역적 변형(localized changes)**을 일으키는 경향이 있다.
- 예를 들어, 28번째 파라미터는 모든 클래스에서 수직 연장(vertical elongation)을, 1번째 파라미터는 두께(thickness)를 제어하는 것으로 관찰되었다.

## 🧠 Insights & Discussion

### 강점

본 논문은 캡슐 네트워크의 고질적인 문제였던 연산 복잡도와 깊은 구조 설계의 어려움을 3D Convolution과 skip connection이라는 CNN의 성공 방정식을 접목하여 해결하였다. 특히 파라미터 수를 크게 줄이면서도 CIFAR10과 같은 복잡한 데이터셋에서 SOTA급 캡슐 네트워크 성능을 달성한 점이 고무적이다.

### 한계 및 논의사항

- **CNN과의 비교**: 캡슐 네트워크 도메인 내에서는 최상위 성적을 거두었으나, DenseNet이나 ResNet과 같은 최신 CNN 모델의 성능에는 여전히 미치지 못한다.
- **파라미터 독립성**: 저자들은 instantiation parameter 공간이 반드시 orthogonal(직교)하지는 않아 일부 파라미터가 공통 속성을 공유할 수 있음을 언급하였다. 향후 연구에서 이러한 파라미터 간의 상관관계를 제거하는 방법이 논의될 필요가 있다.
- **데이터셋 확장**: 현재는 소규모 벤치마크 데이터셋 위주로 검증되었으며, ImageNet과 같은 대규모 데이터셋으로의 확장 가능성을 확인해야 한다.

## 📌 TL;DR

DeepCaps는 **3D Convolution 기반의 새로운 Dynamic Routing**과 **Skip Connection을 포함한 심층 구조**를 도입하여, 기존 캡슐 네트워크의 높은 연산 비용과 학습 저하 문제를 해결한 모델이다. 이를 통해 CIFAR10, SVHN, Fashion-MNIST에서 기존 캡슐 모델 대비 성능을 크게 향상시켰으며, 파라미터 수는 68% 감소시켰다. 또한, **클래스 독립적 디코더**를 통해 이미지의 물리적 속성을 클래스에 상관없이 일관되게 제어할 수 있음을 증명하여, 향후 정밀한 데이터 생성 및 분석 연구에 기여할 가능성이 크다.
