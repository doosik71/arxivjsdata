# Biologically Inspired Oscillating Activation Functions Can Bridge the Performance Gap between Biological and Artificial Neurons

**Authors: Not explicitly mentioned in the provided text (Preprint submitted May 11, 2023)**

## 🧩 Problem to Solve

본 논문은 생물학적 뉴런과 인공 뉴런 사이의 현저한 성능 격차, 특히 개별 뉴런의 표현 능력(representative power) 차이를 해결하고자 한다. 기존의 인공 뉴런은 단일 하이퍼플레인(single hyperplane)만을 결정 경계로 가지기 때문에, 개별 뉴런 수준에서는 선형 분류만을 수행할 수 있다. 이로 인해 XOR 문제와 같이 선형 분리가 불가능한 데이터를 학습하기 위해서는 반드시 다층 네트워크(multilayer network) 구조가 필요하다.

반면, 최근 연구에 따르면 인간의 신피질 피라미드 뉴런(neocortical pyramidal neurons)은 단일 뉴런만으로도 XOR 함수를 학습할 수 있다는 사실이 밝혀졌다. 본 논문의 목표는 이러한 생물학적 특성에서 영감을 얻어, 단일 뉴런이 XOR 문제를 해결할 수 있도록 하는 '진동 활성화 함수(Oscillating Activation Functions)'를 제안하고, 이것이 딥러닝 모델의 학습 속도와 정확도, 그리고 네트워크 효율성에 미치는 영향을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 활성화 함수가 여러 개의 제로 포인트(zero points)를 갖도록 설계하여, 단일 뉴런이 결정 경계에서 여러 개의 하이퍼플레인을 가질 수 있게 하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **새로운 활성화 함수 제안**: 인간의 피라미드 뉴런에서 영감을 받은 4가지의 진동 활성화 함수인 SQU(Shifted Quadratic Unit), NCU(Non-Monotonic Cubic Unit), SSU(Shifted Sinc Unit), DSU(Decaying Sine Unit)를 제안한다.
2. **단일 뉴런의 XOR 해결 능력 입증**: 제안된 함수들이 'XOR Property'를 가지고 있음을 보임으로써, 별도의 특징 공학(feature engineering) 없이 단일 뉴런만으로 XOR 문제를 해결할 수 있음을 증명한다.
3. **성능 및 효율성 향상**: 진동 활성화 함수가 기존의 단조(monotonic) 함수나 단일 제로 포인트 함수보다 그래디언트 흐름(gradient flow)을 개선하여 학습 속도를 높이고, 더 적은 수의 레이어로도 높은 분류 성능을 낼 수 있음을 실험적으로 보여준다.

## 📎 Related Works

기존의 인공 신경망(ANN)에서는 다양한 활성화 함수가 사용되어 왔다.

- **포화 활성화 함수(Saturating Activation Functions)**: Sigmoid나 Tanh와 같은 함수들은 입력값이 일정 범위를 벗어나면 미분값이 0에 가까워지는 포화 현상이 발생하며, 이는 심층 신경망에서 Vanishing Gradient Problem(기울기 소멸 문제)을 야기한다.
- **비포화 활성화 함수(Non-saturating Activation Functions)**: ReLU 및 그 변형들(Leaky ReLU, PReLU, GELU, ELU, SiLU, Mish, Swish 등)은 양수 영역에서 포화되지 않아 심층 네트워크 학습을 가능하게 했다.

**기존 방식과의 차별점**:
대부분의 기존 활성화 함수는 단조 증가하거나(monotonic), 원점에서 단 하나의 제로 포인트만을 가지는 형태이다. 따라서 단일 뉴런의 결정 경계는 항상 단일 하이퍼플레인으로 제한된다. 본 논문은 이와 달리 **비단조(non-monotonic)**이며 **다수의 제로 포인트**를 가진 진동 함수를 도입함으로써, 단일 뉴런의 표현력을 근본적으로 확장했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 결정 경계와 XOR Property
뉴런의 활성화 값이 $a = g(w^T x + b)$일 때, 결정 경계는 $g(z) = 0$을 만족하는 지점들의 집합이다. 만약 $g(z)$가 $z=0$에서만 0을 갖는다면 결정 경계는 단일 하이퍼플레인이 되지만, 제로 포인트가 여러 개라면 단일 뉴런이 여러 개의 하이퍼플레인을 가질 수 있어 XOR 문제를 해결할 수 있다. 이를 'XOR Property'라고 정의한다.

### 2. 활성화 함수의 필수 조건
유용한 활성화 함수가 되기 위해 본 논문은 다음의 조건을 제시한다.
- **원점 근처의 선형성**: 가중치가 작게 초기화되었을 때 학습이 빠르게 시작되도록 원점 근처에서 identity function($f(z)=z$)과 유사해야 하며, $g'(0) \approx 1$을 만족해야 한다.
- **원점 값**: $g(0) = 0$이어야 한다.

### 3. 제안하는 진동 활성화 함수
본 논문에서 제안 및 분석한 주요 함수들은 다음과 같다.

- **SQU (Shifted Quadratic Unit)**:
  $$f_{18}(z) = z^2 + z$$
- **NCU (Non-Monotonic Cubic Unit)**:
  $$f_{19}(z) = z - z^3$$
- **SSU (Shifted Sinc Unit)**:
  $$f_{21}(z) = \pi \text{sinc}(z - \pi)$$
  (단, $\text{sinc}(z)$는 $z=0$일 때 1, 그 외에는 $\frac{\sin(z)}{z}$로 정의됨)
- **DSU (Decaying Sine Unit)**:
  $$f_{23}(z) = \frac{\pi}{2}(\text{sinc}(z - \pi) - \text{sinc}(z + \pi))$$
- **GCU (Growing Cosine Unit)**:
  $$f_{22}(z) = z \cos(z)$$

### 4. 학습 절차
- **최적화 알고리즘**: Adam 및 RMSprop 사용.
- **손실 함수**: 다중 클래스 분류 문제이므로 Sparse Categorical Cross-Entropy Loss를 사용한다.
- **평가 방식**: 가중치 초기화에 따른 영향을 최소화하기 위해 5회 독립 실험의 평균 정확도를 측정한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: CIFAR-10, CIFAR-100, Imagenette.
- **비교 대상**: 총 23가지의 활성화 함수 (Signum, Identity, Sigmoid, Tanh, ReLU, GELU, Mish, Swish 등 포함).
- **지표**: Mean Test Accuracy, Mean Test Loss.

### 2. 정량적 결과
- **정확도 순위**: 모든 벤치마크에서 진동 활성화 함수들이 상위 5위권을 대부분 차지하였다.
  - CIFAR-10: 상위 5개 중 4개가 진동 함수.
  - CIFAR-100: 상위 5개 모두가 진동 함수.
  - Imagenette: 상위 5개 중 4개가 진동 함수.
- **학습 속도**: Training Curve 분석 결과, 진동 활성화 함수를 사용한 네트워크가 기존 함수들보다 일관되게 빠르게 수렴하는 양상을 보였다.

### 3. 레이어 수에 따른 성능 변화
실험 결과, 컨볼루션 레이어의 수가 증가함에 따라 진동 활성화 함수의 정확도 향상 폭이 기존 함수들보다 훨씬 컸다. 이는 단일 뉴런의 표현력이 높기 때문에, 동일한 레이어 수에서 더 높은 정확도를 달성하거나, 반대로 더 적은 수의 뉴런/레이어로도 동일한 성능을 낼 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
- **표현력의 확장**: 본 연구는 활성화 함수의 수학적 성질(다중 제로 포인트)을 변경하는 것만으로도 개별 뉴런의 계산 능력을 획기적으로 높일 수 있음을 보여주었다. 이는 복잡한 특징 추출을 위해 무작정 네트워크를 깊게 쌓는 대신, 뉴런 개별의 효율성을 높이는 방향성을 제시한다.
- **그래디언트 흐름 개선**: 제안된 함수들은 비포화(non-saturating) 특성을 가지며, 특히 NCU와 같은 함수는 기존 함수들보다 큰 미분값을 가져 역전파(backpropagation) 학습 속도를 가속화한다.

### 2. 한계 및 논의사항
- **하이퍼파라미터 의존성**: 실험에서 최적의 하이퍼파라미터 설정을 사용했다고 명시되어 있으나, 진동 함수들의 주기나 진폭이 모델의 안정성에 어떤 영향을 미치는지에 대한 세부적인 분석은 부족하다.
- **일반화 가능성**: 이미지 분류 작업 외에 다른 도메인(NLP 등)에서도 동일한 효율성이 나타날지는 추가적인 연구가 필요하다.
- **추측 배제**: 논문에서는 $z^2 \cos(z)$와 같이 원점에서 미분값이 0인 함수가 성능이 낮게 나왔음을 통해 원점 선형성의 중요성을 강조하고 있다.

## 📌 TL;DR

본 논문은 생물학적 피라미드 뉴런이 단일 세포만으로 XOR 문제를 해결한다는 점에서 영감을 받아, **다수의 제로 포인트를 갖는 진동 활성화 함수(SQU, NCU, SSU, DSU 등)**를 제안하였다. 이 함수들은 단일 뉴런이 여러 개의 결정 하이퍼플레인을 가질 수 있게 하여 표현력을 극대화하며, 결과적으로 **CIFAR-10/100 및 Imagenette 벤치마크에서 기존의 ReLU, Mish, Swish 등을 능가하는 정확도와 빠른 학습 속도**를 보였다. 이 연구는 인공 뉴런의 기본 단위를 개선함으로써 생물학적 뉴런과의 성능 격차를 줄이고, 더 효율적인(더 적은 레이어를 사용하는) 신경망 설계의 가능성을 열었다는 점에서 중요한 의의를 가진다.