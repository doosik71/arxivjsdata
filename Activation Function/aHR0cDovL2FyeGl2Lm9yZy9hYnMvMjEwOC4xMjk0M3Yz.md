# Growing Cosine Unit: A Novel Oscillatory Activation Function That Can Speedup Training and Reduce Parameters in Convolutional Neural Networks

Mathew Mithra Noel, Arunkumar L, Advait Trivedi, Praneet Dutta (2023)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델, 특히 합성곱 신경망(Convolutional Neural Networks, CNNs)에서 사용되는 활성화 함수(Activation Function)의 근본적인 한계를 해결하고자 한다.

기존의 대다수 활성화 함수(ReLU, Sigmoid, Swish, Mish 등)는 단조 증가(monotonically increasing)하거나 비진동성(non-oscillatory) 특성을 가진다. 이러한 특성으로 인해 개별 뉴런은 오직 하나의 선형 결정 경계(linear decision boundary), 즉 하나의 하이퍼플레인(hyperplane)만을 생성할 수 있다. 결과적으로 XOR 문제와 같이 선형적으로 분리 불가능한 문제를 해결하기 위해서는 반드시 다층 구조의 네트워크나 여러 개의 뉴런이 필요하며, 이는 네트워크의 파라미터 수 증가와 계산 복잡도 상승으로 이어진다.

논문의 목표는 진동성(oscillatory) 활성화 함수를 도입하여 단일 뉴런이 비선형 결정 경계를 가질 수 있게 함으로써, 네트워크의 크기를 줄이고 학습 속도를 향상시키며, 모델의 표현 능력을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **Growing Cosine Unit (GCU) 제안**: $C(z) = z \cos(z)$로 정의되는 새로운 진동성 활성화 함수를 제안하였다.
2. **단일 뉴런 기반 XOR 문제 해결**: 전통적으로 최소 3개의 뉴런이 필요했던 XOR 문제를 단 하나의 GCU 뉴런만으로 해결할 수 있음을 입증하였다.
3. **이론적 한계 증명**: 엄격하게 단조 증가하는 활성화 함수나 항등 함수 $I(z) = z$와 부호 동등성(sign-equivalence)을 가진 함수는 단일 뉴런으로 XOR 문제를 해결할 수 없다는 두 가지 정리(Proposition)를 제시하였다.
4. **CNN 성능 향상 및 효율성 입증**: CIFAR-10, CIFAR-100, Imagenette 데이터셋에서 GCU가 ReLU, Swish, Mish보다 높은 정확도와 빠른 수렴 속도를 보임을 실험적으로 증명하였으며, 특히 Swish나 Mish보다 계산 비용이 저렴함을 확인하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 활성화 함수들의 특성과 한계를 설명한다.

- **Sigmoidal functions**: 미분 가능하며 이진 결정으로 해석 가능하지만, 입력값이 0에서 멀어질 때 기울기가 소실되는 Vanishing Gradient 문제로 인해 깊은 네트워크 학습이 어렵다.
- **ReLU (Rectified Linear Unit)**: 기울기 소실 문제를 완화하여 딥러닝의 이정표가 되었으나, 입력값이 음수일 때 기울기가 0이 되어 뉴런이 더 이상 업데이트되지 않는 'Neuron Death' 문제와 출력값의 평균이 0보다 커지는 Bias Shift 문제가 존재한다.
- **Swish 및 Mish**: 최근 제안된 비단조(non-monotonic) 함수들로 우수한 성능을 보이지만, 본 논문의 분석에 따르면 이들은 여전히 $\text{sign}(g(z)) = \text{sign}(z)$를 만족하는 부호 동등성을 가지므로, 단일 뉴런의 결정 경계가 여전히 하나의 하이퍼플레인으로 제한된다는 한계가 있다.

## 🛠️ Methodology

### 1. Growing Cosine Unit (GCU) 정의

제안된 GCU 활성화 함수는 다음과 같이 정의된다.
$$C(z) = z \cos(z)$$

### 2. 결정 경계의 확장

단일 뉴런의 결정 경계 $B$는 활성화 함수 $g$의 결과가 0이 되는 지점들의 집합으로 정의된다.
$$B = \{x \in \mathbb{R}^n : g(w^T x + b) = 0\}$$

기존의 $g(z)=0 \iff z=0$인 함수들은 결정 경계가 단일 하이퍼플레인 $w^T x + b = 0$이 되지만, GCU는 $\cos(z)$ 성분으로 인해 무수히 많은 제로 포인트(zeros)를 가진다. 따라서 GCU 뉴런의 결정 경계는 다음과 같이 균일한 간격을 가진 무수한 평행 하이퍼플레인의 집합이 된다.
$$w^T x + b = \frac{\pi}{2} + n\pi \quad (n \in \mathbb{Z})$$
이로 인해 입력 공간은 서로 다른 클래스가 교대로 배치된 '평행한 띠(parallel strips)' 형태로 분할되며, 이를 통해 단일 뉴런만으로도 XOR와 같은 복잡한 분류 문제를 해결할 수 있다.

### 3. 계산 복잡도 분석

GCU는 하나의 초월 함수(transcendental function) 호출과 한 번의 곱셈만을 사용한다. 이는 3개의 초월 함수 호출과 곱셈을 사용하는 Mish보다 계산 비용이 현저히 낮으며, Leaky ReLU보다는 약간 높지만 최신 상태의 Swish나 Mish보다는 훨씬 효율적이다.

### 4. 학습 절차

- **최적화 도구**: RMSprop Optimizer 사용.
- **손실 함수**: Categorical Cross Entropy (Softmax head).
- **가중치 초기화**: Xavier Uniform Initializer 사용.
- **적용 범위**: GCU는 계산 비용이 ReLU보다 높으므로 합성곱 층(Convolution layers)에만 적용하고, 전결합 층(Dense layers)에는 ReLU를 유지하는 하이브리드 구조를 사용하였다.

## 📊 Results

### 1. 정량적 성능 평가

CIFAR-10, CIFAR-100, Imagenette 데이터셋에 대해 실험한 결과, GCU를 합성곱 층에 적용했을 때 가장 높은 정확도를 기록하였다.

- **Imagenette (VGG-16 backbone)**: GCU 적용 모델이 모든 ReLU 기반 아키텍처보다 약 $7\%$ 더 높은 Top-1 정확도를 보였다.
- **CIFAR-10 & CIFAR-100**: ReLU, Swish, Mish 대비 전반적으로 높은 정확도를 보였으며, 특히 학습 곡선에서 더 빠른 수렴 속도를 나타냈다.

### 2. 정성적 분석 및 시각화

- **필터 시각화**: VGG-16의 각 층에서 필터 출력을 시각화한 결과, GCU는 ReLU보다 객체(예: 새)의 특징을 더 명확하고 강하게(더 큰 출력값) 검출하는 경향을 보였다.
- **기울기 흐름(Gradient Flow)**: 각 층의 기울기 RMS(Root Mean Square) 값을 분석한 결과, ReLU와 Leaky ReLU는 기울기 값이 격하게 진동하는 반면, GCU는 진동이 훨씬 적고 안정적인 흐름을 보였다. 이는 GCU가 기울기가 0이 되는 구간이 매우 국소적인 지점에만 존재하여 Vanishing Gradient 문제를 완화하기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. GCU의 특성 및 강점

GCU는 입력값 $z$가 작을 때 $z \cos(z) \approx z$가 되어 선형 활성화 함수처럼 동작한다. 이는 초기 가중치가 작을 때 네트워크가 선형 분류기처럼 작동하게 하여 정규화(regularizing) 효과를 주고 과적합을 방지한다. 또한, 큰 입력값에 대해서는 진동하며 무제한(unbounded) 함수로 동작하여 표현력을 높인다.

### 2. 이론적 함의

본 논문은 활성화 함수의 '부호 동등성' 개념을 도입하여, 왜 기존의 비단조 함수인 Swish나 Mish조차도 단일 뉴런으로는 XOR 문제를 풀 수 없는지를 수학적으로 증명하였다. 이는 단순히 함수 형태를 조금 바꾸는 것이 아니라, '진동성'이라는 근본적인 특성을 도입해야 뉴런의 표현 능력을 획기적으로 높일 수 있음을 시사한다.

### 3. 한계 및 논의사항

- GCU가 ReLU보다 계산 비용이 높기 때문에 모든 층에 적용하기보다는 합성곱 층에 선택적으로 적용하는 전략을 취했다.
- 실험에서 사용된 데이터셋 외에 더 거대한 규모의 데이터셋이나 다른 아키텍처(예: Transformer)에서의 효과는 명시적으로 다루지 않았다.

## 📌 TL;DR

본 논문은 단일 뉴런이 하나의 선형 경계만 가질 수 있다는 기존의 한계를 깨기 위해 진동성 활성화 함수인 **Growing Cosine Unit ($C(z) = z \cos(z)$)**을 제안하였다. GCU는 단일 뉴런만으로 XOR 문제를 해결할 수 있을 만큼 강력한 표현력을 가지며, CNN의 합성곱 층에 적용했을 때 ReLU, Swish, Mish보다 **더 높은 정확도, 빠른 수렴 속도, 그리고 안정적인 기울기 흐름**을 보였다. 특히 계산 효율성이 Swish/Mish보다 뛰어나 실제 적용 가능성이 높으며, 향후 생물학적 뉴런의 진동 특성을 모사한 더 발전된 활성화 함수 연구의 토대가 될 것으로 기대된다.
