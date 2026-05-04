# Activation Adaptation in Neural Networks

Farnoush Farhadi, Vahid Partovi Nia, Andrea Lodi (2019)

## 🧩 Problem to Solve

신경망 아키텍처를 설계할 때 은닉층의 활성화 함수(Activation Function) 선택은 매우 중요한 결정 사항이다. 일반적으로 신경망은 가중치(Weight)와 편향(Bias) 파라미터를 학습하지만, 활성화 함수는 Sigmoid, ReLU와 같이 사전에 고정된 함수를 사용한다. 이론적으로는 네트워크가 충분히 깊고 넓다면 활성화 함수의 종류와 무관하게 임의의 복잡한 함수를 근사할 수 있다고 알려져 있으나, 실제 환경에서는 활성화 함수의 선택이 학습된 표현(Representation)과 최종 예측 성능에 상당한 영향을 미친다.

그동안 네트워크 하이퍼파라미터 튜닝에 대한 많은 연구가 진행되었음에도 불구하고, 개별 뉴런에 가장 적합한 활성화 함수를 어떻게 선택하고 최적화할 것인가에 대한 연구는 부족했다. 본 논문의 목표는 학습 과정에서 데이터가 직접 활성화 함수의 형태를 추정할 수 있도록 하는 유연한 활성화 함수를 개발하고, 이를 역전파(Back-propagation) 과정에 통합하여 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 활성화 함수를 누적 분포 함수(Cumulative Distribution Function, CDF)의 관점에서 재정의하고, 여기에 학습 가능한 형상 파라미터(Shape Parameter) $\alpha$를 도입하는 것이다.

기존의 가중치가 입력의 스케일을 조절하고 편향이 활성화의 중심을 잡는 역할을 했다면, 본 연구에서 제안하는 형상 파라미터 $\alpha$는 뉴런의 곡률(Curvature)을 직접 조정하여 각 뉴런이 데이터에 최적화된 활성화 함수를 스스로 가질 수 있게 한다. 이를 통해 비대칭성(Asymmetry)과 매끄러움(Smoothness)이라는 두 가지 핵심 속성을 데이터 기반으로 최적화할 수 있다.

## 📎 Related Works

기존에는 ReLU의 한계를 극복하기 위해 Leaky ReLU, ELU, GELU와 같이 특정 기울기를 수정하거나 지수적 감소를 도입한 변형 함수들이 제안되었다. 또한, 최근에는 Leaky ReLU와 ELU를 혼합하여 데이터 기반으로 학습하는 적응형 함수 연구도 진행된 바 있다.

본 논문은 이러한 기존 접근 방식과 달리, 활성화 함수를 확률 분포의 CDF로 정식화함으로써 더 광범위한 클래스의 함수를 일반화한다. 특히 특정 변형 함수를 제안하는 것에 그치지 않고, 형상 파라미터를 통해 Sigmoid에서 Gumbel 분포로, 혹은 불연속적인 ReLU에서 매끄러운(Smooth) 버전의 ReLU로 유연하게 전이될 수 있는 프레임워크를 제공한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 학습 절차

각 뉴런은 이제 가중치 $w$와 편향 $w_0$뿐만 아니라, 활성화 함수의 형태를 결정하는 형상 파라미터 $\alpha$를 개별적으로 보유한다. 학습 과정에서 손실 함수 $L$에 대한 $\alpha$의 기울기를 계산하여, 가중치 업데이트와 동시에 활성화 함수의 형태를 최적화한다.

### 주요 구성 요소 및 방정식

#### 1. Adaptive Gumbel Activation

본 연구는 Sigmoid 함수가 로지스틱 분포의 CDF라는 점에 착안하여, 비대칭적인 Gumbel 분포의 CDF를 이용한 적응형 활성화 함수를 제안한다.

$$ \sigma_\alpha(x) = 1 - \{1 + \alpha \exp(x)\}^{-\frac{1}{\alpha}}, \quad \alpha \in \mathbb{R}^+, x \in \mathbb{R} $$

여기서 $\alpha$는 함수의 형태를 조절하는 파라미터이다. $\alpha$ 값의 변화에 따라 함수는 대칭적인 Sigmoid 형태에서 비대칭적인 Gumbel 형태 사이를 오가게 된다.

#### 2. Adaptive ReLU

ReLU는 비유계(Unbounded) 함수이므로, 이를 유계(Bounded) 성분과 비유계 성분의 곱으로 분해하여 접근한다. 표준 ReLU는 $\sigma(x) = x \cdot \Delta(x)$ (여기서 $\Delta(x)$는 Heaviside 함수)로 표현될 수 있으며, 본 논문은 $\Delta(x)$를 매끄러운 CDF인 지수 분포의 CDF $\Delta_\alpha(x)$로 대체한다.

$$ \Delta_\alpha(x) = (1 - e^{-\alpha x}) I_{\{x>0\}}(x) $$
$$ \sigma_\alpha(x) = x \Delta_\alpha(x), \quad \alpha \in \mathbb{R}^+, x \in \mathbb{R} $$

이 식에서 $\alpha$는 매끄러움(Smoothness)을 결정하는 파라미터이며, $\alpha \to \infty$가 되면 표준 ReLU 함수로 수렴한다. 또한 $\Delta_\alpha(x)$를 로지스틱 CDF로 설정할 경우 SWISH 활성화 함수와 동일해진다.

### 학습 알고리즘 및 역전파

학습률 $\gamma > 0$가 주어졌을 때, 뉴런의 파라미터 $\theta = [\alpha, w_0, w]$는 다음과 같은 규칙으로 업데이트된다.

$$ w_0^l \leftarrow w_0^l - \gamma \frac{\partial L}{\partial w_0^l} $$
$$ w^l \leftarrow w^l - \gamma \frac{\partial L}{\partial w^l} $$
$$ \alpha^l \leftarrow \alpha^l - \gamma \frac{\partial L}{\partial \alpha^l} $$

수치적 계산 시 $\alpha > 0$ 조건을 강제하기 위해 $\alpha$ 대신 $e^\alpha$를 재파라미터화하여 사용することを 권장한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 시뮬레이션 데이터, MNIST(이미지), Movie Review(텍스트), URL 사용자 의도 예측 데이터.
- **아키텍처**: 완전 연결 네트워크(Fully-connected) 및 LeNet5 기반 CNN.
- **지표**: 예측 정확도(Accuracy), 정밀도(Precision), 재현율(Recall).

### 주요 결과

1. **시뮬레이션 데이터**: 적응형 Gumbel이 Sigmoid보다 우수한 성능을 보였으며, 적응형 ReLU는 표준 ReLU와 비슷하거나 깊은 네트워크(8개 층)에서 약간 더 높은 성능을 기록했다.
2. **MNIST**: CNN 구조에서 적응형 Gumbel이 가장 좋은 성능을 보였으며, ReLU가 그 뒤를 이었다.
3. **Movie Review**: 텍스트 데이터 분석 결과, 합성곱 층(Convolutional layer)에는 적응형 ReLU를, 완전 연결 층(Fully-connected layer)에는 적응형 Gumbel을 사용하는 조합이 가장 효과적임을 확인했다.
4. **URL 의도 예측**: 실제 기업 데이터를 활용한 적용 사례에서도 적응형 Gumbel이 소폭 우세한 성능을 보였다.

### 정성적 분석

학습된 $\alpha$ 값의 분포를 분석한 결과, 네트워크의 앞부분(Early layers)에 위치한 뉴런들이 뒷부분보다 더 다양하고 유연하게 활성화 함수를 적응시키는 경향이 발견되었다. 이는 초기 층에서 데이터의 기초적인 표현을 학습할 때 활성화 함수의 형태가 더 중요한 역할을 함을 시사한다.

## 🧠 Insights & Discussion

본 논문은 활성화 함수를 고정된 하이퍼파라미터가 아닌, 학습 가능한 파라미터로 취급함으로써 모델의 유연성을 크게 높였다. 특히 활성화 함수를 CDF로 정식화한 접근 방식은 새로운 적응형 함수를 설계할 때 수학적 기반을 제공한다.

**강점 및 시사점:**

- 역전파 식에 단 하나의 방정식만 추가하면 되므로 계산 비용의 증가가 매우 적다.
- 각 뉴런이 자신의 특성에 맞는 최적의 곡률을 스스로 찾아가게 함으로써, 수동으로 활성화 함수를 튜닝해야 하는 번거로움을 줄였다.
- 특히 얕은 구조의 네트워크에서 이러한 유연성이 성능 향상에 더 큰 기여를 하는 것으로 보인다.

**한계 및 비판적 해석:**

- 실험이 주로 LeNet5와 같은 비교적 단순하고 고전적인 아키텍처 위주로 진행되었다. 최신 deep architecture(예: ResNet, Transformer)에서도 동일한 효과가 나타날지는 명시되지 않았다.
- $\alpha$ 파라미터의 초기값 설정이나 학습률 $\gamma$에 대한 민감도 분석이 부족하여, 실제 적용 시 추가적인 튜닝이 필요할 수 있다.

## 📌 TL;DR

본 논문은 활성화 함수를 누적 분포 함수(CDF)로 정의하고 학습 가능한 형상 파라미터 $\alpha$를 도입하여, 뉴런이 스스로 최적의 활성화 함수를 선택하게 하는 **Adaptive Activation** 방법론을 제안한다. Adaptive Gumbel과 Adaptive ReLU를 통해 비대칭성과 매끄러움을 최적화했으며, MNIST 및 텍스트 분류 실험에서 기존 고정 함수보다 우수하거나 대등한 성능을 보였다. 이 연구는 모델 설계 시 활성화 함수 선택의 고민을 학습 과정으로 전이시켰다는 점에서 향후 효율적인 신경망 설계에 기여할 가능성이 높다.
