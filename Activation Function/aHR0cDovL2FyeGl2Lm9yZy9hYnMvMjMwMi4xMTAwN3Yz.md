# Unification of popular artificial neural network activation functions

Mohammad Mostafanejad (2023)

## 🧩 Problem to Solve

인공신경망(ANN)에서 활성화 함수(Activation Function)는 신경망의 반응 풍부함을 조절하고 보편적 근사자(Universal Approximators)로서의 정확도, 효율성 및 성능을 결정하는 핵심 구성 요소이다. 초기에는 Logistic Sigmoid나 Hyperbolic Tangent와 같은 포화 활성화 함수(Saturating Activation Functions)가 주로 사용되었으나, 이들은 기울기 소실(Vanishing Gradient) 문제로 인해 깊은 망 학습에 어려움이 있었다. 이를 해결하기 위해 ReLU(Rectified Linear Unit)와 그 변형들이 제안되었으며, 이후 Swish, Mish, GELU 등 다양한 함수들이 등장하였다.

그러나 현재까지 모든 데이터 모달리티와 응용 도메인에서 일관되게 우월한 성능을 보이는 단일 활성화 함수는 존재하지 않는다. 연구자들은 최적의 활성화 함수를 찾기 위해 수많은 실험적 설정(데이터 전처리, 옵티마이저, 정규화 등)과 하이퍼파라미터 튜닝, 또는 신경망 구조 탐색(NAS)에 막대한 계산 비용을 소모하고 있다. 따라서 본 논문은 다양한 활성화 함수들 사이의 수학적 연결 고리를 찾아 하나의 통합된 형태로 표현함으로써, 함수 선택의 혼란을 줄이고 학습 데이터에 따라 최적의 형태를 적응적으로 학습할 수 있는 유연한 프레임워크를 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 분수 미적분학(Fractional Calculus)의 **Mittag-Leffler 함수**를 도입하여, 기존의 주요 활성화 함수들을 하나의 수식으로 통합할 수 있는 **통합 게이트 표현식(Unified Gated Representation)**을 제안한 것이다.

중심적인 직관은 서로 다른 특성을 가진 활성화 함수들이 사실상 특정 매개변수 집합을 가진 하나의 일반화된 함수 형태의 특수 사례(Special Cases)라는 점이다. 제안된 표현식은 다음과 같은 이점을 가진다.
1. **상호 보완적 보간(Interpolation):** 매개변수를 조정함으로써 서로 다른 활성화 함수 사이를 매끄럽게 보간할 수 있으며, 이를 통해 포화 동작(Saturation behavior)을 정밀하게 제어하여 기울기 소실 및 폭주 문제를 완화할 수 있다.
2. **적응적 학습 가능성:** 고정된 형태의 함수뿐만 아니라, 활성화 함수의 형태를 결정하는 매개변수 자체를 학습 가능한 변수로 설정하여 데이터에 최적화된 함수 형태를 찾을 수 있다.
3. **수학적 일관성:** 제안된 형태는 미분에 대해 닫혀 있어(Closed under differentiation), 경사 하강법 기반의 역전파(Backpropagation) 알고리즘에 효율적으로 적용 가능하다.

## 📎 Related Works

기존 연구들은 ReLU의 단점인 Dying ReLU 문제를 해결하기 위해 Leaky ReLU, ELU, SELU 등의 변형 함수들을 제안해 왔다. 또한, 고정된 형태가 아닌 학습 가능한 활성화 함수(Trainable Activation Functions) 연구들이 진행되었으나, 이러한 함수들은 종종 제약 조건이 있는 단순한 다층 퍼셉트론(MLP) 하위 네트워크로 대체될 수 있다는 한계가 있었다.

본 논문은 단순히 새로운 함수를 제안하는 것이 아니라, 기존의 수많은 함수(ReLU, Sigmoid, Tanh, Swish, Mish, GELU 등)를 수학적으로 통합하는 관점을 취한다. 이는 개별 함수들을 각각 구현하고 실험하는 기존 방식과 달리, 하나의 통합된 수식 내에서 매개변수 최적화 문제로 접근한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Mittag-Leffler 함수
본 연구의 기초가 되는 Mittag-Leffler 함수는 분수 미적분학에서 지수 함수의 일반화된 형태로 알려져 있다. 1-매개변수 함수는 다음과 같이 정의된다.

$$E_{\alpha}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}, \quad \alpha \in \mathbb{C}$$

또한, 더 일반화된 2-매개변수 Mittag-Leffler 함수는 다음과 같다.

$$E_{\alpha, \beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}, \quad \text{Re}(\alpha) > 0, \beta \in \mathbb{C}$$

### 2. 통합 게이트 표현식 (Unified Gated Representation)
저자는 위 함수들을 이용하여 다음과 같은 통합된 활성화 함수 형태를 제안한다.

$$x\Phi[x] := x \cdot x^{\gamma-1} \left( \frac{E_{\alpha_1, \beta_1}[f(x)]}{E_{\alpha_2, \beta_2}[g(x)]} \right)$$

여기서 $\Phi$는 게이트 함수이며, $f(x)$와 $g(x)$는 실수 집합 $\mathbb{R}$에서 $\mathbb{R}$로 매핑되는 잘 정의된(well-behaved) 함수이다. 이 수식에서 $\gamma, \alpha, \beta$ 등의 매개변수와 $f, g$ 함수의 선택에 따라 기존의 유명한 활성화 함수들을 다음과 같이 구현할 수 있다.

- **ReLU:** $\gamma=1$이며 게이트 함수 결과가 $x>0$일 때 $1$, 그 외 $0$이 되도록 설정.
- **Sigmoid:** $\gamma=0, f(x)=-e^{-x}, g(x)=0$으로 설정하여 $\sigma(x) = \frac{1}{1+e^{-x}}$를 도출.
- **Swish:** $\gamma=1, f(x)=e^{-cx}, g(x)=0$으로 설정하여 $x\sigma(cx)$ 형태 구현.
- **Hyperbolic Tangent (tanh):** $f(x)=g(x)=x^2$와 특정 매개변수 집합을 통해 구현.
- **Mish:** Softplus 함수를 tanh 게이트 함수의 인자로 전달하고 $\gamma=2$로 설정.

### 3. 학습 및 추론 절차
이 표현식은 두 가지 방식으로 사용될 수 있다. 첫째는 특정 활성화 함수를 모사하기 위해 매개변수를 고정하는 **고정 형태(Fixed-shape)** 방식이고, 둘째는 $\beta_2$와 같은 매개변수를 학습 가능하게 설정하여 역전파를 통해 최적의 함수 형태를 찾아가는 **적응적(Adaptive)** 방식이다.

특히, Mittag-Leffler 함수의 도함수는 다시 Mittag-Leffler 함수의 합으로 표현될 수 있으므로(Closed under differentiation), 다음과 같은 일반 미분 식을 통해 효율적인 경사 하강법 구현이 가능하다.

$$\frac{d}{dz} E_{\alpha, \beta}(z) = \frac{1}{\alpha} [E_{\alpha, \alpha+\beta-1}(z) + (1-\beta)E_{\alpha, \alpha+\beta}(z)]$$

## 📊 Results

### 1. 실험 설정
제안된 통합 표현식의 효율성과 정확성을 검증하기 위해 네 가지 이미지 분류 실험을 수행하였다.
- **모델 및 데이터셋:** 
    - LeNet-5 $\rightarrow$ MNIST, CIFAR-10
    - ShuffleNet-v2, ResNet-101 $\rightarrow$ ImageNet-1k
- **평가 지표:** Loss, Accuracy, Precision, Recall, F1-score, 그리고 계산 비용을 측정하기 위한 Wall-clock time 및 Processing rate.
- **기준선(Baseline):** ReLU를 사용한 기본 모델.

### 2. 주요 결과
- **정확도 및 성능:** 
    - **MNIST/CIFAR-10:** LeNet-5 실험 결과, 통합 표현식으로 구현한 활성화 함수들은 기존 built-in 구현체와 거의 동일한 검증 정확도를 보였다. 특히 CIFAR-10에서는 ReLU를 Swish-1이나 Mish로 대체했을 때 정확도가 약 $1.2\%$ 향상되는 결과가 통합 표현식에서도 동일하게 관찰되었다.
    - **ImageNet-1k:** ShuffleNet-v2와 ResNet-101 실험에서도 통합 표현식은 기존 활성화 함수들의 성능을 그대로 재현하였다. ResNet-101의 경우, 통합 표현식 기반의 tanh가 가장 낮은 검증 손실(Validation Loss)을 기록하였다.
- **계산 효율성:**
    - 통합 표현식은 특수 함수(Mittag-Leffler) 계산으로 인해 built-in 함수보다 약간의 추가 시간이 소요되지만, 그 차이는 매우 작았다.
    - 예를 들어, MNIST 실험에서 Mish의 경우 built-in(24.38s) 대비 통합 표현식(40.73s)이 더 오래 걸렸으나, Softsign의 경우 차이가 거의 없었다. 
    - ImageNet-1k의 ShuffleNet-v2 실험에서는 일부 게이트 표현식(Mish)이 오히려 built-in보다 빠른 처리 속도를 보이기도 하였다.

## 🧠 Insights & Discussion

본 논문은 개별적으로 존재하던 수많은 활성화 함수들이 사실은 하나의 통합된 수학적 틀(Mittag-Leffler 기반 게이트 표현식) 안에 존재함을 입증하였다. 이는 단순한 수식의 통합을 넘어 다음과 같은 학술적 의미를 가진다.

첫째, **설계의 유연성**이다. 연구자는 더 이상 "ReLU를 쓸 것인가, Mish를 쓸 것인가"를 고민하는 대신, 통합 표현식의 매개변수 공간에서 최적의 함수 형태를 탐색하거나 학습할 수 있다. 특히 $\beta_2$ 값을 조절하여 선형 함수와 tanh 함수 사이를 매끄럽게 보간하는 결과는, 신경망의 포화 특성을 정밀하게 제어할 수 있는 가능성을 보여준다.

둘째, **구현의 간결함**이다. 다양한 활성화 함수를 위해 각각의 클래스를 정의하는 대신, 하나의 통합 함수와 매개변수 테이블만으로 모든 함수를 대체할 수 있어 코드의 유지보수성이 향상된다.

셋째, **분수 미적분학의 확장성**이다. 본 연구는 활성화 함수를 분수 미적분학의 관점에서 재해석함으로써, 향후 '분수 인공신경망(Fractional ANN)' 및 새로운 역전파 알고리즘 연구로 나아갈 수 있는 이론적 토대를 마련하였다.

다만, 특수 함수 계산에 따른 추가적인 계산 오버헤드가 존재하며, 이는 Mittag-Leffler 함수의 더 효율적인 수치 계산 알고리즘이 도입되어야 해결될 문제이다. 또한, 본 논문에서는 매개변수를 학습 가능한 변수로 설정했을 때의 구체적인 성능 향상 폭보다는 통합 가능성 자체에 집중하였으므로, 실제 적응적 학습을 통한 성능 최적화에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 연구는 분수 미적분학의 **Mittag-Leffler 함수**를 이용하여 ReLU, Sigmoid, Tanh, Swish, Mish, GELU 등 대중적인 활성화 함수들을 하나의 수식으로 통합한 **Unified Gated Representation**을 제안하였다. 실험을 통해 이 통합 표현식이 기존 개별 함수들의 성능을 그대로 유지하면서도, 매개변수 조절을 통해 함수 간 보간 및 적응적 형태 학습이 가능함을 확인하였다. 이는 활성화 함수 선택의 복잡성을 줄이고, 신경망 설계에 수학적 유연성을 제공하며, 향후 분수 미적분 기반의 새로운 신경망 구조 연구에 기여할 가능성이 높다.