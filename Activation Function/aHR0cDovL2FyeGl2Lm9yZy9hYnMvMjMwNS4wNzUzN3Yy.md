# Saturated Non-Monotonic Activation Functions

Junjia Chen, Zhibin Pan (2023)

## 🧩 Problem to Solve

딥러닝 네트워크에서 활성화 함수(Activation Function)는 모델의 비선형성을 결정하는 핵심 요소이다. 기존의 대중적인 활성화 함수들은 대부분 단조 함수(Monotonic Function) 형태를 띠고 있으며, 최근에는 비단조(Non-monotonic) 활성화 함수들이 우수한 성능을 보이면서 연구되고 있다.

그러나 GELU, SiLU, Mish와 같은 비단조 활성화 함수들은 비단조성을 도입함으로써 양수 입력 영역에서도 비선형성을 가지게 되는데, 이는 결과적으로 양수 입력 신호를 왜곡(distortion)시키는 결과를 초래한다. ReLU와 그 변형들이 성공적이었던 이유는 양수 영역에서 입력을 왜곡 없이 그대로 통과시켰기 때문이라는 점을 고려할 때, 비단조 함수들의 양수 영역 왜곡은 불필요하거나 잠재적으로 해로운 요소가 될 수 있다.

따라서 본 논문의 목표는 ReLU가 가진 '양수 영역의 무손실 전달' 특성과 비단조 함수들이 가진 '음수 영역의 강력한 표현력'을 결합하여, 양수 신호의 왜곡은 없애면서 비단조성의 이점은 유지하는 새로운 활성화 함수를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비단조 활성화 함수의 음수 부분만을 취하고, 양수 부분은 ReLU의 선형 함수로 대체하는 **'통과율 포화(Pass Rate Saturation)'** 개념을 도입하는 것이다.

이를 통해 양수 입력에 대해서는 기울기가 항상 1인 상수를 유지함으로써 경사 하강법(Gradient Descent)의 효율성을 높이고, 음수 입력에 대해서는 비단조적인 특성을 유지하여 네트워크의 일반화 성능을 향상시키고자 하였다. 이러한 설계 철학을 바탕으로 SGELU, SSiLU, SMish라는 세 가지 새로운 활성화 함수를 제안하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 활성화 함수들의 특성과 한계를 언급한다.

- **ReLU (Rectified Linear Unit):** 단순함과 빠른 수렴 속도를 가지며 양수 입력을 왜곡 없이 전달하지만, 음수 입력 시 뉴런이 완전히 꺼지는 Dying ReLU 문제로 인해 가중치 업데이트가 중단되는 한계가 있다.
- **LeakyReLU 및 PReLU:** Dying ReLU 문제를 해결하기 위해 음수 영역에 작은 기울기를 도입하거나 학습 가능한 파라미터를 추가하였으나, 여전히 단조 함수라는 틀 안에 있다.
- **비단조 함수 (GELU, SiLU, Mish):** 음수 영역에서 비단조적인 특성을 가져 네트워크의 표현력을 높이고 우수한 성능을 보이지만, 양수 영역에서도 로그 또는 지수 연산으로 인해 미세한 비선형 왜곡이 발생한다.
- **PFLU (Power Function Linear Unit):** 음수 영역의 희소성을 유지하면서 비단조성을 도입한 사례이다.

본 연구는 이러한 기존 연구들과 달리, 양수 영역은 ReLU의 정체성 함수(Identity function)를 그대로 사용하고 음수 영역만 선택적으로 비단조 함수를 결합함으로써 두 접근 방식의 장점만을 취했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 구조 및 설계 방식

제안하는 활성화 함수 $f^s(x)$의 일반적인 구조는 다음과 같이 정의된다.

$$
f^s(x) =
\begin{cases}
x, & x \ge 0 \\
x \cdot s(\beta x), & x < 0
\end{cases}
$$

여기서 $s(\cdot)$는 $[0, 1]$ 범위의 값을 가지는 게이트 함수이며, $\beta$는 하이퍼파라미터 또는 학습 가능한 파라미터이다.

### 제안하는 활성화 함수 종류

위의 구조를 바탕으로 GELU, SiLU, Mish의 음수 부분을 각각 결합하여 세 가지 함수를 정의한다.

1. **SGELU (Saturated GELU):**
   $$\text{SGELU}(x) = \max\left(x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right], x\right)$$
2. **SSiLU (Saturated SiLU):**
   $$\text{SSiLU}(x) = \max\left(\frac{x}{1 + e^{-x}}, x\right)$$
3. **SMish (Saturated Mish):**
   $$\text{SMish}(x) = \max(x \tanh(\ln(1 + e^x)), x)$$

### 기울기(Gradient) 분석

제안된 함수들은 $x \ge 0$ 영역에서 기울기가 항상 $1$이다. 음수 영역의 기울기는 각 기반 함수의 미분값을 따른다. 예를 들어 SGELU의 기울기는 다음과 같다.

$$
\frac{d\text{SGELU}(x)}{dx} =
\begin{cases}
\frac{x}{2\pi}e^{-\frac{x^2}{2}} + \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right], & x < 0 \\
1, & x \ge 0
\end{cases}
$$

이러한 구조는 양수 영역에서 기울기가 소실되거나 왜곡되지 않고 일정하게 유지되게 하여 최적화 과정을 더 효율적으로 만든다.

### 통과율 포화 (Pass Rate Saturation) 이론

저자들은 비단조 함수를 입력 $x$에 확률적 마스크 $m \sim \text{Bernoulli}(F(x))$를 곱한 과정의 기댓값 $E[mx] = x F(x)$로 해석한다. 여기서 $F(x)$는 **통과율 함수(Pass Rate Function)**이다.

- **기존 비단조 함수:** $F(x)$가 $x > 0$에서도 1에 도달하기까지 완만한 곡선을 그리므로 신호 왜곡이 발생한다.
- **ReLU:** $F_{\text{ReLU}}(x)$는 $x \ge 0$일 때 정확히 $1$이며, $x < 0$일 때 $0$이다.
- **제안 방법:** 기존 비단조 함수의 $F(x)$를 $x \ge 0$ 영역에서 강제로 $1$로 포화(Saturate)시켜, 양수 신호를 무손실로 전달하도록 설계한 것이다.

## 📊 Results

### 실험 설정

- **데이터셋:** CIFAR-100 (100개 클래스, 클래스당 학습 이미지 500장, 테스트 이미지 100장).
- **평가 모델:** MobileNet, MobileNetV2, ShuffleNet V2, SqueezeNet, VGG-11, VGG-13.
- **학습 설정:** SGD 옵티마이저 (Momentum 0.9, Weight Decay $5 \times 10^{-4}$), 초기 학습률 0.1 (50, 120, 160 에포크에서 5배씩 감소).
- **비교 대상:** ReLU, LReLU, PReLU, Swish, SiLU, Mish, GELU.

### 정량적 결과 (Top-1 Accuracy)

실험 결과, 제안된 방법들이 대부분의 아키텍처에서 기존 베이스라인보다 우수한 성능을 보였다.

- **SGELU의 성능:** SqueezeNet을 제외한 모든 모델에서 GELU보다 성능이 향상되었으며, ReLU 대비 약 $1\%$에서 $4.3\%$까지 정확도가 상승하였다. 특히 모든 모델을 통틀어 가장 높은 성능을 기록한 경우가 많았다.
- **SSiLU 및 SMish:** 각각 대응되는 SiLU 및 Mish보다 성능이 향상되었으며, 대부분의 네트워크에서 ReLU보다 우수한 성능을 보였다.
- **상대적 순위:** 기반 함수들의 성능 순위가 $\text{GELU} > \text{SiLU} > \text{Mish}$였으며, 제안된 함수들 역시 $\text{SGELU} > \text{SSiLU} > \text{SMish}$ 순으로 성능이 나타났다.

## 🧠 Insights & Discussion

### 강점

본 연구는 단순한 수식의 조합을 넘어, 활성화 함수를 '통과율(Pass Rate)'이라는 확률적 관점에서 해석하여 양수 영역의 왜곡 문제를 명확히 정의하였다. 또한, 이론적 가설(양수 영역의 선형성이 효율적이다)을 다양한 경량 모델(MobileNet 등)과 무거운 모델(VGG 등) 모두에서 검증함으로써 범용성을 입증하였다.

### 한계 및 논의사항

- **데이터셋의 제한:** 실험이 CIFAR-100이라는 상대적으로 작은 규모의 데이터셋에 국한되어 있다. ImageNet과 같은 대규모 데이터셋에서도 동일한 성능 향상이 유지되는지에 대한 검증이 필요하다.
- **계산 비용:** $\text{erf}$나 $\tanh, \ln$과 같은 연산은 단순한 ReLU보다 계산 비용이 높다. 양수 영역은 단순화되었으나 음수 영역의 연산 비용이 전체 추론 속도에 미치는 영향에 대한 분석이 명시되지 않았다.
- **SqueezeNet의 예외성:** SGELU가 SqueezeNet에서만 GELU보다 약간 낮은 성능을 보인 이유에 대해 구체적인 분석이 제시되지 않았다.

## 📌 TL;DR

본 논문은 비단조 활성화 함수(GELU, SiLU, Mish)가 가진 양수 영역의 신호 왜곡 문제를 해결하기 위해, **양수 영역은 ReLU의 선형성을 따르고 음수 영역만 비단조성을 유지하는 SGELU, SSiLU, SMish**를 제안하였다. CIFAR-100 실험 결과, 특히 SGELU가 기존의 단조/비단조 활성화 함수들보다 전반적으로 우수한 성능을 보였으며, 이는 양수 신호의 무손실 전달과 음수 신호의 비선형 표현력이 결합되었을 때 최적의 성능이 나옴을 시사한다. 향후 다양한 대규모 비전 모델의 기본 활성화 함수로 적용될 가능성이 높다.
