# STL: A Signed and Truncated Logarithm Activation Function for Neural Networks

Yuanhao Gong (2021)

## 🧩 Problem to Solve

인공신경망(Neural Networks)에서 활성화 함수(Activation Function)는 비선형성(Non-linearity)을 제공하여 네트워크가 복잡한 데이터 관계를 모델링할 수 있게 하는 핵심 요소이다. 그러나 기존의 활성화 함수들은 수학적 성질의 한계로 인해 학습의 효율성이나 정확도에 영향을 주는 문제점들을 가지고 있다.

본 논문이 해결하고자 하는 구체적인 문제와 목표는 다음과 같다.

- **기존 함수의 한계:** ReLU는 음수 영역에서 기울기가 0이 되는 Dead ReLU 문제와 기함수(Odd function)가 아니라는 점이 있으며, Sigmoid나 Tanh는 출력값이 제한되어 있어 큰 입력값 간의 구분이 어렵고 기울기 소실(Vanishing Gradient) 문제가 발생한다.
- **연구 목표:** 기함수 성질, 단조 증가, 미분 가능성, 무한한 출력 범위, 연속적이고 0이 아닌 기울기, 그리고 계산 효율성이라는 6가지 이상적인 수학적 성질을 모두 만족하는 새로운 활성화 함수를 제안하고 그 성능을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **STL(Signed and Truncated Logarithm)**이라는 새로운 활성화 함수를 설계한 것이다.

STL의 중심 설계 아이디어는 입력값이 작은 구간($|x| \le 1$)에서는 선형적으로 동작하게 하여 원점 근처의 수치적 불안정성을 방지하고, 입력값이 큰 구간($|x| > 1$)에서는 로그 함수를 사용하여 값의 증가 속도를 조절하면서도 출력 범위를 무한히 확장하는 것이다. 이를 통해 기울기 소실 문제를 해결함과 동시에 기함수적 특성을 유지하여 학습 과정에서의 잠재적 편향(Bias)을 제거하였다.

## 📎 Related Works

저자는 기존의 다양한 활성화 함수들을 분석하고 그 한계를 지적한다.

- **Sigmoid & Tanh:** 출력 범위가 유한하여(Bounded) 큰 입력값들에 대해 변별력이 떨어지며, 기울기 소실 문제가 발생한다.
- **ReLU & Variants (PReLU, ELU):** ReLU는 $x \le 0$에서 기울기가 0이 되어 정보 손실이 발생하며, 기함수가 아니기에 학습 시스템에 암시적인 편향을 줄 수 있다.
- **Swish & Serf:** 최신 함수들이지만 여전히 위에서 언급한 6가지 이상적인 성질을 모두 만족하지는 못한다.
- **NLReLU:** 로그 함수를 사용하지만 출력값이 비음수(Non-negative)로 제한된다는 한계가 있다.

STL은 이러한 기존 함수들과 달리, 기함수 성질과 무한한 출력 범위, 연속적인 0이 아닌 기울기를 동시에 확보함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. STL 함수 정의

STL 함수는 다음과 같이 정의되는 조각 함수(Piecewise function)이다.

$$f_{our}(x) = \begin{cases} \alpha x, & \text{when } |x| \le 1 \\ \alpha \delta(x)(\log(|x|) + 1), & \text{else} \end{cases}$$

여기서 $\delta(x)$는 $x$의 부호를 나타내는 $\text{sign}$ 함수이며, $\alpha$는 스케일 파라미터(기본값은 1)이다. $|x| \le 1$ 구간에서 로그 함수를 절단(Truncate)한 이유는 해당 구간에서 로그 함수의 기울기가 급격히 증가하여 발생할 수 있는 수치적 문제를 방지하기 위함이다.

### 2. 기울기(Gradient) 분석

STL의 도함수는 다음과 같이 정의된다.

$$f'_{our}(x) = \begin{cases} \alpha, & \text{when } |x| \le 1 \\ \frac{\alpha}{|x|}, & \text{else} \end{cases}$$

이 식에서 알 수 있듯이 $0 < f'_{our}(x) \le \alpha$가 성립하므로, 기울기가 결코 0이 되지 않아 Vanishing Gradient 문제를 원천적으로 차단하며, 모든 구간에서 연속적이다.

### 3. 계산 효율성 및 구현

로그 계산의 비용을 줄이기 위해 IEEE 754 부동 소수점 표현 방식(부호, 지수 $E$, 가수 $V$)을 활용한 $\log_2$ 근사 기법을 제안한다.

$$\log_2(x) = E - 127 + \log_2(1 + V)$$

$1 \le x < 2$ 범위의 $\log_2(1+V)$ 값은 다항식 근사나 룩업 테이블(Lookup Table)을 통해 빠르게 계산할 수 있어, 실질적인 연산 속도를 ReLU나 Softsign 수준으로 유지할 수 있다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** CIFAR-10, CIFAR-100
- **비교 대상:** ReLU, Mish, Serf
- **평가 지표:** Top-1 Accuracy (%)
- **대상 모델:** SqueezeNet, ResNet-50, WideResNet, ShuffleNet-v2, ResNeXt-50, Inception-v3, DenseNet-121, MobileNet-v2, EfficientNet-B0 등 다수의 아키텍처

### 2. 정량적 결과

- **CIFAR-10:** 거의 모든 모델에서 STL이 가장 높은 정확도를 기록하였다. 특히 ResNet-50에서는 ReLU(86.54%) 대비 STL(88.32%)로 유의미한 성능 향상을 보였다.
- **CIFAR-100:** ResNet-164, WideResNet-28-10 등에서 STL이 타 함수들보다 우세한 성능을 보이며 State-of-the-art(SOTA) 성능을 입증하였다.
- **실행 시간:** 20,000개의 무작위 샘플에 대해 연산 시간을 측정한 결과, ReLU와 Softsign은 0.0054초, STL은 0.0060초가 소요되어 계산 비용 측면에서도 매우 효율적임을 확인하였다. (단, 이는 수치적 가속 기법을 적용하지 않은 기본 $\log_2$ 함수 기준이다.)

## 🧠 Insights & Discussion

### 강점

STL은 수학적으로 매우 견고하게 설계되었다. 특히 기함수(Odd function) 성질을 통해 입력 데이터의 부호에 따른 편향을 없앴으며, 단조 증가(Monotone) 성질을 통해 입력 정보의 순서를 보존하는 전단사 함수(Bijective mapping)를 구현하였다. 또한, 기울기가 0이 되지 않으면서도 유한한 범위 내에 존재하여 학습의 수치적 안정성을 높였다.

### 한계 및 논의사항

- **파라미터 $\alpha$의 영향:** 논문에서는 $\alpha=1$을 기본값으로 사용하며, 이후의 선형 변환 층이 이 값을 흡수할 것이라고 주장한다. 하지만 $\alpha$ 값의 변화가 실제 성능에 미치는 영향에 대한 정밀한 Ablation Study는 제시되지 않았다.
- **범용성 검증:** CIFAR 데이터셋 외에 더 거대한 데이터셋(예: ImageNet)이나 다른 도메인(NLP 등)에서의 검증이 추가적으로 필요하다.
- **계산 복잡도:** 비록 시간 차이가 미미하지만, 단순 덧셈과 최대값 연산만으로 이루어진 ReLU에 비해 로그 연산이 포함된 STL은 하드웨어 가속기(GPU/TPU) 수준에서 구현 시 약간의 오버헤드가 발생할 가능성이 있다.

## 📌 TL;DR

본 논문은 기존 활성화 함수들의 수학적 결함(기울기 소실, 편향, 유한한 범위 등)을 해결하기 위해 **STL(Signed and Truncated Logarithm)** 함수를 제안한다. STL은 기함수, 단조 증가, 무한 범위, 연속적이고 0이 아닌 기울기라는 이상적인 성질을 모두 갖추고 있으며, CIFAR-10/100 실험을 통해 ReLU, Mish, Serf보다 우수한 성능을 입증하였다. 이 연구는 특히 수치적 안정성이 중요한 깊은 신경망 설계에 있어 효율적인 대안이 될 가능성이 높다.
