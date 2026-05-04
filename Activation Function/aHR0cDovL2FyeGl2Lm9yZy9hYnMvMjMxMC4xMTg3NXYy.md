# Fractional Concepts in Neural Networks: Enhancing Activation Functions

Zahra Alijani, Vojtech Molek (2025)

## 🧩 Problem to Solve

인공신경망의 성능을 최적화하기 위해서는 적절한 활성화 함수(Activation Function)를 선택하는 것이 매우 중요하다. 활성화 함수는 모델에 필수적인 비선형성(Nonlinearity)을 제공하여 복잡한 데이터 관계를 학습할 수 있게 한다. 하지만 현재의 일반적인 관행은 수많은 활성화 함수 중에서 수동으로 선택하거나 자동화된 탐색 도구를 사용하는 방식에 의존하고 있으며, 이는 많은 시행착오와 반복적인 재학습을 초래한다.

본 논문은 이러한 정적인 선택 방식에서 벗어나, 분수계 미분(Fractional Calculus)을 도입하여 활성화 함수 자체를 유연하게 조정 가능한 파라미터화된 형태로 변환하고자 한다. 즉, 정수 차수의 미분이 아닌 분수 차수 미분(Fractional Order Derivative, FDO)을 학습 가능한 파라미터로 설정함으로써, 단일 함수가 아닌 함수의 '가족(Family)'을 생성하고 네트워크가 최적의 활성화 형태를 스스로 찾도록 하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 분수계 미분을 활성화 함수에 결합하여, 분수 차수 $\alpha$를 조정함에 따라 함수의 형태가 연속적으로 변화하는 가변적 활성화 함수를 설계하는 것이다.

주요 기여 사항은 다음과 같다.

1. **분수계 활성화 함수의 확장**: 기존의 Sigmoid, GELU, Mish 함수에 분수계 미분을 적용하여 FSig, FGELU, FMish라는 새로운 변형 함수들을 제안하였다.
2. **FALU의 개선**: 기존의 Fractional Adaptive Linear Unit (FALU)에서 $\alpha \in (1, 2]$ 범위의 근사치 부정확성 문제를 발견하고, 이를 해결하기 위한 수식 수정($a \to a-1$)을 제안하였다.
3. **실용적 구현 및 분석**: 분수계 미분의 수치적 계산을 위한 하이퍼파라미터 $N$과 $h$의 상관관계를 정의하고, 이를 기반으로 다양한 표준 데이터셋(CIFAR-10, ImageNet-1K 등)과 아키텍처(ResNet, EfficientNet)에서 성능을 검증하였다.
4. **재현성 제공**: 그동안 분수계 활성화 함수 연구에서 부족했던 공개 구현체와 재현 가능한 실험 설정을 포함한 저장소를 제공한다.

## 📎 Related Works

기존 연구들은 분수계 미분을 신경망에 적용하려는 시도를 지속해 왔다.

- **Mittag-Leffler (M-L) 함수**: 지수 함수를 일반화한 M-L 함수를 사용하여 간단한 MLP 구조에서 논리 연산 및 분류 성능을 높인 연구가 있었다. 하지만 범용적인 파라미터 선택 규칙이 부족했다.
- **Softplus 기반 분수 함수**: Softplus에서 유도된 ReLU, tanh, Sigmoid의 분수 변형을 ResNet에 적용하여 성능 향상을 보고한 연구가 있었으나, 학습 루틴이나 코드가 공개되지 않아 재현성이 떨어졌다.
- **FALU (Fractional Adaptive Linear Unit)**: 감마 함수($\Gamma$)와 같은 계산 복잡도가 높은 항을 제거하여 효율적인 연산을 가능케 한 함수가 제안되었다.
- **FReLU 및 변형**: Maclaurin 급수 전개를 통해 FReLU, FPReLU 등을 제안하고 풍력 발전 예측과 같은 회귀 문제에서 성능을 입증하였다.
- **RLCFD (Riemann–Liouville conformable fractional derivative)**: 최신 연구에서 개선된 RLCFD를 도입했으나, 분수 차수 $\alpha$를 학습 가능한 파라미터가 아닌 수동 설정 파라미터로 사용했다는 한계가 있다.

본 논문은 이러한 선행 연구들이 주로 얕은 MLP나 제한적인 데이터셋에 국한되었다는 점을 지적하며, 더 깊은 네트워크와 대규모 데이터셋에서 학습 가능한 $\alpha$를 사용하는 체계적인 평가를 수행한다.

## 🛠️ Methodology

### 1. 분수계 미분의 수치적 정의

본 논문은 수치적 계산이 용이한 **Grünwald-Letnikov (GL)** 미분 정의를 주로 사용한다. 분수 차수 $\alpha \in \mathbb{R}^+$에 대한 GL 미분은 다음과 같이 정의된다.

$$D^\alpha f(x) = \lim_{h \to 0} \frac{1}{h^\alpha} \sum_{n=0}^{\lfloor x/h \rfloor} (-1)^n C_{n, \alpha} f(x-nh)$$

여기서 $C_{n, \alpha}$는 이항 계수이다. 실제 구현을 위해 무한 합을 유한한 항 $k$개로 제한한 근사식 $F^k(x)$를 사용하며, 이는 다음과 같은 가중치 합(Convex Combination) 형태로 표현될 수 있다.

$$F^k(x) = \sum_{n=0}^k w_n \cdot f(x-nh)$$

### 2. 제안하는 분수계 활성화 함수

- **FGELU (Fractional GELU)**: 기존 GELU 함수에 GL 미분을 적용하여, 입력값 $x$에 대해 분수 차수 $\alpha$에 따라 형태가 변하는 비선형 함수를 생성한다.
- **FMish (Fractional Mish)**: Mish 함수의 특성을 유지하면서 분수계 미분을 통해 유연성을 부여한다.
- **FSig (Fractional Sigmoid)**: Sigmoid 함수에 GL 미분을 적용한다. 특히 내부 층에서 Sigmoid가 갖는 기울기 소실 문제를 분수계 미분을 통한 형태 변화로 완화하고자 한다.
- **개선된 FALU**: $\alpha \in [0, 2]$ 범위에서 매끄러운 전이를 위해 다음과 같이 정의한다.
  - $\alpha \in [0, 1]$ 일 때: $g(x, \beta) + \alpha \sigma(\beta x)(1 - g(x, \beta))$
  - $\alpha \in (1, 2]$ 일 때: $h(x, \beta) + (\alpha - 1) \sigma(\beta x)(1 - 2h(x, \beta))$
  - 여기서 $g(x)=x\sigma(x)$이며, $\sigma$는 Sigmoid 함수이다.

### 3. 학습 절차 및 하이퍼파라미터 튜닝

- **하이퍼파라미터 $N$과 $h$**: 합산의 상한 $N$과 간격 $h$는 서로 얽혀 있다. 본 논문은 계산 구간을 일정하게 유지하기 위해 $h = \frac{1}{\max(1, N-1)}$라는 관계식을 제안한다.
- **학습 가능한 FDO**: 분수 차수 $\alpha$를 네트워크의 학습 가능한 파라미터로 설정한다.
- **Weight Decay 제외**: Weight decay가 $\alpha$를 0으로 강제 수렴시키는 부작용이 있음을 발견하여, $\alpha$에 대해서는 weight decay 계수를 $\gamma=0$으로 설정하여 방지한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: CIFAR-10 (32x32), ImageNet-1K, CalTech-256, Food-101.
- **아키텍처**: ResNet-20, 32, 44, 56, 110 및 EfficientNet-B0.
- **비교 대상**: ReLU, PReLU, GELU, Mish, Sigmoid 및 각 분수 변형 함수들.

### 2. 주요 결과

- **FSig의 효과**: 모든 ResNet 변형에서 FSig가 원본 Sigmoid보다 월등히 높은 정확도를 보였으며, 이는 분수계 미분이 Sigmoid의 치명적인 단점을 효과적으로 보완했음을 시사한다.
- **FGELU 및 FMish**: FGELU는 일부 실험에서 GELU보다 소폭 우수한 성능을 보였으나, FMish는 분수 변형 함수들 중 가장 낮은 성능을 보였으며 일부 설정에서 학습이 불안정하여 $\text{NaN}$이 발생하거나 매우 강한 Gradient Clipping이 필요했다.
- **FALU 성능**: 본 실험 결과, FALU는 ReLU나 PReLU보다 뛰어난 성능을 보이지 않았으며, 이는 기존 문헌의 결과와 차이가 있다. 저자들은 기본 활성화 함수(Baseline)들의 성능이 본 실험에서 더 높게 측정되었기 때문이라고 분석한다.
- **계산 복잡도**: $N$이 증가함에 따라 메모리 사용량과 시간 복잡도가 선형적으로 증가한다. 특히 PyTorch의 역전파를 위해 중간 결과물을 저장해야 하므로 메모리 부담이 커진다.

## 🧠 Insights & Discussion

### 1. 손실 함수 표면(Loss Surface)의 변화

저자들은 분수계 활성화 함수가 훈련 손실(Train Loss)은 잘 낮추지만, 테스트 손실(Test Loss)에서 급격한 스파이크가 발생하거나 일반화 성능이 떨어지는 현상을 발견했다. 이는 분수계 미분 계산의 복잡성(특히 감마 함수 사용)이 **손실 함수 표면을 더 불연속적이거나 좁은 국소 최솟값(Narrow Local Minima)을 갖게 만들어**, 모델이 일반화되기 어려운 비볼록(Non-convex)한 표면을 생성하기 때문이라고 가설을 세운다.

### 2. 파라미터 수렴 경향

학습 결과, 분수 차수 $\alpha$는 주로 0 또는 2 근처로 수렴하는 경향을 보였다. 이는 네트워크가 극단적인 형태의 활성화 함수를 선택함으로써 최적화를 시도했음을 의미한다.

### 3. 한계점

- **계산 효율성**: $N$ 값에 따른 복잡도 증가 문제가 해결되지 않아 실무 적용에 제약이 있다.
- **불안정성**: FMish와 같은 특정 함수에서 나타나는 학습 불안정성 문제는 여전히 해결해야 할 과제이다.

## 📌 TL;DR

본 논문은 활성화 함수에 **분수계 미분(Fractional Order Derivative)**을 도입하여, 함수의 형태를 학습 가능한 파라미터 $\alpha$로 제어하는 유연한 활성화 함수 가족을 제안하였다. 실험 결과, **Fractional Sigmoid (FSig)**는 기존 Sigmoid의 한계를 극복하며 성능을 크게 향상시켰으나, 다른 분수 변형 함수들은 계산 복잡도 증가와 손실 표면의 불안정성으로 인해 ReLU/PReLU 같은 전통적인 함수보다 일관된 우위를 점하지 못했다. 이 연구는 분수계 미분을 통한 신경망 유연성 확장 가능성을 제시함과 동시에, 수치적 안정성과 계산 효율성 개선이라는 향후 연구 방향을 명확히 제시하고 있다.
