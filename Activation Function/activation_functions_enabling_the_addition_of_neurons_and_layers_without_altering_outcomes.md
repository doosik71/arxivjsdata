# Activation functions enabling the addition of neurons and layers without altering outcomes

Sergio López-Ureña (2025)

## 🧩 Problem to Solve

본 논문은 신경망(Neural Networks, NNs)의 아키텍처를 확장할 때, 기존에 학습된 네트워크의 함수적 특성을 그대로 유지하면서 뉴런의 수나 레이어의 깊이를 증가시키는 방법을 해결하고자 한다. 일반적으로 네트워크를 확장(Widening 또는 Deepening)하면 모델의 용량은 커지지만, 초기화 과정에서 발생하는 성능 저하(Performance drop)가 주요 문제로 지적된다.

특히 기존의 Net2Net이나 NetMorph와 같은 함수 보존 변환(Function-preserving transformations) 방식은 파라미터 행렬의 과도한 제로 패딩(Zero-padding)으로 인해 학습 초기 단계에서 성능이 하락하거나, 멱등성(Idempotence)과 같은 매우 제한적인 활성화 함수 조건을 요구하는 한계가 있었다. 따라서 본 연구의 목표는 이러한 제약 없이, 네트워크의 출력을 변경하지 않고도 레이어를 삽입하거나 뉴런을 추가할 수 있는 새로운 활성화 함수 클래스를 제안하고 그 수학적 근거를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Subdivision Theory(분할 이론)**에 기반하여 **Refinable(정제 가능)**하고 **Sum the identity(항등함수의 합)** 특성을 가진 새로운 활성화 함수 클래스를 설계한 것이다.

중심적인 직관은 다음과 같다.

1. **Refinability**: 활성화 함수가 특정 조건 하에서 자신의 스케일링 및 시프트된 복사본들의 합으로 표현될 수 있다면, 하나의 뉴런을 여러 개의 뉴런으로 분할해도 전체 출력값이 유지될 수 있다.
2. **Summing the Identity**: 활성화 함수들의 합이 항등 함수($f(x)=x$)가 된다면, 기존 레이어 연산자를 두 개의 연산자로 분리하여 그 사이에 새로운 레이어를 삽입하더라도 수학적으로 동일한 결과를 낼 수 있다.

이러한 설계를 통해 계산 비용이 많이 드는 루틴이나 과도한 제로 패딩 없이도 파라미터를 명시적으로 계산하여 네트워크를 확장할 수 있는 프레임워크를 제공한다.

## 📎 Related Works

논문에서는 신경망 아키텍처 최적화 및 함수 보존 변환과 관련된 기존 연구들을 언급한다.

- **Net2Net (Chen et al., 2015) & NetMorph (Wei et al., 2016)**: 학습된 지식을 더 넓거나 깊은 네트워크로 전이하는 방식을 제안하였으나, 제로 패딩으로 인한 초기 성능 저하 문제가 존재한다.
- **기존 레이어 삽입 방식**: 멱등성(Idempotence)을 가진 매우 제한적인 활성화 함수를 사용하거나, 항등 함수와의 볼록 조합(Convex combination)으로 구성된 파라미터 의존적 활성화 함수를 사용하였다.
- **Subdivision Theory**: 본 논문은 이를 신경망 설계에 도입하여, 기존의 B-Splines나 다해상도 분석(Multiresolution analysis) 개념을 활성화 함수에 적용함으로써 기존의 제한적인 조건들을 극복하고자 한다.

## 🛠️ Methodology

### 1. 핵심 정의 및 속성

본 방법론은 두 가지 핵심 수학적 속성에 의존한다.

**Definition 1: Refinable Function**
함수 $\sigma: \mathbb{R} \to \mathbb{R}$가 $\tau \in \mathbb{R}$와 계수 $a_l \in \mathbb{R}$에 대해 다음을 만족하면 Refinable하다고 정의한다.
$$\sigma(t) = \sum_{l=0}^{A-1} a_l \sigma(2t + \tau - l), \forall t \in \mathbb{R}$$

**Definition 2: Sum the Identity**
구간 $I$ 내에서 $\mu \in \mathbb{R}$와 $B \in \mathbb{N}$가 존재하여 다음을 만족하면 항등함수의 합 속성을 가진다고 정의한다.
$$t = \sum_{l=0}^{B-1} \sigma(t + \mu - l), \forall t \in I$$

### 2. 네트워크 확장 절차

#### 2.1. 뉴런 수 증가 (Layer Widening)

Refinable 활성화 함수 $\sigma_0$를 사용하는 레이어가 있을 때, Theorem 3에 따라 하나의 뉴런을 $A$개의 뉴런으로 분할할 수 있다. 이때 출력의 변화를 막기 위해 가중치 $W$와 편향 $b$를 다음과 같이 업데이트한다.

- **새로운 가중치 및 편향**:
  $$W^0_{l,:} := 2W^0_{0,:}, \quad b^0_l := 2b^0_0 + \tau - l, \quad W^1_{:,l} := a_l W^1_{:,0} \quad (l=0, \dots, A-1)$$
이 과정을 통해 기존 뉴런의 출력이 $A$개 뉴런의 가중 합으로 분해되어 $L_1 \circ L_0$의 결과가 유지된다.

#### 2.2. 레이어 삽입 (Layer Insertion)

Sum the identity 속성을 가진 활성화 함수 $\sigma_0$를 사용하여 기존 레이어 $L$을 $L_1 \circ L_0$로 분리한다.

- **Theorem 5 ($\bar{n} = B n_0$)**: 입력 차원을 기준으로 확장하는 방식으로, 새로운 레이어의 뉴런 수를 $\bar{n} = B n_0$로 설정한다.
- **Theorem 7 ($\bar{n} = B n_1$)**: 출력 차원을 기준으로 확장하는 방식으로, 새로운 레이어의 뉴런 수를 $\bar{n} = B n_1$로 설정한다.
이때 $\beta$라는 스케일링 인자를 도입하여 입력 데이터가 항등 함수 합 속성이 성립하는 구간 $I$ 내에 들어오도록 조정함으로써, 데이터 집합 $\Omega$에 대해 함수 값이 보존되도록 설계한다.

### 3. Spline 활성화 함수 설계

논문은 구체적인 구현체로 B-Splines 기반의 활성화 함수 $\sigma^B_d$를 제안한다.

- **구성**: B-Spline 기저 함수 $\phi^B_d$를 이용하여 다음과 같이 정의한다.
$$\sigma(t) = -\frac{1}{2} + \sum_{m=0}^{\infty} \phi \left( t + \frac{d}{2} - m \right)$$
- **특성**: 이 함수는 $C^{d-1}$ 연속성을 가지며, Refinable하고 Sum the identity 속성을 모두 만족한다.
- **미분 가능성**: 역전파(Backpropagation)를 위해 $\sigma^B_1, \sigma^B_2$의 도함수를 $\sigma$ 값으로 표현한 closed-form 식을 제공하여 효율적인 학습이 가능하게 한다.

## 📊 Results

본 논문은 실험적인 벤치마크 결과보다는 **수학적 증명과 알고리즘적 타당성**을 제시하는 데 집중하고 있다.

- **정량적/정성적 결과**:
  - **Theorem 3, 5, 7**을 통해 뉴런 추가 및 레이어 삽입 시 네트워크의 출력이 수학적으로 동일함($L \circ L = L \circ L$)을 증명하였다.
  - **Theorem 10, 13**을 통해 제안한 Spline 활성화 함수가 Refinability와 Sum the identity 속성을 이론적으로 완벽히 충족함을 보였다.
- **검증 방법**: 저자는 수식의 정확성을 검증하기 위해 Wolfram Mathematica 노트북을 공개하여 상징적 계산(Symbolic computation) 결과를 제공하였다.
- **구현 가이드**: 부록(Appendix A)을 통해 뉴런 분할(Algorithm 1)과 레이어 삽입(Algorithm 2, 3)을 위한 상세한 의사코드를 제공하여 실제 적용 가능성을 입증하였다.

## 🧠 Insights & Discussion

### 강점

- **이론적 엄밀성**: Approximation Theory와 Subdivision Theory를 딥러닝의 아키텍처 확장 문제와 결합하여, 단순한 휴리스틱이 아닌 수학적 보장을 제공하는 프레임워크를 구축하였다.
- **효율성**: 기존의 제로 패딩 방식이나 복잡한 텐서 분해 방식 없이, 명시적인 수식만으로 파라미터를 초기화할 수 있어 계산 효율성이 매우 높다.
- **범용성**: 특정 활성화 함수에 국한되지 않고, Refinable 및 Sum the identity 조건을 만족하는 모든 함수에 적용 가능한 일반적인 이론을 제시하였다.

### 한계 및 미해결 질문

- **실증적 데이터 부족**: 제안한 방법이 실제 대규모 데이터셋(예: ImageNet, GLUE)에서 기존의 Net2Net 대비 얼마나 빠른 수렴 속도를 보이는지, 혹은 최종 정확도에 어떤 영향을 미치는지에 대한 실험적 결과가 제시되지 않았다.
- **활성화 함수의 제약**: 모든 Refinable 함수가 적절한 것은 아니며, 단조 증가(Non-decreasing) 특성을 유지하기 위해서는 'Monotone subdivision scheme'이라는 추가 조건이 필요함을 명시하고 있다.
- **고차 함수 미분**: $\sigma^B_d$에 대해 $d > 2$인 경우의 효율적인 도함수 계산 식을 찾는 것이 향후 과제로 남아 있다.

### 비판적 해석

본 연구는 신경망의 '구조적 성장'을 수학적 함수 보존 문제로 치환하여 해결한 우수한 이론적 시도이다. 하지만 딥러닝의 실무적 관점에서는 이론적 동일성보다 '학습 가능성'이 더 중요하다. 제안된 활성화 함수가 기존의 ReLU나 GELU만큼의 학습 효율을 낼 수 있는지, 그리고 이 함수들로 교체했을 때 기존 가중치와의 호환성 문제가 없는지에 대한 실증적 분석이 보완되어야 한다.

## 📌 TL;DR

본 논문은 **Subdivision Theory**를 이용해 **Refinable**하고 **Sum the identity** 속성을 가진 새로운 활성화 함수 클래스를 제안한다. 이를 통해 네트워크의 출력값을 전혀 바꾸지 않고도 **뉴런 수를 늘리거나(Widening) 레이어를 추가(Deepening)**할 수 있는 명시적인 파라미터 업데이트 공식을 제공한다. 이는 기존의 제로 패딩 기반 확장 방식에서 발생하는 초기 성능 저하 문제를 해결할 수 있는 이론적 토대를 마련한 연구이며, 특히 **B-Spline 기반 활성화 함수**를 통해 구체적인 구현 가능성을 제시하였다. 향후 구조적 학습(Structural Learning) 및 모델 최적화 분야에서 학습 속도를 가속화하는 핵심 기법으로 활용될 가능성이 크다.
