# Evolution of Activation Functions: An Empirical Investigation

Andrew Nader, Danielle Azar (2021)

## 🧩 Problem to Solve

신경망의 하이퍼파라미터 설정은 전통적으로 전문가의 지식에 의존한 시행착오(trial and error) 과정을 통해 이루어지며, 이는 매우 많은 시간이 소요되는 작업이다. 최근 Neural Architecture Search (NAS) 알고리즘이 등장하여 네트워크 구조 설정을 자동화하려는 시도가 있었으나, 대부분의 NAS는 은닉층의 구성이나 뉴런 간의 연결성에 집중해 왔으며, 완전히 새로운 활성화 함수(activation function)를 찾는 자동화 연구는 상대적으로 부족했다.

활성화 함수는 신경망의 비선형성을 결정하는 핵심 요소임에도 불구하고, 기존 연구들은 주로 사람이 수동으로 설계하거나 미리 정의된 소수의 함수 집합 내에서 선택하는 방식에 머물러 있다. 본 논문의 목표는 진화 알고리즘(evolutionary algorithm)을 통해 특정 문제에 최적화된 완전히 새로운 활성화 함수를 자동으로 발견하고, 이것이 기존의 표준 활성화 함수들보다 우수한 성능을 낼 수 있는지 실험적으로 검증하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 유전 프로그래밍(Genetic Programming, GP)을 사용하여 활성화 함수를 트리 구조로 표현하고, 이를 진화시켜 최적의 수학적 형태를 찾는 것이다. 특히, 활성화 함수의 성능이 가중치 초기화 방식에 크게 의존한다는 점에 착안하여, 활성화 함수 $\phi$와 가중치 초기화 기법 $F$를 하나의 염색체로 묶어 함께 진화시키는 **공진화(co-evolution)** 전략을 제안한다. 이를 통해 사람이 설계한 휴리스틱에 얽매이지 않고 데이터셋과 네트워크 구조에 최적화된 활성화 함수를 탐색할 수 있다.

## 📎 Related Works

논문에서는 전통적인 활성화 함수부터 최신 경향까지 다음과 같이 설명한다.

- **Sigmoidal Functions**: $\tanh$나 logistic function 등이 있으며, 입력값이 커질수록 기울기가 0에 수렴하는 Gradient Vanishing 문제가 발생하여 깊은 망의 학습을 어렵게 만든다.
- **ReLU (Rectified Linear Unit)**: $\phi(x) = \max(0, x)$로 정의되며, 양수 영역에서 기울기가 1로 유지되어 학습 속도를 높이고 희소성(sparsity)을 제공한다. 하지만 음수 영역에서 뉴런이 완전히 비활성화되는 'Dying ReLU' 문제가 존재한다.
- **ReLU 변형 모델**: Leaky ReLU는 음수 영역에 작은 기울기를 추가하여 Dying ReLU 문제를 해결하며, ELU와 SELU는 평균 활성값을 0에 가깝게 밀어내어 학습의 안정성을 높인다. GELU는 확률적 정규화 개념을 도입하여 ReLU와 ELU보다 우수한 성능을 보이기도 한다.
- **자동화된 탐색**: Google Brain의 연구([Ramachandran et al. 2017])는 RNN 컨트롤러와 강화학습을 통해 Swish 함수를 발견하였다. 본 논문은 이러한 자동 탐색의 흐름을 잇되, 미리 정의된 컴포넌트 조합이 아닌 유전 프로그래밍을 통한 더 자유로운 형태의 탐색을 지향한다.

## 🛠️ Methodology

### 1. 염색체 인코딩 (Chromosome Encoding)

본 연구에서 개체(individual)는 $\langle \phi, F \rangle$ 형태의 염색체로 표현된다.

- **활성화 함수 트리 ($\phi$)**: 유전 프로그래밍의 트리 구조를 사용하여 수학 함수를 표현한다. 리프 노드는 입력값 $x$를 나타내며, 내부 노드는 단항(unary) 또는 이항(binary) 연산자로 구성된다.
  - **단항 연산자**: $\text{relu, elu, sigmoid, tanh, swish, sin, cos, atan, sinh, cosh, leaky relu, softplus, erf, \text{absolute value}}$
  - **이항 연산자**: $\text{add, subtract, multiply, maximum, minimum}$
- **가중치 초기화 기법 ($F$)**: $\text{random normal, random uniform, truncated normal, variance scaling, orthogonal, lecun uniform/normal, glorot uniform/normal, he uniform/normal}$ 중 하나를 선택하는 명목 변수이다.

### 2. 진화 절차

- **집단 구성**: 인구 수 100명, 총 50세대 동안 진화시킨다. 상위 4개의 개체를 보존하는 Elitism 전략을 사용한다.
- **선택 및 변이**: Rank selection을 사용하며, 교차율(crossover rate)은 80%, 변이율(mutation rate)은 5%이다.
  - **교차(Crossover)**: 단일 지점 염색체 교차 후, 트리 구조에 대해 leaf-biased one-point crossover를 수행한다.
  - **변이(Mutation)**: 무작위로 가지(branch)를 선택해 그 하위 노드 중 하나로 대체하는 shrink mutation을 사용한다.
- **Bloat 제어**: 트리의 깊이가 10을 초과하면 부모 트리로 대체하여 계산 복잡도를 제한한다.

### 3. 학습 및 평가 절차

- **적합도 측정 (Fitness)**: 훈련 데이터의 10%를 검증셋으로 분리하여 측정한다.
  - 이진 분류: $F_1$ measure
  - 다중 클래스 분류: Categorical Accuracy
  - 회귀: Mean Squared Error (MSE)
- **훈련 세부사항**: ADAM 옵티마이저를 사용하며, 진화 과정 중에는 Early Stopping(10 epoch tolerance)을 적용하여 과적합을 방지한다. 최종 선택된 최우수 함수들은 30회 반복 실행하여 평균과 표준편차를 측정한다.

### 4. 활성화 함수 특성 분석

발견된 함수의 형태를 분석하기 위해 다음 네 가지 속성의 진화 과정을 추적한다.

- **Monotonicity**: 함수가 단조 증가하는지 여부.
- **Zero on Non-Negative Real Axis**: $x \le 0$일 때 $\phi(x) = 0$인지 여부 (Sparsity 확인).
- **Upper Unbounded**: $x \to \infty$일 때 $\phi(x) \to \infty$인지 여부.
- **Lower Unbounded**: $x \to -\infty$일 때 $\phi(x) \to -\infty$인지 여부.

## 📊 Results

실험은 크게 세 가지 태스크(다변량 분류, 회귀, 이미지 분류)에서 수행되었다.

### 1. 다변량 분류 (Multi-variate Classification)

- **데이터셋**: Electricity, Magic Telescope, Robot Navigation, EEG Eye State.
- **결과**: 다수의 데이터셋에서 진화된 함수($\phi_1, \phi_2, \phi_3$)가 ReLU, ELU, SELU 등 베이스라인보다 높은 성능을 보였다. 특히 Magic Telescope 데이터셋에서는 정밀도(Precision)가 약 6% 향상되었다.
- **통계적 유의성**: Tukey HSD 테스트 결과, 대부분의 데이터셋에서 진화된 함수들의 개선 사항이 통계적으로 유의미함($p < 0.001$)이 확인되었다.

### 2. 회귀 (Regression)

- **데이터셋**: Red Wine Quality, White Wine Quality, California Housing.
- **결과**: 모든 회귀 데이터셋에서 진화된 함수가 베이스라인보다 낮은 MSE를 기록하였다.
- **특이점**: 회귀 문제에서는 $\sin, \cos$와 같은 주기적 함수가 포함된 비단조(non-monotonic) 함수들이 발견되었으며, 이들이 기존의 단조 함수들보다 훨씬 우수한 성능을 보였다.

### 3. 이미지 분류 (Image-based Classification)

- **데이터셋**: CIFAR-10, Fashion-MNIST, MNIST (각 5,000개의 샘플 서브셋 사용).
- **결과**: CIFAR-10에서 정확도가 약 3% 향상되었고, MNIST에서도 약 0.7%의 향상이 있었다.
- **특이점**: CIFAR-10과 Fashion-MNIST에서는 ReLU와 유사하게 $x \le 0$에서 0이 되는 특성이 선호되었으나, MNIST에서는 매우 특이한 형태의 함수들이 발견되었다.

## 🧠 Insights & Discussion

본 논문은 다음과 같은 중요한 통찰을 제공한다.
첫째, **전통적인 활성화 함수 설계 휴리스틱의 한계**이다. 일반적으로 활성화 함수는 단조 증가(monotonicity)해야 하며, 상한선이 없어야(upper unbounded) 학습이 잘 된다고 알려져 있다. 하지만 본 실험 결과, 특히 회귀 문제에서는 주기적이고 비단조적인 함수가 최적의 성능을 냈으며, 이는 문제의 특성에 따라 필요한 함수의 형태가 완전히 다를 수 있음을 시사한다.

둘째, **가중치 초기화와의 결합 효과**이다. 동일한 활성화 함수라도 초기화 방식에 따라 성능이 크게 달라지며, 이를 함께 최적화함으로써 더 높은 성능을 달성할 수 있었다.

셋째, **알고리즘의 일반성**이다. Robot Navigation 데이터셋의 경우 알고리즘이 스스로 ReLU 함수를 재발견(recover)하였다. 이는 제안된 방법론이 특정 방향으로 편향되지 않고, 실제 최적이 ReLU라면 이를 찾아낼 수 있음을 입증한다.

**한계점**:

- 계산 자원의 한계로 인해 이미지 데이터셋의 전체 크기를 사용하지 못하고 서브셋만 사용하였다.
- 발견된 함수들이 수학적으로는 복잡하지만, 실제 구현 시 계산 비용(computational cost)에 대한 정밀한 분석이 부족하다.

## 📌 TL;DR

본 연구는 유전 프로그래밍을 통해 **활성화 함수와 가중치 초기화 기법을 자동으로 설계하는 프레임워크**를 제안하였다. 실험 결과, 이렇게 발견된 함수들이 다양한 분류 및 회귀 작업에서 ReLU, ELU, SELU 등 표준 함수들을 통계적으로 유의미하게 능가함을 확인하였다. 특히, 기존의 설계 상식(단조성 등)을 깨는 함수들이 특정 문제에서 더 좋은 성능을 낸다는 점을 밝혀냈으며, 이는 향후 NAS(Neural Architecture Search) 과정에 활성화 함수 탐색을 통합하는 것이 매우 가치 있는 작업임을 시사한다.
