# Learning Specialized Activation Functions for Physics-informed Neural Networks

Honghui Wang, Lu Lu, Shiji Song, and Gao Huang (2023)

## 🧩 Problem to Solve

Physics-informed Neural Networks (PINNs)는 물리 법칙(주로 편미분 방정식, PDE)을 신경망의 손실 함수에 소프트 제약 조건(soft constraints)으로 통합하여 물리 시스템의 역학을 모델링하는 강력한 도구이다. 그러나 PDE 기반의 손실 함수는 최적화 과정을 매우 불안정하게 만들며, 이로 인해 최적화 난이도가 높아지는 일명 'ill-conditioned' 문제가 발생한다.

특히, PINN의 성능은 사용되는 활성화 함수(Activation Function)의 선택에 매우 민감하게 반응한다. 예를 들어, 주기적인 특성을 가진 PDE를 풀 때는 sinusoidal 함수가 유리하고, 급격한 감쇠가 일어나는 시스템에서는 exponential 함수가 더 효과적이다. 하지만 현재까지는 최적의 활성화 함수를 찾기 위해 비효율적인 시행착오(trial-and-error) 방식에 의존해 왔으며, 이는 막대한 계산 자원과 도메인 지식을 요구한다. 따라서 본 논문의 목표는 PDE의 특성에 따라 최적의 활성화 함수를 자동으로 탐색하고 학습할 수 있는 적응형 활성화 함수(Adaptive Activation Function)를 제안하여 PINN의 최적화 난이도를 완화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 여러 후보 활성화 함수들의 선형 결합을 통해 해당 PDE 문제에 특화된 '전문화된 활성화 함수'를 학습하는 것이다. 이를 위해 저자들은 기존의 Adaptive Blending Units (ABU) 개념을 PINN의 특성에 맞게 최적화한 **ABU-PINN**을 제안한다.

주요 설계 전략은 다음과 같다.
1. **물리적 사전 지식 반영**: PDE 시스템의 특성(주기성, 급격한 감쇠 등)을 반영하여 sinusoidal 및 exponential 함수와 같은 기본 함수들을 후보 집합에 포함시킨다.
2. **매끄러움(Smoothness) 보장**: PDE 제약 조건의 고계도 미분(higher-order derivatives) 계산을 위해, ReLU와 같은 조각별 선형 함수(piece-wise linear functions)를 후보 집합에서 제거하여 연속적인 미분 가능성을 확보한다.
3. **탐색 공간 확장**: 각 후보 함수에 학습 가능한 기울기(adaptive slopes)를 도입하여 수렴 속도를 더욱 향상시킨다.

## 📎 Related Works

기존의 적응형 활성화 함수 연구들은 주로 이미지 분류와 같은 컴퓨터 비전 작업에 집중되어 왔으며, 다음과 같은 접근 방식들이 존재한다.
- **SLAF**: Taylor 근사를 기반으로 다항식 기저를 사용하여 함수를 학습한다.
- **PAU**: Padé 근사를 통해 유리 함수(rational functions) 형태의 탐색 공간을 구축한다.
- **ACON**: Maxout 계열 함수들의 매끄러운 근사치를 학습한다.
- **ABU**: 여러 후보 함수의 선형 결합을 통해 최적의 조합을 찾는다.

하지만 이러한 방식들을 PINN에 직접 적용하기에는 한계가 있다. SLAF의 다항식 기저는 고계도 미분 시 값이 폭발하거나 진동하는 문제가 있으며, PAU는 분모의 절대값으로 인해 미분 불연속성이 발생하여 학습이 불안정해진다. ACON은 매끄러움은 충족하지만 탐색 공간의 다양성이 부족하여 다양한 특성을 가진 PDE를 모두 처리하기 어렵다. ABU-PINN은 이러한 한계를 극복하기 위해 PINN에 특화된 후보 집합 구성과 매끄러운 조합 방식을 채택하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 학습 절차
PINN은 기본적으로 다음과 같은 손실 함수 $L(\theta)$를 최소화하도록 학습된다.
$$L(\theta) = L_{ic}(\theta) + L_{bc}(\theta) + L_{r}(\theta)$$
여기서 $L_{ic}$는 초기 조건(initial condition), $L_{bc}$는 경계 조건(boundary condition), 그리고 $L_{r}$은 PDE 잔차(residual) 손실을 의미한다. ABU-PINN은 이 네트워크 내부의 표준 활성화 함수를 적응형 활성화 함수로 대체하여, 네트워크가 학습 과정에서 PDE 시스템에 최적화된 함수 형태를 스스로 찾도록 한다.

### ABU-PINN의 상세 메커니즘
ABU-PINN은 후보 활성화 함수 집합 $F = \{\sigma_1, \sigma_2, \dots, \sigma_N\}$의 가중합으로 정의된다. 제안된 최종 활성화 함수 $f(x)$의 방정식은 다음과 같다.
$$f(x) = \sum_{i=1}^{N} G(\alpha_i) \sigma_i(\beta_i x)$$

각 구성 요소의 역할은 다음과 같다.
- $\sigma_i(\cdot)$: 사전 지식을 기반으로 선택된 후보 활성화 함수(예: $\sin, \exp, \tanh, \text{GELU}$ 등)이다.
- $\alpha_i$: 각 후보 함수의 중요도를 결정하는 학습 가능한 파라미터이다.
- $G(\cdot)$: 게이트 함수로, 본 논문에서는 주로 $\text{softmax}$를 사용하여 가중치의 합이 1이 되도록 제한함으로써 탐색 공간을 후보 함수들의 볼록 껍질(convex hull)로 한정한다.
- $\beta_i$: 각 함수에 적용되는 학습 가능한 기울기(adaptive slope)로, 수렴 속도를 개선한다.

### 추론 및 최적화 절차
학습 가능한 파라미터 $\alpha_i$와 $\beta_i$는 네트워크의 가중치 및 편향과 함께 역전파(backpropagation)를 통해 동시에 최적화된다. 이를 통해 뉴런별(neuron-wise) 또는 층별(layer-wise)로 서로 다른 최적의 활성화 함수 조합을 가질 수 있게 된다.

## 📊 Results

### 실험 설정
저자들은 1D Poisson 방정식(토이 예제), 1D 시간 종속 PDE(Convection, Burgers, Allen-Cahn, KdV, Cahn-Hilliard), 그리고 2D Navier-Stokes 방정식(Lid-driven cavity, Flow past a circular cylinder) 등 광범위한 벤치마크를 사용하여 성능을 측정하였다. 주요 지표로는 $L^2$ 상대 오차(relative error)를 사용하였다.

### 주요 결과
1. **표준 활성화 함수 대비 우위**: ABU-PINN은 모든 테스트 케이스에서 단일 표준 활성화 함수를 사용했을 때보다 낮은 오차율을 기록하였다. 특히 Convection 방정식에서는 최적의 표준 함수 대비 오차를 83% 감소시켰다.
2. **다른 적응형 함수 대비 성능**: SLAF, PAU, ACON과 비교했을 때 ABU-PINN은 훨씬 안정적이고 정확한 결과를 보였다. PAU는 미분 불연속성으로 인해 학습 불안정성이 관찰되었으며, ACON은 유연성 부족으로 성능 향상에 한계가 있었다.
3. **2D 및 역문제(Inverse Problem) 해결**: 2D Navier-Stokes 방정식 실험에서도 우수한 성능을 보였으며, 특히 물성치 $\lambda_1, \lambda_2$를 추정하는 역문제에서 GELU 대비 $\lambda_1$ 추정 오차를 38% 감소시키는 등 강력한 성능을 입증하였다.
4. **계산 비용**: ABU-PINN의 계산 시간은 표준 함수 대비 약 3.1배 증가하였으나, 정확도 측면에서의 이득이 훨씬 크므로 이는 수용 가능한 수준으로 평가된다.

## 🧠 Insights & Discussion

### Neural Tangent Kernel (NTK) 관점의 해석
본 논문은 ABU-PINN의 성능 향상 원인을 NTK 관점에서 분석한다. 표준 신경망의 NTK는 고정된 활성화 함수에 의해 결정되지만, ABU-PINN은 가중치 $\alpha_i$를 학습함으로써 **학습 가능한 NTK(learnable NTK)**를 구축한다. 실험적으로 ABU-PINN은 NTK의 고유값 스펙트럼(eigenvalue spectrum)을 조정하여 평균 고유값을 높였으며, 이는 이론적으로 학습 손실의 더 빠른 수렴 속도로 이어진다.

### 강점 및 한계
- **강점**: 물리적 사전 지식을 신경망 구조(활성화 함수)에 직접 통합함으로써, 단순한 데이터 피팅을 넘어 PDE의 수학적 특성을 반영한 모델링이 가능하다는 점이다. 또한, 학습된 계수 $\alpha_i$를 통해 어떤 함수가 해당 문제 해결에 중요했는지 해석할 수 있는 가능성을 제시한다.
- **한계**: 본 연구는 정수 차수 PDE(integer-order PDEs)만을 다루었으며, 분수 차수(fractional)나 확률적(stochastic) PDE에 대한 검증은 이루어지지 않았다. 또한, 이론적 분석보다는 실험적 결과에 의존한 부분이 많아, PDE 제약 조건이 적응형 알고리즘에 미치는 영향에 대한 엄밀한 이론적 증명은 향후 과제로 남겨두었다.

## 📌 TL;DR

이 논문은 PINN의 최적화 난이도를 해결하기 위해 물리적 특성을 반영한 후보 함수들의 조합을 학습하는 **ABU-PINN**을 제안한다. 제안된 방법은 주기성이나 급격한 감쇠와 같은 PDE의 특성에 맞춰 활성화 함수를 자동으로 최적화하며, NTK의 고유값을 개선하여 수렴 속도와 정확도를 획기적으로 높인다. 이는 향후 복잡한 물리 시스템을 모델링하는 딥러닝 구조 설계에 있어 활성화 함수의 자동 최적화가 중요한 역할을 할 수 있음을 시사한다.