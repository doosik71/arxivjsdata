# Imitation Learning of Stabilizing Policies for Nonlinear Systems

Sebastian East (2021)

## 🧩 Problem to Solve

본 논문은 알려진 시스템 동역학(known system dynamics)에 대해 **안정성(stability)이 보장되는 제어 정책을 학습하는 모방 학습(Imitation Learning, IL)** 문제를 다룬다.

전통적인 모방 학습은 전문가의 상태-입력 쌍 데이터 $(\hat{x}, \hat{u})$를 사용하여 지도 학습(supervised learning) 방식으로 제어기를 설계한다. 그러나 단순한 지도 학습으로 학습된 정책은 실제 시스템에 적용했을 때 시스템을 발산하게 만들 수 있으며, 특히 안전이 중요한(safety-critical) 애플리케이션에서는 제어 정책의 안정성 보장이 필수적이다.

기존 연구들은 주로 선형 시불변(Linear Time-Invariant, LTI) 시스템과 선형 제어기에 집중해 왔다. 하지만 실제 시스템은 특정 평형점 근처에서만 선형적으로 동작하며, 비선형 시스템을 안정화하기 위해서는 비선형 제어기가 필요할 때가 많다. 따라서 본 논문의 목표는 **다항식 시스템(polynomial systems)과 다항식 제어기(polynomial controllers)로 확장하여, 전역적 안정성(global stability)이 보장되는 모방 학습 방법론을 제시하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Sum of Squares (SOS) 기법을 사용하여 비선형 시스템의 안정성 제약 조건을 계산 가능한 형태(tractable form)로 변환**하여 모방 학습 프레임워크에 통합하는 것이다.

구체적으로, Lyapunov 안정성 이론을 기반으로 한 비선형 제약 조건을 SOS 최적화 문제로 변환함으로써, 비선형 제어기의 계수를 결정하는 최적화 문제를 구성한다. 또한, 이 문제가 가지는 Biconvex 특성을 해결하기 위해 **ADMM(Alternating Direction Method of Multipliers)**과 **Projected Gradient Descent** 알고리즘을 제안하여 수치적인 해를 구하는 방법을 제시한다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 한계를 지적한다.

1.  **Palan et al. [2]:** 'Kalman constraint'를 도입하여 LTI 시스템에 대해 최적이며 안정적인 제어기를 학습시켰으나, 시스템이 선형이라는 제약이 있다.
2.  **Havens and Hu [5]:** 선형 행렬 부등식(Linear Matrix Inequalities, LMI)을 사용하여 LTI 시스템의 안정성과 강건성을 보장했다.
3.  **Yin et al. [7]:** 신경망 제어기를 사용했으나, 안정성 보장을 위한 ADMM 과정에서 비볼록(nonconvex) 최적화 문제를 반복적으로 풀어야 하므로 지역 최적점(local minima)에 빠질 위험이 있다.

**차별점:** 기존 연구들이 LTI 시스템과 선형 제어기에 국한되었던 것과 달리, 본 연구는 다항식 시스템과 제어기로 범위를 확장하여 더 넓은 상태 공간에서 안정성을 보장할 수 있게 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 제어기 구조
모방 학습을 지도 학습 문제로 정의한다. 전문가 데이터 $\mathcal{D} = \{(\hat{x}_1, \hat{u}_1), \dots, (\hat{x}_N, \hat{u}_N)\}$가 주어졌을 때, 다항식 제어기 $\pi(x)$를 다음과 같이 설계한다.
$$\pi(x) = K(x)Z(x)$$
여기서 $Z(x)$는 미리 결정된 단항식(monomials) 벡터이며, $K(x)$는 최적화를 통해 결정해야 할 계수를 가진 다항식 행렬이다.

### 2. 최적화 목표 함수
최적의 정책 $\pi^\star$는 다음과 같은 모방 손실(imitation loss) $\ell$과 정규화(regularization) 항 $r$의 합을 최소화하는 방향으로 결정된다.
$$\pi^\star := \arg\min_{\pi} \frac{1}{N} \sum_{i=1}^{N} \ell(\pi(\hat{x}_i), \hat{u}_i) + r(\pi)$$

### 3. 안정성 제약 조건 (Stabilizing Constraint)
시스템 동역학이 $\dot{x} = A(x)Z(x) + B(x)u$로 주어질 때, Lyapunov 함수 $V(x) = Z^\top(x)P^{-1}(\tilde{x})Z(x)$를 정의한다. 시스템이 안정하기 위해서는 다음 두 조건이 만족되어야 한다.
- $P(x) \succ 0$
- $\dot{V}(x) < 0$ (시간 미분값이 음수여야 함)

이 조건들을 구체화하면 다음과 같은 행렬 부등식 형태가 된다.
$$P(\tilde{x})A^\top(x)M^\top(x) + M(x)A(x)P(\tilde{x}) + F^\top(x)B^\top(x)M^\top(x) + M(x)B(x)F(x) - \sum_{j \in J} \frac{\partial P}{\partial x_j}(\tilde{x})[A^j(x)Z(x)] \prec 0$$
여기서 $M(x)$는 $Z(x)$의 자코비안(Jacobian) 행렬이며, $F(x)$는 제어기 $K(x)$와 $P(\tilde{x})$의 관계($K(x) = F(x)P^{-1}(\tilde{x})$)를 정의하는 변수이다.

### 4. Sum of Squares (SOS) 근사
위의 다항식 부등식 제약 조건은 계산적으로 다루기 어렵다. 이를 해결하기 위해 **SOS 기법**을 도입한다. 다항식 $f(x)$가 SOS라면 $f(x) = z^\top(x)Qz(x)$ (단, $Q \succeq 0$) 형태로 표현될 수 있으며, 이는 반양정치 계획법(Semidefinite Programming, SDP)으로 변환 가능하다. 본 논문은 이를 통해 안정성 조건을 다음과 같은 행렬 부등식으로 변환한다.
- $v^\top[P(\tilde{x}) - \epsilon_1 I]v = [z_1(x) \otimes v]^\top Q_1 [z_1(x) \otimes v], \quad Q_1 \succeq 0$
- $\dot{V}$ 관련 식 $\dots = [z_2(x) \otimes v]^\top Q_2 [z_2(x) \otimes v], \quad Q_2 \preceq 0$

### 5. 학습 절차 및 알고리즘
최종 최적화 문제는 다음과 같다.
$$\min_{\{K_i\}, \{F_i\}, \{P_i\}, Q_1, Q_2} \frac{1}{N} \sum_{i=1}^{N} \ell(K(\hat{x}_i)Z(\hat{x}_i), \hat{u}_i) + r(\{K_i\})$$
$$\text{s.t. } K(x)P(\tilde{x}) = F(x) \text{ 및 SOS 제약 조건 (7)-(10)}$$

이 문제는 $K(x)P(\tilde{x}) = F(x)$라는 제약 조건 때문에 **Biconvex** (두 변수 중 하나를 고정하면 볼록함) 특성을 가진다. 이를 풀기 위해 두 가지 휴리스틱 알고리즘을 제안한다.
- **ADMM:** $K$와 $(F, P, Q)$를 번갈아 가며 업데이트하는 방식이다.
- **Projected Gradient Descent:** 목적 함수에 대해 경사 하강법을 수행한 후, SOS 제약 조건이 만족되는 집합으로 투영(projection)하는 방식이다.

## 📊 Results

### 실험 설정
- **시스템 1:** 비선형 시스템 + 선형 전문가 제어기.
- **시스템 2:** 선형 시스템 + 비선형(3차) 전문가 제어기.
- **지표:** 반복 횟수(Iteration)에 따른 모방 손실(Imitation Loss)의 감소 추이.
- **환경:** Python 3.8, Sympy, CVXPY, SCS solver, Jax 사용.

### 주요 결과
1.  **비선형 시스템 실험:** ADMM은 매우 빠르게 수렴하며 전문가 제어기에 근접한 해를 찾았다. 반면, Projected Gradient Descent는 지역 최적점(local minima)에 자주 빠지며 수렴 속도가 현저히 느렸다.
2.  **비선형 제어기 실험:** 전문가 제어기가 3차 다항식인 경우, 학습 제어기의 차수 제한으로 인해 전문가와 완전히 일치하는 해를 찾을 수는 없었다. 그럼에도 불구하고 ADMM이 Projected Gradient Descent보다 우수한 성능을 보였으나, 선형 제어기 실험 때보다는 수렴 속도가 느리고 손실 값이 높았다. 이는 $K(x)P(\tilde{x}) = F(x)$ 제약 조건의 항이 많아지면서 최적화 난이도가 증가했기 때문으로 분석된다.

## 🧠 Insights & Discussion

**강점:**
- 비선형 시스템에 대해 전역적 안정성을 수학적으로 보장하는 모방 학습 프레임워크를 제안하였다.
- SOS 기법을 통해 복잡한 비선형 안정성 제약을 SDP 문제로 변환하여 실질적인 계산이 가능하게 만들었다.
- ADMM이 Biconvex 최적화 문제에서 Projected Gradient Descent보다 효율적인 수렴 성능을 보임을 입증하였다.

**한계 및 비판적 해석:**
- **보수성(Conservatism):** 본 논문은 전역 안정성(global stability)을 추구한다. 하지만 실제로는 전역 안정성을 만족하는 해를 찾기 매우 어려우며, 이는 제약 조건이 너무 엄격하여(conservative) 해가 존재하지 않는(infeasible) 경우가 많음을 의미한다.
- **차수 선택의 문제:** 실험 2에서 보듯, 제어기나 Lyapunov 함수의 차수를 잘못 설정하면 전문가의 동작을 제대로 모방하지 못하는 '대응 문제(correspondence issue)'가 발생한다.

## 📌 TL;DR

본 논문은 비선형 다항식 시스템에서 **안정성이 보장되는 제어 정책을 학습하기 위해 SOS(Sum of Squares) 기법을 적용한 모방 학습 방법**을 제안한다. Lyapunov 안정성 조건을 SDP 형태로 변환하여 최적화 문제로 구성하였으며, **ADMM 알고리즘이 비선형 제어기 학습에서 효율적**임을 보였다. 이 연구는 안전이 필수적인 비선형 시스템의 제어기 설계에 있어 이론적 보장과 학습 기반 접근법을 결합했다는 점에서 중요한 의의가 있으며, 향후 국소 영역(bounded domains)에서의 안정성 확장 연구로 이어질 가능성이 크다.