# Continuous Online Learning and New Insights to Online Imitation Learning

Jonathan Lee, Ching-An Cheng, Ken Goldberg, Byron Boots (2019)

## 🧩 Problem to Solve

본 논문은 전통적인 Online Learning의 적대적(adversarial) 설정이 실제 반복적 알고리즘(iterative algorithms)에서 나타나는 정규성(regularity)을 충분히 반영하지 못한다는 문제에서 출발한다. 특히 Markov Decision Processes (MDPs)와 관련된 최적화 문제에서는 기대값(expectation)을 취하는 과정에서 자연스럽게 연속성(continuity) 특성이 나타나는데, 이를 무시한 기존의 적대적 분석은 이론적 결과가 지나치게 보수적(overly conservative)인 경향이 있다.

따라서 본 논문의 목표는 학습자의 결정에 따라 손실 함수의 그래디언트가 연속적으로 변하는 새로운 설정인 **Continuous Online Learning (COL)**을 정립하고, 이를 통해 달성하기 어렵다고 알려진 **sublinear dynamic regret**의 달성 조건과 효율적인 알고리즘을 연구하는 것이다. 최종적으로는 이러한 이론적 통찰을 Online Imitation Learning (IL)에 적용하여 학습 안정성을 분석하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **Continuous Online Learning (COL) 프레임워크 제안**: 손실 함수가 학습자의 현재 결정에 의존하는 bifunction $f$에 의해 결정되는 새로운 설정을 도입하여, 실제 문제에서 나타나는 정규성을 이론적으로 모델링하였다.
2.  **Dynamic Regret과 Equilibrium Problems (EPs)의 등가성 증명**: COL 설정에서 sublinear dynamic regret을 달성하는 것이 특정 Equilibrium Problem 또는 Variational Inequality (VI)를 해결하는 것과 본질적으로 동일함을 수학적으로 증명하였다.
3.  **Regret Reduction 방법론 제시**: $(\alpha, \beta)$-regularity 조건을 통해 dynamic regret을 static regret과 평형점(equilibrium point)으로의 수렴 속도로 환원(reduction)시켜 분석할 수 있는 이론적 토대를 마련하였다.
4.  **Online Imitation Learning에의 적용**: Online IL 문제를 COL 문제로 정식화하여, 특정 조건($\alpha > \beta$) 하에서 유일한 최적 정책의 존재성을 보이고, 결정론적 및 확률적 피드백 환경에서의 수렴성과 dynamic regret bound를 제시하였다.

## 📎 Related Works

기존의 Online Learning 연구들은 주로 최악의 경우(worst-case)를 상정하는 adversarial setting에 집중해 왔다. 이러한 접근 방식은 static regret을 제한하는 데는 효과적이지만, 매 라운드마다 최적의 결정을 추적해야 하는 dynamic regret 분석에는 한계가 있다. 일부 연구들이 적대적 환경에 제약을 가해 dynamic regret을 분석하려 했으나, 이는 사례별(case-by-case) 분석에 그치거나 학습자의 결정에 따라 적대성이 변하는 상황(예: Online IL)을 모델링하기 어려웠다.

Online Imitation Learning의 대표적 알고리즘인 DAGGER 등은 Online Learning의 형식을 띠고 있으며, 기존 연구들은 주로 static regret 관점에서 분석하였다. 최근 연구들에서 dynamic regret의 중요성이 제기되었으나, 본 논문은 COL이라는 일반화된 프레임워크를 통해 더 정밀한 수렴 조건과 non-asymptotic bound를 제공함으로써 기존 분석을 개선하였다.

## 🛠️ Methodology

### 1. Continuous Online Learning (COL) 정의
COL에서는 상대방(opponent)이 학습자에게 알려지지 않은 bifunction $f: (x, x') \mapsto f_x(x') \in \mathbb{R}$를 가지고 있다고 가정한다. 라운드 $n$에서 학습자가 결정 $x_n$을 내리면, 상대방은 다음과 같은 손실 함수를 제공한다.

$$l_n(x) := f_{x_n}(x)$$

여기서 $x$는 쿼리 인자(query argument)이며, $x'$는 성능을 평가받는 결정 인자(decision argument)이다. COL의 핵심은 $\forall x' \in X$에 대해 $\nabla f_x(x')$가 $x$에 대해 연속적인 맵(continuous map)이라는 점이다. 또한, 피드백 신호에 노이즈나 편향을 추가하여 확률적/적대적 요소를 포함할 수 있도록 설계하였다.

### 2. $(\alpha, \beta)$-Regularity
효율적인 알고리즘 설계를 위해 본 논문은 COL 문제에 다음과 같은 정규성 조건을 도입한다.

-   **$\alpha$-strong convexity**: 모든 $x \in X$에 대해 $f_x(\cdot)$가 $\alpha$-강볼록 함수여야 한다.
-   **$\beta$-Lipschitz continuity**: $\nabla f_\cdot(x)$가 $\beta$-립시츠 연속 맵이어야 한다.

이 조건이 만족될 때, 맵 $\nabla f_x(x)$는 $(\alpha - \beta)$-strong monotonicity를 가지게 되며, 이는 Variational Inequality (VI) 문제의 유일한 해가 존재함을 보장한다.

### 3. Dynamic Regret의 환원 (Reduction)
본 논문은 dynamic regret을 다음과 같이 상한(upper bound) 지을 수 있음을 증명하였다.

$$\text{Regret}^d_N \leq \min\left\{G \sum_{n=1}^N \Delta_n, \text{Regret}^s_N(x^*)\right\} + \sum_{n=1}^N \min\left\{\beta D_X \Delta_n, \frac{\beta^2}{2\alpha}\Delta_n^2\right\}$$

여기서 $x^*$는 평형점이며, $\Delta_n = \|x_n - x^*\|$이다. 이는 평형점이 존재할 때, 결정 $x_n$이 $x^*$ 주변에 머문다면 dynamic regret이 static regret과 유사하게 동작함을 의미한다. 특히 $\alpha > \beta$인 경우, sublinear dynamic regret 달성 문제가 linear loss 하의 static regret 문제로 환원된다.

## 📊 Results

### 1. Online Imitation Learning (IL) 적용
Online IL의 목적 함수 $\min_{\pi \in \Pi} \mathbb{E}_{s \sim d_\pi}[c(s, \pi; \pi^*)]$에서, 라운드 $n$의 손실 함수를 $l_n(\pi) = f_{\pi_n}(\pi) = \mathbb{E}_{s \sim d_{\pi_n}}[c(s, \pi; \pi^*)]$로 정의함으로써 이를 COL 문제로 변환하였다.

### 2. 이론적 분석 결과
-   **유일 해의 존재성**: $\alpha > \beta$ 조건이 만족되면, 자신의 분포 상에서 최적인 유일한 정책 $\hat{\pi}$가 존재한다.
-   **결정론적 피드백 (Deterministic Feedback)**: Online Gradient Descent (OGD)를 사용할 때, 적절한 stepsize $\eta$를 선택하면 정책 $\pi_n$이 $\hat{\pi}$로 선형 수렴(linear convergence)하며, $\text{Regret}^d_N = O(1)$임을 보였다.
-   **확률적 피드백 (Stochastic Feedback)**: $\eta_n = 1/\sqrt{n}$인 OGD 알고리즘을 사용할 때, 기대 dynamic regret이 다음과 같이 sublinear함을 증명하였다.

$$\mathbb{E}[\text{Regret}^d_N] = O(\sqrt{N})$$

이는 MDP에서 샘플링으로 인해 발생하는 노이즈가 있음에도 불구하고, 학습이 안정적으로 이루어짐을 정량적으로 보여준다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 추상적인 Online Learning 이론과 구체적인 MDP 기반 최적화 문제 사이의 간극을 **Bifunction**과 **Continuous Online Learning**이라는 개념으로 메운 점이다. 특히 sublinear dynamic regret이라는 어려운 목표를 Equilibrium Problem과 연결하여, 기존의 VI 및 EP 문헌에서 검증된 알고리즘들을 그대로 적용할 수 있는 이론적 근거를 제시하였다.

비판적 관점에서 볼 때, 본 논문은 $\alpha > \beta$라는 강한 가정을 전제로 한다. 이는 강볼록성($\alpha$)이 분포의 변화율($\beta$)보다 커야 함을 의미하는데, 실제 복잡한 환경에서 이 조건을 정밀하게 측정하거나 보장하는 것은 여전히 어려운 과제이다. 또한, 제안된 bound들이 이론적인 상한을 제시하지만, 실제 하이퍼파라미터(예: $\eta$) 설정에 따른 실증적인 성능 변화에 대한 실험적 데이터는 본 텍스트에 명시되어 있지 않다.

그럼에도 불구하고, Online IL의 수렴 조건을 $\alpha > \beta$라는 단순한 형태로 정립하고, stochastic feedback 환경에서도 sublinear dynamic regret이 가능함을 보인 것은 매우 중요한 학술적 진보라고 판단된다.

## 📌 TL;DR

이 논문은 학습자의 결정에 따라 손실 함수가 연속적으로 변하는 **Continuous Online Learning (COL)** 프레임워크를 제안하여, 어렵다고 알려진 **sublinear dynamic regret** 문제를 **Equilibrium Problem (EP)**으로 환원하여 해결하였다. 이를 Online Imitation Learning에 적용하여 $\alpha > \beta$ 조건 하에서 최적 정책으로의 수렴성과 $O(\sqrt{N})$의 dynamic regret bound를 이론적으로 증명하였다. 이 연구는 향후 MDP 기반의 다양한 반복 최적화 알고리즘을 분석하는 새로운 표준 틀을 제공할 가능성이 크다.