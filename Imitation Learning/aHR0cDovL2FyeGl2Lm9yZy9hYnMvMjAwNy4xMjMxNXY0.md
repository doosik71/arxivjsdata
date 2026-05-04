# Bayesian Robust Optimization for Imitation Learning

Daniel S. Brown, Scott Niekum, Marek Petrik (2020)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning)에서 에이전트가 전문가의 시연(demonstrations)에 포함되지 않은 새로운 상태(novel states)에 직면했을 때, 어떤 행동을 취해야 하는지에 대한 문제를 다룬다. 

역강화학습(Inverse Reinforcement Learning, IRL)은 매개변수화된 보상 함수를 학습함으로써 새로운 상태로의 일반화 능력을 제공하지만, 진정한 보상 함수(true reward function)와 그에 따른 최적 정책에 대한 불확실성(uncertainty)이 여전히 존재한다. 기존의 안전한 모방 학습 방식은 주로 최악의 상황을 가정하는 maxmin 프레임워크를 사용하여 매우 보수적인 정책을 생성하는 경향이 있으며, 반대로 위험 중립적인(risk-neutral) IRL 방식은 평균이나 MAP(Maximum A Posteriori) 보상 함수에 최적화하여 지나치게 공격적이고 불안정한 정책을 생성하는 문제가 있다.

따라서 본 연구의 목표는 이러한 두 극단 사이의 가교 역할을 하는 프레임워크를 제공하는 것이다. 즉, 보상 함수의 불확실성에서 오는 인식적 위험(epistemic risk)과 기대 수익(expected return) 사이의 트레이드오프를 사용자의 위험 감수 성향에 따라 조절할 수 있는 강건한 정책 최적화 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 베이지안 보상 함수 추론(Bayesian reward function inference)과 조건부 가치 위험(Conditional Value at Risk, CVaR)이라는 정합적 위험 척도(coherent risk measure)를 결합하여, 기대 수익과 위험 사이의 균형을 맞추는 정책을 최적화하는 것이다.

주요 기여 사항은 다음과 같다:
1. **BROIL(Bayesian Robust Optimization for Imitation Learning) 제안**: 불확실한 보상 함수 하에서 기대 수익과 CVaR의 균형을 직접적으로 최적화하는 최초의 모방 학습 프레임워크를 제시한다.
2. **효율적인 선형 계획법(Linear Programming, LP) 정식화**: BROIL의 최적 정책을 다항 시간 내에 계산할 수 있는 효율적인 LP 수식을 도출하였다.
3. **두 가지 목적 함수 제안**: 단순히 불확실성에 대해 강건한 'Robust Objective'와 전문가 시연 대비 후회를 최소화하는 'Baseline Regret Objective'를 제안하고 비교 분석하였다.
4. **성능 입증**: BROIL이 기존의 위험 민감형 및 위험 중립형 IRL 알고리즘보다 더 나은 기대 수익과 강건성을 제공하며, 위험 회피 수준에 따라 다양한 솔루션을 생성할 수 있음을 실험적으로 증명하였다.

## 📎 Related Works

기존의 IRL 연구들은 보상 함수의 모호성을 해결하기 위해 다양한 접근 방식을 취해왔다.
- **Robust IRL (Maxmin 접근법)**: FPL-IRL, LPAL, GAIL 등은 최악의 보상 시나리오에서도 성능을 보장하려 한다. 그러나 이는 환경을 지나치게 적대적으로 간주하여 실무적으로는 너무 보수적인 행동(overly conservative)을 유발하는 한계가 있다.
- **Risk-Neutral IRL**: MaxEnt IRL이나 Bayesian IRL의 일부 방식은 평균 보상이나 MAP 보상에 최적화한다. 이는 불확실성에 대해 지나치게 낙관적이며, 고위험 도메인(예: 자율주행, 의료)에서는 치명적인 결과를 초래할 수 있다.
- **Risk-Sensitive IRL**: RS-GAIL 등은 위험 회피 성향을 가진 전문가를 모방하려 하지만, 이는 전이 확률의 불확실성(aleatoric risk)에 집중하며 본 논문이 다루는 보상 함수의 불확실성(epistemic risk)과는 다른 문제이다.
- **VaR-BIRL**: Value at Risk(VaR)를 사용하여 강건성을 측정하려 했으나, VaR의 최적화는 NP-hard 문제이며 꼬리 위험(tail risk)을 무시한다는 단점이 있다.

BROIL은 이러한 한계를 극복하기 위해 convex 최적화가 가능한 CVaR를 도입하고, 베이지안 사후 분포를 통해 보상 함수의 불확실성을 정량적으로 처리함으로써 평균 성능과 최악의 성능을 동시에 고려한다.

## 🛠️ Methodology

### 1. MDP 및 선형 보상 함수 정의
환경을 MDP $\langle S, A, r, P, \gamma, p_0 \rangle$로 모델링하며, 보상 함수 $r$은 특징 행렬 $\Phi$와 가중치 벡터 $w$의 선형 결합인 $r = \Phi w$로 가정한다. 정책 $\pi$의 기대 수익은 상태-행동 점유 빈도(state-action occupancy frequencies) $u_\pi$를 사용하여 $\rho(\pi, r) = u_\pi^T r$로 표현된다.

### 2. 위험 척도: CVaR
본 논문은 $\alpha \in [0, 1]$ 범위의 위험 회피 파라미터를 사용하여 CVaR를 정의한다. $\text{VaR}_\alpha$가 $(1-\alpha)$-분위수의 최악 결과라면, $\text{CVaR}_\alpha$는 $\text{VaR}_\alpha$보다 낮은 값들을 가진 분포의 기댓값이다.
$$\text{CVaR}_\alpha[X] = \mathbb{E}[X | X \le \text{VaR}_\alpha[X]]$$
이는 convex 함수이며, 다음과 같은 최적화 형태로 변환 가능하다:
$$\text{CVaR}_\alpha[X] = \max_{\sigma \in \mathbb{R}} \left( \sigma - \frac{1}{1-\alpha} \mathbb{E}[(\sigma - X)^+] \right)$$

### 3. BROIL 목적 함수
BROIL은 기대 수익(Expected Return)과 $\text{CVaR}_\alpha$ 사이의 가중치 $\lambda \in [0, 1]$를 도입하여 다음과 같은 목적 함수를 최대화한다:
$$\max_{\pi \in \Pi} \lambda \cdot \mathbb{E}[\psi(\pi, R)] + (1-\lambda) \cdot \text{CVaR}_\alpha[\psi(\pi, R)]$$
여기서 $\psi(\pi, R)$은 성능 측정 지표이며, $\lambda=1$이면 위험 중립적(Risk-neutral), $\lambda=0$이면 완전히 강건한(Fully robust) 정책이 된다.

### 4. 선형 계획법(LP) 정식화
베이지안 IRL을 통해 샘플링된 $N$개의 보상 함수 집합 $\{R_1, \dots, R_N\}$과 각 확률 질량 $p_R$이 주어졌을 때, 위 문제는 다음과 같은 LP로 변환되어 효율적으로 풀 수 있다:
$$\text{maximize}_{u, \sigma} \quad \lambda \cdot p_R^T \psi(\pi_u, R) + (1-\lambda) \cdot \left( \sigma - \frac{1}{1-\alpha} p_R^T [\sigma \cdot 1 - \psi(\pi_u, R)]^+ \right)$$
$$\text{subject to} \quad \sum_{a \in A} (I - \gamma P_a^T) u_a = p_0, \quad u \ge 0$$

### 5. 성능 측정 지표 $\psi$의 두 가지 설정
- **Robust Objective**: $\psi(\pi, R) = \rho(\pi, R)$. 단순히 기대 수익의 강건성을 최적화한다.
- **Baseline Regret Objective**: $\psi(\pi, R) = \rho(\pi, R) - \rho(\pi_E, R)$. 전문가 정책 $\pi_E$ 대비 성능 차이(Regret)의 강건성을 최적화한다. 실제로는 전문가의 특징 기대값 $\hat{\mu}_E$를 사용하여 $\psi(\pi_u, R) = R^T u - W^T \hat{\mu}_E$로 계산한다.

## 📊 Results

### 1. Zero-shot Robust Policy Optimization (기계 교체 문제)
전문가 시연 없이 사전 분포(prior)만을 사용하여 강건한 정책을 학습하는 실험이다.
- **설정**: 4개의 상태를 가진 기계 교체 MDP. 부품 교체 비용은 정규분포, 교체하지 않았을 때의 비용은 꼬리가 긴 감마 분포를 따른다.
- **결과**: $\lambda=1$인 위험 중립 정책은 평균 비용만을 고려해 절대 수리하지 않는 선택을 하지만, $\lambda$가 감소함에 따라 정책은 점진적으로 수리 확률을 높인다. $\lambda=0$일 때 최악의 비용 시나리오를 피하기 위해 가장 보수적으로 수리하는 정책이 생성된다. 이는 BROIL이 위험-수익 간의 효율적 경계(Efficient Frontier)를 효과적으로 탐색함을 보여준다.

### 2. Ambiguous Demonstrations (모호한 시연 문제)
시연 데이터가 상태 공간의 일부만 커버하여 보상 함수에 대한 불확실성이 큰 상황을 가정한 그리드월드 실험이다.
- **비교 대상**: MaxEnt IRL (위험 중립), LPAL (Maxmin 강건), BROIL (Robust 및 Regret 설정).
- **결과**:
    - **MaxEnt IRL**: 위험을 무시하고 최단 경로(빨간색 셀 통과)를 택하는 경향이 있어 위험도가 높다.
    - **LPAL**: 지나치게 보수적이며, 전문가의 특징값과 정확히 일치시키려다 보니 정책이 매우 확률적(stochastic)이 되어 빠르게 목표에 도달하지 못한다.
    - **BROIL**: $\lambda$ 값에 따라 위험을 적절히 회피하면서도 효율적인 경로를 찾는 정책의 집합을 생성한다. 특히 'Baseline Regret' 설정은 전문가가 방문하지 않은 위험 지역(빨간색 셀)을 효과적으로 회피하면서도 성능을 극대화하였다.

### 3. 정량적 분석 및 확장성
- **성능**: Figure 5의 효율적 경계 그래프에서 BROIL은 기대 수익과 강건성(CVaR) 모두에서 MaxEnt IRL과 LPAL을 압도(dominate)하는 것으로 나타났다.
- **계산 효율성**: 수천 개의 상태와 수천 개의 보상 함수 샘플이 있는 경우에도 개인용 노트북에서 수백 초 내에 LP를 해결할 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점
- **제어 가능한 위험 수준**: 단일 알고리즘으로 $\lambda$ 파라미터 조절만으로 위험 중립적 행동부터 극도로 보수적인 행동까지 연속적으로 생성할 수 있다.
- **이론적 기반과 실용성의 조화**: CVaR라는 정합적 위험 척도를 사용하면서도, 이를 LP로 정식화하여 계산 복잡도를 낮추고 실용적인 최적화 가능성을 확보했다.
- **에피스테믹 리스크 처리**: 단순한 전이 확률의 불확실성이 아니라, 학습된 보상 함수 자체의 불확실성을 다룸으로써 모방 학습의 근본적인 문제인 '시연 외 상태에서의 행동' 문제를 논리적으로 해결했다.

### 한계 및 미해결 과제
- **샘플링 의존성**: 베이지안 IRL을 통해 보상 함수의 사후 분포를 얻기 위해 MCMC 샘플링을 사용하는데, 이는 복잡한 제어 작업에서 계산 비용이 증가하는 병목 지점이 될 수 있다.
- **이산 상태 공간**: 본 논문의 LP 정식화는 이산적인 상태-행동 공간을 전제로 한다. 연속적인 상태/행동 공간으로 확장하기 위해서는 정책 경사법(Policy Gradient)이나 근사 선형 계획법(Approximate LP) 등의 추가적인 기법이 필요하다.

### 비판적 해석
본 연구는 "안전함"의 정의를 보상 함수의 불확실성에 대한 강건성으로 정의하였다. 하지만 이는 에이전트가 가진 '인식'의 불확실성에 대한 대응일 뿐, 실제 환경의 물리적 제약이나 절대적인 안전 보장(Safety Guarantee)을 의미하는 것은 아니다. 따라서 BROIL 정책이 수학적으로 강건하더라도, 사용자가 설정한 $\lambda$나 $\alpha$ 값이 실제 환경의 위험 임계치와 맞지 않는다면 여전히 위험한 행동을 할 가능성이 존재한다.

## 📌 TL;DR

본 논문은 모방 학습에서 보상 함수의 불확실성으로 인해 발생하는 위험을 관리하기 위해, 기대 수익과 조건부 가치 위험(CVaR)의 가중 합을 최적화하는 **BROIL** 프레임워크를 제안한다. 이를 통해 사용자는 위험 감수 성향($\lambda$)에 따라 최적의 효율성과 최악의 상황 대비 강건성 사이의 균형을 맞춘 정책을 얻을 수 있으며, 이는 효율적인 선형 계획법(LP)을 통해 계산된다. 실험 결과 BROIL은 기존의 위험 중립적 또는 극단적 강건성 기반 IRL보다 우수한 성능을 보였으며, 이는 자율주행이나 의료 로봇과 같이 위험 관리가 필수적인 도메인의 모방 학습에 중요한 기여를 할 것으로 평가된다.