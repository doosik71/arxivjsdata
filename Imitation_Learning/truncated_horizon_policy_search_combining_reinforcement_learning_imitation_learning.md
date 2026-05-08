# TRUNCATEDHORIZONPOLICYSEARCH: COMBINING REINFORCEMENTLEARNING& IMITATIONLEARNING

Wen Sun, J. Andrew Bagnell, Byron Boots (2018)

## 🧩 Problem to Solve

강화 학습(Reinforcement Learning, RL)은 현대 딥러닝 기술과 결합하여 로봇 제어, 비디오 게임 등 복잡한 순차적 의사결정 문제에서 큰 발전을 이루었지만, 성공을 위해서는 막대한 훈련 데이터와 계산 자원을 요구한다. 이러한 비효율성을 개선하기 위해 모방 학습(Imitation Learning, IL)과 같은 접근 방식이 탐색되었다. 모방 학습은 전문가의 시연(expert demonstrations)이나 cost-to-go oracle을 활용하여 학습 과정을 안내함으로써 무작위 탐색을 줄이고 샘플 효율성을 높인다.

그러나 모방 학습은 다음과 같은 본질적인 한계를 지닌다:

1. **전문가의 성능 한계**: 학습된 정책의 성능이 전문가의 성능에 제한되며, 실제 환경에서 전문가는 종종 최적이 아닐 수 있다. 기존의 이론적 보증을 가진 모방 학습 알고리즘(예: DAgger, AGGREVATE)은 전문가 정책만큼의 성능 또는 전문가 정책 대비 한 단계 개선된 성능만을 보장한다. 이는 서브옵티멀(sub-optimal)한 전문가가 있을 경우 서브옵티멀한 정책을 반환할 수 있음을 의미한다.
2. **개선량의 불확실성**: 기존의 IL과 RL 결합 방식(예: Chang et al., 2015)은 전문가 정책보다 더 나은 지역 최적 정책(locally optimal policy)을 찾을 수 있지만, 전문가 대비 얼마나 개선될 수 있는지를 정량적으로 파악하기 어렵다.

따라서 이 연구는 순수한 RL 전략의 샘플 비효율성을 극복하면서도 잠재적으로 서브옵티멀한 전문가를 능가할 수 있는 정책을 학습하기 위해, IL과 RL의 장점을 결합하는 새로운 방법을 모색한다. 목표는 전문가를 활용하여 합리적인 정책을 빠르게 학습하는 동시에, RL을 통해 전문가를 능가할 수 있는 방법을 탐색하는 것이다.

## ✨ Key Contributions

이 논문의 핵심 기여는 다음과 같다.

* **IL과 RL의 통합 프레임워크 제시**: cost shaping과 cost-to-go oracle의 개념을 통해 모방 학습(IL)과 강화 학습(RL)을 통합하는 새로운 방식을 제안한다. 이는 oracle의 정확도에 따라 학습자의 계획 예측 범위(planning horizon)가 조절되는 방식으로, $k=1$일 때 모방 학습, $k=\infty$일 때 순수 강화 학습에 해당하여 두 접근 방식을 보간(interpolate)하는 통찰을 제공한다.
* **oracle 정확도와 계획 예측 범위 간의 관계 규명**: cost-to-go oracle의 정확도가 학습자의 효과적인 계획 예측 범위를 어떻게 단축하는지 이론적으로 분석한다. 전역 최적(globally optimal) oracle은 계획 예측 범위를 1단계로 단축하여 1단계 탐욕적(one-step greedy) 마르코프 의사 결정 과정(Markov Decision Process, MDP)을 최적화하기 쉽게 만들지만, 최적에서 멀리 떨어진 oracle은 유사 최적(near-optimal) 성능을 달성하기 위해 더 긴 계획 예측 범위에 걸쳐 계획해야 함을 보인다.
* **Truncated HORizon Policy Search (THOR) 제안**: 위 통찰을 바탕으로 서브옵티멀 oracle이 주어졌을 때, 유한한 계획 예측 범위 ($k$)에 걸쳐 reshaped cost를 최대화하는 정책을 탐색하는 방법인 THOR를 제안한다.
* **이론적 성능 개선 보장**: $k > 1$일 때, THOR가 AGGREVATE와 같은 기존 1단계 탐욕적 IL 알고리즘보다 더 나은 정책을 학습할 수 있음을 이론적으로 증명하고, 전문가를 능가하는 성능 개선의 차이를 정량적으로 제시한다.
  * 불완전한 oracle이 주어졌을 때 AGGREVATE의 성능 한계를 보여주는 하한(lower bound)을 분석한다 (Theorem 3.1).
  * $k$-step disadvantage를 최적화하는 정책이 최적 정책에 얼마나 가까워질 수 있는지를 상한(upper bound)으로 제시한다 (Theorem 3.2).
* **실용적인 알고리즘 개발**: 연속적인 상태 및 행동 공간에 적용 가능한 모델-프리(model-free), actor-critic 스타일의 기울기 기반(gradient-based) 알고리즘을 제안한다. 이 알고리즘은 Truncated Back-Propagation Through Time(BPTT)과 유사한 방식으로 구현된다.
* **실험적 우수성 입증**: 불완전한 cost-to-go oracle이 주어졌을 때도, THOR가 RL 기준선(TRPO-GAE) 대비 더 빠른 수렴과 높은 샘플 효율성을 보이며, IL 기준선(AGGREVATED) 대비 현저히 더 나은 성능을 달성함을 실험을 통해 입증한다. 특히 희소 보상(sparse rewards) 환경에서 뛰어난 성능을 보인다.

## 📎 Related Works

논문에서 소개한 관련 연구 및 그 한계, 그리고 본 연구와의 차별점은 다음과 같다.

* **계획 예측 범위(Planning Horizon) 단축**:
  * **Farahmand et al. (2016)**: 최적 가치(optimal value-to-go)를 근사하는 종료 값을 가진 $k$-단계 보상 합을 최대화하는 모델 기반 RL 접근 방식을 제안했다. 이 알고리즘은 모델 기반 설정과 이산 상태/행동 공간에 초점을 맞추며, 정책 계산을 위해 $k$-단계 가치 반복(value iteration)을 수행해야 한다.
  * **본 연구와의 차별점**: THOR는 모델-프리(model-free) 방식이며, 연속적인 상태 및 행동 공간에도 적용 가능하다.
* **편향-분산(Bias-Variance) 트레이드오프**:
  * 시간차 학습(Temporal Difference Learning) (Sutton, 1988) 및 정책 반복(policy iteration) 문헌 (Gabillon et al., 2011)에서 $k$-단계 롤아웃(rollouts)을 사용하여 추정된 reward-to-go의 편향과 분산을 교환하는 접근 방식이 광범위하게 연구되었다.
  * **본 연구와의 차별점**: 본 연구는 이러한 트레이드오프를 계획 예측 범위의 관점에서 재해석하며, cost shaping을 통해 IL과 RL을 통합하는 새로운 관점을 제시한다.
* **보상 형성(Reward Shaping)**:
  * **Ng (2003)**: Ng의 박사 학위 논문 Theorem 5는 보상 형성을 위한 잠재 함수(potential function)가 최적 가치 함수에 가깝다면, 원래 MDP의 할인율(discount factor)을 크게 줄여도 최적성을 잃지 않는다고 제안했다.
  * **본 연구와의 차별점**: Ng의 연구와 본 연구 모두 근본적으로 재구성된 MDP의 난이도(계획 예측 범위가 짧을수록 최적화하기 쉬움)와 학습된 정책의 최적성 사이의 트레이드오프를 다룬다. 그러나 본 연구는 계획 예측 단계를 직접 단축하는 방식을 고려하며, 보상 형성을 통해 기존 모방 학습 접근 방식을 이해하고 oracle이 최적 가치 함수에 얼마나 가까운지에 따라 계획 예측 범위를 1에서 무한대까지 변화시킴으로써 IL과 RL을 통합하는 길을 제시한다.
* **IL과 RL의 결합**:
  * **Chang et al. (2015)**: 점진적인 RL 및 IL 업데이트를 확률적으로 교차하여 IL과 RL을 결합하려고 시도했다. 이 방법은 학습된 정책이 전문가 정책만큼 수행하거나 궁극적으로 지역 최적 정책에 도달할 수 있다.
  * **본 연구와의 차별점**: Chang et al.의 연구는 전문가 대비 학습자가 얼마나 개선될 수 있는지 정량적으로 파악하기 어려웠으나, THOR는 이러한 성능 차이를 정확하게 정량화하는 이론적 보증을 제공한다.
* **기존 모방 학습 알고리즘의 한계**:
  * **AGGREVATE (Ross & Bagnell, 2014) 및 AGGREVATED (Sun et al., 2017)**: 이론적으로 강력한 보증을 제공하지만, 서브옵티멀한 전문가가 있을 경우 정책이 최적에 가깝다는 보장을 하지 못하며, 단지 전문가 정책 대비 한 단계 개선된 정책만을 보장한다.
  * **본 연구와의 차별점**: 본 연구는 불완전한 oracle을 사용한 AGGREVATE의 성능 한계를 보여주는 하한 분석(lower bound analysis)을 추가로 제공한다.

이러한 관련 연구와 비교했을 때, 본 연구의 주요 차별점은 cost shaping을 통해 IL과 RL을 계획 예측 범위의 개념으로 통합하고, 불완전한 oracle이 주어졌을 때 $k>1$ 스텝 최적화를 통해 전문가를 능가하는 성능 개선을 이론적으로 보장하며, 연속적인 상태 및 행동 공간에 적용 가능한 모델-프리 actor-critic 알고리즘을 제시한다는 점이다.

## 🛠️ Methodology

이 논문은 cost shaping과 유한한 계획 예측 범위(truncated planning horizon)를 활용하여 모방 학습(IL)과 강화 학습(RL)을 결합하는 방법인 Truncated HORizon Policy Search (THOR)를 제안한다.

### 전체 파이프라인 및 시스템 구조

THOR의 핵심 아이디어는 다음과 같다.

1. **Cost Shaping**: 원본 MDP $M_0$의 cost 함수 $c(s,a)$를 전문가의 cost-to-go oracle $\hat{V}_e(s)$를 잠재 함수(potential function)로 사용하여 새로운 cost 함수 $c'(s,a)$로 변환한다. 이로써 새로운 MDP $M$이 생성된다.
2. **Truncated Horizon Policy Search**: 새로 생성된 MDP $M$ 위에서 유한한 $k$ 단계의 계획 예측 범위에 걸쳐 reshaped cost를 최소화하는 정책을 탐색한다.

### 주요 구성 요소 및 역할

* **Markov Decision Process (MDP)**: $M_0 = (S,A,P,C,\gamma)$.
  * $S$: 상태(state) 집합.
  * $A$: 행동(action) 집합.
  * $P(s'|s,a)$: 상태 $s$에서 행동 $a$를 취했을 때 상태 $s'$로 전이할 확률.
  * $c(s,a)$: 상태 $s$에서 행동 $a$를 취했을 때 발생하는 cost.
  * $\gamma \in [0,1)$ : 할인율(discount factor).
* **Cost-to-Go Oracle ($\hat{V}_e(s)$)**: 훈련 중 전문가의 cost-to-go를 추정하여 제공하는 함수이다. $\hat{V}_e(s)$는 최적 정책의 cost-to-go $V^*_{M_0}(s)$와 반드시 같을 필요는 없으며, 불완전할 수 있다. 논문에서는 전문가 시연으로부터 TD 학습을 통해 $\hat{V}_e(s)$를 학습하는 시나리오에 초점을 맞춘다.
* **Cost Shaping**: 원본 MDP $M_0$와 임의의 잠재 함수 $\Phi: S \to \mathbb{R}$가 주어졌을 때, 새로운 cost $c'(s,a)$는 다음과 같이 정의된다.
    $$ c'(s,a) = c(s,a) + \gamma\Phi(s') - \Phi(s), \quad s' \sim P_{sa} $$
    여기서 $\Phi(s)$는 $\hat{V}_e(s)$로 사용된다. Ng et al. (1999)는 이러한 cost shaping이 원본 MDP $M_0$와 reshaped MDP $M$의 최적 정책을 동일하게 유지함을 보였다.
* **$k$-step Disadvantage (불리 함수)**: 정책 $\pi$가 $k$ 단계에 걸쳐 reshaped cost $c'(s,a)$를 최소화하도록 훈련되는 목표 함수이다. 이는 다음과 같이 정의된다.
    $$ E \left[ \sum_{i=1}^k \gamma^{i-1} c(s_i,a_i) + \gamma^k \hat{V}_e(s_{k+1}) - \hat{V}_e(s_1) \Bigg| s_1=s; a \sim \pi \right], \quad \forall s \in S $$
    여기서 $k=1$일 경우, 이는 AGGREVATE와 같이 전문가에 대한 1단계 disadvantage를 최소화하는 문제로 귀결된다. $k=\infty$일 경우, 이는 원본 MDP의 전체 cost를 최적화하는 문제가 된다.

### 주요 방정식 설명 및 훈련 절차

#### 1. 이론적 근거

* **Theorem 3.1 (불완전 Oracle 사용 시 1단계 탐욕 정책의 한계)**:
    불완전한 oracle $\hat{V}_e(s)$가 최적 $V^*_{M_0}(s)$와 $|\hat{V}_e(s) - V^*_{M_0}(s)| = \epsilon$ 만큼의 차이를 가질 때, 1단계 탐욕 정책 $\hat{\pi}^* = \arg\min_a [c(s,a) + \gamma E_{s' \sim P_{sa}}[\hat{V}_e(s')]$] (AGGREVATE의 정책)의 성능은 최적 정책 $\pi^*$로부터 적어도 $\Omega(\frac{\gamma \epsilon}{1-\gamma})$만큼 떨어진다. 즉, $J(\hat{\pi}^*) - J(\pi^*) \ge \Omega(\frac{\gamma \epsilon}{1-\gamma})$ 이다. 이는 불완전한 oracle에만 의존하는 1단계 탐욕적 정책이 상당한 서브옵티멀리티를 가질 수 있음을 보여준다.

* **Theorem 3.2 (k>1 단계 최적화를 통한 전문가 능가)**:
    $k>1$ 단계에 걸쳐 $k$-step disadvantage (위의 $k$-step disadvantage 공식)를 최소화하는 정책 $\hat{\pi}^*$를 학습하고, oracle $\hat{V}_e(s)$가 최적 $V^*(s)$와 $| \hat{V}_e(s) - V^*(s) | = \Theta(\epsilon)$ 만큼 차이 날 때, $\hat{\pi}^*$의 성능은 최적 정책 $\pi^*$로부터 $J(\hat{\pi}^*) - J(\pi^*) \le O(\frac{\gamma^k \epsilon}{1-\gamma^k})$ 만큼의 성능 차이를 가진다. 이 정리는 $k$를 증가시킴으로써, 불완전한 oracle이 있을 때 1단계 탐욕 정책의 한계($\Omega(\frac{\gamma \epsilon}{1-\gamma})$)를 극복하고 최적 정책에 더 가까워질 수 있음을 보여준다.

#### 2. Truncated HORizon Policy Search (THOR) 알고리즘 (Algorithm 1)

THOR는 파라미터화된 정책 $\pi_\theta$에 대해 기울기 기반 업데이트를 수행하는 actor-critic 스타일의 알고리즘이다.

* **입력**: 원본 MDP $M_0$, 절단 단계(truncation step) $k$, Oracle $V_e$.
* **정책 초기화**: 정책 $\pi_\theta$와 $k$-단계 truncated advantage estimator $\hat{A}_{0,k}^M$를 초기화한다.
* **반복**:
    1. **데이터 수집**: 현재 정책 $\pi_{\theta_n}$을 실행하여 $N$개의 궤적(trajectory) $\{\tau_i\}_{i=1}^N$을 생성한다.
    2. **Cost Reshaping**: 각 궤적 $\tau_i$의 모든 시간 단계 $t$에 대해 cost를 재구성한다:
        $$ c'(s_t,a_t) = c(s_t,a_t) + V_e(s_{t+1}) - V_e(s_t) $$
        *참고: 알고리즘에서는 $\gamma$가 명시되지 않았으나, 일반적인 cost shaping 공식(Eq. 1)에 따르면 $\gamma V_e(s_{t+1})$이 되어야 한다. 여기서는 알고리즘의 명시된 형태를 따른다.*
    3. **기울기 계산**: 정책 파라미터 $\theta$에 대한 정책 기울기를 계산한다. 이 기울기는 $k$-단계 truncated advantage estimator $\hat{A}_{\pi_n,k}^M(s_t,a_t)$를 사용하여 다음과 같이 근사된다.
        $$ \sum_{\tau_i} \sum_t \nabla_\theta (\ln \pi_\theta(a_t|s_t)) \Big|_{\theta=\theta_n} \hat{A}_{\pi_n,k}^M(s_t,a_t) $$
        여기서 $\hat{A}_{\pi_n,k}^M(s_t,a_t)$는 GAE(Generalized Advantage Estimation)를 사용하여 근사된, reshaped cost $c'$에 대한 $k$-단계 truncated advantage function이다. 이는 $Q_{\pi,k}^M(s,a) = E [c'(s,a) + \sum_{i=1}^{k-1} \gamma^i c'(s_i,a_i)]$와 $V_{\pi,k}^M(s) = E [\sum_{t=1}^k \gamma^{t-1} c'(s_t,a_t)]$로부터 유도된다.
        이 기울기 공식은 Truncated Back-Propagation Through Time(BPTT)과 유사하며, $c'_t$가 최대 $k$ 단계까지만 이전 행동에 영향을 미친다고 본다. $k=1$은 AGGREVATED의 "No Back-Propagation Through Time"과 동일하다.
    4. **Disadvantage Estimator 업데이트 (Critic)**: 수집된 궤적과 reshaped cost $c'$를 사용하여 $\hat{A}_{\pi_n,k}^M$를 업데이트한다.
    5. **정책 파라미터 업데이트 (Actor)**: 계산된 기울기를 사용하여 정책 파라미터 $\theta_{n+1}$를 업데이트한다 (예: Stochastic Gradient Descent, Natural Gradient 등).

THOR는 oracle $\hat{V}_e$가 불완전하더라도, $k$ 값을 적절히 조절함으로써 IL (작은 $k$)과 RL (큰 $k$) 사이의 균형을 찾아 전문가를 능가하고 더 효율적인 학습을 가능하게 한다.

## 📊 Results

이 논문은 OpenAI Gym의 로봇 시뮬레이터를 사용하여 THOR의 성능을 평가했다.

**실험 설정**:

* **비교 대상**: TRPO-GAE (강화 학습 기준선), AGGREVATED (모방 학습 기준선, AGGREVATE의 정책 기울기 버전).
* **Oracle 시뮬레이션**:
    1. TRPO-GAE를 수렴할 때까지 훈련하여 전문가 정책($\pi_e$)을 얻는다.
    2. $\pi_e$를 실행하여 궤적(trajectory) 배치를 수집한다.
    3. 수집된 전문가 시연 데이터로부터 TD 학습(Temporal Difference learning)을 사용하여 cost-to-go 함수 $\hat{V}_e$를 훈련한다.
    4. **중요**: THOR는 훈련된 $\hat{V}_e$만을 cost shaping에 사용하며, $\pi_e$ 자체나 상호작용적 전문가 피드백은 사용하지 않는다. 이는 이전 연구(Ross et al., 2011)보다 훨씬 어려운 설정이다. 또한, 정책 또는 critic을 시연 데이터로 사전 훈련하지 않아 순수 RL 접근 방식과 공정한 비교를 보장한다.
* **목표**: 불완전한 oracle $\hat{V}_e$가 주어졌을 때,
  * THOR ($k>1$)가 AGGREVATED ($k=1$)보다 현저히 더 나은 성능을 보이는지.
  * THOR ($k \ll H$, 여기서 $H$는 전체 계획 예측 범위)가 TRPO-GAE보다 더 빠르게 수렴하고 샘플 효율적인지.
* **통계**: 25개의 독립적으로 생성된(i.i.d.) 시드(seed)로부터 평균과 표준 편차를 보고한다.
* **매개변수 튜닝**: TRPO-GAE 코드베이스의 권장 매개변수를 사용했으며, 절단 길이 $k$만 튜닝했다.

### 5.1 이산 행동 제어 (Discrete Action Control)

* **환경**: Mountain-Car, Acrobot, 희소 보상(sparse reward) 버전 CartPole. 이들 환경은 정책이 성공하기 전까지 보상 신호가 거의 없으므로, 무작위 탐색에 의존하는 순수 RL은 보상 희소성(reward sparsity)으로 인해 어려움을 겪는다.
* **결과 (Fig. 1)**:
  * **THOR ($k>1$) vs. AGGREVATED ($k=1$)**: Acrobot에서 THOR ($k>1$)는 AGGREVATED ($k=1$)보다 훨씬 우수한 성능을 보인다. Mountain Car에서는 AGGREVATED의 평균 성능이 좋지만, THOR ($k>1$, 특히 $k=10$)는 훨씬 높은 `평균 + 표준 편차`를 보여 보상 신호를 더 효과적으로 활용하여 oracle보다 더 나은 성능을 달성할 수 있음을 나타낸다.
  * **THOR ($k \ll H$) vs. TRPO-GAE**: $k$ 값이 증가할수록 일반적으로 성능이 향상된다. Acrobot 환경에서 $H=200$으로 설정하여 무작위 정책이 보상을 받기 더 어렵게 만들었을 때, THOR는 다양한 $k$ 설정에서 TRPO-GAE보다 항상 더 빠르게 학습하며, $k=50, k=100$일 때 TRPO-GAE를 평균과 `평균 + 표준 편차` 모두에서 크게 능가한다. 이는 THOR가 보상 신호(AGGREVATED보다 우수하기 위함)와 oracle (TRPO보다 빠르거나 우수하기 위함)을 모두 활용할 수 있음을 나타낸다.

### 5.2 연속 행동 제어 (Continuous Action Control)

* **환경**: MuJoCo 시뮬레이터의 희소 보상 Inverted Pendulum, 희소 보상 Inverted Double Pendulum, Hopper, Swimmer. Hopper와 Swimmer는 보상 희소성이 없다. 이 환경들은 이전 섹션보다 훨씬 크고 복잡한 상태/제어 공간을 가지므로, $\hat{V}_e$의 정확도가 상대적으로 낮을 수 있다.
* **결과 (Fig. 2)**:
  * **일반적인 경향**: 모든 시뮬레이션에서 좋은 성능을 달성하려면 $k$가 원래 계획 예측 범위 $H$의 약 20%~30% 정도가 필요했다.
  * **AGGREVATED ($k=1$)의 한계**: 불완전한 가치 함수 추정기 $\hat{V}_e$ 때문에 AGGREVATED ($k=1$)는 거의 학습하지 못했다.
  * **Reward Shaping의 효과**: $k=H$ (계획 예측 범위에 대한 절단 없음, 즉 순수 RL에 cost shaping 적용) 설정에서도 $\hat{V}_e$를 사용한 reward shaping이 TRPO-GAE보다 더 나은 성능을 보였다. 이는 $\hat{V}_e$가 최적 $V^*$에 가깝지 않더라도, reward shaping 자체가 정책 기울기 방법에 유익할 수 있음을 시사한다.
  * **분산 감소**: 모든 실험에서 THOR는 학습된 정책의 성능 분산(variance)을 현저히 감소시켰다 (예: Swimmer). 이는 $k$가 $H$에 비해 작을 때 절단(truncation)이 정책 기울기 추정의 분산을 크게 줄일 수 있기 때문이다.

**결론적으로**, 실험 결과는 THOR가 불완전한 oracle이 주어졌음에도 불구하고, 기존 RL 및 IL 기준선 대비 뛰어난 성능, 향상된 샘플 효율성, 그리고 감소된 정책 성능 분산을 보여줌으로써, 이론적 통찰을 효과적으로 검증한다.

## 🧠 Insights & Discussion

### 논문에서 뒷받침되는 강점

* **IL과 RL의 효과적인 통합**: 논문은 cost shaping과 cost-to-go oracle을 사용하여 IL과 RL을 자연스럽게 연결하는 새로운 방법을 제시한다. oracle의 정확도에 따라 계획 예측 범위 $k$를 조절하는 아이디어는 두 패러다임 사이의 스펙트럼을 제공하며, 이는 기존 연구들의 한계를 넘어서는 통찰이다.
* **이론적 기반의 견고성**: AGGREVATE와 같은 기존 1단계 탐욕적 IL 알고리즘이 불완전한 oracle 하에서 겪는 성능 한계를 하한(lower bound)으로 정량화하고($\Omega(\frac{\gamma \epsilon}{1-\gamma})$), $k>1$ 단계 최적화를 통해 이러한 한계를 극복하고 전문가를 능가할 수 있음을 상한(upper bound)으로($O(\frac{\gamma^k \epsilon}{1-\gamma^k})$) 이론적으로 보장한다. 이러한 정량적인 분석은 연구의 신뢰성을 높인다.
* **실용적인 적용 가능성**: 모델-프리, actor-critic 스타일의 기울기 기반 알고리즘인 THOR는 복잡하고 연속적인 상태 및 행동 공간을 가진 로봇 제어 문제에 성공적으로 적용될 수 있음을 실험적으로 보여준다. 이는 많은 현실 세계의 시나리오에서 중요한 장점이다.
* **샘플 효율성 및 안정성 향상**: THOR는 특히 희소 보상 환경에서 기존 RL 기준선(TRPO-GAE)보다 훨씬 빠른 수렴 속도와 높은 샘플 효율성을 달성한다. 또한, 정책 기울기 추정의 분산 감소 효과로 인해 학습된 정책의 성능 분산이 현저히 줄어들어 학습의 안정성이 향상된다.
* **불완전한 Oracle에 대한 강건성**: 전문 지식이나 시연 데이터로부터 학습된 불완전한 cost-to-go oracle만으로도 효과적으로 작동함을 입증한다. 이는 실제 환경에서 완벽한 oracle을 얻기 어렵다는 점을 고려할 때 매우 중요한 실용적 가치를 갖는다.
* **Reward Shaping 자체의 이점**: $k=H$ (전체 계획 예측 범위)인 경우에도 reward shaping이 TRPO-GAE보다 더 나은 성능을 보여주는데, 이는 cost-to-go oracle을 활용한 reward shaping이 그 자체로 RL 알고리즘의 성능을 향상시킬 수 있음을 시사한다.

### 한계, 가정 또는 미해결 질문

* **Oracle의 가용성 및 정확도**: 이 논문은 cost-to-go oracle $\hat{V}_e(s)$에 대한 접근을 가정한다. 비록 논문이 이를 TD 학습을 통해 시뮬레이션하지만, 실제 환경에서 충분히 정확한 $\hat{V}_e(s)$를 얻는 것은 여전히 중요한 과제이다. oracle의 정확도가 $k$의 선택과 최종 성능에 결정적인 영향을 미친다.
* **최적 Truncation Step $k$의 결정**: 실험에서 $k$는 환경과 전체 계획 예측 범위 $H$에 따라 튜닝되어야 했다(예: $H$의 20%~30%). 최적의 $k$ 값을 자동으로 또는 적응적으로 결정하는 방법에 대한 명확한 가이드라인이나 메커니즘은 제시되지 않았다. 이는 실용적인 측면에서 추가적인 튜닝 노력을 요구한다.
* **이론적 보증의 한계**: Theorem 3.2에서 제시된 성능 상한은 여전히 oracle의 불완전성 $\epsilon$에 비례한다($O(\frac{\gamma^k \epsilon}{1-\gamma^k})$). $\epsilon$이 매우 크다면, 상당한 개선을 위해 $k$가 매우 커져야 할 수 있으며, 이는 다시 전체 계획 예측 범위 RL 문제와 유사한 계산 복잡성을 야기할 수 있다.
* **시연 데이터 사전 훈련 미사용**: 논문은 공정한 비교를 위해 시연 데이터로 정책이나 critic을 사전 훈련하지 않았다고 명시한다. 하지만 이론적으로 이러한 사전 훈련이 성능을 더욱 향상시킬 수 있다고 언급되어 있다. 이는 THOR의 잠재력을 완전히 탐색하지 않았을 수 있다는 의문을 제기한다.

### 논문에 근거한 간략한 비판적 해석 및 논의사항

THOR는 IL과 RL의 강점을 효과적으로 결합하여, 두 분야의 오랜 문제점인 샘플 비효율성과 전문가의 서브옵티멀리티 한계를 동시에 해결하려는 매우 중요한 시도이다. 계획 예측 범위 $k$를 통해 두 학습 패러다임을 연속적으로 연결하는 아이디어는 직관적이며 이론적으로도 잘 뒷받침된다. 특히 불완전한 oracle이 주어졌을 때, 1단계 탐욕적 정책이 아닌 $k>1$ 단계 최적화를 통해 성능을 향상시킬 수 있다는 이론적 증명은 실질적인 가이드라인을 제공한다.

그러나 oracle의 품질과 $k$ 값 선택에 대한 의존성은 여전히 실용적인 과제로 남아있다. oracle을 동적으로 개선하거나, 환경 특성에 맞춰 $k$를 자동으로 조절하는 메커니즘에 대한 연구는 THOR의 적용 가능성을 더욱 확장할 수 있을 것이다. 예를 들어, DDPG(Deep Deterministic Policy Gradient)와 같은 다른 RL 기술과의 결합, 그리고 훈련 중 전문가 피드백을 통해 oracle을 온라인으로 업데이트하는 접근 방식은 향후 연구 방향으로 논문에서 직접 제안하고 있다. 궁극적으로 THOR는 효율성과 성능이라는 두 마리 토끼를 잡으려는 딥 강화 학습 분야에 중요한 진전을 가져다줄 잠재력을 가지고 있다.

## 📌 TL;DR

이 논문은 "Truncated HORizon Policy Search (THOR)"를 제안하며 모방 학습(IL)과 강화 학습(RL)을 cost shaping과 cost-to-go oracle의 개념을 통해 통합한다. 핵심 아이디어는 oracle의 정확도가 학습자의 효과적인 계획 예측 범위(planning horizon)를 결정한다는 것이다: 완벽한 oracle은 계획 예측 범위를 1단계로 단축시키지만, 불완전한 oracle은 유사 최적 성능을 위해 더 긴 $k$ 단계를 요구한다. 이는 IL($k=1$)과 RL($k=\infty$) 사이의 간극을 연결하는 통찰을 제공한다.

THOR는 불완전한 oracle이 주어졌을 때 유한한 $k$-단계 계획 예측 범위에 걸쳐 reshaped cost를 최적화하는 정책을 탐색한다. 이론적으로, 이 연구는 1단계 탐욕적 IL 방법(예: AGGREVATE)의 성능 한계를 정량화하고, $k>1$ 단계 최적화를 통해 불완전한 oracle로도 전문가를 능가하는 정책을 학습할 수 있음을 보장한다. 실용적인 측면에서, 이 논문은 연속적인 상태 및 행동 공간에 적용 가능한 모델-프리, actor-critic 스타일의 기울기 기반 알고리즘을 제시한다.

실험을 통해 THOR는 불완전한 cost-to-go oracle을 사용하더라도 기존 RL 기준선(TRPO-GAE) 대비 더 빠른 수렴과 높은 샘플 효율성을, 그리고 IL 기준선(AGGREVATED) 대비 현저히 더 나은 성능을 달성함을 입증했다. 특히, 희소 보상 환경에서 뛰어난 성능을 보였고, 학습된 정책의 성능 분산 또한 감소시켰다.

이 연구는 IL과 RL의 주요 한계(샘플 비효율성 및 전문가 서브옵티멀리티)를 극복하며, 두 분야를 통합하는 강력한 프레임워크를 제공한다. THOR는 실제 적용 및 향후 연구에서 순차적 의사결정 문제의 효율적이고 강력한 해결책으로 중요한 역할을 할 가능성이 있다.
