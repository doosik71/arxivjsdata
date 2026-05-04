# Fast Policy Learning through Imitation and Reinforcement

Ching-An Cheng, Xinyan Yan, Nolan Wagener, Byron Boots (2018)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)의 느린 수렴 속도와 모방 학습(Imitation Learning, IL)의 전문가 의존성 문제를 동시에 해결하고자 한다. 

강화학습은 복잡한 의사결정 문제에서 강력한 성능을 보이지만, 최적의 정책을 학습하기 위해 방대한 양의 환경 상호작용이 필요하며 이는 실제 로봇 공학 시스템에 적용할 때 비용과 시간 측면에서 매우 비효율적이다. 반면, 모방 학습은 전문가의 시연(demonstrations)을 활용하여 정책을 빠르게 학습할 수 있다는 장점이 있다. 그러나 모방 학습은 전문가의 정책 품질에 완전히 의존하므로, 제공된 전문가가 최적이 아닌 suboptimal 상태일 경우 학습된 정책 역시 전문가의 성능을 뛰어넘지 못하고 오히려 강화학습보다 낮은 성능을 보일 수 있다.

따라서 본 연구의 목표는 모방 학습의 빠른 초기 수렴 속도와 강화학습의 최적성 확보 능력을 결합하여, suboptimal 전문가를 가지고 시작하더라도 최종적으로는 전문가보다 더 나은 성능을 내면서도 일반적인 RL보다 빠르게 수렴하는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RL과 IL을 단일한 이론적 프레임워크로 통합하고, 이를 기반으로 **LOKI (Locally Optimal search after K-step Imitation)** 알고리즘을 제안한 것이다.

1. **통합 프레임워크 제시**: 다양한 RL 및 IL 알고리즘을 **Mirror Descent**라는 공통의 최적화 프레임워크 내에서 정식화하였다. 이를 통해 RL과 IL의 차이가 본질적으로는 업데이트 방향을 결정하는 first-order oracle의 선택 차이임을 수학적으로 증명하였다.
2. **LOKI 알고리즘 제안**: 정책 학습을 두 단계(Imitation Phase $\rightarrow$ Reinforcement Phase)로 나누어 진행하는 단순하고 효율적인 전략을 제안하였다.
3. **랜덤 스위칭 시간($K$)의 도입**: 모방 학습에서 강화학습으로 전환되는 시점인 $K$를 특정 확률 분포에 따라 무작위로 결정함으로써, 이론적으로 LOKI가 전문가 정책을 초기 조건으로 하여 직접 Policy Gradient를 수행하는 것과 유사한 성능을 낼 수 있음을 보였다.
4. **이론적 및 실험적 검증**: 제안된 방법이 suboptimal 전문가를 능가할 수 있음을 이론적으로 증명하였으며, 다양한 시뮬레이션 환경에서 기존의 하이브리드 방법론들보다 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

기존 연구들은 RL의 비용 정보(cost information)를 IL 과정에 통합하여 전문가 이상의 성능을 내고자 시도하였다.

- **DAGGER 및 AGGREVATED**: DAGGER는 전문가의 행동을 모방하여 covariate shift 문제를 해결하려 했으나 전문가의 성능에 갇히는 한계가 있다. AGGREVATED는 전문가의 Advantage function을 직접 사용하여 전문가를 능가할 가능성을 열었으나, 실제 구현 시 전문가의 가치 함수(Value function)를 정확히 추정하는 것이 매우 어렵다.
- **THOR 및 LOLS**: 최근 연구인 THOR는 truncated horizon RL 문제를 풀이하여 전문가 의존성을 줄이려 했고, LOLS는 전문가 정책과 현재 정책을 혼합하여 학습하는 방식을 제안하였다. 
- **기존 방식의 한계**: 위 방법들은 대개 상태 리셋(state resetting)이 가능하다는 가정이 필요하거나, 전문가의 가치 함수 $\hat{V}^{\pi^*}$에 크게 의존한다. 특히 고차원 상태 공간에서는 covariate shift로 인해 가치 함수 추정 오차가 기하급수적으로 증가하여 성능이 급격히 저하되는 문제가 발생한다.

LOKI는 전문가의 가치 함수를 학습할 필요 없이 단순한 KL-divergence 기반의 모방 학습으로 시작하여 RL로 전환함으로써 이러한 실무적 제약과 수렴 불안정성을 회피한다.

## 🛠️ Methodology

### 1. Mirror Descent Framework
본 논문은 정책 업데이트 규칙을 다음과 같은 Mirror Descent 형태로 정의한다.
$$\theta_{n+1} = \arg \min_{\theta \in \Theta} \langle g_n, \theta \rangle + \frac{1}{\eta_n} D_{R^n}(\theta || \theta_n)$$
여기서 $g_n$은 first-order oracle로부터 받은 업데이트 방향 벡터이며, $D_{R^n}$은 Bregman divergence이다. $R^n$의 선택에 따라 Negative entropy, Quadratic form, Fisher information matrix 등을 사용하여 다양한 RL/IL 알고리즘을 표현할 수 있다.

### 2. First-Order Oracles
RL과 IL의 차이는 $g_n$을 생성하는 oracle의 차이로 귀결된다.
- **Policy Gradient (RL)**: 현재 정책 $\pi_n$을 기준으로 Advantage function $A^{\pi_n}$을 사용하여 업데이트 방향을 결정한다.
  $$g_n \approx \mathbb{E}_{d^{\pi_n}} (\nabla_\theta \mathbb{E}_{\pi} [A^{\pi_n}])$$
- **Imitation Gradient (IL)**: 전문가 정책 $\pi^*$와 학습 정책 $\pi$ 사이의 대리 손실 함수(surrogate loss) $\tilde{c}$를 최소화하는 방향으로 업데이트한다.
  $$g_n \approx \mathbb{E}_{d^{\pi_n}} (\nabla_\theta \mathbb{E}_{\pi} [\tilde{c}])$$

### 3. LOKI 알고리즘 절차
LOKI는 다음의 두 단계로 구성된다.

**Phase 1: Imitation Phase (모방 단계)**
- 학습 시작 전, 정수 $K$를 확률 분포 $P(K=n) = \frac{n^d}{\sum_{m=N_m}^{N_M} m^d}$에 따라 무작위로 샘플링한다.
- $K$번의 반복 동안 모방 학습을 수행한다. 이때 손실 함수로 $\mathbb{E}_{\pi} [\tilde{c}] = KL(\pi^* || \pi)$를 사용하여 전문가 정책에 빠르게 근접하도록 한다. 이 단계에서는 가치 함수 추정기가 필요 없으며, 단순한 정책 거리 최소화만 수행한다.

**Phase 2: Reinforcement Phase (강화 단계)**
- $K$단계 이후에는 Policy Gradient 방법으로 전환한다.
- 이미 모방 단계에서 수집된 데이터를 통해 Advantage function $\hat{A}^{\pi_n}$의 기초 추정치가 마련되어 있으므로, 전문가 정책 근처에서 효율적으로 최적화를 시작하여 전문가의 성능을 추월하는 local optimum을 찾는다.

## 📊 Results

### 실험 설정
- **작업(Tasks)**: Inverted Pendulum, Locomotion (Hopper, 2D Walker, 3D Walker), Robot Manipulator (Reacher) 등 OpenAI Gym 및 DART 물리 엔진 기반 환경.
- **비교 대상(Baselines)**: 
    - TRPO: 순수 RL.
    - TRPO from expert (Ideal): 전문가 정책으로 초기화 후 RL 수행.
    - DAGGERED: 1차 미분 기반의 DAGGER.
    - SLOLS, THOR: RL과 IL을 결합한 기존 하이브리드 방법론.
- **평가 지표**: 누적 보상(Accumulated Rewards).

### 주요 결과
- **LOKI vs. Baselines**: LOKI는 모든 작업에서 전문가 정책을 능가하는 성능을 보였다. 특히 학습 초기에는 DAGGERED처럼 빠르게 상승하고, 이후에는 TRPO처럼 지속적으로 성능을 개선하여 최종적으로는 'Ideal' 설정(전문가 초기화 RL)과 거의 동일한 성능에 도달하였다.
- **복잡도에 따른 성능 차이**: 
    - 저차원 작업(Pendulum, Reacher)에서는 THOR와 SLOLS도 전문가 이상의 성능을 냈으나, 고차원 작업(Walker 시리즈)에서는 성능이 TRPO보다도 낮게 나타났다.
    - 이는 THOR와 SLOLS가 전문가의 가치 함수 $\hat{V}^{\pi^*}$ 추정에 의존하는데, 고차원 상태 공간에서 발생하는 covariate shift로 인해 가치 함수 추정 오차가 심각해지기 때문이다.
- **LOKI의 강점**: LOKI는 전문가의 가치 함수를 직접 사용하지 않고 정책 공간에서의 거리(KL)만 이용하므로, covariate shift 문제로부터 자유로우며 구현이 매우 단순하다.

## 🧠 Insights & Discussion

본 논문은 RL의 탐색 효율성 문제와 IL의 전문가 의존성 문제를 '랜덤 스위칭'이라는 단순한 아이디어로 해결하였다.

- **이론적 통찰**: $K$를 무작위로 설정함으로써 얻는 이점은 매우 크다. Theorem 2에 따르면, LOKI는 기대값 관점에서 전문가 정책을 초기 상태로 하여 Policy Gradient를 수행하는 것과 유사한 효과를 낸다. 이는 단순한 휴리스틱이 아니라 최적화 이론에 근거한 접근임을 시사한다.
- **실무적 강점**: 기존의 RL-IL 결합 알고리즘들은 상태 리셋 가능성이나 정교한 가치 함수 추정기를 요구했지만, LOKI는 기존의 TRPO 같은 off-the-shelf 알고리즘들을 그대로 활용하면서 단계만 나누어 적용하면 되므로 구현 난이도가 매우 낮다.
- **비판적 해석**: 본 논문에서 제시한 $\tilde{c} = KL(\pi^* || \pi)$ 방식은 전문가 정책의 행동 분포만을 모방하는 것이므로, 보상 구조가 매우 복잡하거나 전문가의 행동이 극도로 제한적인 경우 초기 수렴 속도가 예상보다 느릴 수 있다. 또한, 랜덤 변수 $K$의 하이퍼파라미터($d, N_m, N_M$) 설정이 성능에 어느 정도 영향을 미치는지에 대한 민감도 분석이 더 보강되었다면 더 완성도 높은 보고서가 되었을 것이다.

## 📌 TL;DR

본 연구는 모방 학습(IL)의 빠른 초기 수렴 속도와 강화학습(RL)의 최적 성능 달성 능력을 결합한 **LOKI** 알고리즘을 제안한다. LOKI는 무작위로 결정된 시점 $K$까지는 전문가를 모방 학습하고, 그 이후에는 정책 경사법(Policy Gradient)을 통해 성능을 고도화한다. 실험 결과, LOKI는 suboptimal 전문가를 가지고 시작하더라도 전문가를 능가하는 성능을 보였으며, 특히 기존의 하이브리드 방법론들이 겪던 가치 함수 추정 오차(covariate shift) 문제를 효과적으로 해결하였다. 이는 실무적으로 구현이 매우 간단하면서도 강력한 성능을 내는 RL-IL 통합 프레임워크를 제시했다는 점에서 가치가 크다.