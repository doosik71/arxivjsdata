# Policy Gradient Bayesian Robust Optimization for Imitation Learning

Zaynah Javed, Daniel S. Brown, Satvik Sharma, Jerry Zhu, Ashwin Balakrishna, Marek Petrik, Anca D. Dragan, Ken Goldberg (2021)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning) 과정에서 발생하는 **보상 함수 모호성(Reward Function Ambiguity)** 문제를 해결하고자 한다. 실제 환경에서 인간의 시연(demonstration)이나 피드백을 통해 보상 함수를 학습할 때, 동일한 시연 데이터를 설명할 수 있는 서로 다른 보상 함수가 다수 존재할 수 있다. 이는 에이전트가 진정한 보상 함수가 무엇인지에 대해 인식론적 불확실성(Epistemic Uncertainty)을 갖게 함을 의미한다.

기존의 정책 최적화 방식은 주로 기대 성능(expected performance)을 최대화하는 방향으로 설계되었으나, 로봇 공학, 의료, 금융과 같은 실제 응용 분야에서는 최악의 상황을 피하려는 **위험 회피적(risk-averse) 행동**이 필수적이다. 따라서 본 연구의 목표는 보상 함수에 대한 불확실성을 명시적으로 고려하여, 기대 성능과 위험(risk) 사이의 균형을 맞추는 강건한 정책 최적화 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **PG-BROIL (Policy Gradient Bayesian Robust Optimization for Imitation Learning)** 알고리즘을 제안한 것이다. 주요 아이디어는 다음과 같다.

- **Soft-Robust 목적 함수 설계**: 기대 성능과 조건부 가치 위험(Conditional Value at Risk, CVaR)을 동시에 최적화하는 목적 함수를 도입하여, 위험 중립적(risk-neutral) 행동부터 위험 회피적(risk-averse) 행동까지 조절 가능한 정책 학습 프레임워크를 제공한다.
- **연속적 MDP로의 확장**: 기존의 BROIL이 이산 상태/행동 공간과 알려진 전이 역학(known dynamics) 환경으로 제한되었던 한계를 극복하고, 정책 경사(Policy Gradient) 방식을 도입하여 전이 역학을 모르거나 상태/행동 공간이 연속적인 복잡한 MDP에서도 동작하도록 확장하였다.
- **불확실성에 대한 헤징(Hedging)**: 단일한 최적 보상 함수를 찾으려 하기보다, 가능한 여러 보상 함수 가설들에 대해 강건하게 동작하는 정책을 학습함으로써 보상 해킹(reward hacking)이나 오정렬(misalignment) 문제를 완화한다.

## 📎 Related Works

- **강건한 강화학습(Robust RL)**: 기존 연구들은 주로 환경의 전이 역학(transition dynamics)에 대한 불확실성에 집중했으며, 보상 함수는 고정되어 있다고 가정했다.
- **모방 학습(Imitation Learning)**: Behavioral Cloning이나 GAIL 같은 방식은 보상 함수를 명시적으로 학습하지 않고 시연을 직접 모방한다. 역강화학습(IRL) 방식은 보상 함수를 추정하지만, 대개 점 추정치(point-estimate)나 기대 보상 함수만을 사용하여 최적화하므로 불확실성을 충분히 반영하지 못한다.
- **Bayesian IRL**: 보상 함수의 사후 분포(posterior distribution)를 구하지만, 실제 정책 최적화 단계에서는 MAP(Maximum A Posteriori) 또는 평균 보상 함수를 사용하는 경우가 많아 위험 회피적 설계를 반영하기 어렵다.
- **BROIL**: 선형 계획법(Linear Programming)을 통해 위험-기대 성능의 균형을 맞추는 방식을 제안했으나, 연속 공간 및 미지의 역학 환경에서는 적용이 불가능했다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 목적 함수
PG-BROIL은 보상 함수의 분포 $P(R)$이 주어졌을 때, 다음과 같은 **Soft-Robust 목적 함수**를 최대화하는 정책 $\pi_\theta$를 학습한다.

$$\max_{\pi_\theta} \lambda \cdot \mathbb{E}_{P(R)}[\psi(\pi_\theta, R)] + (1-\lambda) \cdot \text{CVaR}_\alpha[\psi(\pi_\theta, R)]$$

여기서 $\psi(\pi_\theta, R)$는 성능 지표(예: 기대 리턴 $v(\pi, R)$)이며, $\lambda \in [0, 1]$는 기대 성능과 위험 회피 사이의 가중치를 조절하는 하이퍼파라미터이다. $\alpha$는 CVaR에서 고려할 하위 꼬리 부분(tail)의 크기를 결정한다.

### 2. 주요 구성 요소 및 수식 설명
- **$\text{CVaR}_\alpha$ (Conditional Value at Risk)**: 분포의 하위 $(1-\alpha)$ 영역의 평균값을 의미한다. 즉, 발생 가능한 최악의 시나리오들의 평균 성능을 최적화함으로써 꼬리 위험(tail risk)을 최소화한다. 본 논문은 이를 위해 다음과 같은 최적화 형태를 사용한다.
  $$\text{CVaR}_\alpha[X] = \max_\sigma \left( \sigma - \frac{1}{1-\alpha} \mathbb{E}[(\sigma - X)^+] \right)$$
  여기서 $\sigma$는 Value at Risk ($\text{VaR}_\alpha$)에 해당하며, $(\cdot)^+$는 $\max(0, \cdot)$를 의미한다.

- **Policy Gradient 업데이트**: CVaR 항은 모든 곳에서 미분 가능하지 않으므로 서브그라디언트(sub-gradient)를 유도하여 사용한다. 최종적인 정책 경사는 다음과 같이 계산된다.
  $$\nabla_\theta \text{BROIL} \approx \frac{1}{|T|} \sum_{\tau \in T} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) w_t(\tau) \right]$$
  여기서 가중치 $w_t(\tau)$는 다음과 같이 정의된다.
  $$w_t(\tau) = \sum_{i=1}^N P(r_i) \Phi_{r_i}^t(\tau) \left( \lambda + \frac{1-\lambda}{1-\alpha} \mathbb{1}_{\sigma^* \ge v(\pi, r_i)} \right)$$
  - $\Phi_{r_i}^t(\tau)$는 보상 함수 $r_i$ 하에서의 Advantage function과 같은 성능 측정치이다.
  - $\mathbb{1}_{\sigma^* \ge v(\pi, r_i)}$는 현재 정책이 보상 함수 $r_i$에서 저조한 성능을 보일 때(즉, 최악의 시나리오에 해당할 때) 가중치를 부여하는 지시 함수이다.

### 3. 학습 절차
1. 현재 정책 $\pi_{\theta^k}$를 통해 궤적 세트 $T$를 수집한다.
2. 샘플링된 $N$개의 보상 함수 가설 $\{r_i\}$ 각각에 대해 기대 리턴 $v(\pi, r_i)$를 추정한다.
3. 선형 탐색(line search)을 통해 $\sigma^*$를 계산한다.
4. 계산된 $w_t$를 사용하여 정책 경사를 추정하고, $\theta$를 업데이트한다. (PPO와 같은 Trust-region 방식 적용 가능)

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 환경**: CartPole, Pointmass Navigation, Reacher, TrashBot, Atari Boxing.
- **비교 대상 (Baselines)**: BC, GAIL, RAIL, PBRL, Bayesian REX.
- **지표**: 기대 리턴, 특정 위험 구역(gray/red region) 진입 횟수, 게임 스코어 등.

### 2. 주요 결과
- **위험-성능 트레이드오프**: $\lambda$ 값을 낮출수록(위험 회피 성향을 높일수록) 에이전트는 기대 리턴은 다소 낮아지더라도 최악의 경우를 피하는 안전한 행동을 보였다. 예를 들어, Pointmass Navigation에서 $\lambda$가 낮을수록 불확실한 비용이 발생하는 회색 구역을 완전히 우회하는 경향이 나타났다.
- **모호한 시연에서의 성능 (TrashBot)**: 매우 적은 수의 선호도(preferences) 데이터만 주어진 상황에서, PG-BROIL은 다른 모든 baseline보다 훨씬 많은 쓰레기를 수집하면서도 회색 구역 진입을 최소화했다.
  - **PBRL/Bayesian REX**: 단일 보상 함수나 평균 보상 함수에 과적합되어 보상 해킹(예: 단순히 흰색 구역에 머무는 행동)에 빠지거나 위험 구역에 진입하는 문제가 발생했다.
  - **PG-BROIL**: 여러 보상 가설을 동시에 고려함으로써 "쓰레기 수집"과 "위험 구역 회피"라는 두 가지 목표를 모두 달성하는 강건한 정책을 학습했다.
- **도메인 시프트 및 복잡한 환경**: Reacher 환경에서 시연 데이터에 없던 위험 구역이 나타났을 때, PG-BROIL은 불확실성을 인식하고 해당 구역을 회피했다. Atari Boxing에서도 단순히 공격만 하는 것이 아니라, 맞을 위험을 고려하여 더 높은 최종 스코어를 기록했다.

## 🧠 Insights & Discussion

### 1. 강점
본 논문은 보상 함수에 대한 **인식론적 불확실성(Epistemic Uncertainty)**을 정책 최적화 단계에서 직접적으로 다루었다는 점에서 큰 의의가 있다. 특히, 단순한 최악 상황 가정(maxmin)이 유발하는 지나친 비관주의(overly pessimistic)를 $\lambda$ 파라미터를 통해 조절할 수 있게 하여, 실용적인 수준의 강건함을 확보했다.

### 2. 한계 및 비판적 해석
- **수치적 불안정성**: $\lambda$ 값이 0에 매우 가까울 때, CVaR의 지시 함수($\mathbb{1}$)로 인해 정책 최적화가 불안정해지는 현상이 관찰되었다. 이는 불연속적인 가중치 업데이트가 그래디언트의 분산을 높이기 때문으로 분석된다.
- **계산 비용**: 매 업데이트마다 $N$개의 보상 함수 가설에 대해 리턴을 계산해야 하므로, 계산량이 표준 정책 경사법보다 $N$배 증가한다. 다만, 저자들은 데이터 수집 단계가 병목이 되는 RL 특성상 이는 수용 가능한 수준이라고 주장한다.
- **보상 함수 샘플링 의존성**: Bayesian REX 등을 통해 보상 함수의 사후 분포를 샘플링하여 사용하므로, 초기 샘플링 품질이 전체 정책의 강건성에 영향을 미칠 가능성이 크다.

## 📌 TL;DR

본 논문은 보상 함수가 불확실한 모방 학습 환경에서 **기대 성능과 꼬리 위험(CVaR)을 동시에 최적화하는 정책 경사 알고리즘인 PG-BROIL**을 제안한다. 이 방법은 연속적인 상태/행동 공간과 미지의 역학을 가진 MDP로 확장 가능하며, 특히 시연 데이터가 부족하거나 모호하여 보상 함수 추정치가 불확실할 때, 단일 보상 함수에 과적합되지 않고 여러 가설에 대해 헤징(hedging)함으로써 보상 해킹을 방지하고 안전한 성능을 보장한다. 향후 픽셀 데이터로부터 직접 강건한 정책을 학습하는 연구로 확장될 가능성이 높다.