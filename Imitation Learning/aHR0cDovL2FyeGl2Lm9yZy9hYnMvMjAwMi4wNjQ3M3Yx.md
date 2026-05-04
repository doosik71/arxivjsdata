# Universal Value Density Estimation for Imitation Learning and Goal-Conditioned Reinforcement Learning

Yannick Schroecker, Charles Isbell (2020)

## 🧩 Problem to Solve

본 논문은 Goal-conditioned Reinforcement Learning (GCRL)과 Imitation Learning (IL)이라는 두 가지 서로 다른 설정에서 발생하는 공통적인 문제, 즉 에이전트가 특정 목표 상태(goal state)나 데모 상태(demonstration state)에 효율적이고 신뢰성 있게 도달하도록 학습시키는 문제를 다룬다.

구체적으로 GCRL에서는 희소 보상(sparse reward)으로 인해 학습 신호가 부족하며, 이를 해결하기 위해 널리 사용되는 Hindsight Experience Replay (HER)가 'hindsight bias'라는 심각한 부작용을 겪는다는 점을 지적한다. Hindsight bias는 실패한 시도에서 목표를 사후적으로 변경하여 학습함으로써, 위험한 행동의 실패 확률을 과소평가하고 결과적으로 최적이 아닌 경로를 선택하게 만드는 편향을 의미한다.

또한, IL에서는 에이전트가 데모 데이터와 다른 상태에 진입했을 때 결정 내리지 못해 오류가 누적되는 문제(accumulating errors)와 매우 적은 양의 데모 데이터만으로 효율적으로 학습해야 하는 Sample-efficiency 문제가 핵심이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 확률적 장기 역학(probabilistic long-term dynamics)과 가치 함수(value function) 사이의 연결 고리를 찾는 것이다. 저자들은 특정 상태에 도달할 확률 밀도(density)가 가치 함수와 직접적인 관련이 있다는 점에 착안하여, 최신 Density Estimation 기술을 활용한 **Universal Value Density Estimation (UVDE)** 방식을 제안한다.

중심적인 직관은 보상이 목표 도달 시에만 주어지는 희소 보상 설정에서, $Q$-함수는 에이전트가 해당 목표에 도달할 확률의 할인된 합(discounted likelihood)과 비례한다는 것이다. 이를 통해 가치 함수를 밀도 추정 문제로 치환함으로써, HER와 같은 편향 없이도 희소 보상 환경에서 밀집된 학습 신호(dense learning signal)를 얻을 수 있다.

## 📎 Related Works

기존의 GCRL 접근 방식인 UVFA(Universal Value Function Approximators)는 무작위 목표 샘플링을 사용하므로 보상 신호가 밀집되어 있어야 한다는 한계가 있다. 이를 극복하기 위해 제안된 HER는 사후적으로 목표를 수정하여 학습 효율을 높였으나, 앞서 언급한 hindsight bias로 인해 확률적 도메인(stochastic domains)에서 성능이 저하된다.

IL 분야에서는 GAIL(Generative Adversarial Imitation Learning)과 같은 적대적 학습 방식이 상태-행동 분포 매칭을 통해 높은 성능을 보였으나, 데모 데이터가 극도로 적을 경우 판별기(discriminator)를 학습시키기 어려워 성능이 급격히 떨어진다는 한계가 있다. 본 논문은 이러한 분포 매칭 문제를 가치 밀도 추정 관점에서 접근하여 데이터 효율성을 극대화한다.

## 🛠️ Methodology

### 1. Value Density Estimation의 정의
보상 함수가 목표 도달 시에만 양의 값을 가지고 그 외에는 0인 경우, 즉 $r^g(s,a) = (1-\gamma)\delta_{h(s,a),g}$인 설정에서 $Q$-함수는 다음과 같이 정의될 수 있다.

$$Q^\mu_{r^g}(s,a) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \int p^\mu(s, a \to s') \delta_{h(s', \mu(s')), g} ds' =: F^\mu_\gamma(g|s,a)$$

여기서 $F^\mu_\gamma(g|s,a)$는 상태-행동 쌍 $(s,a)$에서 시작하여 정책 $\mu$를 따랐을 때 목표 $g$에 도달할 확률 밀도를 나타내는 **Value Density**이다.

### 2. 시스템 구조 및 학습 절차
본 논문은 이 Value Density를 추정하기 위해 **Normalizing Flows** (구체적으로는 RealNVP의 간소화된 버전)를 사용한다. RealNVP는 가역적인 변환을 통해 단순한 분포를 복잡한 밀도 함수로 변환하며, 최대 가능도 추정(Maximum Likelihood Estimation, MLE)을 통해 학습 가능하다.

학습은 크게 두 가지 추정기를 결합하는 방식으로 이루어진다.
1. **Density Estimator ($F_\Phi$):** 짧은 리플레이 버퍼에서 샘플링하여 목표 도달 확률 밀도를 직접 학습한다.
2. **TD-Learner ($Q_\omega$):** 전통적인 Temporal Difference(TD) 학습을 통해 가치 함수를 업데이트한다.

두 추정기는 서로의 단점(TD의 희소 보상 문제, Density Estimation의 높은 분산 및 짧은 타임 호라이즌 문제)을 보완하기 위해 다음과 같이 결합되어 TD 타겟을 형성한다.

$$Q := \max(\tilde{Q}_\omega(s,a;g), F_\Phi(g|s,a))$$

이후 TD 손실 함수는 다음과 같다.
$$L(\omega) := (r^g(s,a) + \gamma Q - \tilde{Q}_\omega(s,a;g))^2$$

### 3. Value Density Imitation (VDI)
IL로 확장하기 위해, 에이전트가 전문가의 상태 분포를 그대로 따르도록 학습시킨다.
- 에이전트의 현재 상태 분포 $d_\omega(s)$를 모델링하는 무조건부 밀도 추정기를 학습시킨다.
- 전문가의 상태 $s$를 목표로 샘플링할 때, 에이전트가 방문하기 어려운 상태(즉, $d_\omega(s)$가 낮은 상태)에 더 높은 확률($\propto 1/d_\omega(s)$)을 부여하여 샘플링한다.
- 이렇게 샘플링된 상태를 목표로 하여 앞서 설명한 UVD 방식으로 정책을 업데이트함으로써 전문가의 상태 분포를 효율적으로 매칭한다.

## 📊 Results

### 1. Goal-conditioned RL 실험
FetchPush, FetchSlide 등 Fetch 로봇 팔 조작 작업에서 실험을 진행하였다.
- **결과:** 결정론적 환경에서는 UVD가 HER와 유사한 성능을 보였다. 그러나 행동에 가우시안 노이즈를 추가한 확률적 환경(FetchSlide + Noise)에서는 HER가 hindsight bias로 인해 성능이 저하되는 반면, **UVD는 편향 없이 정확한 위험도를 평가하여 훨씬 높은 성공률**을 기록하였다.

### 2. Imitation Learning 실험
HalfCheetah-v2와 Humanoid-v2 로코모션 작업에서 GAIL과 비교하였다.
- **결과:** 특히 데모 데이터가 매우 적은 상황에서 VDI의 우위가 두드러졌다. GAIL은 데모 궤적 수가 줄어들면 성능이 급격히 하락하지만, **VDI는 단 하나의 데모 궤적(single demonstrated trajectory)만으로도 전문가 수준의 성능에 근접**하는 탁월한 Sample-efficiency를 보였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 가치 함수를 확률 밀도 추정 문제로 재정의함으로써, RL의 고질적인 문제인 희소 보상과 HER의 hindsight bias를 동시에 해결했다는 점이다. 특히 Normalizing Flows를 도입하여 복잡한 고차원 상태 공간에서도 유연하게 가치 밀도를 추정할 수 있음을 입증하였다.

다만, 고차원 상태 공간에서 밀도 추정을 수행할 때 발생하는 '차원의 저주(curse of dimensionality)' 문제가 수치적 불안정성을 유발할 수 있다. 저자들은 이를 해결하기 위해 Logit의 합 대신 평균을 사용하는 방식($N$-th root approximation)을 제안하였으나, 이는 수학적 근사치이므로 엄밀한 최적성 보장 측면에서는 논의의 여지가 있을 수 있다.

또한, 밀도 추정기의 타임 호라이즌을 제한하여 분산을 줄였는데, 이로 인해 발생하는 가치 과소평가 문제를 TD 학습의 부트스트래핑(bootstrapping)으로 보완한 설계는 매우 실용적인 접근이라 판단된다.

## 📌 TL;DR

본 논문은 가치 함수를 상태 도달 확률 밀도로 해석하여, Normalizing Flows 기반의 **Universal Value Density Estimation (UVD)** 기법을 제안한다. 이 방법은 GCRL에서 HER의 hindsight bias를 제거하여 확률적 환경에서도 안정적인 학습을 가능하게 하며, IL에서는 단 하나의 데모 궤적으로도 학습이 가능한 극강의 데이터 효율성을 달성한다. 이는 향후 데이터 획득 비용이 높은 실제 로봇 제어나 복잡한 목표 지향적 에이전트 학습에 중요한 기여를 할 것으로 보인다.