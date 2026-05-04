# Enabling Off-Policy Imitation Learning with Deep Actor-Critic Stabilization

Sayambhu Sen, Shalabh Bhatnagar (2025)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)의 고질적인 문제인 보상 함수 설계(Reward Engineering)의 어려움을 해결하기 위한 모방 학습(Imitation Learning, IL)의 효율성 문제를 다룬다. 특히, Generative Adversarial Imitation Learning (GAIL)과 같은 최신 모방 학습 방법론들이 매우 낮은 샘플 효율성(Sample Inefficiency)을 보인다는 점에 주목한다.

이러한 샘플 효율성 저하의 근본적인 원인은 GAIL이 TRPO와 같은 On-policy 알고리즘을 기반으로 하기 때문이다. On-policy 방식은 매번 새로운 궤적(Trajectory)을 수집하고 한 번의 경사 하강법 업데이트 후 데이터를 폐기하므로, 환경과의 상호작용 횟수가 기하급수적으로 증가한다. 또한, Actor, Critic, Discriminator 세 네트워크 간의 상호작용으로 인한 불안정성과, 연속적 행동 공간에서 가우시안 분포를 사용할 때 발생하는 행동 값의 클리핑(Clipping) 문제가 학습 속도를 더욱 늦춘다. 따라서 본 연구의 목표는 Off-policy 학습 체계와 안정화 기법을 도입하여 환경 샘플 효율성을 획기적으로 개선한 모방 학습 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 On-policy 기반의 GAIL 구조를 Off-policy Actor-Critic 구조로 전환하고, 네트워크의 복잡도를 줄여 학습의 안정성을 높이는 것이다. 주요 기여 사항은 다음과 같다.

첫째, Replay Buffer를 활용하는 Off-policy 학습 체계를 도입하여 과거의 상호작용 데이터를 재사용함으로써 환경 샘플 효율성을 높였다. 
둘째, 별도의 Discriminator 네트워크를 유지하는 대신, 보상 학습 기능을 Critic 네트워크에 직접 통합하여 최적화해야 할 파라미터 수를 줄이고 네트워크 간 상호작용으로 인한 불안정성을 제거하였다.
셋째, $\tanh$ 활성화 함수를 적용한 Bounded Actor 구조와 노이즈 입력을 통한 재매개변수화(Reparameterization)를 도입하여, 행동 값이 유효 범위를 벗어나 데이터가 낭비되는 문제를 해결하였다.
넷째, Clipped Double Q-learning 및 Soft Target Update를 적용하여 Off-policy RL에서 흔히 발생하는 Q-값의 과대평가(Overestimation) 문제를 억제하고 학습의 수렴 안정성을 확보하였다.

## 📎 Related Works

논문에서는 기존 모방 학습의 세 가지 주요 접근 방식과 그 한계를 설명한다.

1. **Behaviour Cloning (BC)**: 전문가의 상태-행동 쌍을 지도 학습(Supervised Learning) 방식으로 학습하는 가장 단순한 방법이다. 그러나 학습 데이터와 실제 실행 시의 데이터 분포가 달라지는 공변량 변화(Covariate Shift) 문제로 인해, 전문가 데이터가 부족할 경우 오류가 누적되어 치명적인 실패로 이어지는 한계가 있다.
2. **Inverse Reinforcement Learning (IRL)**: 전문가의 행동을 정당화하는 잠재적 비용/보상 함수를 먼저 찾고, 이를 통해 정책을 학습하는 방식이다. 하지만 보상 함수 업데이트(Outer-loop)와 정책 최적화(Inner-loop)가 중첩된 구조를 가지며, 내측 루프에서 매번 RL을 완전히 수렴시켜야 하므로 계산 비용이 매우 높고 샘플 효율성이 낮다.
3. **Generative Adversarial Imitation Learning (GAIL)**: IRL의 Max-entropy 문제를 점유 측정치 매칭(Occupancy Measure Matching) 문제로 변환하여 GAN 구조로 푼 방식이다. 기존 IRL의 중첩 루프 문제를 해결했지만, TRPO와 같은 On-policy 알고리즘에 의존함으로써 환경 샘플 효율성이 극도로 낮다는 단점이 있다.

## 🛠️ Methodology

### 1. Off-Policy Actor-Critic 및 Bounded Actor
본 논문은 DDPG 스타일의 Off-policy Actor-Critic 구조를 채택한다. Actor $\pi_\theta$는 상태 $s$와 노이즈 $z$를 입력으로 받아 행동을 결정하는 결정론적 정책(Deterministic Policy) 형태를 띠며, 최종 출력층에 $\tanh$를 적용하여 행동 범위를 $[-1, 1]$로 제한한다. 이를 통해 행동 값이 클리핑되어 그래디언트 신호가 소실되는 문제를 방지한다. 정책 업데이트는 다음과 같은 그래디언트를 사용한다.

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\beta, z \sim P_z} \left[ \nabla_a Q^{\pi_\theta, \nu}(s, a) \big|_{a=\pi_\theta(s, z)} \nabla_\theta \pi_\theta(s, z) \right]$$

### 2. Reward 및 Value Learning의 통합
본 제안 방법론의 핵심은 보상 함수 $R_\omega(s, a) = \log(r_\omega(s, a))$와 Q-함수 $Q^{\pi_\theta, \nu}(s, a) = \log(q^{\pi_\theta, \nu}(s, a))$를 로그 확률 도메인에서 정의하는 것이다. 수정된 벨만 방정식(Modified Bellman Equation)은 다음과 같다.

$$\log(q^{\pi_\theta, \nu}(s_t, a_t)) = \mathbb{E}_{s_{t+1}, a_{t+1}} \left[ \log(r_\omega(s_t, a_t) \cdot q^{\pi_\theta, \nu}(s_{t+1}, a_{t+1})^\gamma) \right]$$

여기서 Discriminator를 제거하기 위해, 전문가 데이터에 대해서는 $r^*_\omega = 1$ ($\log r^*_\omega = 0$), 비전문가 데이터에 대해서는 $r^*_\omega = 0.5$ ($\log r^*_\omega = -\log 2$)라는 최적 보상 값을 직접 Critic의 타겟에 주입한다. 결과적으로 Critic $\nu$의 최적화 목표는 전문가 데이터($D_E$)와 정책 데이터($D_\pi$)에 대한 두 개의 Jensen-Shannon Divergence (JSD) 손실 합을 최소화하는 것이 된다.

$$\arg \min_\nu \mathbb{E}_{D_E} [D_{JS}(P_\nu \parallel \mathbb{E}[P^\gamma_\nu])] + \mathbb{E}_{D_\pi} [D_{JS}(P_\nu \parallel \mathbb{E}[P^\gamma_\nu / 2])]$$

### 3. 안정화 기법 (Stabilization)
Off-policy 학습의 불안정성을 해결하기 위해 두 가지 기법을 적용한다.
- **Soft Target Updates**: 타겟 네트워크 $\nu_{target}$을 $\nu_{target} \leftarrow \tau \nu + (1-\tau)\nu_{target}$ 형태로 천천히 업데이트하여 학습 목표의 진동을 줄인다.
- **Clipped Double Q-Learning**: 두 개의 독립적인 Critic 네트워크를 운용하며, 타겟 값 계산 시 두 네트워크의 최솟값을 사용하여 Q-값의 과대평가를 방지한다.
  $$Q^{target}(s', a') = \min(Q^{\nu_{1, target}}(s', a'), Q^{\nu_{2, target}}(s', a'))$$

## 📊 Results

### 실험 설정
- **환경**: OpenAI Gym의 `BipedalWalker-v2` (24차원 상태 공간, 4차원 연속 행동 공간).
- **전문가 데이터**: PPO를 통해 500만 스텝 학습 후, 에피소드 보상이 300 이상인 상위 100개의 궤적을 추출하여 사용하였다.
- **비교 대상**: On-policy 기반의 GAIL.
- **지표**: 환경 상호작용 스텝 수에 따른 에피소드 보상(Episodic Return).

### 정량적 결과
실험 결과, 제안된 "Off Imitation" 알고리즘은 GAIL 대비 압도적인 샘플 효율성을 보여주었다. 제안 방법은 약 200,000 스텝 이내에 전문가 수준의 보상인 300점에 도달한 반면, GAIL은 훨씬 느리게 학습하며 더 낮은 성능에서 정체되는 모습을 보였다. 특히, 환경의 실제 보상 함수에 접근하지 않고 오직 전문가의 궤적만을 이용하여 전문가 수준의 성능을 빠르게 복원했다는 점이 중요하다.

## 🧠 Insights & Discussion

본 논문은 Adversarial Imitation Learning에서 발생하는 샘플 효율성 저하의 원인을 정확히 짚어내고, 이를 해결하기 위한 공학적 설계(Off-policy 전환, Discriminator 통합, Bounded Actor)를 체계적으로 제시하였다. 특히 보상 함수 학습을 Critic 내부에 내재화하여 네트워크 복잡도를 줄인 점은 학습 안정성 측면에서 매우 효율적인 접근이다.

다만, 결과 그래프에서 나타나듯 Off-policy 방식 특유의 높은 분산(Variance) 문제가 관찰된다. 이는 학습 과정에서 보상이 급격히 변동하는 양상으로 나타나며, 저자들 또한 이를 한계점으로 언급하고 있다. 이를 해결하기 위해 Actor는 On-policy로, Critic은 Off-policy로 업데이트하는 하이브리드 방식이나, 하위 최적(Suboptimal) 데모데이터에서도 학습 가능한 메커니즘의 도입이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 GAIL의 심각한 샘플 효율성 문제를 해결하기 위해 **Off-policy Actor-Critic 구조와 JSD 기반의 통합 Critic 학습 방법**을 제안한다. 별도의 Discriminator 없이 Critic이 보상 학습을 동시에 수행하며, Clipped Double Q-learning과 Bounded Actor를 통해 안정성을 높였다. 실험 결과, `BipedalWalker-v2` 환경에서 GAIL보다 훨씬 적은 환경 상호작용만으로 전문가 수준의 성능에 도달함으로써 실제 적용 가능성을 입증하였다.