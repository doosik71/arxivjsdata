# Generative Adversarial Self-Imitation Learning

Yijie Guo, Junhyuk Oh, Satinder Singh, Honglak Lee (2018)

## 🧩 Problem to Solve

강화학습(Reinforcement Learning, RL)의 본질적인 문제는 어떤 상태에서 어떤 행동을 취해야 미래에 더 나은 결과로 이어질지를 결정하는 Temporal Credit Assignment 문제이다. 특히 보상 신호가 매우 희소(sparse)하거나 지연(delayed)되어 제공되는 환경에서는 에이전트가 어떤 행동이 보상에 기여했는지 파악하기 어렵기 때문에 학습 속도가 매우 느려지거나 지역 최적점(local optima)에 빠지는 문제가 발생한다.

본 논문의 목표는 에이전트가 과거에 생성했던 '좋은 궤적(good trajectories)'을 모방하도록 유도하는 단순한 정규화 방법론인 Generative Adversarial Self-Imitation Learning (GASIL)을 제안하는 것이다. 이를 통해 보상이 희소한 환경에서도 학습 가능한 밀집된(dense) 보상 신호를 생성하여 Temporal Credit Assignment 문제를 완화하고자 한다.

## ✨ Key Contributions

GASIL의 핵심 아이디어는 Generative Adversarial Imitation Learning (GAIL) 프레임워크를 자기 모방(Self-Imitation)에 적용하는 것이다. 기존의 GAIL이 외부 전문가(Expert)의 데이터를 모방하는 것과 달리, GASIL은 에이전트가 스스로 수집한 데이터 중 보상이 높았던 상위 $K$개의 궤적을 전문가 데이터로 간주한다.

핵심 직관은 판별자(Discriminator)를 학습시켜 '현재의 정책이 생성한 궤적'과 '과거의 좋은 궤적'을 구분하게 하고, 정책(Policy)은 이 판별자를 속이도록(즉, 좋은 궤적을 그대로 재현하도록) 학습하는 것이다. 이 과정에서 판별자는 일종의 학습된 보상 함수(learned reward function) 역할을 수행하며, 에이전트에게 더 정밀하고 밀집된 가이드라인을 제공하게 된다.

## 📎 Related Works

본 논문은 크게 세 가지 관련 연구 분야를 다룬다.

**1. Generative Adversarial Learning 및 GAIL**
GAN(Generative Adversarial Networks)의 적대적 학습 구조를 모방 학습으로 확장한 것이 GAIL이다. GAIL은 전문가의 점유 측정치(occupancy measure)와 정책의 점유 측정치 사이의 Jensen-Shannon Divergence를 최소화한다. GASIL은 여기서 '전문가'의 존재를 '에이전트가 수집한 최선의 경험'으로 대체하여 RL 환경에 적용했다는 점이 차별점이다.

**2. Reward Learning**
환경에서 제공하는 원래의 보상 함수가 반드시 학습에 최적인 것은 아니며, 더 빠른 학습을 가능케 하는 최적 보상 함수(optimal reward function)가 존재할 수 있다는 관점이다. GASIL의 판별자는 이러한 내부 보상 함수를 학습하는 것으로 해석될 수 있으며, 이는 Reward Shaping의 일종으로 볼 수 있다.

**3. Self-Imitation Learning**
과거의 좋은 경험에 집중하여 정책을 학습하는 연구들이 존재한다. 예를 들어 Episodic Control이나 SIL(Self-Imitation Learning)은 과거 경험을 직접적으로 활용한다. GASIL은 이러한 자기 모방 개념을 적대적 학습 프레임워크로 확장함으로써, 단순한 행동 복제(Behavior Cloning)보다 더 효율적인 샘플 효율성을 달성하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

GASIL은 다음과 같은 반복적인 루프를 통해 학습된다.

1. 현재 정책 $\pi_\theta$를 통해 궤적 $\tau_\pi$를 샘플링한다.
2. 수집된 궤적 중 보상이 높은 상위 $K$개를 선택하여 좋은 궤적 버퍼 $B$에 저장한다.
3. 판별자 $D_\phi$를 학습시켜 $\tau_\pi$와 $\tau_E \in B$를 구분하게 한다.
4. 정책 $\pi_\theta$를 업데이트하여 판별자가 $\tau_\pi$를 $\tau_E$라고 믿게 만든다.

### 주요 구성 요소 및 방정식

**1. 판별자(Discriminator) 학습**
판별자 $D_\phi(s,a)$는 주어진 상태-행동 쌍이 정책에서 왔는지, 아니면 좋은 궤적 버퍼에서 왔는지를 판별한다. 목적 함수는 다음과 같은 binary cross-entropy 형태를 가진다.
$$\nabla_\phi L_{GASIL} = E_{\tau_\pi}[\nabla_\phi \log D_\phi(s,a)] + E_{\tau_E}[\nabla_\phi \log(1 - D_\phi(s,a))]$$
여기서 $\tau_\pi$는 현재 정책의 궤적, $\tau_E$는 버퍼 $B$에서 샘플링된 좋은 궤적이다.

**2. 정책(Policy) 학습**
정책은 판별자를 속이기 위해 $\log D_\phi(s,a)$를 최대화하는 방향으로 학습된다. 이때 Score function estimator를 사용하여 다음과 같이 업데이트한다.
$$\nabla_\theta L_{GASIL} = E_{\tau_\pi}[\nabla_\theta \log \pi_\theta(a|s)Q(s,a)] - \lambda \nabla_\theta H(\pi_\theta)$$
여기서 $Q(s,a) = E_{\tau_\pi}[\log D_\phi(s,a)|s_0=s, a_0=a]$이며, $H(\pi_\theta)$는 정책의 엔트로피로 탐색을 장려하기 위해 사용된다.

**3. Policy Gradient와의 결합**
GASIL은 단독으로 사용될 수도 있지만, 기존의 RL 목적 함수 $J_{PG}$와 결합하여 사용할 때 더 강력하다. 다음과 같이 수정된 보상 함수 $r_\alpha(s,a)$를 사용하여 Advantage $\hat{A}_t^\alpha$를 계산한다.
$$r_\alpha(s,a) = r(s,a) - \alpha \log D_\phi(s,a)$$
최종적인 정책 업데이트 방향은 다음과 같이 정의된다.
$$\nabla_\theta J_{PG} - \alpha \nabla_\theta L_{GASIL} = E_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) \hat{A}_t^\alpha]$$
여기서 $\alpha$는 판별자가 제공하는 내부 보상의 가중치를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **환경**: 2D Point Mass (단순 제어), OpenAI Gym MuJoCo (복잡한 제어).
- **비교 대상 (Baselines)**: PPO, PPO+BC (Behavior Cloning), PPO+SIL (Self-Imitation Learning).
- **지표**: 누적 보상(Average Reward) 및 학습 곡선.

### 주요 결과

**1. 2D Point Mass 환경**
PPO는 빠르게 학습하지만 지역 최적점(sub-optimal policy)에 쉽게 빠지는 경향을 보였다. 반면 PPO+GASIL은 초기 학습 속도는 다소 느리지만, 최종적으로는 훨씬 더 높은 보상을 얻는 최적 정책을 찾아냈다. 이는 판별자가 생성하는 내부 보상이 에이전트로 하여금 환경의 단순한 보상에 매몰되지 않고 더 넓은 영역을 탐색하게 했기 때문으로 분석된다.

**2. MuJoCo 환경**
대부분의 MuJoCo 태스크에서 PPO+GASIL이 PPO 및 다른 자기 모방 기법(BC, SIL)보다 우수한 성능을 보였다. 특히 GASIL은 BC보다 샘플 효율성이 높았으며, 이는 적대적 학습 구조가 비정상 데이터(non-stationary data) 환경에서 더 일반화 성능이 좋기 때문으로 해석된다.

**3. 지연된 보상(Delayed Reward) 설정**
보상을 20단계마다 한 번씩 제공하도록 수정한 극한 환경에서 GASIL의 효과가 가장 극명하게 나타났다. PPO와 다른 베이스라인들이 학습에 어려움을 겪는 반면, GASIL은 판별자가 제공하는 밀집 보상 덕분에 성능 하락폭이 훨씬 적었으며 압도적인 성능 향상을 보였다.

**4. 하이퍼파라미터 분석**

- **버퍼 크기**: 너무 작으면 샘플 부족으로 성능이 떨어지고, 너무 크면 낮은 품질의 궤적이 섞여 들어와 성능이 저하되는 Trade-off 관계가 존재함을 확인하였다.
- **판별자 업데이트 횟수**: GAN과 마찬가지로 판별자와 생성자(정책) 사이의 균형이 중요하며, 업데이트 횟수가 너무 적거나 많으면 성능이 떨어진다.

## 🧠 Insights & Discussion

**강점**
GASIL은 환경에서 제공하는 보상이 희소하거나 지연된 상황에서도, 에이전트가 스스로 발견한 '최선의 경험'을 바탕으로 가상의 보상 지도를 그려냄으로써 학습을 가속화한다. 이는 외부 전문가 데이터 없이도 자기 주도적으로 Reward Shaping을 수행하는 효과적인 정규화 방법이다.

**한계 및 비판적 해석**

1. **다중 모드(Multi-modal) 문제**: 논문에서 사용한 Gaussian Policy는 단일 모드 분포만을 표현할 수 있다. 하지만 좋은 궤적 버퍼 $B$에는 서로 다른 시점에 수집된 다양한 형태의 최적 경로(Multi-modal trajectories)가 포함될 수 있으며, 이를 Gaussian Policy로 모방하는 것은 이론적으로 한계가 있다.
2. **결정론적 환경 가정**: 논문에서도 언급되었듯, 자기 모방 기반 방식은 환경이 결정론적(deterministic)일 때 정책 개선이 보장된다. 확률적(stochastic) 환경에서의 강건성은 실험적으로 어느 정도 확인되었으나, 이론적인 보장이나 정교한 처리는 부족하다.
3. **판별자 학습의 불안정성**: GAN 계열의 고질적인 문제인 판별자와 생성자 간의 균형 잡기 문제가 하이퍼파라미터 민감도로 나타났다.

## 📌 TL;DR

본 논문은 에이전트가 수집한 과거의 상위 궤적들을 모방하도록 하는 **Generative Adversarial Self-Imitation Learning (GASIL)**을 제안하였다. 이 방법은 판별자를 통해 학습된 내부 보상을 제공함으로써, 특히 **보상이 희소하거나 지연된 환경에서 Temporal Credit Assignment 문제를 효과적으로 해결**한다. 결과적으로 PPO와 같은 기존 알고리즘의 성능을 유의미하게 향상시켰으며, 이는 향후 전문가 데이터가 없는 환경에서 스스로 학습 가이드를 생성하는 RL 연구에 중요한 기여를 할 것으로 보인다.
