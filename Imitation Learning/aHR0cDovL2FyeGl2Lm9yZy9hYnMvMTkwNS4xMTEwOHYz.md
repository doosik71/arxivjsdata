# SQIL: Imitation Learning via Reinforcement Learning with Sparse Rewards

Siddharth Reddy, Anca D. Dragan, Sergey Levine

## 🧩 Problem to Solve

모방 학습(Imitation Learning)은 전문가의 시범(demonstration)을 통해 에이전트를 학습시키는 중요한 방법론입니다. 기존의 행동 복제(Behavioral Cloning, BC) 방식은 시범 데이터의 행동을 탐욕적으로 모방하여, 에이전트가 시범 상태 분포를 벗어나면 누적 오류로 인해 성능이 저하되는 분포 변화(distribution shift) 문제를 겪습니다. 역강화 학습(Inverse Reinforcement Learning, IRL)이나 GAN 기반 모방 학습(Generative Adversarial Imitation Learning, GAIL)과 같은 최신 RL 기반 방법들은 이 문제를 해결하지만, 알려지지 않은 보상 함수를 추정하기 위해 적대적 학습(adversarial training)과 같은 복잡하고 불안정한 기법을 사용해야 합니다. 본 논문은 이러한 복잡성 없이 RL을 사용하여 분포 변화 문제를 극복하는 더 간단하고 견고한 모방 학습 방법을 제안합니다.

## ✨ Key Contributions

* **SQIL (Soft Q Imitation Learning) 제안**: 학습된 보상 함수나 적대적 학습 없이도 장기적인 모방을 가능하게 하는 간단하고 효과적인 모방 학습 알고리즘을 제안합니다.
* **희소 보상(Sparse Rewards) 활용**: 에이전트가 시범된 상태에서 시범된 행동을 수행하면 상수 보상 $+1$을, 그 외의 모든 행동에는 $0$의 보상을 부여하는 매우 희소한 보상 구조를 사용합니다.
* **BC의 분포 변화 문제 극복**: SQIL은 BC의 주요 단점인 분포 변화 문제를 해결하며, GAIL과 같은 복잡한 방법들과 경쟁할 만한 성능을 보여줍니다.
* **이론적 해석**: SQIL이 동적 정보(dynamics information)를 정책에 통합하고 보상 함수에 희소성 사전(sparsity prior)을 적용하는 정규화된 BC(Regularized BC)의 한 형태로 해석될 수 있음을 증명합니다.
* **구현 용이성**: 표준 Q-학습 또는 오프-정책 액터-크리틱(off-policy actor-critic) 알고리즘에 몇 가지 간단한 수정만으로 쉽게 구현할 수 있습니다.
* **광범위한 환경에서의 성능 입증**: Box2D, Atari, MuJoCo의 다양한 이미지 기반 및 저차원 태스크에서 BC를 능가하고 GAIL과 경쟁적인 결과를 달성했습니다.

## 📎 Related Works

* **행동 복제(Behavioral Cloning, BC)**: 시범 데이터를 이용한 지도 학습 기반 방법으로, 분포 변화 문제에 취약합니다.
* **역강화 학습(Inverse Reinforcement Learning, IRL)**: 전문가의 시범으로부터 보상 함수를 학습하고 이를 바탕으로 정책을 학습하는 RL 기반 방법입니다.
* **GAN 기반 모방 학습(Generative Adversarial Imitation Learning, GAIL)**: IRL의 한 형태로, GAN을 사용하여 전문가의 정책과 에이전트의 정책 간 분포를 일치시키려 합니다.
* **Deep Q-learning from Demonstrations (DQfD) & Normalized Actor-Critic (NAC)**: 시범 데이터로 리플레이 버퍼를 초기화하는 측면에서 SQIL과 유사하지만, 환경으로부터 보상 신호를 받는 RL 알고리즘이라는 점에서 차이가 있습니다.
* **희소성 정규화(Sparsity Regularization)**: Piot et al. [2013, 2014]의 연구에서 제안된, 암시적 보상 함수에 대한 희소성 사전(sparsity prior)을 통해 BC를 정규화하는 기법입니다.
* **상수 보상을 사용하는 동시 연구**: SQIL과 동시에 Sasaki et al. [2019] 및 Wang et al. [2019]에서도 학습된 보상 함수 대신 상수 보상을 사용하는 모방 학습 알고리즘이 개발되었습니다.

## 🛠️ Methodology

SQIL은 소프트 Q-학습(soft Q-learning) 알고리즘을 기반으로 다음 세 가지 핵심적인 수정을 가합니다.

1. **시범 데이터로 리플레이 버퍼 초기화**: 에이전트의 경험 리플레이 버퍼($D_{\text{demo}}$)를 전문가의 시범 데이터로 채우며, 이 시범 경험에 대한 보상을 상수 $r = +1$로 설정합니다.
2. **새로운 경험에 대한 보상 설정**: 에이전트가 환경과 상호작용하여 새로운 경험($D_{\text{samp}}$)을 수집하고 리플레이 버퍼에 추가할 때, 이 새로운 경험에 대한 보상은 상수 $r = 0$으로 설정합니다.
3. **경험 샘플링 균형**: 리플레이 버퍼에서 미니 배치(mini-batch)를 샘플링할 때, 시범 경험과 새로운 경험의 수를 각각 $50\%$씩 균형 있게 샘플링하여 학습에 사용합니다.

이러한 수정은 에이전트에게 시범된 상태에서 전문가를 모방하도록 유도하고, 시범 상태를 벗어났을 때 시범 상태로 되돌아오도록 장려하는 간단한 보상 구조를 만듭니다. 오프-정책(off-policy) 알고리즘이기 때문에, 에이전트는 실제로 시범 상태를 방문하지 않고도 버퍼에 저장된 시범 데이터를 리플레이하여 긍정적인 보상을 학습할 수 있습니다.

SQIL은 다음 손실 함수를 최소화하여 소프트 Q-함수 $Q_{\theta}$를 학습합니다:
$$
\nabla_{\theta} \left( \delta^2(D_{\text{demo}}, 1) + \lambda_{\text{samp}} \delta^2(D_{\text{samp}}, 0) \right)
$$
여기서 $\delta^2(D, r)$은 식 (1)에 정의된 제곱 소프트 벨만 오류(squared soft Bellman error)이며, $r \in \{0, 1\}$은 상수 보상입니다. SQIL은 연속 액션 공간에 대해서는 소프트 액터-크리틱(Soft Actor-Critic, SAC)과 같은 오프-정책 액터-크리틱 방법론에 적용될 수 있습니다.

## 📊 Results

* **이미지 기반 Car Racing**: 초기 상태 분포가 전문가 시범과 다른 경우($S^0_{\text{train}}$), SQIL은 BC와 GAIL-DQL을 크게 능가하며 새로운 상황에 일반화하는 능력을 보였습니다. 초기 상태 분포가 동일한 경우($S^0_{\text{demo}}$)에도 BC와 비슷한 성능을 보였고 GAIL-DQL보다는 우수했습니다.
* **이미지 기반 Atari (Pong, Breakout, Space Invaders)**: 모든 게임에서 SQIL은 BC를 능가했으며, GAIL-DQL보다도 우수한 성능을 보였습니다. 이는 이미지 기반의 판별자 학습이 어려운 GAIL의 단점을 보여줍니다.
* **저차원 MuJoCo (Humanoid, HalfCheetah)**: 연속 액션 제어 태스크에서 SQIL은 BC를 능가하고 GAIL-TRPO와 비교할 만한 성능을 보여주며, 적은 수의 시범 데이터로도 잘 작동함을 입증했습니다.
* **저차원 Lunar Lander (Ablation Study)**: 초기 상태 변화가 있는 환경에서 SQIL은 BC, GAIL 및 모든 SQIL 변형(환경 상호작용 없이, 동적 정보 없이, 무작위 샘플링)보다 훨씬 뛰어난 성능을 보였습니다. 이는 환경과의 상호작용을 통한 샘플링과 동적 정보의 중요성을 확인시켜 줍니다. 또한, SQIL은 정규화된 BC(RBC)보다도 훨씬 우수한 성능을 보여 RBC의 $V(s_0)$ 항이 성능을 저하시킬 수 있음을 시사했습니다.

## 🧠 Insights & Discussion

SQIL의 핵심 통찰은 적대적 학습이나 복잡한 보상 함수 학습 없이도 희소한 상수 보상을 활용하여 RL 기반 모방 학습의 효과를 얻을 수 있다는 것입니다. 이는 복잡한 RL 알고리즘 구현의 어려움을 해소하고 실용성을 높입니다. 이론적으로, SQIL은 전문가 정책과 환경 동역학에 대한 정보를 통합하는 정규화된 행동 복제(BC)로 해석될 수 있으며, 이는 SQIL이 분포 변화 문제를 극복하는 원리를 설명합니다.

**함의**: 이 연구는 단순한 RL과 상수 보상 기반의 모방 방법이 학습된 보상을 사용하는 더 복잡한 방법만큼 효과적일 수 있다는 개념 증명(proof of concept)을 제공합니다. 이는 모방 학습 분야에서 더 간단하고 견고한 접근 방식의 잠재력을 시사합니다.

**한계 및 향후 연구**:

* SQIL이 무한한 시범 데이터가 주어졌을 때 전문가의 상태 점유 측정(state occupancy measure)과 일치하는지는 아직 증명되지 않았습니다.
* 향후 연구 방향으로는 SQIL을 확장하여 전문가의 정책뿐만 아니라 보상 함수도 복구하는 방법을 모색할 수 있습니다. 예를 들어, 상수 보상 대신 매개변수화된(parameterized) 보상 함수를 사용하여 소프트 벨만 오류 항의 보상을 모델링하는 것입니다. 이는 기존 적대적 IRL 알고리즘에 대한 더 간단한 대안을 제공할 수 있습니다.

## 📌 TL;DR

본 논문은 분포 변화 문제와 복잡한 보상 함수 학습 없이 전문가의 행동을 모방하는 간단하고 효과적인 알고리즘인 SQIL을 제안합니다. SQIL은 소프트 Q-학습에 시범 데이터 초기화, 상수 희소 보상(+1/0), 경험 샘플링 균형의 세 가지 수정을 적용하며, 이론적으로는 정규화된 행동 복제의 한 형태로 해석됩니다. 실험 결과, SQIL은 BC를 능가하고 GAIL과 경쟁적인 성능을 보이며, 특히 이미지 기반 태스크와 새로운 초기 상태에 대한 일반화 능력에서 강점을 보여줍니다.
