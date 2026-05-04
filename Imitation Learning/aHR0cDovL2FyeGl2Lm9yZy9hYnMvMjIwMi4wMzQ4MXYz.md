# A Ranking Game for Imitation Learning

Harshit Sikchi, Akanksha Saran, Wonjoon Goo, Scott Niekum (2023)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)에서 보상 함수를 정의하는 어려움과 학습 데이터의 효율성 문제를 해결하고자 한다. 특히, 전문가의 행동(action) 정보 없이 상태 관찰값만으로 학습해야 하는 Learning from Observation (LfO) 설정은 전문가의 행동을 추론해야 하므로 탐색 비용이 매우 높고 학습이 어렵다는 문제가 있다.

기존의 역강화학습(Inverse Reinforcement Learning, IRL) 방식은 전문가의 시연 데이터(expert demonstrations)로부터 학습하지만, 오프라인 선호도(offline preferences) 데이터를 통합할 수 있는 메커니즘이 부족하다. 반면, 선호도 기반 학습은 고차원 보상 함수를 추론하기 위해 방대한 양의 선호도 데이터가 필요하다는 한계가 있다. 또한, 많은 기존 IRL 방법론들이 채택하고 있는 적대적 min-max 게임 형태의 최적화는 학습의 불안정성과 낮은 샘플 효율성을 초래한다. 따라서 본 연구의 목표는 전문가 시연과 선호도 데이터를 통합하여 학습 효율을 높이고, 최적화 과정의 안정성을 확보할 수 있는 새로운 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 모방 학습을 정책 에이전트(Policy Agent)와 보상 에이전트(Reward Agent) 사이의 **랭킹 게임(Ranking Game)**으로 재정의한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **통합 프레임워크 제안**: 전문가 데이터와 선호도 데이터를 동시에 활용할 수 있도록 모방 학습을 랭킹 기반의 게임으로 정식화하였다.
2. **새로운 랭킹 손실 함수 $L_k$ 제안**: 단순한 이진 분류가 아니라, 선호되는 행동과 그렇지 않은 행동 사이에 사용자가 제어 가능한 일정한 성능 격차 $k$를 유도하는 회귀 기반의 손실 함수를 제안하였다.
3. **데이터 증강(Trajectory Mixup)**: 궤적 공간에서 convex combination을 통해 보상 함수의 지형(landscape)을 부드럽게 만들어 정책 최적화를 용이하게 하는 자동 랭킹 생성 기법을 도입하였다.
4. **Stackelberg 게임 구조 적용**: 정책과 보상 에이전트 중 누가 리더(Leader)가 되느냐에 따라 PAL(Policy as Leader)과 RAL(Reward as Leader) 두 가지 알고리즘을 도출하여, 환경 변화에 따른 적응 성능을 최적화하였다.

## 📎 Related Works

기존 모방 학습 연구는 크게 행동 복제(Behavioral Cloning, BC)와 역강화학습(IRL)으로 나뉜다. GAIL, AIRL과 같은 전통적인 IRL 방법론들은 전문가와 에이전트 사이의 상태-행동 분포를 일치시키려 하지만, 이는 적대적 학습의 불안정성이라는 문제를 안고 있다.

선호도 기반 학습(Learning from Preferences) 분야에서는 T-REX와 같은 연구들이 하위 최적(suboptimal) 궤적들 간의 순위를 이용하여 보상을 추론하려 했다. 그러나 이러한 방식들은 대개 전문가 시연 데이터를 함께 사용하는 LfD/LfO 설정과의 유기적인 통합이 부족했다. 특히 LfO 설정에서는 전문가의 행동 정보가 없기 때문에 탐색의 어려움이 극심하며, 기존의 OPOLO와 같은 최신 방법론들조차 복잡한 조작 작업(manipulation tasks)에서는 한계를 보였다. 본 연구는 이러한 한계를 극복하기 위해 오프라인 선호도를 통해 보상 함수를 정규화(regularization)하고 정책 학습의 가이드를 제공하는 방식을 취한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 프레임워크는 두 명의 플레이어가 참여하는 일반-합 게임(general-sum game)으로 구성된다.

- **보상 에이전트(Reward Agent)**: 주어진 데이터셋 $D_p$에 포함된 쌍별 행동 랭킹(pairwise behavior rankings)을 만족하는 보상 함수 $R$을 학습한다.
- **정책 에이전트(Policy Agent)**: 학습된 보상 함수 $R$ 하에서 기대 수익(expected return)을 최대화하는 정책 $\pi$를 학습한다.

### 랭킹 손실 함수 $L_k$

보상 에이전트는 다음과 같은 $L_k$ 손실 함수를 최소화함으로써 학습한다.

$$L_k(D_p; R) = \mathbb{E}_{(\rho_i, \rho_j) \sim D_p} \left[ \mathbb{E}_{s,a \sim \rho_i} [(R(s,a) - 0)^2] + \mathbb{E}_{s,a \sim \rho_j} [(R(s,a) - k)^2] \right]$$

여기서 $\rho_i \preceq \rho_j$는 $\rho_j$가 더 선호되는 행동임을 의미하며, $k$는 두 행동 간에 유도하고자 하는 성능 격차(performance gap)를 나타내는 하이퍼파라미터이다. 이는 보상 함수의 스케일을 직접 제어할 수 있게 하여, 너무 크거나 작은 보상 값으로 인해 발생하는 최적화 문제를 방지한다.

### 데이터 증강 및 선호도 통합

- **자동 생성 랭킹 (auto)**: 두 궤적 $\tau_i, \tau_j$ 사이를 $\lambda$ 값으로 보간(interpolation)하여 $\tau_{\lambda} = \lambda \tau_i + (1-\lambda) \tau_j$ 형태의 중간 궤적들을 생성한다. 이에 대응하는 보상 타겟 역시 선형적으로 배분하여 보상 함수의 지형을 매끄럽게 만든다.
- **오프라인 선호도 (pref)**: 전문가 데이터뿐만 아니라 외부에서 제공된 오프라인 랭킹 데이터셋 $D_{offline}$을 함께 사용한다. 전체 손실 함수는 다음과 같이 가중 합으로 정의된다.
$$\mathcal{L} = \alpha L_k(D_{\pi}; R) + (1-\alpha) L_k(D_{offline}; R)$$

### Stackelberg 게임 최적화 전략

본 연구는 학습의 안정성을 위해 한 플레이어가 리더가 되고 다른 플레이어가 최적 반응(best response)을 보이는 Stackelberg 구조를 제안한다.

- **PAL (Policy as Leader)**: 정책이 리더가 된다. 보상 함수를 현재 정책 데이터에 대해 빠르게 수렴시킨 후, 정책을 소폭 업데이트한다. 의도(intent) 변화가 있는 환경에서 빠르게 적응하는 특성을 보인다.
- **RAL (Reward as Leader)**: 보상이 리더가 된다. 지금까지 수집된 모든 정책 데이터를 누적하여 보상을 보수적으로 업데이트한다. 환경의 동역학(dynamics)이 변하는 상황에서 더 강건한 성능을 보인다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**: MuJoCo의 Locomotion 작업(Hopper, HalfCheetah, Walker2d, Ant, Humanoid)과 복잡한 조작 작업(Door opening, Pen manipulation)을 사용하였다.
- **평가 지표**: 정규화된 성능(normalized performance) 및 샘플 효율성(sample efficiency)을 측정하였다.
- **비교 대상**: GAIfO, DACfO, BCO, f-IRL, OPOLO, IQ-Learn 등 최신 LfO/LfD 방법론들과 비교하였다.

### 주요 결과

1. **샘플 효율성 및 수렴 성능**: MuJoCo locomotion 작업에서 단 하나의 전문가 궤적만으로도 기존 SOTA 방법론인 OPOLO보다 월등히 높은 샘플 효율성을 보였으며, 전문가 수준의 성능에 빠르게 도달하였다.
2. **복잡한 작업 해결 능력**: 기존 LfO 방법론들이 전혀 해결하지 못한 Pen-v0 및 Door-v0 작업에서, 오프라인 선호도를 활용한 `RANK-PAL/RAL (pref)` 방식만이 성공적으로 전문가 행동을 모방하였다. 이는 LfO의 고질적인 문제인 탐색의 어려움을 오프라인 선호도가 효과적으로 가이드했음을 시사한다.
3. **PAL vs RAL 분석**: 의도 변경(Intent-Adaptation) 실험에서는 PAL이, 환경 동역학 변경(Dynamics-Adaptation) 실험에서는 RAL이 더 빠르게 적응함을 확인하여 두 알고리즘의 상보적 특성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 적대적 학습의 불안정성을 회귀 기반의 랭킹 손실 함수 $L_k$로 대체함으로써 학습의 안정성을 확보하였다. 특히, 보상 값의 범위를 $k$라는 하이퍼파라미터로 명시적으로 제어한 것이 Deep RL의 최적화 성능을 높이는 데 결정적인 역할을 하였다.

또한, 궤적 공간에서의 Mixup 기법을 통해 생성된 자동 랭킹이 보상 함수의 gradient signal을 더 명확하게 만들어 샘플 효율성을 극대화했음을 알 수 있다. 이론적으로는 Theorem 4.1을 통해 제안된 랭킹 게임의 평형 상태에서 에이전트와 전문가 간의 성능 격차가 유계(bounded)됨을 증명하여 방법론의 정당성을 부여하였다.

다만, 현실 세계의 선호도 데이터는 매우 노이즈가 심할 수 있는데, 본 논문에서는 이에 대한 구체적인 처리 방식을 제안하지 않았다는 점이 한계로 지적된다. 또한, 보상 함수의 스케일 $k$나 업데이트 빈도와 같은 하이퍼파라미터를 수동으로 설정해야 한다는 점 역시 향후 자동화가 필요한 부분이다.

## 📌 TL;DR

본 연구는 모방 학습을 보상 에이전트와 정책 에이전트 간의 **랭킹 게임**으로 정의하고, 새로운 **회귀 기반 랭킹 손실 함수 $L_k$**를 제안하였다. 이를 통해 전문가 시연 데이터와 오프라인 선호도를 통합하여 학습할 수 있게 되었으며, 특히 LfO 설정에서 극심한 탐색 비용 문제를 해결하여 복잡한 조작 작업을 성공적으로 수행하였다. 이 프레임워크는 샘플 효율성을 획기적으로 높였으며, Stackelberg 게임 구조를 통해 환경 변화에 따른 적응 전략(PAL, RAL)을 제공함으로써 향후 로봇 학습 및 실시간 적응형 모방 학습 연구에 중요한 기여를 할 것으로 보인다.
