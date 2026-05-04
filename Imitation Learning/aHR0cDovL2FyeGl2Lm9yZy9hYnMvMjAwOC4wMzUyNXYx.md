# Non-Adversarial Imitation Learning and its Connections to Adversarial Methods

Oleg Arenz and Gerhard Neumann (2020)

## 🧩 Problem to Solve

본 논문은 Imitation Learning (IL) 및 Inverse Reinforcement Learning (IRL) 분야에서 널리 사용되는 Adversarial formulation의 불안정성 문제를 해결하고자 한다. GAIL(Generative Adversarial Imitation Learning)이나 AIRL(Adversarial Inverse Reinforcement Learning)과 같은 현대적인 방법론들은 Generative Adversarial Networks (GANs)의 구조를 채택하여, 전문가의 상태-행동 분포($q$)와 에이전트의 정책으로 유도된 분포($p_\pi$)를 일치시키는 saddle point problem으로 문제를 정의한다.

그러나 이러한 Adversarial 방식은 최적화 과정에서 매우 불안정하며, 이론적인 수렴성을 보장하기 위해서는 정책 업데이트(policy update)의 크기가 매우 작아야 한다는 제약 조건이 따른다. 만약 업데이트가 너무 크면 보상 신호(reward signal)가 현재 정책에 지나치게 특화되어 빠르게 무효화되는 문제가 발생한다. 따라서 본 논문의 목표는 saddle point problem을 회피하여 더 강력한 수렴 보장과 안정적인 최적화를 제공하는 **Non-Adversarial Imitation Learning (NAIL)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 분포 일치 문제를 $\min \max$ 게임이 아닌, **Reverse Kullback-Leibler (RKL) divergence의 상한선(upper bound)을 반복적으로 좁히고 최적화하는 문제**로 재정의하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **NAIL 프레임워크 제안**: Adversarial 방식과 유사한 구조를 가지면서도 saddle point problem을 포함하지 않아, 정책 업데이트 크기에 상관없이 더 강력한 수렴 보장을 제공하는 Non-Adversarial Imitation Learning 프레임워크를 도입하였다.
2.  **AIRL의 재해석**: 기존의 AIRL이 사실상 본 논문에서 제안한 non-adversarial formulation의 한 사례임을 증명하였다. 이를 통해 AIRL의 이론적 유도 과정을 단순화하고 수렴성 보장을 강화하였다.
3.  **O-NAIL 알고리즘 개발**: 환경과의 상호작용 없이 전문가 데이터만으로 학습하는 Offline Imitation Learning 방법론인 O-NAIL을 제안하였다. 이는 ValueDice 알고리즘에서 영감을 얻었으나, saddle point problem을 해결하지 않고도 수렴 가능함을 보였다.

## 📎 Related Works

기존의 Imitation Learning 연구들은 주로 전문가의 분포와 에이전트의 분포 사이의 거리를 최소화하는 Distribution-matching 방식으로 접근해 왔다. 

-   **KL Divergence 기반 방법**: Forward KL은 모든 전문가 데이터를 커버하려는 mode averaging 특성이 있어 위험한 행동을 유발할 수 있는 반면, Reverse KL은 전문가의 데이터가 있는 영역에 집중하는 mode-seeking 특성이 있어 더 안전한 행동을 유도한다.
-   **Adversarial 기반 방법 (GAIL, AIRL)**: GAN의 구조를 이용하여 implicit distribution을 매칭한다. 특히 AIRL은 Maximum Causal Entropy IRL의 기반 위에서 reward function을 학습하려 하지만, 본 논문에서는 AIRL의 기존 이론적 근거(Maximum Likelihood gradient와의 일치성)가 수렴 후에나 유효하며 일반적인 상황에서는 약하다는 점을 지적한다.
-   **차별점**: 제안된 NAIL은 기존의 adversarial 방법들이 가진 $\min \max$ 구조의 불안정성을 제거하고, 보조 분포(auxiliary distribution)를 도입하여 보상 함수가 현재 정책에 의존하지 않도록 설계함으로써 안정성을 확보하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 핵심 아이디어
NAIL의 핵심은 Reverse KL divergence를 최소화하는 문제를 해결하기 위해, 보조 분포 $\tilde{p}(\tau)$를 도입하여 목적 함수의 하한선(lower bound)을 구성하고 이를 반복적으로 최대화하는 것이다. 이는 Expectation-Maximization (EM) 알고리즘과 유사한 방식으로 동작한다.

### 주요 방정식 및 수식 설명

#### 1. 목적 함수와 하한선 유도
기본 목표는 다음의 Reverse KL divergence를 최소화하는 것이다.
$$\min_\pi D_{RKL}(p_\pi(o) || q(o)) = \max_\pi \int_\tau p_\pi(\tau) r(\tau, \pi) d\tau$$
여기서 $r(\tau, \pi)$는 현재 정책 $\pi$에 의존하므로 불안정하다. 이를 해결하기 위해 보조 분포 $\tilde{p}(\tau)$를 도입한 하한선 $L$을 유도하면 다음과 같다.
$$L = H(\pi) + \int_o p_\pi(o) \log \frac{q(o)}{\tilde{p}(o)} do + (1-\gamma) \sum_t \gamma^t \int_s p^\pi_t(s) \int_a \pi(a|s) \log \tilde{\pi}(a|s) dads$$
여기서 $H(\pi)$는 정책의 discounted causal entropy이다.

#### 2. Non-Adversarial Reward Function
Step-based 설정에서, NAIL의 하한선 보상 함수(lower bound reward)는 다음과 같이 정의된다.
$$r^{\text{lb}}_{\tilde{\pi}}(s, a) = \log \left( \frac{q(s, a)}{p_{\tilde{\pi}}(s, a)} \right) + \log \tilde{\pi}(a|s)$$
이 식의 의미는 단순히 전문가 분포와 보조 분포의 밀도 비율(density ratio)을 보는 것에 그치지 않고, $\log \tilde{\pi}(a|s)$ 항을 더함으로써 이전 정책 $\tilde{\pi}$로부터 너무 멀어지지 않도록 페널티를 부여하는 것과 같다.

#### 3. O-NAIL (Offline Non-Adversarial Imitation Learning)
O-NAIL은 환경 상호작용 없이 Offline 데이터를 사용하여 학습한다. 이를 위해 다음과 같은 Actor-Critic 구조를 사용한다.

-   **Critic 업데이트**: Adversarial reward $r^{\text{adv}}$에 대한 Q-함수 $Q^{\tilde{\pi}}_{r^{\text{adv}}}$를 학습한다. 이때 Lemma 2에 의해, 하한선 보상에 대한 soft Q-함수는 다음과 같이 간단히 표현된다.
    $$Q^{\text{soft}, \tilde{\pi}}_{r^{\text{lb}}}(s, a) = Q^{\tilde{\pi}}_{r}(s, a) + \log \tilde{\pi}(a|s)$$
-   **Actor 업데이트**: 다음과 같은 I-Projection loss를 최소화하여 정책을 업데이트한다.
    $$L^{\text{actor}, \pi^{(i)}}(\pi) = -\mathbb{E}_{s \sim z(s)} \left[ \int_a \pi(a|s) (Q^{\pi^{(i)}}_{\text{adv}} + \log \pi^{(i)}(a|s) - \log \pi(a|s)) da \right]$$

## 📊 Results

### 실험 설정
-   **데이터셋 및 환경**: Mujoco의 Hopper, Walker2d, HalfCheetah, Ant 환경에서 실험을 진행하였다.
-   **비교 대상**: Behavioral Cloning (BC), ValueDice.
-   **지표**: Approximation return (에이전트가 달성한 평균 보상).
-   **초기화**: 모든 정책은 Behavioral Cloning으로 초기화하여 분포의 중첩(overlap)을 확보하였다.

### 주요 결과
-   **성능**: O-NAIL은 대부분의 환경과 전문가 데모 수(1, 2, 5, 10, 20개)에 걸쳐 ValueDice와 BC보다 우수한 성능을 보였다.
-   **업데이트 횟수**: Table 1에 따르면 O-NAIL은 ValueDice보다 훨씬 더 많은 수의 gradient step($N_\pi, N_Q$)을 사용하여 정책과 Q-함수를 업데이트했음에도 불구하고 발산하지 않고 안정적으로 성능이 향상되었다. 이는 saddle point problem을 제거함으로써 얻은 이론적 이점이 실제 실험에서도 유효함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 Adversarial IL의 고질적인 문제인 '작은 업데이트 크기 제약'을 이론적으로 해결하였다. NAIL 프레임워크를 통해 AIRL이 단순한 $\min \max$ 게임이 아니라, 사실상 하한선을 최적화하는 non-adversarial 방법론임을 밝혀낸 점은 학술적으로 매우 중요한 통찰이다. 또한 O-NAIL을 통해 offline 설정에서도 안정적인 학습이 가능함을 입증하였다.

### 한계점 및 비판적 논의
1.  **초기화 의존성**: 실험에서 O-NAIL은 BC로 초기화했을 때만 성공했으며, 무작위 초기화 시에는 학습에 실패하였다. 이는 밀도 비율 추정 시 분포의 support가 겹치지 않으면 gradient가 사라지는 문제가 여전히 존재함을 의미한다.
2.  **다양성 부족**: 현재 NAIL은 Reverse KL divergence에만 국한되어 있다. 다른 $f$-divergence나 Wasserstein distance 등으로 확장 가능한지에 대해서는 아직 미해결 과제로 남아 있다.
3.  **실용적 차이의 모호함**: 알고리즘 구조가 기존 adversarial 방법과 매우 유사하기 때문에, 단순히 성능 수치만으로는 non-adversarial 구조의 이점을 완전히 체감하기 어려울 수 있다. 다만, 업데이트 횟수를 획기적으로 늘릴 수 있다는 점이 실질적인 차별점이다.

## 📌 TL;DR

본 논문은 Imitation Learning의 불안정한 $\min \max$ 최적화 구조를 해결하기 위해, Reverse KL divergence의 하한선을 반복적으로 최적화하는 **Non-Adversarial Imitation Learning (NAIL)** 프레임워크를 제안하였다. 이를 통해 AIRL의 이론적 정당성을 재확립하였으며, offline 환경에서 안정적으로 작동하는 **O-NAIL** 알고리즘을 통해 기존 ValueDice 및 BC보다 뛰어난 성능을 입증하였다. 이 연구는 향후 offline RL에서 일반화 가능한 보상 함수를 학습하는 연구에 중요한 기초를 제공할 것으로 기대된다.