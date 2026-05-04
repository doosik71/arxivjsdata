# Imitation Learning from Observations by Minimizing Inverse Dynamics Disagreement

Chao Yang, Xiaojian Ma, Wenbing Huang, Fuchun Sun, Huaping Liu, Junzhou Huang, Chuang Gan (2019)

## 🧩 Problem to Solve

본 논문은 전문가의 액션(action) 정보 없이 상태(state) 관찰 데이터만으로 정책을 학습하는 Learning from Observations (LfO) 문제를 다룬다. 일반적으로 Learning from Demonstration (LfD)는 전문가의 상태와 액션을 모두 사용하여 학습하므로 상대적으로 쉽지만, 실제 환경에서 전문가의 정확한 액션 데이터를 수집하는 것은 매우 어렵거나 불가능한 경우가 많다(예: 인터넷 비디오 데이터 활용).

LfO의 핵심 난제는 동일한 상태 전이(state transition)에 대해 여러 가지 액션이 존재할 수 있다는 점이다. 특히 로봇 제어와 같이 자유도가 높은 시스템에서는 동일한 포즈 변화를 만들어내는 관절 제어 방법이 무수히 많을 수 있어, 상태 관찰만으로 최적의 액션을 결정하는 것이 매우 어렵다. 따라서 본 논문의 목표는 LfD와 LfO 사이의 이론적 간극을 분석하고, 이를 최소화하여 LfO의 성능을 LfD 수준으로 끌어올리는 새로운 학습 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LfD와 LfO의 최적화 차이가 전문가와 모방자 사이의 Inverse Dynamics 모델의 불일치, 즉 **Inverse Dynamics Disagreement**에서 기인한다는 것을 이론적으로 증명한 것이다.

저자들은 이 불일치를 직접 계산하는 것이 어렵다는 점을 인지하고, 이를 모델 프리(model-free) 방식으로 최소화할 수 있는 상한선(upper bound)을 유도하였다. 구체적으로, Inverse Dynamics Disagreement를 최소화하는 것이 상태-액션 점유 측정치(state-action occupancy measure)의 음의 인과 엔트로피(negative causal entropy)를 최소화하는 것과 연결됨을 보였으며, 이를 통해 Mutual Information(MI) 극대화와 정책 엔트로피 극대화를 통해 LfO의 성능을 향상시키는 **Inverse-Dynamics-Disagreement-Minimization (IDDM)** 방법론을 제안하였다.

## 📎 Related Works

기존의 모방 학습 연구는 크게 두 가지 방향으로 나뉜다.

1.  **Learning from Demonstrations (LfD):** Behavior Cloning (BC)이나 Inverse Reinforcement Learning (IRL) 계열의 방법들이 있으며, 대표적으로 GAIL은 전문가와 에이전트의 점유 측정치(occupancy measure) 간의 차이를 최소화한다. 하지만 이러한 방법들은 모두 정확한 전문가의 액션 가이드가 필요하다는 한계가 있다.
2.  **Learning from Observations (LfO):** 
    *   **Hand-crafted Reward 기반:** DeepMimic과 같이 물리적 특성을 맞추도록 설계된 보상 함수를 사용한다. 하지만 이는 보상 함수를 직접 설계해야 하며, 액션에 의존적인 보상 체계를 가진 작업에는 일반화하기 어렵다.
    *   **Model-Based LfO:** BCO는 역동학 모델(inverse dynamics model)을 통해 액션을 추론하여 BC를 수행한다. 그러나 역동학 모델이 현재 정책에 의존하므로, 최적 정책이 학습되기 전에는 정확한 역동학 모델을 얻을 수 없어 이론적 보장이 어렵고 과적합(overfitting)의 위험이 있다.
    *   **GAIL 기반 LfO:** GAIfO는 GAIL의 상태-액션 쌍을 상태 전이(state transition) 쌍으로 대체하여 학습한다. 본 논문에서는 GAIfO가 Inverse Dynamics Disagreement를 고려하지 않기 때문에 LfD와의 간극을 메울 수 없음을 지적하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
IDDM은 기본적으로 GAIfO의 구조를 계승하되, Inverse Dynamics Disagreement를 최소화하기 위한 추가적인 정규화 항을 목적 함수에 도입한다. 시스템은 상태 전이를 판별하는 Discriminator $D_\phi$와 정책 네트워크 $\pi_\theta$, 그리고 Mutual Information을 추정하는 MI Estimator $I$로 구성된다.

### 핵심 이론 및 방정식
본 논문은 전문가 $\pi^E$와 에이전트 $\pi$ 사이의 Inverse Dynamics Disagreement를 다음과 같이 정의한다.
$$\text{Inverse Dynamics Disagreement} := D_{KL}(\rho^\pi(a|s, s') || \rho^E(a|s, s'))$$
여기서 $\rho^\pi(a|s, s')$는 상태 $s$에서 $s'$로 전이되었을 때 액션 $a$가 선택될 확률을 나타내는 Inverse Dynamics 모델이다.

**Theorem 1**에 따르면, 에이전트와 전문가가 동일한 동역학 시스템을 공유할 때 다음과 같은 관계가 성립한다.
$$D_{KL}(\rho^\pi(a|s, s') || \rho^E(a|s, s')) = D_{KL}(\rho^\pi(s, a) || \rho^E(s, a)) - D_{KL}(\rho^\pi(s, s') || \rho^E(s, s'))$$
이는 LfD의 목적 함수($D_{KL}(\rho^\pi(s, a) || \rho^E(s, a))$)와 Naive LfO의 목적 함수($D_{KL}(\rho^\pi(s, s') || \rho^E(s, s'))$) 사이의 차이가 정확히 Inverse Dynamics Disagreement임을 의미한다.

### 최종 학습 목적 함수
현실적으로 $\rho^E$를 알 수 없으므로, 저자들은 이 간극의 상한선을 유도하여 다음과 같은 최종 목적 함수 $L^\pi$를 제안한다.
$$L^\pi = D_{KL}(\rho^\pi(s, s') || \rho^E(s, s')) - \lambda_p H^\pi(a|s) - \lambda_s I^\pi(s; (s', a))$$

각 항의 역할은 다음과 같다:
1.  **$D_{KL}(\rho^\pi(s, s') || \rho^E(s, s'))$**: Naive LfO 항으로, 전문가의 상태 전이 분포를 모방한다. GAN 구조의 Discriminator를 통해 구현된다.
2.  **$H^\pi(a|s)$**: 정책 엔트로피 항으로, 탐색을 촉진하고 불확실성을 관리한다.
3.  **$I^\pi(s; (s', a))$**: 상태 $s$와 $(s', a)$ 쌍 사이의 Mutual Information 항이다. 결정론적 시스템에서 $H^\pi(s) = I^\pi(s; (s', a))$가 성립하며, 이를 통해 Inverse Dynamics Disagreement의 상한선을 최소화한다.

### 학습 절차
1.  에이전트의 rollout 데이터를 수집하고, MINE(Mutual Information Neural Estimation)를 사용하여 $I^\pi(s; (s', a))$를 업데이트한다.
2.  Discriminator $D_\phi$를 업데이트하여 에이전트의 상태 전이와 전문가의 상태 전이를 구분하도록 학습시킨다.
3.  PPO(Proximal Policy Optimization)를 사용하여 정책 $\pi_\theta$를 업데이트하며, 이때 위에서 정의한 $L^\pi$를 최소화하는 방향으로 경사 하강법을 적용한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 환경:** CartPole, Pendulum, DoublePendulum, Hopper, HalfCheetah, Ant 등 6가지 MuJoCo 기반 연속 제어 벤치마크를 사용하였다.
- **비교 대상:** DeepMimic, BCO, GAIfO, GAIfO-s(단일 상태 입력 버전), 그리고 Oracle 기준인 GAIL을 비교군으로 설정하였다.
- **지표:** 각 환경에서 정의된 원래의 보상(reward) 합계를 측정하였다.

### 주요 결과
1.  **정량적 성능:** IDDM은 쉬운 작업(CartPole)에서는 기존 LfO 방법들과 유사한 성능을 보였으나, 고차원 제어가 필요한 어려운 작업(Ant, Hopper)에서 다른 LfO 방법론들을 압도하는 성능 향상을 보였다. 특히 Ant 환경에서는 GAIfO 대비 매우 높은 보상을 기록하며 GAIL(LfD)에 근접하는 성능을 나타냈다.
2.  **학습 안정성:** BCO의 경우 학습이 진행됨에 따라 성능이 급격히 떨어지는 현상이 관찰되었으나, IDDM은 model-free 방식으로서 일관되게 안정적인 학습 곡선을 보여주었다.
3.  **데모 데이터 수량 영향:** 데모 데이터의 수가 적은 상황에서도 IDDM은 GAIfO보다 우수한 성능을 유지하였으며, 데이터가 증가함에 따라 GAIL과 대등한 수준까지 성능이 향상됨을 확인하였다.
4.  **Toy Experiment (Gridworld):** 액션 선택지의 수를 늘려 Inverse Dynamics Disagreement를 인위적으로 증가시켰을 때, GAIfO와 GAIL의 성능 격차가 벌어지며, IDDM이 그 중간에서 간극을 효과적으로 메우고 있음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 LfO의 성능 저하 원인을 단순히 '데이터 부족'이 아닌 'Inverse Dynamics Disagreement'라는 이론적 관점에서 정의하고 이를 해결했다는 점에서 큰 강점을 가진다. 

특히, Inverse Dynamics 모델을 직접 학습시켜 액션을 예측하는 BCO 방식이 가지는 과적합 위험을 지적하고, 이를 엔트로피와 Mutual Information이라는 정보 이론적 도구를 통해 model-free 방식으로 해결한 점이 인상적이다. Ablation study를 통해 $\lambda_s$(MI weight)와 $\lambda_p$(policy entropy weight)가 모두 성능 향상에 기여함을 보였으며, 특히 MI 항이 고차원 환경에서 결정적인 역할을 함을 확인하였다.

다만, 본 연구는 결정론적(deterministic) 시스템을 가정하고 있으며, 실제 환경의 노이즈가 심한 확률적 동역학 시스템에서도 이 상한선 유도가 그대로 적용될 수 있을지에 대해서는 추가적인 논의가 필요해 보인다. 또한, 현재는 제어 문제에 집중하고 있으나, 저자들이 언급했듯이 표현 학습(representation learning)과 결합한다면 서로 다른 도메인 간의 모방 학습(cross-domain imitation)으로 확장될 가능성이 높다.

## 📌 TL;DR

본 논문은 전문가의 액션 없이 상태 관찰만으로 학습하는 LfO에서, LfD와의 성능 격차가 **Inverse Dynamics Disagreement**에서 발생함을 이론적으로 증명하고 이를 해결하는 **IDDM** 방법론을 제안하였다. IDDM은 Mutual Information 극대화와 정책 엔트로피 관리를 통해 이 간극을 model-free 방식으로 최소화하며, 특히 고차원 로봇 제어 작업에서 기존 LfO 방법론들을 압도하는 성능과 안정성을 보여주었다. 이 연구는 비디오 데이터와 같은 상태 전이 데이터만으로 고수준의 기술을 학습시키려는 향후 연구에 중요한 이론적 토대를 제공한다.