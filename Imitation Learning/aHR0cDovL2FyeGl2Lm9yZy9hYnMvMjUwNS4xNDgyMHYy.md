# Imitation Learning via Focused Satisficing

Rushit N. Shah, Nikolaos Agadakos, Synthia Sasulski, Ali Farajzadeh, Sanjiban Choudhury, Brian Ziebart (2025)

## 🧩 Problem to Solve

기존의 모방 학습(Imitation Learning)은 시연자(demonstrator)가 어떤 고정된 비용 함수(cost function)에 대해 최적 또는 최적에 가까운 행동을 한다는 가정을 전제로 한다. 그러나 실제 인간의 행동은 '만족화 이론(Satisficing Theory)'에 따라, 최적성을 추구하기보다 개인의 열망 수준(aspiration levels)에 비추어 '수용 가능한(acceptable)' 수준의 행동을 선택하는 경향이 있다.

이러한 차이는 다음과 같은 문제를 야기한다:
1. **가치 불일치(Value Misalignment):** 시연자가 최적이 아님에도 최적이라고 가정하고 학습하면, 모방자가 시연자의 실제 의도나 수용 기준과 다른 행동을 생성할 수 있다.
2. **성능 한계:** 단순히 행동을 복제하는 Behavioral Cloning(BC)은 시연자의 성능을 초과할 수 없으며, Inverse Reinforcement Learning(IRL) 계열은 추정된 단일 스칼라 보상 함수에 지나치게 의존하여 특정 지표만 최적화하고 다른 중요한 수용 기준을 무시하는 경향이 있다.

본 논문의 목표는 시연자의 구체적인 수용 기준을 명시적으로 학습하지 않고도, 시연자가 수용할 수 있는(acceptable) 행동을 생성하며 동시에 시연자의 성능을 능가하는 정책을 학습하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Subdominance(하위 우세)**라는 개념을 도입하여, 보상 함수를 추정하는 대신 정책의 최적화 목표를 직접적으로 "시연자의 행동을 Pareto-dominating(파레토 지배)하는 것"으로 설정하는 것이다.

- **MinSubFI (Minimally Subdominant Focused Imitation):** 시연자의 궤적보다 모든 비용 특징(cost features) 차원에서 더 낮은 비용을 갖도록(즉, 파레토 지배하도록) 유도하는 마진 기반의 목적 함수를 제안한다.
- **Snippet-focused Learning:** 전체 궤적이 아닌 궤적의 일부(snippets)에 집중하여, 노이즈가 많은 시연 데이터 속에서도 고품질의 부분 행동을 효율적으로 학습한다.
- **보상 함수 없는 최적화:** 명시적인 스칼라 보상 함수를 학습하고 이를 다시 최적화하는 복잡한 파이프라인 대신, Subdominance를 강화학습의 목적 함수로 직접 사용하여 정책을 최적화한다.

## 📎 Related Works

- **Behavioral Cloning (BC):** 시연자의 상태-행동 쌍을 직접 모방한다. 단순하지만 시연자의 성능을 넘어서지 못하며, 수용 가능성에 대한 보장이 없다.
- **Inverse Reinforcement Learning (IRL) 및 GAIL:** 시연자를 정당화하는 보상 함수를 추정한다. 하지만 추정된 보상 함수가 실제 수용 기준과 일치하지 않을 수 있으며, 단일 스칼라 값으로 압축하는 과정에서 정보 손실이 발생한다.
- **T-REX / D-REX:** 순위가 매겨진 시연을 통해 시연자를 능가하는 성능을 추구한다. 그러나 여전히 추정된 보상 함수에 의존하며, 보상 함수의 표현력 한계로 인해 특정 특징만 과도하게 최적화하는 문제가 발생할 수 있다.

## 🛠️ Methodology

### 1. Satisficing 및 Subdominance 정의
시연자의 수용 가능성(acceptability)은 비용 함수 $w$와 임계값 $\nu$에 대해 $\text{cost}_w(\xi) < \nu$일 때 충족된다고 정의한다. 이를 위해 본 논문은 **Subdominance**를 사용하여 모방자가 시연자를 얼마나 충분히 능가하지 못하고 있는지를 측정한다.

특징 $k$에 대한 Subdominance $\text{subdom}_k$는 다음과 같이 정의된다:
$$\text{subdom}_k^{\alpha_k}(\xi, \tilde{\xi}) = \left[ \alpha_k (f_k(\xi) - f_k(\tilde{\xi})) + 1 \right]^+$$
여기서 $[x]^+ = \max(x, 0)$는 힌지 함수(hinge function)이며, $\alpha_k$는 마진의 기울기를 결정하는 파라미터이다. 전체 Subdominance $\text{subdom}_\alpha$는 각 특징에 대한 값들의 합 또는 최대값으로 계산된다. $\text{subdom} = 0$이라는 것은 모방자의 궤적 $\xi$가 시연자의 궤적 $\tilde{\xi}$를 모든 특징에서 마진만큼 파레토 지배함을 의미한다.

### 2. MinSubFI 목적 함수
스토캐스틱 정책 $\pi_\theta$에 대해, 기대 Subdominance를 최소화하는 것이 목표이다:
$$\min_{\theta} \min_{\alpha \succeq 0} \sum_{\text{task } i} \frac{|\tilde{\Xi}_i|}{|\tilde{\Xi}|} \text{subdom}_\alpha(\pi_\theta, \tilde{\Xi}_i) + \lambda_\alpha \|\alpha\|^2 + \lambda_\theta \|\theta\|^2$$
이 식은 정책 $\pi_\theta$와 마진 $\alpha$를 동시에 최적화하여, 시연자 집단 $\tilde{\Xi}_i$에 대해 최대한 수용 가능한 행동을 생성하도록 유도한다.

### 3. Snippet-focused Subdominance
전체 궤적이 아닌 부분 궤적(snippet) $\xi_{sn}$에 집중하여 학습하는 방식이다.
$$\text{subdom}_{\text{snip} S}^\alpha(\xi, \tilde{\xi}) = \max_{(\xi_{sn}, \tilde{\xi}_{sn}) \in S(\xi, \tilde{\xi})} \text{subdom}_\alpha(\xi_{sn}, \tilde{\xi}_{sn})$$
이는 전체 궤적의 품질이 낮더라도 그 안에 포함된 고품질의 부분 행동을 포착하여 학습할 수 있게 한다.

### 4. 정책 경사 최적화 (Policy Gradient)
Subdominance를 강화학습의 신호로 사용하기 위해 다음과 같은 정책 경사(Policy Gradient) 형태를 도출한다:
$$\nabla_\theta \mathbb{E}_{\xi_i \sim \pi_\theta \times \tau_i} [\text{subdom}_\alpha(\xi_i, \tilde{\Xi}_i)] = \mathbb{E}_{\xi_i \sim \pi_\theta \times \tau_i} \left[ \text{subdom}_\alpha(\xi_i, \tilde{\Xi}_i) \sum_{(s,a) \in \xi_i} \nabla_\theta \log \pi_\theta(a|s) \right]$$
또한, 궤적 단위의 Subdominance를 상태 단위로 분해(Corollary 5)하여 PPO와 같은 최신 강화학습 알고리즘에 적용할 수 있도록 하여, 시간적 일관성을 가진 크레딧 할당(credit assignment)이 가능하게 했다.

### 5. 학습 절차 및 변형
- **Online MinSubFI:** 환경에서 직접 샘플링하며 Subdominance를 최소화한다.
- **Offline MinSubFI:** 중요도 샘플링(Importance Weighting)을 통해 기존 시연 데이터만으로 경사도를 추정한다.
- **MinSubFI-LCF:** 쌍별 선호도(pairwise preferences)를 이용하여 잠재적 비용 특징(latent cost features) $f_\psi$를 먼저 학습한 후 정책을 최적화한다.

## 📊 Results

### 실험 설정
- **환경:** Cartpole, LunarLander, Hopper, HalfCheetah, Walker2d (OpenAI Gym).
- **데이터:** PPO로 학습된 하위 최적(suboptimal) 정책의 시연 100개 및 LunarLander의 경우 실제 인간 시연 데이터 사용.
- **비교 대상:** BC, GAIL, AIRL, T-REX, $\text{T-REX}_{\text{CF}}$ (비용 특징 기반 T-REX).

### 주요 결과
1. **시연자 수용도 (Acceptability):** $\gamma$-satisficing 값(시연자가 수용할 확률)을 측정한 결과, MinSubFI가 모든 환경에서 기준선들보다 압도적으로 높은 수용도를 보였다. 특히 $\text{MinSubFI}_{\text{SNIP}}$이 매우 높은 성능을 기록했다.
2. **실제 리턴 (True Returns):**
   - 대부분의 환경에서 MinSubFI가 시연자의 평균 리턴을 상회하며, 많은 경우 기준선들보다 높은 성능을 보였다.
   - 특히 인간 시연 데이터를 사용한 LunarLander에서 다른 방법론들은 실패한 반면, $\text{MinSubFI}_{\text{ON}}$은 유의미한 성능 향상을 보였다.
3. **데이터 품질에 대한 강건성:** 시연 데이터셋에서 상위/하위 일부를 제거하여 데이터 품질을 낮추었을 때, MinSubFI는 T-REX나 AIRL보다 성능 저하가 훨씬 적어 노이즈 섞인 데이터에 강건함을 입증했다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **다차원 최적화의 이점:** T-REX와 같은 방법은 단일 스칼라 보상을 최적화하므로, 하나의 특징(예: Cartpole의 폴 각도)만 극단적으로 최적화하고 다른 특징(예: 카트의 위치)을 무시하는 '보상 게이밍' 현상이 발생한다. 반면 MinSubFI는 모든 비용 특징에서 파레토 지배를 추구하므로, 시연자가 중요하게 생각하는 여러 기준을 동시에 충족하는 균형 잡힌 정책을 학습한다.
- **Snippet 학습의 효율성:** 전체 궤적을 비교하는 것보다 부분 궤적을 비교하는 것이 학습 신호를 더 명확하게 제공하며, 이는 데이터의 국소적 고품질 구간을 효과적으로 추출하는 역할을 한다.

### 한계 및 논의
- **특징 설계 의존성:** 여전히 비용 특징 $f$를 수동으로 정의해야 하는 부담이 있다. LCF(Learned Cost Features)를 통해 이를 완화하려 했으나, 보조적인 주석(annotation) 없이 특징 표현을 완벽하게 학습하는 것은 여전히 어려운 문제로 남아 있다.
- **Degenerate Solution 방지:** 비용 특징을 단순히 최소화하면 에피소드를 빨리 종료하여 비용을 줄이려는 경향이 발생할 수 있다. 이를 방지하기 위해 'Trajectory Padding' 기법을 도입하여 일정 시간 동안은 행동하도록 강제했는데, 이는 특징 설계와 작업 목표 간의 미세한 불일치를 해결하기 위한 임시 방편이다.

## 📌 TL;DR

본 논문은 인간이 최적이 아닌 '수용 가능한' 수준에서 행동한다는 **Satisficing 이론**을 모방 학습에 접목하였다. 보상 함수를 추정하는 대신, 모방자가 시연자의 궤적을 모든 특징 차원에서 능가하도록 하는 **Subdominance 최소화** 목적 함수를 제안하였다. 특히 궤적의 일부만 사용하는 **Snippet-focused** 접근법을 통해 노이즈가 많은 데이터에서도 강건하게 학습하며, 실험적으로 시연자의 수용 가능성을 보장함과 동시에 시연자의 성능을 상회하는 결과를 얻었다. 이 연구는 향후 정교한 보상 설계 없이도 인간의 의도에 부합하는 고성능 에이전트를 학습시키는 데 중요한 기여를 할 것으로 보인다.