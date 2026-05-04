# Imitation Learning by State-Only Distribution Matching

Damian Boborzi, Christoph-Nikolas Straehle, Jens S. Buchner and Lars Mikelsons (2022)

## 🧩 Problem to Solve

본 논문은 전문가의 행동(action) 정보 없이 상태(state) 정보만 주어진 상황에서 정책을 학습하는 **Learning-from-Observations (LfO)** 문제를 다룬다. 현실 세계의 많은 시나리오(예: 비디오 녹화 데이터, 조감도 기반의 교통 데이터)에서는 전문가가 어떤 액션을 취했는지 알 수 없는 경우가 많으며, 이 경우 상태 정보만을 이용해 모방 학습을 수행해야 한다.

기존의 state-only 모방 학습 방식은 주로 **Adversarial Imitation Learning (AIL)**에 기반하고 있다. 그러나 AIL은 다음과 같은 치명적인 한계를 가진다:
1. **학습의 불안정성**: 적대적 학습(adversarial training) 특성상 최적화 과정이 불안정하며 수렴 보장이 어렵다.
2. **성능 추정의 어려움**: 환경의 실제 보상 함수(true reward function)를 알 수 없는 상황에서, 학습 중인 모델이 얼마나 잘 수행하고 있는지 판단할 수 있는 신뢰할 만한 수렴 지표(convergence estimator)가 부족하다. 이는 결국 실제 환경에서 성능이 낮은 모델을 선택하게 되는 결과로 이어진다.

따라서 본 논문의 목표는 적대적 학습을 배제한 **비적대적(non-adversarial)** 방식의 LfO 접근법을 제안하고, 이를 통해 학습의 안정성을 높이며 해석 가능한 성능 지표를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 정책이 유도하는 상태 전이 궤적(state transition trajectories)과 전문가의 궤적 간의 **Kullback-Leibler Divergence (KLD)**를 직접 최소화하는 것이다.

주요 기여 사항은 다음과 같다:
- **SOIL-TDM (State Only Imitation Learning by Trajectory Distribution Matching)** 제안: 적대적 최적화 없이 KLD를 최소화하는 비적대적 LfO 방법론을 제시하였다.
- **Max-Entropy RL로의 변환**: KLD 최소화 문제를 적절한 보상 함수를 정의함으로써 **Maximum Entropy Reinforcement Learning** 문제로 재정의하였으며, 이를 통해 **Soft Actor-Critic (SAC)** 알고리즘을 적용하여 효율적으로 최적화하였다.
- **Normalizing Flows 활용**: 전문가의 상태 전이 분포, 환경의 forward/backward dynamics를 모델링하기 위해 복잡한 확률 분포 표현에 능한 **Normalizing Flows (RealNVP)**를 도입하였다.
- **신뢰할 수 있는 성능 지표**: KLD 기반의 손실 함수를 통해 학습 과정 중 정책의 성능을 객관적으로 추정할 수 있는 지표를 제공함으로써, 실제 보상 함수 없이도 최적의 모델을 선택할 수 있게 하였다.

## 📎 Related Works

### 관련 연구 및 한계
1. **Learning-from-Demonstrations (LfD)**: 상태-액션 쌍$(s, a)$이 제공되는 설정이다. 하지만 실제 환경에서는 액션 데이터를 수집하는 비용이 매우 높거나 불가능한 경우가 많다.
2. **Adversarial Imitation Learning (AIL)**: GAIL, OPOLO, f-IRL 등이 대표적이다. 판별기(discriminator)를 통해 보상을 추정하지만, 앞서 언급한 대로 학습 불안정성과 수렴 판단의 어려움이 있다.
3. **Model-based LfO**: forward dynamics나 inverse action 모델을 학습하여 액션을 추론하는 방식(BCO 등)이 존재한다.

### 기존 접근 방식과의 차별점
- **OPOLO 및 f-IRL과의 차이**: 적대적 min-max 최적화를 사용하지 않으므로 학습이 훨씬 안정적이며, 추측에 의존하는 판별기 보상이 아닌 KLD라는 명확한 통계적 지표를 사용한다.
- **FORM과의 차이**: FORM 역시 조건부 밀도 추정(conditional density estimation)을 사용하지만, 본 논문은 정책의 상태-다음 상태 분포를 정책 엔트로피, forward dynamics, inverse action 모델로 분해하여 정의함으로써 **엔트로피의 중복 계산(double accounting)** 문제를 해결하고 샘플 효율성을 높였다.

## 🛠️ Methodology

### 전체 파이프라인
SOIL-TDM의 전체 구조는 전문가 데이터로부터 상태 전이 모델을 오프라인으로 학습한 후, 에이전트가 환경과 상호작용하며 수집한 데이터를 통해 dynamics 모델과 정책을 반복적으로 업데이트하는 구조이다.

### 핵심 방법론 및 방정식

#### 1. 목적 함수 (KLD 최소화)
정책 $\pi_\theta$가 유도하는 궤적 분포 $\mu_{\pi_\theta}$와 전문가의 궤적 분포 $\mu_E$ 사이의 KLD를 최소화하는 것이 목표이다.
$$J_{SOIL-TDM} = \sum_{i=0}^{T-1} \mathbb{E}_{(s_{i+1}, s_i) \sim \mu_{\pi_\theta}} [\log \mu_{\pi_\theta}(s_{i+1}|s_i) - \log \mu_E(s_{i+1}|s_i)]$$

#### 2. 보상 함수의 정의
위의 KLD 최소화 문제는 다음과 같은 보상 함수 $r(a_i, s_i)$를 사용하는 Max-Entropy RL 문제로 변환될 수 있다.
$$r(a_i, s_i) := \mathbb{E}_{s_{i+1} \sim p(s_{i+1}|a_i, s_i)} [-\log p(s_{i+1}|a_i, s_i) + \log \pi'_\theta(a_i|s_{i+1}, s_i) + \log \mu_E(s_{i+1}|s_i)]$$
여기서:
- $p(s_{i+1}|a_i, s_i)$: 환경의 **Forward Dynamics** 모델 (상태와 액션이 주어졌을 때 다음 상태의 확률)
- $\pi'_\theta(a_i|s_{i+1}, s_i)$: 환경의 **Inverse Action** 모델 (현재 상태와 다음 상태가 주어졌을 때 취해진 액션의 확률)
- $\mu_E(s_{i+1}|s_i)$: **Expert State Transition** 모델 (전문가가 $s_i$에서 $s_{i+1}$로 이동할 확률)

#### 3. 학습 절차
1. **오프라인 단계**: 전문가 데이터 $\mathcal{D}_E$를 사용하여 $\mu_E(s_{t+1}|s_t)$를 학습한다. 이때 과적합을 방지하기 위해 상태 값에 가우시안 노이즈를 추가하고 점진적으로 줄이는 기법을 사용한다.
2. **온라인 반복 단계**:
    - **데이터 수집**: 현재 정책 $\pi_\theta$로 환경과 상호작용하여 $(s, a, s')$ 데이터를 Replay Buffer $\mathcal{D}_{RB}$에 저장한다.
    - **Dynamics 모델 업데이트**: $\mathcal{D}_{RB}$의 샘플을 이용하여 Forward Dynamics 모델 $\mu_\phi$와 Inverse Action 모델 $\mu_\eta$를 최대 가능도 추정(MLE) 방식으로 학습한다.
    - **정책 최적화**: 정의된 보상 함수 $r$을 사용하여 **SAC (Soft Actor-Critic)** 알고리즘으로 정책 $\pi_\theta$와 Q-함수를 업데이트한다.

### 모델 아키텍처
모든 밀도 모델($\mu_E, \mu_\phi, \mu_\eta$)은 **RealNVP** 기반의 Conditional Normalizing Flows를 사용하여 구현되었다. 이는 고차원 연속 공간에서 복잡한 확률 분포를 유연하게 모델링하기 위함이다.

## 📊 Results

### 실험 설정
- **환경**: Pybullet 물리 시뮬레이터의 연속 제어 작업 (Ant, HalfCheetah, Hopper, Walker2D, Humanoid).
- **비교 대상**: OPOLO, f-IRL, FORM.
- **지표**: 전문가 성능을 1로 정규화한 상대적 보상(Relative Reward).
- **평가 시나리오**: 
    - **Scenario A (Unknown True Reward)**: 실제 보상을 모르고, 각 알고리즘이 제공하는 수렴 지표(KLD, loss 등)만을 이용해 최적 모델을 선택한 경우.
    - **Scenario B (Best True Reward)**: 실제 보상 함수를 사용하여 가장 성능이 좋은 체크포인트를 선택한 경우.

### 주요 결과
1. **학습 안정성 및 모델 선택**: 실제 보상을 알 수 없는 Scenario A에서 SOIL-TDM은 다른 baseline들에 비해 매우 낮은 분산을 보이며 안정적인 성능을 기록하였다. 특히 적대적 방식인 OPOLO와 f-IRL은 선택 지표의 불안정성으로 인해 성능 편차가 매우 컸다.
2. **데이터 효율성**: 전문가 궤적의 수가 적은 상황(1~2개)에서도 SOIL-TDM은 FORM보다 우수한 성능을 보였으며, OPOLO/f-IRL과 경쟁하거나 이를 능가하였다.
3. **성능 우위**: 실제 보상을 사용하여 최적 모델을 선택한 Scenario B에서도 SOIL-TDM은 경쟁력 있는 성능을 유지하거나 타 방법론을 압도하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구의 가장 큰 성과는 **"적대적 학습 없이도 분포 매칭(Distribution Matching)이 가능하다"**는 것을 증명한 점이다. 특히 KLD를 최소화하는 문제를 Max-Entropy RL 프레임워크로 통합함으로써, 기존 RL의 강력한 최적화 도구(SAC)를 그대로 사용할 수 있게 되었다. 또한, 학습 과정에서 계산되는 KLD 값이 실제 성능과 강한 상관관계를 가지므로, 보상 함수가 없는 현실 환경에서 모델의 수렴 여부를 판단할 수 있는 실용적인 도구를 제공한다.

### 한계 및 비판적 논의
- **가정 사항**: 논문은 전문가 데이터셋이 전문가의 분포를 정확하게 표현하고 있다고 가정한다. 데이터가 극도로 부족한 경우, Normalizing Flows의 밀도 추정 오류가 보상 함수의 노이즈로 작용할 가능성이 있다.
- **계산 복잡도**: 세 가지 Normalizing Flow 모델(Expert, Forward, Inverse)을 동시에 학습하고 유지해야 하므로, 단순한 MLP 기반 모델보다 계산 비용이 높을 수 있다.
- **보상 값의 범위**: SAC를 적용하기 위해 로그 확률 값들을 clipping($[-15, 1e9]$) 하였다. 이는 이론적인 KLD 최소화와 실제 구현 사이의 간극을 메우기 위한 조치이며, clipping 범위 설정에 따라 성능이 달라질 수 있는 하이퍼파라미터 민감도가 존재한다.

## 📌 TL;DR

본 논문은 전문가의 액션 정보 없이 상태 전이 궤적만을 이용하여 정책을 학습하는 **비적대적 모방 학습 방법론인 SOIL-TDM**을 제안한다. 정책과 전문가 간의 KLD를 최소화하는 문제를 Max-Entropy RL로 변환하여 SAC로 최적화하며, Normalizing Flows를 통해 정교한 상태 전이 확률을 모델링한다. 실험 결과, 기존 적대적 방식(OPOLO, f-IRL)보다 학습이 훨씬 안정적이며, 특히 **실제 보상 함수를 알 수 없는 상황에서 신뢰할 수 있는 성능 지표를 제공**함으로써 최적의 모델을 선택할 수 있다는 점에서 자율주행이나 로보틱스와 같은 실제 적용 가능성이 매우 높다.