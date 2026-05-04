# Deconfounding Imitation Learning with Variational Inference

Risto Vuorio, Pim de Haan, Johann Brehmer, Hanno Ackermann, Daniel Dijkman, and Taco Cohen (2022)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)에서 전문가(Expert)와 모방자(Imitator)가 관찰하는 정보의 불일치로 인해 발생하는 **인과적 혼동(Causal Confounding)** 문제를 해결하고자 한다. 구체적으로, 전문가는 환경의 잠재 변수(Latent variable) $\theta$를 관찰하여 행동을 결정하지만, 모방자는 이를 관찰할 수 없는 상황을 다룬다.

이러한 설정에서 표준적인 행동 복제(Behavioral Cloning, BC)를 수행하면 모방자는 과거의 상태와 행동 기록을 통해 $\theta$를 추론하려 시도한다. 그러나 모방자가 자신의 과거 행동을 $\theta$에 대한 증거로 잘못 사용하는 **인과적 망상(Causal Delusion)** 현상이 발생한다. 예를 들어, 자율주행차가 "내가 빠르게 달리고 있으니 도로에 얼음이 없을 것"이라고 잘못 판단하는 것과 같다. 이는 모방자가 전문가의 데이터 분포(Conditional distribution)는 학습하지만, 실제 환경에서의 개입(Intervention) 결과인 interventional policy를 학습하지 못하기 때문에 발생한다.

논문의 목표는 전문가에게 쿼리를 보내지 않고도, 변분 추론(Variational Inference)을 통해 잠재 변수를 식별함으로써 인과적 혼동이 제거된 모방 학습 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **전문가 데이터가 아닌 탐색 데이터(Exploration data)를 사용하여 잠재 변수 추론 모델을 학습시키는 것**이다.

전문가 데이터는 $\theta$에 의해 행동과 상태 전이가 모두 결정되어 있어 인과적으로 얽혀(confounded) 있지만, 모방자의 탐색 데이터는 $\theta$와 독립적인 정책으로 수집되므로 상태 전이($s \to s'$)만을 통해 $\theta$를 순수하게 추론할 수 있다. 이를 통해 모방자는 전문가의 행동에 의존하지 않고 환경의 역학(Dynamics)만을 이용해 $\theta$를 추론하는 **Interventional Policy**를 구현할 수 있게 된다.

## 📎 Related Works

기존의 모방 학습 및 인과 관계 관련 연구들은 다음과 같은 한계점을 가진다.

1. **Behavioral Cloning (BC):** 앞서 언급한 인과적 망상 문제로 인해, 전문가와 모방자의 관찰 정보가 다를 때 성능이 급격히 저하된다.
2. **DAgger (Dataset Aggregation):** 학습 과정에서 전문가에게 지속적으로 정답 쿼리를 요청함으로써 문제를 해결한다. 하지만 실제 환경에서 전문가(예: 숙련된 인간 운전자)에게 매번 쿼리를 보내는 것은 비용이 너무 크거나 불가능할 수 있다.
3. **Inverse Reinforcement Learning (IRL) 및 GAIL:** 보상 함수를 역으로 추론하여 해결하려 하지만, 적대적 학습(Adversarial learning)의 불안정성과 확장성 문제로 인해 실제 구현 시 학습이 어렵고 불안정한 경우가 많다.
4. **Causality-aware IL:** Ortega et al. (2021) 등이 인과적 망상 문제를 지적했으나, 이들의 해결책 역시 전문가 쿼리에 의존하는 경향이 있다.

본 논문은 전문가 쿼리 없이, 환경과의 상호작용(탐색)과 변분 추론만을 사용하여 이 문제를 해결한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Interventional Policy의 정의

논문은 단순한 조건부 정책(Conditional policy) $\pi_{cond}$ 대신, 과거의 행동을 개입(intervention)으로 처리하여 오직 환경의 역학만을 증거로 사용하는 **Interventional Policy** $\pi_{int}$를 목표로 한다.

$$\pi_{int}(a_t | s_{1:t}, a_{1:t-1}) = \mathbb{E}_{\theta \sim p_{int}(\theta | \tau)} [\pi_{exp}(a_t | s_t, \theta)]$$
$$p_{int}(\theta | \tau) \propto p(\theta) \prod_{t} p(s_{t+1} | s_t, a_t, \theta)$$

여기서 $p_{int}$는 과거 행동 $a_t$가 $\theta$에 의해 결정된 것이 아니라 외부에서 주어진 것이라고 가정하고 $\theta$를 추론하는 분포이다.

### 2. 전체 시스템 구조 및 학습 절차

제안된 방법론은 두 단계의 학습 과정으로 구성된다 (Figure 3 참조).

#### 단계 1: 잠재 변수 추론 모델 학습 (Deconfounding)

모방자의 탐색 정책 $\pi_{expl}$을 사용하여 수집한 데이터를 통해 인코더 $q_\phi$와 역학 모델 $p_\psi$를 학습한다. 탐색 정책은 $\theta$에 의존하지 않으므로, 여기서 학습된 $q_\phi$는 자연스럽게 interventional latent를 추론하게 된다. 학습 목적 함수는 ELBO(Evidence Lower Bound)를 최대화하는 것이다.

$$\hat{L}_{VI} = \mathbb{E}_{\tau \sim p_{expl}} \left[ \mathbb{E}_{\hat{\theta} \sim q_\phi(\hat{\theta}|\tau_{:t})} [\log p_\psi(s_{t+1}|s_t, a_t, \hat{\theta})] - \beta D_{KL}(q_\phi(\hat{\theta}|\tau_{:t}) \| q_\phi(\hat{\theta}|\tau_{:t-1})) \right]$$

이 과정에서 인코더 $q_\phi$는 궤적 $\tau$를 입력받아 잠재 변수의 분포를 출력하며, 역학 모델 $p_\psi$는 $\hat{\theta}$가 주어졌을 때 다음 상태 $s_{t+1}$을 예측한다.

#### 단계 2: 잠재 조건부 정책 학습 (Imitation)

학습된 인코더 $q_\phi$를 전문가 데이터 $\tau_e$에 적용하여 각 궤적의 잠재 변수 $\hat{\theta}$를 추론한다. 이후, 이 $\hat{\theta}$를 조건으로 하여 전문가의 행동을 복제하는 정책 $\pi_\eta(a|s, \hat{\theta})$를 학습시킨다.

$$L_{BC} = \mathbb{E}_{\tau \sim p} \mathbb{E}_{\hat{\theta} \sim q_\phi(\hat{\theta}|\tau)} [\log \pi_\eta(a_t | s_t, \hat{\theta})]$$

### 3. 추론 및 실행 (Test Time)

테스트 시, 에이전트는 다음과 같이 동작한다.

1. 초기 잠재 변수 $\hat{\theta}$를 샘플링한다.
2. $\pi_\eta(a|s, \hat{\theta})$에 따라 행동하고 상태 전이를 관찰한다.
3. 관찰된 전이 데이터를 인코더 $q_\phi$에 넣어 $\hat{\theta}$에 대한 믿음(belief)을 업데이트한다.
4. 업데이트된 $\hat{\theta}$를 사용하여 다시 행동한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 환경:**
  - Multi-armed Bandit (인과적 망상 분석용)
  - LunarLander-v2 (키 바인딩 latent)
  - HalfCheetah (목표 속도 latent)
  - AntGoal (목표 위치 latent)
- **비교 대상:** Naive BC (RNN 기반), DAgger, GAIL, Oracle (정답 latent를 아는 경우)
- **평가 지표:** Episode Return, 정답 Arm 선택 확률

### 2. 주요 결과

- **Multi-armed Bandit:** Naive BC는 전문가 데이터에서는 높은 확률로 정답을 맞히지만, 실제 배포 시에는 자신의 과거 행동을 근거로 잘못된 추론을 하여 성능이 급락했다. 반면 제안 방법은 True Interventional Policy와 거의 일치하는 성능을 보였다.
- **Control Environments:**
  - **Naive BC:** 모든 환경에서 인과적 망상으로 인해 매우 낮은 성능을 보였다.
  - **GAIL:** 이론적으로는 가능해야 하나, 실제로는 학습의 불안정성으로 인해 Naive BC보다 약간 나은 수준이거나 오히려 낮은 성능을 보였다.
  - **Proposed Method (Deconfounded IL):** Naive BC와 GAIL을 압도하였으며, 많은 경우 DAgger의 성능에 근접하였다. 특히 LunarLander와 HalfCheetah에서 탁월한 성능 향상을 보였다.
  - **DAgger:** 전문가 쿼리를 사용하므로 가장 높은 성능(Upper bound)을 보였으나, 실용성 측면에서 제안 방법이 우위에 있다.

## 🧠 Insights & Discussion

### 강점

본 연구는 모방 학습에서 발생하는 '인과적 망상'이라는 난제를 변분 추론과 탐색 데이터의 결합이라는 명쾌한 방법으로 해결하였다. 특히 전문가 쿼리 없이도 Interventional Policy를 식별할 수 있음을 이론적으로 증명하고 실증적으로 보여주었다는 점이 매우 강력하다.

### 한계 및 가정

1. **정적 잠재 변수 가정:** 본 논문은 $\theta$가 에피소드 동안 일정하다고 가정한다. 도로의 구간마다 특성이 바뀌는 실제 주행 환경과 같은 동적 잠재 변수 상황에서는 적용에 한계가 있을 수 있다.
2. **탐색 데이터 필요성:** 전문가 쿼리는 필요 없지만, 환경과의 상호작용을 통한 탐색 데이터가 필요하다. 완전히 offline 데이터만으로 학습해야 하는 상황에서는 제안된 알고리즘의 변형(Appendix C)이 필요하며, 이는 더 어려운 학습 문제로 이어진다.
3. **재귀성(Recurrence) 가정:** 이론적 증명을 위해 모든 상태-행동 쌍이 무한히 자주 방문된다는 강한 가정을 사용하였다.

### 비판적 해석

GAIL과 같은 IRL 기반 방법론이 이론적 가능성에도 불구하고 실제로는 성능이 낮게 나온 점은 주목할 만하다. 이는 모방 학습에서 단순한 보상 함수 추론보다, 인과 관계를 고려한 잠재 변수 식별이 훨씬 더 안정적이고 효율적인 접근법일 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 전문가가 가진 추가 정보(잠재 변수)를 모방자가 갖지 못할 때 발생하는 **인과적 망상(Causal Delusion)** 문제를 해결하기 위해, **변분 추론 기반의 Deconfounding 모방 학습** 방법을 제안한다. 탐색 데이터를 통해 환경의 역학만을 이용하는 잠재 변수 추론 모델을 먼저 학습하고, 이를 기반으로 전문가의 행동을 복제함으로써 전문가 쿼리 없이도 최적의 **Interventional Policy**를 학습할 수 있음을 입증하였다. 이 연구는 고차원 제어 환경에서 Naive BC 및 GAIL보다 훨씬 뛰어난 성능을 보이며, 실용적인 인과 관계 인식 모방 학습의 가능성을 열었다.
