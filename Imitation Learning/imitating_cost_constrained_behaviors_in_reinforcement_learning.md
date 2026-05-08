# Imitating Cost-Constrained Behaviors in Reinforcement Learning

Qian Shao, Pradeep Varakantham, Shih-Fen Cheng (2024)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)의 모방 학습(Imitation Learning, IL) 과정에서 **비용 제약 조건(Cost Constraints)**이 존재하는 환경을 다룬다. 일반적인 모방 학습은 전문가의 행동을 관찰하여 보상 모델을 학습하거나 정책(Policy)을 직접 복제하는 데 집중하며, 주로 연료 제한이나 안전 거리와 같은 제약 조건이 없는 Unconstrained setting을 가정한다.

그러나 실제 환경(예: 자율 주행 배달 차량, 레이싱 카)에서는 전문가의 결정이 단순히 보상(Reward)뿐만 아니라, 가용 연료, 시간 제한, 안전 경계 준수와 같은 비용 제약에 의해 결정된다. 특히, 환경에서 제공하는 비용 신호(Cost signals)는 알 수 있지만, 구체적으로 어느 정도의 수치까지가 안전한지에 대한 **최대 비용 임계값(Maximum cost limit)을 모르는 상태**에서 전문가의 궤적(Trajectories)만을 통해 이 제약을 학습하고 준수해야 하는 문제가 발생한다. 본 연구의 목표는 전문가의 행동 분포를 모방함과 동시에, 전문가가 준수한 비용 제약 조건을 만족하는 정책을 학습하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비용 제약이 있는 환경에서 전문가의 분포를 매칭하기 위해 **Lagrangian relaxation**과 **Meta-gradient**, 그리고 **Alternating gradient** 기법을 결합한 세 가지 방법론을 제안하는 것이다.

1. **CCIL (Cost-Constrained Imitation Lagrangian):** Lagrangian multiplier를 도입하여 보상 최대화와 비용 제약 준수 사이의 균형을 맞추는 3방향 그래디언트 업데이트 방식을 제안한다.
2. **MALM (Meta-gradient for Lagrangian Approach):** CCIL의 하이퍼파라미터인 Lagrangian penalty를 최적화하기 위해 온라인 교차 검증(Online cross-validation) 기반의 Meta-gradient 기법을 적용하여 성능을 개선한다.
3. **CVAG (Cost-Violation-based Alternating Gradient):** Lagrangian multiplier에 의존하지 않고, 현재 정책의 비용 준수 여부(Feasibility)에 따라 보상 최대화와 비용 최소화 방향으로 그래디언트 업데이트를 교차로 수행하는 직관적인 접근법을 제안한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구를 언급하며 차별점을 제시한다.

- **Malik et al. (2021):** 보상 함수가 이미 정의된 상태에서 전문가의 궤적으로부터 비용 제약을 학습하는 연구이다.
- **Cheng et al. (2023) [LGAIL]:** 비용 신호와 최대 비용 임계값이 모두 제공되지만 보상 신호는 알 수 없는 상황을 다룬다.

**차별점:** 본 논문은 환경으로부터 오는 비용 신호는 알 수 있지만, **안전 제약(최대 비용 임계값)은 주어지지 않는다**는 점이 핵심이다. 즉, 센서를 통해 현재 상태의 비용(예: 배터리 온도)은 측정 가능하지만, 어느 지점이 위험 수치인지(임계값)는 알 수 없으며, 이를 오직 전문가의 데이터를 통해서만 유추해야 한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 GAIL(Generative Adversarial Imitation Learning) 프레임워크를 기반으로 하며, 학습자의 점유 측정치(Occupancy measure, $\rho^\pi$)를 전문가의 점유 측정치($\rho^{\pi_E}$)에 맞추는 동시에 비용 제약을 만족시키는 것을 목표로 한다.

### 1. CCIL (Cost-Constrained Imitation Lagrangian)

CCIL은 다음과 같은 목적 함수를 최적화하여 saddle point $(\theta, \omega, \lambda)$를 찾는다.

$$L(\omega, \lambda, \theta) \triangleq \min_{\theta} \max_{\omega, \lambda} \mathbb{E}_{\pi_\theta}[\log D_\omega(s, a)] + \mathbb{E}_{\pi_E}[\log(1 - D_\omega(s, a))] + \lambda(\mathbb{E}_{\pi_\theta}[d(s, a)] - \mathbb{E}_{\pi_E}[d(s, a)]) - \beta H(\pi_\theta)$$

- **구성 요소:**
  - **Discriminator ($D_\omega$):** 전문가의 상태-행동 쌍과 학습자의 생성 데이터를 구별하며, 이 출력값이 학습자의 보상 신호로 사용된다.
  - **Causal Entropy ($H(\pi_\theta)$):** 정책의 엔트로피를 최대화하여 다양한 대안을 탐색하게 한다.
  - **Lagrangian Penalty ($\lambda$):** 학습자의 기대 비용이 전문가의 기대 비용을 초과할 때 페널티를 부여한다.

- **학습 절차:**
  - $\omega$ 업데이트: Adam Optimizer를 사용하여 Discriminator의 손실 함수를 최대화한다.
  - $\theta$ 업데이트: **TRPO (Trust Region Policy Optimization)**를 사용하여 정책을 업데이트한다. 이때 Surrogate advantage는 $\text{Advantage}(\text{Reward}) - \lambda \cdot \text{Advantage}(\text{Cost})$ 형태로 계산된다.
  - $\lambda$ 업데이트: $\nabla_\lambda L = (\mathbb{E}_{\pi_\theta}[d(s, a)] - \mathbb{E}_{\pi_E}[d(s, a)])$를 이용하여 비용 위반 정도에 따라 $\lambda$를 조정한다.

### 2. MALM (Meta-gradient for Lagrangian Approach)

MALM은 $\lambda$를 단순히 업데이트하는 대신, 검증 데이터셋(Validation set)을 이용한 **Outer loss**를 최소화하는 방향으로 $\lambda$를 조정한다.

$$\mathcal{L}_{\text{outer}}(\lambda) = \mathbb{E}_{\pi_\theta} [ (A_r(s, a) - \lambda d(s, a))^2 ]$$

이 방식은 보상 최대화와 비용 제약 준수 사이의 최적의 trade-off를 찾기 위해 메타-그래디언트를 사용하여 $\lambda$를 튜닝함으로써 CCIL보다 안정적인 성능을 제공한다.

### 3. CVAG (Cost-Violation-based Alternating Gradient)

CVAG는 복잡한 multiplier 없이 현재 상태의 **실행 가능성(Feasibility)**에 따라 목표를 전환한다.

- **Case 1 (비용 준수 시):** 학습자의 평균 비용이 전문가의 비용보다 낮거나 같으면, 보상(Return)을 최대화하는 방향으로 $\theta$를 업데이트한다.
- **Case 2 (비용 위반 시):** 학습자의 평균 비용이 전문가의 비용을 초과하면, 비용($d$)을 최소화하는 방향으로 $\theta$를 업데이트한다.

## 📊 Results

### 실험 설정

- **환경:** Safety Gym (Point, Car, Doggo / Goal, Button 작업) 및 MuJoCo (HalfCheetah, Hopper, Ant, Swimmer, Walker2d, Humanoid)
- **비교 대상 (Baselines):** GAIL, IQ-Learn, BC (비용 제약 무시), LGAIL (임계값 알고 있다고 가정)
- **측정 지표:**
  - **Normalized Penalized Return ($R^{pen}$):** 보상과 비용 위반 페널티의 trade-off를 측정 (높을수록 좋음).
  - **Recovered Return ($R^{rec}$):** 전문가의 보상을 얼마나 복원했는지 측정 (100에 가까울수록 좋음).
  - **Cost Violation ($\phi_d$):** 전문가의 비용을 얼마나 초과했는지 측정 (낮을수록 좋음).

### 주요 결과

- **MALM의 우수성:** Safety Gym의 대부분 작업과 MuJoCo의 HalfCheetah, Ant 작업에서 가장 높은 성능을 보였으며, 보상과 비용 준수 사이의 최적의 균형을 달성하였다.
- **CVAG의 특성:** 특히 속도 제한(Speed Limit)과 관련된 Walker2d, Swimmer 환경에서 강점을 보였다.
- **Baseline의 한계:**
  - BC와 IQ-Learn은 비용과 보상 모두에서 저조한 성능을 보였다.
  - GAIL은 보상은 높게 달성하지만, 비용 제약을 거의 지키지 못해 가장 높은 Cost violation을 기록하였다.
  - LGAIL은 일부 작업에서 우수했으나, 전반적으로 CCIL과 비슷하거나 더 많은 비용을 발생시켰다.

## 🧠 Insights & Discussion

본 논문은 비용 제약이 있는 환경에서 단순한 모방 학습이 위험할 수 있음을 정량적으로 입증하였다. 특히, 보상 함수를 모르는 상태에서 비용 제약만을 지켜야 하는 상황에서 **Meta-gradient를 통한 $\lambda$의 동적 조정(MALM)**이 매우 효과적임을 보여주었다.

**강점 및 한계:**

- **강점:** 다양한 환경(Safety Gym, MuJoCo)에서 제안한 방법론들이 기존 Baseline보다 보상-비용 trade-off를 잘 처리함을 입증하였다.
- **한계:** 전문가의 궤적이 충분히 제공되어야 하며, 전문가가 이미 최적의 비용-보상 균형을 찾았다는 가정 하에 작동한다. 만약 전문가의 데이터가 비효율적이라면 학습된 정책 역시 비효율적일 가능성이 크다.
- **비판적 해석:** CVAG의 경우 매우 단순한 스위칭 메커니즘을 사용함에도 불구하고 특정 환경(속도 제한 등)에서 효과적이었다는 점은, 모든 문제에 복잡한 Lagrangian 방식이 필요하지 않을 수 있음을 시사한다.

## 📌 TL;DR

이 논문은 **비용 임계값을 모르는 상태에서 전문가의 행동과 비용 제약을 동시에 모방**하는 새로운 프레임워크를 제안한다. Lagrangian 기반의 **CCIL**, 이를 메타-학습으로 최적화한 **MALM**, 그리고 실행 가능성에 따라 목표를 바꾸는 **CVAG** 세 가지 방법을 제시하였다. 실험 결과, 특히 **MALM**이 보상을 최대화하면서도 비용 제약을 엄격히 준수하는 데 가장 뛰어난 성능을 보였으며, 이는 실제 안전이 중요한 자율 시스템의 모방 학습에 중요한 기여를 할 것으로 평가된다.
