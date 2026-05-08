# Implicit Actor-Critic Coupling via a Supervised Learning Framework for RLVR

Jiaming Li, Longze Chen, Ze Gong, Yukun Chen, Lu Wang, Wanwei He, Run Luo, Min Yang (2025)

## 🧩 Problem to Solve

본 논문은 검증 가능한 보상(Verifiable Rewards)을 이용한 강화학습(RLVR, Reinforcement Learning with Verifiable Rewards) 환경에서 대규모 언어 모델(LLM)의 추론 능력을 향상시킬 때 발생하는 고질적인 문제들을 해결하고자 한다.

RLVR의 핵심은 수학 문제 풀이나 프로그래밍과 같이 정답 여부를 명확히 검증할 수 있는 영역에서 결과 보상(Outcome Reward)을 통해 정책을 최적화하는 것이다. 그러나 기존의 RL 기반 접근 방식은 다음과 같은 치명적인 한계가 있다. 첫째, 전체 응답이 생성된 후에만 단 한 번의 보상이 주어지는 Sparse Reward 문제로 인해 개별 토큰 수준의 Credit Assignment가 어렵다. 둘째, PPO와 같은 Value-model-based 방법은 가치 모델을 별도로 학습시켜야 하므로 연산 복잡도와 메모리 오버헤드가 크다. 셋째, GRPO와 같은 Value-model-free 방법은 Monte Carlo 추정치에 의존하여 분산(Variance)이 매우 높으며, 이로 인해 Advantage Collapse나 학습 불안정성, 성능 저하가 빈번하게 발생한다.

따라서 본 연구의 목표는 RL의 불안정성과 복잡성을 제거하고, 직접적인 감독 신호(Direct Supervision)를 활용하여 더 안정적이고 효율적으로 정책을 최적화할 수 있는 새로운 학습 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **RLVR 문제를 지도학습(Supervised Learning) 과제로 재구성**하는 것이다.

기존처럼 보상을 사용하여 정책 경사(Policy Gradient)를 업데이트하는 대신, 결과 보상 $R(q,o)$ 자체를 예측 가능한 라벨(Label)로 취급한다. 정책 모델이 생성한 응답의 품질을 예측하는 Score Function $\psi$를 학습시키고, 이를 Binary Cross-Entropy (BCE) 손실 함수로 최적화함으로써 RL 문제를 SL 문제로 변환한다.

이 설계의 중심적인 직관은 이러한 지도학습 형태의 목적 함수가 수학적으로는 클래식한 Policy Gradient 업데이트를 복원함과 동시에, Actor(정책 개선)와 Critic(보상 추정)의 역할을 단일 모델 내에서 암시적으로 결합(Implicit Coupling)한다는 점이다. 이를 통해 별도의 가치 모델 없이도 안정적인 보상 추정과 정책 업데이트를 동시에 달성할 수 있다.

## 📎 Related Works

논문에서는 LLM의 추론 능력을 극대화하기 위한 Reasoning Models(OpenAI-o1, DeepSeek-R1 등)와 RLVR 기술들을 소개한다.

기존의 RLVR 접근 방식은 크게 두 가지로 나뉜다.

1. **Value-model-based (예: PPO, VAPO):** 명시적인 가치 함수를 학습하여 Advantage를 추정한다. 안정적이지만 모델 복잡도가 높고 연산 비용이 크다는 한계가 있다.
2. **Value-model-free (예: GRPO, DAPO):** 그룹 내 상대적 보상을 이용한 Monte Carlo 추정 방식을 사용한다. 구조는 단순하지만 Gradient Variance가 매우 높아 학습이 불안정하다는 단점이 있다.

PACS는 이러한 두 방식의 절충안을 넘어, 보상 예측이라는 지도학습 구조를 통해 분산을 낮추면서도 가치 모델의 오버헤드를 제거한 차별화된 접근 방식을 취한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 목적 함수

PACS는 쿼리 $q$와 그에 대한 응답 $o$를 입력으로 하고, 최종 결과 보상 $R(q,o) \in \{0, 1\}$를 라벨로 하는 이진 분류 문제로 RLVR을 정의한다. 최적화하고자 하는 목적 함수는 다음과 같은 Binary Cross-Entropy 손실 함수이다.

$$L(\theta) = -\mathbb{E}_{q \sim P(Q), o \sim \pi_\theta(\cdot|q)} \left[ R(q,o) \log(\sigma(\psi(q,o; \pi_\theta))) + (1-R(q,o)) \log(1-\sigma(\psi(q,o; \pi_\theta))) \right]$$

여기서 $\sigma$는 Sigmoid 함수이며, $\psi(q,o; \pi_\theta)$는 정책 모델 $\pi_\theta$에 의해 매개변수화된 Score Function이다.

### 2. Gradient Analysis 및 Implicit Coupling

본 논문은 위 손실 함수의 기울기(Gradient)를 분석하여 이 구조가 어떻게 Actor-Critic 역할을 동시에 수행하는지 증명한다. 손실 함수의 기울기는 다음과 같이 분해된다.

$$\nabla_\theta L(\theta) = -\mathbb{E}_{q, o \sim \pi_\theta} \left[ \underbrace{l(q,o; \pi_\theta) \nabla_\theta \log \pi_\theta(o|q)}_{\text{ACTOR: policy improvement}} + \underbrace{(R(q,o) - \sigma(\psi(q,o; \pi_\theta))) \nabla_\theta \psi(q,o; \pi_\theta)}_{\text{CRITIC: reward estimation}} \right]$$

- **Actor 역할:** 첫 번째 항은 표준적인 Policy Gradient 업데이트 형태를 띠며, 예측 보상과 실제 보상의 정렬 정도($l$)에 따라 업데이트 강도가 조절된다.
- **Critic 역할:** 두 번째 항은 예측 보상 $\sigma(\psi)$와 실제 보상 $R$ 사이의 오차(Prediction Error)를 줄이는 방향으로 $\psi$를 학습시킨다.

이처럼 하나의 모델이 $\pi_\theta(o|q)$를 통해 샘플링(Actor)하고, $\psi$를 통해 품질을 평가(Critic)함으로써 두 역할이 암시적으로 결합되어 학습 효율성이 극대화된다.

### 3. Score Function $\psi$의 구현

$\psi$를 구체적으로 구현하기 위해, 본 논문은 **RLOO (REINFORCE Leave-One-Out)** 추정기를 도입하여 그룹 내 상대적 이점을 측정한다.

먼저, 로그 확률 비율에 기반한 Reward Proxy $\hat{r}$을 다음과 같이 정의한다.
$$\hat{r}(q,o_i; \pi_\theta) = \beta \sum_{t=1}^{|o_i|} \left( \log \pi_\theta(o_{i,t} | q, o_{i,<t}) - \log \pi_{ref}(o_{i,t} | q, o_{i,<t}) \right)$$

이후, $G$개의 샘플 중 하나를 제외한 나머지 샘플들의 평균과 비교하는 방식으로 Score Function $\psi$를 계산한다.
$$\psi(q,o_i; \pi_\theta) = \hat{r}(q,o_i; \pi_\theta) - \frac{1}{G-1} \sum_{j \neq i} \hat{r}(q,o_j; \pi_\theta)$$

또한, 학습 안정성을 위해 참조 정책 $\pi_{ref}$를 주기적으로 현재 정책 $\pi_\theta$의 스냅샷으로 업데이트하는 Hard-reset 전략을 사용한다.

### 4. 데이터 불균형 처리

정답 샘플과 오답 샘플의 비율이 극도로 불균형할 경우 모델이 한쪽으로 편향될 수 있다. 이를 해결하기 위해 correct 샘플과 incorrect 샘플에 서로 다른 가중치를 부여하는 Class Imbalance Treatment 방식을 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** DeepScaleR (약 40,000개의 수학 문제-답변 쌍)
- **평가 벤치마크:** MATH 500, AMC 23, AIME 2024, AIME 2025
- **사용 모델:** Qwen2.5-3B, Qwen2.5-7B
- **비교 대상:** PPO, GRPO 및 Base 모델
- **측정 지표:** $\text{pass@k}$ (k개의 샘플 중 적어도 하나가 정답일 확률)

### 주요 결과

PACS는 모든 벤치마크에서 PPO와 GRPO를 유의미하게 상회하는 성능을 보였다. 특히 난이도가 매우 높은 AIME 시리즈에서 그 효과가 극명하게 나타났다.

- **AIME 2025 (Qwen2.5-7B):** $\text{pass@256}$ 기준, PACS는 **58.22%**를 달성하여 PPO와 GRPO 대비 각각 **15.31%p, 11.77%p** 높은 성능 향상을 기록하였다.
- **AIME 2024 (Qwen2.5-7B):** $\text{pass@256}$ 기준 **59.78%**를 기록하며, PPO(46.46%)와 GRPO(45.42%)를 크게 앞질렀다.
- **MATH 500 및 AMC 23:** 모든 $k$ 값 설정에서 Base 모델 및 RL 베이스라인보다 우수한 성능을 유지하였다.

### Ablation Study

- **$\beta$ 값의 영향:** $\beta=1$일 때 최적의 성능을 보였으며, 특히 AIME와 같은 고난도 과제일수록 $\beta$ 값에 대한 민감도가 높게 나타났다.
- **Advantage Estimator 비교:** RLOO 기반의 PACS가 GRPO나 Dr. GRPO 방식보다 특히 고난도 과제(AIME)에서 더 정밀한 Credit Assignment를 수행하여 월등한 성능을 보였다.
- **가중치 적용 유무:** 가중치 메커니즘을 제거했을 때(w/o weight) 성능이 하락함을 확인하여, 데이터 불균형 처리가 필수적임을 입증하였다.

## 🧠 Insights & Discussion

### 학습 역학(Training Dynamics) 분석

본 논문은 PPO/GRPO와 PACS의 학습 과정을 비교하여 다음과 같은 통찰을 제시한다.

1. **Entropy 유지:** PPO와 GRPO는 학습 초기에 Entropy가 급격히 감소하는 Entropy Collapse 현상을 보이며 보수적인 출력을 생성하는 경향이 있다. 반면, PACS는 Entropy가 일정 수준 유지되거나 비단조적으로 변화하며 더 활발한 탐색(Exploration)을 수행한다.
2. **Gradient Norm:** PACS는 다른 방법론들에 비해 지속적으로 높은 Gradient Norm을 유지한다. 이는 모델이 학습 후반부까지도 유의미한 파라미터 업데이트를 지속하며 더 효율적으로 최적화됨을 의미한다.
3. **응답 길이:** PACS는 더 상세하고 긴 추론 과정(Chain-of-Thought)을 생성하는 경향을 보였으며, 이는 향상된 성능과 밀접한 관련이 있는 것으로 분석된다.

### 비판적 해석

PACS는 RL의 복잡한 메커니즘을 SL의 단순한 구조로 치환하여 안정성을 확보했다는 점에서 매우 영리한 접근이다. 특히 $\psi$ 함수를 통해 Actor와 Critic을 통합함으로써 연산 효율성을 잡은 점이 돋보인다. 다만, 본 연구는 수학적 추론이라는 Verifiable Reward가 명확한 도메인에 집중되어 있어, 보상 함수가 모호한 일반적인 텍스트 생성 작업으로의 확장 가능성에 대해서는 추가적인 검증이 필요할 것으로 보인다.

## 📌 TL;DR

PACS는 RLVR 문제를 **지도학습(Supervised Learning)의 이진 분류 문제로 재구성**하여, RL 특유의 Sparse Reward 문제와 학습 불안정성을 해결한 프레임워크이다. 이 방법은 단일 모델 내에서 **Actor와 Critic의 역할을 암시적으로 결합**하여 연산 효율성을 높이면서도, RLOO 기반의 Score Function을 통해 안정적인 정책 업데이트를 가능케 한다. 실험 결과, Qwen2.5 모델을 사용하여 AIME 2025 등 고난도 수학 벤치마크에서 PPO와 GRPO를 압도하는 성능을 보여주었으며, 이는 LLM의 추론 능력 향상을 위한 포스트 트레이닝의 새로운 효율적 방향성을 제시한다.
