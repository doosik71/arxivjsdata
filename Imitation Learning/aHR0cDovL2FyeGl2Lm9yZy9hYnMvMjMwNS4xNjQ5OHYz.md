# Coherent Soft Imitation Learning

Joe Watson, Sandy H. Huang, and Nicolas Heess (2023)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)의 두 가지 주류 접근 방식인 행동 복제(Behavioral Cloning, BC)와 역강화학습(Inverse Reinforcement Learning, IRL) 사이의 간극을 메우고자 한다. 

BC는 구현이 간단하지만, 평가 단계에서 훈련 데이터와 다른 상태에 진입했을 때 오류가 누적되는 **Covariate Shift** 문제에 취약하다. 반면, IRL은 보상 함수를 추론하여 이 문제를 해결하지만, 보상과 정책을 동시에 학습하는 과정이 복잡하며 특히 고차원 제어 문제에서 saddle-point 최적화의 불안정성과 하이퍼파라미터 민감도 문제가 발생한다.

기존의 하이브리드 전략들은 BC로 사전 학습된 정책을 IRL로 미세 조정하려 했으나, 부정확한 초기 보상 함수로 인해 정책 최적화가 진행되면서 BC로 얻은 이점이 사라지는 'unlearning' 현상이 발생했다. 따라서 본 연구의 목표는 BC의 단순성과 IRL의 구조적 강점을 결합하여, BC 정책을 효율적으로 초기화하고 이를 환경과의 상호작용을 통해 안정적으로 개선할 수 있는 **Coherent**한 모방 학습 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 엔트로피 정규화(Entropy-regularized) 강화학습 설정에서 정책 업데이트 식을 역전시켜, **BC 정책이 최적해가 되는 형태의 보상 함수(Shaped Reward)를 유도**하는 것이다. 이를 통해 다음과 같은 기여를 한다.

1. **Coherent Reward의 유도**: BC 정책을 통해 해당 정책이 최적이 되도록 하는 보상 함수를 정의함으로써, RL 미세 조정 단계에서 초기 성능을 유지하며 정책을 개선할 수 있는 '일관성(Coherence)'을 확보하였다.
2. **HetStat 아키텍처 제안**: 연속 제어 환경에서 정책이 데이터 분포 밖(Out-of-Distribution)에서 사전 확률 분포(Prior)로 돌아가도록 하기 위해, 정지 과정(Stationary Process) 이론을 적용한 신경망 구조인 `HetStat`을 도입하였다.
3. **안정적인 스케일링**: Adversarial 방식의 불안정성을 피하고, 고차원 및 이미지 기반 작업에서도 하이퍼파라미터 튜닝을 최소화하며 안정적인 성능 향상을 입증하였다.

## 📎 Related Works

기존 모방 학습 연구들은 크게 다음과 같이 분류된다.

- **Adversarial IL (GAIL, AIRL, DAC 등)**: 판별기(Discriminator)를 통해 보상을 학습하며 $\min_{\pi} \max_{r}$ 형태의 게임 이론적 목적 함수를 사용한다. 그러나 고차원 상태 공간에서 최적화가 불안정하고 하이퍼파라미터에 매우 민감하다.
- **Implicit IRL (IQ-Learn, PPIL 등)**: 명시적인 보상 함수 없이 가치 함수(Value Function)를 통해 직접 정책을 학습하려 한다. 하지만 여전히 saddle-point 최적화의 특성을 가지며, BC 사전 학습과 결합했을 때 초기 성능을 빠르게 잃어버리는 경향이 있다.
- **Proxy Reward (SQIL 등)**: 전문가 데이터에 대해 단순한 이진 보상을 부여하는 방식이다. 구현은 쉽지만 이론적 보장이 부족하고 수렴 속도가 느리다.

본 논문의 **CSIL**은 이러한 방식들과 달리, BC 정책으로부터 유도된 **Log-policy ratio**를 보상으로 사용함으로써, 정책과 보상 사이의 일관성을 보장하고 BC의 초기 이점을 보존하며 RL로의 전이를 가능하게 한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. Coherent Reward의 유도 (Policy Inversion)

본 논문은 엔트로피 정규화된 정책 업데이트 식을 역전시켜, 특정 정책 $\pi$가 최적이 되게 하는 보상 함수 $r$을 찾는다. **Theorem 1**에 따르면, 사전 정책 $p$와 사후 정책 $q_\alpha$가 주어졌을 때, 다음과 같이 Critic $Q$와 보상 $r$을 표현할 수 있다.

$$Q(s, a) = \alpha \log \frac{q_\alpha(a|s)}{p(a|s)} + V_\alpha(s)$$
$$r(s, a) = \alpha \log \frac{q_\alpha(a|s)}{p(a|s)} + V_\alpha(s) - \gamma \mathbb{E}_{s' \sim P(\cdot|s, a)} [V_\alpha(s')]$$

여기서 $\alpha$는 온도 파라미터이며, $V_\alpha(s)$는 soft value function이다. 특히, $V_\alpha(s)$를 잠재 함수(potential function)로 사용하는 보상 shaping 이론(Ng et al., 1999)에 따라, 다음과 같은 **Coherent Reward** $\tilde{r}$을 정의할 수 있다.

$$\tilde{r}(s, a) = \alpha \log \frac{q_\alpha(a|s)}{p(a|s)}$$

이 보상은 전문가 데이터 영역에서는 양수, 데이터 밖에서는 음수 또는 0의 값을 가지며, 정책이 다시 전문가 분포로 돌아오도록 유도하는 역할을 한다.

### 2. Stationary Processes for Continuous Control (HetStat)

연속 제어에서 $\tilde{r}$이 효과적으로 작동하려면, 정책 $q_\theta$가 데이터가 없는 영역에서 사전 분포 $p$와 일치해야 한다. 일반적인 MLP는 외삽(extrapolation) 문제로 인해 OOD 영역에서 예측값이 발산하는 경향이 있다. 이를 해결하기 위해 본 논문은 **Stationary Process** 이론을 도입한 `HetStat` 구조를 제안한다.

- **구조**: MLP의 마지막 레이어에 주기적 활성화 함수(Periodic activation function, 예: $\sin$)를 적용하여 가우시안 프로세스(Gaussian Process)를 근사한다.
- **효과**: 이를 통해 모델이 학습 데이터가 없는 영역에서 자연스럽게 사전 분포(Prior)로 회귀하게 만들며, 결과적으로 $\tilde{r}$이 정의된 대로 shaping 효과를 내도록 보장한다.

### 3. 전체 학습 절차 (CSIL Algorithm)

CSIL의 전체 파이프라인은 다음과 같이 진행된다.

1. **Behavioral Cloning (BC)**: 전문가 데이터 $\mathcal{D}$를 사용하여 정규화된 BC를 수행하고 초기 정책 $q_{\theta_1}$을 학습한다.
2. **Coherent Reward 정의**: 학습된 $q_{\theta_1}$과 사전 분포 $p$를 이용하여 고정된 보상 함수 $\tilde{r}_{\theta_1}(s, a) = \alpha (\log q_{\theta_1}(a|s) - \log p(a|s))$를 생성한다.
3. **RL Refinement**: 생성된 $\tilde{r}_{\theta_1}$을 보상으로 사용하여 Soft Actor-Critic(SAC)과 같은 Soft Policy Iteration 알고리즘을 통해 정책을 미세 조정한다. 이때 미세 조정 온도 $\beta$는 초기 온도 $\alpha$보다 작게 설정($\beta < \alpha$)하여 정책이 개선될 여지를 둔다.

### 4. 추가적인 안정화 기법

- **Critic Jacobian Regularization**: Critic $Q$가 전문가 행동 근처에서 1차 최적성(first-order optimality)을 갖도록, 행동에 대한 Jacobian의 노름을 최소화하는 보조 손실 함수를 추가한다.
  $$\min_\phi \mathbb{E}_{s, a \sim \mathcal{D}} [\|\nabla_a Q_\phi(s, a)\|^2]$$
- **Reward Refinement**: 학습 과정에서 발생하는 근사 오류를 줄이기 위해, replay buffer의 데이터를 사용하여 보상 함수를 minimax 방식으로 미세 조정한다.
- **Faithful Heteroscedastic Regression**: 데이터의 신호를 노이즈로 오인하는 문제를 방지하기 위해, 평균 제곱 오차(MSE)와 음의 로그 가능도(NLL)를 결합한 'faithful' 손실 함수를 사용하여 BC를 수행한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 작업**: MuJoCo Gym (Locomotion), Adroit (Dexterous Manipulation), Robomimic (Robot Manipulation)
- **비교 대상**: BC, SQIL, DAC, IQ-Learn, PPIL, PWIL 등
- **측정 지표**: 정규화된 Return (Expert=1, Random=0), 성공률(Success Rate)

### 2. 주요 결과
- **Online Imitation (Gym & Adroit)**: CSIL은 거의 모든 환경에서 최신 SOTA 방법론들과 대등하거나 더 높은 성능을 보였다. 특히 고차원 제어 작업인 Adroit에서 saddle-point 최적화 기반 방법론들이 불안정하게 무너지는 반면, CSIL은 매우 안정적으로 수렴하며 BC 성능을 상회하였다.
- **Offline Imitation (Gym)**: 오프라인 설정은 온라인보다 훨씬 어려웠으나, 전문가 데이터의 양이 증가함에 따라 CSIL의 성능이 꾸준히 향상되는 양상을 보였다.
- **Image-based Tasks (Robomimic)**: 이미지 관측값을 사용하는 환경에서도 확장 가능함을 보였으며, 특히 `NutAssemblySquare` 작업에서 기존 BC 모델보다 높은 성공률을 기록하였다.
- **Tabular MDP**: 'Windy' 환경(외란이 존재하는 환경) 실험을 통해, CSIL이 단순 BC보다 훨씬 강건하며 GAIL과 유사한 가치 함수를 생성함을 확인하였다.

## 🧠 Insights & Discussion

### 강점
- **안정성**: Adversarial 방식의 불안정성과 하이퍼파라미터 민감도 문제를 해결하였다. BC로 시작하여 RL로 끝내는 구조 덕분에 학습 초기 단계부터 높은 성능을 보장한다.
- **일관성(Coherence)**: 정책 인버전을 통해 유도된 보상을 사용함으로써, RL 미세 조정 단계에서 BC의 성과를 'unlearning' 하지 않고 오히려 개선하는 구조를 구축하였다.
- **확장성**: `HetStat` 구조와 Coherent Reward의 결합은 고차원 상태-행동 공간과 이미지 입력 환경에서도 효과적으로 작동함을 입증하였다.

### 한계 및 논의사항
- **BC 의존성**: CSIL의 성능은 초기 BC 정책의 품질에 크게 의존한다. 실험 결과, 데이터가 매우 부족하거나 모델이 너무 복잡하여 BC 자체가 전혀 작동하지 않는 경우, CSIL 역시 해당 작업을 해결하지 못하는 경향이 있었다.
- **보상 함수의 성격**: 본 방법론은 '진정한 보상(True Reward)'을 찾는 것이 아니라, 전문가 정책이 최적이 되도록 'shaping된 보상'을 찾는 것이다. 실무적으로는 충분하지만, 이론적인 보상 추론 관점에서는 한계가 있을 수 있다.
- **향후 연구**: MLP를 넘어 RNN과 같은 순환 신경망이나 멀티모달 행동 분포를 가진 정책 클래스에 CSIL을 적용하여 하위 최적(sub-optimal) 인간 데이터에 대한 처리 능력을 높일 필요가 있다.

## 📌 TL;DR

본 논문은 BC의 효율성과 IRL의 강건함을 결합한 **Coherent Soft Imitation Learning (CSIL)**을 제안한다. 핵심은 BC 정책을 역전시켜 해당 정책이 최적이 되는 **Coherent Reward**를 유도하고, 이를 통해 BC 이후의 RL 미세 조정 과정에서 성능 저하(unlearning) 없이 정책을 개선하는 것이다. 또한, OOD 영역에서의 안정성을 위해 **정지 과정(Stationary Process)** 기반의 `HetStat` 아키텍처를 도입하였다. 실험 결과, CSIL은 고차원 제어 및 이미지 기반 작업에서 기존 Adversarial/Implicit IL 방법론보다 훨씬 안정적이고 샘플 효율적인 성능을 보였으며, 이는 향후 복잡한 로봇 제어 시스템의 모방 학습에 중요한 기여를 할 것으로 기대된다.