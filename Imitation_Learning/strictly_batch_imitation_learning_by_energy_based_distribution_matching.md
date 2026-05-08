# Strictly Batch Imitation Learning by Energy-based Distribution Matching

Daniel Jarrett, Ioana Bica, Mihaela van der Schaar (2020)

## 🧩 Problem to Solve

본 논문은 **Strictly Batch Imitation Learning (SBIL)** 문제를 해결하고자 한다. 일반적인 Imitation Learning(모방 학습)은 강화 학습 신호(reinforcement signals)에 접근하거나, 환경의 전이 역학(transition dynamics)을 알고 있거나, 혹은 환경과 직접 상호작용하며 정책을 개선하는 과정을 거친다. 하지만 의료 서비스(healthcare)와 같이 실제 환경에서의 실험 비용이 매우 높거나 위험한 분야에서는 이러한 조건들이 충족되기 어렵다.

따라서 본 연구는 다음과 같은 제약 조건 하에서의 학습을 목표로 한다:

1. 강화 학습 신호(보상 함수)에 접근할 수 없다.
2. 환경의 전이 역학(dynamics)에 대한 지식이 없다.
3. 학습 과정 중 환경과의 추가적인 상호작용(online interaction)이 완전히 배제된다.

기존의 Behavioral Cloning(BC)은 이러한 오프라인 설정에서 작동하지만, 전문가가 방문한 상태 분포(state distribution) 정보를 활용하지 못해 성능이 제한적이다. 반면, 기존의 Apprenticeship Learning이나 Adversarial IL 방식들은 본질적으로 온라인 환경을 가정하므로, 이를 오프라인으로 적용하려면 부정확한 Off-policy evaluation이나 모델 추정에 의존해야 하는 비효율성이 존재한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Energy-based Distribution Matching (EDM)** 기술을 통해 정책의 파라미터와 상태 분포의 에너지 함수를 동일한 파라미터로 공유하여 공동 학습(Joint Learning)하는 것이다.

가장 중심적인 직관은 **정책의 판별적(discriminative) 모델(액션 조건부 확률)을 상태 분포의 생성적(generative) 에너지 함수와 동일하게 파라미터화**함으로써, 별도의 환경 상호작용 없이도 전문가의 Occupancy Measure(방문 분포)와 모방자의 방문 분포 사이의 Divergence를 최소화할 수 있다는 점이다. 이를 통해 정책 학습과 동시에 상태 분포 모델을 학습함으로써, 데이터가 부족한 상황에서도 더 일반화된 성능을 낼 수 있게 한다.

## 📎 Related Works

논문에서는 기존의 접근 방식을 다음과 같이 분류하고 한계를 지적한다.

1. **Behavioral Cloning (BC):** 본질적으로 오프라인 방식이지만, 롤아웃 분포(rollout distribution)의 내생성을 무시하고 단순 지도 학습으로 처리하므로, 상태 방문 분포의 귀중한 정보를 활용하지 못한다.
2. **Classic Inverse Reinforcement Learning (IRL):** 보상 함수 $R$을 먼저 추정하고 이를 통해 정책을 학습하는 간접적인 방식을 취한다. 오프라인 설정에서 적용하려면 매 단계마다 Off-policy evaluation이 필요하며, 이는 분산이 크고 계산 비용이 높다.
3. **Adversarial Imitation Learning (AIL):** 생성적 적대 신경망(GAN)과 유사하게 Discriminator를 통해 방문 분포를 맞추려 한다. 하지만 이는 본질적으로 모방자의 정책으로 생성한 궤적(trajectories) 샘플링이 필요하므로, 완전한 오프라인 설정에서는 적용이 어렵거나 복잡한 Max-min 최적화 과정에서 편향된 그래디언트 문제가 발생한다.

EDM은 이러한 방식들과 달리, 정책 파라미터 $\theta$를 통해 정책과 상태 분포 모델을 동시에 최적화함으로써, 환경 상호작용 없이도 분포 매칭 효과를 얻는다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파라미터화

먼저 정책 $\pi_\theta(a|s)$를 다음과 같은 Softmax 형태로 정의한다:
$$\pi_\theta(a|s) = \frac{e^{f_\theta(s)[a]}}{\sum_{a'} e^{f_\theta(s)[a']}}$$
여기서 $f_\theta(s)[a]$는 액션 $a$에 대한 로짓(logits)이다.

### 에너지 기반 모델 (EBM)의 도입

본 논문은 정책의 로짓 $f_\theta$를 사용하여 상태 분포 $\rho_\theta(s)$에 대한 에너지 함수 $E_\theta(s)$를 다음과 같이 정의한다:
$$E_\theta(s) = -\log \sum_a e^{f_\theta(s)[a]}$$
즉, 정책의 파라미터 $\theta$가 결정되면 자동으로 상태 분포의 에너지 함수가 결정되는 구조이다.

### 학습 목표 및 손실 함수 (Surrogate Objective)

최종 목표는 전문가의 방문 분포 $\rho^D$와 모방자의 분포 $\rho^\theta$ 사이의 KL Divergence를 최소화하는 것이다. 하지만 $\rho^\theta(s)$를 직접 계산하는 것은 불가능하므로, 다음과 같은 **Surrogate Objective** $L_{surr}(\theta)$를 제안한다:
$$L_{surr}(\theta) = L_\rho(\theta) + L_\pi(\theta)$$

1. **정책 손실 ($L_\pi$):** 표준적인 Behavioral Cloning 손실함수(Cross-Entropy)이다.
   $$L_\pi(\theta) = -\mathbb{E}_{s,a \sim \rho^D} \log \pi_\theta(a|s)$$
2. **방문 분포 손실 ($L_\rho$):** 전문가 데이터의 에너지와 모델이 생성한 샘플의 에너지 차이를 최소화한다.
   $$L_\rho(\theta) = \mathbb{E}_{s \sim \rho^D} E_\theta(s) - \mathbb{E}_{s \sim \rho^\theta} E_\theta(s)$$

### 학습 절차 및 최적화

$\mathbb{E}_{s \sim \rho^\theta} E_\theta(s)$ 항을 계산하기 위해, 모델 $\rho^\theta$로부터 상태 샘플을 생성해야 한다. 이를 위해 다음과 같은 절차를 사용한다:

- **SGLD (Stochastic Gradient Langevin Dynamics):** 에너지 함수의 그래디언트를 따라 상태 공간에서 샘플링을 수행한다.
  $$\tilde{s}_{i} = \tilde{s}_{i-1} - \alpha \frac{\partial E_\theta(\tilde{s}_{i-1})}{\partial \tilde{s}_{i-1}} + \sigma \mathcal{N}(0, I)$$
- **PCD (Persistent Contrastive Divergence):** 매번 처음부터 샘플링하는 대신, 샘플 버퍼(Buffer)를 유지하여 계산 효율성을 높인다.

최종적으로 $\nabla_\theta L_\rho + \nabla_\theta L_\pi$를 통해 네트워크 파라미터 $\theta$를 업데이트한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업:**
  - **Control Tasks:** OpenAI Gym의 CartPole, Acrobot, LunarLander, BeamRider.
  - **Healthcare:** MIMIC-III (중환자실 환자의 치료 궤적 데이터).
- **비교 대상:** Behavioral Cloning (BC), Reward-Regularized Classification (RCAL), Deep Successor Feature Network (DSFN), Value-Dice (VDICE).
- **측정 지표:** Gym 환경에서는 평균 리턴(Average Returns), MIMIC-III에서는 액션 매칭 정확도(ACC), AUC, APR을 사용한다.

### 주요 결과

1. **정량적 성능:** EDM은 모든 환경에서 BC 및 다른 오프라인 적응 알고리즘들보다 일관되게 우수한 성능을 보였다.
2. **데이터 효율성:** 특히 전문가의 궤적(trajectories) 수가 매우 적은 **Low-data regime**에서 EDM의 성능 향상 폭이 가장 컸다. 이는 EBM이 희소한 데이터 $\rho^D$를 매끄러운(smoothed) 분포 모델 $\rho^\theta$로 대체하여 학습함으로써 발생하는 효과이다.
3. **의료 데이터 적용:** MIMIC-III 데이터셋에서도 ACC, AUC, APR 모든 지표에서 타 모델 대비 우위를 점하며 실제 의료 환경에서의 적용 가능성을 입증하였다.
4. **반-지도 학습 (Semi-supervised):** 액션 정보가 없는 상태 전용(state-only) 데이터만 추가로 제공했을 때도 성능이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

EDM의 가장 큰 강점은 정책 학습과 상태 분포 학습을 단일 파라미터 $\theta$로 묶어 **멀티태스크 학습(Multitask Learning)** 형태로 구현했다는 점이다. 이를 통해 별도의 보상 함수 추정이나 환경 상호작용 없이도, 정책이 방문하게 될 상태 분포를 간접적으로 제어할 수 있게 되었다. 특히, BC가 단순히 전문가의 행동을 복제하는 것에 그치는 반면, EDM은 "어떤 상태에 머물러야 하는가"에 대한 생성적 정보를 함께 학습함으로써 분포 불일치(distribution mismatch) 문제를 완화한다.

### 한계 및 비판적 논의

- **액션 공간의 제한:** 본 연구에서는 Joint EBM 구조를 사용하므로 액션이 범주형(categorical)인 경우에만 적용 가능하다. 연속적인 액션 공간으로 확장하기 위해서는 EBM의 회귀(regression) 적용 방식에 대한 추가 연구가 필요하다.
- **EBM 학습의 불안정성:** 일반적으로 에너지 기반 모델은 학습이 어렵고 수렴 여부를 판단하기 어렵다. 저자들은 본 실험 환경에서는 큰 문제가 없었다고 주장하지만, 더 고차원적인 상태 공간에서는 SGLD 샘플링의 효율성과 안정성이 문제가 될 가능성이 크다.
- **데이터 대표성 가정:** $\rho^D$가 실제 전문가 분포를 충분히 대표한다는 가정하에 작동하므로, 데이터 자체가 심하게 편향되어 있을 경우 모델이 잘못된 상태 분포를 학습할 위험이 있다.

## 📌 TL;DR

본 논문은 환경 상호작용과 전이 역학 지식이 전혀 없는 **Strictly Batch Imitation Learning** 상황에서, 정책 파라미터와 상태 분포의 에너지 함수를 공유하여 공동 학습하는 **EDM(Energy-based Distribution Matching)** 방법을 제안한다. 이 방법은 단순한 행동 복제(BC)를 넘어 상태 방문 분포를 모델링함으로써, 특히 데이터가 부족한 상황에서 매우 강력한 성능을 보이며 의료 데이터와 같은 실전 오프라인 환경에 적용 가능성이 높다.
