# ADAIL: Adaptive Adversarial Imitation Learning

Yiren Lu, Jonathan Tompson (2020)

## 🧩 Problem to Solve

본 논문은 단일 소스 도메인에서 수집된 소수의 전문가 시연(demonstrations)만을 이용하여, 환경의 역학(dynamics)이 서로 다른 다양한 환경에서도 동작할 수 있는 적응형 정책(adaptive policies)을 학습하는 문제를 다룬다.

로봇 학습 분야에서 이 문제는 매우 중요하다. 실제 환경에서는 다음과 같은 제약 사항이 존재하기 때문이다.

1. 보상 함수(reward functions)를 정확하게 정의하고 얻기가 매우 어렵다.
2. 소스 도메인과 타겟 도메인 간의 통계적 특성 차이로 인해, 한 환경에서 학습된 정책을 다른 환경에 그대로 배포하기 어렵다.
3. 역학이 완전히 제어되거나 알려진 여러 환경에서 전문가 시연을 각각 수집하는 것은 현실적으로 불가능에 가깝다.

따라서 본 연구의 목표는 보상 함수 없이, 단일 도메인의 시연 데이터만으로 다양한 역학 환경에 적응 가능한 정책을 학습하는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **역학 임베딩(dynamics embedding)에 기반한 정책 조건화**와 **역학 불변 판별자(dynamics-invariant discriminator)의 학습**이다.

- **적응형 정책 설계**: 정책 $\pi_\theta$가 현재 환경의 역학을 나타내는 잠재 변수 $c$에 조건화되도록 설계하여, 온라인 시스템 식별(online system ID)을 통해 환경 변화에 따라 다르게 행동할 수 있게 한다.
- **역학 불변 판별자**: 판별자가 단순히 환경의 역학적 특성만을 보고 전문가와 모방자를 구분하는 것을 방지하기 위해 Gradient Reversal Layer (GRL)를 도입한다. 이를 통해 판별자가 역학적 차이가 아닌 '행동의 특성'에 집중하게 하여 유의미한 보상 신호를 제공하도록 한다.

## 📎 Related Works

- **Behavioral Cloning (BC)**: 단순하지만 Covariate Shift로 인한 누적 오차 문제로 인해 방대한 양의 데이터나 전문가 정책에 대한 접근이 필요하다는 한계가 있다.
- **Inverse Reinforcement Learning (IRL) 및 GAIL**: 보상 함수를 추론하거나 판별자를 통해 전문가의 상태-행동 분포를 모방한다. 하지만 환경의 역학이 변할 경우, 판별자가 역학적 차이만으로 전문가 여부를 판단해버려 유효한 보상 신호를 주지 못하는 문제가 있다.
- **Dynamics Randomization**: 시뮬레이션에서 역학을 다양하게 변화시켜 강건한 정책을 학습한다. 그러나 도메인 시프트가 매우 클 경우 전문가의 행동 자체가 타겟 도메인에서 유효하지 않을 수 있으며, 본 논문과 달리 명시적인 역학 임베딩을 통한 적응 능력을 제공하지 않는 경우가 많다.
- **Meta Learning**: 다양한 작업에 대해 초기값을 학습하고 타겟 환경에서 미세 조정(fine-tuning)을 수행한다. 반면 ADAIL은 테스트 환경에서의 추가적인 미세 조정 없이도 적응이 가능하다.

## 🛠️ Methodology

### 전체 시스템 구조

ADAIL은 크게 세 가지 구성 요소로 이루어진다: **Dynamics Posterior $Q_\phi$**, **Policy $\pi_\theta$**, 그리고 **Discriminator $D_\omega$**.

1. **Dynamics Posterior**: 현재 궤적(trajectory) $\tau$를 입력받아 해당 환경의 역학 임베딩 $c$를 추론한다.
2. **Policy**: 상태 $s$와 추론된 역학 임베딩 $c$를 입력으로 받아 행동 $a$를 결정한다. 즉, $\pi_\theta(a|s, c)$ 형태로 동작한다.
3. **Discriminator**: 상태-행동 쌍 $(s, a)$를 입력받아 전문가의 행동인지 모방자의 행동인지 판별한다.

### 학습 목표 및 손실 함수

전체 학습 목적 함수는 다음과 같이 정의된다:
$$\min_{\theta} \max_{\omega, \phi} \mathbb{E}_{\pi_E}[\log D_\omega(s, a)] + \mathbb{E}_{\pi_\theta(\cdot|c)}[\log(1 - D_\omega(s, a))] + \mathbb{E}_{\tau \sim \pi_\theta(\cdot|c)}[\log Q_\phi(c|\tau)]$$

여기서 첫 번째와 두 번째 항은 GAIL의 표준 목적 함수로, 판별자 $D_\omega$는 전문가 $\pi_E$와 모방자 $\pi_\theta$를 구분하려 하고, 정책 $\pi_\theta$는 판별자를 속이려 한다. 세 번째 항은 역학 포스테리어 $Q_\phi$가 환경 파라미터 $c$를 정확히 추론하도록 학습하는 항이다.

### 역학 불변 판별자 (Dynamics-Invariant Discriminator)

판별자가 역학 정보를 이용해 정답을 맞히는 것을 막기 위해 Gradient Reversal Layer (GRL)를 사용한다. 판별자 네트워크에 역학 파라미터를 예측하는 보조 헤드 $D_R(c|s, a)$를 추가하고, 이 헤드와 공유하는 특징 추출 층 사이에 GRL을 배치한다. GRL은 역전파 시 그래디언트의 부호를 반전시켜, 특징 추출 층이 역학 정보를 최대한 제거한(역학 불변한) 특징만을 학습하도록 강제한다.

### 역학 임베딩 학습 방법

본 논문은 두 가지 임베딩 학습 방식을 제안한다.

1. **Direct Supervised Learning**: 정답 물리 파라미터 $c$가 있을 때, Huber Loss를 사용하여 $Q_\phi$가 이를 직접 회귀(regression)하도록 학습한다.
2. **VAE-based Unsupervised Learning**: 정답 라벨이 없는 경우, Conditional VAE 구조를 사용한다. 디코더를 Forward Dynamics 모델로 설계하여 $(s, a, c)$를 통해 $s'$를 예측하게 하며, 다음과 같은 ELBO를 최대화한다:
   $$\text{ELBO} = \mathbb{E}_{Q_\phi(c|s, a, s')}[\log P_\psi(s'|s, a, c)] - KL(Q_\phi(c|s, a, s') || P(c))$$
   또한, 인코더가 단순히 $s'$를 복제하는 identity mapping을 학습하는 것을 방지하기 위해 Contrastive Regularization loss $\mathcal{L}_{\text{contrastive}}$를 추가한다.

## 📊 Results

### 실험 설정

- **환경**: CartPole-V0, Hopper, HalfCheetah, Ant (MuJoCo 기반).
- **변동 파라미터**: 중력의 x성분($G_x$) 및 마찰력($F_r$) 등을 범위 내에서 균등하게 샘플링하여 변화시킴.
- **데이터**: 각 환경의 소스 도메인에서 16개의 전문가 시연 데이터만 사용.
- **비교 대상**: PPO Expert, UP-true (Yu et al., 2017), GAIL with Dynamics Randomization.

### 주요 결과

1. **역학 적응 능력**: CartPole 실험에서 GAIL-rand는 힘의 방향이 바뀌는 경우($F_m < 0$) 실패하였으나, ADAIL은 역학 임베딩을 통해 성공적으로 적응하였다.
2. **GRL의 효과**: Hopper 환경에서 GRL을 사용하지 않았을 때보다 사용했을 때 판별자가 더 유의미한 신호를 제공하여 최종 성능이 향상됨을 확인하였다.
3. **전반적 성능**: MuJoCo의 세 가지 환경(HalfCheetah, Ant, Hopper) 모두에서 ADAIL이 GAIL-rand 및 PPO Expert(unseen dynamics)보다 높은 누적 보상을 기록하였다. 특히 정답 파라미터를 사용한 경우(ADAIL-true)는 상한선인 UP-true에 근접하거나 일부 뛰어넘는 성능을 보였다.
4. **일반화 성능**: 학습 시 경험하지 못한 'Blackout' 영역(held-out parameters)에서도 정책은 어느 정도 작동하였으나, $Q_\phi$의 추론 오차(RMSE)가 증가함에 따라 실제 성능(ADAIL-pred)이 하락하는 경향을 보였다. 이는 정책의 성능이 역학 추론의 정확도에 의존함을 시사한다.

## 🧠 Insights & Discussion

### 강점

본 연구는 실제 로봇 학습 환경의 제약(단일 소스 데이터, 보상 함수 부재)을 정확히 반영한 문제 정의를 내렸으며, 이를 해결하기 위해 GRL과 역학 임베딩이라는 단순하지만 효과적인 구조를 제안하였다. 특히, 판별자가 환경의 특성(dynamics)과 행동의 특성(behavior)을 구분하도록 강제한 점이 핵심적인 기여이다.

### 한계 및 비판적 해석

1. **추론 오차에 대한 의존성**: 실험 결과에서 나타나듯, Dynamics Posterior $Q_\phi$의 예측 오차가 발생하면 정책의 성능이 직접적으로 하락한다. 즉, 시스템 식별(System ID)의 정확도가 전체 시스템의 병목 지점이 될 수 있다.
2. **잠재 공간의 복잡도**: VAE 기반의 비지도 학습 방식이 제시되었으나, 실제 복잡한 고차원 역학 환경에서 이러한 잠재 변수가 얼마나 유의미하게 학습될 수 있을지에 대한 심층적인 분석은 부족하다.
3. **가정**: 학습 단계에서 환경 생성기 $g(c)$를 통해 다양한 환경을 샘플링할 수 있다고 가정하는데, 이는 시뮬레이션 환경에서는 가능하나 실제 환경에서는 여전히 어려운 과제일 수 있다.

## 📌 TL;DR

ADAIL은 단일 소스 도메인의 소수 시연만으로 다양한 역학 환경에 적응하는 모방 학습 알고리즘이다. 역학 임베딩을 통해 정책을 조건화하고, GRL을 이용해 역학에 불변한 판별자를 학습함으로써 환경 변화에도 강건하게 전문가의 행동을 모방할 수 있게 한다. 이 연구는 보상 함수가 없고 데이터가 제한적인 실제 로봇 시스템의 Sim-to-Real 전이 및 적응형 제어 연구에 중요한 기여를 할 가능성이 높다.
