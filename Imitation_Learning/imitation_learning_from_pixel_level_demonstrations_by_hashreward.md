# Imitation Learning from Pixel-Level Demonstrations by HashReward

Xin-Qiang Cai, Yao-Xiang Ding, Yuan Jiang, Zhi-Hua Zhou (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 고차원 상태 공간(High-dimensional state environments), 특히 가공되지 않은 픽셀(raw pixels) 입력을 사용하는 환경에서 Imitation Learning(IL)의 일반화 성능이 급격히 저하되는 현상이다.

일반적으로 Adversary-based IL 알고리즘은 전문가의 시연(demonstrations)으로부터 보상 함수를 학습하여 정책을 최적화한다. 하지만 고차원 환경에서는 다음과 같은 문제가 발생한다.

- **제한된 샘플 수와 차원의 저주**: 전문가의 데이터는 한정되어 있는 반면 상태 공간은 매우 넓어, Discriminator가 학습 데이터에 과적합(overfitting)되기 쉽다.
- **Discrimination-Rewarding Trade-off**: Discriminator가 너무 강력해지면, 전문가의 데이터와 조금만 달라도 매우 낮은 보상을 주게 된다. 이로 인해 Learner는 유의미한 보상 신호를 받지 못하고 학습에 실패하게 된다.

따라서 본 논문의 목표는 고차원 환경에서도 Discriminator가 효과적인 보상 신호를 생성할 수 있도록, 차원 축소(Dimensionality Reduction, DR)와 판별 학습(Discriminative training) 사이의 균형을 맞추는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Supervised Hashing**을 도입하여 차원 축소 과정과 Discriminator 학습을 하나의 통합된 절차로 수행하는 것이다.

기존의 Unsupervised DR(예: Autoencoder)은 판별에 필요한 핵심 정보를 손실할 위험이 있으며, 단순한 Discriminator 학습은 과적합 문제를 야기한다. HashReward는 이를 해결하기 위해 다음과 같은 직관을 사용한다.

- **재구성 정보(Reconstructive information)**를 통해 입력 데이터의 구조를 유지하면서 차원을 축소한다.
- **감독 학습 기반의 해싱(Supervised hashing)**을 통해, 전문가 샘플과 Learner 샘플을 효과적으로 구분할 수 있는 판별적 특성을 해싱 코드에 직접 반영한다.
- 이를 통해 Discriminator가 너무 강력해져서 보상 신호가 사라지는 것을 방지하고, 동시에 고차원 입력에서 발생하는 노이즈를 제거하여 안정적인 학습을 가능하게 한다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들을 소개하며 차별점을 제시한다.

- **GAIL (Generative Adversarial Imitation Learning)**: 저차원 상태 공간에서는 매우 효과적이지만, 고차원 픽셀 입력 환경에서는 Discriminator의 과적합으로 인해 성능이 심각하게 저하된다.
- **VAIL (Variational Adverserary Imitation Learning)**: 정보 이론적 정규화(Information-theoretic regularization)를 통해 특성 표현을 학습하여 GAIL을 개선했다. 하지만 DR 과정에서 명시적인 Supervised Loss를 사용하지 않아 고차원 환경에서의 한계가 존재한다.
- **GIRIL**: VAE를 사용하여 액션 신호를 인코딩하고 Curiosity 메커니즘을 통해 보상을 생성하지만, DR 문제를 직접적으로 해결하지는 않는다.
- **Unsupervised Hashing/Autoencoder**: 단순한 차원 축소는 판별에 필수적인 정보를 유실할 가능성이 커서, Discriminator 학습과 분리된 DR 방식은 성능이 낮다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

HashReward는 크게 **Autoencoder 모듈**과 **Discriminator 모듈**로 구성된다. 입력 픽셀 데이터는 Encoder를 통해 이진 해싱 코드(Binary hashing code)로 변환되며, 이 코드는 Decoder를 통해 다시 복원됨과 동시에 Action 신호와 결합되어 Discriminator의 입력으로 사용된다.

### 2. 학습 목표 및 손실 함수

HashReward의 전체 손실 함수 $\mathcal{L}$은 해싱 학습 손실 $\mathcal{L}_H$와 판별자 학습 손실 $\mathcal{L}_D$의 합으로 정의된다.
$$\mathcal{L} = \mathcal{L}_H + \mathcal{L}_D$$

#### (1) Hashing Training Loss ($\mathcal{L}_H$)

$\mathcal{L}_H$는 다음의 수식으로 계산되며, 세 가지 목적을 동시에 달성한다.
$$\mathcal{L}_H(\{s_i, y_i\}, \{s_j, y_j\}) = \underbrace{\|s_i - s'_i\|_2^2 + \|s_j - s'_j\|_2^2}_{\text{Reconstruction Error}} + \underbrace{\lambda (\|1 - |b(s_i)|\|_2^2 + \|1 - |b(s_j)|\|_2^2)}_{\text{Binarization Regularization}} + \text{Supervised Terms}$$

여기서 $\text{Supervised Terms}$는 다음과 같다.
$$\frac{1}{2} I(y_{ij}) \|b(s_i) - b(s_j)\|_2^2 + \frac{1}{2} (1 - I(y_{ij})) \max(2\delta - \|b(s_i) - b(s_j)\|_2^2, 0)$$

- **재구성 오차**: 복원된 상태 $s'$가 원본 상태 $s$와 유사하게 유지하여 정보 손실을 방지한다.
- **이진화 정규화**: 해싱 코드 $b(s)$가 $\{-1, 1\}$ 값에 가까워지도록 강제한다.
- **감독 학습 항**: 라벨 $y$가 같으면(둘 다 전문가이거나 둘 다 Learner이면) 해싱 코드가 서로 가까워지게 하고, 라벨이 다르면 일정 거리 $\delta$ 이상으로 멀어지게 하여 판별력을 확보한다.

#### (2) Discriminator Training Loss ($\mathcal{L}_D$)

$\mathcal{L}_D$는 GAIL의 목적 함수와 유사하지만, 입력으로 픽셀 $s$ 대신 이진 해싱 코드 $b(s)$와 액션 $a$를 사용한다. 이는 Discriminator가 저차원의 효율적인 표현 위에서 판별을 수행하게 하여 과적합을 방지한다.

### 3. 추론 및 학습 절차

1. **Pretraining**: 전문가 시연 데이터와 랜덤 정책 데이터를 사용하여 Autoencoder를 먼저 사전 학습시킨다.
2. **Iterative Update**:
   - Learner 정책 $\pi_G$를 통해 궤적(trajectories)을 생성한다.
   - 전문가 데이터와 Learner 데이터를 샘플링하여 HashReward 네트워크(Encoder, Decoder, Discriminator)를 업데이트한다.
   - Discriminator의 출력값 $-\log D(s, a)$를 유사 보상(pseudo reward)으로 사용하여 RL 알고리즘(예: PPO)으로 정책 $\pi_G$를 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **환경**: Atari 2600 게임 15종(고차원 픽셀 입력), MuJoCo 시뮬레이터 5종.
- **비교 대상**: GAIL, VAIL, GIRIL, GAIL-AE(Autoencoder 사용), GAIL-UH(Unsupervised Hashing 사용) 등.
- **지표**: 누적 보상(Return).

### 2. 주요 결과

- **정량적 성과**: HashReward는 Atari 15종 중 12종, MuJoCo 5종 중 3종에서 최적의 성능을 기록하였다. 특히 Boxing, Qbert, SpaceInvaders 등에서 전문가 수준의 보상을 달성하였다.
- **정성적 분석 (Pong)**: 전문가의 정교한 'kill-shot'(패들의 짧은 면으로 공을 치는 동작)을 다른 방법론들은 모방하지 못했으나, HashReward는 이를 정확히 학습하여 구현해냈다.
- **보상 곡선 분석**: GAIL이나 VAIL의 경우, 실제 보상(True Reward)이 증가해도 유사 보상(Pseudo Reward)이 정체되거나 하락하는 '과판별' 현상이 나타났다. 반면 HashReward의 유사 보상은 실제 보상의 추세를 매우 유사하게 따라갔다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

본 논문은 고차원 IL의 실패 원인이 단순히 차원이 높아서가 아니라, **Discriminator가 너무 빨리 학습되어 유의미한 보상 신호를 제공하지 못하는 불균형**에 있음을 이론적(Generalization Bound) 및 실험적으로 증명하였다. Supervised Hashing을 통해 DR과 판별 학습을 결합함으로써, 정보 손실 없이 차원을 축소하고 Discriminator의 학습 속도를 제어하여 안정적인 Reward Shaping을 가능하게 했다.

### 2. 한계 및 논의사항

- **사전 학습 의존성**: Autoencoder의 사전 학습 단계가 포함되어 있어, 초기 데이터 수집 및 학습 시간이 추가로 소요된다.
- **해싱 코드 길이**: 64-bit 등의 고정된 해싱 코드 길이를 사용하는데, 환경의 복잡도에 따른 최적의 코드 길이에 대한 분석이 더 필요하다.
- **탐색 문제**: 본 논문은 모방 학습에 집중하고 있으나, Montezuma's Revenge와 같이 극심한 탐색(exploration)이 필요한 게임에서의 성능은 여전히 도전 과제로 남아 있다.

## 📌 TL;DR

본 논문은 고차원 픽셀 환경에서 Adversary-based Imitation Learning이 Discriminator의 과적합으로 인해 실패한다는 점을 분석하고, 이를 해결하기 위해 **Supervised Hashing 기반의 HashReward**를 제안한다. 재구성 오차와 감독 학습 기반의 판별 손실을 결합하여 차원 축소와 판별 성능의 균형을 맞췄으며, 그 결과 Atari 및 MuJoCo 환경에서 기존 SOTA 방법론들을 크게 상회하는 성능을 보였다. 이는 고차원 상태 공간을 가진 로봇 제어나 복잡한 게임 환경의 모방 학습에 매우 중요한 기여를 할 것으로 평가된다.
