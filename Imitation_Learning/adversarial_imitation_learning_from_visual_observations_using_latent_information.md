# Adversarial Imitation Learning from Visual Observations using Latent Information

Vittorio Giammarino, James Queeney, Ioannis Ch. Paschalidis (2024)

## 🧩 Problem to Solve

본 논문은 전문가의 비디오 영상만을 학습 소스로 사용하는 **Visual Imitation from Observations (V-IfO)** 문제에 집중한다. V-IfO는 기존의 모방 학습(Imitation Learning)과 비교하여 다음과 같은 두 가지 핵심적인 난제를 가진다.

1. **전문가 액션의 부재 (Absence of Expert Actions):** 학습 에이전트는 전문가가 어떤 행동을 취했는지에 대한 정보 없이 오직 관측값(비디오 프레임)만을 통해 학습해야 한다.
2. **환경의 부분 관측성 (Partial Observability):** 픽셀 기반의 관측값만으로는 환경의 실제 상태(Ground-truth state)를 완전히 알 수 없으므로, 이는 POMDP(Partially Observable Markov Decision Process) 문제로 정의된다.

이러한 제약 조건 하에서 복잡한 로봇 제어 작업을 수행할 수 있는 알고리즘을 개발하는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 V-IfO 문제를 이론적으로 분석하고, 이를 바탕으로 효율적인 잠재 공간(Latent Space) 기반의 모방 학습 알고리즘인 **LAIfO (Latent Adversarial Imitation from Observations)**를 제안한 것이다.

- **이론적 분석:** 에이전트의 성능 저하(Suboptimality)가 전문가와 에이전트의 **잠재 상태-전이 분포(Latent state-transition distributions)** 간의 Divergence에 의해 상한(Upper bound)이 결정됨을 수학적으로 증명하였다.
- **LAIfO 알고리즘:** 위 이론적 분석을 바탕으로 V-IfO 문제를 (1) 관측 시퀀스로부터 적절한 잠재 표현을 추정하는 문제와 (2) 잠재 공간에서 전문가와 에이전트의 분포 차이를 최소화하는 문제로 환원하여 해결하였다.
- **효율성 증명:** 픽셀 공간에서 직접 모방 학습을 수행하는 기존 방식과 달리, 저차원의 잠재 공간에서 Adversarial Imitation을 수행함으로써 계산 효율성을 극대화하였다.

## 📎 Related Works

논문은 모방 학습의 네 가지 프레임워크를 구분하며 기존 연구의 한계를 지적한다.

- **Imitation Learning (IL):** 상태가 완전 관측 가능하고 전문가의 상태-액션 쌍이 제공되는 설정이다.
- **Imitation from Observations (IfO):** 상태는 완전 관측 가능하나, 전문가의 액션 정보가 없는 설정이다.
- **Visual Imitation Learning (V-IL):** 픽셀 관측을 사용하지만, 전문가의 액션 정보가 제공되는 설정이다.
- **Visual Imitation from Observations (V-IfO):** 픽셀 관측만 가능하며 전문가 액션 정보도 없는 가장 어려운 설정이다.

기존의 V-IfO 접근 방식인 **PatchAIL**은 픽셀 공간에서 직접 Adversarial IL을 수행하기 때문에 고차원 데이터로 인한 막대한 계산 비용이 발생한다는 한계가 있다. 반면, V-IL의 최신 기법인 **VMAIL**은 모델 기반(Model-based) 접근법을 사용하여 세계 모델(World Model)을 학습시켜야 하므로 학습 과정이 불안정하고 계산량이 많다는 단점이 있다.

## 🛠️ Methodology

### 1. 이론적 배경 및 분석

연구진은 POMDP 환경에서 에이전트 $\pi_\theta$와 전문가 $\pi_E$의 성능 차이를 분석하였다. 보상 함수가 상태 전이($S \times S$)에 의존한다고 가정할 때, 다음의 정리를 도출하였다.

**Theorem 2:**
$$|J(\pi_E) - J(\pi_\theta)| \le \frac{2R_{\max}}{1-\gamma} D_{TV}(\rho^{\pi_\theta}(z, z'), \rho^{\pi_E}(z, z'))$$
여기서 $D_{TV}$는 Total Variation distance이며, $\rho(z, z')$는 잠재 상태 전이 방문 분포(Latent state-transition visitation distribution)이다. 즉, 잠재 공간에서의 전이 분포만 잘 일치시키면 전문가의 성능에 근접할 수 있음을 의미한다.

### 2. LAIfO 알고리즘 구조

LAIfO는 크게 **잠재 변수 추정**과 **잠재 공간에서의 적대적 모방 학습** 두 단계로 구성된다.

#### (1) 잠재 변수 추정 (Latent Variable Estimation)

에이전트는 픽셀 관측값 $x$로부터 상태의 충분 통계량(Sufficient statistic)인 잠재 변수 $z$를 추정해야 한다.

- **Observation Stacking:** 최근 $d$개의 관측 프레임을 쌓아서 입력으로 사용한다.
- **Data Augmentation:** Crop과 같은 데이터 증강 기법을 적용하여 특징 추출기의 강건성을 높인다.
- **Feature Extractor:** $\phi_\delta : X^d \to Z$ 함수를 통해 $z = \phi_\delta(\text{aug}(x_{t-d+1:t}))$로 잠재 상태를 생성한다.

#### (2) 적대적 모방 학습 (Off-policy Adversarial IfO)

잠재 전이 $(z, z')$를 구분하는 판별기(Discriminator) $D_\chi$를 학습시켜 전문가의 분포를 모방한다.

- **판별기 손실 함수:**
$$\max_\chi \mathbb{E}_{(z, z') \in \mathcal{B}_E} [\log(D_\chi(z, z'))] + \mathbb{E}_{(z, z') \in \mathcal{B}} [\log(1 - D_\chi(z, z'))] + g(\nabla_\chi D_\chi)$$
여기서 $g(\nabla_\chi D_\chi)$는 학습 안정성을 위한 Gradient Penalty 항이다.
- **에이전트 학습:** 판별기의 출력값 $D_\chi(z, z')$를 보상 $r_\chi$로 사용하여, Off-policy RL(DDPG 기반)을 통해 정책 $\pi_\theta$를 최적화한다.

#### (3) 학습 절차의 특이점

특징 추출기 $\phi_\delta$는 Critic 네트워크의 손실 함수를 통해 업데이트되지만, **판별기 $D_\chi$와 정책 $\pi_\theta$로부터 오는 그래디언트 전파는 차단(Stop-gradient)**한다. 이는 잠재 변수 $z$가 특정 플레이어(판별기 혹은 정책)에게 편향되지 않고, 오직 성능 결정에 필요한 정보만을 담도록 하기 위함이다.

### 3. RL 효율성 향상 기법

전문가 비디오가 있을 때, 표준 RL 목적 함수에 LAIfO의 목적 함수를 결합하여 샘플 효율성을 높인다.
$$\max_\theta \mathbb{E}_{\tau^\theta} \left[ \sum_{t=0}^\infty \gamma^t (R(s_t, a_t) + r_\chi(z_t, z_{t+1})) \right]$$
즉, 환경의 실제 보상 $R$에 판별기가 제공하는 보상 $r_\chi$를 더해 학습함으로써 가이드라인을 제공한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** DeepMind Control Suite (13가지 작업).
- **비교 대상:** PatchAIL-W (V-IfO SOTA), VMAIL (V-IL SOTA), Behavioral Cloning (BC).
- **평가 지표:** 에피소드 당 평균 리턴, 전문가 성능 대비 도달 시간(Wall-clock time).

### 2. 주요 결과

- **V-IfO 성능:** LAIfO는 PatchAIL-W와 대등하거나 더 높은 최종 성능을 보였다. 특히 **계산 효율성**에서 압도적인 우위를 점했는데, 이는 픽셀 공간이 아닌 저차원 잠재 공간에서 학습했기 때문이다. (Wall-clock time ratio가 많은 작업에서 0.1~0.6 수준으로 매우 낮음)
- **V-IL 성능:** 전문가 액션이 제공되는 설정(LAIL)에서 모델 기반 방식인 VMAIL보다 더 높은 성능과 빠른 수렴 속도를 보였다. 이는 모델 프리(Model-free) 방식이 세계 모델 학습의 부담과 불안정성을 제거했기 때문이다.
- **RL 가속화:** Humanoid 작업에서 LAIfO를 결합한 RL이 최신 RL 알고리즘인 DrQv2보다 훨씬 빠른 샘플 효율성을 보이며 성능을 향상시켰다.

## 🧠 Insights & Discussion

### 강점

본 논문은 단순한 알고리즘 제안을 넘어, POMDP 환경에서의 모방 학습을 잠재 전이 분포의 일치 문제로 공식화한 이론적 토대가 매우 견고하다. 이를 통해 "왜 잠재 공간에서 학습해야 하는가"에 대한 정당성을 부여했으며, 실제 실험을 통해 계산 효율성과 성능을 동시에 잡았음을 입증하였다.

### 한계 및 가정

- **동일 환경 가정:** 전문가와 에이전트가 동일한 POMDP 내에서 동작한다고 가정한다. 현실 세계에서는 카메라 각도나 환경 설정이 다른 **Domain Mismatch** 문제가 발생하며, 이를 해결하기 위한 Visual Domain Adaptation 연구가 추가로 필요하다.
- **적대적 학습의 불안정성:** GAN 기반의 적대적 학습은 하이퍼파라미터에 민감하고 학습이 불안정할 수 있다. Gradient Penalty 등으로 완화했으나, 다른 Divergence 최소화 기법에 대한 탐색 가능성이 남아있다.

### 비판적 해석

잠재 변수 추정을 위해 사용한 'Observation Stacking'과 'Data Augmentation'은 매우 단순한 기법이다. 논문에서는 이것만으로도 충분하다고 주장하지만, 더 복잡한 환경이나 장기 의존성(Long-term dependency)이 필요한 작업에서는 Variational Inference나 Contrastive Learning 같은 고도화된 표현 학습 기법이 필수적일 것으로 보인다.

## 📌 TL;DR

본 논문은 전문가 비디오만으로 학습하는 V-IfO 문제를 해결하기 위해, **잠재 상태-전이 분포를 일치시키는 것이 핵심**이라는 이론적 증명과 함께 **LAIfO** 알고리즘을 제안하였다. 이 알고리즘은 픽셀 데이터를 저차원 잠재 공간으로 투영한 후 적대적 모방 학습을 수행함으로써, 기존 SOTA 대비 **비슷하거나 더 높은 성능을 내면서도 계산 시간을 획기적으로 단축**시켰다. 또한, 이 기법을 표준 RL에 결합하여 고차원 로봇 제어 작업의 학습 효율을 크게 높일 수 있음을 보여주었다.
