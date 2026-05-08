# Imitation Learning from Purified Demonstrations

Yunke Wang, Minjing Dong, Yukun Zhao, Bo Du, Chang Xu (2024)

## 🧩 Problem to Solve

본 논문은 순차적 의사 결정 문제(sequential decision-making problems)를 해결하기 위한 Imitation Learning(IL)에서 발생하는 **불완전한 전문가 시연(Imperfect Demonstrations)** 문제를 다룬다. 전통적인 IL 방식은 전문가의 시연 데이터가 최적(optimal)이라는 가정하에 작동하지만, 실제 환경에서 수집된 데이터에는 상당한 노이즈가 포함되어 있어 학습된 정책의 성능을 저하시키는 경우가 많다.

기존의 연구들은 불완전한 시연 데이터를 활용해 최적화하는 방향으로 접근했으나, 여전히 일정 비율 이상의 최적 시연 데이터가 확보되어야 성능이 보장된다는 한계가 있다. 또한, 많은 Confidence-based 방법들이 신뢰도 추정과 정책 학습을 동시에 수행하는 bi-level optimization 구조를 가지고 있어 학습 과정이 불안정하고 수렴이 어렵다는 문제가 존재한다. 따라서 본 논문의 목표는 시연 데이터에서 노이즈를 먼저 제거하는 **Purification(정제)** 과정을 도입하여, 정제된 데이터를 통해 안정적으로 최적의 정책을 학습하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Diffusion Model의 정방향(Forward) 및 역방향(Reverse) 확산 과정을 이용하여 불완전한 시연 데이터를 정제**하는 것이다.

중심적인 직관은 sub-optimal한 시연 데이터에 의도적으로 노이즈를 추가하여 기존의 노이즈 패턴을 매끄럽게(smooth) 만든 후, 최적 데이터로 학습된 Diffusion Model을 통해 이를 다시 복구함으로써 최적의 시연 데이터에 가까운 형태로 변환한다는 점이다. 이를 통해 복잡한 bi-level optimization 없이도 데이터 정제와 정책 학습을 분리하여 수행할 수 있으며, 정제된 데이터를 Behavioral Cloning(BC)이나 GAIL과 같은 다양한 IL 알고리즘에 유연하게 적용할 수 있다.

## 📎 Related Works

### 1. Imitation Learning from Imperfect Demonstrations

- **Confidence-based methods**: WGAIL, SAIL, BCND, CAIL 등은 각 시연 데이터에 가중치를 부여하여 중요도를 조절한다. 그러나 이러한 방식은 가중치 추정과 IL 학습이 얽혀 있어 최적화가 불안정하며, 일부 최적 데이터가 반드시 필요하다는 제약이 있다.
- **Preference-based methods**: T-REX, D-REX, SSRR 등은 인간의 선호도나 랭킹 정보를 이용하여 보상 함수를 학습함으로써 최적의 정책을 유도한다. 이는 정교한 랭킹 정보가 필요하다는 전제가 따른다.

### 2. Diffusion Model in Imitation Learning

최근 Diffusion Model을 정책 네트워크 자체로 정의하거나(Diffusion Policy), GAIL의 판별자(Discriminator)로 사용하는 연구들이 등장하였다. 하지만 이러한 연구들은 생성 능력의 향상에 집중할 뿐, 입력 데이터 자체가 불완전할 때 발생하는 노이즈 제거 문제(Imperfect Demonstration issue)를 직접적으로 해결하지는 않는다.

## 🛠️ Methodology

### 전체 파이프라인

제안된 **Diffusion Purified Imitation Learning (DP-IL)**은 크게 두 단계로 구성된다:

1. **Purification**: 소량의 최적 시연 데이터($D_o$)로 Diffusion Model을 학습시킨 후, 이를 이용해 다량의 sub-optimal 시연 데이터($D_s$)를 정제한다.
2. **Imitation Learning**: 정제된 데이터 $\hat{D}_s$와 최적 데이터 $D_o$를 합쳐 일반적인 IL 알고리즘(BC, GAIL 등)으로 정책을 학습한다.

### 상세 구성 요소 및 절차

#### 1. Diffusion Model 학습

최적 시연 데이터 $x_o = (s, a)$를 사용하여 DDPM(Denoising Diffusion Probabilistic Models) 기반의 모델 $\epsilon_\phi$를 학습시킨다. 학습 목표는 다음과 같은 손실 함수를 최소화하는 것이다:
$$\min_{\phi} \mathbb{E}_{x_o, \epsilon, i} \left[ \left\| \epsilon - \epsilon_\phi (\sqrt{\bar{\alpha}_i} \cdot x_o + \sqrt{1 - \bar{\alpha}_i} \epsilon, i) \right\|^2 \right]$$
여기서 $\epsilon$은 가우시안 노이즈이며, $\bar{\alpha}_i$는 노이즈 스케줄에 따른 계수이다.

#### 2. 시연 데이터 정제 (Purification)

학습된 $\epsilon_\phi$를 사용하여 sub-optimal 데이터 $x_s$를 정제한다. 이 과정은 두 단계로 나뉜다.

- **Forward Diffusion (Noise Injection)**: 특정 역전 지점(inverse point) $i_r$까지 노이즈를 추가하여 기존의 노이즈 패턴을 제거한다.
$$\hat{x}_{i_r} = \sqrt{\bar{\alpha}_{i_r}} \cdot x_s + \sqrt{1 - \bar{\alpha}_{i_r}} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
- **Reverse Diffusion (Denoising)**: $i_r$ 단계부터 $0$ 단계까지 역과정을 통해 데이터를 복구한다.
$$\hat{x}_{i-1} = \frac{1}{\sqrt{\alpha_i}} \left( \hat{x}_i - \frac{1 - \alpha_i}{\sqrt{1 - \bar{\alpha}_i}} \epsilon_\phi(\hat{x}_i, t) \right) + \sqrt{\beta_i} z, \quad z \sim \mathcal{N}(0, I)$$
최종적으로 도출된 $\hat{x}_0$가 정제된 시연 데이터가 된다.

### 이론적 분석 및 최적 $i_r$ 선택

논문은 역전 지점 $i_r$ (또는 연속 시간 $t_r$)의 선택이 성능에 결정적인 영향을 미침을 이론적으로 증명한다.

- **Theorem 1**: 정방향 확산 과정에서 시간이 지날수록 최적 분포와 sub-optimal 분포 사이의 KL divergence가 단조 감소함을 보인다. 즉, $t_r$이 클수록 노이즈 제거 효과가 커진다.
- **Theorem 2 & Proposition 3**: 정제된 데이터와 최적 데이터 사이의 $L_2$ 거리의 상한과 하한을 정의하며, $t_r$이 증가함에 따라 이 거리의 경계값이 함께 증가함을 보인다.
- **Theorem 4**: 따라서 노이즈 제거(큰 $t_r$ 필요)와 데이터 구조 유지(작은 $t_r$ 필요) 사이의 Trade-off가 존재하며, 최적의 $t_r^*$는 특정 변곡점 $t_p$ 근처에 위치한다. 또한, 입력 데이터의 노이즈 $\delta$가 클수록 더 큰 $t_r^*$가 필요함을 시사한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**: MuJoCo (Ant-v2, HalfCheetah-v2, Walker2d-v2, Hopper-v2) 및 RoboSuite (Nut Assembly) 작업에서 평가하였다.
- **시연 데이터 생성**:
  - $D_o$: TRPO로 학습된 최적 정책으로 생성.
  - $D_s$: 최적 액션에 가우시안 노이즈를 추가하거나(D1), RL 학습 과정의 중간 체크포인트를 활용(D2)하여 생성. 노이즈 수준은 L1(강함), L2(중간), L3(약함)로 구분하였다.
- **비교 대상**: BC, BCND, DWBC, DemoDICE (Offline), 2IWIL, IC-GAIL, WGAIL (Online).
- **지표**: MuJoCo에서는 누적 보상(Average cumulative reward), RoboSuite에서는 성공률(Success rate)을 측정하였다.

### 주요 결과

- **정량적 결과**: DP-BC와 DP-GAIL은 모든 MuJoCo 작업과 다양한 노이즈 수준에서 기존 baseline들을 압도하는 성능을 보였다. 특히, 노이즈가 매우 심한 L1 상황에서도 다른 방법론보다 훨씬 안정적인 성능 향상을 기록하였다.
- **정성적 분석**: $i_r$ 값에 따른 성능 변화를 분석한 결과, 이론적 예측대로 노이즈가 심한 데이터(L1)일수록 더 큰 $i_r$에서 최적 성능이 나타났으며, 노이즈가 적은 데이터(L3)일수록 작은 $i_r$에서 좋은 성능을 보였다.
- **RoboSuite 결과**: 실제 인간의 조작 데이터가 포함된 RoboSuite 환경에서도 DP-BC가 가장 높은 성공률(0.86)을 기록하며 강건함을 입증하였다.
- **노이즈 강건성**: 가우시안 노이즈 외에 Uniform noise, Salt-and-pepper noise 환경에서도 DP-BC가 일관되게 최선의 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점

본 연구의 가장 큰 강점은 **데이터 정제와 정책 학습의 분리(Decoupling)**이다. 기존의 Confidence-based 방법들이 학습 중에 가중치를 동적으로 조절하려다 발생하는 불안정성을 완전히 제거하였으며, 정제된 데이터를 생성하는 모듈이므로 BC나 GAIL 등 어떤 IL 알고리즘과도 결합 가능한 높은 범용성을 가진다.

### 한계 및 논의사항

- **최적 데이터 의존성**: Diffusion Model을 학습시키기 위해 소량의 최적 시연 데이터($D_o$)가 반드시 필요하다. 만약 완전히 최적인 데이터가 단 하나도 없는 환경이라면 본 방법론을 그대로 적용하기 어렵다.
- **하이퍼파라미터 $i_r$**: 이론적 가이드라인이 제시되었으나, 실제 환경에서는 여전히 $i_r$을 튜닝해야 하는 부담이 있다. 다만, 실험을 통해 간단한 작업은 $i_r \in [5, 10]$, 어려운 작업은 $[10, 50]$ 범위에서 좋은 성능이 나온다는 경험적 가이드를 제공하였다.

## 📌 TL;DR

본 논문은 불완전한 전문가 시연 데이터로 인해 발생하는 Imitation Learning의 성능 저하 문제를 해결하기 위해, **Diffusion Model을 이용한 데이터 정제 과정(DP-IL)**을 제안한다. 최적 데이터를 통해 학습된 Diffusion Model로 sub-optimal 데이터에 노이즈를 주입했다가 다시 복구하는 2단계 과정을 통해 노이즈를 제거하며, 이를 통해 BC나 GAIL과 같은 기존 IL 알고리즘의 성능을 획기적으로 높였다. 이 연구는 데이터 정제-정책 학습의 분리라는 새로운 패러다임을 제시하며, 향후 실제 환경의 저품질 데이터를 활용한 로봇 학습 및 제어 분야에 중요하게 적용될 가능성이 높다.
