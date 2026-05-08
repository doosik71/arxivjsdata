# Domain Adaptive Imitation Learning

Kuno Kim, Yihong Gu, Jiaming Song, Shengjia Zhao, Stefano Ermon (2020)

## 🧩 Problem to Solve

본 논문은 Embodiment(신체 구조), Viewpoint(시점), Dynamics(역학)의 불일치와 같이 도메인 간의 차이가 존재하는 상황에서 어떻게 전문가의 작업을 모방할 것인가에 대한 문제를 다룬다.

기존의 많은 모방 학습 연구들은 두 도메인 간에 시간적으로 정렬된 쌍(paired, aligned) 형태의 시연 데이터가 필요하거나, 환경과의 상호작용이 필수적인 강화학습(RL) 단계를 추가로 거쳐야 했다. 그러나 실제 환경에서 정렬된 시연 데이터를 얻는 것은 매우 어렵고, RL 절차는 비용이 많이 든다는 한계가 있다.

따라서 본 논문의 목표는 정렬되지 않은(unpaired, unaligned) 시연 데이터만을 이용하여, 추가적인 RL 단계 없이 새로운 도메인에서 작업을 최적으로 수행할 수 있도록 하는 **Domain Adaptive Imitation Learning (DAIL)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 '정렬(Alignment)'과 '적응(Adaptation)'이라는 두 단계의 접근 방식을 통해 도메인 간의 간극을 극복하는 것이다.

1. **GAMA (Generative Adversarial MDP Alignment)**: 정렬되지 않은 시연 데이터로부터 상태(state)와 행동(action)의 대응 관계를 학습하는 새로운 비지도 MDP 정렬 알고리즘을 제안한다.
2. **Zero-shot Imitation**: 학습된 정렬 맵을 활용하여 추가적인 환경 상호작용이나 RL 과정 없이 전문가의 정책을 자신의 도메인으로 즉시 전이하여 모방하는 제로샷 모방을 구현한다.
3. **MDP Alignability 이론**: 어떤 조건에서 DAIL이 정렬과 적응을 통해 해결 가능한지를 설명하는 이론적 프레임워크인 MDP Reduction 및 Alignability 개념을 도입하여 연구의 이론적 기반을 마련한다.

## 📎 Related Works

기존의 도메인 전이 학습 연구들은 주로 다음과 같은 접근 방식을 취했다.

- **선형 투영 및 매니폴드 정렬**: CCA(Canonical Correlation Analysis)나 UMA(Unsupervised Manifold Alignment)와 같은 방법들이 상태 간의 대응 관계를 찾으려 했으나, 이는 주로 선형적인 맵에 의존하거나 수작업으로 설계된 특징(hand-crafted features)을 필요로 했다.
- **정렬된 데이터 기반 학습**: IF(Invariant Features)나 IfO(Imitation from Observation) 등은 시간적으로 정렬된 시연 데이터가 있다는 가정하에 상태 맵을 학습했다. 이는 데이터 수집 비용이 매우 높다는 단점이 있다.
- **시점 불일치 해결**: TPIL(Third Person Imitation Learning) 등은 도메인 공통의 특징 공간을 학습하여 시점 차이를 극복하려 했으나, 시점 차이가 매우 큰 경우에는 성능이 저하되는 한계가 있었다.

본 논문은 정렬되지 않은 데이터만을 사용한다는 점, 행동 맵($g$)을 함께 학습하여 RL 단계 없이 제로샷 전이가 가능하다는 점, 그리고 신체 구조와 시점, 역학 불일치를 하나의 통합된 프레임워크로 해결한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. MDP Reduction 및 Alignability 이론

논문은 두 MDP $M_x$와 $M_y$ 사이에 구조를 보존하는 맵 $r = (\phi, \psi)$가 존재할 때, $M_x$가 $M_y$로 **Reduction** 된다고 정의한다. 여기서 $\phi: S_x \to S_y$는 상태 맵, $\psi: A_x \to A_y$는 행동 맵이다.

- **$\pi$-optimality**: $M_y$에서 최적인 상태-행동 쌍은 $M_x$에서도 최적이어야 하며, $M_y$의 모든 최적 쌍은 $M_x$에서 대응되는 쌍이 존재해야 한다.
- **Dynamics preservation**: $M_y$의 전이 함수 $P_y$는 $M_x$의 전이 함수 $P_x$를 통해 $\phi(P_x(s_x, a_x)) = P_y(\phi(s_x), \psi(a_x))$를 만족해야 한다.

이러한 Reduction이 존재하면, 전문가 도메인 $M_y$의 최적 정책 $\pi_y$를 $\hat{\pi}_x = g \circ \pi_y \circ f$ (여기서 $f=\phi, g=\psi^{-1}$) 형태로 변환하여 $M_x$에서도 최적 정책을 얻을 수 있다.

### 2. GAMA (Generative Adversarial MDP Alignment)

GAMA는 위에서 정의한 MDP Reduction을 학습하기 위해 적대적 학습(Adversarial Training) 구조를 사용한다.

#### 훈련 목표 및 손실 함수

GAMA는 다음의 두 가지 목적 함수를 최적화한다.

1. **정책 성능 최대화**: 자신의 도메인($x$)에서 모방 정책 $\hat{\pi}_x$가 얼마나 잘 수행되는지를 측정한다.
2. **전이 분포 일치**: $M_x$에서 $\hat{\pi}_x$를 실행했을 때 발생하는 전이 튜플 $(s_y, a_y, s'_y)$의 분포 $\sigma_{x \to y}^{\hat{\pi}_x}$가 전문가 도메인 $M_y$의 전이 분포 $\sigma_y^{\pi_y}$와 일치하도록 만든다.

전체 목적 함수는 다음과 같이 정의된다.
$$\min_{f,g} \max_{\{D_{\theta_i D}\}} \sum_{i=1}^N \left( \mathbb{E}_{s_x \sim \pi^*_{x,T_i}} [D_{KL}(\pi^*_{x,T_i}(\cdot|s_x) || \hat{\pi}_{x,T_i}(\cdot|s_x))] + \lambda (\mathbb{E}_{\pi^*_{y,T_i}} [\log D_{\theta_i D}(s_y, a_y, s'_y)] + \mathbb{E}_{\pi^*_{x,T_i}} [\log(1 - D_{\theta_i D}(\hat{s}_y, \hat{a}_y, \hat{s}'_y))]) \right)$$

여기서 첫 번째 항은 **Behavioral Cloning**을 통한 정책 성능 최적화이며, 두 번째 항은 **GAN** 형태의 손실 함수로, 판별기(Discriminator) $D$가 실제 전문가의 전이와 $\hat{\pi}_x$에 의해 생성된 전이를 구분하지 못하도록 상태 맵 $f$와 행동 맵 $g$를 학습시킨다.

### 3. 추론 및 적응 절차

1. **Alignment Phase**: 여러 정렬 작업 세트 $D_{x,y}$를 통해 공통의 상태 맵 $f_{\theta_f}$와 행동 맵 $g_{\theta_g}$를 학습한다.
2. **Adaptation Phase**: 새로운 타겟 작업 $T$에 대해 전문가 도메인의 시연 데이터로부터 $\pi_{y,T}$를 학습하고, 이를 $\hat{\pi}_{x,T} = g_{\theta_g} \circ \pi_{y,T} \circ f_{\theta_f}$로 결합하여 자신의 도메인 정책을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 환경**: Pendulum, Cartpole, Reacher, Snake 등의 환경을 사용하였다.
- **비교 대상**: CCA, UMA, IF, IfO, TPIL, SAIL 등의 기존 베이스라인과 비교하였다.
- **평가 지표**:
  - **Alignment Complexity**: 제로샷 모방이 가능해질 때까지 필요한 정렬 작업(Task)의 수.
  - **Adaptation Complexity**: 타겟 작업을 성공적으로 모방하기 위해 필요한 전문가 시연 데이터의 양.
  - **$\ell_2$ Loss**: 학습된 상태 맵과 Ground-truth 정렬 맵 사이의 거리.

### 주요 결과

- **정량적 결과**: 상태 맵 학습 성능 평가에서 GAMA는 베이스라인 대비 평균적으로 $17.3\times$ 더 낮은 $\ell_2$ 손실을 기록하였다. 특히 정렬되지 않은 데이터셋 환경에서 다른 방법론들이 실패하는 반면 GAMA는 성공적인 정렬을 보여주었다.
- **DAIL 성능**:
  - **Dynamics, Embodiment, Viewpoint 불일치** 모든 시나리오에서 GAMA는 제로샷 모방에 성공하였다.
  - **Adaptation Complexity** 측정 결과, GAMA가 생성한 적응된 시연 데이터가 실제 자신의 도메인 시연 데이터(Self-Demo)와 유사한 수준의 효율성을 보였다.
  - **Visual Inputs**: 딥 공간 오토인코더(Deep Spatial Autoencoder)를 사용하여 고차원 이미지 입력 환경에서도 GAMA가 효과적으로 동작함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 서로 다른 도메인 간의 모방 학습 문제를 'MDP Reduction'이라는 수학적 프레임워크로 정형화하여 이론적 근거를 제시하였다. 특히, 기존 연구들이 필수적으로 요구했던 '정렬된 데이터'와 'RL 상호작용'이라는 두 가지 큰 제약 조건을 모두 제거하고도 높은 성능을 냈다는 점이 매우 고무적이다.

### 한계 및 논의사항

- **이론적 가정**: 이론적 증명 과정에서 MDP가 Unichain이며 결정론적 역학(Deterministic Dynamics)을 가진다고 가정하였다. 실제 환경은 확률적(Stochastic)인 경우가 많으므로, 이에 대한 확장이 필요하다.
- **Weakly Alignable**: 저자들은 이론적으로 완벽하게 Alignable하지 않은 경우(예: snake4 $\leftrightarrow$ snake3)에도 GAMA가 잘 작동함을 보였는데, 이는 이론과 실제 구현 사이에 간극이 존재함을 시사한다. 향후 '약한 정렬 가능성'에 대한 이론적 정립이 필요할 것으로 보인다.
- **최소 정렬 세트**: 현재는 충분한 양의 정렬 작업 세트가 있다고 가정하지만, 실제 적용을 위해서는 최소한의 데이터로 정렬을 수행할 수 있는 방법론에 대한 연구가 추가되어야 한다.

## 📌 TL;DR

본 논문은 신체 구조, 시점, 역학이 서로 다른 도메인 간의 모방 학습을 해결하기 위해 **GAMA**라는 비지도 MDP 정렬 알고리즘을 제안한다. GAMA는 정렬되지 않은 시연 데이터로부터 상태와 행동의 대응 관계를 적대적 학습으로 찾아내며, 이를 통해 추가적인 RL 단계 없이 전문가의 기술을 자신의 도메인으로 즉시 전이하는 **Zero-shot Imitation**을 가능하게 한다. 이 연구는 로봇이 인간의 시연을 보거나 다른 형태의 로봇으로부터 기술을 배울 때 발생하는 도메인 격차를 줄이는 데 핵심적인 역할을 할 것으로 기대된다.
