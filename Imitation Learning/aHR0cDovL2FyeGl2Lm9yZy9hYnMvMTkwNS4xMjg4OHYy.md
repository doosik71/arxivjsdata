# Imitation Learning as f-Divergence Minimization

Liyiming Ke, Sanjiban Choudhury, Matt Barnes, Wen Sun, Gilwoo Lee, and Siddhartha Srinivasa

## 🧩 Problem to Solve

본 논문은 여러 모드를 포함하는 시연(multi-modal demonstrations)을 이용한 모방 학습(Imitation Learning, IL) 문제를 다룹니다. 기존 최첨단 방법인 GAIL(Generative Adversarial Imitation Learning) 및 행동 복제(Behavior Cloning, BC)는 손실 함수 선택으로 인해 이러한 모드들을 잘못 보간(interpolate)하여 안전하지 않은 행동(예: 장애물로 돌진)을 유발하는 문제가 있습니다. 저자들은 많은 작업에서 모든 모드를 학습하는 대신 전문가 시연의 단일 모드를 모방하는 것으로 충분하다고 주장합니다.

## ✨ Key Contributions

- 학습자 및 전문가 궤적 분포(trajectory distributions) 간의 f-다이버전스(f-divergence) 최소화를 위한 통합 모방 학습 프레임워크를 제안합니다.
- 임의의 f-다이버전스 추정치를 최소화하는 알고리즘(f-VIM)을 제안합니다. 이 프레임워크는 서로 다른 다이버전스(예: Kullback-Leibler, Jensen-Shannon, Total Variation)를 적용하여 기존 IL 알고리즘(Behavior Cloning, GAIL, DAgger)을 재현할 수 있음을 보여줍니다.
- 역 KL 다이버전스(Reverse KL divergence, I-projection)가 "모드 추구(mode-seeking)" 특성을 가지고 있어 다중 모드 입력에 효과적임을 주장합니다.
- 제안하는 근사 역 KL 다이버전스(RKL-VIM) 기법이 GAIL 및 BC보다 다중 모드 행동을 더 안정적으로 모방하며, 안전하게 단일 모드로 수렴함을 경험적으로 입증합니다.

## 📎 Related Works

- **모방 학습 (IL):** 로봇 공학에서 오랜 역사를 가지고 있으며, 강화 학습(Reinforcement Learning, RL)의 초기화에 활용되기도 합니다.
- **행동 복제 (Behavior Cloning):** 지도 학습(supervised learning)으로 전문가와 동일한 행동을 선택하지만, 작은 오류가 누적되어 분포 불일치(distribution mismatch)로 이어질 수 있습니다.
- **상호작용 학습 (Interactive Learning, DAgger):** 분포 불일치를 완화하지만, 일부 도메인에서는 온-정책 전문가 레이블(on-policy expert labels)을 얻기 비실용적이며 바람직하지 않은 행동을 초래할 수 있습니다.
- **역 강화 학습 (Inverse Reinforcement Learning, IRL):** 전문가 행동을 설명하는 보상 함수나 Q-값을 복구합니다. GANs(Generative Adversarial Networks)와 유사한 게임 이론적 프레임워크(GAIL, AIRL 등)로 확장되었습니다.
- **f-다이버전스:** 상대 엔트로피(relative entropy), 대칭 교차 엔트로피(symmetric cross-entropy)와 같은 특정 다이버전스 측정값을 최소화하는 IL 방법론과 연결됩니다.
- **다중 모드 시연 처리:** 데이터를 클러스터링하여 각 클러스터별로 학습하거나(InfoGAN/InfoGAIL), 최대 엔트로피 정식화(maximum entropy formulations)를 확장하여 다중 모드 정책을 학습하는 방법이 있었습니다.

## 🛠️ Methodology

1. **문제 정의:** MDP(Markov Decision Process), 궤적 분포 $\rho_{\pi}(\tau)$, 상태-행동 분포 $\rho_{\pi}(s)\pi(a|s)$ 및 f-다이버전스 $D_{f}(p,q) = \sum_x q(x)f(\frac{p(x)}{q(x)})$를 정의합니다.
2. **IL 목표:** 학습자 정책 $\pi$가 전문가 정책 $\pi^*$의 궤적 분포와 유사하도록 $D_{f}(\rho_{\pi^*}(\tau),\rho_{\pi}(\tau))$를 최소화하는 것을 목표로 합니다.
3. **근사:** 궤적 분포 대신 **평균 상태-행동 분포** 간의 f-다이버전스 $D_{f}(\rho_{\pi^*}(s)\pi^*(a|s),\rho_{\pi}(s)\pi(a|s))$를 최소화합니다. 이는 원래 목표의 하한(lower bound)입니다 (Theorem 1).
4. **변분 근사 (Variational Approximation):** $f(\cdot)$의 변분 형태를 사용하여 샘플만 있을 때 다이버전스를 추정합니다.
    $$ D_{f}(p,q) \ge \sup_{\phi \in \Phi} \left( \mathbb{E}_{x \sim p}[\phi(x)] - \mathbb{E}_{x \sim q}[f^*(\phi(x))] \right) $$
5. **f-VIM 프레임워크 (Algorithm 1):** 변분 모방 학습(Variational Imitation, VIM)을 위해 다음과 같은 최적화 문제를 해결합니다.
    $$ \hat{\pi}= \arg \min_{\pi \in \Pi} \max_w \mathbb{E}_{(s,a)\sim\rho_{\pi^*}}[g_f(V_w(s,a))] - \mathbb{E}_{(s,a)\sim\rho_{\pi}}[f^*(g_f(V_w(s,a)))] $$
    여기서 $V_w$는 무제한 판별자(discriminator), $g_f$는 활성화 함수(activation function)입니다. 이 프레임워크는 판별자 $V_w$와 학습자 $\pi_{\theta}$를 반복적으로 업데이트하여 saddle point를 찾습니다.
6. **기존 IL 알고리즘 복구:**
    - **KL-VIM:** $f(u) = u \log u$를 사용하여 행동 복제(Behavior Cloning)를 재현합니다. 이는 "모드 커버링(mode-covering)" 특성을 보입니다.
    - **JS-VIM:** $f(u) = -(u+1)\log\frac{1+u}{2} + u\log u$를 사용하여 GAIL(Jensen-Shannon)을 재현합니다. 이는 모드 커버링 또는 혼합 특성을 보입니다.
    - **RKL-VIM:** $f(u) = -\log u$를 사용하여 역 KL 다이버전스(Reverse KL)를 최소화합니다. 이는 "모드 추구(mode-seeking)" 특성을 보입니다.
    - **DAgger:** 총 변동(Total Variation, TV) 거리를 사용한 목표로 설명됩니다.
7. **역 KL 최소화를 위한 대체 기법 (상호작용 학습):** 상호작용 전문가를 활용할 수 있는 경우, 역 KL 다이버전스를 직접 행동 분포 다이버전스로 변환하여 최적화할 수 있습니다.
    - **변분 행동 다이버전스 최소화 (RKL-iVIM):** 행동 다이버전스를 직접 최소화하여 더 쉬운 추정치를 제공합니다.
    - **무후회 온라인 학습을 통한 밀도 비율 최소화:** 밀도 비율 추정기(Density Ratio Estimator, DRE)와 DAgger와 유사한 온라인 학습을 사용하여 행동 다이버전스의 상한을 최소화합니다.

## 📊 Results

- **저차원 태스크 (Bandit, GridWorld):**
  - **가설 H1 (글로벌 최적점):** 역 KL 다이버전스(RKL)의 글로벌 최적 정책은 전문가 시연의 단일 모드로 수렴하는 반면, KL 및 JS 다이버전스는 모드 간을 보간하여 안전하지 않은 행동을 보였습니다.
  - **가설 H2 (다이버전스 추정):** KL 및 JS 다이버전스에 대한 샘플 기반 추정치는 선호하는 정책에 대해 실제 다이버전스를 RKL보다 더 많이 과소평가하는 경향이 있었습니다.
  - **가설 H3 (정책 기울기 최적화):** 정책 기울기 최적화 결과도 RKL-VIM이 단일 모드로 수렴하는 정책을 생성하는 반면, JS-VIM 및 KL-VIM은 그렇지 못하여 가설 H1과 일치했습니다.
- **고차원 연속 제어 태스크 (Mujoco):**
  - RKL-VIM과 JS-VIM(GAIL)을 Mujoco 환경에서 테스트했습니다.
  - 수렴을 돕기 위해 생성기 손실에 실제 보상의 작은 비율을 혼합했습니다.
  - Ant, Hopper, HalfCheetah 환경에서 RKL-VIM이 JS-VIM(GAIL)보다 더 높은 평균 에피소드 보상으로 수렴했습니다. RKL-VIM은 "학습자가 방문하는 상태보다 전문가가 방문하는 상태에 더 높은 가중치"를 부여하여 수렴이 다소 느리지만 더 높은 보상으로 종료되었습니다.

## 🧠 Insights & Discussion

- 본 논문은 f-다이버전스를 기반으로 하는 모방 학습의 통합 프레임워크를 제시하며, 기존의 Behavior Cloning, GAIL, DAgger 등의 다양한 방법론을 포괄합니다.
- 다중 모드 시연 시, 역 KL 다이버전스(RKL)는 안전하고 효율적으로 모드 중 일부로 수렴하는 반면, KL 및 JS 다이버전스는 종종 안전하지 않은 보간된 행동을 생성합니다.
- **한계점:** 프레임워크는 다이버전스의 근사 추정치(하한)를 최소화하며, KL 다이버전스만이 샘플로부터 정확하게 측정될 수 있습니다. RKL의 경우, 최적 추정기가 발산할 수 있어 많은 샘플이 필요할 수 있습니다. 또한, 유한한 샘플 집합에서 f-다이버전스에 대한 엄격한 상한을 도출하기는 어렵습니다.
- **실용적 해결책:** "노이즈가 있는" 다이버전스 최소화(가우시안 노이즈 추가)를 고려하여 분포의 절대 연속성(absolute continuity)을 보장하고 다이버전스 크기의 상한을 설정할 수 있습니다. 또한, IL을 전문가 행동의 우도(likelihood)를 최대화하는 것과 학습자가 전문가가 방문하지 않는 상태를 방문하는 것을 제재하는 것 사이의 균형을 찾는 것으로 볼 수 있으며, 외부 정보(예: 장애물 제약)를 보조 페널티 항으로 활용할 수 있습니다.
- **향후 연구 방향:** 최대 엔트로피 모멘트 매칭(maximum entropy moment matching)과의 통합 및 측정 가능한 정의를 갖는 적분 확률 메트릭(Integral Probability Metrics, IPM) 클래스(MMD, Total Variation, Earth-mover's distance 등) 탐색을 제안합니다.

## 📌 TL;DR

**문제:** 기존 모방 학습(IL) 방법(Behavior Cloning, GAIL)은 다중 모드 전문가 시연에서 모드 간을 잘못 보간하여 안전하지 않은 정책을 생성합니다.

**제안 방법:** 본 논문은 학습자와 전문가 궤적 분포 간의 f-다이버전스 최소화를 위한 통합 IL 프레임워크를 제안합니다. 이 프레임워크는 기존 IL 알고리즘들을 다양한 f-다이버전스를 통해 재현할 수 있음을 보여줍니다. 특히, 다중 모드 시나리오에서 "모드 추구(mode-seeking)" 특성을 가진 역 KL 다이버전스(I-projection)를 활용하는 RKL-VIM 방법을 제안합니다.

**핵심 결과:** RKL-VIM은 다중 모드 행동을 단일 전문가 모드로 수렴시킴으로써 더 안전하고 강력하게 모방합니다. 반면, KL 및 JS 다이버전스는 "모드 커버링(mode-covering)" 경향으로 인해 보간된, 잠재적으로 안전하지 않은 정책을 만듭니다. 이러한 결과는 저차원 벤치마크와 고차원 Mujoco 환경 모두에서 경험적으로 검증되었습니다.
