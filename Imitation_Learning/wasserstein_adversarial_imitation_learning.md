# Wasserstein Adversarial Imitation Learning

Huang Xiao, Michael Herman, Joerg Wagner, Sebastian Ziesche, Jalal Etesami, Thai Hong Linh

## 🧩 Problem to Solve

모방 학습(Imitation Learning)은 전문가 시연(demonstrations)으로부터 전문가 정책(expert policy)을 복구하는 것을 목표로 합니다. 기존 접근 방식들은 다음과 같은 한계를 가집니다:

* **역강화 학습(Inverse Reinforcement Learning, IRL)의 한계:**
  * 샘플 효율성은 높지만, 문제에 특화된 보상 함수(reward function)나 (태스크) 특정 보상 함수 정규화(regularization)가 필요합니다.
  * IRL 문제는 종종 '잘못 제기된 문제(ill-posed)'로, 여러 보상 함수가 동일한 관찰된 행동을 설명할 수 있습니다.
  * 내부 루프에서 강화 학습(Reinforcement Learning, RL) 문제를 풀어야 하므로 고차원 연속 공간(high-dimensional continuous space)에서 비효율적입니다.
* **생성적 적대적 모방 학습(Generative Adversarial Imitation Learning, GAIL)의 한계:**
  * GAN 기반으로 정책을 직접 학습하여 IRL의 계산적 이점을 유지하지만, 수렴 시 판별자(discriminator)가 유효한 보상 함수로 해석될 수 없습니다 (모든 곳에서 $0.5$로 수렴).
  * 표준 GAN 훈련은 훈련 불안정성(training instabilities), 즉 모드 붕괴(mode collapse) 또는 기울기 소실(vanishing gradient)에 취약합니다.
  * Wasserstein GAN 기반의 목적 함수(objective)는 성능 향상을 보였지만, 이에 대한 이론적 정당성이 부족했습니다.

궁극적으로, 이 논문은 효율적이고 안정적인 모방 학습을 위해 일반적이고 바람직한 속성(예: 부드러움, smoothness)을 가진 보상 함수를 도출하는 방법의 필요성을 제기합니다.

## ✨ Key Contributions

* **이론적 정당화:** 견습 학습(apprenticeship learning)과 역강화 학습(IRL)이 Integral Probability Metrics (IPM), 특히 Wasserstein 거리(Wasserstein distance)와 자연스럽게 연결됨을 이론적으로 입증했습니다.
  * 이를 통해 보상 함수 공간(reward function space)을 넓히고, 부드러움과 같은 바람직한 속성을 가진 보상 함수를 사용할 수 있게 합니다.
  * 최적 운송(Optimal Transport, OT) 문제의 이중 형식(dual form)에 등장하는 칸토로비치 퍼텐셜(Kantorovich potentials)이 유효한 보상 함수로 해석될 수 있음을 보였습니다.
* **새로운 알고리즘 제안:** 이러한 관찰을 바탕으로 **Wasserstein Adversarial Imitation Learning (WAIL)**이라는 새로운 모방 학습 접근 방식을 제안합니다.
  * WAIL은 칸토로비치 퍼텐셜을 보상 함수로 간주하며, 대규모 애플리케이션을 위해 정규화된 최적 운송(regularized optimal transport)을 활용합니다.
* **실험적 우수성:** 여러 로봇 실험에서 WAIL이 기존 베이스라인(GAIL 및 행동 복제, Behavioral Cloning, BC)보다 평균 누적 보상(average cumulative rewards) 측면에서 우수한 성능을 보였습니다.
  * 특히 단 하나의 전문가 시연만으로도 상당한 샘플 효율성(sample-efficiency) 개선을 달성했습니다.
  * 학습된 보상 함수가 GAIL에 비해 훨씬 부드럽고 안정적임을 시각적으로 증명했습니다.

## 📎 Related Works

* **견습 학습(Apprenticeship Learning, AL) 및 시연으로부터 학습(Learning from Demonstration):** [4]
* **행동 복제(Behavioral Cloning, BC):** [6] 전문가 행동을 직접 모방하는 방법.
* **역강화 학습(Inverse Reinforcement Learning, IRL):** [25] 전문가 정책에서 미지의 보상 함수를 추정. [1, 2, 11, 19, 20, 21, 22, 30, 31, 37].
  * **최대 인과 엔트로피 IRL (Maximum Causal Entropy IRL, MCE-IRL):** [35, 36] 정책의 최적성 가정을 완화.
* **생성적 적대적 모방 학습(Generative Adversarial Imitation Learning, GAIL):** [16] GAN 기반으로 정책을 직접 학습하여 점유도 측정(occupancy measures)을 일치시킴.
  * **Adversarial IRL:** [12] 판별자를 유효한 보상 함수로 분리.
* **Wasserstein GAN (WGAN) 및 관련 연구:** [5, 15, 32, 33] Jensen-Shannon 발산 대신 Wasserstein 거리를 사용하여 GAN 훈련의 불안정성을 개선.
* **Integral Probability Metrics (IPM):** [29] 두 확률 분포 사이의 거리를 측정하는 일반적인 프레임워크.
* **최적 운송(Optimal Transport, OT) 이론 및 계산:** [7, 9, 10, 13, 23, 28, 34] Wasserstein 거리의 근간이며, 대규모 문제에 적용하기 위한 계산 방법론 개발.
* **신뢰 영역 정책 최적화(Trust Region Policy Optimization, TRPO):** [26] 정책 업데이트에 사용된 강화 학습 알고리즘. [27] Generalized Advantage Estimation와 함께.

## 🛠️ Methodology

1. **IRL과 Wasserstein 거리의 연결:**
    * 인과 엔트로피 정규화된 견습 학습 (Eq. 4)의 목적 함수 후반부가 Induced Occupancy Measures $\rho_E$와 $\rho_\pi$ 간의 Integral Probability Metric (IPM) $\phi_R(\rho_E, \rho_\pi)$으로 해석될 수 있음을 보입니다.
    * 특히, 보상 함수 공간 $R$을 상태-행동 공간에서 거리 함수 $d$에 대해 Lipschitz(1) 연속인 함수들의 클래스인 Lip(1)으로 선택하면, 이 IPM이 잘 알려진 1-Wasserstein 거리 $W_d^1(\rho_\pi, \rho_E)$가 됨을 증명합니다 (Proposition 3.1).
    * 이는 최적 운송 문제의 이중 형식(dual form)에 나타나는 칸토로비치 퍼텐셜(Kantorovich potential)을 보상 함수 $r(s,a)$로 해석함으로써 가능합니다.
2. **Wasserstein Adversarial Imitation Learning (WAIL) 알고리즘:**
    * **정규화된 최적 운송(Regularized Optimal Transport):** Lipschitz 연속성 제약 조건을 직접 적용하기 어렵기 때문에, OT 문제의 엔트로피 정규화 또는 $L_2$ 정규화 형태를 사용합니다 (Eq. 9).
        $$ W_d(\rho_\pi, \rho_E) = \sup_{r:S \times A \to R} E_{y \sim \rho_E}[r(y)] - E_{x \sim \rho_\pi}[r(x)] + E_{(x,y) \sim \rho_\pi \times \rho_E}[\Omega_{d,\epsilon}(r,x,y)] $$
        여기서 $\Omega_{d,\epsilon}(r,x,y)$는 엔트로피 정규화 또는 $L_2$ 정규화 항입니다.
    * **보상 함수 업데이트 (칸토로비치 퍼텐셜 학습):** 심층 신경망으로 매개변수화된 보상 함수 $r_w$를 stochastic ascent 방식으로 업데이트하여 위 목적 함수를 최대화합니다. 이는 판별자(discriminator)의 역할과 유사합니다. 지상 비용(ground cost)으로 상태-행동 공간의 유클리드 거리 $d(x,y) = \|y-x\|$를 사용하며, $L_2$ 정규화가 더 안정적이어서 주로 사용됩니다.
    * **정책 업데이트 (생성자 역할):** 이전 단계에서 추정된 보상 $\hat{r}$을 고정 보상으로 간주하고, 정책 $\pi_\theta$의 인과 엔트로피 $H(\pi_\theta)$를 정규화 요소 $\lambda$로 포함하는 정책 기울기(policy gradient) 단계 (Eq. 10)를 수행합니다.
        $$ \nabla_\theta \pi_\theta = \nabla_\theta E_{\rho_{\pi_\theta}}[\hat{r}(s,a)] + \lambda \nabla_\theta H(\pi_\theta) $$
        이 정책 업데이트는 KL 제약(KL-constrained)이 있는 자연 기울기 단계(natural gradient step)인 Trust Region Policy Optimization (TRPO) [26]를 통해 이루어집니다.
    * 알고리즘 (1)은 적대적 학습 스타일로 보상과 정책을 반복적으로 업데이트하며, Theorem 4.1을 통해 최적 해 $(r^*, \pi^*)$로 수렴함을 이론적으로 증명합니다.

## 📊 Results

* **실험 환경:** OpenAI Gym의 고전 제어 태스크(Cartpole, Mountaincar, Acrobot)와 고차원 MuJoCo 환경(Hopper, Walker2d, HalfCheetah, Ant, Humanoid, Reacher) 총 9가지.
* **베이스라인:** GAIL과 Behavior Cloning (BC).
* **주요 발견:**
  * **WAIL의 압도적인 성능:** WAIL은 거의 모든 학습 태스크에서 GAIL과 BC를 능가했습니다. 특히, 단 하나의 전문가 궤적(trajectory)만으로도 전문가 행동에 근접하는 **탁월한 샘플 효율성**을 보였습니다.
  * **고차원 환경에서의 강점:** MuJoCo 환경에서는 BC가 충분한 시연이 있을 때만 전문가 행동을 모방했고, GAIL은 BC보다 나은 결과를 보였습니다. 하지만 WAIL은 대부분의 샘플 크기에서, 심지어 단 하나의 전문가 시연으로도 다른 방법들을 압도했습니다.
  * **Humanoid 태스크의 특이점:** 데이터 크기 증가 시 Humanoid 태스크에서 성능 하락이 관찰되었는데, 이는 전문가 정책의 높은 분산 때문일 수 있다고 분석했습니다.
  * **Reacher 태스크 (고난이도):** WAIL과 BC는 다양한 데이터 크기에 걸쳐 일관된 성능을 보였으나, GAIL은 전문가 성능을 달성하기 위해 80개 이상의 시연이 필요했습니다.
* **보상 표면 시각화:** Humanoid 환경에서의 보상 표면 시각화 결과 (Fig. 2)는 WAIL의 핵심 장점을 보여줍니다.
  * **WAIL의 부드러운 보상 함수:** WAIL이 학습한 보상 함수는 데이터 크기에 관계없이 GAIL보다 훨씬 부드러운 보상 표면을 형성했습니다. 전문가 상태-행동 방향을 따라 높은 보상을 할당하며, 적은 데이터에서도 일관된 패턴을 보이다가 데이터가 많아질수록 더욱 부드럽고 잘 정의된 보상 함수를 생성합니다.
  * **GAIL 판별자의 문제점:** GAIL의 판별자는 데이터가 적을 때 포화되어 전문가 모방에 실패하고, 데이터가 많아져도 거의 모든 곳에서 상수 보상을 할당하여 이후 정책 업데이트에 유용하지 않았습니다.
  * WAIL은 Wasserstein 거리의 기하학적 속성 덕분에 전문가 지원 영역(expert support)을 벗어나는 상태-행동에 대해서도 보상 점수가 잘 정의됩니다.

## 🧠 Insights & Discussion

* **이론적 기반의 강점:** 이 연구는 견습 학습, 최적 운송, IRL 간의 자연스러운 연결성을 이론적으로 정당화함으로써, 상태-행동 공간에서 부드러운 보상 함수를 선택할 수 있는 기반을 마련했습니다.
* **유효하고 유용한 보상 함수:** WAIL은 모델 프리(model free) 방식임에도 불구하고, 학습된 이중 함수(dual function)인 칸토로비치 퍼텐셜이 훈련 후에도 유효하고 유용한 보상 함수로 기능합니다. 이는 수렴 시 쓸모없게 되는 GAIL의 판별자와 대비됩니다.
* **뛰어난 샘플 효율성:** 최적 운송 문제의 최적화 특성과 부드러운 보상 함수 덕분에 WAIL은 극히 적은 수의 전문가 시연만으로도 전문가 행동을 달성하는 뛰어난 샘플 효율성을 보여주었습니다.
* **향후 연구 방향:**
  * WAIL의 샘플 복잡성(sample complexity)에 대한 심층 분석.
  * **지상 비용(ground cost) 활용:** 상태-행동 공간에 대한 도메인 지식(domain knowledge)을 인코딩하는 적절하게 정의된 사전(prior)으로 지상 비용을 강화하거나, 이를 동시에 추정하는 연구.
  * **표현력이 풍부한 보상 함수:** 신경망으로 매개변수화된 부드러운 보상 함수가 더욱 표현력이 풍부한 보상 함수군을 나타낼 수 있음을 활용.
  * **보상 함수의 전이 가능성:** 최적 운송이 본질적으로 상태-행동 공간에서 운송 맵(transportation map)을 학습하므로, 이러한 맵을 추정할 수 있다면 보상 함수를 다른 학습 태스크로 전이(transfer)할 수 있는 가능성.

## 📌 TL;DR

**문제:** 기존 모방 학습(IRL, GAIL)은 보상 함수 정의의 어려움, 샘플 비효율성, 훈련 불안정성, 유효한 보상 함수 부재 등의 한계를 가지고 있었다.

**제안 방법:** 이 논문은 견습 학습, 역강화 학습(IRL), 최적 운송(Optimal Transport) 간의 이론적 연결을 규명하고, Wasserstein 거리에 기반한 새로운 모방 학습 알고리즘인 **Wasserstein Adversarial Imitation Learning (WAIL)**을 제안한다. WAIL은 최적 운송의 이중 형식에 나타나는 칸토로비치 퍼텐셜을 부드럽고 유효한 보상 함수로 활용하며, 정규화된 최적 운송을 통해 대규모 적용을 가능하게 한다.

**주요 결과:** WAIL은 다양한 로봇 제어 태스크에서 기존 베이스라인(GAIL, BC) 대비 우수한 성능을 보였으며, 특히 단 하나의 전문가 시연만으로도 전문가 행동에 근접하는 탁월한 샘플 효율성을 입증했다. 또한, WAIL이 학습한 보상 함수는 GAIL보다 훨씬 부드럽고 안정적임을 보상 표면 시각화를 통해 보여주었다.
