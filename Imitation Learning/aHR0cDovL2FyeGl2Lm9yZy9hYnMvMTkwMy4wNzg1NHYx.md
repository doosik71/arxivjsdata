# Hindsight Generative Adversarial Imitation Learning

Naijun Liu, Tao Lu, Yinghao Cai, Boyao Li, and Shuo Wang (2019)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)의 핵심적인 제약 사항인 **전문가 시연 데이터(Expert Demonstration Data)의 의존성 문제**를 해결하고자 한다. 일반적으로 모방 학습은 전문가의 데이터를 통해 제어 정책(Control Policy)을 효율적으로 학습할 수 있는 강력한 패러다임이지만, 실제 환경에서 고품질의 전문가 데이터를 수집하는 것은 매우 비용이 많이 들고 노동 집약적인 작업이다.

특히, 보상 함수(Reward Function)를 설계하기 어려운 복잡한 작업에서 전문가 데이터 없이 정책을 학습시키는 것은 매우 도전적인 과제이다. 따라서 본 연구의 목표는 **전문가 시연 데이터가 전혀 없는 상황에서도 성공적으로 모방 학습을 수행할 수 있는 알고리즘을 개발**하는 것이며, 이를 통해 모방 학습의 적용 범위를 획기적으로 확장하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Hindsight(사후 분석)** 개념을 **Generative Adversarial Imitation Learning (GAIL)** 프레임워크에 결합하는 것이다.

중심적인 직관은 에이전트가 환경과 상호작용하며 생성한 일반적인 궤적(Rolled-out Trajectories)을 **Hindsight Transformation** 기법을 통해 '전문가와 유사한(Expert-like)' 데이터로 변환하여 자가 합성(Self-synthesize)하는 것이다. 이를 통해 외부의 전문가 데이터 없이도 생성자와 판별자 간의 적대적 학습을 가능하게 한다. 또한, 학습 과정에서 생성자의 수준이 올라감에 따라 합성되는 전문가 유사 데이터의 수준도 함께 높아지는데, 이는 결과적으로 **커리큘럼 학습(Curriculum Learning)** 메커니즘을 내재적으로 구현하게 되어 학습의 안정성과 효율성을 높인다.

## 📎 Related Works

본 논문에서는 다음과 같은 관련 연구들을 소개하고 차별점을 제시한다.

1. **Imitation Learning (IL):**
    * **Behavior Cloning (BC):** 지도 학습 방식으로 단순하지만, 데이터가 많이 필요하며 Covariate Shift로 인한 오차 누적 문제가 발생한다.
    * **Inverse Reinforcement Learning (IRL):** 전문가 데이터로부터 보상 함수를 추론하여 정책을 학습한다.
    * **GAIL:** GAN의 구조를 차용하여 생성자(정책)가 전문가의 샘플을 모사하도록 학습시키며, 보상 함수를 직접 설계하지 않아도 된다는 장점이 있다.

2. **Hindsight Experience Replay (HER):**
    * 강화 학습에서 희소 보상(Sparse Reward) 문제를 해결하기 위해, 실패한 궤적이라도 도달한 상태를 목표(Goal)로 재설정함으로써 성공한 경험으로 변환하여 학습하는 기법이다.

3. **Learning with Few Data & Self-Imitation:**
    * 최근 적은 양의 데이터로 학습하거나, 외부 데이터 없이 과거의 좋은 경험을 모방하는 Self-Imitation Learning 연구들이 진행되었다. 특히 GASIL은 과거의 상위 $K$개 궤적을 긍정 샘플로 사용한다. 본 논문의 HGAIL은 단순히 과거의 좋은 궤적을 선택하는 대신, Hindsight 개념을 통해 데이터를 직접 변환하여 전문가 유사 데이터를 생성한다는 점에서 GASIL과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

HGAIL은 정책 $\pi_\theta$(생성자)와 판별자 $D_\omega$로 구성된다. 기본적으로 GAIL의 목적 함수를 따르며 다음과 같은 수식을 기반으로 한다.

$$\min_{\theta} \max_{\omega} L(\theta, \omega) = \mathbb{E}_{\pi_\theta} [\log(D_\omega(s, a))] + \mathbb{E}_{\pi_E} [\log(1 - D_\omega(s, a))] - \lambda H(\pi_\theta)$$

여기서 $\pi_E$는 전문가 정책이며, $H(\pi_\theta)$는 탐색을 장려하기 위한 정책의 엔트로피이다. HGAIL의 핵심은 데이터가 없는 상황에서 $\pi_E$의 역할을 대신할 $\tau^h$(Hindsight transformed trajectories)를 생성하는 것이다.

### 2. Hindsight Transformation (전문가 유사 데이터 합성)

에이전트가 생성한 일반 궤적 $\tau = \{\tau_0, \tau_1, \dots, \tau_N\}$를 다음 절차에 따라 전문가 유사 궤적 $\tau^h$로 변환한다.

* 각 궤적 $\tau_i$의 모든 타임 스텝 $t$에 대해 확률 $p_{ht}$로 변환 여부를 결정한다.
* 선택된 타임 스텝 $j$에 대해, 해당 시점 이후($j$부터 $T$까지)에 실제로 도달한 임의의 위치 $p_l$을 새로운 목표(Goal)로 설정한다.
* 즉, "원래 목표는 달성하지 못했더라도, 나중에 도달한 그 지점을 목표로 삼았다면 성공한 궤적이다"라는 관점에서 데이터를 변환하는 것이다.

### 3. 학습 절차 및 손실 함수

**가. 생성자(정책) 최적화:**
PPO(Proximal Policy Optimization) 알고리즘을 사용하여 판별자가 주는 보상을 최대화하도록 학습한다.
$$\min_{\theta} \mathbb{E}_{\tau} [\log D_\omega(s, a)] - \lambda \frac{1}{H(\pi_\theta)}$$

**나. 판별자 최적화:**
판별자는 일반 궤적 $\tau$(부정 샘플)와 Hindsight 변환된 궤적 $\tau^h$(긍정 샘플)를 구분하도록 학습한다.
$$\max_{\omega} \mathbb{E}_{\tau} [\log(D_\omega(s, a))] + \mathbb{E}_{\tau^h} [\log(1 - D_\omega(s, a))]$$

**다. 전체 파이프라인:**

1. 무작위 가중치로 $\tau_0$ 생성 $\to$ $\tau^h_0$ 합성.
2. $\tau^h_0$를 이용해 생성자를 MLE(Maximum Likelihood Estimation)로 사전 학습.
3. $\tau_0$와 $\tau^h_0$를 이용해 판별자를 사전 학습.
4. 생성자와 판별자를 교대로 업데이트하며 수렴할 때까지 반복한다.

## 📊 Results

### 1. 실험 설정

* **작업:** Fetch 로봇의 Target Reaching(목표 지점 도달) 및 Target Grasping(물체 잡기).
* **조건:** 매우 어려운 **Binary Sparse Reward** 환경 (성공 시 0, 그 외 -1).
* **비교 대상:** GAIL-demo(전문가 데이터 있음), PPO, GASIL, HGAIL-no(Hindsight 변환 제외).
* **평가 지표:** 성공률(Success Rate) 및 거리 오차(Distance Error).

### 2. 주요 결과

* **정량적 결과:** HGAIL은 전문가 데이터가 없음에도 불구하고, 전문가 데이터를 사용한 GAIL-demo와 유사한 수준의 성능을 보였다.
* **비교 분석:**
  * **PPO:** Sparse Reward 환경에서 단독으로는 학습에 실패하였다.
  * **GASIL:** HGAIL보다 수렴 속도가 느리고 최종 성능이 낮았다.
  * **HGAIL-no:** Hindsight 변환이 없을 경우 성능이 급격히 저하되어, 본 기법이 핵심적임을 입증하였다.

### 3. Ablation Study

* **Curriculum Learning:** 초기 데이터 $\tau_0$로만 전문가 데이터를 고정해서 사용하는 것보다, 학습 과정에서 생성되는 최신 데이터를 계속 변환하여 사용하는 것이 훨씬 안정적이고 높은 성능을 보였다.
* **Transformation 전략:** 마지막 도달 지점만 사용하는 'Final Hindsight'보다, 이후의 임의 지점을 사용하는 'Future Hindsight'가 더 효과적이었다.
* **확률 $p_{ht}$:** 변환 확률이 1일 때 가장 좋은 성능을 보였다.
* **보상 함수:** $r_1(s, a) = -\log(1 - \text{sig}(\text{dis}(s, a)))$ 형태의 보상 함수가 가장 빠른 수렴과 높은 성능을 보였다.

### 4. Sim-to-Real Transfer

시뮬레이션에서 학습된 정책을 실제 UR5 로봇에 직접 적용한 결과, 추가 학습 없이도 높은 성공률과 낮은 거리 오차로 Reaching 및 Picking 작업을 성공적으로 수행하였다.

## 🧠 Insights & Discussion

본 논문은 전문가 데이터 없이도 모방 학습을 수행할 수 있다는 가능성을 제시하였다. 특히, 단순히 데이터를 증강하는 것이 아니라 **Hindsight Transformation을 통해 '성공의 정의'를 재설정**함으로써 에이전트가 스스로 학습 가이드를 생성하게 만든 점이 인상적이다.

또한, 생성자가 발전함에 따라 합성되는 전문가 유사 데이터의 질이 점진적으로 향상되는 구조는, 명시적으로 설계되지 않았음에도 불구하고 매우 효과적인 **내재적 커리큘럼 학습(Implicit Curriculum Learning)**으로 작용하였다. 이는 복잡한 작업에서 학습 초기 단계의 불안정성을 줄이는 핵심 요인이 된다.

다만, 본 연구는 상대적으로 단순한 로봇 조작 작업(Reaching, Grasping)에 집중되어 있다. 더 복잡한 계층적 작업이나 매우 긴 시퀀스의 작업에서도 이 방식이 유효할지는 추가적인 연구가 필요해 보인다. 또한, $p_{ht}=1$이 최적이라는 결과는 HER의 일반적인 설정과 다소 차이가 있어, 적대적 학습 환경에서의 데이터 분포 특성에 대한 더 깊은 분석이 요구된다.

## 📌 TL;DR

* **핵심 기여:** 전문가 시연 데이터 없이 모방 학습을 가능하게 하는 HGAIL 알고리즘 제안.
* **방법론:** 에이전트의 궤적을 Hindsight Transformation을 통해 '전문가 유사 데이터'로 자가 합성하고, 이를 GAIL의 적대적 학습 구조에 활용.
* **결과:** 전문가 데이터를 사용한 GAIL에 근접하는 성능을 보였으며, 내재된 커리큘럼 학습 효과로 학습 안정성을 확보함. 실제 UR5 로봇으로의 Sim-to-Real 전이 성공.
* **의의:** 데이터 수집 비용이 높은 환경에서 모방 학습의 진입 장벽을 낮추었으며, 향후 복잡한 조작 기술 학습 및 계층적 강화 학습으로의 확장 가능성이 높음.
