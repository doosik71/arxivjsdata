# Domain-Adversarial and $\lambda$-Conditional State Space Model for Imitation Learning

Ryo Okumura, Masashi Okada, and Tadahiro Taniguchi (2021)

## 🧩 Problem to Solve

본 논문은 부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Processes, POMDPs) 환경에서 로봇 제어를 위한 상태 표현 학습(State Representation Learning, SRL) 시 발생하는 **도메인 시프트(Domain Shift)** 문제를 해결하고자 한다.

모방 학습(Imitation Learning)에서 전문가(Expert)의 데이터와 에이전트(Agent)의 데이터 사이에는 외관, 배경, 카메라 각도, 혹은 로봇과 인간의 손가락 차이와 같이 제어 성능과는 무관하지만 시각적으로 뚜렷한 도메인 시프트가 존재한다. 기존의 SRL 방법론들은 이러한 도메인 의존적 정보(Domain-dependent information)를 제거하지 못하며, 이로 인해 모방 학습의 보상 함수 역할을 하는 판별자(Discriminator)가 제어와 무관한 외형적 특징에 집중하게 되어 학습 효율이 저하되는 문제가 발생한다. 따라서 본 연구의 목표는 도메인에 구애받지 않으면서도(Domain-agnostic), 작업의 특성과 시스템의 동역학을 충분히 반영하는(Task- and dynamics-aware) 상태 표현을 학습하는 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **DAC-SSM(Domain-Adversarial and $\lambda$-Conditional State Space Model)**이라는 새로운 상태 공간 모델을 제안한 것이다. 이 모델의 중심 설계 아이디어는 다음과 같다.

1. **도메인 적대적 학습(Domain-Adversarial Training):** 도메인 판별자(Domain Discriminator)를 도입하고, 상태 표현이 어느 도메인에서 왔는지 판별하지 못하도록 하는 도메인 혼동 손실(Domain Confusion Loss)을 추가하여 도메인 불변 특징을 추출한다.
2. **도메인 조건부 구조(Domain-Conditional Structure):** Conditional VAE(CVAE)의 아이디어를 차용하여, 인코더와 디코더가 도메인 레이블($y$)을 조건으로 입력받게 함으로써 도메인 의존적 정보와 제어 관련 정보를 분리(Disentanglement)한다.
3. **통합 최적화:** 상태 추론, 관측치 재구성, 순방향 동역학, 보상 모델을 함께 최적화하여 도메인 불변성을 유지하면서도 제어에 필수적인 동역학 정보를 보존한다.

## 📎 Related Works

본 논문에서 언급하는 관련 연구와 그 한계는 다음과 같다.

* **State Representation Learning (SRL):** RSSM(Recurrent State Space Model) 기반의 PlaNet과 같은 연구들이 고차원 이미지 데이터에서 효율적인 상태 표현을 학습하여 계획(Planning) 성능을 높였으나, 이러한 방법들은 도메인 시프트가 존재하는 상황에서 도메인 의존적 노이즈를 제거하는 메커니즘이 부족하다.
* **Domain-Agnostic Feature Representation:** 도메인 적대적 학습이나 CVAE를 통한 특징 분리 연구들이 존재하며, 일부는 Sim-to-Real 전이에 사용되었다. 하지만 이를 POMDP 환경의 순차적 상태 공간 모델 및 모방 학습과 직접적으로 결합한 사례는 드물다.
* **Imitation Learning:** GAIL과 같은 적대적 모방 학습이 제안되었으나, 관측치 수준에서의 도메인 시프트가 클 경우 판별자가 작업 수행 능력보다는 외형적 차이에 반응하여 부적절한 보상을 제공하는 한계가 있다.
* **Imitation Learning from Observation (IfO):** 행동(Action) 데이터 없이 관측치만으로 학습하는 방식들이 제안되었으나, 대부분 수작업으로 설계된 특징(Hand-designed features)에 의존하거나 정교한 제어 작업에서는 성능이 떨어진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 상태 공간 모델

DAC-SSM은 POMDP를 기반으로 하며, 역사적 정보를 전달하기 위해 다음과 같은 혼합 결정론적/확률적 동역학을 가진 RSSM 구조를 따른다.

* **전이 모델(Transition model):** $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$ (RNN으로 구현)
* **상태 모델(State model):** $s_t \sim p(s_t | h_t)$
* **관측 모델(Observation model):** $o_t \sim p(o_t | h_t, s_t, y)$

여기서 $h_t$는 결정론적 상태, $s_t$는 확률적 상태, $o_t$는 이미지 관측치, $a_t$는 연속적 행동, $y$는 도메인 레이블이다. 특히 관측 모델(디코더)과 상태 추론 모델(인코더)은 도메인 레이블 $y$를 조건으로 입력받아 도메인별 특성을 분리한다.

### 2. 손실 함수 및 학습 절차

모델은 다음의 손실 함수들을 통해 공동 최적화된다.

**A. RSSM 손실 ($L_{RSSM}$):**
관측치 시퀀스의 로그 가능도를 최대화하기 위해 ELBO(Evidence Lower Bound)를 사용한다.
$$L_{RSSM} = - \sum_{t=1}^{T} \mathbb{E}_{q(s_t|...)} [\ln p(o_t|h_t, s_t, y)] + \mathbb{E}_{q(s_{t-1}|...)} [KL(q(s_t|...)||p(s_t|h_t))]$$

**B. 도메인 판별자 및 혼동 손실 ($L_{D_d}$):**
도메인 판별자 $D_d$는 상태 $h_t$가 에이전트 도메인인지 전문가/초보자 도메인인지 구분한다.
$$L_{D_d} = 2 \times \mathbb{E}_{h_t \sim \mathcal{B}_A} [\ln D_d(h_t)] + \mathbb{E}_{h_t \sim \mathcal{B}_E} [\ln(1 - D_d(h_t))] + \mathbb{E}_{h_t \sim \mathcal{B}_N} [\ln(1 - D_d(h_t))]$$
최종 상태 공간 학습 손실은 다음과 같이 정의되어, 판별자를 속이는 방향으로 학습된다.
$$L_{DAC} = L_{RSSM} - \lambda L_{D_d}$$

**C. 최적성 판별자 ($D_O$):**
모방 보상을 제공하기 위해, 상태-행동 쌍 $(h_t, a_t)$가 전문가의 것인지 판별하는 $L_{D_O}$를 학습한다.

### 3. 계획 알고리즘 (Planning)

학습된 상태 공간 위에서 MPC(Model Predictive Control)를 수행하며, 최적의 행동 시퀀스를 찾기 위해 CEM(Cross Entropy Method)을 사용한다. 목적 함수는 다음과 같이 작업 보상($r_t$)과 모방 보상(최적성 판별자 $D_O$의 출력)의 합을 최대화하는 것이다.
$$\text{Objective} = \sum_{t=1}^{H} [\mathbb{E}_{p(r_t|h_t, s_t)}[r_t] + \ln D_O(h_t, a_t)]$$

## 📊 Results

### 1. 실험 설정

* **환경:** MuJoCo 시뮬레이터의 세 가지 작업 (Cup-Catch, Finger-Spin, Connector-Insertion).
* **도메인 시프트 설정:** 색상, 배경, 시점(Tilted view), 그리고 전문가(인간 손가락) vs 에이전트(로봇 손가락)의 차이를 부여하였다.
* **비교 대상:** PlaNet (기존 SRL), PlaNet + $D_O$ (단순 모방 보상 추가), DA-SSM (적대적 학습만 적용), DC-SSM (조건부 구조만 적용).

### 2. 정량적 결과

표 I와 그림 6에 따르면, 제안된 **DAC/dual**(작업 보상과 모방 보상을 모두 사용한 DAC-SSM)이 모든 작업에서 베이스라인보다 월등히 높은 성능을 보였다. 특히 Connector-Insertion 작업에서는 베이스라인들이 거의 성공하지 못한 반면, DAC-SSM은 전문가 수준에 근접한 성능을 달성하였다.

### 3. 정성적 결과 및 분석

* **재구성 실험 (Fig 7):** DAC-SSM의 상태 표현에서 도메인 레이블 $y$를 바꾸어 이미지를 재구성했을 때, 관절 각도와 같은 제어 관련 정보는 유지되지만 바닥이나 물체의 색상은 레이블에 따라 변하는 것을 확인하였다. 이는 상태 표현이 도메인 불변적임을 시각적으로 입증한다.
* **하이퍼파라미터 $\lambda$ 영향:** $\lambda$ 값이 커질수록(적대적 학습이 강해질수록) Connector-Insertion과 같은 작업에서 성능이 향상되는 경향을 보였으며, 이는 도메인 의존적 정보를 제거하는 것이 실제로 중요하다는 것을 의미한다.

## 🧠 Insights & Discussion

**강점:**
본 논문은 단순히 적대적 학습만 사용하거나 단순히 조건부 인코더/디코더만 사용하는 것보다, 두 가지를 결합했을 때 가장 강력한 도메인 불변 상태 표현이 학습됨을 보였다. 특히, RSSM의 동역학 학습과 도메인 불변성 학습을 결합함으로써, "제어에 유용한 정보"와 "도메인에 특화된 정보"를 성공적으로 분리해냈다.

**한계 및 논의사항:**

1. **$\lambda$의 작업 의존성:** 실험 결과 $\lambda$ 값에 따라 성능 차이가 발생하는데, 이는 각 작업마다 제거해야 할 도메인 정보의 양이 다르기 때문으로 추정된다. $\lambda$를 동적으로 조절하는 메커니즘이 필요할 수 있다.
2. **모달리티의 차이:** 본 연구는 시각적 외형의 차이(색상, 각도 등)에 집중하였으나, 데이터의 모달리티 자체가 완전히 다른 경우(예: 비디오 $\rightarrow$ 텍스트)에도 적용 가능할지는 미지수이다.
3. **실제 로봇 적용:** 시뮬레이션 환경에서는 성공적이었으나, 실제 로봇 환경의 훨씬 큰 불확실성과 노이즈를 처리하기 위해서는 완전히 확률적인 상태 표현(Fully stochastic state representation)에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 전문가와 에이전트 간의 외형적 차이(도메인 시프트)가 모방 학습의 성능을 저하시키는 문제를 해결하기 위해, **도메인 적대적 학습과 조건부 인코더/디코더를 결합한 DAC-SSM**을 제안하였다. 이를 통해 도메인에 무관하면서도 제어에 필수적인 동역학 정보를 포함하는 상태 표현을 학습하였으며, MuJoCo 시뮬레이션의 다양한 작업에서 기존 SRL 방식 대비 2배 이상의 성능 향상을 달성하였다. 이 연구는 향후 Sim-to-Real 전이나 인간의 시연을 통한 로봇 학습에서 도메인 간극을 줄이는 핵심적인 방법론이 될 가능성이 높다.
