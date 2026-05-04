# Burn-In Demonstrations for Multi-Modal Imitation Learning

Alex Kuefler, Mykel J. Kochenderfer (2017)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 **Multi-modal Imitation Learning(다중 모드 모방 학습)** 환경에서 전문가의 다양한 스타일(Style)을 일관성 있게 재현하는 것이다. 특히 자율주행 상황에서 운전자의 주행 스타일은 공격성, 주의 집중도 등 측정하기 어려운 잠재적 요인(Latent factors)에 의해 결정되는데, 기존의 접근 방식들은 다음과 같은 한계를 가진다.

1.  **불안정한 정책 생성**: 지도 학습(Supervised Learning) 기반의 모방 학습은 작은 예측 오류가 누적되어 결과적으로 정책이 불안정해지는 문제(Cascading errors)가 발생한다.
2.  **스타일 일관성 부족**: InfoGAIL과 같은 기존의 다중 모드 모방 학습 알고리즘은 매 시도(Trial)마다 잠재 코드(Latent code) $z$를 무작위로 샘플링한다. 이로 인해 실제 운전자의 초기 상태(속도, 방향 등)가 주어졌음에도 불구하고, 샘플링된 $z$가 실제 운전자의 스타일과 일치하지 않아 이후의 행동이 일관되지 않은 결과를 초래한다.

따라서 본 논문의 목표는 전문가의 부분적인 궤적(Trajectory)을 통해 잠재적인 주행 스타일을 추론하고, 이를 바탕으로 장기적인 시계열(Long time horizons)에서도 전문가의 행동을 안정적으로 모방하는 **Burn-InfoGAIL** 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 **Burn-in Demonstration**이라는 개념을 도입하는 것이다. 이는 정책이 실제 제어를 시작하기 전, 전문가가 수행한 짧은 구간의 데이터(Burn-in)를 먼저 관찰하고, 이 데이터를 조건(Condition)으로 하여 해당 전문가의 잠재 코드 $z$를 추론하는 방식이다.

핵심 직관은 "운전자의 스타일은 주행 시작 직후의 짧은 행동 패턴 속에 이미 녹아 있다"는 점이다. 이를 통해 무작위 샘플링이 아닌, 관찰된 데이터에 기반한 스타일 추론이 가능해지며, 결과적으로 실제 전문가의 주행 스타일과 일관성을 유지하는 정책을 생성할 수 있다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 차별점을 제시한다.

-   **VAE (Variational Autoencoders)**: 인간의 시연 데이터에서 잠재 임베딩을 발견하는 데 사용되었다. 하지만 행동 복제(Behavioral Cloning) 방식에 의존하기 때문에 시뮬레이터 없이 학습할 경우 누적 오류로 인해 경로를 이탈하는 등 불안정한 정책을 생성하는 한계가 있다.
-   **GAIL (Generative Adversarial Imitation Learning)**: 전문가와 정책의 행동을 구분하는 판별자(Discriminator)를 통해 보상 함수를 학습한다. 하지만 GAIL은 기본적으로 단일 모드(Unimodal) 행동을 학습하려 하므로, 다양한 주행 스타일을 가진 다중 모드 데이터를 처리하기 어렵다.
-   **InfoGAIL**: GAIL에 상호 정보량(Mutual Information) 최대화를 추가하여 잠재 요인을 발견한다. 그러나 앞서 언급했듯이 잠재 코드를 무작위로 샘플링하기 때문에, 특정 전문가의 초기 상태에서 시작하는 시나리오에서 스타일 일관성을 보장할 수 없다.

## 🛠️ Methodology

### 전체 시스템 구조
본 논문은 운전 스타일을 Dynamic Bayesian Network 모델로 정의한다. 전체 프로세스는 두 단계로 나뉜다.
1.  **Burn-in 단계**: 전문가 정책 $\pi_E$가 스타일 $z$에 따라 $T$ 시간 동안 주행하며 궤적 $\tau = (s_0, a_0, \dots, s_T, a_T)$를 생성한다.
2.  **Rollout 단계**: 학습된 정책 $\pi_\theta$가 $\tau$로부터 추론된 $z$를 조건으로 하여 $s_{T+1}$부터 행동을 생성한다.

### 주요 구성 요소 및 학습 목표
시스템은 정책 네트워크 $\pi_\theta$, 판별자(Critic) $D_\omega$, 그리고 추론 모델(Inference model) $Q_\psi$로 구성된다.

#### 1. Imitation Learning (모방 학습)
전문가의 행동을 모방하기 위해 Wasserstein GAIL의 목적 함수를 사용한다.
$$W(\theta, \omega) = \mathbb{E}_{a \sim \pi_\theta(\cdot|s)} [D_\omega(s, a)] - \mathbb{E}_{a \sim \pi_E(\cdot|s)} [D_\omega(s, a)]$$
판별자 $D_\omega$의 출력값은 대리 보상 함수(Surrogate reward) $\tilde{r}(s, a)$로 사용되며, 이는 다음과 같이 정의되어 항상 양수 값을 갖게 함으로써 조기 종료(충돌 등)를 방지한다.
$$\tilde{r}(s, a) = \log(1 + e^{D_\psi(s, a)})$$

#### 2. Information Maximization (정보 최대화)
정책과 추론 모델 간의 상호 정보량(Mutual Information)을 최대화하여 잠재 코드 $z$가 실제 스타일을 잘 반영하게 한다.
$$I^q(z; s, a) = H(Q_\psi(z')) - C(\theta, \psi)$$
여기서 $C(\theta, \psi)$는 시도 시작 시 샘플링된 $z'$와 시도 종료 시 $Q_\psi$가 예측한 코드 간의 교차 엔트로피(Cross Entropy) 오차이다. $H(Q_\psi(z'))$는 잠재 코드의 엔트로피로, 특정 코드 하나로 붕괴되는 현상(Mode collapse)을 막기 위해 최대화한다.

### 최종 목적 함수
최종적으로 Burn-InfoGAIL은 다음의 통합 목적 함수를 최적화한다.
$$\min_\theta \max_{\omega, \psi} W(\theta, \omega) - C(\theta, \psi) + \lambda H(\hat{\mathbb{E}}_\tau [Q_\psi(z'|\tau)])$$
-   $W(\theta, \omega)$: 전문가 데이터 모방 유도.
-   $C(\theta, \psi)$: 행동을 통해 스타일 $z$를 예측 가능하게 함.
-   $\lambda H(\dots)$: 모든 잠재 코드가 균등하게 샘플링되도록 보장.

### 학습 절차 및 아키텍처
-   **아키텍처**: 모든 모델은 $\tanh$ 활성화 함수를 사용하는 다층 퍼셉트론(MLP)이다. 잠재 코드 $z$는 선형 임베딩을 통해 정책 네트워크의 후반부 은닉층에 결합(Concatenation)된다.
-   **최적화**: 
    -   정책 $\pi_\theta$: TRPO(Trust Region Policy Optimization)를 사용한다.
    -   추론 모델 $Q_\psi$: Adam 최적화 도구를 사용하여 분류 작업처럼 학습한다.
    -   판별자 $D_\omega$: K-Lipschitz 속성을 유지하기 위해 RMSProp을 사용한다.

## 📊 Results

### 실험 설정
-   **환경**: 타원형 레이스 트랙 시뮬레이션.
-   **전문가 스타일**: 4가지 클래스(Aggressive, Passive, Speeder, Tailgating)로 구분된다.
-   **데이터**: 960개의 학습 시연, 480개의 검증 시연 (각 5초/50타임스텝).
-   **지표**: Adjusted Mutual Information (AMI)를 통해 스타일 클러스터링 성능을 측정하고, RMSE를 통해 전문가 궤적과의 오차를 측정한다.

### 주요 결과
1.  **스타일 추론 성능 (AMI)**: Burn-InfoGAIL은 검증 세트에서 **0.38의 AMI**를 기록하여, InfoGAIL(0.16)이나 VAE+K-Means(0.24)보다 월등히 높은 성능을 보였다.
2.  **주행 안정성**: '위험 이벤트(Offroad, Collision, Reversal)' 발생 빈도에서 Burn-InfoGAIL은 특히 경로 이탈(Offroad) 비율을 VAE(0.756)나 GAIL(0.165)보다 현저히 낮게(0.074) 유지하며 안정적인 주행을 보였다.
3.  **장기 궤적 재현 (RMSE)**: 30초의 장기 주행 시나리오에서 Burn-InfoGAIL은 속도와 위치 모두에서 가장 낮은 RMSE를 기록하였다. 특히 GAIL 기반 방식들이 시간이 지남에 따라 평균적인 행동으로 회귀(Regress to the mean)하며 오차가 커지는 반면, 본 모델은 전문가의 종단점(End-point)에 가장 가깝게 도달하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
-   **엔트로피 항($\lambda$)의 중요성**: $\lambda=0$일 때 추론 모델이 모든 데이터를 단 하나의 라벨로 예측하는 붕괴 현상이 발생함을 확인하였다. 이를 통해 잠재 공간의 다양성을 유지하는 엔트로피 최대화가 다중 모드 학습에 필수적임을 입증하였다.
-   **일관성 유지**: Burn-in demonstration을 통해 $z$를 결정함으로써, 정책이 주행 내내 일관된 스타일을 유지할 수 있게 되었다. 이는 단순히 모방하는 것을 넘어, "누가 운전하고 있는가"라는 정체성을 부여한 것과 같다.

### 한계 및 향후 과제
-   **시뮬레이터 의존성**: 롤아웃(Rollout)을 위해 시뮬레이션 환경이 반드시 필요하므로, 완전한 비지도 클러스터링 방법으로 사용하기에는 제약이 있다. 저자들은 이를 해결하기 위해 학습된 환경 동역학 모델(Learned dynamics model)을 사용하는 방안을 제시한다.
-   **사전 분포 가정**: 본 연구는 스타일이 균등하게 분포되어 있다고 가정하고 엔트로피를 최대화했으나, 실제 데이터의 불균등한 분포를 반영하기 위해 KL Divergence 기반의 복잡한 Prior를 도입할 필요가 있다.

## 📌 TL;DR

본 논문은 다중 모드 모방 학습에서 발생하는 스타일 불일치 문제를 해결하기 위해, 전문가의 초기 궤적을 관찰하여 스타일을 추론하는 **Burn-in Demonstration** 개념과 **Burn-InfoGAIL** 알고리즘을 제안하였다. 실험 결과, 제안 방법은 기존 InfoGAIL 및 VAE 대비 스타일 분류 정확도(AMI)가 높고, 장기 주행 시에도 전문가의 스타일을 일관되게 유지하며 낮은 오차(RMSE)를 보였다. 이 연구는 자율주행과 같이 운전자의 개별적 스타일이 중요한 도메인에서 매우 실용적인 모방 학습 프레임워크를 제공한다.