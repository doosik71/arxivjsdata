# LEARNING WHILE NAVIGATING: A PRACTICAL SYSTEM BASED ON VARIATIONAL GAUSSIAN PROCESS STATE-SPACE MODEL AND SMARTPHONE SENSORY DATA

Ang Xie, Feng Yin, Bo Ai, Shuguang Cui (2020)

## 🧩 Problem to Solve

본 논문은 스마트폰의 센서 데이터를 활용한 실내 내비게이션(Indoor Navigation) 시스템의 정확도 향상 문제를 다룬다. 실내 환경에서는 GPS 수신이 어려우므로 보통 WiFi의 수신 신호 강도(Received Signal Strength, RSS)와 관성 측정 장치(Inertial Measurement Unit, IMU) 데이터를 결합한 상태 공간 모델(State-Space Model, SSM)을 사용한다.

기존의 파라메트릭(Parametric) SSM은 시스템의 동역학(Dynamics)이 명확히 정의되지 않은 경우 모델링이 어렵고, 데이터로부터 유용한 이동 패턴을 학습하는 능력이 부족하다는 한계가 있다. 따라서 본 연구의 목표는 비파라메트릭(Non-parametric) 모델인 Gaussian Process State-Space Model(GPSSM)을 실내 내비게이션에 적용하고, 이를 구현하기 위한 실용적인 변분 학습(Variational Learning) 절차를 제안하여 내비게이션의 정밀도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GPSSM의 강력한 비파라메트릭 표현력과 최신 내비게이션 기술을 결합하여, 실제 스마트폰 환경에서 작동 가능한 실용적인 학습 프레임워크를 구축하는 것이다.

주요 기여 사항은 다음과 같다.

1. **순차적 학습 절차 제안**: 측정 함수($g$)와 전이 함수($f$)를 순차적으로 학습함으로써 최적화 복잡도를 줄이고, 모델의 식별 불가능성(Non-identifiability) 문제를 완화하였다.
2. **실용적 데이터 융합**: 저비용 WiFi RSS 기반 위치 추정 기술과 보행자 추측 항법(Pedestrian Dead Reckoning, PDR)을 GPSSM 프레임워크 내에 통합하였다.
3. **변분 희소 GP(Variational Sparse GP) 적용**: 계산 복잡도를 낮추기 위해 유도 지점(Inducing points)을 도입한 변분 추론 방식을 적용하여 실제 환경에서의 구현 가능성을 입증하였다.

## 📎 Related Works

실내 내비게이션은 전통적으로 Kalman Filter, Extended Kalman Filter, Unscented Kalman Filter 및 Particle Filter와 같은 베이지안 필터링 알고리즘을 기반으로 한 SSM을 통해 수행되었다. 그러나 이러한 방식들은 전술한 바와 같이 모델의 파라미터를 사전에 정의해야 한다는 제약이 있다.

이를 해결하기 위해 함수 근사 능력이 뛰어난 Gaussian Process(GP)를 SSM에 도입한 GPSSM 연구들이 진행되었다. 초기 GPSSM은 latent state의 최대 사후 확률(MAP) 추정치를 찾는 방식으로 학습되었으나, 이후 완전한 확률적 학습을 위해 Particle Markov Chain Monte Carlo(PMCMC) 방식이 제안되었다. 하지만 PMCMC는 계산 부하가 매우 커서, 이를 해결하기 위해 변분 희소 GP(Variational Sparse GP) 프레임워크를 기반으로 한 다양한 변분 학습 절차들이 개발되었다. 본 논문은 이러한 이론적 배경을 바탕으로, 실제 실내 내비게이션 시나리오에서 검증되지 않았던 변분 GPSSM의 실용적 구현 방법을 제시하며 기존 연구와 차별점을 둔다.

## 🛠️ Methodology

### 1. GPSSM 기본 구조

GPSSM은 잠재 상태(Latent state) $x_t$와 관측치 $y_t$ 사이의 관계를 다음과 같은 두 함수로 정의한다.

- **전이 함수(Transition Function)**: $x_t = f(x_{t-1}, u_t) + q_{t-1}$
- **측정 함수(Measurement Function)**: $y_t = g(x_t) + r_t$

여기서 $u_t$는 제어 입력(Control input), $q$와 $r$은 각각 프로세스 노이즈와 측정 노이즈이다. GPSSM에서는 $f$와 $g$를 모두 GP로 모델링하며, 각각 평균 함수 $m(\cdot)$과 커널 함수 $k(\cdot, \cdot)$에 의해 정의된다.

$$f(x, u) \sim \text{GP}(m_f(x, u), k_f((x, u), (x', u')))$$
$$g(x) \sim \text{GP}(m_g(x), k_g(x, x'))$$

### 2. 측정 함수 $g$의 학습 (Measurement Function Learning)

측정 함수 학습은 WiFi RSS 데이터를 이용한 위치 추정으로 시작한다.

- **사전 학습(Pretraining)**: 선형 로그-거리 경로 손실 모델(Linear log-distance path loss model)을 사용하여 각 AP의 파라미터를 추정하고, 최대 우도 추정(Maximum Likelihood Estimation)을 통해 거친(Coarse) 위치 추정치 $y$를 얻는다.
- **GP를 통한 정밀화(Refinement)**: 단순한 RSS 추정치는 노이즈가 심하므로, 단일 GP 모델 $y = g(x) + r$을 통해 이를 정밀화한다. 이때 로그 주변 우도(Log-marginal likelihood) 함수를 최대화하여 커널 하이퍼파라미터 $\theta_g$와 노이즈 분산 $R$을 최적화한다.

### 3. 전이 함수 $f$의 학습 (Transition Function Learning)

전이 함수는 스마트폰 IMU 센서를 활용한 PDR 데이터를 제어 입력 $u_t$로 사용하여 학습한다.

- **PDR 기반 입력**: 가속도계, 자이로스코프, 지자기 센서를 이용해 걸음 수, 걸음 길이 $L_t$, 걷는 방향 $\psi_t$를 추정하여 $u_t = L_t [\sin(\psi_t), \cos(\psi_t)]^T$를 산출한다.
- **변분 학습(Variational Learning)**: 전이 함수 $f$는 관측되지 않은 입력/출력이 많아 계산이 어렵다. 이를 해결하기 위해 $M$개의 유도 지점(Inducing points) $v_{1:M}$을 도입하고, 증거 하한(Evidence Lower BOund, ELBO)을 최대화하는 변분 추론을 수행한다.

ELBO의 수식은 다음과 같다:
$$\log p(y_{1:T}) \geq \mathbb{E}_{q(x_{0:T}, f_{1:T}, v_{1:M})} \left[ \log \frac{p(y_{1:T}, x_{0:T}, f_{1:T}, v_{1:M})}{q(x_{0:T}, f_{1:T}, v_{1:M})} \right]$$

학습은 다음 과정을 반복적으로 수행한다:

1. 평활화 분포(Smoothing distribution) $q^*(x_{0:T})$로부터 샘플링.
2. $q^*(v_{1:M})$의 자연 파라미터 업데이트.
3. 경사 하강법(Gradient descent)을 통해 하이퍼파라미터 $\{\theta_f, Q, z_{1:M}\}$ 최적화.

## 📊 Results

### 실험 환경 및 설정

- **장소**: 홍콩 과연대학교(CUHK) 심천 캠퍼스의 $1600\text{m}^2$ 규모 사무실.
- **데이터**: 26개의 WiFi AP 배치, HUAWEI 스마트폰(Android 7.0) 사용, IMU 샘플링 레이트 100Hz.
- **작업**: "U"자 모양의 보행 경로 복원.
- **비교 대상**: WiFi 위치 추정 전용, PDR 전용, 선형 가우시안 상태 공간 모델(LGSSM), 학습 데이터 수에 따른 GPSSM(1, 3, 5개 궤적).
- **지표**: 평균 절대 오차(Mean Absolute Error, MAE).

### 실험 결과

정량적 결과는 다음과 같다 (Table 1 참조):

| 모델 | MAE (m) |
| :--- | :--- |
| WiFi Localization Only | 3.55m |
| PDR Only | 5.34m |
| LGSSM | 2.72m |
| GPSSM (1 Traj) | 2.29m |
| GPSSM (3 Trajs) | 2.16m |
| GPSSM (5 Trajs) | 2.11m |

- **분석**: WiFi 단독 방식은 AP가 부족한 급커브 구간에서 오차가 크고, PDR 단독 방식은 누적 오차로 인해 경로가 드리프트(Drift)되는 현상이 발생한다.
- **LGSSM vs GPSSM**: 파라메트릭 모델인 LGSSM은 측정 노이즈를 과소평가하여 WiFi의 잘못된 추정치에 쉽게 휘둘리는 경향을 보였다. 반면, 비파라메트릭 GPSSM은 학습 데이터(궤적 수)가 많아질수록 실제 경로에 훨씬 근접한 복원 성능을 보였으며, 가장 낮은 MAE($2.11\text{m}$)를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 GPSSM이 복잡한 실내 환경의 비선형적인 신호 특성과 보행자의 이동 패턴을 효과적으로 학습할 수 있음을 입증하였다. 특히, 측정 함수와 전이 함수를 순차적으로 학습하는 실용적인 절차를 통해 GPSSM의 고질적인 문제였던 계산 복잡도와 식별 불가능성 문제를 해결하여 실제 시스템에 적용 가능한 수준으로 끌어올렸다는 점이 강점이다.

다만, 본 연구에서는 보행자의 걸음 길이를 상수($L_{\text{const}}$)로 가정하였다는 한계가 있다. 실제 보행자의 걸음 길이는 개인이나 상황에 따라 달라지므로, 이를 동적으로 추정하는 메커니즘이 추가된다면 성능이 더욱 향상될 수 있을 것이다. 또한, 유도 지점(Inducing points)의 선택 기준이 결과에 영향을 미칠 수 있으므로, 이에 대한 더 정교한 최적화 방안에 대한 논의가 필요하다.

## 📌 TL;DR

이 논문은 스마트폰의 WiFi RSS와 IMU 데이터를 융합하여 실내 위치를 추적하는 **변분 GPSSM(Variational GPSSM) 기반 내비게이션 시스템**을 제안한다. 비파라메트릭 모델의 특성을 활용해 복잡한 실내 환경을 정밀하게 모델링하였으며, 순차적 학습 절차와 변분 추론을 통해 실제 구현 가능성을 증명하였다. 실험 결과, 기존의 선형 모델이나 단일 센서 방식보다 월등한 정확도를 보였으며, 이는 향후 데이터 기반의 정밀 실내 위치 추적 연구에 중요한 기여를 할 것으로 평가된다.
