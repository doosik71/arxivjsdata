# Active Learning in Gaussian Process State Space Model

Hon Sum Alec Yu, Dingling Yao, Christoph Zimmer, Marc Toussaint, and Duy Nguyen-Tuong (2021)

## 🧩 Problem to Solve

본 논문은 Gaussian Process State Space Model(GPSSM)에서 데이터 효율적인 학습을 위한 Active Learning(AL) 전략을 연구한다. 구체적으로, 시스템의 기저 동역학(underlying dynamics)을 최적으로 학습하기 위해, 시스템을 어떤 latent state(잠재 상태)로 유도할 것인지 결정하는 최적의 입력값(control inputs)을 능동적으로 선택하는 문제를 해결하고자 한다.

일반적인 동역학 시스템 학습에서는 많은 양의 데이터가 필요하지만, 실제 물리 시스템에서는 데이터 수집 비용이 매우 높다. 따라서 최소한의 데이터만으로 모델의 불확실성을 효과적으로 줄이는 AL 기법이 필수적이다. 특히 기존의 AL 기반 GPSSM 연구들은 상태 변수(state)가 관측 가능하고 측정 가능하다는 가정을 전제로 하였으나, 실제 많은 시스템에서는 상태 변수가 latent state로 존재하여 직접 관측할 수 없다는 문제가 있다. 본 논문의 목표는 상태 변수가 관측되지 않는 latent state-space 상황에서도 동작하는 효율적인 AL 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mutual Information(상호 정보량)을 AL의 기준(criterion)으로 사용하여, 모델의 불확실성을 가장 많이 줄여줄 수 있는 제어 입력 $c_t$를 선택하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Latent State를 위한 MI 추정치 도출**: 상태 변수가 관측되지 않는 GPSSM 환경에서 계산 가능한(tractable) 근사 Mutual Information 추정 방식을 유도하였다.
2. **두 가지 AL 전략 제안**: 최신 관측치와 예측 전이 함수 간의 관계를 보는 latest Mutual Information(latMI)과, 전체 시퀀스의 정보를 고려하는 total Mutual Information(totMI) 전략을 제안하였다.
3. **물리 시스템 검증**: 시뮬레이션 및 실제 물리 시스템(Pendulum, Cart-Pole, Twin-Rotor Aerodynamical System) 실험을 통해 제안한 AL 전략이 무작위 탐색보다 훨씬 적은 데이터로 높은 정확도를 달성함을 입증하였다.

## 📎 Related Works

기존의 State-Space Model(SSM) 연구들은 주로 모델링과 추론에 집중해 왔으며, Bayesian 관점에서 불확실성을 다루기 위해 Gaussian Process(GP) prior를 도입한 GPSSM이 발전해 왔다. 최근에는 Variational Inference(VI)를 통해 계산 복잡도를 해결하려는 시도가 주를 이루고 있다.

AL 분야에서는 GP regression을 이용한 데이터 선택 연구가 활발하였으나, 이를 GPSSM과 결합한 연구는 드물다. Buisson et al. [6]과 Capone et al. [8]이 AL을 GPSSM에 적용한 바 있으나, 이들은 상태 변수가 관측 가능하다(observable)고 가정하였다. 상태가 관측 가능하다면 문제는 단순한 지도 학습 기반의 GP regression으로 환원되지만, 본 논문은 상태가 관측되지 않는 latent state 상황을 다룸으로써 기존 연구들과 차별점을 가진다.

## 🛠️ Methodology

### 1. GPSSM 모델 정의
본 논문에서 사용하는 GPSSM은 이산 시간 시퀀스로 정의되며, 다음과 같은 확률 구조를 가진다.

- **전이 함수(Transition Function)**: $f \sim GP(m(\cdot), k(\cdot, \cdot))$
- **상태 전이(State Transition)**: $$x_t | f(x_{t-1}, c_{t-1}) \sim \mathcal{N}(x_t | f_{t-1}, Q)$$
- **관측 모델(Observation Model)**: $$y_t | x_t \sim \mathcal{N}(y_t | Cx_t + d, R)$$

여기서 $x_t$는 latent state, $c_t$는 제어 입력, $y_t$는 관측값이다. $Q$와 $R$은 각각 프로세스 노이즈와 관측 노이즈의 공분산 행렬이다.

### 2. 학습 및 추론 (Variational Inference)
Latent state와 전이 함수를 추론하기 위해 Variational Inference를 사용하며, Evidence Lower Bound(ELBO)를 최대화하는 방향으로 학습한다. 계산 효율성을 위해 Sparse GP 기법을 도입하여 $M$개의 inducing points $u$를 사용한다. 본 논문에서는 $q(x_i|f_i) = \mathcal{N}(x_i | A_{i-1}x_{i-1} + b_{i-1}, S_{i-1})$ 형태의 선형 팩토라이징 근사를 채택하여 학습 안정성과 속도를 높였다.

### 3. Active Learning 전략
최적의 제어 입력 $c_t^*$를 선택하기 위해 다음과 같은 MI 기반의 목적 함수를 사용한다.

#### (1) Latest Mutual Information (latMI)
가장 최근의 관측치 $\hat{y}_{t+1}$과 예측된 전이 함수 $f_{t+1}$ 사이의 MI를 최대화한다.
$$c_t^* = \text{argmax}_{c_t \in C} I(\hat{y}_{t+1}; f_{t+1})$$
이 값은 Girard [24]의 Gaussian approximate integral을 통해 근사적으로 계산하며, 최종적으로 다음과 같은 closed-form으로 표현된다.
$$I(y_t; f_t) \approx \frac{1}{2} \log \left( \frac{\det(R + C(V_t + Q)C^T)}{\det(R + CQC^T)} \right)$$

#### (2) Total Mutual Information (totMI)
시간 범위 내의 모든 관측치와 전이 함수 간의 MI를 고려하는 더 일반적인 형태이다.
$$c_t^* = \text{argmax}_{c_t \in C} I(y_{1:t}, \hat{y}_{t+1}; f_{1:t+1})$$
본 논문은 이를 계산하기 위해 ELBO를 활용하여 다음과 같은 상한선(bound)을 도출하였다.
$$i_s \leq \sum_{i=1}^t \log(\mathcal{N}^s(y_i | Cf_i + d, R + CQC^T)) - L_{t,s}$$
여기서 $L_{t,s}$는 $s$번째 샘플에 대한 ELBO 값이다.

## 📊 Results

### 실험 설정
- **대상 시스템**: Simulated function(modified kink), Pendulum, Cart-Pole, Twin-Rotor Aerodynamical System(TRAS).
- **비교 대상**: Random exploration, latMI strategy, totMI strategy.
- **평가 지표**: Root Mean Square Error(RMSE) 및 데이터 효율성(탐색 포인트 수 대비 정확도).

### 주요 결과
1. **정확도 및 효율성**: 모든 실험에서 totMI 전략이 Random 및 latMI보다 빠르게 RMSE를 낮추는 것으로 나타났다. 특히 TRAS와 같이 복잡한 동역학을 가진 시스템에서 totMI의 우수성이 두드러졌다.
2. **latMI vs totMI**: latMI 역시 무작위 탐색보다는 나은 성능을 보였으나, 전체 시퀀스의 정보를 활용하는 totMI가 훨씬 안정적이고 높은 성능을 보였다.
3. **계산 시간**: totMI는 모델 학습 시 사용되는 추론 스킴(inference scheme)을 그대로 재사용하기 때문에, 수치적 적분을 반복해야 하는 latMI보다 계산 시간이 훨씬 짧았다.
4. **상태 관측 가능 시 비교**: 상태 변수를 알 수 있다는 가정하에 수행된 Capone et al. [8]의 locAL 방식과 비교했을 때, totMI는 상태 정보 없이도 경쟁력 있는 성능을 보였으며, 최종 수렴 시점에서는 더 우수한 성능을 나타냈다.

## 🧠 Insights & Discussion

본 연구는 latent state-space를 가진 GPSSM에서 MI를 기반으로 한 AL 전략이 가능함을 이론적으로 증명하고 실험적으로 검증하였다. 

**강점 및 통찰**:
- **전역적 정보 활용**: 단순히 최신 상태의 불확실성만 보는 것이 아니라, 전체 시퀀스의 상호 정보량을 고려하는 totMI가 훨씬 효과적이라는 점을 밝혀냈다.
- **추론 스킴의 재사용**: AL을 위한 MI 계산을 위해 별도의 복잡한 함수를 도입하는 대신, 이미 모델 학습에 사용되는 ELBO를 활용함으로써 계산 효율성을 극대화하였다.
- **RL과의 차이점**: 강화학습(RL)의 POMDP에서도 MI 최대화 전략이 사용되지만, 본 논문의 'control'은 보상을 최대화하는 정책(policy)을 찾는 것이 아니라, 모델의 불확실성을 줄이기 위한 '탐색 도구'로서의 입력값이라는 점에서 본질적인 차이가 있다.

**한계 및 향후 과제**:
- 본 논문은 상태 변수가 latent라고 가정하지만, 실험의 검증을 위해 일부 시스템에서는 Ground-truth 상태를 사용하여 RMSE를 측정하였다. 실제 완전히 관측 불가능한 환경에서의 정밀한 평가 지표에 대한 논의가 더 필요할 수 있다.

## 📌 TL;DR

이 논문은 상태 변수를 직접 관측할 수 없는 **Latent GPSSM 환경에서 데이터 효율적인 학습을 위한 Active Learning 전략**을 제안한다. 특히 Mutual Information을 근사하여 최적의 제어 입력을 선택하는 **totMI(total Mutual Information)** 방식을 제안하였으며, 이는 계산 효율성과 학습 정확도 측면에서 무작위 탐색 및 기존 방식보다 월등한 성능을 보인다. 이 연구는 데이터 수집 비용이 높은 실제 복잡한 물리 시스템의 동역학 모델링에 중요한 기여를 할 수 있다.