# Imitation Learning by Estimating Expertise of Demonstrators

Mark Beliaev, Andy Shih, Stefano Ermon, Dorsa Sadigh, Ramtin Pedarsani (2022)

## 🧩 Problem to Solve

현대적인 Imitation Learning (IL) 데이터셋은 규모와 다양성을 확보하기 위해 여러 명의 시연자(demonstrators)로부터 데이터를 수집하는 경우가 많다. 그러나 각 시연자는 서로 다른 수준의 전문성(expertise)을 가지고 있으며, 특히 환경의 특정 상태(state)에 따라 능숙도가 다를 수 있다.

기존의 표준적인 IL 알고리즘들은 모든 시연자를 동일한 수준의 전문가로 간주하는 동질적(homogeneous) 접근 방식을 취한다. 이로 인해 모델이 비최적(suboptimal) 시연자의 잘못된 행동까지 그대로 학습하게 되어, 결과적으로 학습된 정책의 성능이 저하되는 문제가 발생한다. 본 논문의 목표는 시연자의 정체성(identity) 정보를 활용하여 각 시연자의 전문성 수준을 비지도 학습(unsupervised learning) 방식으로 추정하고, 이를 통해 최적의 행동만을 필터링하여 학습함으로써 시연자들보다 더 뛰어난 성능의 정책을 도출하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시연자의 전문성이 상태에 따라 다를 수 있다는 점에 착안하여, **상태 의존적 전문성(state-dependent expertise)**을 추정하는 통합 모델을 설계한 것이다.

- **통합 최적화 모델 (Joint Model):** 모방하고자 하는 정책 $\pi_\theta$와 각 시연자의 전문성 수준을 동시에 최적화한다.
- **상태-시연자 임베딩:** 상태 임베딩 $f_\phi(s)$와 시연자 임베딩 $\omega_i$의 내적을 통해 특정 상태에서의 시연자 능숙도를 계산함으로써, 어떤 시연자가 특정 상황에서 더 능숙한지를 파악한다.
- **이론적 보장:** 제안 방법론이 표준 Behavioral Cloning (BC)을 일반화하며, 특정 조건 하에서 최대 가능도 추정(Maximum Likelihood Estimation)을 통해 최적 정책을 회복할 수 있음을 이론적으로 증명하였다.
- **범용적 검증:** 시뮬레이션 환경(MiniGrid), 실제 로봇 제어 데이터(Robomimic), 그리고 체스 엔드게임 데이터 등 이산 및 연속 액션 공간 모두에서 성능 향상을 입증하였다.

## 📎 Related Works

기존의 모방 학습 연구들은 크게 두 가지 방향으로 진행되어 왔다. 첫째는 최적의 전문가 데이터만을 가정하는 전통적인 IL 방식이며, 둘째는 비최적 데이터를 처리하기 위해 보상 신호(reward signal)나 환경 역학(environment dynamics)을 사용하는 방식이다.

그러나 본 논문이 지적하듯, 실제 크라우드 소싱 데이터에서는 비최적 데이터가 불가피하게 포함되며, 모든 상황에서 보상 함수를 정의하거나 환경 시뮬레이터에 접근하는 것은 비용이 많이 들거나 불가능할 수 있다. 최근의 BC 기반 접근법들이 비최적 데이터에서 상대적으로 좋은 성능을 보였으나, 여전히 시연자의 개별적 특성을 고려하지 않는다는 한계가 있다. 본 연구는 보상이나 환경 역학 없이 오직 시연자의 ID 정보만을 활용하여 전문성을 추정한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전문성 수준 모델링 (Expertise Levels)

시연자 $i$가 상태 $s$에서 가지는 전문성 $\rho_\phi(s, \omega_i)$를 다음과 같이 정의한다.

$$\rho_\phi(s, \omega_i) = \sigma(\langle f_\phi(s), \omega_i \rangle)$$

여기서 $f_\phi(s)$는 상태 $s$를 $d$차원 벡터로 매핑하는 상태 임베딩 네트워크이며, $\omega_i$는 시연자 $i$의 능숙도를 나타내는 $d$차원 임베딩 벡터이다. $\sigma$는 시그모이드 함수로, 결과값 $\rho$는 $0$과 $1$ 사이의 값을 가진다. $\rho=1$은 완전한 전문가, $\rho=0$은 완전히 무작위적인 행동을 하는 상태를 의미한다.

### 2. 시연자의 행동 분포 (Demonstrator's Action Distribution)

최적 정책 $\pi_\theta^\star$와 추정된 전문성 $\rho$를 결합하여 시연자의 실제 행동 분포를 모델링한다.

- **이산 액션 공간 (Discrete Action Space):** 최적 정책과 균등 분포(uniform random) 사이를 선형 보간한다.
$$\pi(a|s, \omega_i, \phi, \pi_\theta^\star) = \rho_\phi(s, \omega_i)\pi_\theta^\star(a|s) + \frac{1 - \rho_\phi(s, \omega_i)}{|A|}$$

- **연속 액션 공간 (Continuous Action Space):** 최적 정책을 Gaussian Mixture Model (GMM)으로 정의하고, 전문성 $\rho$를 이용하여 각 컴포넌트의 분산을 스케일링한다.
$$\pi(a|s, \omega_i, \phi, \pi_\theta^\star) = \sum_{j=1}^{k} \alpha_j \mathcal{N}(a; \mu_j^\star(s), \sigma_j^\star(s)/\rho_\phi(s, \omega_i))$$
전문성이 낮을수록($\rho \to 0$) 분산이 커져 행동의 불확실성이 증가하게 된다.

### 3. 학습 절차 및 손실 함수

모델은 최적 정책 $\pi_\theta$, 상태 임베딩 $\phi$, 시연자 임베딩 $\omega$를 동시에 학습한다. 목적 함수는 데이터의 음의 로그 가능도(Negative Log-Likelihood, NLL)를 최소화하는 것이다.

$$L(\theta, \phi, \omega) = -\mathbb{E}_{i, (s,a)} [\log \pi(a|s, \omega_i, \phi, \pi_\theta)]$$

또한, 상태 임베딩 $f_\phi$의 표현력을 높이기 위해 DeepMDP 프레임워크를 보조 작업(auxiliary task)으로 사용한다. 이는 잠재 공간(latent space)에서 다음 상태의 임베딩을 예측하도록 하여, 환경의 역학 구조가 임베딩에 반영되도록 유도한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 환경:** MiniGrid (Empty, Lava, Obstacles, Unlock), Robomimic (Square - 로봇 팔 제어), Lichess (체스 엔드게임).
- **비교 대상 (Baselines):** BC, BC-RNN, GAIL, IRIS.
- **평가 지표:** 에피소드당 평균 보상(Mean Episodic Reward) 및 시연자 전문성 추정의 정확도.

### 2. 주요 결과

- **시뮬레이션 (MiniGrid):** Obstacles 환경에서 BC와 GAIL은 데이터에 노이즈가 적음에도 불구하고 낮은 성능을 보였으나, ILEED는 일관되게 높은 성능을 유지하였다. 특히 시연자들의 전문성 분포가 다양할 때 성능 향상 폭이 컸다.
- **실제 인간 데이터 (Robomimic):** 인간 시연자의 수준(Better, Okay, Worse)이 섞인 데이터셋에서 ILEED는 모든 설정에서 BC-RNN과 IRIS를 능가하였으며, 평균적으로 약 $4.8\%$의 보상 향상을 보였다.
- **전문성 추정 능력 (Chess):** 체스 플레이어의 실제 레이팅에 따라 5개 그룹으로 나눈 결과, ILEED가 추정한 전문성 $\rho$ 값이 레이팅 순위와 단조 증가(monotonicity) 관계에 있음을 확인하여, 비지도 방식으로 실제 숙련도를 정확히 추정할 수 있음을 입증하였다.
- **다중 기술 학습 (Multi-skill):** 서로 다른 기술을 가진 시연자들이 섞인 환경에서 상태 의존적 전문성 모델을 사용했을 때, 개별 시연자 중 가장 뛰어난 사람(Best Demonstrator)보다 더 높은 성능의 정책을 학습하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 보상 함수나 환경 시뮬레이터 없이 오직 **시연자의 ID**라는 최소한의 정보만으로 비최적 데이터셋에서 최적 정책을 추출할 수 있음을 보여주었다. 특히 전문성을 '상태 의존적'으로 모델링함으로써, 각 시연자의 강점만을 취합하여 개별 시연자 모두를 뛰어넘는 "Super-expert" 정책을 생성할 수 있다는 점이 매우 인상적이다.

### 한계 및 비판적 해석

- **상태 탐색의 의존성:** 이론적 보장과 실제 성능이 상태 임베딩의 품질에 크게 의존한다. 만약 시연자들이 상태 공간을 충분히 탐색하지 않았다면, 정확한 전문성 추정이 어려울 수 있다.
- **단순한 노이즈 모델:** 본 모델은 전문성을 단순히 분산의 증가나 균등 분포로의 보간으로 정의하였다. 하지만 실제 인간의 실수는 단순한 무작위 노이즈가 아니라 특정한 편향(bias)이나 체계적인 오류(systematic error)를 가질 가능성이 높다. 이러한 복잡한 오류 패턴을 포착하기 위한 더 정교한 모델링이 필요하다.
- **계산 비용:** $\theta, \phi, \omega$를 동시에 최적화해야 하므로 학습의 불안정성이 존재할 수 있으며, 본 논문에서도 이를 해결하기 위해 20번의 재시작(restarts)을 통해 최적의 모델을 선택하는 방식을 사용하였다.

## 📌 TL;DR

본 논문은 여러 명의 수준이 서로 다른 시연자로부터 데이터를 수집할 때, 각 시연자의 **상태 의존적 전문성(state-dependent expertise)**을 비지도 학습으로 추정하여 최적의 정책을 학습하는 **ILEED** 알고리즘을 제안한다. 제안 방법론은 보상 신호 없이도 비최적 데이터를 효과적으로 필터링하며, 결과적으로 개별 시연자보다 더 뛰어난 성능을 내는 정책을 도출한다. 이는 향후 대규모 크라우드 소싱 데이터셋을 활용한 로보틱스 및 자율 주행 제어 연구에 있어 데이터 효율성을 극대화하는 중요한 기여를 할 것으로 기대된다.
