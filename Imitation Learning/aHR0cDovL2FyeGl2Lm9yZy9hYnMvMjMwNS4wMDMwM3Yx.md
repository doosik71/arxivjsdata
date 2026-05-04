# A Coupled Flow Approach to Imitation Learning

Gideon Freund, Elad Sarafian, Sarit Kraus (2023)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL) 및 강화 학습(Reinforcement Learning, RL)에서 핵심적인 역할을 하는 **상태 분포(State Distribution)** 및 **상태-행동 분포(State-Action Distribution)**의 명시적 모델링 문제를 해결하고자 한다.

상태 분포는 Policy Gradient Theorem의 기초가 되며, 많은 분포 매칭(Distribution Matching) 기반의 모방 학습 알고리즘에서 핵심적인 요소이다. 하지만 고차원 데이터에서 확률 밀도 추정(Density Estimation)의 어려움으로 인해, 기존 연구들은 상태 분포를 이론적인 도구로만 사용했을 뿐 이를 명시적으로 모델링하는 경우는 드물었다. 

특히 기존의 분포 매칭 방식들은 전문가의 데이터가 적거나, 행동 정보 없이 상태 정보만 주어진 상황(Learning from Observations, LFO), 혹은 데이터가 희소한(Subsampled) 상황에서 성능이 급격히 저하되는 한계가 있다. 따라서 본 논문의 목표는 **Normalizing Flows(NF)**를 활용하여 이러한 분포들을 명시적으로 모델링하고, 이를 통해 적은 양의 전문가 데이터만으로도 전문가의 행동을 효과적으로 모방하는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **두 개의 Normalizing Flow 모델을 Donsker-Varadhan (DV) 표현식을 통해 결합(Couple)**하여 로그 분포 비율(Log Distribution Ratio)을 정확하게 추정하는 것이다.

기존에는 전문가 분포 $p_e$와 에이전트 분포 $p_\pi$를 각각 독립적인 밀도 추정기로 모델링하려 했으나, 이는 두 모델이 서로 다른 분포 영역에서 평가되는 **Out-of-Distribution (OOD)** 문제로 인해 실패했다. 저자들은 DV 표현식의 최적점(Optimality Point)이 로그 분포 비율에서 발생한다는 점에 착안하여, 두 flow의 차이로 보상 함수를 정의하고 이를 적대적(Adversarial) 방식으로 학습시키는 **Coupled Flow Imitation Learning (CFIL)**을 제안하였다.

## 📎 Related Works

### 관련 연구 및 한계
1. **GAIL 및 DAC**: JS Divergence를 최소화하는 적대적 학습 방식을 사용한다. 하지만 전문가 데이터가 매우 적은 상황에서는 불안정하거나 성능이 낮을 수 있다.
2. **DICE Family (ValueDICE 등)**: Reverse KL Divergence를 최소화하며, DV 표현식을 사용하여 off-policy 목적 함수를 유도했다. 그러나 상태-행동 쌍이 모두 필요하며, 데이터가 희소할 경우 성능이 떨어진다는 지적이 있다.
3. **NDI (Imitation with Neural Density Models)**: Normalizing Flows를 사용하여 전문가 분포를 모델링한다. 하지만 CFIL과 달리 비적대적(Non-adversarial) 방식이며, 하한선(Lower Bound)을 사용하는 느슨한 접근 방식을 취해 성능 면에서 CFIL에 뒤처진다.

### CFIL의 차별점
CFIL은 NF를 사용하여 분포를 명시적으로 모델링한다는 점에서는 NDI와 유사하지만, **DV 표현식을 통한 두 flow의 결합**과 **적대적 학습 구조**를 도입함으로써 OOD 문제를 해결하고, 단 하나의 전문가 궤적(Single expert trajectory)만으로도 전문가 수준의 성능을 낼 수 있도록 설계되었다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 핵심 아이디어
CFIL은 Reverse KL Divergence를 최소화하는 문제를 RL 문제로 변환하여 푼다. 즉, 로그 분포 비율을 보상(Reward)으로 정의하여 정책 $\pi$를 최적화한다.

$$\arg \min_{\pi} D_{KL}(p^\pi || p^e) = \arg \max_{\pi} \mathbb{E}_{p^\pi(s,a)} \left[ \log \frac{p^e(s,a)}{p^\pi(s,a)} \right]$$

여기서 보상 함수 $r = \log \frac{p^e}{p^\pi}$를 정확히 추정하기 위해 두 개의 Normalizing Flow 모델 $p_\psi$와 $q_\phi$를 사용한다.

### 2. Coupled Flow 추정기 및 DV 표현식
단순히 두 flow를 독립적으로 학습시키는 대신, DV 표현식을 사용하여 결합한다. DV 표현식에 따르면 KL Divergence는 다음과 같이 정의된다.

$$D_{KL}(p^\pi || p^e) = \sup_{x: S \times A \to \mathbb{R}} \mathbb{E}_{p^\pi(s,a)}[x(s,a)] - \log \mathbb{E}_{p^e(s,a)}[e^{x(s,a)}]$$

이 식의 최적점 $x^*$는 $\log \frac{p^\pi}{p^e} + C$가 된다. 따라서 CFIL은 $x$를 다음과 같은 inductive bias를 가진 네트워크로 모델링한다.

$$x_{\psi, \phi}(s, a) = \log p_\psi(s, a) - \log q_\phi(s, a)$$

여기서 $p_\psi$와 $q_\phi$는 **Masked Autoregressive Flow (MAF)** 아키텍처를 사용한다. 이렇게 결합된 구조는 학습 과정에서 두 flow가 서로의 데이터를 함께 처리하게 하여 OOD 문제를 방지한다.

### 3. 학습 절차 및 손실 함수
학습은 정책 $\pi$를 업데이트하는 단계와 flow 파라미터 $\psi, \phi$를 업데이트하는 단계를 교대로 수행하는 적대적 방식으로 진행된다.

**A. Flow 업데이트 (Minimization Step):**
다음의 목적 함수 $J$를 최소화하여 $\psi$와 $\phi$를 업데이트한다.

$$J = \log \frac{1}{M} \sum_{i=1}^M e^{x(s^e_i, a^e_i)} - \frac{1}{M} \sum_{i=1}^M x(s_i, a_i)$$

이때, 수치적 안정성을 위해 $x$에 **Squashing function** $\sigma = 6 \tanh(x/15)$를 적용한다. 또한, 선택적으로 다음과 같은 **Flow Regularization** 손실 $L$을 추가할 수 있다.

$$L = -\frac{1}{M} \sum_{i=1}^M \log q_\phi(s^e_i, a^e_i) + \log p_\psi(s_i, a_i)$$

**B. 정책 업데이트 (Maximization Step):**
업데이트된 $x_{\psi, \phi}$의 음수 값($-x^*$)을 보상으로 사용하여 SAC(Soft Actor-Critic)와 같은 RL 알고리즘으로 정책 $\pi$를 최적화한다.

**C. Smoothing 기법:**
입력 데이터의 정밀도 문제를 해결하기 위해 각 차원에 uniform noise를 더하는 smoothing 기법을 적용한다.
$$(s, a) += \beta \cdot (s, a) \odot u, \quad u \sim \text{Uniform}(-1/2, 1/2)$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Mujoco 벤치마크 (HalfCheetah, Walker2d, Ant, Hopper, Humanoid).
- **기준선(Baselines)**: ValueDICE, DAC, Behavioral Cloning (BC).
- **지표**: 정규화된 비동기적 보상(Normalized asymptotic reward).
- **특이사항**: 단 하나의 전문가 궤적(Single expert trajectory)만 사용.

### 2. 주요 결과
- **Standard State-Action Setting**: CFIL은 단 하나의 궤적만으로 모든 작업에서 전문가 수준의 성능에 도달하였다. 특히 Ant와 Humanoid 환경에서 ValueDICE와 DAC보다 월등한 성능을 보였다.
- **Learning from Observations (LFO)**: 행동 정보 없이 상태 정보만 주어진 설정에서, CFIL은 기존 SOTA인 OPOLO를 압도하는 성능을 기록하였다. 특히 단일 상태 분포 $d(s)$만 모델링했을 때도 성공적으로 동작하는 놀라운 범용성을 보였다.
- **Subsampled Regime**: 데이터를 10, 20, 50, 100배로 희소하게 샘플링했을 때도, CFIL은 매우 적은 데이터(예: 단 10개의 상태-행동 쌍)만으로도 전문가의 행동을 어느 정도 회복할 수 있음을 입증하였다.

## 🧠 Insights & Discussion

### 1. BC Graph를 통한 분석
저자들은 제안하는 추정기가 보상 함수로서 얼마나 유효한지 측정하기 위해 **BC Graph**라는 분석 도구를 제시하였다. 이는 BC 에이전트의 학습 과정 중 생성된 궤적들에 대해 추정기가 부여한 보상과 실제 환경 보상의 상관관계를 나타낸다.
- **Uncoupled Flow**: BC Graph가 우상향하지 않아 RL 학습자가 잘못된 보상 신호로 인해 실패함을 확인하였다.
- **Coupled Flow**: BC Graph가 뚜렷한 우상향 추세를 보이며, 이는 RL 학습자가 전문가의 행동을 따라갈 수 있는 적절한 인센티브를 제공함을 의미한다.

### 2. 강점 및 한계
- **강점**: 하이퍼파라미터 튜닝 없이도 다양한 환경(LFO, Sparse data)에서 매우 견고한 성능을 보인다. 특히 Normalizing Flows의 명시적 밀도 추정 능력을 IL에 성공적으로 결합하였다.
- **한계**: DV 표현식을 통한 적대적 학습의 특성상 학습의 안정성을 위해 squashing function과 같은 휴리스틱한 장치가 필요하며, flow 모델 자체의 학습이 매우 섬세한 작업임을 언급하였다.

## 📌 TL;DR

본 논문은 Normalizing Flows를 활용해 상태-행동 분포를 명시적으로 모델링하는 **CFIL (Coupled Flow Imitation Learning)**을 제안한다. 핵심은 두 flow를 **Donsker-Varadhan 표현식**으로 결합하여 OOD 문제를 해결하고 정확한 로그 분포 비율을 추정하여 이를 RL의 보상으로 사용하는 것이다. 실험 결과, 단 하나의 전문가 궤적만으로도 SOTA 성능을 달성했으며, 특히 상태 정보만 주어진 상황(LFO)이나 데이터가 극도로 희소한 상황에서 기존 방식들을 압도하는 범용성과 효율성을 입증하였다. 이는 향후 RL/IL 분야에서 상태 분포의 명시적 모델링이 중요한 역할을 할 수 있음을 시사한다.