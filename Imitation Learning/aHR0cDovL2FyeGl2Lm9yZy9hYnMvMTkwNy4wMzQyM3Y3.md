# On-Policy Robot Imitation Learning from a Converging Supervisor

Ashwin Balakrishna, Brijen Thananjeyan, Jonathan Lee, Felix Li, Arsh Zahed, Joseph E. Gonzalez, Ken Goldberg (2019)

## 🧩 Problem to Solve

기존의 On-policy imitation learning 알고리즘(예: DAgger)은 고정된(fixed) 전문가 supervisor가 존재한다고 가정한다. 하지만 실제 환경에서는 새로운 작업을 수행하는 인간 전문가나, 학습을 통해 성능이 개선되는 알고리즘 기반의 controller와 같이 시간이 지남에 따라 행동이 변화하고 수렴하는 converging supervisor가 제공되는 경우가 많다.

또한, 이러한 supervisor들은 대체로 실행 속도가 매우 느리다는 문제점이 있다. 인간은 로봇을 통해 고주파수의 정밀한 동작을 수행하는 데 어려움을 겪으며, Model Predictive Control(MPC)와 같은 모델 기반 제어 기법은 복잡한 역학 모델 위에서 확률적 최적화를 수행해야 하므로 계산 비용이 매우 크다.

따라서 본 논문의 목표는 시간이 흐르며 성능이 개선되지만 실행 속도가 느린 converging supervisor의 능력을, 효율적으로 실행 가능한 reactive policy로 증류(distill)하는 프레임워크를 제안하고 그 이론적 보장과 실용성을 입증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 'Converging Supervisor Framework(CSF)'라는 새로운 온-폴리시 모방 학습 프레임워크를 제안한 것이다. 

중심적인 직관은 학습 과정 중에는 중간 단계의 supervisor들로부터 레이블을 제공받더라도, 최종적으로 수렴한 supervisor($\psi_N$)를 기준으로 했을 때의 Static 및 Dynamic regret이 sublinear하게 유지될 수 있음을 이론적으로 증명한 것이다. 이를 통해 supervisor의 구조에 제약 없이 어떠한 오프-폴리시(off-policy) 방법론(RL 알고리즘 또는 인간)도 supervisor로 사용할 수 있는 유연성을 확보하였다.

실용적 관점에서는 데이터 효율성이 뛰어난 Deep MBRL 알고리즘인 PETS를 improving supervisor로 활용하여, PETS의 높은 데이터 효율성은 유지하면서 추론 속도는 획기적으로 높인 모델-프리(model-free) learner policy를 학습시켰다.

## 📎 Related Works

기존의 On-policy 모방 학습인 DAgger는 전문가의 고정된 정책을 가정하여 learner의 궤적 분포에서 피드백을 받는 방식을 사용하였다. 최근에는 고정된 supervisor 설정에서 dynamic regret에 대한 분석이 이루어졌으나, supervisor 자체가 변화하는 설정에 대한 분석은 부족했다.

또한, Model-based planning을 reactive policy로 증류하려는 Dual Policy Iteration(DPI) 연구들이 존재했다. 그러나 기존의 DPI 방식들은 주로 이산 상태 공간(discrete state spaces)에 적용되었거나, supervisor의 구조에 대해 특정한 제약 조건을 두어 최신 Deep MBRL 알고리즘의 강력한 모델 용량을 충분히 활용하지 못하는 한계가 있었다. 본 논문은 supervisor의 구조적 가정을 제거함으로써 이러한 한계를 극복하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
본 프레임워크는 $\text{MDP}(S, A, P, T, R)$ 환경에서 동작하며, 다음과 같은 반복적인 절차를 따른다.
1. 현재의 learner policy $\pi_{\theta_i}$를 환경에 배포하여 궤적(trajectory)을 생성한다.
2. 생성된 궤적의 각 상태 $s$에 대해, 현재 단계의 supervisor $\psi_i$로부터 최적 행동(label)을 제공받는다.
3. 제공받은 레이블을 사용하여 supervised learning loss를 최소화하도록 $\pi_{\theta_i}$를 업데이트하여 $\pi_{\theta_{i+1}}$을 생성한다.

### 손실 함수 및 학습 목표
Learner는 supervisor $\psi_i$와의 차이를 줄이기 위해 다음과 같은 surrogate loss 함수를 최소화하는 것을 목표로 한다.

$$l_i(\pi_\theta, \psi_i) = \mathbb{E}_{\tau \sim p(\tau|\theta_i)} \left[ \frac{1}{T} \sum_{t=1}^T \|\pi_\theta(s_t^i) - \psi_i(s_t^i)\|^2 \right]$$

여기서 $p(\tau|\theta_i)$는 정책 $\pi_{\theta_i}$에 의해 생성된 궤적의 분포이며, $\pi_\theta(s)$와 $\psi_i(s)$는 각각 learner와 supervisor가 상태 $s$에서 선택한 행동이다.

### 이론적 분석: Regret Analysis
논문은 고정된 supervisor가 아닌, 최종 수렴한 supervisor $\psi_N$에 대한 regret을 분석한다.

- **Static Regret**: 전체 학습 기간 동안 $\pi_{\theta_i}$와 $\psi_N$을 기준으로 한 최적 정책 $\pi_{\theta^*}$ 사이의 누적 손실 차이를 측정한다.
- **Dynamic Regret**: 각 라운드 $i$마다 현재 정책 $\pi_{\theta_i}$와 해당 라운드에서 $\psi_N$을 기준으로 한 최적 정책 $\pi_{\theta^*_i}$ 사이의 손실 차이를 측정한다.

본 논문은 다음과 같은 핵심 정리를 통해 converging supervisor 설정에서도 sublinear regret이 가능함을 보인다.
- **Theorem 4.1**: $\psi_N$에 대한 static regret은 일반적인 static regret과 supervisor의 변화율(convergence rate)에 비례하는 항의 합으로 상한선을 정의할 수 있다.
- **Theorem 4.2**: 궤적 분포의 Lipschitz 연속성 가정이 충족되고 supervisor가 수렴한다면, $\psi_N$에 대한 dynamic regret 또한 sublinear하게 달성 가능하다.

### 실용적 구현: PETS as Supervisor
이론적 프레임워크를 구현하기 위해 PETS(Probabilistic Ensembles with Trajectory Sampling)를 supervisor로 사용한다. PETS는 앙상블 역학 모델을 통해 데이터 효율적인 제어를 수행하지만 계산량이 많다. 이를 해결하기 위해 다음과 같은 절차를 수행한다.
1. Model-free learner policy를 통해 롤아웃을 수행한다.
2. 롤아웃이 종료된 후, 수집된 데이터를 기반으로 PETS가 각 상태에서의 최적 행동을 오프라인으로 계산하여 레이블을 생성한다.
3. DAgger 방식을 통해 learner policy를 PETS의 레이블로 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**: MuJoCo 시뮬레이션(PR2 Reacher, Pusher) 및 실제 로봇 실험(da Vinci Surgical Robot, dVRK).
- **비교 대상(Baseline)**: SAC, TD3 (Model-free RL), ME-TRPO (Hybrid RL), PETS (Model-based RL).
- **측정 지표**: 누적 보상(Return), 정책 평가 시간(Policy evaluation time), 쿼리 시간(Query time).

### 주요 결과
1. **성능 및 데이터 효율성**: 시뮬레이션 실험에서 CSF learner는 PETS와 유사한 수준의 보상을 달성하였으며, SAC, TD3, ME-TRPO 등 다른 deep RL 베이스라인보다 뛰어난 성능을 보였다. 이는 PETS의 데이터 효율성을 유지하면서 reactive policy로의 증류가 성공적이었음을 의미한다.
2. **실행 속도 향상**: 
   - 시뮬레이션 환경에서 CSF learner의 정책 평가 속도는 PETS 대비 최대 **80배** 빨랐다.
   - 실제 dVRK 로봇 실험에서도 쿼리 시간이 최대 **20배** 감소하였다.
3. **물리 로봇 적용**: dVRK의 Single-arm 및 Double-arm Reacher 작업에서 CSF learner가 PETS supervisor를 효과적으로 추적하며 성공적으로 학습됨을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 Model-based RL(MBRL)의 높은 데이터 효율성과 Model-free RL의 빠른 실행 속도라는 두 마리 토끼를 잡을 수 있는 이론적/실무적 토대를 제공하였다. 특히, supervisor가 고정되어 있지 않고 학습 과정 중에 계속 변한다는 현실적인 설정을 수용하면서도 이론적인 regret 보장을 이끌어낸 점이 돋보인다.

다만, 본 논문은 supervisor가 실제로 MDP의 보상 함수 $R$에 대해 개선된다는 가정을 전제로 하고 있다. 만약 supervisor가 잘못된 방향으로 수렴하거나 발산한다면 learner 역시 잘못된 정책을 학습하게 될 것이다. 또한, 실제 로봇 실험에서 하드웨어 제어 주파수의 한계로 인해 이론적인 쿼리 시간 단축이 실제 전체 평가 시간 단축으로 완전히 이어지지 않은 점은 향후 고주파 제어가 가능한 시스템에서의 검증이 필요함을 시사한다.

비판적으로 보자면, learner가 supervisor의 분포가 아닌 자신의 분포에서 데이터를 수집하므로 발생하는 분포 차이(distribution shift) 문제를 DAgger 형태의 온-폴리시 업데이트로 해결하려 했으나, 이 과정에서 발생하는 샘플 효율성 저하에 대한 정량적 분석이 더 보완되었다면 좋았을 것이다.

## 📌 TL;DR

본 논문은 시간이 지남에 따라 성능이 개선되는 'Converging Supervisor'로부터 로봇 정책을 효율적으로 학습하는 프레임워크(CSF)를 제안하고, 최종 수렴 정책에 대한 sublinear regret을 이론적으로 증명하였다. 이를 PETS 알고리즘에 적용하여 MBRL의 데이터 효율성을 유지하면서도 추론 속도를 최대 80배까지 가속화하였으며, 이를 실제 수술 로봇(dVRK)에 적용하여 실용성을 입증하였다. 이 연구는 무거운 모델 기반 제어기를 가벼운 신경망 정책으로 빠르게 증류하는 표준적인 방법론을 제시했다는 점에서 향후 실시간 로봇 제어 연구에 중요한 기여를 할 가능성이 높다.