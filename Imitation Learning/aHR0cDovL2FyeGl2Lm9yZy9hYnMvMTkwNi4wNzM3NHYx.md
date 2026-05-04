# Sample-efficient Adversarial Imitation Learning from Observation

Faraz Torabi, Sean Geiger, Garrett Warnell, Peter Stone (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Imitation from Observation (IfO, 관찰로부터의 모방 학습) 프레임워크에서 발생하는 심각한 샘플 효율성(sample inefficiency) 문제이다. IfO는 전문가의 행동(action) 정보 없이 오직 상태(state) 궤적만을 관찰하여 작업을 학습하는 방식이다. 최근 Generative Adversarial Imitation from Observation (GAIfO)와 같은 적대적 학습 방식이 복잡한 행동 모방에서 뛰어난 성능을 보였으나, 이러한 알고리즘들은 정책이 수렴하기까지 방대한 양의 데모 데이터와 수많은 학습 반복 횟수를 요구한다.

이러한 높은 샘플 복잡도는 물리적 로봇에 알고리즘을 직접 적용하는 것을 어렵게 만든다. 시뮬레이션 환경에서는 데이터를 빠르게 생성할 수 있지만, 실제 로봇은 물리적 제약과 비용 문제로 인해 데이터 수집 속도가 제한적이기 때문이다. 따라서 본 연구의 목표는 적대적 모방 학습의 높은 성능을 유지하면서도, 물리적 로봇에 적용 가능할 수준으로 샘플 효율성을 높인 새로운 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 적대적 학습 알고리즘의 성능과 궤적 중심 강화학습(trajectory-centric reinforcement learning)의 샘플 효율성을 결합하는 것이다. 구체적으로, 기존의 GAIfO 프레임워크에 Linear Quadratic Regulator (LQR)을 통합하여 제안한 **LQR+GAIfO** 알고리즘이 핵심 기여이다.

LQR은 선형 동역학과 이차 비용 함수(quadratic cost)를 가정하여 최적 제어기를 매우 효율적으로 찾아내는 기법이다. 저자들은 LQR의 효율적인 궤적 최적화 능력을 GAIfO의 적대적 보상 학습 구조에 접목함으로써, 딥러닝 기반의 Model-free RL 방식보다 훨씬 적은 샘플만으로도 전문가의 행동을 빠르게 모방할 수 있도록 설계하였다.

## 📎 Related Works

논문에서는 모방 학습의 기존 접근 방식을 크게 두 가지로 분류하여 설명한다. 첫째는 상태에서 행동으로의 직접적인 매핑을 배우는 Behavioral Cloning (BC)이며, 둘째는 전문가가 최적화하고자 하는 비용 함수를 추정하는 Inverse Reinforcement Learning (IRL)이다. 최근에는 GAIL과 같이 생성적 적대 신경망(GAN) 구조를 이용해 전문가와 학습자의 분포 차이를 줄이는 방식이 주목받았으며, 이를 행동 정보가 없는 상태-전이 관찰 데이터로 확장한 것이 GAIfO이다.

그러나 GAIfO는 Model-free RL 알고리즘(예: TRPO, PPO)을 기반으로 하기에 데이터 효율성이 매우 낮다는 한계가 있다. 반면, Guided Policy Search (GPS)와 같은 궤적 중심 강화학습은 LQR을 통해 국소적인 선형 모델을 학습하고 이를 이용해 전역적인 정책 학습을 가이드함으로써 높은 샘플 효율성을 달성하였다. 본 논문은 이러한 GPS의 효율성을 IfO 영역으로 가져와 GAIfO의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인
LQR+GAIfO는 전문가의 상태 궤적 $\tau_E = \{s_t\}$만을 입력으로 받으며, 다음의 반복적인 루프를 통해 정책을 최적화한다.

1.  **데이터 수집 및 동역학 학습**: 현재의 Linear Gaussian controller $p(a|s)$를 실행하여 상태-행동 궤적 $\{(s, a, s')\}$를 수집한다. 이후 Bayesian linear regression을 사용하여 각 타임스텝별 동역학 모델 $p(s'|s, a) = \mathcal{N}(F_t[s_t, a_t] + f_t, \sigma^2)$을 학습한다.
2.  **판별자(Discriminator) 업데이트**: 신경망 기반의 판별자 $D_\theta$를 학습시켜 전문가의 상태 전이 $(s, s')$와 학습자의 상태 전이를 구분한다. 이때 학습 안정성을 위해 Wasserstein loss와 Gradient penalty를 사용하며, 목적 함수는 다음과 같다.
    $$\min_{D_\theta} \mathbb{E}_{\tau_{p(a|s)}} [D_\theta(s, s')] - \mathbb{E}_{\tau_E} [D_\theta(s, s')]$$
3.  **비용 함수 근사**: 판별자 $D_\theta$를 비용 함수로 활용한다. 하지만 LQR을 적용하기 위해서는 비용 함수가 이차 형태(quadratic)여야 하므로, 동역학 모델 $f_t$와 판별자 $D_\theta$를 결합한 합성 함수 $C(s_t, a_t) = (D_\theta \circ f_t)(s_t, a_t)$를 구성하고, 이를 2차 테일러 전개를 통해 이차 비용 함수 $c^q(s_t, a_t)$로 근사한다.
    $$c^q(s_t, a_t) = \frac{1}{2} [s_t, a_t]^T \nabla^2_{s,a} C(s_t, a_t) [s_t, a_t] + [s_t, a_t]^T \nabla_{s,a} C(s_t, a_t)$$
4.  **제어기 최적화**: 근사된 이차 비용 함수 $c^q$를 기반으로 LQR을 수행하여 새로운 선형-가우시안 제어기를 생성한다. 이때 급격한 정책 변화를 막기 위해 이전 궤적 분포와의 KL-Divergence를 이용하여 업데이트 크기를 제한한다.

### 주요 구성 요소의 역할
- **Linear Gaussian Controller**: 정책의 복잡도는 낮으나 학습 속도가 매우 빠른 제어기 역할을 한다.
- **Bayesian Linear Regression**: 적은 데이터로도 환경의 동역학을 효율적으로 모델링한다.
- **Wasserstein Discriminator**: 전문가와 학습자의 차이를 측정하는 적대적 보상 함수 역할을 한다.
- **Taylor Expansion**: 비선형적인 신경망 판별자의 출력을 LQR이 처리할 수 있는 수학적 형태로 변환한다.

## 📊 Results

### 실험 설정
- **플랫폼**: 6자유도 로봇 팔인 UR5 및 Gazebo 시뮬레이터.
- **작업**: 정해진 시작 지점에서 Cartesian 공간의 특정 목표 지점까지 도달하는 Reaching task.
- **비교 대상**: PPO(Proximal Policy Optimization)를 기반으로 구현된 GAIfO.
- **평가 지표**: 정규화된 성능 점수 (0.0: 무작위 정책, 1.0: 전문가 수준).

### 주요 결과
1.  **학습 속도 비교**: 시뮬레이션 결과, LQR+GAIfO는 GAIfO보다 훨씬 빠르게 성능이 상승하였으며, 약 30회 반복(iteration) 시점에서 정점에 도달하였다. 이는 제안 방법이 압도적인 샘플 효율성을 가짐을 보여준다.
2.  **일반화 능력**: 전문가 데이터에 포함되지 않은 새로운 목표 지점으로 이동하는 테스트를 수행하였다. 데모 데이터의 수가 많아질수록 LQR+GAIfO의 일반화 성능이 향상되는 경향을 보였다.
3.  **물리적 로봇 적용**: 시뮬레이션과 실제 UR5 로봇 팔에서 동일한 실험을 진행한 결과, 실제 로봇에서의 성능이 시뮬레이션보다 오히려 약간 더 높게 나타났다. 이는 실제 환경의 노이즈가 오히려 탐색(exploration)을 도와 정책 개선을 가속화했을 가능성이 있다.

## 🧠 Insights & Discussion

본 연구는 LQR과 GAIfO의 결합이 모방 학습의 샘플 효율성을 획기적으로 높일 수 있음을 입증하였다. 특히 물리적 로봇에서 실현 가능한 수준의 학습 속도를 보여준 점이 고무적이다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, LQR+GAIfO의 성능은 약 60회 반복 이후 감소하는 경향을 보이는데, 이는 판별자의 성능 향상 속도에 비해 선형 가우시안 제어기의 표현력(capacity)이 제한적이기 때문에 발생하는 현상으로 분석된다. 결국 매우 긴 학습 시간과 방대한 데이터를 투입한다면, 복잡한 정책 표현이 가능한 신경망 기반의 GAIfO가 최종 성능 면에서는 앞설 가능성이 크다.

둘째, 일반화 능력의 경우 판별자가 학습한 일반적인 비용 함수 덕분에 어느 정도 가능하지만, 더 높은 수준의 일반화를 위해서는 단순 LQR을 넘어 딥러닝 정책을 가이드하는 완전한 GPS(Guided Policy Search) 구조로의 확장이 필요하다.

## 📌 TL;DR

본 논문은 적대적 모방 학습(GAIfO)의 고질적인 문제인 샘플 비효율성을 해결하기 위해, 궤적 최적화 기법인 LQR을 결합한 **LQR+GAIfO** 알고리즘을 제안하였다. 이 방법은 판별자의 출력을 이차 비용 함수로 근사하여 LQR로 최적화함으로써, 기존 방식보다 훨씬 적은 데이터와 반복 횟수로 전문가의 행동을 모방할 수 있게 하며, 실제 물리적 로봇 팔(UR5)에서도 성공적으로 작동함을 입증하였다. 향후 딥러닝 정책과 GPS 구조를 통합한다면 효율성과 최종 성능을 모두 잡을 수 있을 것으로 기대된다.