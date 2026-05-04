# Unsupervised Meta-Learning for Reinforcement Learning

Abhishek Gupta, Benjamin Eysenbach, Chelsea Finn, Sergey Levine (2020)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)의 학습 속도를 높이기 위해 과거의 경험을 재사용하는 Meta-RL의 고질적인 문제인 **메타-훈련 작업 설계의 부담(Human burden of task design)**을 해결하고자 한다.

기존의 Meta-RL 알고리즘들은 새로운 작업을 빠르게 학습하기 위해 사전에 정의된 작업 분포(Task Distribution)에서 샘플링된 수많은 메타-훈련 작업들을 필요로 한다. 그러나 이러한 작업 분포를 수동으로 설계하는 과정은 매우 번거롭고 많은 양의 감독(Supervision)을 요구하며, 실제 복잡한 현실 세계의 문제에 적용하기 어렵다. 또한, Meta-RL의 성능은 설계된 메타-훈련 작업 분포에 매우 의존적이며, 훈련 시 사용된 분포와 유사한 작업에 대해서만 일반화 성능이 높다는 한계가 있다.

따라서 본 논문의 목표는 사람이 직접 메타-훈련 작업을 설계하지 않고도, 환경과의 상호작용만을 통해 자동으로 작업 분포를 획득하여 새로운 작업에 빠르게 적응할 수 있도록 하는 **Unsupervised Meta-RL** 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **상호 정보량(Mutual Information, MI)에 기반한 작업 제안(Task Proposal) 메커니즘**을 통해 최적의 메타-학습기를 훈련시킬 수 있다는 통찰이다.

주요 기여 사항은 다음과 같다:

1. **Unsupervised Meta-RL 프레임워크 제안**: 수동적인 작업 설계 없이 환경의 역학(Dynamics)만을 이용하여 환경 특화적 학습 절차(Environment-specific learning procedure)를 자동으로 획득하는 방법을 제시한다.
2. **이론적 분석**: 상호 정보량 기반의 작업 제안 방식이 최악의 경우의 후회(Worst-case regret)를 최소화하는 minimax 의미에서의 최적성을 가질 수 있음을 증명한다.
3. **실용적 알고리즘 구현**: DIAYN(Diversity is All You Need)을 통한 작업 획득과 MAML(Model-Agnostic Meta-Learning)을 통한 메타-학습을 결합하여, 수동 설계된 작업 분포를 사용한 Oracle 방식에 근접하는 성능을 보임을 실험적으로 입증한다.

## 📎 Related Works

본 논문은 Meta-RL, Goal Generation, 그리고 Unsupervised Exploration의 접점에 위치한다.

- **Meta-RL**: MAML, RL$^2$ 등 기존 연구들은 빠른 적응을 위해 다수의 작업을 학습하지만, 이는 모두 사람이 정의한 작업 분포에 의존한다.
- **Goal Generation 및 Exploration**: DIAYN이나 VIME와 같은 연구들은 보상 함수 없이 상태 공간을 넓게 탐색하거나 다양한 기술(Skill)을 발견하는 데 집중한다. 하지만 이들은 새로운 작업에 빠르게 적응하는 '학습 방법' 자체를 최적화하는 Meta-learning 관점의 접근은 부족했다.
- **차별점**: 기존의 Goal-conditioned RL은 특정 목표 상태에 도달하는 것에 집중하지만, 본 논문은 임의의 보상 함수(Arbitrary reward functions)에 대해 빠르게 적응할 수 있는 '범용적인 학습 절차'를 획득하는 것을 목표로 한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Unsupervised Meta-RL은 크게 두 가지 구성 요소로 이루어진다: **작업 제안 메커니즘(Task Proposal Mechanism)**과 **메타-학습 알고리즘(Meta-learning Algorithm)**이다.

- **작업 제안 메커니즘**: 잠재 변수 $z \sim p(z)$를 보상 함수 $r_z(s, a)$로 매핑하여, 감독 없이 다양한 작업들을 자동으로 생성한다.
- **메타-학습 알고리즘**: 생성된 작업 분포를 바탕으로, 어떤 새로운 작업이 주어지더라도 빠르게 적응할 수 있는 RL 알고리즘 $f$를 학습한다.

### 2. 이론적 배경 및 최적성

본 논문은 보상 함수가 없는 **Controlled Markov Process (CMP)** $\mathcal{C} = (\mathcal{S}, \mathcal{A}, P, \gamma, \rho)$를 가정한다.

- **최악의 경우의 후회(Worst-case Regret)**:
  특정 작업 분포 $p(r_z)$에 대해 최적의 학습 절차 $f^*$와 현재 학습 절차 $f$ 사이의 기대 보상 차이를 후회(Regret)라고 정의한다. Unsupervised 설정에서는 어떤 작업 분포가 올지 모르므로, 최악의 분포에 대한 후회를 최소화하는 $\min_f \max_{p(r_z)} \text{REGRET}(f, p(r_z))$ 문제를 푼다.
- **Goal-reaching Tasks 분석**:
  분석 결과, 목표 상태 도달 작업의 경우 **상태 공간에 대해 균등한(Uniform) 분포**로 목표를 샘플링하여 학습했을 때 최악의 후회가 최소화됨을 보였다.
- **상호 정보량(Mutual Information)의 역할**:
  잠재 변수 $z$와 최종 상태 $s_T$ 사이의 상호 정보량 $I(s_T; z)$를 최대화하면, 결과적으로 $s_T$의 주변 분포가 균등 분포가 됨을 증명하였다. 즉, MI 최대화는 최적의 Unsupervised 메타-훈련 작업 분포를 생성하는 방법이 된다.

### 3. 실용적 알고리즘 (UML-DIAYN)

이론적 분석을 바탕으로 다음과 같은 파이프라인을 구현한다.

1. **작업 획득 (Task Acquisition)**: DIAYN을 사용하여 판별자(Discriminator) 네트워크 $D_\phi(z|s)$를 학습시킨다. 이 네트워크는 현재 상태 $s$가 어떤 잠재 변수 $z$에 의해 생성되었는지 예측한다.
2. **보상 함수 정의**: 판별자의 출력값을 보상으로 사용한다:
   $$r_z(s, a) = \log D_\phi(z|s)$$
   이를 통해 $z$마다 서로 다른 보상 함수가 정의되어, 다양한 '가상 작업'들이 생성된다.
3. **메타-학습 (Meta-Learning)**: 위에서 정의된 $r_z$를 사용하여 **MAML**을 수행한다. MAML은 새로운 작업에 대해 몇 번의 경사 하강법(Gradient Descent)만으로 최적의 정책을 찾을 수 있는 초기 파라미터 $\theta$를 찾는 알고리즘이다.

## 📊 Results

### 1. 실험 설정

- **환경**: 2D Point Navigation, HalfCheetah (2D locomotion), Ant (3D locomotion).
- **평가 작업**: 메타-훈련 시에는 전혀 사용되지 않은 새로운 목표 지점 도달 또는 목표 속도 달성 작업.
- **비교 대상**:
  - **RL from scratch**: 메타-학습 없이 처음부터 학습.
  - **VIME-init**: VIME로 사전 학습 후 파인튜닝.
  - **Oracle**: 사람이 직접 설계한 최적의 작업 분포로 메타-학습.
  - **UML-Random**: DIAYN 대신 무작위 초기화된 판별자를 사용한 Unsupervised Meta-RL.

### 2. 주요 결과

- **학습 속도 향상**: 모든 환경에서 `UML-DIAYN`은 RL from scratch나 VIME-init보다 훨씬 빠르게 새로운 작업에 적응하였다. 이는 무감독 환경 상호작용을 통해 획득한 Prior가 학습 속도를 획기적으로 높임을 의미한다.
- **Oracle과의 경쟁력**: 2D Navigation과 HalfCheetah에서는 사람이 설계한 Oracle 방식과 거의 대등한 성능을 보였다. Ant 작업에서는 Oracle이 약간 우세했으나, 이는 Unsupervised 작업 제안 알고리즘의 한계일 뿐 Meta-RL 구조 자체의 문제는 아니라고 분석한다.
- **무작위 판별자의 효과**: 놀랍게도 `UML-Random` 역시 RL from scratch보다는 좋은 성능을 보였다. 이는 정교한 MI 최대화 없이도 환경과의 상호작용만으로 어느 정도의 유용한 적응 전략을 학습할 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 시사점

본 논문은 Meta-RL의 가장 큰 병목이었던 '작업 설계' 문제를 해결함으로써, 에이전트가 특정 환경에 최적화된 **'학습하는 법(How to learn)'**을 스스로 깨우칠 수 있음을 보여주었다. 특히 이론적으로 MI 최대화가 최악의 후회를 줄이는 균등 분포 생성으로 이어진다는 점을 연결하여 방법론의 당위성을 확보하였다.

### 한계 및 비판적 해석

- **결정론적 역학 가정**: 이론적 분석이 Deterministic Dynamics(결정론적 역학)에 국한되어 있어, 확률적인 환경에서의 최적성에 대해서는 추가 연구가 필요하다.
- **Trajectory-matching의 단순화**: 임의의 보상 함수를 Trajectory-matching 문제로 치환하여 분석하였으나, 실제 복잡한 보상 함수(Intermediate rewards가 존재하는 경우)에서는 Posterior sampling 방식의 탐색이 최적이 아닐 수 있다.
- **작업 분포의 불일치**: 시각화 결과, DIAYN이 발견한 작업들이 실제 테스트 작업들과 완전히 일치하지 않음에도 성능이 높게 나왔다. 이는 MAML이 어느 정도의 분포 변화(Distribution shift)를 견딜 수 있음을 의미하지만, 동시에 어떤 특성이 전이(Transfer)에 결정적인 역할을 했는지에 대한 추가 분석이 필요하다.

## 📌 TL;DR

본 논문은 사람이 직접 작업을 설계해야 하는 기존 Meta-RL의 한계를 극복하기 위해, **상호 정보량(Mutual Information) 최대화**를 통해 자동으로 작업 분포를 생성하고 학습하는 **Unsupervised Meta-RL**을 제안한다. DIAYN으로 가상 작업을 만들고 MAML로 학습하는 이 방식은, 수동 설계된 작업 없이도 새로운 환경 작업에 대해 매우 빠른 적응 속도를 보이며, 특정 사례에서는 전문가가 설계한 작업 분포를 사용한 성능에 근접하는 결과를 달성하였다. 이 연구는 향후 로봇 제어와 같이 작업 정의가 어려운 도메인에서 효율적인 사전 학습(Pre-training) 방법론으로 활용될 가능성이 높다.
