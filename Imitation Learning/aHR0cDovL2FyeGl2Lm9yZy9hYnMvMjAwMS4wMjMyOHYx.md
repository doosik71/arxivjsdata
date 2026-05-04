# The Past and Present of Imitation Learning: A Citation Chain Study

Nishanth Kumar (2020)

## 🧩 Problem to Solve

본 논문은 전문가의 시연(demonstrations)을 관찰하여 일반적인 기술을 습득하는 Imitation Learning(모방 학습) 분야의 발전 과정을 분석하는 것을 목표로 한다. 모방 학습의 핵심 문제는 학습자가 전문가의 행동을 관찰하지 못한 새로운 상황에서도 전문가처럼 행동할 수 있도록 일반화(generalization)된 정책을 학습하는 것이다.

저자는 지난 30년 동안 모방 학습 방법론이 어떻게 변화해 왔는지 탐구하기 위해, 서로를 인용하며 발전한 4편의 랜드마크 논문을 선정하여 그 핵심 아이디어와 영향력을 분석한다. 이를 통해 단순한 행동 복제에서부터 이론적인 분포 일치(distribution matching) 관점으로 진화한 모방 학습의 흐름을 체계적으로 정리하고자 한다.

## ✨ Key Contributions

본 보고서의 중심적인 기여는 모방 학습의 방법론적 진화를 다음과 같은 논리적 흐름으로 연결하여 분석한 점이다.

1. **Supervised Learning(지도 학습) 관점**: 상태에서 행동으로의 직접적인 매핑을 통한 단순 모방.
2. **Inverse Reinforcement Learning(역강화 학습) 관점**: 전문가의 보상 함수(reward function)를 추론하여 정책을 도출하는 방식.
3. **Generative Adversarial Networks(생성적 적대 신경망) 관점**: 보상 함수 추론의 비효율성을 해결하기 위해 전문가의 상태-행동 분포를 직접 모방하는 방식.
4. **Divergence Minimization(발산 최소화) 관점**: 위의 모든 방법론을 수학적 발산 최소화 문제로 통합하여 이론적 근거를 제시하는 방식.

## 📎 Related Works

논문은 모방 학습의 뿌리가 소프트웨어 개발의 'programming by example'에서 시작되었으며, 이후 로보틱스와 AI 분야로 확장되어 'Learning from Demonstration' 또는 'Imitation Learning'이라는 용어로 정착되었음을 설명한다.

기존의 접근 방식들은 크게 두 갈래로 나뉜다. 하나는 전문가의 결정(decision)을 그대로 따라 하는 지도 학습 방식이고, 다른 하나는 전문가가 최적화하려고 했던 내재적 목표(보상 함수)를 찾아내려는 역강화 학습 방식이다. 저자는 이 두 방식이 가진 각각의 한계(지도 학습의 오차 누적 문제, 역강화 학습의 데이터 효율성 및 연산 비용 문제)를 지적하며, 이를 해결하기 위해 등장한 최신 방법론들의 차별점을 논의한다.

## 🛠️ Methodology

본 논문은 4개의 핵심 연구를 통해 모방 학습의 방법론적 발전을 설명한다.

### 1. ALVINN (Pomerleau, 1989)

모방 학습을 가장 단순한 형태의 Supervised Learning 문제로 정의한다. 전문가의 궤적 $\mathcal{T} = [(S_t, a^E_t)]$ 가 주어졌을 때, 상태 $S_t$를 행동 $a_t$로 매핑하는 함수를 학습한다.

- **구조**: Deep Neural Network(DNN)를 사용하여 비디오 데이터와 레이저 거리 측정기 데이터를 입력받고, 차량이 주행해야 할 곡률(curvature) 벡터를 출력한다.
- **특징**: 실제 데이터 수집의 어려움을 해결하기 위해 '도로 시뮬레이터'를 구축하여 학습 데이터를 생성하였다.

### 2. Apprenticeship Learning via IRL (Abbeel and Ng, 2004)

지도 학습 대신 Reinforcement Learning(RL)을 활용한다. 전문가의 행동 뒤에 숨겨진 보상 함수 $R^E(s, a)$를 먼저 찾아내고, 이를 최대화하는 정책 $\pi$를 학습하는 Inverse Reinforcement Learning(IRL) 방식을 제안한다.

- **절차**: 전문가 궤적 $\rightarrow$ IRL $\rightarrow$ 보상 함수 $R^E$ 추출 $\rightarrow$ RL $\rightarrow$ 최적 정책 $\pi$ 도출.
- **한계**: 보상 함수를 학습하기 위해 상태에 대한 특징 집합(features $\phi^s$)이라는 인간의 추가적인 정의가 필요하다.

### 3. Generative Adversarial Imitation Learning (GAIL, Ho and Ermon, 2016)

IRL-RL의 반복적인 최적화 과정이 매우 느리다는 점을 해결하기 위해 GAN 구조를 도입한다.

- **핵심 아이디어**: IRL과 RL을 거쳐 정책을 만드는 과정은 결과적으로 학습된 정책 $\pi$의 상태-행동 분포 $\rho^\pi$를 전문가의 분포 $\rho^{\pi_E}$와 일치시키는 과정과 같다.
- **방법**: GAN의 판별자(Discriminator)가 전문가의 분포와 학습자의 분포를 구분하게 하고, 생성자(Generator/Policy)는 판별자를 속이도록 학습함으로써 직접적으로 분포를 일치시킨다. 이는 매 반복마다 RL을 수렴시킬 필요가 없어 데이터 효율성이 매우 높다.

### 4. Divergence Minimization Perspective (Ghasemipour et al., 2019)

모든 모방 학습 방법을 수학적인 분포 간의 발산(divergence) 최소화 문제로 통합한다.

- **수학적 정의**:
  - **Supervised Learning**: 조건부 분포 간의 발산을 최소화한다.
    $$\min \text{div}(\rho^{\pi_E}(a_t|s_t), \rho^\pi(a_t|s_t))$$
  - **GAIL 및 IRL 기반 방법**: 결합 분포(joint distribution) 간의 발산을 최소화한다.
    $$\min \text{div}(\rho^{\pi_E}(s_t, a_t), \rho^\pi(s_t, a_t))$$
- **FAIRL**: 전문가 시연 없이도 사용자가 직접 정의한 상태-행동 분포 $\rho$를 학습자가 따르도록 하는 방식을 제안한다.

## 📊 Results

각 방법론의 실험 결과는 다음과 같이 요약된다.

- **ALVINN**: CMU 캠퍼스 내 400m 구간에서 초속 0.5m의 속도로 자율 주행에 성공하였으며, 이는 당시의 수동 설계 알고리즘과 대등한 성능이었다.
- **Abbeel & Ng (IRL)**: 시뮬레이션 환경에서 차선 유지, 타 차량 회피, 의도적 충돌 등 5가지의 복잡한 주행 행동을 학습함을 보였다.
- **GAIL**: 고차원 시뮬레이션 환경(Humanoid, Ant)에서 단 50개의 타임스텝 궤적만으로 걷기 동작을 학습하였다. 특히 Behavior Cloning(지도 학습) 및 기존 IRL 방법론보다 훨씬 적은 전문가 데이터만으로 우수한 성능을 달성하였다.
- **Ghasemipour et al.**: 이론적 분석을 통해 결합 분포를 일치시키는 방식(GAIL 등)이 조건부 분포를 일치시키는 방식(SL)보다 고차원 공간에서 더 강건함을 실험적으로 증명하였다. 또한 FAIRL을 통해 전문가 데이터 없이 수동 정의된 분포만으로 'Fetch' 및 'Pusher' 작업의 정책을 학습할 수 있음을 보여주었다.

## 🧠 Insights & Discussion

본 분석 보고서를 통해 도출된 주요 통찰은 다음과 같다.

첫째, **오차 누적 문제(Compounding Errors)**이다. 지도 학습 기반의 모방 학습은 매 순간의 결정만을 복제하므로, 초기에 발생한 작은 오차가 시간이 흐를수록 누적되어 궤적에서 크게 벗어나는 문제가 발생한다. 반면, 보상 함수나 분포를 학습하는 IRL/GAIL 방식은 전문가의 '목표'를 학습하므로 스스로 오차를 수정할 수 있는 능력을 갖는다.

둘째, **데이터 효율성 대 환경 상호작용의 트레이드오프**이다. GAIL은 전문가의 시연 데이터 사용량은 획기적으로 줄였으나, 정책을 학습시키기 위해 환경과 상호작용해야 하는 횟수는 여전히 많다는 한계가 있다. 반면 지도 학습은 환경 상호작용 없이 시연 데이터만으로 학습이 가능하다.

셋째, **이론적 통합의 중요성**이다. 서로 다른 접근법으로 보였던 SL과 IRL/GAIL이 결국 어떤 분포의 발산을 최소화하느냐의 차이였다는 분석은, 향후 더 효율적인 발산 측정 지표(예: KL divergence vs JS divergence)를 선택함으로써 성능을 개선할 수 있는 방향성을 제시한다.

## 📌 TL;DR

본 논문은 모방 학습의 역사를 **지도 학습 $\rightarrow$ 역강화 학습(IRL) $\rightarrow$ 적대적 모방 학습(GAIL) $\rightarrow$ 발산 최소화 이론**으로 이어지는 흐름으로 분석하였다. 핵심 기여는 단순한 행동 복제에서 벗어나 상태-행동의 결합 분포를 일치시키는 것이 고차원 작업에서 더 효율적이고 강건하다는 이론적/실험적 근거를 제시한 것이다. 이 연구는 향후 전문가 시연 없이 언어 명령 등으로 상태-행동 분포를 직접 정의하여 학습시키는 새로운 모방 학습 방법론의 가능성을 시사한다.
