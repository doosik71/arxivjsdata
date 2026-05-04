# RILe: Reinforced Imitation Learning

Mert Albaba, Sammy Christen, Thomas Langarek, Christoph Gebhardt, Otmar Hilliges, Michael J. Black (2024/2025)

## 🧩 Problem to Solve

본 논문은 고차원 환경(high-dimensional settings)에서 인공지능 에이전트가 복잡한 행동을 습득할 때 발생하는 학습의 어려움을 해결하고자 한다. 기존의 학습 방법들은 각각 다음과 같은 한계를 지닌다.

첫째, 전통적인 Reinforcement Learning (RL)은 보상 함수(reward function)를 설계하기 위해 막대한 수동 작업(manual effort)이 필요하며, 이는 시간 소모가 크고 오류가 발생하기 쉽다.

둘째, Inverse Reinforcement Learning (IRL)은 전문가의 시연(expert demonstrations)으로부터 보상 함수를 추론하여 수동 설계를 대체하지만, 정책 학습과 보상 함수 업데이트를 반복하는 반복적 프로세스(iterative process)를 거치므로 계산 비용이 매우 높다.

셋째, Imitation Learning (IL)은 전문가의 행동과 에이전트의 행동을 직접 비교함으로써 효율성을 높였으나, 고차원 환경에서는 단순히 행동의 유사성만을 판단하는 비교 메커니즘이 학습을 위한 세밀한 피드백(fine-grained feedback)을 제공하지 못한다. 특히 Adversarial Imitation Learning (AIL)이나 AIRL과 같은 방식은 보상 함수가 판별자(discriminator)의 출력에 너무 밀접하게 결합되어 있어, 학습 단계별로 필요한 적응형 가이드를 제공하는 데 한계가 있다.

따라서 본 논문의 목표는 IRL의 적응형 보상 학습 능력과 AIL의 계산 효율성을 결합하여, 고차원 작업에서도 전문가의 복잡한 행동을 효과적으로 모방할 수 있는 밀집 보상 함수(dense reward function) 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Trainer-Student 프레임워크**를 통해 보상 함수 학습과 정책 학습을 협력적인 관계로 재정의하는 것이다.

기존의 AIL이 판별자와 에이전트가 서로를 속이려는 경쟁적(competitive) 관계였다면, RILe는 Trainer 에이전트가 Student 에이전트의 성장 단계에 맞춰 보상 신호를 동적으로 조정하는 협력적(cooperative) 관계를 구축한다. Trainer는 RL을 통해 보상 함수 자체를 학습하며, 판별자의 피드백을 바탕으로 Student가 전문가의 행동에 가까워지도록 유도하는 최적의 보상 전략을 탐색한다. 이를 통해 고차원 작업에서 필수적인 '맥락 민감형 가이드(context-sensitive guidance)'를 제공함으로써 학습 효율과 최종 성능을 극대화한다.

## 📎 Related Works

본 논문은 전문가 시연으로부터 학습하는 두 가지 주요 흐름을 검토한다.

1. **Imitation Learning (IL):** Behavioral Cloning (BC)은 정답 행동을 직접 모방하는 지도 학습 방식이며, GAIL은 판별자를 통해 전문가와 에이전트의 분포 차이를 줄이는 적대적 학습 방식을 사용한다. 하지만 이러한 방식들은 고차원 환경에서 단순한 행동 매칭이나 이진 분류 기반의 신호만으로는 세밀한 안내가 불가능하다는 한계가 있다.
2. **Inverse Reinforcement Learning (IRL):** 전문가의 내재적 보상 함수를 복원하는 것이 목표이다. MaxEnt IRL이나 AIRL 등이 대표적이지만, 정책이 수렴할 때까지 기다렸다가 보상을 업데이트하는 반복 루프 구조로 인해 계산 효율성이 매우 낮다는 점이 문제로 지적된다.

RILe는 IRL의 보상 함수 추론 능력과 AIL의 동시 학습 효율성을 결합하여, 반복적인 재학습 루프 없이도 실시간으로 적응하는 보상 함수를 학습함으로써 기존 연구들과 차별화된다.

## 🛠️ Methodology

RILe는 Student Agent, Trainer Agent, 그리고 Discriminator라는 세 가지 주요 구성 요소로 이루어져 있다. 전체적인 시스템은 Figure 2에 묘사된 바와 같이 상호작용하며 학습된다.

### 1. Student Agent

Student 에이전트는 환경과 상호작용하며 전문가의 행동을 모방하는 정책 $\pi_S$를 학습한다. 특이점은 Student가 환경으로부터 직접 보상을 받는 것이 아니라, Trainer 에이전트의 정책 $\pi_T$가 생성하는 값을 보상으로 사용한다는 점이다.
즉, Student의 보상 $r_S$는 다음과 같이 정의된다.
$$r_S = \pi_T((s_S, a_S))$$
Student의 최적화 목표는 Trainer가 제공하는 보상의 기대값을 최대화하는 것이다.
$$\min_{\pi_S} -\mathbb{E}_{(s_S, a_S) \sim \pi_S} [\pi_T(s_S, a_S)]$$

### 2. Discriminator

판별자 $D_\phi$는 전문가의 상태-행동 쌍 $(s, a) \sim \tau_E$와 Student의 상태-행동 쌍 $(s, a) \sim \pi_S$를 구분하는 이진 분류기이다. 판별자의 목적 함수는 다음과 같다.
$$\max_{\phi} \mathbb{E}_{(s, a) \sim \tau_E} [\log(D_\phi(s, a))] + \mathbb{E}_{(s, a) \sim \pi_S} [\log(1 - D_\phi(s, a))]$$

### 3. Trainer Agent

Trainer 에이전트는 Student를 전문가의 행동으로 유도하는 보상 함수를 학습하는 RL 에이전트이다. Trainer는 Student의 상태-행동 쌍 $s_T = (s_S, a_S)$를 관측하고, 스칼라 값인 $a_T \in [-1, 1]$를 출력하며, 이 값이 곧 Student의 보상 $r_S$가 된다.

Trainer는 자신의 행동(보상 값 부여)이 적절했는지를 판별자의 출력값과 비교하여 학습한다. 판별자의 출력 $D_\phi \in [0, 1]$를 $[-1, 1]$ 범위로 확장하기 위해 스케일링 함수 $\upsilon(x) = 2x - 1$를 사용한다. Trainer의 보상 $R^T$는 다음과 같이 정의된다.
$$R^T = e^{-|\upsilon(D_\phi(s_T)) - a_T|}$$
이 식은 Trainer가 부여한 보상 $a_T$가 판별자가 판단한 전문가 유사도 $\upsilon(D_\phi)$와 일치할수록 높은 보상을 받게 함으로써, Trainer가 판별자의 기준에 맞는 보상 함수를 학습하도록 유도한다. Trainer의 목표는 다음과 같다.
$$\max_{\pi_T} \mathbb{E}_{s_T \sim \pi_S, a_T \sim \pi_T} [e^{-|\upsilon(D_\phi(s_T)) - a_T|}]$$

### 4. 학습 절차 및 안정화 전략

RILe는 Student, Trainer, Discriminator를 동시에 학습시킨다. 다만, 동적 보상 체계로 인한 불안정성을 해결하기 위해 다음 세 가지 전략을 사용한다.

- **Trainer 동결 (Freezing):** Trainer의 Critic 네트워크가 수렴하면 Trainer 학습을 중단하여 과적합을 방지한다.
- **작은 리플레이 버퍼 사용:** Trainer에게는 Student보다 작은 버퍼를 사용하여 최근의 Student 정책 변화에 더 빠르게 적응하도록 한다.
- **Student 탐색 강화:** $\epsilon$-greedy 전략 등을 통해 Student의 탐색률을 높여 Trainer가 더 다양한 상태-행동 쌍에 대한 보상을 학습할 수 있게 한다.

## 📊 Results

### 1. 정성적 및 정량적 분석 (Ablation Studies)

- **보상 함수의 동적 변화:** Maze 환경 실험에서 RILe의 보상 함수는 Student의 진행 단계에 따라 보상 영역을 이동시키며 '커리큘럼'과 같은 역할을 수행함을 확인하였다. 반면 GAIL과 AIRL은 보상 지형이 정적인 모습을 보였다.
- **보상 함수 역동성 지표:** RFDC(보상 분포 변화량), FS-RFDC(고정 상태에서의 보상 변화량) 지표에서 RILe가 가장 높은 적응성을 보였으며, CPR(성능-보상 상관관계) 분석에서도 DRAIL-RILe가 가장 높은 양의 상관관계를 기록하여 학습 효율성을 입증하였다.
- **노이즈 강건성:** MuJoCo Humanoid-v2 환경에서 전문가 데이터에 강한 노이즈($\Sigma=0.5$)가 섞인 경우에도 RILe는 다른 베이스라인들보다 월등히 높은 성능을 유지하였다.
- **공변량 변화(Covariate Shift):** 학습된 보상 함수를 고정하고 환경에 노이즈를 추가했을 때, RILe의 보상 함수가 AIRL보다 더 강건한 성능을 보였다.

### 2. 벤치마크 성능 평가

- **MuJoCo Task:** Humanoid, Walker2d 등 4개 작업에서 RILe는 GAIL, AIRL, IQ-Learn 대비 경쟁력 있거나 우월한 성능을 보였으며, 특히 고차원 작업인 Humanoid에서 그 격차가 두드러졌다.
- **LocoMujoco Task:** 모션 캡처 데이터를 이용한 고차원 로봇 제어 작업(Walk, Carry)에서 RILe는 기존 AIL/IL/IRL 방식들을 크게 상회하였으며, 특히 DRAIL-RILe 변형 모델은 전문가 성능(Expert)에 근접하는 결과를 달성하였다.

## 🧠 Insights & Discussion

본 연구는 보상 함수를 고정된 식이나 단순한 판별자 출력값으로 사용하는 대신, RL을 통해 **'학습 가능한 에이전트(Trainer)'**로 설계함으로써 고차원 모방 학습의 난제를 해결하였다. 특히 Trainer가 Student의 현재 수준을 고려하여 보상을 동적으로 조정하는 메커니즘은 복잡한 행동을 단계적으로 습득하게 하는 커리큘럼 학습 효과를 제공한다.

하지만 논문에서는 동적으로 변화하는 보상 함수로 인해 정책의 안정성을 확보하는 것이 여전히 도전 과제임을 언급한다. Trainer를 중간에 동결하는 방식이 임시방편이 될 수 있으나, 이는 추가적인 적응을 멈추게 하는 트레이드-오프가 존재한다. 또한 판별자가 빠르게 과적합되는 문제 역시 해결해야 할 지점이다. 향후 연구에서는 판별자 없는 보상 학습이나 협력적 다중 에이전트 RL(multi-agent RL)의 도입을 통해 이러한 문제를 해결할 가능성을 제시한다.

## 📌 TL;DR

RILe는 Student(정책 학습)와 Trainer(보상 함수 학습)가 협력하는 새로운 모방 학습 프레임워크이다. Trainer가 RL을 통해 Student의 학습 단계에 맞춘 적응형 보상을 실시간으로 생성함으로써, 기존 AIL/IRL의 계산 효율성 및 정밀도 문제를 동시에 해결하였다. 특히 고차원 로봇 제어 작업에서 전문가 수준에 근접한 성능을 보였으며, 이는 향후 복잡한 물리 시스템의 행동 복제 및 제어 연구에 중요한 기여를 할 것으로 기대된다.
