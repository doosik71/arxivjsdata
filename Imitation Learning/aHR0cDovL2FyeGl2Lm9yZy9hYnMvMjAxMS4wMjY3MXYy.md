# HILONet: Hierarchical Imitation Learning from Non-Aligned Observations

Shanqi Liu, Junjie Cao, Wenzhou Chen, Licheng Wen, and Yong Liu (2021)

## 🧩 Problem to Solve

본 논문은 전문가의 행동(action) 정보 없이 관측값(observation)만으로 학습하는 Imitation Learning from Observation (ILfO)에서 발생하는 **non-time-aligned(비시간 정렬)** 환경 문제를 해결하고자 한다.

기존의 많은 ILfO 방법론들은 에이전트가 전문가의 시연 궤적을 단계별로(step-by-step) 그대로 따라 하는 것을 목표로 한다. 그러나 실제 환경에서는 시연자와 학습 에이전트의 물리적 능력이 다르거나 환경이 상이하여, 매 시간 단계마다 전문가의 상태를 동일하게 달성하는 것이 불가능한 경우가 많다. 또한, 기존 방법들은 주로 단일 궤적만을 활용하므로 여러 전문가 궤적에서 얻을 수 있는 풍부한 정보를 충분히 활용하지 못한다는 한계가 있다.

따라서 본 연구의 목표는 전문가의 시연 궤적들로부터 현재 상태에서 달성 가능한 **feasible sub-goals(실행 가능한 하위 목표)**를 동적으로 선택하여 학습하는 계층적 모방 학습 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 계층적 강화학습(Hierarchical Reinforcement Learning, HRL) 구조를 도입하여, 전문가의 궤적 전체에서 현재 상황에 적합한 관측값을 하위 목표로 설정하고 이를 달성하도록 유도하는 것이다.

주요 기여 사항은 다음과 같다:

1. **계층적 ILfO 구조 제안**: 고수준 정책(High-level policy)이 전문가 궤적 중 달성 가능한 sub-goal을 선택하고, 저수준 정책(Low-level policy)이 이를 달성하는 구조를 통해 non-time-aligned 환경 문제를 해결한다.
2. **유연한 행동 제어를 위한 보상 구조**: 보상 함수 내의 파라미터 $\alpha$를 조절함으로써, 에이전트가 전문가의 궤적을 엄격하게 모방할지(mimic) 또는 목표 지점까지 더 효율적인 새로운 경로를 탐색할지(explore)를 제어할 수 있게 한다.
3. **학습 효율성 증대 기법**: HRL의 고질적인 문제인 non-stationarity(비정상성)를 해결하기 위해 Hindsight replacement, Asynchronous delayed update, 그리고 차별화된 Replay buffer 크기 설정을 제안하여 샘플 효율성을 높였다.

## 📎 Related Works

### 1. Imitation Learning from Observation (ILfO)

- **Model-based**: dynamics model을 학습하여 전문가의 행동을 추론하는 BCO나 ILPO 등이 있다.
- **Model-free**: GAILfO와 같은 적대적 학습 방식이나, 전문가 상태와 에이전트 상태 간의 유클리드 거리를 이용한 reward engineering 방식(TCN 등)이 존재한다.
- **한계점**: 기존의 reward engineering 방식들은 대개 시간 단계별 정렬을 가정하므로 non-time-aligned 환경에서 성능이 저하되며, 단일 궤적에 의존하는 경향이 크다.

### 2. Hierarchical Reinforcement Learning (HRL)

- HIRO, Option-Critic, FeUdal Networks 등이 대표적이며, 최근에는 HAC와 같이 hindsight 기법을 사용하여 비정상성 문제를 해결하려는 시도가 있었다.
- **차별점**: 기존의 HRL 기반 모방 학습 연구가 있었으나, 본 논문처럼 모든 전문가 궤적 내에서 하위 목표를 동적으로 선택하여 non-time-aligned 문제를 해결하려는 접근은 차별적이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

HILONet은 고수준 정책 $\pi_{high}$와 저수준 정책 $\pi_{low}$의 이층 구조로 구성된다. 두 정책 모두 DDPG(Deep Deterministic Policy Gradient) 알고리즘을 사용하여 학습된다.

- **고수준 정책 ($\pi_{high}$)**: 현재 관측값 $o_t$를 입력받아 전문가 궤적 집합 $D$에서 하위 목표 $o_g$를 선택한다. 출력값은 2차원 액션 공간으로 구성되며, 각각 선택할 궤적의 인덱스와 해당 궤적 내 상태의 인덱스를 나타낸다.
$$o_g = d_{ij} = \pi_{high}(a^h_1, a^h_2 | o_t; \theta_h)$$
- **저수준 정책 ($\pi_{low}$)**: 현재 상태 $o_t$와 고수준 정책이 부여한 하위 목표 $o_g$를 입력받아 환경과 상호작용하는 액션 $a_t$를 출력한다.
$$a_t = \pi_{low}(a_t | o_t, o_g; \theta_l)$$

### 2. 보상 구조 (Reward Structure)

전문가의 액션 라벨이 없으므로 관측값 기반의 보상을 설계하였다.

- **저수준 보상 ($r_{low}$)**: 하위 목표와의 거리 기반 보상과 목표 달성 시 부여되는 희소 보상(sparse reward) $r$을 결합한다.
$$r_{low}(o_t, o_g) = \begin{cases} -\|o_g - o_t\|^2 & \text{if } \|o_g - o_t\| > \epsilon \\ -\|o_g - o_t\|^2 + r & \text{if } \|o_g - o_t\| < \epsilon \end{cases}$$
- **고수준 보상 ($r_{high}$)**: 저수준 정책이 목표를 달성했을 때만 보상을 주며, 전문가 궤적 내에서의 진행도($I(o_g)$)의 증가분을 반영한다.
$$r_{high}(o^g_t, o^g_{t-\Delta t}) = \begin{cases} 1 + \alpha \cdot (I(o^g_t) - I(o^g_{t-\Delta t})) & \text{if } \|o^g_i - o_t\| < \epsilon \\ 0 & \text{if } \|o^g_i - o_t\| > \epsilon \end{cases}$$
여기서 $\alpha$는 에이전트의 행동 패턴을 결정하는 핵심 파라미터이다. $\alpha$가 높으면 전문가의 궤적을 정밀하게 따라가려 하고, 낮으면 목표 지점까지의 새로운 경로를 탐색하려는 경향을 보인다.

### 3. 비정상성(Non-stationarity) 해결 방법

계층적 구조에서 저수준 정책의 변화가 고수준 정책의 학습을 방해하는 문제를 해결하기 위해 다음 세 가지 방법을 도입하였다.

1. **Hindsight Transitions**: 저수준 정책이 원래의 $o_g$는 달성하지 못했더라도 전문가 궤적 내의 다른 상태 $o_{t+\Delta t}$에 도달했다면, 이를 하위 목표로 대체하여 저장함으로써 성공적인 전이(transition) 데이터로 활용한다.
2. **Asynchronous Delayed Update**: 저수준 정책은 매 스텝 학습시키지만, 고수준 정책은 일정 시간 지연(Time-delay) 후에 업데이트하여 저수준 정책이 어느 정도 안정된 후 고수준 정책이 이를 기반으로 학습하게 한다.
3. **Smaller Experience Replay Buffer**: 고수준 정책의 리플레이 버퍼 크기를 작게 유지하여, 과거의 부정확한 저수준 정책 하에서 생성된 오래된 데이터를 빠르게 제거한다.

## 📊 Results

### 1. 실험 설정

- **환경**: MountainCar, LunarLander (단일 목표 작업), Swimmer, 3Dball (순차적 목표 작업), Reacher (혼합 작업) 총 5가지 환경에서 테스트하였다.
- **비교 대상**: GAIL (액션 라벨 사용), GAILfO (관측값만 사용), TSRE (시간 순서대로 모방하는 Baseline)
- **지표**: 환경에서 얻는 누적 리턴(Environmental Return)을 성능 지표로 사용하였다.

### 2. 주요 결과

- **정량적 결과**: 모든 환경에서 HILONet은 GAILfO와 TSRE보다 우수한 성능을 보였으며, 액션 라벨을 사용하는 GAIL과 비교해도 근접한 성능을 나타냈다. 특히 3Dball 환경에서는 GAIL보다 높은 점수를 기록하였다.
- **정성적 결과 (LunarLander)**: 시각화 결과, HILONet은 전문가의 궤적을 단순히 복제하는 것이 아니라, 목표 지점에 도달하기 위한 새로운 최적 경로를 학습하는 능력을 보였다.
- **범용성**: 단일 목표 작업($\alpha=0.5$)과 순차적 목표 작업($\alpha=10$) 모두에서 성공적으로 동작함을 확인하여, 제안한 보상 구조의 유효성을 입증하였다.
- **Ablation Study**: Hindsight replacement와 Time-delay training을 제거했을 때 성능이 급격히 저하되는 것을 통해, 비정상성 해결 기법들이 학습에 필수적임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 ILfO의 고질적인 문제인 non-time-aligned 상황을 해결하기 위해 **'동적 목표 선택'**이라는 계층적 접근 방식을 제안하였다.

**강점**:

- 전문가의 궤적을 그대로 따라가는 대신, 현재 상태에서 달성 가능한 최적의 지점을 하위 목표로 설정함으로써 물리적 능력이 다른 에이전트도 효율적으로 모방 학습을 수행할 수 있게 하였다.
- $\alpha$라는 단순한 파라미터 조절만으로 '정밀 모방'과 '경로 탐색'이라는 서로 다른 성격의 작업을 모두 수행할 수 있는 유연성을 확보하였다.

**한계 및 논의**:

- 실험에서 GAIL과 성능 차이가 적었으나, GAIL은 액션 라벨이라는 매우 강력한 정보가 제공되는 상황이다. 반면 HILONet은 오직 관측값만으로 이 정도 성능을 냈다는 점에서 의미가 크다.
- 다만, 모든 실험이 시뮬레이션 환경에서 이루어졌으며, 실제 로봇 환경에서의 하드웨어 제약이나 노이즈가 섞인 관측값 상황에서도 동일한 강건함(robustness)을 보일지는 추가적인 검증이 필요하다.

## 📌 TL;DR

HILONet은 전문가의 액션 정보 없이 관측값만으로 학습하는 ILfO에서, 시간적 정렬이 맞지 않는 문제를 해결하기 위해 **계층적 강화학습 구조**를 도입한 모델이다. 고수준 정책이 전문가 궤적에서 실행 가능한 하위 목표를 동적으로 선택하고, 저수준 정책이 이를 달성하는 방식을 취한다. 특히 비정상성 해결을 위한 Hindsight 기법과 시간 지연 업데이트를 통해 학습 효율을 높였으며, 단일 목표 및 순차 목표 작업 모두에서 기존 SOTA ILfO 방법론들을 상회하는 성능을 입증하였다. 이 연구는 액션 데이터 확보가 어려운 실제 로봇 제어나 자율주행 분야의 모방 학습에 중요한 기여를 할 것으로 기대된다.
