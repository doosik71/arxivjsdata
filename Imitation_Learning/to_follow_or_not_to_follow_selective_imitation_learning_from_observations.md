# To Follow or not to Follow: Selective Imitation Learning from Observations

Youngwoon Lee, Edward S. Hu, Zhengyu Yang, and Joseph J. Lim (2019)

## 🧩 Problem to Solve

본 논문은 전문가의 시연(demonstration)을 통해 기술을 습득하는 모방 학습(Imitation Learning)에서, 시연자와 학습자 간의 신체적 능력 차이나 환경적 제약으로 인해 발생하는 문제를 해결하고자 한다. 기존의 많은 모방 학습 방법론은 시연의 모든 단계를 순차적으로 따라 하는 방식을 취한다. 하지만 학습자와 시연자의 로봇 하드웨어가 다르거나, 학습자의 환경에 시연 당시에는 없었던 장애물이 존재하는 경우, 시연의 모든 상태를 그대로 재현하는 것은 불가능하다.

따라서 본 연구의 목표는 전문가의 동작(action) 정보 없이 오직 관측값(observation)으로만 구성된 시연 데이터로부터, 학습자가 자신의 상황에 맞게 도달 가능한 상태만을 선택적으로 모방하여 과업을 완수하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Selective Imitation Learning from Observations (SILO)**라는 계층적 강화학습(Hierarchical Reinforcement Learning) 구조를 설계한 것이다.

단순히 시연을 그대로 복제하는 것이 아니라, 현재 상태에서 도달 가능한 목표 상태(sub-goal)를 시연 데이터 내에서 유연하게 선택하고, 그 목표에 도달하기 위한 동작을 학습하는 방식을 제안한다. 이를 통해 학습자는 시연자의 속도와 자신의 속도가 다르더라도 대응할 수 있으며, 환경적 제약으로 인해 도달 불가능한 시연의 일부 구간을 능동적으로 건너뛸 수 있는 유연성을 갖게 된다.

## 📎 Related Works

기존의 모방 학습 연구들은 주로 상태-동작 쌍(state-action pairs)을 사용하는 방식이었으나, 이는 전문가의 동작 데이터를 수집하는 비용이 매우 크다는 단점이 있다. 동작 정보 없이 관측값만으로 학습하는 Learning from Observations (LfO) 연구들이 제안되었으나, 이들 대부분은 시연과 학습자의 실행 과정이 시간적으로 정렬(temporally aligned)되어 있다고 가정한다.

반면, 본 논문이 제안하는 SILO는 다음과 같은 점에서 기존 연구와 차별화된다. 첫째, 시간적 정렬 가정 없이 학습자가 시연의 진행 속도를 조절하거나 일부 프레임을 건너뛸 수 있다. 둘째, 시연자와 학습자의 물리적 능력이 다르더라도 도달 가능한 상태만을 선택적으로 모방함으로써 전이 가능성(transferability)을 높였다.

## 🛠️ Methodology

### 전체 시스템 구조

SILO는 상위 수준의 **Meta Policy**와 하위 수준의 **Low-level Policy**로 구성된 계층적 구조를 가진다.

1. **Meta Policy**: 시연 데이터 $\tau = \{o^\tau_1, o^\tau_2, \dots, o^\tau_T\}$ 중에서 현재 관측 상태 $o_t$에서 도달 가능한 최적의 서브골(sub-goal) $o^\tau_g$를 선택한다.
2. **Low-level Policy**: Meta Policy가 선택한 서브골 $o^\tau_g$를 입력으로 받아, 해당 상태에 도달하기 위한 구체적인 동작 $a_t$를 생성한다.

### 주요 메커니즘 및 방정식

Meta Policy는 현재 상태와 시연의 미래 상태들 사이의 관계를 학습하며, 다음과 같은 Q-함수 기반의 최적화 문제를 푼다.

$$g \sim \pi^{meta}(g|o_t, \{o^\tau_{i+1}, \dots, o^\tau_{i+w}\}; \theta) = \text{argmax}_{g \in \{i+1, i+w\}} \gamma^{g-i-1} Q(o_t, o^\tau_g; \theta)$$

여기서 $w$는 Meta window의 크기로, Meta Policy가 서브골을 선택할 수 있는 탐색 범위이다. $\gamma$는 할인 인자(discounting factor)로, 너무 많은 프레임을 건너뛰지 않고 최대한 시연을 밀접하게 따르도록 유도한다.

Low-level Policy는 목표 조건부 정책(goal-conditioned policy) $\pi^{low}(a|o_t, o^\tau_g; \phi)$로 구현되며, 서브골에 도달할 때까지 동작을 생성한다. 도달 여부는 다음 조건으로 판단한다.

$$\|o^\tau_g - o_{t+1}\| < \epsilon$$

### 학습 절차 및 손실 함수

- **Meta Policy 학습**: Low-level Policy가 선택된 서브골에 성공적으로 도달하면 $+1$의 보상을 받고, 실패하면 $0$의 보상을 받는다. 이를 통해 도달 불가능한 상태는 피하고, 최대한 많은 시연 상태를 달성하도록 Double DQN을 통해 학습한다.
- **Low-level Policy 학습**: 동작 레이블이 없으므로, 서브골 도달 여부만을 보상으로 사용하는 Soft Actor-Critic (SAC) 알고리즘을 사용한다. 또한, 데이터 효율성을 높이기 위해 가상의 목표를 생성하여 학습하는 Hindsight Experience Replay (HER)를 적용한다.
- **Embedding**: 서로 다른 환경에서 수집된 데이터를 비교하기 위해 Raw state, 예측된 3D 좌표(Predicted 3D location), AprilTag 등을 임베딩 공간으로 사용하여 상태 간의 유사도를 측정한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**:
  - **Obstacle Push**: 장애물을 피해 블록을 목표 지점까지 밀기 (시연 궤적에 장애물이 배치됨).
  - **Pick-and-Place**: 블록을 집어 옮기기 (학습자의 팔 도달 범위가 시연자보다 짧음).
  - **Furniture Assembly**: 가구 부품을 정렬하여 조립하기.
  - **Obstacle Push (Real)**: 실제 Sawyer 로봇을 이용하여 인간의 시연을 모방하되 장애물을 회피하기.
- **비교 대상 (Baselines)**:
  - **Sequential**: 시연의 다음 프레임을 무조건 서브골로 선택하는 방식.
  - **Random-skip**: 윈도우 내에서 무작위로 서브골을 선택하는 방식.
- **지표**: 성공률(Success rate) 및 시연 프레임 충족률(Coverage).

### 주요 결과

- **정량적 결과**: 모든 시뮬레이션 작업에서 SILO가 베이스라인보다 월등한 성능을 보였다. 특히 Sequential 방식은 장애물이나 물리적 제약으로 인해 도달 불가능한 상태가 존재할 경우 성공률이 $0.0$으로 수렴했다. Random-skip은 운 좋게 장애물을 피할 때만 성공하여 낮은 성공률을 기록했다.
- **실제 로봇 실험**: 실제 환경의 Obstacle Push 작업에서 SILO는 약 $0.25$의 성공률을 기록하며, 무작위 건너뛰기 방식($0.1$)보다 효과적으로 장애물을 우회하여 목표에 도달함을 입증했다.
- **Ablation Study**:
  - 학습 시연 데이터의 수가 10개만 있어도 높은 성능을 보였으며, 데이터 수를 늘려도 성능 차이가 미미했다.
  - Meta window 크기를 늘렸을 때, 더 많은 프레임을 건너뛰어야 하는 상황에서도 유연하게 대응할 수 있음을 확인했다.

## 🧠 Insights & Discussion

### 강점

SILO의 가장 큰 강점은 **유연한 모방(Flexible Imitation)**이다. 시연자와 학습자의 물리적/환경적 차이를 '서브골의 선택적 채택'이라는 메커니즘으로 해결했다. 특히 Pick-and-place 실험에서 Meta Policy가 없으면 로봇이 블록을 집지 않고 밀어버리는 지역 최솟값(local minimum)에 빠지는 경향이 있었으나, Meta Policy를 통해 이를 방지하고 정답 궤적을 빠르게 학습하는 것을 확인했다.

### 한계 및 논의

본 연구에서는 상태 유사도를 측정하기 위해 단순한 유클리드 거리($\epsilon$ 임계값)를 사용했다. 하지만 Obstacle Push 실험에서 3D 위치 예측기(position predictor)를 사용했을 때 성능이 저하된 점은, 임베딩 공간의 정확도가 Meta Policy의 결정과 Low-level Policy의 도달 판단에 직접적인 영향을 미침을 시사한다. 즉, 복잡한 환경일수록 단순한 거리 기반의 보상보다는 정교한 시각적 임베딩이나 보상 함수 설계가 필요할 수 있다.

## 📌 TL;DR

본 논문은 시연자와 학습자의 환경/신체적 차이로 인해 발생하는 '도달 불가능한 상태' 문제를 해결하기 위해, 시연 중 도달 가능한 상태만을 선택적으로 모방하는 **SILO (Selective Imitation Learning from Observations)** 프레임워크를 제안한다. Meta Policy가 도달 가능한 서브골을 선택하고 Low-level Policy가 이를 달성하는 계층적 구조를 통해, 동작 정보 없는 관측값만으로도 유연하게 기술을 습득할 수 있음을 입증했다. 이 연구는 향후 인간의 비디오 시연을 로봇이 자신의 능력에 맞춰 적응적으로 학습하는 연구에 중요한 기반이 될 것으로 보인다.
