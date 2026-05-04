# Interactive Imitation Learning in State-Space

Snehal Jauhri, Carlos Celemin, Jens Kober (2020)

## 🧩 Problem to Solve

본 논문은 Imitation Learning(IL, 모방 학습)에서 발생하는 데이터 품질 문제와 특히 전문가가 아닌 일반인이 에이전트를 가르칠 때 겪는 어려움을 해결하고자 한다. 기존의 IL 및 Interactive IL 기법들은 주로 Action-space(행동 공간)에서 시연 데이터나 피드백을 제공받는다. 그러나 로봇 팔의 관절 토크나 각도와 같은 구체적인 행동 값은 일반인(Non-expert)이 직관적으로 이해하거나 제공하기 매우 어렵다.

인간은 일반적으로 작업을 배울 때 정밀한 행동 값보다는 상태의 전이(State transition), 즉 "물체 쪽으로 이동하라"와 같은 State-space(상태 공간) 상의 변화를 통해 학습한다. 따라서 본 연구의 목표는 행동 공간이 아닌 상태 공간에서의 인간 피드백을 활용하여 에이전트의 행동을 학습시키고 개선하는 새로운 Interactive Learning 기법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **TIPS(Teaching Imitative Policies in State-space)**라는 프레임워크를 제안한 것이다. TIPS의 중심 아이디어는 다음과 같다.

1.  **State-space Corrective Feedback**: 시연자가 에이전트의 현재 상태를 관찰하고, 특정 상태 차원의 값을 증가시키거나 감소시켜야 한다는 단순한 이진 신호를 통해 가이드를 제공한다.
2.  **Indirect Inverse Dynamics**: 상태 공간의 피드백을 행동 공간의 명령으로 변환하기 위해, 직접적인 Inverse Dynamics Model(IDM) 대신 Forward Dynamics Model(FDM)을 이용한 간접적인 방식을 채택한다. 이는 상태 전이가 불가능한 경우나 부분적인 상태 정보만 제공된 경우의 모호성을 해결한다.
3.  **Non-expert Friendly**: 정밀한 수치 대신 '방향성'만으로 피드백을 줄 수 있게 하여, 전문가가 아닌 사람도 직관적으로 에이전트를 가르칠 수 있도록 설계하였다.

## 📎 Related Works

기존의 Interactive IL 연구들은 주로 다음과 같은 방식으로 접근하였다.
- **Corrective Action Labels/Feedback**: 에이전트가 수행하는 행동에 대해 수정된 행동 값을 제공하는 방식이다. 하지만 앞서 언급했듯 행동 공간에서의 피드백은 직관적이지 않다.
- **Evaluative Feedback**: 행동의 좋고 나쁨을 점수로 매기는 방식이나, 여러 하위 최적(sub-optimal) 행동들 사이에서 일관된 점수를 부여하는 것이 모호하다는 한계가 있다.
- **Imitation from Observation (IfO)**: 행동 정보 없이 상태 궤적만을 통해 학습하는 방법이다. 많은 IfO 기법이 상태 전이를 행동으로 매핑하기 위해 Inverse Dynamics Model(IDM)을 사용한다.

TIPS는 IfO의 상태 전이 개념과 Interactive Learning의 프레임워크를 결합하였다. 특히, 기존의 IDM이 가진 한계(불가능한 상태 전이 처리 불가, 부분 상태 정보의 모호성)를 극복하기 위해 FDM을 활용한 행동 샘플링 방식을 도입하여 차별점을 두었다.

## 🛠️ Methodology

### 1. Corrective Feedback (상태 공간 피드백)
인간 시연자는 매 시간 단계 $t$마다 상태의 각 차원에 대해 $h_t \in \{-1, 0, +1\}$ 형태의 이진 신호를 제공한다. 여기서 $0$은 피드백 없음, $+1$은 값의 증가, $-1$은 값의 감소를 의미한다. 이를 통해 계산된 인간이 원하는 목표 상태 $s^{des}_{t+1}$은 다음과 같다.

$$s^{des}_{t+1} = s_t + h_t \cdot e$$

여기서 $e$는 각 상태 차원에 대해 설정된 에러 상수(error constant) 하이퍼파라미터이다. 시연자는 전체 상태가 아닌 일부 관찰 가능한 부분 상태(partial state)에 대해서만 피드백을 제공할 수 있다.

### 2. Mapping State Transitions to Actions (행동 매핑)
현재 상태 $s_t$에서 목표 상태 $s^{des}_{t+1}$로 이동하기 위한 최적의 행동 $a^{des}_t$를 찾기 위해 간접적인 역동역학(Indirect Inverse Dynamics) 방식을 사용한다. 

먼저, 학습된 Forward Dynamics Model(FDM)인 $f$를 사용하여 가능한 행동 후보군 $a \in A$들을 샘플링하고, 각 행동이 가져올 다음 상태 $\hat{s}_{t+1} = f(s_t, a)$를 예측한다. 이후 예측된 상태와 목표 상태 사이의 거리가 최소가 되는 행동을 선택한다.

$$a^{des}_t = \arg \min_{a} \| f(s_t, a) - s^{des}_{t+1} \|$$

이 과정에서 $N_a$개의 행동 샘플을 균일하게 샘플링하여 계산한다.

### 3. Training Mechanism (학습 절차)
정책 $\pi(s)$는 피드포워드 인공신경망으로 구현되며, 학습은 다음 두 단계로 진행된다.

- **초기 모델 학습 단계 (Initial Model-Learning Phase)**: 무작위 탐색 정책 $\pi^e$를 실행하여 경험 샘플 $\{s_i, a_i, s_{i+1}\}$을 수집하고, 이를 통해 초기 FDM $f_\theta$를 학습시킨다.
- **교습 단계 (Teaching Phase)**: 
    - 시연자가 피드백 $h_t$를 주면, 위의 수식을 통해 $a^{des}_t$를 계산하고 이를 즉시 실행한다.
    - $(s_t, a^{des}_t)$ 쌍을 Demonstration Buffer $D$에 저장하고, 즉시 정책 $\pi_\phi$를 업데이트한다.
    - 주기적으로 $D$에서 샘플링한 배치를 사용하여 정책을 추가 학습시킨다.
    - 에피소드가 끝날 때마다 수집된 경험 데이터를 통해 FDM $f_\theta$를 지속적으로 업데이트하여 모델의 정확도를 높인다.

## 📊 Results

### 1. 시뮬레이션 평가
OpenAI Gym의 CartPole, Reacher, LunarLanderContinuous 작업을 통해 성능을 평가하였다.
- **성능 비교**: TIPS는 Behavioral Cloning(BC), GAIL 및 행동 공간 기반의 D-COACH보다 높은 누적 보상(Return)을 기록하였다. 특히 CartPole과 Reacher에서 학습 효율과 최종 성능이 월등히 높았다.
- **인간 부하(Task Load)**: NASA-TLX 설문 결과, TIPS를 사용한 시연자들이 D-COACH(행동 공간 피드백)를 사용했을 때보다 정신적 요구도(Mental Demand)와 좌절감(Frustration)이 유의미하게 낮게 나타났다. (예: CartPole에서 정신적 요구도 약 40% 감소)

### 2. 실제 로봇 검증 (KUKA LBR iiwa)
- **Fishing Task**: 흔들리는 공을 컵에 넣는 작업이다. 약 60 에피소드 이후 성공적으로 작업을 수행했으며, 시간이 지남에 따라 시연자의 피드백 빈도가 줄어드는 경향을 보였다.
- **Laser Drawing Task**: 레이저 포인터로 화이트보드에 글자를 그리는 작업이다. 약 80 에피소드 학습 후 기준 궤적과 유사한 글자를 그리는 데 성공하였다.

## 🧠 Insights & Discussion

**강점 및 의의**
TIPS는 전문가가 아닌 사용자도 상태 공간에서의 직관적인 피드백만으로 복잡한 로봇 제어 정책을 학습시킬 수 있음을 입증하였다. 특히 FDM을 이용한 간접 매핑 방식은 행동 공간의 복잡성을 시연자로부터 격리시켜 인지적 부하를 크게 줄였다는 점이 고무적이다.

**한계 및 논의사항**
1.  **계산 비용**: 현재의 행동 선택 방식은 행동 공간 전체에서 샘플링을 수행하므로, 행동 공간의 차원이 높아질 경우 계산량이 기하급수적으로 증가하여 실시간성을 보장하기 어렵다.
2.  **모델 의존성**: FDM의 정확도가 낮으면 계산된 행동 $a^{des}_t$가 부정확해지며, 이는 시연자의 추가적인 수정 노력을 유발한다. 이를 해결하기 위해 더 효율적인 탐색 전략(smarter exploration)이 필요하다.
3.  **작업 특성**: LunarLander와 같이 시스템이 매우 불안정하거나 상태-행동 관계가 복잡한 경우, 상태 공간 피드백만으로는 정밀한 제어(예: 정확한 착륙)를 가르치는 데 한계가 있을 수 있음이 관찰되었다.

## 📌 TL;DR

본 논문은 비전문가가 로봇을 쉽게 가르칠 수 있도록 상태 공간(State-space)에서 이진 피드백을 받아 정책을 학습시키는 **TIPS** 프레임워크를 제안하였다. Forward Dynamics Model을 이용해 상태 전이 요청을 최적의 행동으로 변환하는 방식을 통해, 기존 행동 공간 기반 학습보다 시연자의 인지적 부담을 낮추고 에이전트의 성능을 향상시켰다. 이 연구는 향후 고차원 행동 공간에서의 효율적인 행동 샘플링 기법이 보완된다면, 실제 산업 현장에서 전문가 없이도 로봇을 빠르게 튜닝하는 도구로 활용될 가능성이 높다.