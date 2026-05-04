# A Dual Approach to Imitation Learning from Observations with Offline Datasets

Harshit Sikchi, Caleb Chuck, Amy Zhang, Scott Niekum (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 전문가의 행동(Action) 정보가 없는 상태에서 관찰(Observation) 데이터만을 이용하여 에이전트를 학습시키는 **Learning from Observations (LfO)** 문제이다. 특히, 전문가의 행동 데이터가 부족하거나 없는 상황에서 에이전트가 기존에 보유한 임의의 품질을 가진 오프라인 데이터셋(Offline Dataset)을 활용하여 전문가의 행동을 모방하도록 하는 것을 목표로 한다.

이 문제는 다음과 같은 이유로 중요하다. 첫째, 로봇의 형태(Morphology)가 복잡하거나 직관적이지 않을 때 전문가가 에이전트의 Action 공간에서 직접 시연을 하는 것은 매우 어렵다. 둘째, 튜토리얼 비디오나 다른 로봇의 데이터와 같이 Observation만 존재하는 데이터셋은 풍부하지만, 이를 효율적으로 활용하는 방법은 제한적이다. 

기존의 LfO 방식들은 Inverse Dynamics Model(IDM)을 통해 행동을 추론하거나 Discriminator를 통해 의사 보상(Pseudo-reward)을 생성하는 중간 단계 모델을 학습시킨다. 그러나 오프라인 설정에서는 데이터 제한으로 인해 이러한 중간 모델의 오차가 누적되는 **Compounding Errors** 문제가 발생하며, 이는 최종 정책 학습 및 배포 단계에서 성능 저하의 핵심 원인이 된다.

## ✨ Key Contributions

본 논문의 핵심 기여는 중간 단계의 일단계(One-step) 모델 학습을 완전히 배제하고, **Duality(쌍대성)** 원리를 이용하여 전문가의 상태 방문 분포(Visitation Distribution)를 직접 일치시키는 **DILO (Dual Imitation Learning from Observations)** 알고리즘을 제안한 것이다.

DILO의 중심 아이디어는 상태 $s$에서 다음 상태 $s'$로의 전이가 전문가의 방문 분포와의 장기적 발산(Divergence)에 어떤 영향을 미치는지 정량화하는 **Multi-step Utility Function** $V(s, s')$를 직접 학습하는 것이다. 이를 통해 LfO 문제를 단순한 Actor-Critic 구조의 최적화 문제로 환원시켰으며, 이는 다음과 같은 이점을 제공한다.
1. 중간 모델(IDM, Discriminator) 학습을 생략하여 compounding error를 방지한다.
2. 임의의 하위 최적(Suboptimal) 오프라인 데이터를 활용하여 off-policy 학습이 가능하다.
3. 단일 플레이어(Single-player) 목적 함수를 최적화함으로써 학습의 안정성과 성능을 높인다.

## 📎 Related Works

### 기존 LfO 접근 방식 및 한계
- **Discriminator 기반 방식 (GAIL, SMODICE 등):** 전문가와 에이전트의 상태 방문 분포를 일치시키기 위해 Discriminator를 학습하고 이를 보상 함수로 사용한다. 그러나 오프라인 설정에서는 Discriminator가 과적합(Overfitting)되기 쉬우며, 여기서 발생한 오차가 RL 과정에서 누적되어 성능을 크게 떨어뜨린다.
- **IDM 기반 방식 (BCO 등):** 오프라인 데이터를 이용해 $s, s'$로부터 $a$를 예측하는 Inverse Dynamics Model을 학습한 뒤, 전문가 궤적에 행동 라벨을 붙여 Behavior Cloning(BC)을 수행한다. 이 방식은 IDM의 예측 오류가 BC의 compounding error와 결합되어 증폭되며, 오프라인 데이터셋에 포함된 복구 행동(Recovery behaviors)을 활용하지 못한다는 한계가 있다.

### 기존 연구와의 차별점
DILO는 분포 일치(Distribution Matching) 문제를 선형 제약 조건이 있는 볼록 최적화(Convex Program) 문제로 정의하고, 이를 Dual form으로 변환하여 해결한다. 이는 기존의 Dual RL이나 DICE 계열 방법론의 통찰을 LfO로 확장한 것이며, 특히 전문가의 Action 정보 없이 $\{s, s'\}$ 쌍의 joint visitation distribution을 매칭함으로써 Action-free 학습을 가능하게 했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
DILO는 크게 두 단계의 프로세스로 구성된다. 첫째, 전문가의 방문 분포를 모방하기 위한 유틸리티 함수 $V$를 학습하는 단계이며, 둘째, 학습된 $V$를 기반으로 오프라인 데이터셋에서 정책 $\pi$를 추출하는 단계이다.

### 상세 방법론 및 방정식

#### 1. Primal Objective: Joint Visitation Distribution Matching
에이전트와 전문가의 $\{s, s'\}$ joint visitation distribution을 일치시키는 것을 목표로 한다. 여기서 $\tilde{d}^\pi(s, s', a')$는 정책 $\pi$ 하에서 상태 $s$에서 $s'$로 전이하고 행동 $a'$를 취할 확률 분포이다. 목표 함수는 다음과 같이 $f$-divergence를 최소화하는 것이다.
$$\min_{\pi} D_f(\tilde{d}^\pi(s, s', a') \| \tilde{d}^E(s, s', a'))$$

#### 2. Dual Objective: DILO
위의 제약 조건이 있는 최적화 문제를 Lagrangian Duality를 통해 제약 조건이 없는 Dual 문제로 변환한다. DILO의 목적 함수는 다음과 같다.
$$\min_{V} \beta(1-\gamma)\mathbb{E}_{s,s' \sim \tilde{d}^0} [V(s, s')] + \mathbb{E}_{s,s' \sim \text{Mix}_\beta(\tilde{d}^E, \rho)} [f^*( \gamma \mathbb{E}_{s'' \sim p(\cdot|s', a')} [V(s', s'')] - V(s, s') )] - (1-\beta)\mathbb{E}_{s,s' \sim \rho} [ \gamma \mathbb{E}_{s'' \sim p(\cdot|s', a')} [V(s', s'')] - V(s, s') ]$$
여기서 $V: S \times S \to \mathbb{R}$는 전이의 유용성을 나타내는 Lagrange dual variable이며, $f^*$는 $f$-divergence의 conjugate function이다. $\text{Mix}_\beta$는 전문가 데이터 $\tilde{d}^E$와 오프라인 데이터 $\rho$의 혼합 분포를 의미한다.

#### 3. Policy Extraction
학습된 유틸리티 함수 $V^*$를 사용하여, 오프라인 데이터셋 $\rho$ 상에서 $V^*$ 값을 최대화하는 방향으로 정책을 학습한다. 이를 위해 Value-weighted regression을 사용하며 손실 함수는 다음과 같다.
$$\mathcal{L}(\psi) = -\mathbb{E}_{s, a, s' \sim \rho} [e^{\tau V^*(s, s')} \log \pi_\psi(a|s)]$$
이 과정에서 $\tau$는 온도 매개변수로, $V^*$ 값이 높은 전이의 행동을 더 강하게 학습하도록 유도한다.

#### 4. 학습 절차 및 알고리즘 흐름
1. **V-network 학습:** DILO 목적 함수를 최소화하도록 $V_\phi$를 학습한다. 이때 학습 안정성을 위해 **Orthogonal Gradient Update (ODICE)** 방식을 채택하여 $V(s, s')$와 $V(s', s'')$ 사이의 그래디언트 충돌을 해결한다.
2. **Policy-network 학습:** 학습된 $V$를 가중치로 하여 $\pi_\psi$를 업데이트한다.
3. 위 과정을 수렴할 때까지 반복한다.

## 📊 Results

### 실험 설정
- **데이터셋:** MuJoCo (D4RL, Robomimic) 및 실제 로봇(UR5e) 환경.
- **비교 대상 (Baselines):** BC, IQ-Learn, ReCOIL (Action-labeled), ORIL, SMODICE (Action-free).
- **지표:** 누적 보상(Cumulative Return), 성공률(Success Rate).

### 주요 결과
1. **MuJoCo 시뮬레이션:** 24개의 데이터셋 전반에서 DILO가 기존 LfO 방식인 SMODICE, ORIL보다 월등한 성능을 보였다. 특히 전문가 데이터가 매우 적은 'few-expert' 설정이나 고차원 관찰 공간(Dextrous Manipulation)에서 강점을 보였다.
2. **이미지 관찰 학습 (Robomimic):** SMODICE는 고차원 이미지 공간에서 Discriminator의 과적합으로 인해 성능이 급락했으나, DILO는 하이퍼파라미터 튜닝 없이도 이미지 기반 모방 학습에서 유의미한 성능 향상을 달성했다.
3. **실제 로봇 실험 (Air Hockey):** 
   - **Safe Object Manipulation:** 장애물을 피해 목표물로 이동하는 작업에서 DILO가 가장 높은 성공률을 기록했다.
   - **Dynamic Puck Hitting:** 매우 정밀한 타이밍과 역학 이해가 필요한 작업에서 BCO와 SMODICE는 거의 실패했으나, DILO는 높은 성공률을 보이며 복잡한 Inverse Dynamics 문제에서도 강건함을 증명했다.

## 🧠 Insights & Discussion

### 강점 및 해석
DILO의 가장 큰 강점은 **중간 모델의 제거**이다. 기존 방식들이 "행동을 예측"하거나 "보상을 추론"하는 단계를 거치며 오차를 생성했던 반면, DILO는 "어떤 상태로 전이하는 것이 유리한가"라는 유틸리티 함수를 직접 학습함으로써 compounding error의 고리를 끊었다. 또한, Dual-RL의 특성상 off-policy 데이터의 분포 변화(distribution shift)에 덜 민감하여 하위 최적 데이터셋에서도 효과적으로 전문가의 행동을 추출할 수 있다.

### 한계 및 비판적 논의
1. **관찰 공간 일치 가정:** 본 논문은 에이전트와 전문가의 관찰 공간이 동일하다고 가정한다. 실제 환경에서 카메라 각도가 다르거나 다른 Embodiment를 가질 경우, 단순한 분포 매칭만으로는 한계가 있으며 이를 해결하기 위해 Semantic Space에서의 매칭이 필요할 것이다.
2. **전문가의 최적성 가정:** 전문가는 항상 최적이라고 가정하지만, 실제 데이터에는 전문가의 편향(Bias)이 포함될 수 있다.
3. **보수적 행동 (Conservatism):** 실험 분석에서 DILO가 BCO보다 보수적으로 행동하여 목표 지점 직전에 멈추거나 속도가 느린 경향이 발견되었다. 이는 $\tau$ 매개변수 설정에 따른 외삽(extrapolation) 정도의 조절 문제로 보이며, 적응형 $\tau$ 선택 방법이 향후 연구 과제로 남아있다.

## 📌 TL;DR

본 논문은 전문가의 Action 정보 없이 Observation만으로 학습하는 LfO 문제에서, 기존의 중간 모델(IDM, Discriminator) 학습으로 인한 오차 누적 문제를 해결하기 위해 **DILO**라는 Dual 기반 알고리즘을 제안했다. DILO는 $\{s, s'\}$ joint visitation distribution을 직접 매칭하는 유틸리티 함수를 학습함으로써, 복잡한 모델링 없이도 오프라인 데이터셋을 활용해 고성능의 모방 정책을 생성한다. 시뮬레이션과 실제 로봇 실험을 통해 고차원 이미지 입력 및 복잡한 물리 역학 환경에서도 기존 방식보다 훨씬 강건하고 뛰어난 성능을 입증하였다.