# CROSS-DOMAIN IMITATION LEARNING VIA OPTIMAL TRANSPORT

Arnaud Fickinger, Samuel Cohen, Stuart Russell, Brandon Amos (2022)

## 🧩 Problem to Solve

본 논문은 **Cross-Domain Imitation Learning (CDIL)**, 즉 전문가(Expert)와 모방 에이전트(Agent)의 신체 구조(Embodiment)나 형태(Morphology)가 서로 다를 때, 전문가의 시연 데이터를 통해 에이전트를 학습시키는 문제를 해결하고자 한다.

전통적인 모방 학습(Imitation Learning)은 전문가와 에이전트가 동일한 상태 공간(State Space)과 행동 공간(Action Space), 그리고 전이 역학(Transition Dynamics)을 공유한다고 가정한다. 그러나 실제 환경에서는 인간의 동작을 로봇이 모방하거나, 서로 다른 하드웨어를 가진 로봇 간에 지식을 전달해야 하는 경우가 많다. 이때 두 도메인의 상태-행동 공간의 차원(Dimensionality)이 서로 다를 수 있어, 단순히 궤적(Trajectory)이나 분포를 직접적으로 비교하는 것이 불가능하다는 점이 핵심 난제이다.

기존의 CDIL 연구들은 두 도메인 간의 매핑(Mapping) 함수를 학습시키기 위해, 두 에이전트가 모두 최적으로 동작하는 '프록시 태스크(Proxy Tasks)' 데이터를 필요로 했다. 하지만 이는 새로운 로봇으로의 전이나 한 번도 본 적 없는 전문가의 데이터를 사용하는 상황에서 적용 가능성을 크게 제한한다. 따라서 본 논문의 목표는 **프록시 태스크 없이도 서로 다른 도메인 간의 구조적 유사성을 이용하여 모방 학습을 수행하는 방법론을 제안하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Gromov-Wasserstein (GW) 거리**를 사용하여 서로 다른 공간에 존재하는 상태-행동 분포를 정렬하고 비교하는 것이다.

기존의 Wasserstein 거리가 동일한 공간 내의 두 분포 사이의 거리를 측정하는 것과 달리, GW 거리는 각 공간 내부의 **상대적 거리 구조(Pairwise Distances)**를 비교한다. 즉, "전문가 공간에서 $x$와 $x'$가 얼마나 떨어져 있는가"와 "에이전트 공간에서 $y$와 $y'$가 얼마나 떨어져 있는가"의 유사성을 측정함으로써, 절대적인 좌표값이 아닌 공간의 '기하학적 구조'를 맞추는 방식이다. 이를 통해 공유된 잠재 공간(Shared Latent Space)을 학습하거나 프록시 태스크를 사용하지 않고도 서로 다른 차원의 공간을 직접 비교할 수 있게 되었다.

## 📎 Related Works

- **Imitation Learning (IL):** Behavioral Cloning, Inverse RL, GAIL 등이 있으며, 최근에는 분포 매칭 문제로 접근하는 Primal Wasserstein IL(PWIL)과 Sinkhorn IL(SIL)이 제안되었다. 본 연구는 이러한 Wasserstein 기반 IL을 Gromov-Wasserstein 설정으로 확장하여 Cross-Domain 설정으로 넓힌 것이다.
- **Domain Transfer:** 기존의 전이 학습 연구들은 주로 상태 공간 간의 선형 매핑을 찾거나(Manifold Alignment), 정렬된 시연 데이터(Paired/Time-aligned demonstrations)를 사용하여 매핑 함수를 학습했다. 하지만 이러한 방법들은 프록시 태스크가 필수적이라는 한계가 있다.
- **차별점:** 제안 방법인 GWIL은 명시적인 교차 도메인 매핑 함수를 학습하지 않으며, 프록시 태스크 없이 단일 전문가 시연만으로도 학습이 가능하다.

## 🛠️ Methodology

### 전체 파이프라인
GWIL은 전문가의 상태-행동 점유 분포(Occupancy Measure)와 에이전트의 분포 사이의 GW 거리를 최소화하는 방향으로 에이전트의 정책을 학습시킨다. 직접적인 최적화가 어렵기 때문에, GW 거리를 기반으로 한 **의사 보상(Pseudo-reward)**을 설계하고 이를 강화학습(RL) 알고리즘으로 최적화하는 구조를 가진다.

### 핵심 구성 요소 및 수식 설명

**1. Metric MDP**
본 논문은 MDP에 거리 함수 $d: S \times A \to \mathbb{R}^+$가 추가된 **Metric MDP** 개념을 도입하여, 상태-행동 공간 내의 거리 개념을 정의한다.

**2. Gromov-Wasserstein (GW) 거리**
두 메트릭 측정 공간 $(X, d_X, \mu_X)$와 $(Y, d_Y, \mu_Y)$ 사이의 GW 거리는 다음과 같이 정의된다:
$$GW((X,d_X,\mu_X), (Y,d_Y,\mu_Y))^2 = \min_{u \in U(\mu_X, \mu_Y)} \sum_{x,x' \in X} \sum_{y,y' \in Y} |d_X(x,x') - d_Y(y,y')|^2 u_{x,y} u_{x',y'}$$
여기서 $u$는 두 분포 사이의 결합(Coupling)이며, 이 수식은 두 공간의 내부 거리 구조가 얼마나 유사한지를 측정한다.

**3. Isomorphic Policies (동형 정책)**
두 정책 $\pi^E$와 $\pi^A$가 **Isomorphic(동형)**하다는 것은, 두 정책의 점유 분포 지원 집합(Support) 사이에 거리 구조를 보존하는 등거리 사상(Isometry) $\phi$가 존재하여 $\rho^{\pi^A}$가 $\rho^{\pi^E}$의 푸시포워드(Push-forward) 측도가 되는 상태를 의미한다.

**4. GW-based Pseudo-reward**
RL 에이전트가 GW 거리를 최소화하도록 유도하기 위해, 다음과 같은 의사 보상 $r^{GW}$를 정의한다:
$$r^{GW}(z_A) = -\frac{1}{\rho^{\pi}(z_A)} \sum_{z_E, z'_E \in Z_E, z'_A \in Z_A} |d_E(z_E, z'_E) - d_A(z_A, z'_A)|^2 u^\star_{z_E, z_A} u^\star_{z'_E, z'_A}$$
여기서 $z$는 상태-행동 쌍 $(s, a)$를 의미하며, $u^\star$는 GW 거리를 최소화하는 최적 결합이다. 에이전트는 이 보상을 최대화함으로써 전문가의 궤적 구조와 가장 유사한 궤적을 생성하도록 학습된다.

### 학습 절차 (Algorithm 1)
1. 전문가의 시연 $\tau$와 각 도메인의 거리 척도 $d_E, d_A$를 입력받는다.
2. 에이전트의 정책 $\pi_\theta$와 가치 함수 $V_\theta$를 초기화한다.
3. 에이전트가 에피소드 $\tau'$를 수집한다.
4. 수집된 $\tau'$와 전문가의 $\tau$ 사이의 GW 거리를 계산한다.
5. 계산된 GW 거리를 바탕으로 각 상태-행동 쌍에 대한 의사 보상 $r$을 할당한다.
6. **Soft Actor-Critic (SAC)** 알고리즘을 사용하여 해당 보상을 최적화하도록 $\pi_\theta$와 $V_\theta$를 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업:** Mujoco 및 DeepMind Control Suite 환경의 연속 제어 작업.
- **평가 지표:** 전문가의 행동을 얼마나 성공적으로 복원하는지 정성적/정량적으로 평가.
- **기준선:** 동일 도메인에서 Wasserstein 거리를 최소화하는 베이스라인 및 SAC 단독 학습.

### 주요 결과
논문은 세 가지 시나리오를 통해 GWIL의 유효성을 검증하였다:

1. **강체 변환(Rigid Transformation):** 전문가의 미로 환경을 반전(Reflection)시킨 에이전트 환경에서 실험했다. 이론(Theorem 1)에 따라 GWIL은 성공적으로 최적 정책을 복원하였다.
2. **약간 다른 상태-행동 공간:** Pendulum(전문가, 3차원)과 Cartpole(에이전트, 5차원) 간의 전이를 실험했다. 차원이 다름에도 불구하고 GWIL은 Cartpole에서 Pendulum의 스윙업 동작과 유사한 최적 행동을 학습했다.
3. **매우 다른 상태-행동 공간:** Cheetah(전문가)와 Walker(에이전트)라는 완전히 다른 형태의 로봇 간 전이를 실험했다. Walker는 Cheetah의 달리기 구조를 모방하여 앞으로 가거나 뒤로 가는 동작을 학습했으며, 이는 GWIL이 매우 추상적인 구조적 유사성만으로도 모방이 가능함을 보여준다.

### 추가 분석
- **희소 보상(Sparse Reward) 환경:** 외부 보상이 거의 없는 환경에서도 GWIL의 의사 보상이 가이드 역할을 하여, 동일 도메인의 Wasserstein 기반 모방 학습과 경쟁 가능한 수준의 성능을 보였다.
- **계산 효율성:** GW 거리 계산의 병목은 궤적의 길이에 의존하며 상태-행동 공간의 차원에는 영향을 받지 않는다. 실험 결과, GWIL의 학습 시간은 Wasserstein 기반 방법과 유사하며, 일부 경우 SAC 단독 학습보다 빠르게 목표 속도에 도달했다.

## 🧠 Insights & Discussion

### 강점
본 논문은 프록시 태스크라는 무거운 제약 조건 없이, **최적 운송(Optimal Transport)의 기하학적 관점**을 도입하여 서로 다른 도메인 간의 모방 학습을 가능하게 했다. 특히 차원이 다른 공간 간의 비교를 위해 GW 거리를 사용한 점이 매우 효율적이며, 이론적 보장(Theorem 1)을 통해 최적성 복원 가능성을 제시한 점이 우수하다.

### 한계 및 비판적 해석
1. **등거리 사상(Isometry)의 한계:** Theorem 1에서 명시하듯, GWIL은 최적 정책을 '등거리 사상' 범위 내에서만 복원한다. 이는 실제 환경에서 완전히 동일한 최적해가 아닌, 기하학적으로 유사한 형태의 해가 도출될 수 있음을 의미한다. (예: Cheetah를 모방하는 Walker가 앞으로 가는 최적해와 뒤로 가는 차선책을 모두 가질 수 있음)
2. **시간적 구조 무시:** GW 거리는 집합(Set) 간의 구조적 유사성을 측정하므로, 궤적 내의 **시간적 순서(Temporal Ordering)** 정보를 완전히 무시한다. 이는 동작의 '순서'가 중요한 복잡한 작업에서는 치명적인 한계가 될 수 있으며, 논문의 결론에서도 Dynamic Time Warping(DTW) 등의 도입 필요성을 언급하고 있다.
3. **거리 함수 $d$에 대한 의존성:** 본 실험에서는 유클리드 거리를 사용했으나, 실제 복잡한 고차원 공간에서 의미 있는 거리 함수 $d$를 어떻게 정의하느냐에 따라 성능이 크게 좌우될 것으로 보인다.

## 📌 TL;DR

본 논문은 전문가와 에이전트의 신체 구조와 상태-행동 공간이 서로 다른 **Cross-Domain Imitation Learning** 문제를 해결하기 위해, 공간의 상대적 거리 구조를 비교하는 **Gromov-Wasserstein (GW) 거리** 기반의 방법론을 제안한다. 프록시 태스크 없이 단일 시연 데이터만으로도 서로 다른 차원의 공간 간 모방이 가능함을 이론적으로 증명하고 실험적으로 검증하였다. 이 연구는 로봇의 형태가 바뀌어도 기존의 동작 지식을 전이시킬 수 있는 가능성을 열어주었으며, 향후 시간적 구조를 고려한 거리 척도로 확장될 경우 더욱 높은 실용성을 가질 것으로 기대된다.