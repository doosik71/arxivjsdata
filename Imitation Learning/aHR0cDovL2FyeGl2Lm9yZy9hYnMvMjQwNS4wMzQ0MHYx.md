# Robotic Constrained Imitation Learning for the Peg Transfer Task in Fundamentals of Laparoscopic Surgery

Kento Kawaharazuka, Kei Okada, and Masayuki Inaba (2024)

## 🧩 Problem to Solve

본 논문은 복강경 수술(Laparoscopic Surgery)의 자율 수행을 위한 로봇 제어 문제를 다룬다. 특히 복강경 수술 환경에서 발생하는 두 가지 핵심적인 기술적 난제를 해결하고자 한다.

첫째는 **물리적 구속 조건(Kinematic Constraints)**의 문제이다. 복강경 수술에서는 체표면에 형성된 포트(Port)를 지점(Fulcrum)으로 하여 기구를 조작해야 하므로, 로봇 팔의 움직임이 포트 위치에 의해 강하게 제한된다.

둘째는 **깊이 정보 인지(Depth Perception)**의 어려움이다. 실제 수술 환경에서는 단안 카메라(Monocular Camera)를 통해 모니터로 영상을 확인하므로, 작업자가 깊이 방향의 정보를 정확히 파악하기 어렵다. 기존 연구들은 주로 깊이 이미지(Depth Image)나 대상 물체의 모델이 존재한다고 가정했으나, 이는 실제 단안 내시경 환경과는 거리가 있다.

따라서 본 논문의 목표는 깊이 이미지나 정밀한 모델 없이, 단안 이미지와 숙련자의 단일 시연 데이터에서 추출한 구속 조건을 활용하여 복강경 수술의 기본 과제인 Peg Transfer Task를 성공적으로 수행하는 자율 로봇 시스템을 구현하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **구속 기반 모방 학습(Constrained Imitation Learning)**이다. 단순히 많은 양의 데이터를 수집하는 대신, 다음과 같은 전략적 파이프라인을 제안한다.

1. **단일 전문가 시연(Single Exemplary Demonstration)으로부터의 구속 추출**: 숙련자의 정밀한 단일 시연 경로에서 각 단계(Phase)별 움직임의 구속 조건(특히 깊이 방향의 최솟값과 최댓값)을 추출한다.
2. **구속 기반의 데이터 수집**: 추출된 구속 조건을 햅틱 장치의 Force Feedback으로 구현하여, 이후 수집되는 다수의 학습 데이터가 전문가의 경로 범위를 벗어나지 않도록 가이드한다.
3. **학습 효율성 증대**: 정제된 데이터를 통해 단안 이미지 기반의 모방 학습 모델이 깊이 방향의 불확실성을 극복하고 더 높은 정확도로 작업을 수행하게 한다.

## 📎 Related Works

기존의 수술 로봇 연구들은 RRT-connect 기반의 경로 계획, 바늘 가이드 메커니즘 최적화, 혹은 스테레오 이미지 및 엣지 인식을 이용한 자율 제거술(Debridement) 등을 다루어 왔다. 최근에는 딥러닝 기반의 조직 추적, 강화 학습을 이용한 패턴 커팅, 모방 학습을 이용한 안과 수술 등이 제안되었다.

그러나 이러한 기존 접근 방식들의 공통적인 한계는 깊이 이미지나 타겟의 기하학적 모델이 가용하다는 가정을 전제로 한다는 점이다. 본 논문은 이러한 가정 없이 오직 단안 이미지와 전문가의 시연에서 유도된 구속 조건만을 사용하여 자율성을 높였다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 시스템 구성 및 구속 역기구학(Constrained IK)

본 시스템은 두 대의 Franka Emika Panda 로봇 팔과 햅틱 장치(Touch Haptic Device), 그리고 FLS(Fundamentals of Laparoscopic Surgery) 트레이닝 박스로 구성된다.

포트의 물리적 제약을 해결하기 위해 **가상 선형 조인트(Virtual Linear Joint)** 개념을 도입한다. 로봇 손(Robot Hand) 위치 $\mathbf{x}_{\text{hand}}$, 포트 위치 $\mathbf{x}_{\text{port}}$, 포셉 팁(Forceps Tip) 위치 $\mathbf{x}_{\text{forcep}}$ 사이의 관계를 설정하고, 다음과 같은 역기구학(IK)을 계산하여 $\mathbf{x}_{\text{forcep}}$가 목표 위치 $\mathbf{x}_{\text{ref\_forcep}}$에 도달함과 동시에 가상 조인트의 끝단 $\mathbf{x}_{\text{virtual}}$이 포트 위치 $\mathbf{x}_{\text{port}}$와 일치하도록 제어한다.

$$\theta, \theta_{\text{virtual}} = \text{IK}(\mathbf{x}_{\text{forcep}}, \mathbf{x}_{\text{ref\_forcep}}, \mathbf{x}_{\text{virtual}}, \mathbf{x}_{\text{port}})$$

### 2. 구속 기반 데이터 수집 절차

단안 이미지의 깊이 인지 문제를 해결하기 위해 다음의 4단계 절차를 수행한다.

**1) 단계 전환 조건(Phase Transition Condition) 정의**: 전체 동작을 여러 페이즈로 나눈다. 예를 들어 포셉 팁의 속도 $\|\dot{\mathbf{x}}_{\text{forcep}}\|_2$가 일정 시간 동안 임계값 이하로 유지되거나, 그리퍼의 개폐 상태 $h$가 변할 때 페이즈가 전환되는 것으로 정의한다.

**2) 모션 구속(Motion Constraint) 추출**: 단일 전문가 시연에서 각 페이즈 $i$에 해당하는 $z$축(깊이 방향)의 최솟값 $z_0$와 최댓값 $z_2$를 추출하여 구속 조건 $C_i$를 생성한다.

- 예: $C_1: z_1 < z_{\text{ref\_forcep}} < z_2$

**3) Force Feedback 기반 데이터 수집**: 추출된 $C_i$를 햅틱 장치의 피드백 힘 $F_z$로 변환하여 작업자에게 전달한다.
$$F_z = \begin{cases} k_p(z_1 - z_{\text{ref\_forcep}}) & \text{if } z_{\text{ref\_forcep}} < z_1 \\ k_p(z_2 - z_{\text{ref\_forcep}}) & \text{if } z_{\text{ref\_forcep}} > z_2 \end{cases}$$
이를 통해 데이터 수집 단계에서 작업자가 깊이 방향으로 실수하는 것을 방지하고 일관된 데이터를 수집한다.

**4) 모방 학습 수행**: 수집된 데이터를 바탕으로 예측 모델을 학습시킨다.

### 3. RNNPB (Recurrent Neural Network with Parametric Bias)

다양한 동작 속도와 스타일을 학습하기 위해 Parametric Bias $p$를 도입한 네트워크를 사용한다.

- **예측 모델**: $(s_{t+1}, u_{t+1}) = f(s_t, u_t, p)$
  - $s$: AutoEncoder를 통해 12차원으로 압축된 이미지 정보 $\xi$
  - $u$: 좌우 포셉 팁의 목표 위치 $\mathbf{x}_{\text{ref\_forcep}}$(6차원) 및 그리퍼 상태 $h$
  - $p$: 각 시연별로 학습되는 파라메트릭 바이어스(Parametric Bias)
- **네트워크 구조**: 4개의 FC 레이어 $\to$ 2개의 LSTM 레이어 $\to$ 4개의 FC 레이어로 구성된 10층 구조이다.
- **학습**: MSE(Mean Squared Error) 손실 함수를 사용하며, Adam 옵티마이저로 가중치 $W$와 $p_k$를 동시에 최적화한다.

## 📊 Results

### 1. 데이터 수집의 안정성 분석

구속 조건이 없는 'Normal' 수집 방식과 구속 조건이 있는 'Constrained' 수집 방식을 비교한 결과, Constrained 방식에서 $z$축 이동 경로의 분산($\sigma_{\text{ave}}$)이 현저히 낮았다.

- **Left Arm**: Normal (6.56) $\to$ Constrained (5.65)
- **Right Arm**: Normal (4.78) $\to$ Constrained (2.74)

### 2. 작업 수행 성공률

Peg Transfer Task의 세부 단계(물체 집기 $\to$ 전달하기 $\to$ 삽입하기)에 대한 성공률을 측정한 결과, Constrained 데이터를 통해 학습한 모델이 Normal 모델보다 월등히 높은 성능을 보였다. 특히 '물체 집기(Taking)' 단계에서 Normal 방식은 많은 실패가 발생했으나, Constrained 방식은 매우 높은 성공률을 기록하였다.

### 3. 잠재 공간(Latent Space) 분석

LSTM의 잠재 공간을 PCA로 시각화한 결과, Constrained 방식에서는 세 개의 서로 다른 펙(Peg)에 대한 궤적이 명확히 구분되어 나타났다. 반면 Normal 방식에서는 궤적이 서로 겹치거나 모호하게 나타나, 모델이 각 상황에 맞는 정밀한 제어 능력을 습득하지 못했음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 단안 카메라 환경에서 깊이 방향의 조작 어려움을 **'데이터 수집 단계의 물리적 가이드(Force Feedback)'**라는 단순하지만 강력한 방법으로 해결하였다. 학습 알고리즘 자체를 수정하는 대신, 학습 데이터의 질을 강제적으로 높임으로써 모방 학습의 성능을 끌어올린 점이 인상적이다.

**한계점 및 향후 과제**:

- **수동 설정의 의존성**: 페이즈 전환 조건과 구속 변수를 사람이 직접 결정해야 한다는 한계가 있다. 이를 자동화하여 Task-agnostic하게 확장하는 연구가 필요하다.
- **환경의 가변성**: 실제 인체 내부(위장 등)는 개인차가 크고 환경이 유동적이므로, 현재의 고정된 구속 조건보다는 더 적응적인(Adaptive) 시스템이 요구된다.
- **단일 시연의 충분성**: 단 하나의 시연만으로 구속 조건을 추출하는 것이 모든 작업에 일반화될 수 있는지에 대한 추가 연구가 필요하다.

## 📌 TL;DR

이 논문은 단안 내시경 환경의 깊이 인지 문제와 포트의 물리적 제약을 극복하기 위해, **단일 전문가 시연에서 추출한 모션 구속 조건을 햅틱 피드백으로 활용하여 고품질 데이터를 수집하고 이를 RNNPB로 학습**시키는 방법을 제안하였다. 실험 결과, 구속 기반 데이터 수집이 단순 수집보다 훨씬 안정적인 제어 성능과 높은 Task 성공률을 보였으며, 이는 복잡한 깊이 센서 없이도 정밀한 수술 로봇 제어가 가능함을 시사한다.
