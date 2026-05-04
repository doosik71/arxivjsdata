# Human-In-The-Loop Task and Motion Planning for Imitation Learning

Ajay Mandlekar, Caelan Garrett, Danfei Xu, Dieter Fox (2023)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 로봇의 복잡한 조작(manipulation) 능력을 학습시키기 위한 데이터 수집의 효율성과 시스템의 강건성을 동시에 확보하는 것이다.

기존의 Imitation Learning(IL)은 인간의 시연(demonstration)을 통해 복잡한 기술을 배울 수 있지만, 특히 long-horizon 작업(긴 단계의 작업)의 경우 데이터 수집에 막대한 시간과 노동력이 소모된다는 단점이 있다. 반면, Task and Motion Planning(TAMP) 시스템은 자동화되어 있으며 long-horizon 작업을 해결하는 데 능숙하지만, 정밀한 물리 모델과 지각(perception) 정보가 필요하기 때문에 접촉이 빈번한 contact-rich 작업(예: 정밀 삽입, 뚜껑 닫기 등)에서는 적용이 매우 어렵다.

따라서 본 논문의 목표는 TAMP의 자동화된 계획 능력과 인간 시연의 유연성 및 정밀함을 결합하여, 데이터 수집 효율을 극대화하면서도 contact-rich 작업까지 수행 가능한 하이브리드 시스템인 HITL-TAMP를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **TAMP-gated control mechanism**을 통해 TAMP 시스템과 인간 텔레오퍼레이터(teleoperator) 간의 제어권을 선택적으로 전환하는 것이다.

1. **제어권의 전략적 분배**: 자유 공간 이동(free-space motion)이나 단순 운반과 같이 자동화가 쉬운 구간은 TAMP가 담당하고, 모델링이 어렵고 정밀도가 필요한 contact-rich 구간은 인간이 담당하도록 설계하였다.
2. **Constraint Learning (제약 조건 학습)**: 소수의 시연 데이터로부터 TAMP가 인간에게 제어권을 넘겨주기 전 도달해야 할 최적의 상태(pre-contact pose)를 학습하는 메커니즘을 제안하였다.
3. **Queueing System (대기열 시스템)**: 인간 운영자가 한 번에 한 대의 로봇만 제어하고, 나머지 로봇들은 TAMP가 자동으로 수행하게 함으로써, 한 명의 운영자가 여러 대의 로봇 플릿(fleet)을 동시에 관리할 수 있도록 하여 데이터 수집 처리량을 획기적으로 높였다.
4. **TAMP-gated Policy**: 수집된 인간의 데이터만을 사용하여 정책을 학습시키고, 이를 TAMP 시스템과 결합하여 배포함으로써 전체 작업을 자동화하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들의 한계를 극복하고자 한다.

- **Imitation Learning**: 대규모 데이터셋이 필수적이지만, long-horizon 작업에서 인간이 모든 과정을 시연하는 것은 매우 비효율적이다.
- **TAMP**: 상징적(symbolic) 계획과 연속적(continuous) 모션 계획을 결합하여 복잡한 작업을 수행하지만, contact-rich 상호작용을 모델링하는 것이 매우 어렵다.
- **Hybrid Approaches**: 기존의 일부 연구(예: LEAGUE)는 시뮬레이션 내에서 TAMP를 통해 데이터를 생성하거나 RL을 사용하지만, 실세계의 복잡한 접촉 상황을 해결하는 데는 여전히 한계가 있다.

HITL-TAMP는 TAMP가 인간에게 '언제' 제어권을 넘겨야 하는지를 결정하는 게이트 역할을 수행하게 함으로써, 인간의 개입을 최소화하면서도 시스템의 성공률을 높이는 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 파이프라인

HITL-TAMP는 TAMP 시스템이 전체 작업 계획을 세우고, 계획된 액션 중 '인간의 개입이 필요한 액션'에 도달했을 때만 제어권을 전환하는 구조를 가진다.

- **TAMP 세그먼트**: PDDLStream 프레임워크를 사용하여 논리적 계획을 수립하고, Joint Position Controller를 통해 로봇을 이동시킨다.
- **인간 세그먼트**: Operational Space Controller(OSC)를 사용하여 인간이 스마트폰 인터페이스로 로봇의 end-effector를 6-DoF로 정밀 제어한다.

### 2. Constraint Learning (제약 조건 학습)

TAMP가 인간에게 제어권을 넘기기 전, 인간이 작업을 성공적으로 수행할 수 있는 최적의 시작 지점을 제공해야 한다. 이를 위해 `PreAttach`와 `AttachGrasp`라는 제약 조건을 학습한다.

- **PreAttach**: 인간이 목표 상태(`GoodAttach`)를 달성하기 직전의 상대적 포즈(pose)를 수집한다. 구체적으로는 시연 데이터에서 역방향으로 탐색하여, 로봇이 물체를 잡고 있고 물체 간 거리가 $\delta$ cm 이상 떨어진 첫 번째 상태를 기록한다.
- **AttachGrasp**: 인간이 성공적으로 부착 작업을 수행할 수 있게 하는 유효한 파지(grasp) 포즈의 집합을 수집한다.

### 3. TAMP-Gated Teleoperation 알고리즘

제어 흐름은 다음과 같은 루프를 따른다:

1. 현재 상태 $s$를 관찰하고 목표 $G$ 달성 여부를 확인한다.
2. 달성하지 못했다면 $\text{PLAN-TAMP}(s, G)$를 통해 액션 시퀀스 $\vec{a}$를 생성한다.
3. $\vec{a}$의 각 액션 $a$에 대해:
    - $a$가 자동화 가능한 액션이면 $\text{EXECUTE-JOINT-COMMANDS}(a)$를 수행한다.
    - $a$가 인간의 개입이 필요한 액션이면, 상태가 액션의 후행 조건(postcondition) $a.\text{eff}$를 만족할 때까지 $\text{EXECUTE-TELEOP}()$를 통해 인간이 제어한다.
4. 후행 조건이 만족되면 다시 TAMP로 제어권이 돌아오며 재계획(replanning)을 수행한다.

### 4. 데이터 수집 효율화를 위한 Queueing System

인간 운영자의 유휴 시간을 줄이기 위해 FIFO 큐를 도입하였다.

- **로봇 프로세스**: TAMP 제어 $\to$ 인간 대기 $\to$ 인간 제어의 3가지 상태를 가지며 비동기적으로 작동한다.
- **인간 프로세스**: 큐에서 대기 중인 로봇을 하나씩 꺼내어 제어하고, 작업이 끝나면 다시 큐에 넣는다.
- **이론적 로봇 수**: 인간의 소비율 $R_H$와 TAMP의 생산율 $R_T$를 고려할 때, 필요한 로봇 수 $N_{robot}$은 다음과 같다:
$$N_{robot} \ge 1 + \frac{R_H}{R_T}$$
(인간의 duty cycle $X$를 고려할 경우 $N_{robot} \ge 1 + \frac{R_H}{R_T \cdot (X/100)}$)

### 5. 학습 절차 및 손실 함수

수집된 데이터 중 **인간이 제어한 세그먼트**만을 사용하여 Behavioral Cloning(BC) 정책 $\pi_\theta$를 학습시킨다.

- **목표 함수**:
$$\arg \min_{\theta} \mathbb{E}_{(x,u) \in D} ||\pi_{\theta}(x) - u||^2$$
여기서 $x$는 이미지(RGB) 또는 저차원 상태 정보이며, $u$는 6-DoF end-effector 액션이다.

## 📊 Results

### 1. 실험 설정

- **태스크**: Coffee, Square, Three Piece Assembly, Tool Hang 등 12가지 contact-rich 및 long-horizon 작업.
- **지표**: 데이터 수집량(demos), 작업 성공률(Success Rate, SR), 인간의 인지적 부하(NASA-TLX).
- **비교 대상**: 기존의 단순 텔레오퍼레이션 시스템(Conventional Teleoperation).

### 2. 주요 결과

- **데이터 수집 효율**: 동일 시간 대비 HITL-TAMP가 기존 시스템보다 2.5배에서 4.5배 더 많은 시연 데이터를 수집하였다.
- **학습 성능**:
  - 단 10분의 비전문가 데이터만으로도 75% 이상의 성공률을 보이는 에이전트를 학습시켰다.
  - Coffee 작업의 경우, HITL-TAMP 데이터로 학습한 정책의 성공률이 100%에 도달한 반면, 기존 방식은 76%에 그쳤다.
- **실제 로봇 검증**: 실제 환경에서 Coffee Preparation, Tool Hang 등의 복잡한 작업을 수행하였으며, 특히 Tool Hang 작업에서는 기존 연구(3%)보다 훨씬 높은 64%의 성공률을 기록하였다.
- **사용자 경험**: NASA-TLX 설문 결과, HITL-TAMP가 정신적/물리적/시간적 요구량 및 좌절감이 훨씬 낮게 나타났다.

## 🧠 Insights & Discussion

### 강점 및 통찰

- **상호 보완적 결합**: TAMP가 '거시적 이동'을 담당하고 IL이 '미시적 정밀 조작'을 담당하게 함으로써, 두 방식의 단점을 완벽히 상쇄하였다.
- **지각 오차에 대한 강건성**: TAMP가 정밀한 목표 지점이 아닌 '전-접촉 상태(pre-contact pose)'까지만 이동시키고, 최종 정밀 조작은 시각 기반 정책이나 인간이 처리하므로, TAMP의 입력으로 들어오는 객체 포즈 추정치에 어느 정도 노이즈가 있어도 시스템이 성공적으로 작동한다.
- **모듈성**: 학습된 `attach` 기술은 TAMP의 프레임워크 내에서 모듈로 작동하므로, 새로운 작업에 재학습 없이 조합하여 사용할 수 있는 가능성을 보여주었다.

### 한계 및 비판적 해석

- **사전 지식 의존성**: 어떤 부분이 TAMP로 가능하고 어떤 부분이 인간의 개입이 필요한지에 대한 고수준의 사전 정의가 필요하다. 이를 자동으로 탐지하는 메커니즘(예: 불확실성 추정)은 아직 구현되지 않았다.
- **도메인 제한**: 주로 테이블 위(tabletop) 환경의 단순한 객체들에 집중되어 있어, 더 다양한 환경과 객체에 대한 일반화 성능 검증이 추가적으로 필요하다.
- **모델 의존성**: PDDLStream을 사용하므로 여전히 거친 수준의 객체 모델과 포즈 추정이 필요하다는 가정이 존재한다.

## 📌 TL;DR

본 논문은 TAMP의 자동 계획 능력과 인간의 정밀 조작 능력을 결합한 **HITL-TAMP** 시스템을 제안하였다. TAMP-gated 제어 메커니즘과 큐잉 시스템을 통해 데이터 수집 효율을 3배 이상 높였으며, 수집된 데이터로 학습한 정책을 TAMP와 결합하여 contact-rich한 long-horizon 작업을 성공적으로 수행하였다. 이 연구는 로봇 학습에서 인간의 개입을 최소화하면서도 고품질의 데이터를 효율적으로 수집하고 배포하는 새로운 프레임워크를 제시했다는 점에서 향후 실세계 로봇 운용 및 학습에 중요한 기여를 할 것으로 평가된다.
