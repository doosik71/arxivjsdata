# Descriptor State Space Modeling of Power Systems

Yitong Li, Timothy C. Green, Yunjie Gu (2023)

## 🧩 Problem to Solve

전력 시스템의 동특성 분석과 모델링에 널리 사용되는 표준 상태 공간(Standard State Space) 모델은 시스템의 제로(Zero) 수가 폴(Pole) 수를 초과하지 않는 'Proper System'만을 표현할 수 있다는 치명적인 제한이 있다. 전력 시스템의 많은 구성 요소들은 'Terminal-inductive' 특성을 가지는데, 이는 시스템의 입력, 출력, 상태 변수를 자유롭게 선택할 수 없게 만든다.

이러한 제한으로 인해 서로 다른 서브시스템을 연결할 때 포트 불일치(Port mismatch) 문제가 발생한다. 기존 연구에서는 이를 해결하기 위해 매우 큰 가상 저항(Virtual resistor)이나 매우 작은 가상 커패시터(Virtual capacitor)를 추가하여 전압이나 전류를 정의하는 방식을 사용했다. 그러나 이러한 가상 소자의 추가는 물리적 구조를 왜곡하여 모델링 오차를 유발하고 모델 구축의 유연성을 떨어뜨린다.

전달 함수(Transfer function) 방식은 Proper 및 Improper 시스템을 모두 표현할 수 있어 대안으로 제시되었으나, 고차원 다입출력(MIMO) 시스템에서 수치적 정확도가 낮고 계산 속도가 느리다는 단점이 있다. 무엇보다 전달 함수는 상태 정보가 은닉되어 있어, 불안정성의 근본 원인을 파악하는 상태 참여도 분석(State participation analysis)이 매우 어렵다. 따라서 본 논문의 목표는 Proper와 Improper 시스템을 모두 수용하면서도 물리적 상태 정보를 유지할 수 있는 모델링 프레임워크를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 자동 제어 이론에서 제안되었으나 전력 공학 분야에서는 거의 활용되지 않았던 Descriptor State Space(또는 Implicit State Space)를 전력 시스템 모델링에 도입하는 것이다.

핵심 기여 사항은 다음과 같다.

1. Proper 및 Improper 시스템을 모두 모델링할 수 있는 Descriptor State Space 프레임워크를 제안하여, 가상 소자 추가 없이 서브시스템 간의 포트 불일치 문제를 해결하였다.
2. Descriptor State Space 모델의 역행렬(Inverse), 연결(Connection), 변환(Transform)을 위한 수학적 알고리즘을 유도하였다.
3. 제안된 알고리즘이 전체 시스템 모델 내에서 각 서브시스템의 상태 변수를 보존(Preserve)하도록 설계함으로써, 시스템 불안정성의 근본 원인과 모드 참여도를 분석할 수 있게 하였다.

## 📎 Related Works

기존의 전력 시스템 모델링은 크게 두 가지 방향으로 진행되었다. 첫째는 표준 상태 공간 모델을 이용한 방식이다. 이는 물리적 변수와 직접 연결된 'White-box' 모델로서 모드 및 참여도 분석이 용이하지만, 앞서 언급한 Proper 시스템으로의 제한으로 인해 포트 불일치 해결을 위해 가상 소자를 추가하는 한계가 있었다.

둘째는 임피던스/어드미턴스 기반의 전달 함수 모델링 방식이다. 이 방식은 블랙박스 분석이 가능하고 Improper 시스템을 자연스럽게 표현할 수 있으나, 상태 변수 정보가 손실되어 참여도 분석이 어렵고 고차원 시스템에서 수치적 불안정성이 나타나는 한계가 있다. 본 논문은 Descriptor State Space를 통해 이 두 방식의 장점(물리적 해석 가능성 및 모델링 유연성)을 동시에 취하고자 한다.

## 🛠️ Methodology

### 1. Descriptor State Space의 정의

표준 상태 공간 모델이 $\dot{x}=Ax+Bu, y=Cx+Du$로 표현되는 반면, Descriptor State Space는 다음과 같이 정의된다.

$$E \dot{x} = Ax + Bu$$
$$y = Cx + Du$$

여기서 $E$는 추가된 행렬이며, $E$가 가역적(Invertible)일 때는 표준 상태 공간으로 변환 가능하므로 Proper 시스템을 나타낸다. 반면 $E$가 특이 행렬(Singular matrix)인 경우, 이는 Improper 시스템을 나타내며 표준 상태 공간으로 직접 변환할 수 없다.

### 2. 모델 조작 알고리즘

논문은 모듈형 모델링을 위해 다음과 같은 알고리즘을 제안한다.

- **Inverse (역모델 생성):** 입력 $u$와 출력 $y$를 바꾸기 위해, 기존의 입력 $u$를 새로운 가상 상태 변수로 추가하는 방식을 사용한다. 물리적으로는 어드미턴스 모델(Proper)을 임피던스 모델(Improper)로 바꾸기 위해 $C_v=0$인 가상 커패시터를 병렬로 추가하는 것과 같다.
- **Connection (연결):** 합(Sum), 추가(Append), 행렬 추가(Matrix append), 피드백(Feedback) 등의 규칙을 통해 서브시스템들을 결합한다. 이를 통해 인덕터의 직렬 연결이나 커패시터의 병렬 연결과 같은 Improper 시스템의 결합을 자연스럽게 처리한다.
- **Transformation (표준 상태 공간으로의 변환):** 물리적으로 구현 가능한 시스템은 결국 Proper 시스템이므로, 분석을 위해 다시 표준 상태 공간으로 변환하는 절차가 필요하다. $E$가 가역적이지 않을 경우, $A_{22}$의 영공간(Null space)을 정의하는 행렬 $N$을 도입하여 가상 상태 변수를 제거하고 실제 상태 변수만을 남기는 변환 식을 유도하였다.

### 3. 상태 추적 및 참여도 분석

본 방법론의 가장 큰 특징은 모든 조작 과정에서 상태 변수가 추적된다는 점이다.

- 역모델 생성 시: $\text{New state} = \text{Old state} \cup \{u\}$
- 모델 변환 시: $\text{New state} \subset \text{Old state}$
- 모델 연결 시: $\text{New state} = \bigcup \text{Subsystem states}$

최종 모델의 상태 변수는 항상 특정 서브시스템의 물리적 상태로 매핑될 수 있으며, 이를 통해 다음과 같은 참여도 계수(Participation factor)를 계산하여 불안정성의 원인을 분석한다.

$$\text{Participation} = \psi_{ik} \phi_{ji}$$

여기서 $\phi$와 $\psi$는 각각 Descriptor State Space의 우측 및 좌측 고유벡터(Right and Left Eigenvector)이다.

## 📊 Results

### 1. 실험 설정

수정된 IEEE 14-bus 시스템을 대상으로 수치 분석 및 EMT(Electromagnetic Transient) 시뮬레이션을 수행하였다. 해당 시스템은 3개의 동기 발전기(SG)와 3개의 인버터 기반 자원(IBR, Type-IV 풍력 발전)으로 구성된다.

### 2. 주요 분석 결과

- **인덕턴스 동기 모드 (Synchronous Mode):** $50\text{Hz}$ 부근의 모드를 분석한 결과, 감쇠가 낮은 모드는 $X/R$ 비율이 매우 높은(20) 변압기 권선(Branch 7-9)의 참여도가 높았으며, 감쇠가 적절한 모드는 $X/R$ 비율이 낮은(1.97) 선로(Branch 6-13)의 참여도가 높음을 확인하였다.
- **SG 스윙 모드 (Swing Mode):** $2.2\text{Hz}$ 모드는 SG1과 SG3의 스윙에 의해, $3.5\text{Hz}$ 모드는 SG6의 스윙에 의해 지배됨을 확인하였다. 이는 각 발전기의 로터 감쇠(Rotor damping) 계수를 변경했을 때의 고유값 궤적(Mode locus)과 EMT 시뮬레이션 결과와 정확히 일치하였다.
- **IBR 제어 상호작용 모드 (Control Interaction Mode):** $24.7\text{Hz}$ 모드는 IBR8의 AC 전류 제어, PLL, DC-link 제어 루프 간의 상호작용으로 발생함을 밝혀냈다. 특히 전류 제어 대역폭을 좁히거나, PLL/DC-link 제어 대역폭을 넓혔을 때 시스템이 불안정해지는 경향을 고유값 분석과 EMT 시뮬레이션을 통해 입증하였다.

## 🧠 Insights & Discussion

본 논문은 전력 시스템 모델링에서 오랫동안 간과되었던 'Proper system'의 제약을 Descriptor State Space라는 수학적 도구를 통해 해결하였다.

**강점:**

- 가상 소자를 추가하여 억지로 모델을 맞추던 기존의 관행을 없애고, 모델링 오차를 제거하였다.
- 전달 함수의 유연성과 상태 공간 모델의 해석 가능성을 동시에 확보하였다.
- 특히, 복잡한 IBR-SG 혼성 그리드에서 특정 제어 루프나 물리적 소자가 시스템 전체의 진동 모드에 어떻게 기여하는지 정량적으로 분석할 수 있는 경로를 제시하였다.

**한계 및 논의:**

- 본 논문은 선형 상태 공간 모델을 기반으로 하므로, 시스템의 비선형성이 매우 강한 운전 영역에서의 분석에는 한계가 있을 수 있다.
- Descriptor State Space의 계산 복잡도가 표준 상태 공간보다 높을 수 있으나, 논문에서는 수치적 정확도와 계산 속도 면에서 전달 함수보다 우위에 있음을 주장하고 있다.

## 📌 TL;DR

이 논문은 전력 시스템 모델링 시 발생하는 포트 불일치 문제를 해결하기 위해 **Descriptor State Space** 기법을 도입하였다. 이를 통해 가상 소자 추가 없이 Proper/Improper 시스템을 모두 표현할 수 있으며, 특히 모델링 과정에서 물리적 상태 변수를 보존함으로써 **불안정성 모드의 근본 원인(Root cause)을 분석하는 참여도 분석(Participation Analysis)을 가능하게 하였다.** 이 연구는 향후 IBR 비중이 높아지는 현대 전력망의 소신호 안정도 분석 및 제어 파라미터 최적화에 중요한 도구로 활용될 가능성이 매우 높다.
