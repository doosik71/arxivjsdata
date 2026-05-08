# An Agent-based Classification Model

Feng Gu, Uwe Aickelin, Julie Greensmith (n.d.)

## 🧩 Problem to Solve

본 논문은 UCI Wisconsin Breast Cancer 데이터셋을 사용하여 데이터를 정상(normal)과 비정상(anomalous)의 두 가지 범주로 분류하는 문제를 다룬다. 이러한 작업은 컴퓨터 시스템에서 비정상적인 행동을 정상적인 행동과 구별하는 Anomaly Detection(이상 탐지)의 일종으로 볼 수 있다.

연구의 주된 목표는 생물학적 면역 체계에서 영감을 얻은 Dendritic Cell Algorithm (DCA)을 AnyLogic이라는 Agent-based Simulation(에이전트 기반 시뮬레이션) 환경에서 재구현하고, 그 타당성을 평가하는 것이다. 저자들은 지능형 에이전트가 면역 엔티티(immune entities)를 표현하는 데 최적의 방법이라고 판단하였으며, 이를 통해 향후 더 복잡하고 적응적인 Artificial Immune Systems (AIS) 모델을 구현할 수 있는 기반을 마련하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 DCA의 생물학적 메커니즘을 에이전트 기반의 시뮬레이션 모델로 전이시킨 것이다. 기존의 알고리즘 중심적 접근 방식에서 벗어나, 면역 체계의 구성 요소들을 독립적인 행동 주체인 에이전트로 설계함으로써 실제 생물학적 면역 반응과 유사한 상호작용 구조를 구축하였다. 이를 통해 모델의 유연성을 높이고 실시간 모니터링 및 동적인 실험 환경을 제공하는 구현 방식을 제시하였다.

## 📎 Related Works

본 연구는 Artificial Immune Systems (AIS)의 하위 분야인 Dendritic Cell Algorithm (DCA)을 기반으로 한다. AIS는 이론적 면역학 및 관찰된 면역 기능에서 영감을 얻어 문제 해결에 적용한 적응형 시스템이다. 특히 DCA는 이상 탐지를 위해 특화되어 개발된 알고리즘으로, 이미 컴퓨터 보안의 침입 탐지(intrusion detection) 분야에서 성공적으로 적용된 바 있다.

기존의 DCA 접근 방식은 수치적 계산과 알고리즘적 흐름에 집중했으나, 본 논문은 이를 에이전트 기반 모델링으로 전환하여 구현함으로써 면역 엔티티의 개별적 특성과 상호작용을 보다 직관적으로 묘사했다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. Dendritic Cell Algorithm (DCA) 메커니즘

DCA는 자연계의 수지상 세포(Dendritic Cells, DCs)가 항원을 처리하고 성숙하는 과정을 모방한다. DCs는 Immature(미성숙), Semimature(반성숙), Mature(완전 성숙)의 세 가지 상태를 가진다.

- **신호 처리**: Immature DC는 주변의 세 가지 신호를 감지한다.
  - $\text{PAMPs}$ (Pathogen Associated Molecular Patterns) 및 $\text{Danger signals}$: DC를 Mature 상태로 유도한다.
  - $\text{Safe signals}$: DC를 Semimature 상태로 유도한다.
- **성숙 결정**: DC는 누적된 $\text{Csm}$ (Costimulation signal) 값이 개별적으로 할당된 $\text{migration threshold}$를 초과할 때 성숙 단계로 진입한다. 이때 누적된 $\text{Semi}$ 신호와 $\text{Mat}$ 신호를 비교하여, $\text{Semi} > \text{Mat}$이면 Semimature 상태가 되고, 그렇지 않으면 Mature 상태가 된다.
- **최종 분류**: 성숙한 DC는 샘플링한 항원에 대해 컨텍스트(context)를 반환한다. Mature DC는 '1'(비정상)을, Semimature DC는 '0'(정상)을 반환한다. 항원별로 수집된 컨텍스트들의 비율인 $\text{MCAV}$ (Mature Context Antigen Value)를 계산하여 분류를 결정한다.

$$\text{MCAV} = \frac{\text{number of context '1'}}{\text{number of all contexts}}$$

$\text{MCAV}$가 설정된 $\text{anomalous threshold}$보다 크면 해당 항원을 비정상(anomalous)으로 분류한다.

### 2. 에이전트 기반 구현 (AnyLogic)

본 모델은 AnyLogic 환경에서 세 가지 유형의 에이전트를 정의하여 시스템을 구축하였다.

- **Antigen Agents (항원 에이전트)**: 데이터 항목을 운반하는 역할이다. 데이터셋의 각 항목을 하나의 에이전트로 생성하며, 선택된 속성을 추출하고 기록한다. 무작위로 DC 에이전트들을 선택해 'picked' 메시지를 보내며, 최종적으로 수집된 컨텍스트를 통해 $\text{MCAV}$를 계산하고 분류 결과를 도출한다.
- **DC Agents (수지상 세포 에이전트)**: 데이터 처리기 역할이다. Immature 상태로 시작하여 $\text{migration threshold}$를 생성한다. 항원 에이전트로부터 메시지를 받으면 속성을 복사해 신호 처리 함수를 실행하고, 누적 신호가 임계치를 넘으면 Mature 또는 Semimature 상태로 전이한 뒤 컨텍스트를 반환하고 소멸한다. 소멸 시 새로운 Immature DC가 생성되어 전체 개체 수를 일정하게 유지한다.
- **TC Agent (T-세포 에이전트)**: 통계 분석기 역할이다. 모든 항원 에이전트의 $\text{MCAV}$를 수집하여 분포도($\text{MCAV diagram}$)를 생성하고, 실제 정답(original category)과 비교하여 전체적인 $\text{True Positive}$ 또는 정확도(accuracy)를 측정한다.

## 📊 Results

본 연구에서는 UCI Wisconsin Breast Cancer 데이터셋을 사용하여 실험을 진행하였다. 구체적인 수치 결과는 텍스트에 명시되지 않았으나, 시뮬레이션 실험 결과 기존의 DCA 연구$[2]$에서 제시된 결과와 동일한 결과를 생성하였음을 확인하였다. 이를 통해 에이전트 기반 시뮬레이션 환경에서도 DCA의 핵심 알고리즘이 성공적으로 작동함이 입증되었다. 성능 측정 지표로는 $\text{True Positive}$ 및 정확도를 사용하였으며, $\text{TC agent}$를 통해 $\text{MCAV}$ 분포와 분류 결과를 시각화하였다.

## 🧠 Insights & Discussion

본 모델은 다음과 같은 강점을 가진다. 첫째, 에이전트 기반 접근 방식은 각 에이전트가 자신의 반응적/능동적 행동을 관리하므로, 자연 면역 체계의 구조와 매우 유사하여 구현이 직관적이다. 둘째, 시뮬레이션 소프트웨어를 통해 에이전트 간의 복잡한 상호작용을 보다 쉽게 구현할 수 있다. 셋째, 에이전트의 상태와 상호작용을 실시간으로 모니터링할 수 있으며, 시뮬레이션 속도를 조절할 수 있어 실험적 유연성이 높다.

다만, 본 연구에서는 신호원으로 사용할 속성(attributes)을 수동으로 선택해야 한다는 한계가 있다. 데이터셋마다 최적의 속성이 다르므로, 이에 대한 자동화된 선택 메커니즘이 부족하다. 저자들은 이를 해결하기 위해 향후 이전 분류 성능을 학습하여 신호원을 자동으로 선택하는 $\text{intelligent tissue agent}$를 도입할 계획임을 밝히고 있다.

## 📌 TL;DR

본 논문은 생물학적 면역 체계를 모방한 이상 탐지 알고리즘인 DCA를 AnyLogic 에이전트 기반 시뮬레이션으로 구현하여, UCI 유방암 데이터셋 분류에 적용한 연구이다. 항원, 수지상 세포, T-세포를 각각의 에이전트로 설계하여 생물학적 메커니즘을 충실히 재현하였으며, 기존 알고리즘과 동일한 성능을 확인하였다. 이 연구는 향후 더 복잡한 적응형 면역 모델을 시뮬레이션 환경에서 구현하고, 지능형 에이전트를 통해 속성 선택을 자동화하는 연구로 확장될 가능성이 크다.
