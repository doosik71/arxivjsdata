# An Invertible State Space for Process Trees

Gero Kolhof, Sebastiaan J. van Zelst (2024)

## 🧩 Problem to Solve

본 논문은 프로세스 마이닝(Process Mining) 분야에서 널리 사용되는 모델링 형식인 **Process Tree**의 계산 효율성을 높이기 위한 정형적인 상태 공간(State Space) 정의 문제를 해결하고자 한다.

Process Tree는 Petri Net의 엄격한 부분집합으로, 구조적 특성 덕분에 알고리즘적 관점에서 매우 유용하며, 특히 Alignments와 같은 정합성 확인(Conformance Checking) 아티팩트의 계산 가능성을 보장한다. 그러나 Process Tree의 정형적 속성과 이를 이용해 일반적인 계산 문제(예: 상태 공간 내 최단 경로 탐색)의 효율성을 개선하려는 연구는 부족한 실정이다.

따라서 본 연구의 목표는 Process Tree를 위한 **가역적인 상태 공간(Invertible State Space)** 정의를 제안하고, 이를 통해 상태 공간 탐색의 시간을 단축할 수 있는 양방향 탐색(Bidirectional Search) 전략을 도입하여 전체적인 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Process Tree의 상태 공간을 정형화하고, 이 상태 공간이 **가역성(Invertibility)**을 가진다는 것을 이론적으로 증명한 것이다.

중심 아이디어는 특정 Process Tree $T$의 상태 공간 그래프가 해당 트리의 역트리(Inverse Tree) $T^{-1}$의 상태 공간 그래프와 **동형(Isomorphic)**이라는 점이다. 이러한 이론적 토대는 상태 공간 탐색 문제를 해결할 때, 시작 상태에서 목표 상태로 가는 단방향 탐색 대신, 양 끝단에서 동시에 탐색하여 중간에서 만나는 **양방향 탐색(Bidirectional Search)** 기법을 적용할 수 있게 함으로써 탐색 범위를 획기적으로 줄이고 성능을 최적화할 수 있게 한다.

## 📎 Related Works

Process Tree는 블록 구조 프로세스 모델링(Block-structured process modeling) 형식에서 영감을 받았으며, 기존 연구들은 주로 다음과 같은 방향으로 진행되었다.

1. **언어 보존 축소 규칙:** Process Tree의 언어를 유지하면서 트리를 단순화하는 축소 규칙들이 제안되었다.
2. **워크플로우 넷(Workflow Net)과의 관계:** 모든 Process Tree가 건전한(sound) Free-choice Workflow Net으로 변환될 수 있음이 알려져 있으며, 반대로 임의의 Workflow Net이 언어적으로 동등한 Process Tree를 갖는지 탐색하는 알고리즘 연구가 존재한다.
3. **프로세스 발견(Process Discovery):** 유전 알고리즘이나 재귀적 알고리즘을 통해 이벤트 데이터로부터 Process Tree를 자동으로 발견하는 연구들이 수행되었다.

기존 접근 방식들이 주로 모델의 구조적 변환이나 발견에 집중했다면, 본 논문은 **상태 공간의 정형적 정의와 그 가역성**을 통해 탐색 알고리즘의 효율성을 극대화한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 상태 및 전이 정의 (States and Transitions)

본 논문에서는 Process Tree의 각 정점(Vertex) $v$에 대해 세 가지 로컬 상태 $S = \{F, O, C\}$를 정의한다.

- **Future (F):** 현재 열려 있지 않으며, 향후 열리거나(Open) 닫힐(Closed) 수 있는 상태이다.
- **Open (O):** 정점이 현재 활성화되어 실행 중인 상태이다.
- **Closed (C):** 정점이 실행을 완료하여 닫혔거나, 혹은 실행되지 않고 건너뛰어 닫힌 상태이다.

전체 Process Tree의 상태 $\vec{s}$는 모든 정점에 대한 상태의 튜플 $\vec{s} = (s_1, s_2, \dots, s_n)$로 표현된다. 상태 전이는 $v[X \to Y]$ 형태로 정의하며, 다음과 같은 기본 제약 조건을 따른다.

- 전이는 $F \to O$, $O \to C$, $C \to F$ 순으로 가능하며, $C \to O$ 전이는 불가능하다.
- 정점이 $O$ 상태가 되기 위해서는 부모 정점이 $O$ 상태여야 하며, 모든 자식 정점들이 $F$ 상태여야 한다.
- 정점이 $C$ 상태가 되기 위해서는 부모 정점이 $O$ 상태여야 하며, 모든 자식 정점들이 $C$ 상태여야 한다.

또한, 각 연산자(Sequence $\to$, Reverse Sequence $\leftarrow$, Exclusive Choice $\times$, Parallelism $+$, Loop $\circlearrowleft$)에 따라 형제 정점(Siblings)들의 상태에 기반한 세부 전이 규칙이 적용된다.

### 2. 가역적 상태 공간 (Invertible State Space)

본 논문은 상태의 역함수 $s^{-1}$를 다음과 같이 정의한다.
$$
s^{-1}(v) =
\begin{cases}
F & \text{if } s(v) = C \\
O & \text{if } s(v) = O \\
C & \text{if } s(v) = F
\end{cases}
$$

이 정의를 바탕으로, Process Tree $T$에서 상태 $s_1$에서 $s_2$로의 전이가 가능하면, 역트리 $T^{-1}$에서는 $s_2^{-1}$에서 $s_1^{-1}$로의 역전이가 가능하다는 **전이 가역성(Transition Inversibility)**을 Lemma 1을 통해 증명한다. 결과적으로 $T$의 상태 공간 그래프 $RG(T, \vec{F})$와 $T^{-1}$의 상태 공간 그래프 $RG(T^{-1}, \vec{F})$는 동형(Isomorphic)임이 보장된다.

### 3. 상태 공간 축소 및 매칭 (Reduction & Matching)

탐색 효율을 높이기 위해 두 가지 최적화 기법을 도입한다.

- **Fast-forwarding:** 정점이 $F \to C$ 또는 $C \to F$로 전이될 때, 해당 정점의 모든 자손(Descendants)들에게도 재귀적으로 동일한 상태 변경을 적용하여 불필요한 중간 상태를 제거한다.
- **State Matching:** 양방향 탐색 중 $\vec{s}$와 $\vec{s}'^{-1}$가 일치하는 지점을 찾아 두 탐색 경로를 결합함으로써 최단 경로를 산출한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Pm4Py 라이브러리를 사용하여 5~15개의 액티비티를 가진 150,000개의 서로 다른 Process Tree를 생성하였다. 연산자 분포는 Dirichlet 분포를 통해 다양하게 샘플링하였다.
- **비교 대상:**
  - **UD (Unidirectional):** 시작 상태에서 종료 상태까지 단방향 BFS 탐색을 수행하는 베이스라인이다.
  - **BD (Bidirectional):** $T$와 $T^{-1}$에서 동시에 탐색을 수행하는 방식이다.
  - **BDP (Bidirectional Parallelized):** BD를 멀티스레드로 구현하여 각 탐색 방향을 병렬로 처리한 방식이다.
- **측정 지표:** 메모리 소비량(탐색된 고유 상태 수)과 실행 시간(Execution Time)을 측정하였다.

### 주요 결과

- **메모리 및 시간 효율성:** BD와 BDP 모두 UD에 비해 메모리 소비와 실행 시간을 유의미하게 줄였다. 특히 약 50%의 트리에서 감소 폭이 $[1.3, 4]$ 범위에 달했다.
- **연산자별 영향:** 병렬 연산자($+$)의 비중이 높을수록 상태 공간의 분기 계수(Branching Factor)가 커지므로 양방향 탐색의 효율(Reduction Factor)이 증가하는 경향을 보였다. 반면, 순차 연산자($\to$) 비중이 높을 때는 분기 계수가 낮아 효율 증가 폭이 적었다.
- **병렬화 효과:** BDP는 BD보다 실행 시간을 더 단축시켰으며, 탐색해야 할 상태 공간의 크기가 클수록 성능 향상 폭이 커져 이론적 최대치인 2배 속도 향상에 근접하는 모습을 보였다.

## 🧠 Insights & Discussion

본 논문은 Process Tree의 상태 공간에 대한 수학적 가역성을 정의함으로써, 단순한 알고리즘 구현을 넘어 이론적 보장을 갖춘 최적화 가능성을 제시하였다. 특히 양방향 탐색의 도입이 상태 공간의 지수적 증가 문제를 완화할 수 있음을 실험적으로 입증하였다.

**강점:**

- 상태 공간의 동형성(Isomorphism)을 증명함으로써 양방향 탐색의 정당성을 확보하였다.
- Fast-forwarding과 같은 실용적인 축소 기법을 통해 실제 계산 시간을 단축했다.

**한계 및 비판적 해석:**

- 실험에 사용된 트리의 크기가 액티비티 5~15개로 매우 작다. 상태 공간이 지수적으로 증가하는 특성상, 더 큰 규모의 트리에서는 BFS 기반의 양방향 탐색만으로는 한계가 있을 것이며, 휴리스틱 기반의 탐색(예: A*)과의 결합이 필요할 것으로 보인다.
- 효율성 개선 정도가 트리의 연산자 구성(특히 병렬성)에 크게 의존하므로, 모든 Process Tree에 대해 일관된 성능 향상을 기대하기는 어렵다.

## 📌 TL;DR

본 연구는 Process Tree를 위한 가역적인 상태 공간(Invertible State Space)을 정의하고, 원본 트리와 역트리의 상태 공간 그래프가 동형임을 증명하였다. 이를 통해 양방향 탐색(Bidirectional Search)을 적용함으로써 단방향 탐색 대비 메모리 사용량과 실행 시간을 크게 줄였으며, 특히 병렬 구조가 많은 트리에서 그 효과가 극대화됨을 확인하였다. 이 결과는 향후 Process Tree 기반의 정합성 확인(Conformance Checking) 및 Alignments 계산 효율성을 높이는 데 중요한 기초 이론으로 활용될 수 있다.
