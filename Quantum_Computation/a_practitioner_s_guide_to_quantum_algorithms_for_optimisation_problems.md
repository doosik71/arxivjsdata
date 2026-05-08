# A Practitioner’s Guide to Quantum Algorithms for Optimisation Problems

Benjamin C. B. Symons, David Galvin, Emre Sahin, Vassil Alexandrov, and Stefano Mensa (2023)

## 🧩 Problem to Solve

본 논문은 산업 현장의 물류, 금융 등 다양한 분야에서 빈번하게 발생하는 $\text{NP-hard}$ 최적화 문제를 해결하기 위한 양자 컴퓨팅 기술의 현재 능력과 한계를 분석하고, 실무자들에게 종합적인 가이드를 제공하는 것을 목표로 한다.

전통적인 고전 컴퓨터는 변수가 많거나 제약 조건이 복잡한 조합 최적화(Combinatorial Optimisation) 문제에서 가능한 모든 해를 탐색하는 데 기하급수적인 시간이 소요되는 계산적 난제에 직면한다. 양자 컴퓨팅은 중첩(Superposition)과 얽힘(Entanglement)이라는 고유한 특성을 이용해 이러한 문제를 더 빠르게 해결할 가능성을 제시하지만, 실제 구현 단계에서는 하드웨어의 소음(Noise)과 게이트 구현의 정확도 문제로 인해 실질적인 '양자 이점(Quantum Advantage)'을 증명하는 데 어려움을 겪고 있다. 따라서 본 논문은 이론적 배경과 실제 적용 사례를 대조하여 양자 최적화 알고리즘의 실질적인 상태를 진단하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 양자 최적화의 이론적 프레임워크와 실제 하드웨어의 제약 사항을 연결하여 실무적인 관점에서의 분석 보고서를 제공했다는 점이다. 주요 기여 사항은 다음과 같다.

1. **고전-양자 최적화의 병렬 구조 분석**: 고전적인 이산 최적화 문제와 양자 알고리즘 간의 유사성과 차이점을 분석하고, 어떤 문제가 양자 가속의 혜택을 받을 수 있는지 정의한다.
2. **하드웨어 패러다임의 비교**: 아날로그 방식의 양자 어닐링(Quantum Annealing)과 디지털 방식의 게이트 기반 양자 컴퓨팅(Gate-based Quantum Computing)의 작동 원리와 범용성 차이를 명확히 설명한다.
3. **주요 양자 알고리즘의 상세 분석**: $\text{QAO}$ 알고리즘과 $\text{QAOA}$ 프레임워크의 수학적 구조, $\text{QUBO}$ 모델에서 이싱(Ising) 해밀토니언으로의 매핑 과정, 그리고 제약 조건 처리를 위한 믹서(Mixer) 설계 방식을 상세히 다룬다.
4. **$\text{NISQ}$ 시대의 현실적 진단**: 현재의 $\text{Noisy Intermediate-Scale Quantum (NISQ)}$ 장치에서 발생하는 소음 문제가 회로 깊이(Circuit Depth)와 성능에 미치는 영향을 분석하고, 단순한 하드웨어 확장이 아닌 알고리즘적 개선의 필요성을 제기한다.

## 📎 Related Works

논문에서는 고전 최적화의 복잡도 이론부터 최신 양자 알고리즘까지 폭넓은 관련 연구를 소개한다.

- **고전 최적화 연구**: 외판원 문제($\text{TSP}$), $\text{Max-Cut}$, $\text{Max-Flow}$, 배낭 문제($\text{Knapsack Problem}$)와 같은 전형적인 $\text{NP-hard}$ 문제들을 언급하며, 고전 알고리즘이 최적해를 찾기 위해 휴리스틱(Heuristic) 및 근사 방법(Approximation methods)에 의존하고 있음을 설명한다.
- **양자 최적화 접근법**:
  - **$\text{AQC}$ 및 $\text{QA}$**: 단열 양자 계산($\text{Adiabatic Quantum Computation}$)과 양자 어닐링의 이론적 토대를 다룬 연구들을 소개하며, 특히 $\text{D-Wave}$와 같은 상용 시스템의 가능성과 한계를 논한다.
  - **$\text{VQA}$ (Variational Quantum Algorithms)**: 하이브리드 양자-고전 알고리즘의 일종으로, 클래식 옵티마이저를 통해 양자 회로의 파라미터를 최적화하는 방식의 연구들을 검토한다.
- **기존 방식과의 차별점**: 본 논문은 개별 알고리즘의 성능 개선보다는, 실무자가 직면할 수 있는 '신화와 사실'을 구분하고 하드웨어 제약($\text{Noise}$, $\text{Connectivity}$) 하에서 알고리즘이 어떻게 동작하는지에 대한 통합적인 관점을 제공한다는 점에서 기존 개별 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

양자 최적화의 기본 흐름은 **[문제 정의 $\rightarrow$ 수학적 모델링 ($\text{QUBO/Ising}$) $\rightarrow$ 양자 회로 설계 $\rightarrow$ 파라미터 최적화 $\rightarrow$ 측정 및 해 추출]** 순으로 진행된다.

### 2. 핵심 알고리즘 및 수학적 설명

#### A. $\text{QAO}$ ($\text{Quantum Approximate Optimisation}$) 알고리즘

$\text{QAO}$는 하이브리드 변분 알고리즘으로, 비용 함수 $C$를 해밀토니언 $\hat{H}_C$로 매핑하여 최솟값을 찾는다.

- **비용 유니터리 (Cost Unitary)**: $\hat{U}_C(\gamma) = \exp(-i\gamma \hat{H}_C)$
- **믹서 유니터리 (Mixer Unitary)**: $\hat{U}_M(\mu) = \exp(-i\mu \hat{H}_M)$
- **전체 회로 구성**: 깊이 $p$에 대해 초기 상태 $|\phi_0\rangle$에 $p$번의 블록을 적용한다.
    $$|\phi(\mu, \gamma)\rangle = \prod_{i=1}^p \hat{U}_M(\mu_i) \hat{U}_C(\gamma_i) |\phi_0\rangle$$
- **목표**: 고전 옵티마이저를 사용하여 다음 기대값을 최소화하는 $\mu_i, \gamma_i$를 찾는다.
    $$\langle \phi(\mu, \gamma) | \hat{H}_C | \phi(\mu, \gamma) \rangle$$

#### B. $\text{QUBO}$에서 이싱(Ising) 모델로의 매핑

제약 없는 이차 이진 최적화($\text{QUBO}$) 문제는 다음과 같은 형태로 정의된다.
$$f_Q(x) = \sum_{i=0}^n \sum_{j=i}^n Q_{ij} x_i x_j \quad (x_i \in \{0, 1\})$$
이를 양자 컴퓨터에서 처리하기 위해 스핀 변수 $s_i \in \{-1, 1\}$를 사용하는 이싱 해밀토니언으로 변환하며, 변환식은 $s = 2x - 1$을 따른다. 최종적으로 이는 파울리 $Z$ 행렬($\hat{Z}$)의 조합으로 표현되어 양자 하드웨어에 구현된다.

#### C. $\text{QAOA}$ ($\text{Quantum Alternating Operator Ansatz}$) 및 제약 조건 처리

$\text{QAO}$가 제약 없는 문제에 집중한다면, $\text{QAOA}$는 제약 조건이 있는 문제를 해결하기 위해 설계된 일반화된 프레임워크이다.

- **Soft Constraint**: 비용 함수에 페널티 항(Penalty term)을 추가하여 제약 위반 시 비용을 크게 높이는 방식이다.
- **Hard Constraint**: 믹서 유니터리 $\hat{U}_M$ 자체를 설계하여, 오직 **실행 가능한 상태(Feasible states)** 사이에서만 전이가 일어나도록 제한한다.
  - 예: $\text{XY-mixer}$는 해밍 무게(Hamming weight)를 보존하여 특정 합계 제약을 유지한다.

### 3. 양자 세미데피니트 프로그래밍 ($\text{Quantum SDP}$)

볼록 최적화(Convex Optimisation)의 일종인 $\text{SDP}$를 양자 알고리즘으로 해결하려는 시도이다. $\text{SDP}$ 문제는 양의 세미데피니트 행렬 $X \succeq 0$에 대해 $\min \langle C, X \rangle$를 찾는 문제이며, 최근에는 $\text{NISQ}$ 장치를 위해 변분 양자 알고리즘(VQA) 형태의 $\text{SDP}$ 솔버가 제안되고 있다.

## 📊 Results

논문은 직접적인 실험보다는 기존의 다양한 케이스 스터디를 종합하여 분석한 결과를 제시한다.

- **측정 지표**: 근사 비율(Approximation Ratio) $r$을 사용하여 성능을 평가한다.
    $$r = \frac{C_{max} - C^*}{C_{max} - C_{min}}$$
    여기서 $C^*$는 알고리즘이 찾은 값이며, $r$이 1에 가까울수록 최적해에 가깝다.
- **주요 정량적/정성적 결과**:
  - **$\text{Max-Cut}$ 문제**: 시뮬레이션상으로는 회로 깊이 $p$가 증가함에 따라 성능이 향상되지만, 실제 하드웨어에서는 $p$가 증가할수록 소음으로 인해 성능이 급격히 저하된다.
  - **하드웨어 제약**: 구글의 $\text{Sycamore}$ QPU 테스트 결과, 평면 그래프(Planar graph) 문제는 비교적 안정적이었으나, 비평면 그래프 문제는 $\text{SWAP}$ 게이트 추가로 인한 회로 깊이 증가로 인해 성능이 빠르게 퇴화했다.
  - **제약 조건 처리**: 페널티 항을 이용한 $\text{Soft Constraint}$ 방식은 실행 가능한 해를 찾을 확률이 매우 낮았으며, $\text{Hard Constraint}$ 방식은 이론적으로 우수하지만 회로 깊이가 훨씬 깊어져 소음에 극도로 취약함을 확인했다.

## 🧠 Insights & Discussion

### 1. 강점 및 가능성

- **물리적 영감의 도입**: $\text{QED-mixer}$와 같이 물리학의 게이지 불변성(Gauge invariance)을 이용해 네트워크 흐름 제약을 자연스럽게 해결하려는 시도는 매우 유망한 방향이다.
- **하이브리드 접근**: $\text{Warm-start}$ 기법(고전 알고리즘으로 초기 해를 찾고 양자로 정교화하는 방식)은 $\text{NISQ}$ 장치의 한계를 극복할 실질적인 대안이 될 수 있다.

### 2. 한계 및 비판적 해석

- **$\text{VQA}$의 역설**: 변분 양자 알고리즘($\text{VQA}$)을 통해 $\text{NP-hard}$ 문제를 풀려 하지만, 정작 그 알고리즘의 파라미터 $\theta$를 최적화하는 고전적 과정 자체가 또 다른 $\text{NP-hard}$ 문제(비볼록 최적화)가 되는 모순이 존재한다.
- **소음의 지배적 영향**: 현재의 양자 우위 주장은 대부분 소음이 없는 시뮬레이션에 기반하고 있다. 실제 $\text{NISQ}$ 하드웨어에서는 회로 깊이의 아주 작은 증가만으로도 양자 상태가 최대 혼합 상태(Maximally mixed state)로 빠르게 붕괴되어 이점이 사라진다.
- **하드웨어 맹신 경계**: 단순히 큐비트 수를 늘리는 것보다 게이트 충실도(Fidelity)의 획기적인 개선이나 오류 수정(Error Correction) 기술 없이는 실질적인 양자 이점을 달성하기 어려울 것으로 보인다.

## 📌 TL;DR

본 논문은 양자 최적화 알고리즘($\text{QAO, QAOA, SDP}$)의 이론적 구조와 실제 적용 가능성을 분석한 실무 지침서이다. 결론적으로 **이론적 잠재력은 매우 크나, 현재의 $\text{NISQ}$ 하드웨어에서는 소음과 회로 깊이의 제약으로 인해 실질적인 양자 이점이 아직 증명되지 않았다**고 진단한다. 특히 $\text{Hard Constraint}$를 처리하는 정교한 믹서 설계와 소음 내성 알고리즘의 개발이 향후 연구의 핵심이 될 것이며, 이는 실제 산업 적용을 위한 필수 과제이다.
