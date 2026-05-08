# Adiabatic Quantum Computation is Equivalent to Standard Quantum Computation

Dorit Aharonov, Wim van Dam, Julia Kempe, Zeph Landau, Seth Lloyd, Oded Regev (2008)

## 🧩 Problem to Solve

본 논문은 단열 양자 계산(Adiabatic Quantum Computation, 이하 AQC) 모델의 계산 능력이 표준 양자 계산(Standard Quantum Computation, 회로 기반 모델)과 비교하여 어느 정도인지 규명하고자 한다.

단열 양자 계산은 시스템의 Hamiltonian을 서서히 변화시켜 초기 상태의 ground state에서 최종 Hamiltonian의 ground state로 전이시키는 방식으로 작동한다. 기존에 AQC가 표준 양자 컴퓨터에 의해 효율적으로 시뮬레이션될 수 있다는 점은 알려져 있었으나, 반대로 표준 양자 회로의 임의의 계산을 AQC로 효율적으로 구현할 수 있는지, 즉 두 모델이 다항 시간 내에 서로를 시뮬레이션할 수 있는 '다항 동등성(polynomial equivalence)'을 갖는지에 대해서는 밝혀지지 않은 상태였다. 이 문제의 해결은 새로운 양자 알고리즘 설계의 관점을 제공하고, 결함 허용(fault-tolerant) 양자 컴퓨터 구축을 위한 새로운 방향성을 제시한다는 점에서 매우 중요하다.

## ✨ Key Contributions

본 논문의 핵심 기여는 표준 양자 계산 모델과 단열 양자 계산 모델이 다항 시간 내에 동등함을 증명한 것이다. 구체적인 설계 아이디어와 기여 사항은 다음과 같다.

1. **표준 양자 회로의 단열 시뮬레이션**: 임의의 양자 회로를 AQC로 효율적으로 시뮬레이션할 수 있음을 보였다. 이를 위해 3-local Hamiltonian을 사용한 구성을 제안하였다.
2. **History State의 활용**: 최종 상태를 직접 지정하는 대신, 양자 회로의 모든 계산 단계의 중첩 상태인 'history state'를 ground state로 갖는 Hamiltonian을 설계함으로써, 최종 출력 상태를 알지 못해도 AQC를 구성할 수 있는 방법을 제시하였다.
3. **물리적 실현 가능성 증명**: 3-local을 넘어, 2차원 그리드 상에서 인접한 입자 간의 2-local 상호작용만을 사용하는 AQC 모델로도 범용 양자 계산이 가능함을 증명하였다. 이때 입자는 6-state particle로 구성된다.
4. **Spectral Gap 분석**: Markov chain의 전도도 경계(conductance bound) 이론을 적용하여, 단열 진화 과정에서의 minimal spectral gap이 다항 시간 내에 유지됨을 수학적으로 증명하였다.

## 📎 Related Works

- **Farhi et al. [14]**: AQC를 이용해 SAT와 같은 고전적 최적화 문제를 해결하려는 시도를 시작하였다. 그러나 최근 연구들[9, 10, 29]은 NP-complete 문제에 대해 AQC 알고리즘이 최악의 경우 지수 시간을 소요할 수 있음을 시사한다.
- **Grover의 검색 알고리즘 [16]**: 비정렬 검색에 대한 Grover의 이차 가속(quadratic speed-up)이 AQC로 구현될 수 있음이 이미 알려져 있었다[9, 30].
- **Kitaev의 Local Hamiltonian 문제 [22]**: Kitaev는 Local Hamiltonian 문제가 Quantum NP-complete임을 증명하며, 양자 회로의 시간 전파(time propagation)를 검사하는 Hamiltonian을 정의하였다. 본 논문은 이 아이디어를 확장하여 AQC의 $H_{final}$을 설계하는 데 사용하였다.
- **기존 연구와의 차별점**: 이전의 연구들이 주로 최종 Hamiltonian($H_{final}$)이 대각 행렬인 '단열 최적화(adiabatic optimization)'에 집중했다면, 본 논문은 $H_{final}$이 일반적인 local Hamiltonian인 더 넓은 범위의 AQC 모델을 다룸으로써 범용 양자 계산과의 동등성을 증명하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인

양자 회로의 계산 과정을 AQC로 변환하는 과정은 다음과 같다.
$\text{Quantum Circuit} \rightarrow \text{History State 구성} \rightarrow \text{Initial/Final Hamiltonian 설계} \rightarrow \text{Adiabatic Evolution} \rightarrow \text{Measurement}$

### 2. 핵심 구성 요소 및 방법론

#### History State ($|\eta\rangle$)

표준 양자 회로의 최종 상태 $|\alpha(L)\rangle$을 직접 ground state로 만드는 것은 불가능에 가깝다(최종 상태를 미리 알 수 없기 때문). 따라서 본 논문은 Kitaev의 아이디어를 빌려, 계산의 모든 단계 $\ell$에서의 상태 $|\alpha(\ell)\rangle$과 이를 나타내는 clock 상태 $|1_\ell 0_{L-\ell}\rangle_c$의 중첩인 history state를 정의한다.
$$|\eta\rangle := \frac{1}{\sqrt{L+1}} \sum_{\ell=0}^{L} |\alpha(\ell)\rangle \otimes |1_\ell 0_{L-\ell}\rangle_c$$
이 상태는 최종 상태 $|\alpha(L)\rangle$에 대해 무시할 수 없는 투영(projection) 값을 가지므로, 이를 ground state로 하는 $H_{final}$을 설계한다.

#### Hamiltonian 설계

- **$H_{init}$**: 초기 상태 $|\gamma_0\rangle = |0^n\rangle \otimes |0^L\rangle_c$를 유일한 ground state로 갖도록 설계하며, 이는 단순한 tensor product 상태이므로 local Hamiltonian으로 쉽게 구현 가능하다.
- **$H_{final}$**: history state $|\eta\rangle$를 ground state로 갖는 Kitaev의 Local Hamiltonian을 사용한다. 이는 계산의 각 단계 $\ell-1$에서 $\ell$로의 전이가 유니타리 게이트 $U_\ell$에 의해 올바르게 수행되었는지를 검사하는 항들의 합으로 구성된다.
- **$\text{AQC Process}$**: 시스템은 다음과 같은 경로를 통해 진화한다.
$$H(s) = (1-s)H_{init} + sH_{final}, \quad s \in [0, 1]$$

#### Spectral Gap 및 시간 복잡도 분석

AQC의 실행 시간 $T$는 minimal spectral gap $\Delta(H(s))$에 의해 결정된다. 본 논문은 Hamiltonian을 Markov chain으로 매핑하고, 전도도 경계(conductance bound)를 사용하여 $\Delta(H(s)) = \Omega(L^{-2})$임을 증명하였다. 결과적으로 실행 시간은 $L$에 대한 다항 시간 $O(poly(L))$이 된다.

#### Locality의 확장 (5-local $\rightarrow$ 3-local $\rightarrow$ 2-local)

- **3-local**: 5-local Hamiltonian에서 clock 관련 제약 조건을 강하게 부여하는 penalty 항($J \cdot H_{clock}$)을 추가하여, 유효한 subspace 내에서 3-local Hamiltonian으로 시뮬레이션이 가능함을 보였다.
- **2-local (2D Grid)**: 입자를 6-state particle로 정의하고, clock을 별도의 레지스터가 아닌 입자들의 '형태(shape)'로 인코딩하였다. snake-like 패턴으로 그리드를 훑으며 계산이 진행되도록 설계하여, 오직 인접한 입자 간의 2-local 상호작용만으로 범용 양자 계산을 구현하였다.

## 📊 Results

### 1. 정량적 결과 및 정리

- **Theorem 1.1**: 단열 양자 계산 모델은 표준 양자 계산 모델과 다항 시간 내에 동등하다.
- **Corollary 1.2**: Explicit sparse Hamiltonian을 사용하는 AQC 모델 역시 표준 양자 계산과 다항 동등하다.
- **Theorem 1.3**: 모든 양자 계산은 2차원 그리드 상의 6-state 입자들 사이의 2-local nearest neighbor Hamiltonian을 이용한 AQC로 효율적으로 시뮬레이션될 수 있다.

### 2. 복잡도 및 성능

- 5-local Hamiltonian을 사용한 경우, 실행 시간은 대략 $O(L^5)$ 수준이다.
- 3-local Hamiltonian으로 최적화했을 때, 실행 시간은 대략 $O(L^{14})$까지 증가할 수 있으나 여전히 다항 시간 내에 수행된다.
- 2-local 2D grid 모델에서도 다항 시간 복잡도를 유지하며 물리적 구현 가능성을 높였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 서로 완전히 다른 두 양자 계산 모델(회로 기반 vs Hamiltonian 진화 기반) 사이의 동등성을 증명함으로써, 양자 알고리즘 설계의 새로운 도구를 제공하였다. 특히, 알고리즘의 효율성 문제를 행렬의 spectral gap 분석 문제로 치환함으로써, 수학적 물리학, expander theory, Markov chain 등의 강력한 수학적 도구들을 양자 알고리즘 설계에 도입할 수 있게 하였다.

### 한계 및 미해결 질문

- **실행 시간 최적화**: 제안된 시뮬레이션의 다항 시간 차수가 다소 높으며($L^{14}$ 등), 이를 단축할 방법이 남아 있다.
- **물리적 제약**: 2D 그리드 모델에서 사용된 '6-state particle'의 차원을 더 낮출 수 있는지, 혹은 1차원 그리드에서도 가능한지는 여전히 열린 문제이다.
- **단열 최적화의 능력**: $H_{final}$이 대각 행렬로 제한되는 '단열 최적화(adiabatic optimization)' 모델이 실제로 고전 튜링 머신으로 효율적으로 시뮬레이션 가능한지는 명확히 밝혀지지 않았다.

### 비판적 해석

본 논문은 이론적인 '동등성'을 증명하는 데 집중하고 있으나, 실제 물리적 시스템에서 $L^{14}$와 같은 높은 차수의 다항 시간 복잡도는 실질적인 구현에 있어 매우 큰 장애물이 될 수 있다. 또한, noise가 존재하는 환경에서의 fault-tolerant AQC에 대한 구체적인 메커니즘보다는 가능성 제시에 그친 점이 아쉽다.

## 📌 TL;DR

본 논문은 **단열 양자 계산(AQC)이 표준 양자 회로 모델과 다항 시간 내에 동등(polynomially equivalent)**함을 증명하였다. 특히 Kitaev의 history state 개념을 도입하여 임의의 양자 회로를 AQC로 변환하는 방법을 제시하였으며, 이를 **3-local Hamiltonian** 및 **2차원 그리드의 2-local nearest neighbor Hamiltonian**로 구현할 수 있음을 보였다. 이 연구는 양자 알고리즘 설계를 spectral gap 분석이라는 새로운 관점에서 접근할 수 있게 하였으며, 향후 하드웨어 수준의 범용 양자 컴퓨터 구현을 위한 중요한 이론적 토대를 마련하였다.
