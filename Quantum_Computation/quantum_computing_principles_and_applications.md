# QUANTUM COMPUTING: PRINCIPLES AND APPLICATIONS

Guanru Feng, Dawei Lu, Jun Li, Tao Xin, and Bei Zeng (2023)

## 🧩 Problem to Solve

본 논문은 현대 양자 컴퓨팅의 급격한 발전 속에서, 양자 컴퓨팅의 기초 원리부터 실제 물리적 구현까지의 전 과정을 포괄적으로 설명하는 것을 목표로 한다. 특히, 현재의 양자 컴퓨팅은 노이즈가 존재하는 중간 단계 양자(Noisy Intermediate-Scale Quantum, NISQ) 시대에 머물러 있으며, 하드웨어의 불완전성으로 인해 이론과 실제 구현 사이에 간극이 존재한다.

본 연구의 핵심 목적은 양자 컴퓨팅의 이론적 기초를 정립하고, 다양한 물리적 플랫폼 중 특히 실험적 성숙도가 높은 핵자기공명(Nuclear Magnetic Resonance, NMR) 플랫폼을 중심으로 양자 알고리즘을 실제로 어떻게 구현하는지에 대한 상세한 가이드를 제공하는 것이다. 이를 통해 교육적 목적의 양자 컴퓨팅 입문과 실험적 검증을 위한 학술적 토대를 마련하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 양자 컴퓨팅의 이론-아키텍처-물리적 구현으로 이어지는 통합적인 프레임워크를 제시한 점이다. 주요 기여 사항은 다음과 같다.

- **다층 아키텍처 제시**: 양자 알고리즘, 소프트웨어, 컴파일러, 명령어 집합(ISA), 물리적 플랫폼으로 이어지는 양자 컴퓨터의 계층적 구조를 정의하여 하드웨어와 소프트웨어 간의 연결 고리를 설명하였다.
- **NMR 기반 구현 방법론 구체화**: NMR 플랫폼에서 큐비트를 정의하고, Hamiltonian 제어를 통해 양자 게이트를 구현하는 상세 과정을 서술하였다. 특히 Pseudo-pure state(PPS) 개념을 도입하여 상온 NMR의 혼합 상태(mixed state) 문제를 해결하는 방법을 제시하였다.
- **최적 제어 알고리즘(GRAPE) 소개**: 목표로 하는 유니타리 행렬 $U$에 도달하기 위해 펄스 형태를 최적화하는 GRAPE(Gradient Ascent Pulse Engineering) 알고리즘의 수학적 원리와 적용 방법을 상세히 다루었다.
- **실무적 알고리즘 구현 사례**: Deutsch, Grover, Bernstein-Vazirani 알고리즘 및 양자 조화 진동자(Quantum Harmonic Oscillator) 시뮬레이션과 같은 구체적인 사례를 통해 이론이 실제 하드웨어에서 어떻게 작동하는지 증명하였다.

## 📎 Related Works

논문은 양자 컴퓨팅의 다양한 물리적 구현 플랫폼에 대한 기존 연구들을 검토한다.

- **초전도 큐비트(Superconducting Qubits)**: Google과 IBM 등이 주도하고 있으며, 최근 '양자 우위(Quantum Supremacy)'를 달성하는 등 가장 지배적인 플랫폼으로 평가받지만, 극저온 환경이 필수적이라는 제약이 있다.
- **이온 트랩(Ion Traps)**: 긴 결맞음 시간(coherence time)과 높은 제어 정밀도를 가지지만, 다수 큐비트 확장 시 레이저 냉각 효율 저하 및 모드 크로스토크(mode crosstalk) 문제가 발생한다.
- **다이아몬드 NV 센터(Diamond NV Center)**: 상온 작동이 가능하다는 강력한 장점이 있으며, 전자 스핀과 핵 스핀을 동시에 활용할 수 있어 정밀 측정 및 통신에 유리하다.
- **실리콘 양자점(Silicon Quantum Dots)**: 기존 반도체 공정을 활용할 수 있어 확장성이 매우 높으나, 환경과의 상호작용으로 인한 짧은 결맞음 시간이 한계로 지적된다.

NMR 플랫폼의 경우, 확장성(scalability) 측면에서는 한계가 있어 범용 양자 컴퓨터로서는 부적합할 수 있으나, 제어 기술이 매우 성숙해 있어 초기 양자 알고리즘 검증 및 교육용 플랫폼으로서 독보적인 가치를 지닌다고 차별점을 제시한다.

## 🛠️ Methodology

### 1. 양자 상태와 수학적 표현

양자 상태는 힐베르트 공간(Hilbert space)의 벡터 $|\psi\rangle$로 표현되며, $N$차원 공간에서 다음과 같은 선형 결합(중첩)으로 나타낼 수 있다.
$$|\psi\rangle = c_0|0\rangle + c_1|1\rangle + \cdots + c_{N-1}|N-1\rangle$$
여기서 각 상태의 확률은 $|c_i|^2$이며, 전체 확률의 합은 $\sum |c_i|^2 = 1$을 만족해야 한다. 시스템의 통계적 특성을 나타내기 위해 밀도 행렬(Density Matrix) $\rho = \sum p_i |\phi_i\rangle \langle \phi_i|$를 사용하며, 이는 순수 상태(pure state)와 혼합 상태(mixed state)를 모두 표현할 수 있게 한다.

### 2. NMR 시스템의 Hamiltonian 및 제어

NMR 시스템의 동역학은 시스템 Hamiltonian $H$에 의해 결정된다. 전체 Hamiltonian은 다음과 같이 구성된다.
$$H = H_{static} + H_{rf} + H_{chemical\_shift} + H_{J-coupling}$$

- **Static Field ($H_{static}$)**: 제만 분리(Zeeman splitting)를 일으키며, $\omega_0 = \gamma B_0$의 라모어 주파수(Larmor frequency)를 결정한다.
- **RF Field ($H_{rf}$)**: 라모어 주파수에 공명하는 라디오 주파수 펄스를 인가하여 큐비트의 상태를 회전시킨다.
- **J-Coupling**: 인접한 핵 스핀 간의 상호작용을 통해 2-큐비트 게이트(예: CNOT)를 구현하는 핵심 기제로 작용한다.

### 3. Pseudo-pure State (PPS) 준비

상온 NMR은 열평형 상태에서 매우 낮은 분극도(polarization)를 가지는 혼합 상태이다. 이를 극복하기 위해 다음과 같은 Pseudo-pure state를 정의하여 순수 상태와 동일한 동역학적 거동을 보이도록 한다.
$$\rho_{pps} = (1-\eta)\frac{I \otimes n}{2^n} + \eta|00\cdots0\rangle \langle 00\cdots0|$$
여기서 $\eta$는 유효 순도(effective purity)이며, 정체성 행렬(Identity matrix) 부분은 측정 신호에 기여하지 않으므로, 관측되는 신호는 순수 상태 $|00\cdots0\rangle$의 진화와 동일하게 나타난다.

### 4. GRAPE 알고리즘을 이용한 펄스 최적화

목표 유니타리 연산 $\bar{U}$와 실제 구현된 연산 $U(t_p)$ 사이의 Fidelity $F$를 최대화하기 위해 GRAPE 알고리즘을 사용한다.
$$F(U(t_p), \bar{U}) = \frac{|\text{Tr}(U(t_p) \cdot \bar{U}^\dagger)|^2}{2^{2n}}$$
GRAPE는 전체 시간을 $N$개의 세그먼트로 나누고, 각 세그먼트의 펄스 진폭 $u(j)$에 대한 기울기(gradient) $\frac{\partial F}{\partial u(j)}$를 계산하여 경사 상승법(gradient ascent)으로 최적의 펄스 형태를 찾아낸다.

## 📊 Results

본 논문은 SpinQ사의 데스크톱 NMR 플랫폼(Gemini, Triangulum)을 통해 다음과 같은 실험적 결과를 제시한다.

- **단일 큐비트 및 2-큐비트 상태 관측**: Bloch Sphere를 통해 $|0\rangle, |1\rangle$ 및 중첩 상태의 구현을 확인하였으며, Quantum State Tomography를 통해 실험적 밀도 행렬과 이론적 밀도 행렬 사이의 일치성을 검증하였다.
- **양자 게이트 구현**: Hadamard(H), Pauli-X, CNOT 게이트를 구현하였다. 특히 CNOT 게이트의 경우, 제어 큐비트의 상태에 따라 타겟 큐비트가 반전되는 진리표(Truth Table)를 실험적으로 완벽히 재현하였다.
- **벨 상태(Bell State) 생성**: $\Phi^- = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$ 상태를 준비하여 최대 얽힘 상태를 구현하였으며, Fidelity 측정을 통해 이론적 예측값과의 근접성을 확인하였다.
- **양자 알고리즘 실행**:
  - **Deutsch 알고리즘**: 함수가 상수 함수(constant)인지 균형 함수(balanced)인지 단 한 번의 쿼리로 판별하는 데 성공하였다.
  - **Grover 알고리즘**: 4개의 데이터 중 타겟을 찾는 과정을 구현하여, 양자 중첩과 진폭 증폭을 통해 정답 확률을 높이는 과정을 입증하였다.
  - **Bernstein-Vazirani 알고리즘**: 얽힘 없이 중첩만으로 비밀 비트열을 단 한 번의 쿼리로 찾아내는 변형 알고리즘을 성공적으로 구현하였다.
- **양자 시뮬레이션**: 2-큐비트 시스템을 이용하여 4-레벨 양자 조화 진동자(QHO)의 에너지 준위와 위상 진화를 시뮬레이션하였으며, $\Omega t$ 변화에 따른 밀도 행렬의 위상 변화를 관찰하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 논문은 양자 컴퓨팅의 이론적 추상함을 실제 물리적 장치(NMR)와 연결하여 구체적인 구현 경로를 제시하였다. 특히, 고가의 극저온 장비 없이 데스크톱 수준의 NMR 장비로 양자 알고리즘을 실습할 수 있음을 보여줌으로써 양자 컴퓨팅 교육의 진입 장벽을 낮추었다는 점이 큰 강점이다.

### 한계 및 비판적 해석

- **확장성 문제**: NMR 플랫폼은 큐비트 수를 늘릴수록 신호 세기가 지수적으로 감소하는 문제가 있어, 실제 대규모 범용 양자 컴퓨터로 발전하기에는 구조적 한계가 명확하다.
- **얽힘의 부재**: 저자들은 NMR 시스템이 매우 노이즈가 많은 혼합 상태이며, 실제로는 얽힘(entanglement)이 존재하지 않는 임계값 이하의 상태일 수 있음을 언급한다. 이는 DQC1 모델(One-bit of quantum information)과 연결되어, 양자 가속의 원천이 반드시 얽힘이어야 하는가에 대한 근본적인 질문(Quantum Discord의 역할)을 던진다.
- **가정의 단순화**: 실험 결과 분석에서 디코히런스(decoherence)와 펄스 오차를 Fidelity 저하의 원인으로 언급하였으나, 각 큐비트별 구체적인 $T_1, T_2$ 시간과 그에 따른 오차 분석이 더 세밀하게 이루어졌다면 분석의 깊이가 더했을 것이다.

## 📌 TL;DR

본 논문은 양자 컴퓨팅의 기초 이론부터 하드웨어 아키텍처, 그리고 NMR 플랫폼을 이용한 실제 구현 방법까지를 다룬 종합 리뷰 보고서이다. 큐비트의 수학적 정의, GRAPE 알고리즘을 통한 정밀 제어, Pseudo-pure state를 이용한 초기화 기법을 상세히 설명하며, 이를 통해 Grover, Deutsch 등 주요 양자 알고리즘과 양자 조화 진동자 시뮬레이션을 실험적으로 증명하였다. 비록 NMR이 확장성 면에서 한계가 있으나, 교육용 및 기초 연구용 플랫폼으로서의 가치를 극대화하였으며, 특히 '얽힘 없는 양자 가속'의 가능성을 시사하는 혼합 상태 컴퓨팅의 관점을 제공한다.
