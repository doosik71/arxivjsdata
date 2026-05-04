# Fundamentals In Quantum Algorithms: A Tutorial Series Using Qiskit Continued

Daniel Koch, Saahil and Patel, Laura Wessing, Paul M. Alsing (202X)

## 🧩 Problem to Solve

본 문서는 양자 컴퓨팅의 소프트웨어와 하드웨어가 분리되는 시점에서, 물리, 수학, 컴퓨터 과학 전반에 걸쳐 양자 알고리즘을 체계적으로 학습시키기 위한 튜토리얼 시리즈이다. 특히 고수준 양자 컴퓨팅 언어인 Qiskit을 활용하여 이론적 기초와 실제 구현 사이의 간극을 메우는 것을 목표로 한다.

해결하고자 하는 핵심 문제는 다음과 같다. 첫째, Quantum Fourier Transform(QFT) 및 Quantum Phase Estimation(QPE)과 같은 핵심 서브루틴의 수학적 원리를 이해하고 이를 게이트 수준에서 구현하는 방법을 제시하는 것이다. 둘째, 이를 확장하여 Shor's Algorithm, Quantum Counting, Q-Means Clustering, QAOA, VQE와 같은 복잡한 알고리즘을 실제로 구동 가능한 형태로 설계하는 방법론을 제공하는 것이다.

## ✨ Key Contributions

본 튜토리얼의 중심적인 기여는 양자 알고리즘의 이론적 배경을 구체적인 Qiskit 코드와 연결하여, 학습자가 직접 실험하며 원리를 깨닫게 하는 '실습 중심의 학술적 가이드'를 제공했다는 점이다.

특히, 단순한 알고리즘 소개에 그치지 않고, 다음과 같은 설계 아이디어를 강조한다.
- **서브루틴의 모듈화**: QFT $\rightarrow$ QPE $\rightarrow$ Shor's/Counting으로 이어지는 논리적 흐름을 구축하여 복잡한 알고리즘을 단계별로 이해하도록 설계하였다.
- **하이브리드 접근법**: QAOA와 VQE 섹션을 통해 양자 시스템의 상태 준비와 클래식 최적화 알고리즘(Gradient Descent, Nelder-Mead)을 결합하는 하이브리드 양자-클래식 컴퓨팅(Hybrid Quantum-Classical Computing)의 구조를 상세히 다룬다.
- **데이터 매핑 전략**: Q-Means Clustering에서 클래식 데이터를 Bloch Sphere 상의 양자 상태로 매핑하여 유클리드 거리를 내적(Inner Product)으로 변환하는 직관적인 방법론을 제시한다.

## 📎 Related Works

본 문서는 다음과 같은 기존 연구 및 알고리즘을 기반으로 한다.
- **Grover's Algorithm**: 데이터 검색을 위한 진폭 증폭(Amplitude Amplification) 원리를 다루며, 이를 Quantum Counting의 기초로 사용한다.
- **Shor's Algorithm**: 정수 인수분해 문제를 주기 찾기(Period Finding) 문제로 변환하여 지수적 속도 향상을 달성하는 고전적 연구를 기반으로 한다.
- **Ising Model**: 물리적 스핀 시스템의 에너지를 모델링하는 Ising Energy Model을 통해 QAOA의 Cost Function을 정의한다.
- **Variational Principle**: 시스템의 에너지 기대값이 항상 바닥 상태(Ground State) 에너지보다 크거나 같다는 원리를 VQE의 핵심 이론으로 채택한다.

## 🛠️ Methodology

본 보고서는 튜토리얼에서 다루는 5가지 핵심 주제의 방법론을 상세히 설명한다.

### 1. Quantum Fourier Transform (QFT) 및 Quantum Adder
QFT는 이산 푸리에 변환(DFT)을 양자 상태에 적용한 것으로, Hadamard 게이트와 Controlled-Phase($CP$) 게이트의 조합으로 구현된다.
- **구조**: $n$개의 큐비트에 대해 Hadamard 게이트를 적용한 후, 점진적으로 감소하는 위상 회전 게이트를 적용하여 상태를 변환한다.
- **Quantum Adder**: Draper의 방식을 따라, 숫자를 QFT 도메인으로 변환한 후 위상 회전을 통해 덧셈을 수행하고, 다시 $QFT^\dagger$를 통해 결과값을 얻는다.

### 2. Quantum Phase Estimation (QPE) 및 Quantum Counting
Unitary 연산자 $U$의 고유값 $e^{2\pi i \phi}$에서 위상 $\phi$를 추정하는 알고리즘이다.
- **절차**: 제어 큐비트들에 Hadamard를 적용하여 중첩 상태를 만든 뒤, $U^{2^j}$ 형태의 Controlled-U 연산을 수행하여 위상 정보를 제어 큐비트에 인코딩한다. 이후 $QFT^\dagger$를 적용하여 $\phi$ 값을 측정 가능한 상태로 변환한다.
- **Quantum Counting**: Grover 연산자 $G$를 $U$로 설정하여 QPE를 수행함으로써, 마킹된 상태의 개수 $M$을 추정한다.

### 3. Shor's Algorithm
정수 $N$을 인수분해하기 위해 다음의 파이프라인을 따른다.
- **수학적 변환**: $a^r \equiv 1 \pmod N$을 만족하는 최소 주기 $r$을 찾는 문제로 변환한다.
- **양자 서브루틴**: Modulo Exponentiation 연산을 통해 주기 $r$을 양자 상태로 인코딩하고, QFT를 통해 이를 추출한다.
- **클래식 후처리**: Continued Fractions(연분수) 알고리즘을 사용하여 측정된 값으로부터 정수 $r$을 근사하고, $\gcd(a^{r/2} \pm 1, N)$을 통해 인수를 찾는다.

### 4. Q-Means Clustering 및 SWAP Test
클래식 $k$-means clustering의 거리 계산 병목 현상을 양자 내적으로 해결한다.
- **SWAP Test**: 두 양자 상태 $|\psi\rangle$와 $|\phi\rangle$의 유사도를 측정한다. 측정 결과 $|0\rangle$ 상태가 나올 확률은 $P(0) = \frac{1}{2} + \frac{1}{2}|\langle\psi|\phi\rangle|^2$ 이며, 이를 통해 내적 값을 추정한다.
- **데이터 매핑**: 2D 데이터를 Bloch Sphere의 $(\theta, \phi)$ 좌표로 매핑하여, 유클리드 거리를 양자 상태의 내적 값으로 변환하여 처리한다.

### 5. QAOA 및 VQE (Variational Algorithms)
최적화 문제를 해결하기 위해 양자 회로와 클래식 최적화 도구를 결합한다.
- **QAOA**: Cost Function $C$를 위상 연산자 $U(C, \gamma) = e^{-i\gamma C}$로, Mixing 연산자를 $U(B, \beta) = e^{-i\beta B}$로 정의한다. 파라미터 $(\gamma, \beta)$를 클래식하게 최적화하여 최적 상태를 찾는다.
- **VQE**: Hamiltonian $H$의 최솟값(바닥 상태 에너지)을 찾기 위해 Ansatz 회로를 설계하고, 기대값 $\langle \Psi(\vec{\theta}) | H | \Psi(\vec{\theta}) \rangle$을 최소화하는 $\vec{\theta}$를 Nelder-Mead 등의 최적화 기법으로 검색한다.

## 📊 Results

본 문서는 특정 벤치마크 데이터셋보다는 알고리즘의 성공적인 구현 사례를 정성적/정량적으로 제시한다.

- **Shor's Algorithm**: $N=35$와 같은 작은 수에 대해 주기 $r$을 성공적으로 찾아 인수를 분리하는 과정을 시뮬레이션으로 증명하였다.
- **Quantum Counting**: 마킹된 상태의 개수가 증가함에 따라 QPE의 측정 결과가 해당 개수의 수학적 기대치와 일치함을 확인하였다.
- **Q-Means**: SWAP Test를 통해 데이터 포인트 간의 유사도를 측정하고, 이를 통해 클래식 $k$-means와 유사한 클러스터링 결과가 도출됨을 시각화(Heatmap)를 통해 보여주었다.
- **QAOA & VQE**: Ising Model 및 Hamiltonian 시스템에서 Gradient Descent와 Nelder-Mead 최적화기를 사용하여 에너지 최솟값에 수렴하는 과정을 확인하였다. 특히 VQE에서는 단일 큐비트 Ansatz를 통해 바닥 상태 에너지를 정확히 추정하는 결과를 얻었다.

## 🧠 Insights & Discussion

본 튜토리얼을 통해 도출된 주요 통찰은 다음과 같다.

1. **정밀도와 자원의 트레이드오프**: QPE에서 큐비트 수 $n$을 늘리면 위상 $\phi$의 추정 정밀도는 높아지지만, 회로의 깊이가 깊어져 실제 하드웨어에서는 노이즈에 취약해지는 문제가 발생한다.
2. **측정의 확률적 특성**: VQE와 QAOA에서 기대값을 계산하기 위해 수많은 반복 측정(Shots)이 필요하며, 이는 계산 비용을 증가시킨다. 특히 Gradient Descent는 측정 노이즈로 인해 'Bumpy'한 에너지 랜드스케이프를 형성하여 지역 최솟값(Local Minima)에 빠질 위험이 크다.
3. **매핑의 중요성**: Q-Means에서 보았듯이, 클래식 데이터를 양자 상태로 어떻게 매핑하느냐에 따라 거리 측정의 왜곡 정도가 달라진다. 대칭적 매핑(Symmetric Mapping)이 비대칭적 매핑보다 더 안정적인 거리 측정 결과를 제공함을 확인하였다.
4. **하이브리드 구조의 유연성**: VQE와 QAOA는 하드웨어의 제약을 클래식 최적화 단계에서 보완할 수 있는 구조를 가지고 있어, NISQ(Near-term Intermediate-Scale Quantum) 시대에 가장 현실적인 대안임을 시사한다.

## 📌 TL;DR

본 보고서는 Qiskit을 이용한 양자 알고리즘의 핵심 서브루틴(QFT, QPE)부터 고급 알고리즘(Shor's, QAOA, VQE, Q-Means)까지의 이론과 구현을 상세히 분석한다. 특히 **수학적 모델 $\rightarrow$ 양자 회로 설계 $\rightarrow$ 클래식 최적화 $\rightarrow$ 결과 검증**으로 이어지는 통합적 방법론을 제시하며, 양자 컴퓨팅이 단순한 계산 속도 향상을 넘어 최적화 및 데이터 분석 분야에서 어떻게 활용될 수 있는지에 대한 실전적 가이드를 제공한다.