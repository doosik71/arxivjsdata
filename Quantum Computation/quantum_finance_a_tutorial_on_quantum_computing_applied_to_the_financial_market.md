# Quantum Finance: um tutorial de computação quântica aplicada ao mercado financeiro

Askery Canabarro, Taysa M. Mendonça, Ranieri Nery, George Moreno, Anton S. Albino, Gleydson F. de Jesus, and Rafael Chaves (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 금융 시장의 **포트폴리오 최적화(Portfolio Optimization)** 문제이다. 포트폴리오 최적화란 투자자가 보유한 자산들 사이의 비중을 조절하여, 주어진 리스크 수준에서 기대 수익을 최대화하거나, 목표 수익 수준에서 리스크를 최소화하는 최적의 조합을 찾는 것이다.

이 문제는 고전적인 Markowitz의 현대 포트폴리오 이론(Modern Portfolio Theory)을 통해 접근할 수 있으나, 실제 투자 환경에서는 다음과 같은 제약 조건들이 추가된다.

- **양수 제약(Positivity constraints):** 공매도를 허용하지 않고 자산을 매수만 하는 경우.
- **정수 제약(Integer constraints):** 자산을 특정 단위로만 투자해야 하는 경우.
- **상한선 제약(Upper limits):** 유동성 문제로 인해 특정 자산에 투자할 수 있는 최대 금액이 정해진 경우.

이러한 제약 조건이 추가될 경우, 포트폴리오 최적화는 **조합 최적화(Combinatorial Optimization)** 문제로 변하며, 탐색해야 할 경우의 수가 변수 개수에 따라 지수적으로 증가($2^n$)한다. 따라서 고전 컴퓨터로는 변수가 많아질 때 계산 시간이 기하급수적으로 늘어나는 한계가 있으며, 이를 효율적으로 해결하기 위한 양자 알고리즘의 적용이 필요하다.

## ✨ Key Contributions

본 논문의 주요 기여는 양자 컴퓨팅의 기초부터 시작하여, 이를 금융 시장의 구체적인 문제에 적용하는 전체 과정을 다루는 **학술적 튜토리얼**을 제공하는 것이다. 핵심적인 설계 아이디어는 다음과 같다.

1. **QAOA의 적용:** 조합 최적화 문제에 강점을 가진 **Quantum Approximate Optimization Algorithm (QAOA)**을 사용하여 포트폴리오 선택 문제를 해결한다.
2. **이진 최적화 매핑:** 어떤 자산을 포트폴리오에 포함할지 여부를 결정하는 이진 변수 $x \in \{0, 1\}^n$ 문제로 정식화하고, 이를 양자 시스템의 Hamiltonian으로 매핑한다.
3. **실제 데이터 기반 검증:** 브라질 증권거래소의 실제 주식 데이터(Braskem, Itaú, Klabin, Vale)를 사용하여 고전적 최적화 결과와 양자 알고리즘의 결과가 일치하는지 증명하는 Proof-of-Concept를 제시한다.
4. **양자-고전 하이브리드 접근:** 양자 회로를 통해 상태를 준비하고, 클래식 최적화 도구를 통해 매개변수를 조정하는 변분 양자 알고리즘(Variational Quantum Algorithm) 구조를 채택한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 배경을 설명한다.

- **Markowitz의 포트폴리오 이론 (1952):** 리스크와 수익의 상관관계를 분석하여 '효율적 투자선(Efficient Frontier)' 개념을 도입한 고전적 접근 방식이다.
- **NISQ (Noisy Intermediate-Scale Quantum) 시대:** 현재의 양자 컴퓨터는 오류 수정(Error Correction)이 불가능하고 노이즈가 많은 초기 단계이다. 따라서 하드웨어의 한계를 인정하고, 일부 계산을 고전 컴퓨터가 분담하는 하이브리드 알고리즘(VQE, QAOA 등)이 주목받고 있다.
- **Adiabatic Quantum Computing (AQC):** 시스템을 매우 천천히 변화시켜 초기 상태의 기저 상태(Ground State)를 최종 Hamiltonian의 기저 상태로 유지하며 최적해를 찾는 방식이다. QAOA는 이 AQC의 연속적인 진화를 이산적으로 근사한 알고리즘으로 이해될 수 있다.

## 🛠️ Methodology

### 1. 포트폴리오 최적화의 수학적 정식화

논문에서는 다음과 같은 목적 함수를 최소화하는 문제를 정의한다.

$$\min_{x \in \{0,1\}^n} bx^T \sigma x - \mu^T x$$
$$\text{subject to: } \mathbf{1}^T x = B$$

- $x$: 자산 선택 여부를 나타내는 이진 벡터 ($1$: 포함, $0$: 제외).
- $\mu$: 각 자산의 기대 수익률 벡터.
- $\sigma$: 자산 간의 공분산 행렬(Covariance Matrix).
- $b$: 투자자의 리스크 회피 성향을 조절하는 파라미터.
- $B$: 포트폴리오에 포함할 자산의 개수(예산).

제약 조건 $\mathbf{1}^T x = B$는 페널티 항 $P(x) = \alpha(B - \sum x_i)^2$를 목적 함수에 추가함으로써 해결한다.

### 2. QAOA (Quantum Approximate Optimization Algorithm)

QAOA는 다음과 같은 하이브리드 절차를 따른다.

**A. Hamiltonian 구성**

- **Cost Hamiltonian ($H_C$):** 위에서 정의한 목적 함수를 양자 연산자로 변환한 것이다. 이진 변수 $x_i$는 Pauli-Z 연산자를 이용하여 $\hat{x}_i = \frac{I - Z_i}{2}$로 매핑된다. 결과적으로 $H_C$는 $Z$ 연산자의 합과 $Z_i Z_j$ 곱의 형태로 표현된다.
- **Mixer Hamiltonian ($H_B$):** 상태 간의 전이를 가능하게 하며, 모든 큐비트에 대해 Pauli-X 연산자를 적용한 $\sum X_i$ 형태로 정의된다.

**B. 상태 준비 및 진화**
초기 상태 $|\psi_{ini}\rangle$ (모든 상태의 균일한 중첩 상태)에서 시작하여, 두 유니타리 연산자 $U_C(\gamma) = e^{-i\gamma H_C}$와 $U_B(\beta) = e^{-i\beta H_B}$를 $p$번 교대로 적용한다.

$$|\psi(\beta, \gamma)\rangle = U_B(\beta_p) U_C(\gamma_p) \dots U_B(\beta_1) U_C(\gamma_1) |\psi_{ini}\rangle$$

**C. 고전적 최적화**
양자 컴퓨터에서 측정된 기대값 $\langle \psi(\beta, \gamma) | H_C | \psi(\beta, \gamma) \rangle$을 최소화하도록 클래식 최적화 알고리즘이 $\beta$와 $\gamma$ 파라미터를 업데이트한다.

### 3. 구현 세부 사항

- **회로 구성:** $U_B$는 $R_x$ 회전 게이트로 구현하며, $U_C$는 $R_z$ 게이트와 CNOT 게이트(두 큐비트 간 상호작용 구현용)의 조합으로 구현한다.
- **시뮬레이터:** ATOS QLM (Quantum Learning Machine) 시뮬레이터를 사용하여 노이즈 환경을 모사하고 실험을 수행하였다.

## 📊 Results

### 1. 실험 설정

- **대상 자산:** 브라질 주식 4종 (BRKM5, ITUB4, KLBN4, VALE3).
- **제약 조건:** 포트폴리오에 포함할 자산 수 $B=2$.
- **리스크 파라미터:** $b=0.5$.
- **비교 대상:** 고전적 Markowitz 분석을 통해 얻은 최적 포트폴리오.

### 2. 정량적 결과

- **최적해 발견:** QAOA를 통해 20개 레이어($p=20$)를 적용했을 때, 최적의 상태인 $|1001\rangle$ (Braskem과 Vale 선택)이 약 **90%의 확률**로 측정되었다. 이는 고전적 분석 결과와 일치한다.
- **레이어 수($p$)에 따른 성능:**
  - **확률:** 레이어 수가 증가함에 따라 최적해 $|1001\rangle$을 찾을 확률이 대체로 증가하지만, 단조 증가하지 않고 진동하는 경향을 보였다.
  - **에너지:** 찾은 솔루션의 에너지는 $p$가 증가함에 따라 점차 낮아지며, 약 16~20 레이어 구간에서 기저 상태 에너지 $\langle H \rangle_{min} = -0.6165$에 수렴하는 양상을 보였다.

## 🧠 Insights & Discussion

**강점 및 유효성:**
본 논문은 복잡한 금융 최적화 문제를 양자 Hamiltonian으로 변환하고, 이를 NISQ 시대의 알고리즘인 QAOA로 해결할 수 있음을 성공적으로 보여주었다. 특히 실제 시장 데이터를 사용하여 이론적 가능성을 실증적으로 입증했다는 점에서 가치가 있다.

**한계 및 논의 사항:**

1. **비단조적 수렴:** 레이어 수 $p$를 늘린다고 해서 반드시 확률이 높아지지 않는 현상이 관찰되었다. 이는 변분 알고리즘 특성상 클래식 최적화 도구가 지역 최솟값(Local Minima)에 빠질 수 있음을 시사한다.
2. **확장성 문제:** 본 실험은 4개의 자산만을 다루었으나, 자산의 수가 늘어날 경우 필요한 큐비트 수와 게이트 깊이가 증가하며, 이는 현재의 노이즈가 많은 양자 하드웨어에서 결맞음 시간(Coherence time) 문제로 이어질 수 있다.
3. **최적화 도구 의존성:** 결과가 사용된 클래식 최적화 알고리즘(COBYLA 등)의 성능에 크게 의존한다는 점이 명시되었다.

## 📌 TL;DR

본 논문은 금융 시장의 포트폴리오 최적화 문제를 이진 조합 최적화 문제로 정의하고, 이를 **QAOA(Quantum Approximate Optimization Algorithm)**를 통해 해결하는 방법을 제시한 튜토리얼 성격의 연구이다. 브라질 주식 4종 데이터를 활용한 실험 결과, 양자 시뮬레이터를 통해 고전적 최적해와 일치하는 결과를 얻었으며, 레이어 수 $p$가 증가함에 따라 최적해의 에너지 값이 수렴함을 확인하였다. 이 연구는 향후 더 많은 자산을 포함한 실제 규모의 금융 최적화 문제에 양자 컴퓨팅을 적용하기 위한 기초적인 가이드라인과 가능성을 제시한다.
