# Efficient Universal Quantum Compilation: An Inverse-free Solovay-Kitaev Algorithm

Adam Bouland, Tudor Giurgiță-Tiron (2021)

## 🧩 Problem to Solve

본 논문은 양자 컴퓨팅의 핵심 과제인 임의의 Unitary 연산을 유한한 Universal gate set을 이용하여 효율적으로 구현하는 '양자 컴파일(Quantum Compilation)' 문제를 다룬다. 기존의 Solovay-Kitaev (S-K) 알고리즘은 임의의 Unitary를 $\text{poly-log}(\epsilon^{-1})$의 게이트 길이로 근사할 수 있음을 보장하지만, 치명적인 제약 조건이 존재한다. 바로 사용되는 게이트 집합이 'Inverse-closed'여야 한다는 점이다. 즉, 집합 내의 모든 게이트에 대해 그 정확한 역행렬(Exact inverse)이 집합 내에 존재해야만 한다.

이러한 제약은 수학적으로는 편리하지만, 실제 물리적 하드웨어에서 구현되는 게이트 집합이 항상 역행렬을 포함한다는 보장이 없으므로 실용적인 한계가 된다. 따라서 본 연구의 목표는 Inverse-closure 조건 없이도 poly-logarithmic한 시간 복잡도를 유지하며 임의의 Unitary를 근사할 수 있는 'Inverse-free Solovay-Kitaev 알고리즘'을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 게이트 집합의 수학적 구조(역행렬의 존재 여부)에 관계없이, universality만 만족한다면 효율적인 컴파일이 가능함을 증명하고 이를 위한 알고리즘을 제시한 것이다.

중심 아이디어는 **Self-correcting sequences(자가 수정 시퀀스)**의 도입이다. 저자들은 $\epsilon$-정밀도로 구현된 근사 Pauli 연산자들을 특정 순서로 조합하면, 개별 연산자의 오차들이 서로 상쇄되어 결과적으로 Identity 연산자에 $\mathcal{O}(\epsilon^2)$ 수준으로 매우 가깝게 접근하는 시퀀스를 구성할 수 있음을 보였다. 이를 통해 $\mathcal{O}(\epsilon)$ 정밀도의 근사 역행렬을 $\mathcal{O}(\epsilon^2)$ 정밀도의 고정밀 역행렬로 변환하는 'Inverse factory'를 구축함으로써, S-K 알고리즘의 재귀 구조를 유지하면서도 역행렬 조건의 제약을 제거하였다.

## 📎 Related Works

기존의 Solovay-Kitaev 알고리즘[Kit97, DN05]은 게이트 집합이 Inverse-closed라는 가정하에 Group commutator($VWV^\dagger W^\dagger$)를 사용하여 오차를 $\epsilon \to \mathcal{O}(\epsilon^{3/2})$로 빠르게 줄여나간다.

이후 일부 연구[SCHL16, BO18]에서는 Inverse-closure 조건을 완화하려 시도하였다. 이들은 게이트 집합이 특정 유한 그룹의 Exact irreducible representation(irrep)을 포함하고 있다면 효율적인 컴파일이 가능함을 보였다. 하지만 이 역시 '정확한(Exact)' 구조가 필요하다는 점에서 완전한 Inverse-free라고 볼 수 없었다. 또한, 정보 이론적 관점에서 역행렬 없이 짧은 시퀀스가 존재한다는 비구성적(Non-constructive) 증명[OSH20]은 있었으나, 이를 실제로 찾아낼 수 있는 알고리즘은 제시되지 않았다. 본 논문은 이러한 한계를 넘어, 아무런 구조적 가정 없이 알고리즘적으로 고정밀 근사를 수행하는 방법을 제시함으로써 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 재귀 구조

본 알고리즘은 표준 S-K 알고리즘의 재귀적 구조를 따른다. $n$번째 단계에서 목표 Unitary $U$에 대한 $\epsilon_n$-근사치 $U_n$을 얻기 위해, 이전 단계의 근사치 $U_{n-1}$과의 차이를 Group commutator로 메우는 방식을 취한다. 이때 핵심은 $\mathcal{O}(\epsilon^2)$ 정밀도의 역행렬을 생성하는 Inverse factory를 사용하여 S-K의 오차 감소율을 유지하는 것이다.

### 2. Self-correcting Sequences

역행렬이 없는 환경에서 오차를 제어하기 위해, 저자들은 Generalized Pauli group의 특성을 이용한다. $\epsilon$-근사된 Pauli 연산자 $X', Z'$가 있을 때, 다음과 같은 시퀀스를 구성하면 $\mathcal{O}(\epsilon^2)$ 정밀도로 Identity에 수렴하게 된다.

- **Qubit ($d=2$)의 경우:**
  $$J_2(X', Y') \equiv X'Y'X'Y'^2X'Y'X' = I + \mathcal{O}(\epsilon^2)$$
- **일반 차원 ($d \ge 2$)의 경우:**
  $$J_d(A, B) \equiv [A^d B]^{d-1} A [B^d A]^{d-1} B = I + \mathcal{O}(\epsilon^2)$$

이 시퀀스의 직관적 원리는 두 개의 부분 그룹(X-generated subgroup과 Z-generated subgroup)에 대해 중첩된 Group twirl을 수행하는 것이다. 이를 통해 각 연산자의 오차가 서로 보완적으로 상쇄되어 선형 항(linear order)의 오차가 사라지게 된다.

### 3. Inverse Factory

Self-correcting sequence를 활용하여, 임의의 Unitary $V$에 대한 저정밀도 근사 역행렬 $\hat{V}^\dagger$를 고정밀도 역행렬 $\hat{\hat{V}}^\dagger$로 변환한다.

- **작동 원리:**
  근사 역행렬 $\hat{V}^\dagger$와 $V$의 곱인 $\hat{V}^\dagger V$는 Identity의 $\epsilon$-근사치이다. 이를 Pauli 연산자와 조합하여 $\epsilon$-근사된 Pauli 연산자를 만들고, 앞서 언급한 Self-correcting sequence $J_d$에 대입한다.
- **결과:**
  이렇게 구성된 시퀀스에서 한쪽 끝의 $V$를 제거하면, 결과적으로 $\hat{\hat{V}}^\dagger V = I + \mathcal{O}(\epsilon^2)$를 만족하는 고정밀 역행렬 $\hat{\hat{V}}^\dagger$를 얻게 된다.

### 4. 최종 알고리즘 흐름

1. $n=0$에서 $\epsilon_0$-net을 통해 기초 근사치를 찾는다.
2. 재귀적으로 $U_{n-1}, V_{n-1}, W_{n-1}$ 및 이들의 근사 역행렬 $\hat{V}^\dagger_{n-1}, \hat{W}^\dagger_{n-1}$를 구한다.
3. 미리 계산된 근사 Pauli 연산자들을 이용하여 Inverse factory를 가동, $\mathcal{O}(\epsilon^2_{n-1})$ 정밀도의 역행렬 $\hat{\hat{V}}^\dagger_{n-1}, \hat{\hat{W}}^\dagger_{n-1}$를 생성한다.
4. 이를 Group commutator $U_n \equiv V_{n-1} W_{n-1} \hat{\hat{V}}^\dagger_{n-1} \hat{\hat{W}}^\dagger_{n-1} U_{n-1}$에 대입하여 오차를 $\mathcal{O}(\epsilon^{3/2}_{n-1})$로 줄인다.

## 📊 Results

### 1. 정량적 결과 및 복잡도 분석

본 알고리즘은 임의의 Universal gate set에 대해 $\epsilon$-근사를 수행하며, 게이트 시퀀스의 길이는 $\mathcal{O}(\text{polylog}(\epsilon^{-1}))$의 복잡도를 가진다.

- **게이트 길이의 지수(Exponent) 비교:**
  - 표준 S-K (Inverse-closed): $\gamma \approx 3.97$
  - 본 연구의 Inverse-free S-K (Qubit): $\gamma \approx 8.62$
- **차원 $d$에 따른 확장성:**
  지수 $\gamma_d$는 차원 $d$에 대해 $\Theta(\log d)$의 속도로 증가한다. 구체적으로 $\gamma_d = \frac{\log(8d^2+1)}{\log(3/2)}$로 표현된다.

### 2. 계산 시간(Runtime)

시퀀스의 길이는 늘어났지만, 실제 알고리즘을 수행하는 계산 시간은 더 효율적이다. 근사 Pauli 연산자들을 미리 계산하여 저장해 둔다면, 각 단계에서 필요한 재귀 호출 횟수는 5회로 고정된다. 따라서 런타임 복잡도는 $\mathcal{O}(\log^{2.7} \epsilon^{-1})$로, 표준 S-K와 동일한 수준을 유지한다.

### 3. 적용 범위 확장

본 방법론은 $SU(d)$뿐만 아니라 Special Linear group $SL(d, \mathbb{C})$로도 자연스럽게 확장 가능하다. 이는 Inverse factory의 핵심 원리가 $\det(V)=1$이라는 조건에 기반하고 있기 때문이다.

## 🧠 Insights & Discussion

### 1. 강점 및 의의

본 연구는 양자 컴파일링의 오랜 난제였던 Inverse-closure 제약을 완전히 제거하였다. 이는 하드웨어 제약이 많은 실제 양자 소자에서 특정 게이트 세트만을 사용하여 임의의 연산을 구현할 때 매우 강력한 이론적 도구가 된다. 특히, 비공선(non-collinear) 축에 대한 두 개의 무리수 회전만으로도 효율적인 컴파일이 가능하다는 점을 시사한다.

### 2. 한계 및 비판적 해석

가장 큰 trade-off는 게이트 시퀀스의 길이 증가이다. 역행렬을 직접 사용할 수 없는 대신 Self-correcting sequence를 통한 '우회로'를 택했기 때문에, 최종 시퀀스의 길이를 결정하는 지수 $\gamma$가 약 2배 이상 증가하였다($3.97 \to 8.62$). 이는 실제 구현 시 더 많은 게이트를 사용해야 함을 의미하며, 이는 decoherence 시간이 제한적인 현재의 NISQ 장치에서는 부담이 될 수 있다.

### 3. 향후 연구 방향

저자들은 지수 $\gamma$를 낮추기 위해 nested group commutator 접근 방식을 결합하는 방안을 제시하였다. 또한, Pauli group 외에 다른 그룹 구조에서도 유사한 self-correcting sequence를 찾을 수 있을지에 대한 가능성을 열어두었다.

## 📌 TL;DR

본 논문은 역행렬(Inverse)이 없는 Universal gate set에서도 효율적인 양자 컴파일이 가능함을 증명한 **Inverse-free Solovay-Kitaev 알고리즘**을 제안한다. 근사 Pauli 연산자들을 조합해 오차를 스스로 상쇄하는 **Self-correcting sequences**를 설계하고, 이를 통해 고정밀 역행렬을 생성하는 **Inverse factory**를 구축함으로써 $\mathcal{O}(\text{polylog}(\epsilon^{-1}))$의 복잡도를 달성하였다. 비록 표준 S-K보다 게이트 길이는 길어지지만, 게이트 집합의 구조적 제약을 없앴다는 점에서 양자 복잡도 이론 및 실제 컴파일러 설계에 중요한 기여를 한다.
