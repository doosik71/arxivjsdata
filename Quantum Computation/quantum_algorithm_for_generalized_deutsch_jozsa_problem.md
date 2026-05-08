# Quantum Algorithm for Generalized Deutsch-Jozsa Problem

Dong Pyo Chi, Jinsoo Kim, and Soojoon Lee (2000)

## 🧩 Problem to Solve

본 논문은 양자 컴퓨팅의 초기 핵심 문제 중 하나인 Deutsch-Jozsa(DJ) 문제를 일반화하고, 이를 효율적으로 해결하기 위한 양자 알고리즘을 제안한다.

기존의 Deutsch-Jozsa 문제는 불리언 함수 $f: \mathbb{Z}_2^n \to \mathbb{Z}_2$가 상수 함수(constant)인지 아니면 균형 함수(balanced, 입력값의 정확히 절반이 0이고 절반이 1인 함수)인지를 판별하는 문제이다. 이 문제는 고전적 컴퓨터로는 최악의 경우 모든 입력을 탐색해야 하지만, 양자 컴퓨터로는 단 한 번의 함수 평가(evaluation)만으로 해결 가능하다.

본 연구의 목표는 함수 $f$의 치역을 $\mathbb{Z}_M$으로 확장하고 'balanced'의 개념을 'evenly distributed'로 일반화하여, 일반화된 Deutsch-Jozsa 문제를 단 한 번의 함수 평가로 해결하는 양자 알고리즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **문제의 일반화**: 함수 $f: \mathbb{Z}_N \to \mathbb{Z}_M$에 대하여 'evenly distributed'라는 개념을 정의함으로써 기존의 DJ 문제를 확장하였다.
2. **단일 평가 알고리즘 제안**: 보조 레지스터(auxiliary register)의 초기 상태를 특정 상태 $|\Psi\rangle$로 설정함으로써, 일반화된 DJ 문제를 단 한 번의 $U_f$ 연산으로 해결하는 방법을 제시하였다.
3. **초기화 없는 알고리즘 설계**: 보조 레지스터의 초기 상태를 미리 설정(initialization)하지 않고도 문제를 해결할 수 있는 변형 알고리즘을 제안하였다. 이는 $U_f$ 연산을 두 번 수행함으로써 보조 레지스터의 상태를 원래대로 복구하며 위상 변화만을 추출하는 방식이다.

## 📎 Related Works

논문은 다음과 같은 관련 연구를 언급한다.

- **Deutsch-Jozsa (1992)**: 양자 컴퓨터의 효율성을 증명한 초기 promise problem으로, 본 연구의 기초가 된다.
- **Collision 및 Claw problems**: $\nu$-to-one 함수(여러 입력이 동일한 출력값을 갖는 함수)에 대한 논의가 있으며, 본 논문에서 정의한 'evenly distributed' 함수는 $\nu$-to-one 함수의 특수한 경우에 해당한다.
- **NMR 양자 컴퓨터 구현**: DJ 알고리즘이 단순함에도 불구하고 양자 우위(quantum advantage)를 보여주기 좋기 때문에 많은 NMR 구현 연구가 진행되었음을 언급하며, 특히 Cleve et al.의 버전과 조건부 위상 변환(conditional phase transform) 방식의 차이를 논한다.

## 🛠️ Methodology

### 1. Evenly Distributed 함수의 정의

함수 $f: \mathbb{Z}_N \to \mathbb{Z}_M$이 다음과 같은 조건을 만족할 때 **evenly distributed**라고 정의한다.

- $f$가 동일한 간격(equally spaced)을 가진 $K$개의 값을 가지며, $\nu = N/K$인 $\nu$-to-one 함수인 경우.
- 즉, $\{f(x) : x \in \mathbb{Z}_N\} = \{j\mu + t : j \in \mathbb{Z}_K\}$를 만족하며, 각 $j$에 대해 $f(x) = j\mu + t$를 만족하는 $x$의 개수가 모두 동일해야 한다.

### 2. 보조 레지스터 초기화를 이용한 알고리즘

함수의 입력 크기를 $N=2^n$, 출력 크기를 $M=2^m$이라고 가정한다. 제어 레지스터(control register)는 $|0^n\rangle$으로, 보조 레지스터(auxiliary register)는 다음과 같이 초기화한다.
$$|\Psi\rangle = F|-\xi\rangle = \frac{1}{\sqrt{M}} \sum_{z=0}^{M-1} \omega^{-\xi z / M} |z\rangle \quad (\xi \in \mathbb{Z}_M, \xi \neq 0)$$
여기서 $\omega_M = e^{2\pi i/M}$이며 $F$는 양자 푸리에 변환(QFT)이다.

**알고리즘 절차:**

1. $W^n \otimes I$ 적용 (Walsh-Hadamard 연산)
2. $U_f$ 적용 (함수 평가 연산: $|x\rangle |y\rangle \to |x\rangle |y+f(x)\rangle$)
3. $W^n \otimes I$ 적용

최종 상태의 제어 레지스터의 상태 $|y\rangle$의 계수 $S^y$는 다음과 같다.
$$S^y = \frac{1}{N} \sum_{x=0}^{N-1} (-1)^{x \cdot y} \omega^{\xi f(x) / M}$$

- $f$가 상수 함수이면: $y=0$일 때 $S^0 \neq 0$이고, $y \neq 0$이면 $S^y = 0$이 되어 측정 결과는 $|0^n\rangle$이 된다.
- $f$가 evenly distributed이면: $y=0$일 때 $S^0 = 0$이 되어 측정 결과는 $|0^n\rangle$이 될 수 없다.
따라서 측정 결과가 $|0^n\rangle$이면 $f$는 evenly distributed가 아니며, 그렇지 않으면 $f$는 상수 함수가 아니라고 결론 내릴 수 있다.

### 3. 보조 레지스터 초기화가 필요 없는 알고리즘

보조 레지스터의 초기 상태 $|\Psi\rangle = a|0\rangle + b|1\rangle$가 임의의 상태일 때, 다음과 같은 시퀀스를 통해 $|x\rangle \to (-1)^{f(x)}|x\rangle$ 위상 변환을 구현하고 보조 레지스터를 복구한다.
$$\text{Procedure: } W^n \otimes I \to U_f \to I \otimes \sigma_z \to U_f \to I \otimes \sigma_z \to W^n \otimes I$$
이 과정은 보조 레지스터의 상태를 변화시키지 않으면서 함수 $f$의 정보를 위상에 인코딩한다.

이를 일반화된 문제로 확장하기 위해 비트와이즈 연산 $\oplus$와 패리티 함수 $p: \mathbb{Z}_2^m \to \mathbb{Z}_2$를 도입하여 다음과 같이 구성한다.

- 보조 레지스터의 상태: $|\Psi\rangle = \bigotimes_{j=0}^{m-1} (a_j|0\rangle + b_j|1\rangle)$
- 연산 순서: $W^n \otimes I \to U^\oplus_f \to I \otimes \sigma_z^{\otimes m} \to U^\oplus_f \to I \otimes \sigma_z^{\otimes m} \to W^n \otimes I$
이 알고리즘은 $f$의 패리티가 상수인지 혹은 evenly distributed인지를 판별한다.

## 📊 Results

### 정량적 비교 (Complexity)

- **고전적 알고리즘**: $K$를 알고 있는 경우 최악의 상황에서 $\nu + 1$번의 평가가 필요하며, $K$를 모르는 경우 $N/2 + 1$번의 평가가 필요하다.
- **제안된 양자 알고리즘**: 보조 레지스터를 초기화할 수 있다면 단 **1회**의 함수 평가로 해결 가능하다. 보조 레지스터 초기화 없이 구현할 경우 **2회**의 함수 평가가 필요하다.

### 실험적 결과

본 논문은 수식 기반의 이론적 증명을 중심으로 하며, 구체적인 수치 데이터나 벤치마크 결과보다는 알고리즘의 수학적 정당성과 복잡도 감소를 입증하는 데 집중한다. 제안된 알고리즘이 $f$가 상수 함수일 때와 evenly distributed일 때 서로 직교(orthogonal)하는 상태를 생성함을 보임으로써 판별 가능성을 증명하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 매우 단순한 형태의 DJ 문제를 일반적인 함수 영역으로 확장함으로써, 양자 알고리즘이 가진 위상 간섭(phase interference)의 원리가 더 넓은 범위의 문제(예: 주기 찾기, $\nu$-to-one 함수 판별)에 적용될 수 있음을 이론적으로 보여주었다.

### 한계 및 논의사항

- **보조 레지스터의 제약**: 초기화 없는 알고리즘의 경우, 보조 레지스터의 초기 상태가 분리 가능(separable)하다는 가정이 필요하다. 이를 제거하기 위해서는 추가적인 기법이 필요함이 언급되었다.
- **평가 횟수의 트레이드-오프**: 보조 레지스터의 초기 상태를 정확히 준비하는 것은 물리적으로 어려울 수 있다. 이를 해결하기 위해 $U_f$를 두 번 사용하여 초기화 과정을 생략하는 대안을 제시한 점은 실용적인 관점에서 중요한 통찰이다.
- **물리적 구현**: NMR 양자 컴퓨터와 같은 실제 환경에서 $U_f$ 연산을 구현하는 비용이 고전적 연산 비용보다 훨씬 클 수 있다는 점은 논문에서 명시적으로 다루지 않았으나, 실제 적용 시 고려해야 할 요소이다.

## 📌 TL;DR

본 논문은 기존의 Deutsch-Jozsa 문제를 $f: \mathbb{Z}_N \to \mathbb{Z}_M$ 형태의 일반적인 함수로 확장하여, 함수가 상수 함수인지 또는 'evenly distributed' 함수인지를 판별하는 양자 알고리즘을 제안하였다. 보조 레지스터를 특정 상태로 초기화하면 단 1회의 함수 평가로 해결 가능하며, 초기화가 불가능한 경우 2회의 평가를 통해 동일한 문제를 해결할 수 있음을 수학적으로 증명하였다. 이 연구는 양자 푸리에 변환과 위상 인코딩을 이용한 주기성 및 분포 판별 문제의 이론적 토대를 제공한다.
