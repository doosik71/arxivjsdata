# Foundational Patterns for Efficient Quantum Computing

Austin Gilliam, Charlene Venci, Sreraman Muralidharan, Vitaliy Dorum, Eric May, Rajesh Narasimhan, and Constantin Gonciulea (2021)

## 🧩 Problem to Solve

본 논문은 고전 컴퓨터로는 해결이 어렵거나 불가능한 NP-hard 문제들을 해결하기 위해, 양자 컴퓨팅의 추상적인 알고리즘을 실제 문제 해결에 적용 가능한 구체적인 '패턴(Pattern)'으로 정형화하는 것을 목표로 한다.

기존의 양자 컴퓨팅 접근 방식은 선형 대수학이나 양자 역학의 복잡한 수학적 표기법에 과도하게 의존하는 경향이 있어, 컴퓨터 과학 관점에서의 접근성이 떨어진다는 문제가 있었다. 따라서 본 연구는 수학적 복잡성을 최소화하고 기하학적 직관을 활용한 시각적 접근 방식을 통해, 양자 알고리즘을 실제 비즈니스 및 컴퓨터 과학 문제에 적용할 수 있는 효율적인 설계 패턴을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 양자 컴퓨팅을 알고리즘의 집합이 아닌, 재사용 가능한 '패턴'의 관점에서 재정의한 것이다. 가장 중심적인 설계 아이디어는 다음과 같다.

1. **Quantum Dictionary Pattern 도입**: 키(Key)와 값(Value) 레지스터를 얽힘(Entanglement) 상태로 구성하여, 클래식 컴퓨터의 딕셔너리 구조를 양자 상태로 구현하는 패턴을 제안한다.
2. **기하학적 시각화 접근법**: 복잡한 수식 대신 화살표(Amplitude)와 회전(Rotation)이라는 기하학적 직관을 통해 양자 상태와 게이트의 동작을 설명한다.
3. **숫자 표현의 정형화**: Phase Estimation을 이용하여 숫자를 각도로 매핑하고, 이를 통해 양자 상태 내에서 숫자를 인코딩하고 조작하는 방법을 체계화하였다.
4. **실질적 문제 적용**: 제안한 패턴을 Quadratic Unconstrained Binary Optimization (QUBO) 및 Subset Sum 문제와 같은 NP-hard 문제 해결에 적용하여 그 효용성을 입증하였다.

## 📎 Related Works

논문은 Shor의 소인수분해 알고리즘, Grover의 탐색 알고리즘, Quantum Fourier Transform (QFT) 등 기존의 근본적인 양자 알고리즘들을 기반으로 한다.

기존의 연구들이 특정 알고리즘의 이론적 복잡도나 수학적 증명에 집중했다면, 본 연구는 이러한 알고리즘들을 조합하여 실제 문제를 풀기 위한 '빌딩 블록'으로 사용하는 패턴 언어에 집중한다는 점에서 차별점을 갖는다. 특히, 단순한 알고리즘 적용을 넘어 'Quantum Dictionary'라는 상위 수준의 추상화 계층을 도입하여, 다양한 비즈니스 문제를 동일한 프레임워크 내에서 공식화하고 해결할 수 있는 경로를 제시한다.

## 🛠️ Methodology

### 1. 시각적 양자 시스템 모델링

본 논문은 Qubit을 확률적 비트의 일반화된 형태로 정의하며, 각 상태의 확률 진폭을 2차원 벡터(화살표)인 Amplitude로 표현한다. 측정 시 상태는 가능한 결과 중 하나로 붕괴(Collapse)하며, 측정 결과의 확률은 해당 화살표 길이의 제곱에 비례한다.

### 2. 주요 양자 게이트 및 연산

상태 변환을 위해 다음과 같은 게이트들을 정의하고 사용한다.

- **Single-Qubit Gates**: $X$ (상태 반전), $Z$ (위상 반전), $H$ (중첩 상태 생성), $Y$ (교환 및 회전) 게이트와 이들의 매개변수화된 회전 버전인 $R_X(\theta), R_Z(\theta), R_Y(\theta)$를 사용한다.
- **Conditional Gates**: 제어 큐비트가 1일 때만 타겟 큐비트에 게이트를 적용하는 $cX$와 같은 조건부 게이트를 통해 큐비트 간의 얽힘을 생성한다.

### 3. Phase Estimation 및 QFT

각도(Phase) 정보를 크기(Magnitude) 정보로 변환하기 위해 Quantum Fourier Transform (QFT)을 활용한다. Phase Estimation 알고리즘은 다음과 같은 절차를 따른다.

1. 연산자 $U$를 통해 기하급수적 시퀀스 $[1, \lambda, \lambda^2, \dots, \lambda^{N-1}]$를 생성한다.
2. Inverse QFT를 적용하여 위상 $\theta$에 대응하는 정수 $p$를 측정한다.
이때 측정 결과의 확률 분포는 정규화된 Fejer 커널(Fejer Distribution)을 따른다는 점을 명시하였다.

### 4. Quantum Dictionary Pattern

본 논문의 핵심 방법론으로, 키 레지스터($n$ qubits)와 값 레지스터($m$ qubits)를 구성한다.

- **인코딩 절차**:
  - 숫자를 기본 각도 $\frac{2\pi}{M}$의 배수로 매핑하여 회전 게이트로 표현한다.
  - 인코딩 연산자를 통해 키와 값을 얽히게 만든다.
  - 값 레지스터에 Inverse QFT를 적용하여 진폭 인코딩(Amplitude encoding)을 기저 인코딩(Basis encoding)으로 변환한다.
- **함수 구현**:
  - **Classically-Defined**: 미리 계산된 함수 값을 회전 게이트 시퀀스로 인코딩한다.
  - **Quantumly-Defined**: 회전 게이트의 제어 조건을 활용하여 $f(x)=x^2$ 또는 QUBO 식과 같은 연산을 직접 회로로 구현한다.

### 5. Grover Iteration 및 Quantum Counting

- **Grover Iterate**: 오라클(Oracle)과 확산 연산자(Diffusion Operator)를 통해 '정답' 상태의 진폭을 증폭시킨다.
- **Quantum Counting**: Grover Iterate를 Phase Estimation의 연산자로 사용하여, 오라클이 인식하는 'Good states'의 개수를 계산한다.
  $$ \text{count} = 2^m \cos^2\left(\frac{p\pi}{2^n}\right) $$

## 📊 Results

### 1. 실험 설정 및 지표

본 연구는 특정 데이터셋보다는 알고리즘의 동작 검증에 집중하며, 시뮬레이션 및 양자 회로의 측정 결과(Histogram)를 통해 정성적/정량적 결과를 제시한다.

### 2. 주요 적용 결과

- **QUBO 문제**: $f(x_0, x_1, x_2) = 12x_0 + x_1 - 15x_2 + 3x_0x_1 - 9x_1x_2$와 같은 2차 다항식의 최솟값을 찾는 문제에 적용하였다. Quantum Dictionary를 통해 모든 가능한 함숫값을 인코딩하고, 'Inequality-Based Value Matching'을 통해 특정 값보다 작은 결과가 존재하는지 반복적으로 확인하여 최솟값 $-23$을 찾아내는 과정을 보여주었다.
- **Subset Sum 문제**: $\{1, 0, 2, -1\}$ 집합에서 합이 0이 되는 부분집합의 개수를 찾는 문제에 적용하였다. Value Counting 패턴을 통해 전수 조사 없이 합이 0인 부분집합이 4개임을 정확히 계산하였다.
- **피보나치 수 계산**: 연속된 1이 없는 이진 문자열의 개수가 피보나치 수와 같다는 성질을 이용하여, QUBO 패턴으로 이를 구현하고 $n=3$일 때 5번째 피보나치 수인 5를 도출하였다.

## 🧠 Insights & Discussion

본 논문은 양자 컴퓨팅의 진입 장벽을 낮추기 위해 수학적 엄밀함보다는 설계 패턴과 기하학적 직관을 강조하였다. 특히 Quantum Dictionary라는 프레임워크를 통해 복잡한 비즈니스 제약 조건을 양자 상태로 인코딩하는 표준적인 방법을 제시했다는 점이 강점이다.

다만, 논문 내에서도 언급되었듯이 효율적인 오라클(Oracle)을 설계하는 것은 여전히 "예술과 과학의 영역"이며, 문제에 따라 오라클의 복잡도가 급격히 증가할 수 있다는 한계가 있다. 또한, 실제 하드웨어에서의 노이즈 문제나 오류 정정(Error Correction)에 대한 논의는 생략되어 있어, 제안된 패턴들이 실제 NISQ(Noisy Intermediate-Scale Quantum) 장치에서 어느 정도의 정확도로 동작할지에 대해서는 추가적인 검증이 필요하다.

비판적 관점에서 볼 때, 본 논문은 알고리즘의 새로운 이론적 돌파구보다는 기존 알고리즘들을 어떻게 '조립'하여 사용할 것인가에 대한 가이드라인에 가깝다. 하지만 이는 실무적 관점에서 매우 중요한 기여이며, 특히 QUBO와 같은 최적화 문제를 양자 딕셔너리 구조로 풀어내려는 시도는 향후 양자 소프트웨어 공학 발전에 기여할 가능성이 높다.

## 📌 TL;DR

본 논문은 복잡한 수학 대신 기하학적 직관을 기반으로 양자 컴퓨팅을 접근하며, 특히 키-값 쌍을 얽힘 상태로 관리하는 **Quantum Dictionary Pattern**을 제안한다. 이를 통해 QUBO, Subset Sum, 피보나치 수 계산과 같은 NP-hard 문제들을 해결하는 구체적인 구현 경로를 제시하였다. 이 연구는 추상적인 양자 알고리즘을 실제 비즈니스 문제 해결을 위한 소프트웨어 패턴으로 변환했다는 점에서 향후 양자 알고리즘의 실용적 적용에 중요한 역할을 할 것으로 보인다.
