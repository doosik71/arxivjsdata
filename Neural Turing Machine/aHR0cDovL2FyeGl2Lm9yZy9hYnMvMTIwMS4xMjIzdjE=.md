# Turing Machines and Understanding Computational Complexity

Paul M.B. Vitányi (2012)

## 🧩 Problem to Solve

본 논문은 계산 가능성(computability)이라는 직관적인 개념을 어떻게 수학적으로 정형화하고, 이를 통해 계산 복잡도(computational complexity)의 기초를 어떻게 이해할 수 있는지를 다룬다. 구체적으로는 앨런 튜링(Alan Turing)이 제안한 Turing Machine(튜링 기계)의 작동 원리를 상세히 기술하고, 이것이 현대 컴퓨터 과학의 이론적 토대가 되는 과정을 설명하는 것을 목표로 한다. 특히, 모든 계산 가능한 함수가 튜링 기계로 구현될 수 있다는 Turing's thesis와 Church-Turing thesis를 통해, 무엇이 계산 가능하고 무엇이 계산 불가능한지를 정의하는 문제에 집중한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 튜링 기계라는 추상적 장치를 통해 계산의 한계를 명확히 규정하고, 이를 현대적인 복잡도 클래스(P, NP, PSPACE 등)의 정의로 연결했다는 점이다. 특히 Universal Turing Machine(범용 튜링 기계)의 개념을 통해 프로그램과 데이터의 구분이 모호해지는 현대 컴퓨터의 소프트웨어적 구조를 이론적으로 뒷받침한다. 또한, 정지 문제(Halting Problem)의 불가능성을 증명함으로써 수학적 논리 체계 내에서 결정 불가능한(undecidable) 영역이 존재함을 명시적으로 보여준다.

## 📎 Related Works

논문은 앨런 튜링의 1936년 연구를 기본 바탕으로 하며, Alonzo Church의 유효 계산 함수(effectively calculable function) 개념과 S. C. Kleene의 Computability thesis를 함께 언급한다. 기존의 직관적인 '효과적인 절차(effective procedure)'에 대한 정의들이 서로 상이했으나, 결과적으로 튜링 기계, 람다 계산법(lambda calculus) 등이 모두 동일한 계산 가능 클래스를 정의한다는 점을 통해 Church-Turing thesis의 타당성을 설명한다. 또한, Kurt Gödel의 불완전성 정리(Incompleteness Theorem)를 언급하며, 튜링의 정지 문제 분석이 괴델의 논리적 한계 증명과 궤를 같이한다는 점을 서술한다.

## 🛠️ Methodology

### 1. Turing Machine의 정형적 정의

튜링 기계는 유한한 상태 제어 장치(finite control), 선형적으로 배열된 셀들의 집합인 테이프(tape), 그리고 테이프의 셀을 읽고 쓰는 헤드(head)로 구성된다.

- **구성 요소**:
  - 상태 집합 $Q$와 테이프 알파벳 $A = \{0, 1, B\}$ (여기서 $B$는 Blank)가 존재한다.
  - 시간은 $0, 1, 2, \dots$와 같이 이산적으로 흐른다.
- **동작 규칙**: 튜링 기계의 동작은 다음과 같은 4-튜플(quadruple) 형태의 규칙 집합으로 정의된다.
    $$(p, s, a, q)$$
    여기서 $p$는 현재 상태, $s$는 현재 스캔 중인 기호, $a$는 수행할 작업(기호 쓰기 또는 헤드 이동 $L, R$), $q$는 작업 후의 다음 상태를 의미한다.
- **결정론적 동작**: 두 규칙의 앞선 두 요소$(p, s)$가 동일하지 않다면 이 기계는 결정론적(deterministic)으로 동작하며, 규칙이 없는 상태에 도달하면 기계는 정지(halt)한다.

### 2. 계산 가능 함수 (Computable Functions)

튜링 기계는 입력값에 대해 정지했을 때 테이프에 남은 값을 출력값으로 내놓음으로써 부분 함수(partial function)를 정의한다. 모든 입력에 대해 정지하는 경우 이를 전계산 가능 함수(total computable function)라고 하며, 결과값이 $\{0, 1\}$인 경우 이를 서술자(predicate)라고 한다.

### 3. 범용 튜링 기계 (Universal Turing Machine, UTM)

모든 튜링 기계 $T$는 그 규칙 집합을 이진 문자열로 인코딩하여 자연수(Gödel number) $n(T)$로 매핑할 수 있다. 범용 튜링 기계 $U$는 특정 튜링 기계의 인코딩 $E(T)$와 입력값 $p$를 동시에 입력받아 $T$의 동작을 그대로 모방하는 기계이다.
$$U(E(T)p) = T(p)$$
이는 현대의 CPU가 프로그램(인코딩된 $T$)을 읽어 실행하는 원리와 동일하다.

### 4. 계산 복잡도 (Computational Complexity)

계산의 효율성을 측정하기 위해 시간 복잡도 $t(n)$과 공간 복잡도 $s(n)$을 정의한다.

- **DTIME / NTIME**: 결정론적/비결정론적 튜링 기계가 시간 $O(t(n))$ 내에 언어를 수용하는 집합이다.
- **복잡도 클래스의 정의**:
  - $P = \bigcup_{c} \text{DTIME}[n^c]$
  - $NP = \bigcup_{c} \text{NTIME}[n^c]$
  - $PSPACE = \bigcup_{c} \text{DSPACE}[n^c]$
- 이들 간에는 $P \subseteq NP \subseteq PSPACE$의 관계가 성립한다.

## 📊 Results

본 논문은 실험적인 수치를 제시하는 대신, 계산 이론의 핵심적인 정리를 결과로 제시한다.

- **정지 문제의 불가능성 (Theorem 1)**: 정지 집합 $K^0 = \{ \langle x, y \rangle : \phi_x(y) < \infty \}$는 계산 가능하지 않다. 즉, 임의의 프로그램이 주어진 입력에 대해 정지할지 영원히 실행될지를 판별할 수 있는 일반적인 알고리즘은 존재하지 않는다.
- **논리 체계의 불완전성 (Theorem 2)**: 페아노 산술(Peano arithmetic)을 확장하고 건전한(sound) 모든 공리화 가능한 이론 $T$에 대해, 참이지만 증명 불가능한 문장 $\text{"}n \notin K^0\text{"}$이 존재한다.
- **복잡도 관계**: Savitch의 정리에 의해 $\text{NSPACE}[s(n)] = \text{DSPACE}[s(n)^2]$임이 밝혀졌으며, 이를 통해 비결정론적 공간 복잡도와 결정론적 공간 복잡도의 관계가 정립되었다.

## 🧠 Insights & Discussion

본 논문은 튜링 기계라는 단순한 모델이 어떻게 현대 컴퓨터의 아키텍처와 알고리즘 분석의 표준이 되었는지를 잘 보여준다. 특히 인상적인 점은, 물리적인 하드웨어의 발전과 상관없이 '계산 가능성'이라는 본질적인 한계는 수학적으로 이미 정의되어 있었다는 사실이다.

비판적 관점에서 본다면, 본 논문은 튜링 기계의 이론적 정립에 집중하고 있어 실제 현대 컴퓨팅에서 발생하는 병렬 처리나 확률적 계산의 효율성 문제를 깊게 다루지는 않는다. 하지만 마지막 섹션에서 Feynman의 양자 컴퓨터 제안과 Deutsch의 Quantum Turing Machine을 언급하며, 고전적 튜링 기계의 한계를 넘어서는 새로운 계산 모델로의 확장 가능성을 제시하고 있다. 이는 계산 이론이 정적인 상태에 머물지 않고 물리적 실체(양자 역학 등)와 결합하여 계속 진화하고 있음을 시사한다.

## 📌 TL;DR

본 논문은 튜링 기계(Turing Machine)의 정형적 정의부터 범용 튜링 기계(UTM), 정지 문제의 불가능성, 그리고 현대의 복잡도 클래스(P, NP, PSPACE)에 이르는 계산 이론의 핵심 흐름을 체계적으로 정리한 보고서이다. 튜링 기계는 단순한 가상 장치를 넘어 '계산 가능성'의 표준을 제시하였으며, 이는 향후 양자 컴퓨팅과 같은 차세대 계산 모델을 분석하는 이론적 기반이 된다.
