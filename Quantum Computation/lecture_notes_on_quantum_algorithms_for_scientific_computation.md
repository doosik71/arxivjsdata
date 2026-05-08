# Lecture Notes on Quantum Algorithms for Scientific Computation

Lin Lin (2022)

## 🧩 Problem to Solve

본 논문(강의 노트)은 양자 컴퓨터를 단순한 이론적 탐구 대상이 아닌, 과학 및 공학 분야의 도전적인 계산 문제를 해결하기 위한 실질적인 도구로 활용하는 방법을 다룬다. 기존의 양자 컴퓨팅 교과서들이 복잡도 이론, 물리적 구현, 오류 정정 등 광범위한 주제를 다루는 반면, 정작 과학 계산의 핵심인 선형 시스템 해결, 고유값 문제, 최소제곱법, 미분 방정식 풀이 및 수치 최적화와 같은 구체적인 응용 방법론에 대한 설명은 부족하다는 문제의식에서 출발한다.

따라서 본 연구의 목표는 미래의 결함 허용(Fault-tolerant) 양자 컴퓨터를 사용하여 과학적 계산, 특히 행렬 계산(Matrix Computation)을 수행하기 위한 핵심 양자 알고리즘들과 그 방법론을 체계적으로 정리하고 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 과학 계산을 위한 양자 알고리즘의 파이프라인을 '블록 인코딩(Block Encoding) $\rightarrow$ 큐비타이제이션(Qubitization) $\rightarrow$ 양자 신호 처리(QSP) 및 양자 특이값 변환(QSVT)'으로 이어지는 일관된 흐름으로 구축하여 제시했다는 점이다.

중심적인 설계 아이디어는 비단일 연산자(Non-unitary operator)인 일반 행렬 $A$를 더 큰 단위 행렬(Unitary matrix) $U_A$의 부분 행렬로 내장하는 블록 인코딩 기법을 사용하고, 이를 통해 행렬의 고유값 또는 특이값을 회전 각도로 변환하여 다항식 근사를 수행하는 것이다. 이를 통해 고전적인 행렬 계산의 복잡도를 획 uma지수적으로 낮출 수 있는 방법론적 토대를 제공한다.

## 📎 Related Works

저자는 Nielsen과 Chuang의 고전적인 교과서를 포함한 기존의 양자 컴퓨팅 자료들이 양자 푸리에 변환(QFT)이나 그로버 검색(Grover's search)과 같은 기초 알고리즘에 치중되어 있어, 이를 어떻게 실제 과학 계산 문제와 연결할지에 대한 가교 역할이 부족하다고 지적한다.

기존의 접근 방식은 주로 Trotter 분할(Trotter splitting)과 같은 해밀토니안 시뮬레이션에 의존했으나, 본 논문은 이를 넘어 LCU(Linear Combination of Unitaries)와 QSVT와 같은 최신 기법들을 소개하며, 단순한 시뮬레이션을 넘어 일반적인 행렬 함수 $f(A)$를 계산하는 더 일반적이고 효율적인 프레임워크를 제시함으로써 차별성을 갖는다.

## 🛠️ Methodology

본 논문은 과학 계산을 위해 다음과 같은 단계적 방법론을 제안한다.

### 1. Block Encoding (블록 인코딩)

비단일 행렬 $A \in \mathbb{C}^{N \times N}$를 직접 양자 회로로 구현할 수 없으므로, $A$를 더 큰 단위 행렬 $U_A$의 좌상단 블록으로 포함시키는 기법이다.
$$A = (\langle 0^m| \otimes I_n) U_A (|0^m\rangle \otimes I_n)$$
여기서 $\alpha$는 정규화 계수이며, $U_A$는 $(\alpha, m)$-블록 인코딩이라고 한다. 이를 통해 행렬-벡터 곱 $A|b\rangle$을 $U_A|0, b\rangle$의 측정 결과로 얻을 수 있다.

### 2. Qubitization (큐비타이제이션)

블록 인코딩된 행렬 $U_A$와 투영 연산자 $\Pi = |0^m\rangle\langle 0^m| \otimes I_n$를 이용해, 각 고유벡터를 2차원 부분 공간으로 '큐비트화'하는 과정이다.
회전 연산자 $O = U_A Z_\Pi$를 정의하면, $O$의 $k$번 반복 적용은 체비쇼프 다항식 $T_k(A)$의 블록 인코딩이 된다. 즉, 행렬의 고유값 $\lambda$를 $\cos \theta$로 매핑하여 다항식 계산을 회전 연산으로 변환한다.

### 3. Linear Combination of Unitaries (LCU)

여러 단위 행렬의 선형 결합 $T = \sum \alpha_i U_i$를 구현하는 기법이다.

- **Prepare Oracle ($V$):** 계수 $\sqrt{\alpha_i}$를 상태로 준비한다.
- **Select Oracle ($U$):** 제어 큐비트의 값에 따라 적절한 $U_i$를 적용한다.
최종 연산자는 $W = (V^\dagger \otimes I) U (V \otimes I)$ 형태로 구현되며, 이는 $T$의 블록 인코딩을 생성한다.

### 4. Quantum Signal Processing (QSP) & QSVT

Qubitization과 LCU의 복잡성을 개선한 기법으로, 단위 행렬과 위상 회전(Phase rotation)의 교차 적용을 통해 임의의 다항식 $P(A)$를 구현한다.
$$U_\Phi = e^{i\phi_0 Z} \prod_{j=1}^d (U_A e^{i\phi_j Z})$$
위상 각도 $\Phi = (\phi_0, \dots, \phi_d)$를 적절히 선택하면, $U_\Phi$의 좌상단 블록이 우리가 원하는 행렬 함수 $P(A)$가 된다. 이를 일반 행렬로 확장한 것이 QSVT(Quantum Singular Value Transformation)이며, 특이값 $\sigma_i$에 대해 함수 $f(\sigma_i)$를 적용하는 효과를 준다.

## 📊 Results

### 1. 선형 시스템 해결 (HHL 알고리즘)

$Ax = b$ 문제를 해결하기 위해 QPE를 사용하여 고유값 $\lambda_j$를 추출하고, 이를 $1/\lambda_j$로 회전시킨 후 역-QPE를 수행한다.

- **복잡도:** 쿼리 복잡도는 $O(\kappa^2/\epsilon)$이며, 진폭 증폭(Amplitude Amplification)을 사용하면 $O(\kappa^2/\epsilon)$ 수준으로 최적화될 수 있다 ($\kappa$는 조건수).
- **결과:** 고전 알고리즘의 $O(N\kappa \log(1/\epsilon))$ 대비, $N$이 매우 클 때 지수적 속도 향상을 기대할 수 있다.

### 2. 포아송 방정식 (Poisson's Equation)

1차원 포아송 방정식의 이산화 행렬 $A$에 대해 $\kappa(A) = O(N^2)$임을 보였다.

- **분석:** 1차원에서는 쿼리 복잡도가 $O(N^4/\epsilon)$이 되어 이득이 적으나, $d$-차원으로 확장 시 조건수는 $d$에 무관하게 유지되는 반면 고전 알고리즘의 비용은 $N^d$에 비례하므로, 고차원 문제에서 압도적인 이점을 가진다.

### 3. 미분 방정식 풀이

Forward Euler 방법을 사용하여 상미분 방정식(ODE)을 선형 시스템 $Ax=b$ 형태로 변환하여 해결하는 방식을 제시한다. 이때 조건수 $\kappa(A) = O(1/\Delta t)$가 됨을 수학적으로 증명하였다.

## 🧠 Insights & Discussion

본 논문은 양자 알고리즘의 발전 방향이 **'근사적 시뮬레이션 $\rightarrow$ 다항식 근사 $\rightarrow$ 최적 위상 제어'**로 진화하고 있음을 보여준다. 특히 Trotter 분할 방식의 오차는 시스템 크기에 따라 증가하는 반면, QSP/QSVT 방식은 위상 각도 $\Phi$의 최적화 문제로 치환되어 훨씬 효율적인 회로 깊이를 달성할 수 있다.

**강점 및 한계:**

- **강점:** 단순한 알고리즘 소개를 넘어, 블록 인코딩부터 QSVT까지 이어지는 수학적 체계를 완벽하게 구축하여 일반적인 행렬 함수 계산 프레임워크를 제시하였다.
- **한계:** 제안된 알고리즘들은 모두 '결함 허용' 양자 컴퓨터를 가정한다. 현재의 NISQ 장치에서는 회로 깊이와 정밀도 문제로 인해 직접 구현이 어려우며, 특히 $A$의 블록 인코딩 $U_A$를 효율적으로 구현하는 것이 실제 적용의 핵심 병목 구간이 될 것이다.

**비판적 해석:**
HHL 알고리즘의 조건수 $\kappa$에 대한 의존성은 여전히 높다. 비록 $N$에 대해서는 지수적 이득이 있지만, $\kappa$가 매우 큰 'Ill-conditioned' 문제의 경우 양자 이득이 상쇄될 수 있다. 따라서 행렬의 성질에 따른 전처리기(Preconditioner)의 양자 버전 개발이 향후 중요한 연구 과제가 될 것으로 보인다.

## 📌 TL;DR

이 논문은 과학 계산의 핵심인 행렬 연산을 양자 컴퓨터에서 수행하기 위한 종합적인 방법론을 제시한다. **Block Encoding $\rightarrow$ Qubitization $\rightarrow$ QSP/QSVT**로 이어지는 파이프라인을 통해 일반 행렬의 함수 $f(A)$를 효율적으로 계산하는 방법을 정립하였으며, 이를 통해 선형 시스템 해결, 미분 방정식 풀이, 바닥 상태 에너지 추정 등에서 고전 컴퓨터 대비 지수적인 속도 향상을 달성할 수 있는 이론적 토대를 마련하였다. 이는 향후 결함 허용 양자 컴퓨터 시대의 과학적 컴퓨팅 표준 프레임워크로 활용될 가능성이 매우 높다.
