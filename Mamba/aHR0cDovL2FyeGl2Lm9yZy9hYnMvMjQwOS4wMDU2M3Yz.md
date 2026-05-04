# SPARSEMAMBA: INTRODUCING CONTROLLABILITY, OBSERVABILITY, AND STABILITY TO STRUCTURAL STATE SPACE MODELS

Emadeldeen Hamdan, Hongyi Pan, Ahmet Enis Cetin (2024)

## 🧩 Problem to Solve

본 연구는 최신 State Space Models(SSMs)인 Mamba 및 Mamba2 아키텍처가 가진 이론적 결여와 그로 인한 효율성 문제를 해결하고자 한다. Transformer 기반 모델들은 긴 시퀀스 처리 시 연산 복잡도가 제곱 단위로 증가하는 문제가 있으며, 이를 해결하기 위해 등장한 Mamba 계열 모델들은 선형 시간 복잡도로 이를 개선하였다.

하지만 기존의 Mamba 모델들은 제어 이론(Control Theory)의 핵심 개념인 가제어성(Controllability)과 가관측성(Observability)에 대한 명시적인 강화가 부족하다. 이로 인해 각 타임스텝에서 $A, B, C, D$ 행렬을 계산하는 과정에서 복잡성과 계산 비용이 증가하는 문제가 발생한다. 또한, Mamba2의 경우 상태 행렬 $A$가 항상 안정적(Stable)이지 않아 모델의 견고성에 한계가 있다. 따라서 본 논문의 목표는 제어 이론의 정준형(Canonical Form)과 안정성 조건을 도입한 Sparse-Mamba(S-Mamba)를 통해 파라미터 수를 줄이고, 학습 시간 단축 및 모델 성능(Perplexity)을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SSM의 파라미터 행렬에 제어 이론의 구조적 제약 조건을 부여하여 희소성(Sparsity)을 확보하고 시스템의 동역학적 특성을 최적화하는 것이다.

1.  **S-Mamba 제품군 제안**: 가제어 정준형(Controllable Canonical Form)을 적용한 **SC-Mamba**, 가관측 정준형(Observable Canonical Form)을 적용한 **SO-Mamba**, 그리고 Mamba2의 안정성을 강화한 **ST-Mamba2**를 제안한다.
2.  **구조적 희소성 도입**: $n \times n$ 크기의 상태 행렬 $A$를 정준형으로 설계하여, 불필요한 파라미터를 제거하고 오직 $n$개의 자유 파라미터만 갖도록 하여 메모리 및 연산 효율을 극대화하였다.
3.  **안정성 강제**: Mamba2의 $A$ 행렬 내 고윳값(Eigenvalue)이 음의 실수 영역에 있도록 강제하여 상태 궤적의 발산을 방지하고 모델의 예측 가능성을 높였다.

## 📎 Related Works

논문에서는 RNN에서 Transformer, 그리고 SSM으로 이어지는 시퀀스 모델링의 발전 과정을 설명한다.

-   **RNN 및 LSTM**: 초기 NLP에서 사용되었으나 기울기 소실/폭주(Vanishing/Exploding Gradients) 문제로 인해 긴 시퀀스 학습에 어려움이 있었다.
-   **Transformers**: Attention 메커니즘을 통해 긴 시퀀스를 효과적으로 처리하지만, 추론 시 전체 시퀀스를 저장해야 하므로 연산 비용이 $O(L^2)$으로 매우 높다.
-   **S4 (Structured State Spaces)**: SSM의 효율성을 높이기 위해 대각 구조(Diagonal structure)를 도입하고 FFT를 이용한 컨볼루션 뷰(Convolution view)를 통해 학습 속도를 개선하였다.
-   **Mamba & Mamba2**: 입력 값에 따라 SSM 파라미터가 변하는 선택적 메커니즘(Selection mechanism)을 도입하여 Transformer에 필적하는 성능과 선형 시간 복잡도를 동시에 달성하였다.

기존 Mamba 계열 모델들은 효율적인 연산에 집중한 나머지, 제어 이론의 기초인 가제어성, 가관측성 및 안정성이라는 수학적 보장을 간과했다는 점이 본 연구의 차별점이다.

## 🛠️ Methodology

### 전체 시스템 구조
S-Mamba는 기본적으로 SSM의 상태 방정식인 다음 두 식을 기반으로 한다.
$$\dot{x}(t) = Ax(t) + Bu(t)$$
$$y(t) = Cx(t) + Du(t)$$
여기서 $x$는 상태 벡터, $u$는 입력, $y$는 출력이며, $A, B, C, D$는 각각 상태, 입력, 출력, 피드스루 행렬이다.

### 1. SC-Mamba (Sparse Controllable Mamba)
가제어 정준형(Controllable Canonical Form, CCF)을 적용하여 시스템이 어떤 초기 상태에서도 유한 시간 내에 원하는 최종 상태로 도달할 수 있도록 보장한다. 행렬 구조는 다음과 같다.

-   **상태 행렬 $A$**: 다음과 같이 하단 행에 특성 다항식의 계수가 위치하는 구조를 갖는다.
$$A = \begin{bmatrix} 0 & 1 & 0 & \cdots & 0 \\ 0 & 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & 1 \\ -a_{n-1} & -a_{n-2} & -a_{n-3} & \cdots & -a_0 \end{bmatrix}$$
-   **입력 행렬 $B$**: 입력이 마지막 상태 변수에만 영향을 주도록 설정한다.
$$B = [0, 0, \dots, 1]^T$$
-   **출력 행렬 $C$**: 상태 변수들의 가중치 합으로 출력을 생성한다.
$$C = [b_{n-1}, b_{n-2}, \dots, b_0]$$
-   **$D$ 행렬**: 기본적으로 $D=0$으로 설정하되, 이후 학습 가능한 파라미터로 둔다.

### 2. SO-Mamba (Sparse Observable Mamba)
가관측 정준형(Observable Canonical Form, OCF)을 적용하여 출력 측정값으로부터 시스템의 내부 상태를 추정할 수 있도록 보장한다. 구조적으로는 SC-Mamba의 전치(Transpose) 형태를 띤다.

-   **상태 행렬 $A$**: SC-Mamba의 $A$ 행렬을 전치한 형태이며, 마지막 열에 계수들이 위치한다.
-   **입력 행렬 $B$**: SC-Mamba의 $C$ 행렬과 유사한 계수 벡터 형태를 갖는다.
-   **출력 행렬 $C$**: 마지막 상태만을 관측하도록 설정한다.
$$C = [0, 0, \dots, 1]$$

### 3. ST-Mamba2 (Stable Mamba2)
Mamba2의 대각 행렬 $A$의 안정성을 강제한다. 시스템이 안정적이려면 $A$의 모든 고윳값이 음의 실수부(Negative real part)를 가져야 한다. 이를 위해 $A$의 대각 성분 $a_i$에 대해 다음과 같은 조건부 수정을 가한다.
$$a_i = \begin{cases} a_i, & \text{if } a_i < 0 \\ -1 \times 10^{-5}, & \text{if } a_i \ge 0 \end{cases}$$
양수이거나 0인 값을 매우 작은 음수로 변환함으로써 상태 궤적이 무한히 발산하는 것을 방지하고 모델의 견고성을 높인다.

## 📊 Results

### 실험 설정
-   **데이터셋**: CodeParrot (1M), OpenWebText (1M), ArXiv (전체), Cosmopedia (100K)
-   **평가 지표**: Perplexity (PPL), 학습 시간(Training Time), 파라미터 수(Number of Parameters)
-   **학습 조건**: 총 7 에포크(Epochs) 학습 후 마지막 에포크 기준 측정

### 주요 결과
1.  **성능 및 효율성 (Mamba vs S-Mamba)**:
    -   **Perplexity**: SC-Mamba가 기존 Mamba 대비 약 5%의 성능 향상을 보였다.
    -   **학습 시간**: SO-Mamba와 SC-Mamba 모두 기존 Mamba 대비 약 3%의 학습 시간 단축 효과를 거두었다. (Table 2 참조)
2.  **안정성 강화 (Mamba2 vs ST-Mamba2)**:
    -   ST-Mamba2는 모든 테스트 데이터셋에서 Mamba2보다 낮은 Perplexity를 기록하며, 안정성 강제가 모델 성능 향상으로 이어짐을 증명하였다. (Table 4 참조)
3.  **파라미터 감소**:
    -   S-Mamba 제품군은 정준형의 희소 구조 덕분에 파라미터 수가 크게 감소하였다. 특히 기존 Mamba 대비 약 100K 수준의 파라미터 감소가 확인되었으며, 이는 Mamba2보다도 적은 수치이다. (Table 3 참조)

## 🧠 Insights & Discussion

본 연구는 딥러닝 모델인 SSM에 고전 제어 이론의 정준형(Canonical Form)을 결합함으로써, 단순히 데이터 기반의 학습에 의존하는 것이 아니라 수학적 구조 제약을 통해 효율성을 얻을 수 있음을 보여주었다.

**강점**:
-   **이론적 근거**: 가제어성, 가관측성, 안정성이라는 제어 이론의 명확한 기준을 아키텍처에 투영하여 모델의 예측 가능성을 높였다.
-   **자원 효율성**: 행렬의 희소성을 강제함으로써 파라미터 수를 줄였음에도 불구하고 오히려 성능(PPL)이 향상되는 결과를 얻었다.

**한계 및 논의사항**:
-   **범용성**: 제안된 정준형 구조가 다양한 도메인의 데이터셋에서 일관되게 작동하는지에 대한 더 광범위한 검증이 필요하다.
-   **하이퍼파라미터**: ST-Mamba2에서 사용된 임계값($-1 \times 10^{-5}$)이 최적의 값인지, 혹은 데이터셋에 따라 동적으로 변해야 하는지에 대한 분석이 부족하다.

결론적으로, 본 논문은 SSM의 행렬 구조를 제어 이론 관점에서 재설계함으로써 '더 가볍지만 더 강력한' 모델을 만들 수 있음을 시사하며, 이는 향후 Mamba3와 같은 차세대 SSM 설계의 핵심 키(Gate key)가 될 가능성이 크다.

## 📌 TL;DR

본 논문은 Mamba 및 Mamba2의 상태 공간 모델에 제어 이론의 **가제어성, 가관측성, 안정성** 개념을 도입한 **S-Mamba** 제품군을 제안한다. 행렬을 정준형(Canonical Form)으로 구성하여 파라미터 수를 대폭 줄이고 희소성을 확보하였으며, 이를 통해 **Perplexity 5% 개선, 학습 시간 3% 단축** 및 모델 안정성 향상을 달성하였다. 이 연구는 SSM 설계 시 수학적 구조 제약이 연산 효율과 모델 성능을 동시에 잡을 수 있는 효과적인 방법임을 입증하였다.