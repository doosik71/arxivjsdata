# Poly-Mamba: A SSM Improved from Multivariate Time Series

Haixiang Wu (2024)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 예측(Multivariate Time Series Forecasting, MTSF) 작업에서 기존의 State Space Model(SSM)과 Transformer 기반 방법론들이 해결하지 못한 **시간에 따른 채널 간 의존성 변화(Channel Dependency variations with Time, CDT)**를 명시적으로 모델링하는 문제를 해결하고자 한다.

다변량 시계열 데이터에서 각 채널(변수)들은 서로 복잡한 상호작용을 하며, 이러한 관계는 고정되어 있지 않고 시간에 따라 계속해서 변화한다. 기존 연구들은 시간적 토큰 간의 의존성이나 채널 토큰 간의 정적인 의존성만을 학습하는 경향이 있었으며, CDT라는 특수한 복잡성을 직접적으로 묘사하지 못했다. 이를 직접 모델링하는 것은 계산 복잡도를 급격히 증가시키고 일반화 성능을 떨어뜨릴 위험이 있으므로, 본 논문은 SSM의 수학적 근간인 직교 함수 기저(orthogonal function basis)의 근사를 통해 효율적으로 CDT를 포착하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SSM의 기본 원리인 '연속적으로 업데이트되는 함수를 직교 함수 기저의 선형 결합으로 근사하는 것'을 다변량 공간으로 확장하는 것이다.

중심적인 설계 아이디어는 단변량 직교 함수 기저 공간을 변수 간의 혼합 항(variable mixing terms)이 포함된 **다변량 직교 함수 공간(multivariate orthogonal function space)**으로 확장하고, 이 공간으로의 투영(projection)을 통해 가중치 계수로 CDT를 명시적으로 기술하는 것이다. 이를 구현하기 위해 저자는 다음과 같은 세 가지 핵심 구성 요소를 제안한다.

1. **MOPA (Multivariate Orthogonal Polynomial Approximation):** 다변량 다항식 기저 공간으로의 매핑을 단순화하여 복잡한 채널 간 의존성을 효율적으로 모델링한다.
2. **LCM (Linear Channel Mixing):** 채널 간의 단순한 선형 관계를 포착하기 위한 매핑 연산을 제공한다.
3. **Order Combining:** 게이팅(Gating) 메커니즘을 통해 각 채널의 저차원 추세 정보는 유지하면서, 고차원 성분에 대해서는 LCM과 MOPA 중 적절한 CDT 패턴을 적응적으로 선택하여 결합한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 배경으로 한다.

- **SSM 및 Mamba:** S4와 Mamba는 Transformer의 이차 복잡도 문제를 해결하고 선형 복잡도로 긴 의존성을 모델링할 수 있는 구조이다. 특히 Mamba는 선택적 메커니즘(selective mechanism)을 통해 정보의 중요도를 판단한다.
- **MTSF 모델:** PatchTST는 시계열을 패치 단위로 나누어 토큰화하며, iTransformer는 채널 간 의존성을 모델링하기 위해 구조를 뒤집는 방식을 사용한다.
- **기존 SSM 기반 MTSF의 한계:** S-Mamba와 같은 기존 SSM 기반 시계열 모델들은 주로 Transformer의 모델링 패러다임을 따라 시간적 또는 채널적 토큰 간의 의존성만을 학습했을 뿐, 시간에 따라 변하는 채널 간의 상호작용인 CDT를 명시적으로 다루지 않았다.

## 🛠️ Methodology

Poly-Mamba는 시계열 패치를 토큰으로 입력받아 다변량 은닉 상태(multivariate hidden states)를 생성하며, 이 은닉 상태에 MOPA, LCM, Order Combining을 적용하여 채널 간 상관관계가 강화된 새로운 상태를 생성한다.

### 1. MOPA (Multivariate Orthogonal Polynomial Approximation)

이론적으로는 단변량 직교 다항식을 다음과 같이 다변량 직교 다항식으로 확장한다.
$$P_{n_1, n_2, ..., n_C}(x_1, x_2, ..., x_C) = P_{n_1}(x_1)P_{n_2}(x_2)...P_{n_C}(x_C)$$
여기서 $x_1x_2$와 같은 혼합 항(mixing term)들이 채널 간의 상호의존성을 나타낸다. 하지만 이를 그대로 구현하면 차원이 기하급수적으로 증가하므로, 저자는 이를 다음과 같이 단순화된 연산으로 대체한다.
$$\text{MOPA}(x(t)) = M \cdot x(t)$$
여기서 $M$은 $C \times N$ 크기의 변환 파라미터 행렬이며, 은닉 상태(계수 행렬)와 Hadamard 곱(element-wise product)을 수행하여 복잡한 CDT 패턴을 효율적으로 모델링한다.

### 2. LCM (Linear Channel Mixing)

모든 채널 관계가 복잡한 고차원 관계는 아니며, 단순한 선형 관계인 경우가 많다. 이를 위해 학습 가능한 파라미터 행렬 $L$ (크기 $C \times C$)을 사용하여 다음과 같이 선형 매핑을 수행한다.
$$\text{LCM}(x(t)) = L x(t)$$
이를 통해 각 채널은 다른 채널로부터 선형적으로 결합된 정보를 전달받는다.

### 3. Order Combining

LCM과 MOPA의 출력을 적응적으로 결합하기 위해 게이팅 메커니즘을 사용한다.

- **게이팅 연산:** $\text{G}_{\{2,...,N\}} = \text{softmax}(P_L \text{LCM}_{\{2,...,N\}}, P_M \text{MOPA}_{\{2,...,N\}})$
- **결합 방식:** 저차원(0, 1차) 성분은 채널 자체의 추세 정보가 중요하므로 LCM 결과만을 유지하고, 고차원($2 \sim N$차) 성분에 대해서만 게이팅을 통해 선택된 값을 결합한다.
$$\text{OC} = \text{Concat}(\text{LCM}_{\{0,1\}}, \text{G}_{\{2,...,N\}})$$

### 전체 추론 절차 (Algorithm 1 요약)

1. 입력 데이터 $X$를 패치화하여 토큰으로 처리한다.
2. SSM의 recurrence 식 $\mathbf{h}_t = \mathbf{A}\mathbf{h}_{t-1} + \mathbf{B}\mathbf{X}_t$를 통해 기본 은닉 상태를 계산한다.
3. 계산된 $\mathbf{h}_t$에 대해 LCM과 MOPA를 각각 적용한다.
4. Order Combining을 통해 저차원 LCM 성분과 게이팅된 고차원 성분을 결합하여 $\mathbf{h}'_t$를 생성한다.
5. 최종적으로 $\mathbf{y}_t = \mathbf{C}\mathbf{h}'_t$를 통해 예측값을 출력한다.

## 📊 Results

### 실험 설정

- **데이터셋:** ETT (4개), ECL, Exchange, Traffic, Weather, Solar-Energy 등 6개의 실제 데이터셋을 사용하였다.
- **비교 대상:** S-Mamba(SSM 기반), iTransformer, PatchTST, Crossformer(Transformer 기반), DLinear, TiDE(Linear 기반), TimesNet(CNN 기반).
- **평가 지표:** MSE(Mean Squared Error)와 MAE(Mean Absolute Error).
- **입력/출력 길이:** 입력 길이 $L=96$, 예측 길이 $T \in \{96, 192, 336, 720\}$.

### 주요 결과

- **예측 성능:** 표 1에서 확인 가능하듯, Poly-Mamba는 대부분의 데이터셋에서 SOTA 성능을 기록하였다. 특히 Weather 데이터셋과 같이 채널 간 실시간 상관관계가 강한 시나리오에서 우수한 성능을 보였다.
- **장기 예측 안정성:** 예측 길이 $T$가 $336, 720$으로 길어질 때, Poly-Mamba는 다른 모델들에 비해 더 안정적인 예측 결과를 나타냈다. 이는 SSM의 역사 정보 압축 능력이 효과적으로 작용했음을 시사한다.
- **효율성:** 표 3에 따르면, Mamba 대비 파라미터 행렬 $L, M$ 및 게이팅 가중치가 추가되었음에도 불구하고 메모리 사용량과 추론 속도(ms/iter)의 증가폭이 매우 작아, 실용적인 효율성을 유지하고 있다.

## 🧠 Insights & Discussion

### 강점 및 해석

- **명시적 CDT 모델링:** 기존 모델들이 암묵적으로 채널 관계를 학습했다면, Poly-Mamba는 다변량 직교 다항식의 개념을 도입하여 채널 간 상호작용을 수학적으로 정의하고 이를 단순화된 행렬 연산으로 구현함으로써 성능과 효율성을 동시에 잡았다.
- **적응적 선택:** Order Combining을 통해 데이터의 특성에 따라 선형 관계(LCM)와 복잡한 관계(MOPA)를 선택적으로 사용할 수 있게 하여, 모델의 범용성을 높였다.

### 한계 및 비판적 해석

- **구현의 단순화:** 저자는 MOPA가 이론적인 다변량 투영 과정을 완전히 구현한 것이 아니라 '단순화된 변형(simplified variation)'이라고 명시하였다. 따라서 이론적 근거와 실제 구현된 Hadamard 곱 연산 사이의 수학적 간극이 완벽하게 메워졌는지는 추가적인 검토가 필요하다.
- **가정:** 본 모델은 SSM의 기본 구조가 시계열의 전역적 특성을 잘 포착한다는 가정 하에 작동한다. 하지만 매우 불규칙하거나 비정상성(non-stationary)이 극심한 데이터에서의 강건성에 대한 분석은 다소 부족하다.

## 📌 TL;DR

Poly-Mamba는 다변량 시계열의 **시간에 따른 채널 간 의존성 변화(CDT)**를 포착하기 위해, SSM의 직교 함수 기저를 다변량 공간으로 확장한 모델이다. **MOPA**(복잡한 관계), **LCM**(선형 관계), **Order Combining**(적응적 결합)의 세 가지 모듈을 통해 채널 간 상관관계를 효율적으로 학습하며, 특히 채널 수가 많고 상관관계가 복잡한 데이터셋에서 기존 SOTA 모델들을 능가하는 성능과 안정성을 보여준다. 향후 대규모 시계열 사전학습(pre-training)에 적용될 가능성이 크다.
