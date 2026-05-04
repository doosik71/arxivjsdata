# From Markov to Laplace: How Mamba In-Context Learns Markov Chains

Marco Bondaschi, Nived Rajaraman, Xiuying Wei, Kannan Ramchandran, Razvan Pascanu, Caglar Gulcehre, Michael Gastpar, Ashok Vardhan Makkuva (2025)

## 🧩 Problem to Solve

본 논문은 최근 Transformer의 대안으로 주목받고 있는 Selective State Space Sequence Models (SSMs), 특히 Mamba(S6)와 Mamba-2의 **In-Context Learning (ICL)** 능력에 대한 근본적인 메커니즘을 규명하고자 한다. Transformer 기반 모델들이 뛰어난 성과를 보였음에도 불구하고 연산 복잡도 문제가 심각하며, Mamba는 이를 해결하며 유사하거나 더 우수한 성능을 보였으나 그 학습 능력이 어떻게 작동하는지에 대한 이론적 이해는 부족한 상태이다.

연구의 핵심 목표는 무작위 마르코프 체인(Random Markov Chains)이라는 통제된 환경을 통해 Mamba가 어떻게 문맥 내에서 정보를 학습하고 예측하는지 체계적으로 분석하는 것이다. 특히, Mamba가 통계적으로 최적인 추정자인 Laplacian smoothing을 구현할 수 있는지, 그리고 이를 가능하게 하는 아키텍처적 핵심 요소가 무엇인지를 밝히는 데 집중한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 Mamba가 문맥 내에서 **Laplacian smoothing estimator**를 효율적으로 학습한다는 사실을 발견하고, 이를 이론적으로 증명한 것이다. 구체적인 핵심 아이디어는 다음과 같다.

- **최적 추정자의 구현**: 단일 레이어의 Mamba만으로도 모든 마르코프 차수(Markov order)에 대해 Bayes 및 minimax 최적인 Laplacian smoothing을 구현할 수 있음을 보였다.
- **Convolution의 결정적 역할**: Mamba의 입력 선택성(Input selectivity) 과정에 포함된 **Convolution** 메커니즘이 최적의 Laplacian smoothing을 표현하는 데 핵심적인 역할을 수행함을 밝혀냈다.
- **이론적 정당성 부여**: 단순화된 모델인 MambaZero를 제안하여, 1차 마르코프 과정에 대해 Mamba가 최적 추정자를 이론적으로 표현할 수 있음을 증명하였다.
- **메모리 하한선 제시**: recurrent 아키텍처가 마르코프 차수 $k$에 대해 Laplacian 추정자를 구현하기 위해서는 은닉 차원(hidden dimension)이 $k$에 대해 지수적으로 증가해야 함($\Omega(2^k)$)을 수학적으로 입증하였다.

## 📎 Related Works

기존의 State Space Models(SSMs)는 선형 시불변(LTI) 시스템을 기반으로 하여 Transformer의 쿼드라틱 복잡도 문제를 해결하려 하였다. 최근의 Mamba와 Mamba-2는 Selective SSM으로서 입력에 따라 파라미터를 동적으로 변경하는 선택성 메커니즘을 도입하여 성능을 비약적으로 향상시켰다.

ICL과 관련하여, Transformer에서는 'Induction Heads'가 카운팅 추정자를 구현하여 ICL을 수행한다는 점이 알려져 있으며, 이를 위해서는 최소 2개의 레이어가 필요하다는 연구 결과가 있다. 반면 SSM의 ICL 능력에 대해서는 기존 연구들이 주로 경험적인 분석에 그쳤으며, Mamba의 능력이 Transformer에 비해 우수하다는 주장과 열세라는 주장이 대립하고 있었다. 본 논문은 이러한 경험적 분석을 넘어, 최적 통계 추정자와 Mamba 아키텍처 간의 첫 번째 공식적인 연결 고리를 이론적으로 제시함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 실험 설정 및 데이터 (Random Markov Chains)

모델의 ICL 능력을 측정하기 위해 무작위 마르코프 체인 데이터를 사용한다.

- **데이터 생성**: 각 상태 전이 확률은 디리클레 분포 $\text{Dir}(\beta \cdot \mathbf{1})$에서 독립적으로 샘플링된다. 모든 시퀀스는 서로 다른 전이 행렬 $P$를 가지므로, 모델은 추론 시 반드시 문맥 내에서 다음 토큰을 예측해야 한다.
- **학습 목표**: 다음 토큰 예측(Next-token prediction)을 위해 cross-entropy 손실 함수를 최소화하도록 학습한다.

### 2. Mamba 아키텍처

본 논문은 Mamba-2를 기준으로 분석하며, 핵심 연산 과정은 다음과 같다.

- **상태 업데이트**:
  $$H_t = a_t H_{t-1} + e_x^t b_t^\top$$
  여기서 $H_t$는 상태 행렬이며, $a_t \in (0, 1)$는 입력에 따라 결정되는 감쇠 계수(decay factor)이다.
- **출력 생성**:
  $$y_t = H_t c_t, \quad z_t = y_t \odot \text{ReLU}(W_z x_t), \quad o_t = W_o z_t$$
- **입력 선택성 (Input Selectivity)**: $a_t, e_x^t, b_t, c_t$는 모두 입력 $x_t$와 그 주변 문맥에 대한 **Convolution** 연산을 통해 계산된다.

### 3. 최적 추정자: Laplacian Smoothing

Bayesian 관점에서 최적인 예측치는 **Laplacian smoothing**(또는 add-$\beta$ estimator)으로 정의된다.
$$\text{P}^{(k)}_\beta(x_{t+1}=1 | x_1^t) = \frac{n_1 + \beta}{n + 2\beta}$$
여기서 $n$은 현재의 $k$차 문맥이 나타난 횟수이고, $n_1$은 해당 문맥 뒤에 토큰 1이 나타난 횟수이다. $\beta$는 보정값으로, 보지 못한 이벤트에 0의 확률을 할당하는 것을 방지한다.

### 4. 이론적 분석 모델: MambaZero

분석을 단순화하기 위해 ReLU와 Gating을 제거하고 Embedding, Convolution, Mamba Block, Linear Layer만 남긴 **MambaZero** 모델을 제안한다. 이를 통해 Convolution이 어떻게 카운팅 메커니즘을 구현하는지 증명한다.

## 📊 Results

### 1. Mamba vs Transformer ICL 성능

- **정량적 결과**: 단일 레이어 Mamba는 최적의 Laplacian smoothing 추정자와 매우 유사한 예측 확률을 생성하였다.
- **비교**: 단일 레이어 Transformer는 이 과제를 수행하는 데 실패하였으며, 2개 레이어 Transformer는 성공하였으나 Mamba보다 정밀도가 떨어지고 변동성이 컸다.

### 2. Convolution의 중요성 (Ablation Study)

- **Convolution 제거 시**: Mamba에서 Convolution을 제거하면 성능이 급격히 저하되어 과제 수행에 실패하였다.
- **MambaZero**: 단순화된 MambaZero 모델조차 Convolution만 있다면 풀 모델과 유사한 성능을 보였다.
- **윈도우 크기 ($w$)**: 마르코프 차수가 $k$일 때, Convolution 윈도우 크기 $w \ge k+1$이어야만 학습이 가능함을 확인하였다.

### 3. Switching Markov Model (선택성 검증)

문맥 중간에 마르코프 체인이 바뀌는 'Switch token'이 존재하는 환경에서 실험하였다.

- **결과**: Mamba는 Switch token이 등장할 때 $a_t \approx 0$으로 설정하여 과거의 카운트 정보를 완전히 삭제하고, 새로운 체인에 대해 다시 카운팅을 시작하는 선택적 메커니즘을 정확히 구현하였다.

### 4. 실제 자연어 데이터 적용 (WikiText-103)

- **결과**: Mamba-2 모델에서 Convolution 유무에 따른 Perplexity를 측정한 결과, Convolution이 있을 때 $27.55$, 없을 때 $30.68$로 성능 향상이 뚜렷하였다. 반면 Transformer는 Convolution 추가로 인한 성능 이득이 미미하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 함의

본 연구는 Mamba의 Convolution 연산이 단순한 전처리가 아니라, **문맥 내 통계적 카운팅(Counting)**을 수행하는 핵심 엔진임을 밝혔다는 점에서 큰 가치가 있다. 특히 단일 레이어 SSM이 다층 Transformer보다 특정 ICL 과제(마르코프 체인 예측)에서 더 효율적일 수 있음을 시사한다.

### 한계 및 비판적 해석

- **차수 확장성**: 1차 마르코프 과정에 대해서는 수학적 증명을 완료하였으나, $k > 1$인 경우에 대해서는 추측(Conjecture) 단계에 머물러 있다. 이는 전이 횟수 간의 상관관계 분석이 복잡하기 때문이며, 향후 엄밀한 증명이 필요하다.
- **차원의 제약**: Theorem 2에서 제시한 $\Omega(2^k)$ 하한선은 마르코프 차수가 높아질수록 모델의 hidden dimension이 지수적으로 커져야 함을 의미한다. 이는 Mamba가 효율적이라고는 하나, 매우 높은 차수의 마르코프 의존성을 가진 데이터를 처리할 때는 메모리 효율성이 급격히 떨어질 수 있음을 암시한다.

## 📌 TL;DR

본 논문은 Mamba가 문맥 내에서 최적의 통계 추정자인 Laplacian smoothing을 구현함을 발견하고, 그 핵심 동력이 **Convolution** 메커니즘에 있음을 이론적·경험적으로 증명하였다. 이는 Mamba가 단순한 RNN의 대체제가 아니라, 특정 형태의 문맥 학습을 수행하는 매우 효율적인 통계적 추정기로 동작할 수 있음을 보여주며, 향후 SSM 아키텍처 설계 및 표현력 분석에 중요한 이론적 토대를 제공한다.
