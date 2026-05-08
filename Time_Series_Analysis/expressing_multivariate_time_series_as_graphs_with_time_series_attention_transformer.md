# Expressing Multivariate Time Series as Graphs with Time Series Attention Transformer

William T. Ng, K. Siu, Albert C. Cheung and Michael K. Ng ([n.d.])

## 🧩 Problem to Solve

본 논문은 다변량 시계열 예측(Multivariate Time Series Forecasting)에서 각 변수가 가지는 개별적인 시간적 흐름(intra-series relationship)과 변수들 간의 상호 의존성(inter-series dependency)을 동시에 효율적으로 캡처하는 것을 목표로 한다.

다변량 시계열 데이터에서 변수들은 자신의 과거 값뿐만 아니라 다른 변수와의 잠재적인 관계에 영향을 받는다. 예를 들어, 특정 도시의 기온은 계절적 패턴을 가지는 동시에 인접한 도시의 기온과 유사한 경향을 보일 수 있다. 기존의 단변량 방법론은 이러한 교차 학습(cross-learning)이 불가능하며, 기존의 딥러닝 모델(RNN, LSTM, Transformer 등)은 비선형 패턴은 잘 잡지만 변수 간의 상호 의존성을 명시적으로 고려하지 못한다는 한계가 있다. 따라서 본 연구는 시계열 데이터를 그래프 구조로 표현하여 이러한 복잡한 관계를 학습하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다변량 시계열을 **에지 강화 동적 그래프(edge-enhanced dynamic graph)**로 표현하고, 이를 처리하기 위해 Transformer의 Self-attention 메커니즘을 수정하여 노드, 에지, 그래프 구조 정보를 동시에 통합하는 것이다.

주요 기여 사항은 다음과 같다:

1. **에지 강화 동적 그래프 제안**: 시계열의 시간적 정보와 변수 간 상호 의존성을 동적 그래프 형태로 추상화하여 더 나은 표현 학습 및 피처 엔지니어링을 가능하게 하였다.
2. **수정된 Self-attention 메커니즘**: 노드 특징, 에지 특징, 그리고 그래프 구조를 단일 레이어에서 통합하여 집계하는 방식을 통해 시계열 예측 문제를 그래프 임베딩 문제로 변환하였다.
3. **성능 검증**: 실제 데이터셋과 벤치마크 데이터셋을 통해 최신 SOTA(State-of-the-art) 모델들보다 우수한 예측 성능을 입증하였다.

## 📎 Related Works

기존의 시계열 예측을 위한 GNN(Graph Neural Networks) 연구는 크게 두 가지 방향으로 나뉜다:

- **물리적 그래프 구조 기반**: STGCN이나 DCRNN과 같은 모델은 교통망과 같이 미리 정의된 물리적 그래프 구조를 가정한다. 하지만 이러한 방식은 문제 특성마다 그래프를 새로 정의해야 하며, 도메인 지식이 필수적이라는 한계가 있다.
- **학습 기반 그래프 구조**: MTGNN이나 TEGNN은 입력 데이터로부터 인접 행렬(adjacency matrix)을 적응적으로 학습한다. 그러나 이러한 모델들은 주로 노드 특징(node features)에 의존하며, 에지(edge)가 가질 수 있는 풍부한 정보를 충분히 활용하지 못한다.

본 논문은 이러한 한계를 극복하기 위해 **Super-empirical Mode Decomposition (SMD)**을 사용하여 에지 특징을 추출하고, 시간이 지남에 따라 변화하는 동적 그래프 구조를 도입하여 기존 GNN 기반 접근 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. Super-empirical Mode Decomposition (SMD)

TSAT는 입력 시계열에서 특징을 추출하기 위해 SMD를 사용한다. SMD는 입력 시계열 $x(t)$를 다음과 같이 유한한 수의 고유 모드 함수(Intrinsic Mode Functions, IMFs)와 잔차(residual)의 합으로 분해한다:
$$x(t) = \sum_{i=1}^{K} f_i(t) + R(t)$$
여기서 $f_i(t) = A_i(t) \cos(2\pi\phi_i(t))$이며, $A_i(t)$는 순시 진폭, $\phi'_i(t)$는 순시 주파수를 의미한다. $R(t)$는 시계열의 전체적인 추세(trend)를 나타낸다.

### 2. 동적 그래프(Dynamic Graph) 정의

다변량 시계열을 $G_t = (X_t, E_t, A_t)$라는 트리플로 정의한다:

- **노드 행렬 $X_t \in \mathbb{R}^{N \times L_x}$**: $N$개 변수의 과거 $L_x$ 길의의 관측값들이 노드가 된다.
- **에지 텐서 $E_t \in \mathbb{R}^{N \times N \times K}$**: 변수 $i$와 $j$ 사이의 $k$번째 IMF 간의 상관관계를 계산하여 에지 특징 $e_{ijk}$를 정의한다.
  $$e_{ijk} = \frac{f_{i,k}^T f_{j,k}}{\|f_{i,k}\|^2 \|f_{j,k}\|^2}$$
- **인접 행렬 $A_t \in \mathbb{R}^{N \times N}$**: 두 변수의 잔차(trend) 간의 상관관계 $\rho_{ij}$를 계산하고, 임계값 $c$보다 크면 연결된 것으로 간주한다.
  $$\rho_{ij} = \frac{R_i^T R_j}{\|R_i\|^2 \|R_j\|^2}, \quad a_{ij}^t = \begin{cases} 1 & \text{if } |\rho_{ij}| > c \\ 0 & \text{otherwise} \end{cases}$$

### 3. TSAT 아키텍처

전체 시스템은 **Time Embedding Layer $\rightarrow$ TSAT Blocks $\rightarrow$ Read-out** 순으로 구성된다.

- **Time Embedding Layer**: Transformer는 순서 정보가 없으므로, RNN 레이어를 통해 노드 행렬 $X_t$의 시간적 순서 정보를 인코딩한다: $h(X_t) = \text{RNN}(X_t)$.
- **수정된 Time Series Self-Attention**: 기존의 Self-attention을 수정하여 노드, 에지, 구조 정보를 동시에 통합한다.
  $$A_i = \left( \alpha_0 \sigma\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) + \sum_{k=1}^{K} \alpha_k \sigma(D^{imf}_k) + \alpha_{K+1} A \right) V_i$$
  - 첫 번째 항($\alpha_0$): 노드 내의 시간적 관계(intra-series)를 캡처하는 기존 self-attention이다.
  - 두 번째 항($\alpha_k$): $k$번째 IMF의 공분산 행렬 $D^{imf}_k$를 통해 변수 간 상관관계(inter-series)를 반영한다.
  - 세 번째 항($\alpha_{K+1}$): SMD 잔차 기반의 인접 행렬 $A$를 통해 그래프 구조 정보를 반영한다.
  - $\alpha$ 값들은 학습 가능한 파라미터이다.

- **Read-out**: $M$개의 TSAT 블록을 거친 후, Layer Normalization과 Global Average Pooling을 적용하여 임베딩된 동적 그래프 $\hat{G}_t$를 생성하고, 최종적으로 Fully Connected Layer를 통해 예측값 $Y_t$를 도출한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ETTh1, ETTh2, ETTm1 (전력 변압기 온도), Weather (기상), Electricity (전력 소비량) 총 5개 케이스를 사용하였다.
- **비교 모델**:
  - 통계 기반: ARIMA, DeepAR
  - 딥러닝 기반: Informer, LSTNet
  - GNN 기반: MTGNN, Graph WaveNet
- **평가 지표**: RMSE(Root Mean Square Error)와 MAE(Mean Absolute Error)를 사용하였다.

### 주요 결과

- **정량적 성능**: TSAT는 대부분의 데이터셋과 예측 기간(horizon)에서 기존 모델들보다 우수한 성능을 보였다. 총 46개 케이스 중 30개에서 최고 성능을 기록하였다.
- **GNN 기반 모델과의 비교**: 특히 변수가 많은 대규모 데이터셋(Weather, Electricity)에서 TSAT와 GNN 기반 모델들이 일반 딥러닝 및 통계 모델보다 월등한 성능을 보였으며, 그중에서도 TSAT가 가장 우수하였다.
- **Ablation Study**: 에지 특징과 인접 행렬 정보를 모두 제거했을 때 성능이 가장 크게 하락하였으며, 두 정보가 모두 존재할 때 최적의 성능을 냄으로써 제안한 동적 그래프 표현의 유효성을 입증하였다.
- **시각화 분석**: t-SNE 분석 결과, 원본 데이터에서는 서로 겹쳐 보였던 유사한 시계열들이 TSAT를 거친 후에는 명확히 구분되는 임베딩 공간으로 투영됨을 확인하여, 표현 학습 능력이 향상되었음을 보였다.

## 🧠 Insights & Discussion

본 논문은 다변량 시계열 데이터를 단순히 수치 행렬로 보는 것이 아니라, **물리적/통계적 의미가 담긴 그래프**로 변환하여 접근했다는 점에서 강점이 있다. 특히 SMD를 통해 시계열을 다양한 주파수 성분(IMF)으로 분해하고, 이를 통해 에지 특징을 생성한 점은 단순한 상관계수 계산보다 훨씬 정교한 변수 간 관계를 포착하게 해준다.

다만, 본 논문에서 제시된 한계나 가정 중 주의 깊게 볼 점은 다음과 같다:

- **임계값 $c$의 설정**: 인접 행렬을 생성할 때 사용되는 임계값 $c$가 성능에 영향을 줄 수 있으나, 이에 대한 최적화 방법론이 구체적으로 논의되지 않았다.
- **계산 복잡도**: Transformer 구조에 GNN의 특성을 결합하고 SMD 분해 과정을 추가했으므로, 단순한 RNN/CNN 모델에 비해 연산 비용이 증가했을 가능성이 크다.

결과적으로, TSAT는 그래프 표현 학습을 시계열 예측에 성공적으로 접목하여, 특히 변수 간 복잡한 상호작용이 중요한 대규모 다변량 시계열 데이터셋에서 매우 강력한 도구가 될 수 있음을 보여준다.

## 📌 TL;DR

본 논문은 다변량 시계열의 내부 패턴과 변수 간 관계를 동시에 학습하기 위해, 시계열을 **에지 강화 동적 그래프**로 변환하는 **TSAT (Time Series Attention Transformer)**를 제안한다. SMD를 통해 추출한 IMF 성분으로 에지를 구성하고, 이를 Transformer의 Attention 메커니즘에 통합함으로써 기존 SOTA 모델들을 뛰어넘는 예측 성능을 달성하였다. 이 연구는 복잡한 다변량 시계열 데이터의 표현 학습을 위해 그래프 구조와 Attention 메커니즘을 결합하는 새로운 방향성을 제시한다.
