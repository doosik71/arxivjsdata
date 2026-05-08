# Entropy Causal Graphs for Multivariate Time Series Anomaly Detection

Falih Gozi Febrinanto, et al. (2025)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series) 데이터에서 이상치 탐지(Anomaly Detection) 성능을 향상시키기 위해 변수 간의 내재적인 인과 관계를 모델링하는 문제를 다룬다.

기존의 다변량 시계열 이상치 탐지 프레임워크들은 대부분 변수 간의 단순한 상관관계나 독립적인 패턴에 의존하며, 변수들 사이의 실제적인 인과 관계(Causal Relationship)를 고려하지 않는다. 하지만 복잡한 시스템(제조, 에너지, 교통 등)에서 발생하는 데이터는 한 센서의 변화가 다른 센서에 영향을 주는 인과적 특성을 가지며, 이를 무시할 경우 이상치 탐지 성능이 저하된다. 따라서 본 논문의 목표는 전이 엔트로피(Transfer Entropy)를 통해 인과 그래프를 구축하고, 이를 그래프 신경망(GNN)과 결합하여 보다 정밀한 이상치 탐지를 수행하는 CGAD 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 아이디어는 단순한 상관관계를 넘어선 **인과적 전이 엔트로피(Transfer Entropy) 기반의 동적 그래프 구조를 활용**하여 다변량 시계열의 공간적-시간적 패턴을 동시에 학습하는 것이다.

1. **인과 그래프 생성**: 정보 이론 기반의 Transfer Entropy를 사용하여 변수 간의 인과 관계를 정량화하고, 이를 통해 가중치가 부여된 인과 그래프(Causal Graph)를 자동으로 구축한다.
2. **Weighted GNN 예측 모델**: 구축된 인과 그래프를 입력으로 하는 가중치 기반 GCN(Graph Convolutional Network)과 시간적 패턴을 추출하는 Causal Convolution(인과 합성곱)을 결합하여 다음 시점의 값을 예측하는 모델을 설계하였다.
3. **강건한 이상치 점수 산출**: 예측 오차를 단순 계산하는 대신, 중앙값 절대 편차(Median Absolute Deviation, MAD) 기반의 정규화를 적용한 수정된 Z-score 방식을 도입하여 이상치 판단의 강건성(Robustness)을 높였다.

## 📎 Related Works

### 기존 연구 및 한계점

1. **재구성 기반 방식(Reconstruction-based)**: OmniAnomaly, USAD 등은 정상 데이터를 재구성하도록 학습하여 재구성 오차를 이용한다. 하지만 시간적 패턴 학습에 한계가 있을 수 있다.
2. **예측 기반 방식(Forecasting-based)**: GDN, GTA 등은 미래 값을 예측하고 실제 값과의 차이를 이용한다. 최근 GNN을 도입하여 변수 간 관계를 모델링하려는 시도가 늘고 있다.
3. **그래프 생성 기술의 한계**:
    - **Gumbel-softmax 기반**: 학습 과정에서 이산적인 인접 행렬을 생성하지만, 도메인 지식을 충분히 활용하지 못하고 과적합(Overfitting) 위험이 있다.
    - **Cosine Similarity/Top-K 기반**: 유사도 기반으로 상위 $k$개의 관계만 선택하므로, 모든 노드가 동일한 차수를 가지게 되어 유연성이 떨어지며 중요한 소수 관계를 놓칠 수 있다.
    - **완전 연결 그래프(Fully Connected)**: 모든 관계를 포함하므로 불필요한 정보와 노이즈가 포함된다.

### 차별점

CGAD는 단순 유사도가 아닌 **인과성(Causality)**에 집중한다. 특히 기존의 인과 기반 모델(CauGNN 등)이 정보 흐름의 지배적 방향만을 고려하여 단방향 관계만 추출했던 것과 달리, CGAD는 양방향 인과성을 모두 보존하여 그래프의 유연성을 확보하였다.

## 🛠️ Methodology

### 1. Causal Graph Generation (인과 그래프 생성)

변수 $Y$가 변수 $X$에 미치는 영향력을 측정하기 위해 Transfer Entropy(TE)를 사용한다. 이는 $X$의 과거 정보뿐만 아니라 $Y$의 과거 정보를 함께 고려했을 때 $X$의 미래 값을 더 잘 예측할 수 있는지를 측정하는 정보 이론적 접근법이다.

- **정보 엔트로피(Information Entropy)**:
  $$H(X) = -\sum_{i \in X} p(i) \log_2 p(i)$$
- **전이 엔트로피(Transfer Entropy)**:
  $$TE_{Y \rightarrow X} = H(X_t | X_{t-1}) - H(X_t | X_{t-1}, Y_{t-1})$$
  여기서 $X_t$는 시간 $t$에서의 값이며, $H(X_t | X_{t-1})$는 조건부 엔트로피이다.

최종적으로 인접 행렬 $\mathcal{A}$의 요소 $\mathcal{A}_{ij}$는 다음과 같이 결정된다:
$$\mathcal{A}_{ij} = \begin{cases} TE_{x_j \rightarrow x_i} & \text{if } TE_{x_j \rightarrow x_i} > c \\ 0 & \text{otherwise} \end{cases}$$
여기서 $c$는 약한 관계를 제거하기 위한 제어 상수이다. 계산 효율성을 위해 Kraskov의 k-최근접 이웃(k-NN) 전략을 사용하며, 긴 시계열 데이터를 작은 청크(chunk)로 나누어 샘플링하여 평균 인접 행렬을 구하는 방식을 취한다.

### 2. Weighted GNN Forecasting (가중치 GNN 예측)

모델은 공간적 분석을 위한 GCN 모듈과 시간적 분석을 위한 TCN 모듈로 구성된다.

- **Weighted GCN**:
  인과 그래프 $\mathcal{A}$를 사용하여 노드 간의 공간적 의존성을 학습한다.
  $$\mathcal{R}^{(l+1)} = \sigma(\hat{\mathcal{D}}^{-1/2} \hat{\mathcal{A}} \hat{\mathcal{D}}^{-1/2} \mathcal{R}^{(l)} \Theta^{(l)})$$
  $\hat{\mathcal{A}}$는 셀프 루프가 추가된 정규화된 인접 행렬이며, $\hat{\mathcal{D}}$는 차수 행렬(Degree Matrix)이다.

- **Temporal Convolution Module**:
  Causal Convolution을 사용하여 시간적 특징을 추출한다. 특히 다양한 시간 범위의 패턴을 잡기 위해 **Dilated Inception Layer**를 도입하여 $1\times2, 1\times3, 1\times5, 1\times6$ 크기의 필터를 병렬로 사용하고 그 결과를 결합(Concatenate)한다. 이후 $\tanh$와 $\sigma$를 이용한 Gating Mechanism을 통해 정보의 흐름을 제어한다.

- **학습 목표**: 예측값 $\hat{x}_{:,t}$와 실제값 $x_{:,t}$ 사이의 평균 제곱 오차(MSE)를 최소화하는 것을 목표로 한다.
  $$loss_{MSE} = \frac{1}{T-d} \sum_{t=d+1}^{T} \|\hat{x}_{:,t} - x_{:,t}\|_2^2$$

### 3. Median Deviation Scoring (중앙값 편차 점수 산출)

예측 오차를 기반으로 이상치 점수를 계산하며, 이상치에 민감한 평균 대신 중앙값을 사용하여 강건함을 높였다.

- **개별 노드 오차**: $error_{i,t} = |\hat{x}_{i,t} - x_{i,t}|$
- **수정된 Z-score (MAD 기반)**:
  $$a_{i,t} = \frac{error_{i,t} - med_i}{MAD_i}$$
  여기서 $med_i$는 오차의 중앙값이며, $MAD_i = med(|error_{i,t} - med_i|)$이다.
- **시스템 전체 점수(Collective Score)**: 모든 노드 중 가장 높은 점수를 선택하는 Max aggregation을 사용한다.
  $$s_t = \max_{i \in N} (a_{i,t})$$
- **최종 판정**: $s_t$가 동적으로 결정된 임계값 $\tau$(POT 전략 사용)보다 크면 이상치($y_t=1$)로 판정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: SWAT, WADI, SMAP, MSL, SMD, PSM 등 6개의 실제 세계 다변량 시계열 데이터셋을 사용하였다.
- **비교 모델**: LSTM-NDT, OmniAnomaly, USAD, TranAD(비그래프 기반), MTAD-GAT, GDN, GTA, DVGCRN(그래프 기반) 등 SOTA 모델들과 비교하였다.
- **평가 지표**: Point-wise F1, Composite Score ($F1_c$), Point-Adjustment F1 ($F1_{PA}$) 세 가지 지표를 사용하여 다각도로 평가하였다.

### 주요 결과

1. **정량적 성능**: CGAD는 모든 데이터셋에 대해 평균적으로 **약 9%의 성능 향상**을 보였다. 특히 $F1_c$와 $F1_{PA}$ 지표에서 압도적인 성능을 보였는데, 이는 개별 시점의 탐지보다 연속적인 이상치 이벤트(Event-level)를 탐지하는 능력이 매우 뛰어남을 의미한다.
2. **Ablation Study**:
    - **인과 그래프 제거(-Caugraph)**: 성능이 크게 하락하여, 단순 연결 그래프보다 인과 관계 기반 구조가 훨씬 유용한 지식을 제공함을 확인하였다.
    - **GCN 제거(-GConv)**: 노드 간 정보 교환이 사라져 성능이 저하되었다.
    - **MAD 정규화 제거(-Zscore)**: 성능 하락 폭이 가장 컸으며(평균 44.37% 감소), 이는 MAD 기반의 점수 산출이 노이즈가 많은 환경에서 이상치를 구분하는 데 핵심적인 역할을 함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **유연한 그래프 구조**: qualitative analysis 결과, Top-K 방식은 모든 노드의 차수가 동일하여 경직된 반면, CGAD의 인과 그래프는 실제 인과 관계의 유무에 따라 노드별 차수가 다양하게 나타나는 높은 유연성(Flexibility)을 보였다.
- **해석 가능성(Interpretability)**: Transfer Entropy를 통해 어떤 변수가 다른 변수에 영향을 주었는지 시각화할 수 있어, 이상 발생 시 어떤 센서가 원인이 되었는지 진단하는 '이상 진단(Anomaly Diagnosis)'이 가능하다.
- **강건한 점수 체계**: MAD를 활용한 정규화는 이상치 데이터가 섞여 있을 때 평균값이 왜곡되는 문제를 방지하여 탐지 정밀도를 높였다.

### 한계 및 향후 연구

- **데이터 분포 변화**: 정상 패턴의 분포가 시간에 따라 변하는 Distribution Shift 문제에 대한 고려가 부족하다.
- **재학습 비용**: 새로운 데이터가 들어올 때마다 인과 그래프를 다시 생성하거나 모델을 재학습해야 하는 부담이 있을 수 있다. 저자들은 이를 해결하기 위해 Lifelong Learning 메커니즘의 도입을 제안한다.

## 📌 TL;DR

본 논문은 다변량 시계열 데이터에서 변수 간의 **인과 관계를 Transfer Entropy로 추출하여 그래프를 구축**하고, 이를 **Weighted GNN과 Causal Convolution**으로 학습하여 미래 값을 예측하는 **CGAD** 프레임워크를 제안한다. 특히 **MAD 기반의 수정된 Z-score**를 통해 이상치 판정의 강건성을 확보하였으며, 실험 결과 기존 SOTA 모델들보다 이벤트 레벨의 이상치 탐지 성능이 월등히 높음을 증명하였다. 이 연구는 복잡한 시스템의 센서 데이터에서 단순 상관관계를 넘어선 인과적 통찰을 통해 이상치를 정밀하게 탐지하고 진단하는 데 중요한 기여를 한다.
