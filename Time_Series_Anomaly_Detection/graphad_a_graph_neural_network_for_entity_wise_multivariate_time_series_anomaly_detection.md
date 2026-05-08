# GraphAD: A Graph Neural Network for Entity-Wise Multivariate Time-Series Anomaly Detection

Xu Chen, Qiu Qiu, Changshan Li, Kunqing Xie (2022)

## 🧩 Problem to Solve

본 논문은 O2O(Online to Offline) 비즈니스 시나리오에서 발생하는 **Entity-Wise Multivariate Time-Series Anomaly Detection (EMTAD)** 문제를 해결하고자 한다. O2O 플랫폼(예: Ele.me)에서는 수많은 소매업체(Entity)들이 활동하며, 각 업체는 주문 수와 같은 핵심 성과 지표(Key Performance Indicator, KPI)를 포함한 다변량 시계열 데이터를 생성한다.

기존의 시계열 이상 탐지 방법들은 주로 시간적 패턴이나 속성 간의 상관관계에 집중하였으나, 서로 다른 엔티티(소매업체) 간의 개별적인 특성을 무시하는 경향이 있었다. 실제 비즈니스 환경에서 소매업체들은 사업 규모, 운영 패턴, 업종 등에 따라 KPI의 평균 수준과 변동성이 매우 상이하다. 모든 엔티티를 동일하게 취급하여 모델링할 경우, 개별 업체의 고유한 특성을 반영하지 못하고 공통적인 패턴만을 학습하게 되어 정밀한 이상 탐지가 어려워진다. 따라서 본 연구의 목표는 엔티티 간의 이질성을 고려하면서 속성 및 시간적 관계를 동시에 포착할 수 있는 새로운 이상 탐지 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 Graph Neural Network (GNN)를 활용하여 **속성(Attribute), 엔티티(Entity), 그리고 시간(Temporal)의 세 가지 관점에서 상관관계를 모델링**하는 것이다.

중심적인 직관은 다음과 같다. 첫째, KPI를 안정적인 성분(Stable component)과 변동성 성분(Volatility component)으로 분리하여 각각 다른 특성에 맞는 패턴을 추출해야 한다. 둘째, 동일한 업종의 업체들은 유사한 정적 특성을 가지며, 최근 성장세가 비슷한 업체들은 유사한 동적 패턴을 가진다는 점에 착안하여 이를 그래프 구조로 표현한다. 셋째, 속성 간의 관계 역시 고정된 것이 아니라 동적으로 변화하므로 Graph Attention Network (GAT)를 통해 이를 적응적으로 학습한다.

## 📎 Related Works

시계열 이상 탐지 연구는 크게 단변량(Univariate) 탐지와 다변량(Multivariate) 탐지로 나뉜다. ARIMA, STL, RobustSTL 등 단변량 방법론은 KPI 단일 지표의 패턴을 분석하지만, 다변량 데이터가 제공하는 풍부한 정보를 활용하지 못한다는 한계가 있다. 반면, Isolation Forest, VAE, LSTM-AE와 같은 다변량 방법론은 여러 속성 간의 상관관계를 추출하여 성능을 높였다.

최근에는 GDN이나 MTAD-GAT와 같이 GNN을 도입하여 센서나 속성 간의 의존성을 모델링하는 시도가 있었다. 하지만 이러한 기존 GNN 기반 방법론들은 다변량 시계열 데이터 자체에는 집중하였을 뿐, EMTAD 시나리오에서 필수적인 **엔티티 간의 차이(Entity-wise difference)**를 처리하는 메커니즘을 포함하지 않았다. 즉, 여러 엔티티가 존재하더라도 각 엔티티를 독립적으로 처리하거나 모두 동일한 특성을 가졌다고 가정하는 한계가 있었다.

## 🛠️ Methodology

GraphAD의 전체 파이프라인은 KPI 분해, 속성 그래프 학습, 엔티티-시간 그래프 학습, 그리고 최종 이상 탐지의 단계로 구성된다.

### 1. KPI Decomposition Module (K-Decom)

KPI의 안정적 성분과 변동성 성분이 서로 다른 진화 과정을 거치므로, 이를 분리하여 GNN에 입력함으로써 정보 손실을 방지한다. 입력 KPI $x$를 두 개의 독립적인 완전 연결 층(Fully-connected layers) $f_s$와 $f_v$를 통해 다음과 같이 분해한다.

$$x_s = f_s(x, \Theta_s) = \sigma(x\Theta_s)$$
$$x_v = f_v(x, \Theta_v) = \sigma(x\Theta_v)$$

여기서 $\Theta_s, \Theta_v$는 학습 가능한 파라미터이며, $\sigma(\cdot)$는 활성화 함수이다. 이때 Information Bottleneck (IB) 이론에 근거하여, 안정적 성분 $x_s$는 원본 KPI와의 상호 정보량(Mutual Information)을 최대화하고, 변동성 성분 $x_v$는 이를 최소화하도록 제약 조건을 부여한다.

$$\max_{\Theta_s} I(x, x_s), \quad \min_{\Theta_v} I(x, x_v)$$

### 2. Attribute Graph Attention Network (A-GAT)

속성 간의 복잡하고 동적인 관계를 포착하기 위해 속성 그래프 $\mathcal{G}_{nt}^A$를 구축한다. 각 노드는 입력 속성을 나타내며, 시계열 유사도 $\text{TiSim}(v_i, v_j)$가 높은 상위 $k$개의 노드를 연결하여 엣지를 생성한다. 엣지 생성 식은 다음과 같다.

$$e_{ij} = \mathbb{1}\{\text{TiSim}(v_i, v_j) \in \text{top-k}\{\text{TiSim}(v_i, v_n)\}\}$$

이후 GAT를 통해 각 속성의 잠재 표현 $z_i$를 학습한다.

$$z_i = \text{GAT}(\mathcal{G}, h_i) = \sigma(h_i + \sum_{v_j \in N(v_i)} \alpha_{ij} h_j)$$

여기서 $\alpha_{ij}$는 속성 간의 중요도를 나타내는 어텐션 스코어이다. 이 과정은 분리된 $x_s$와 $x_v$에 대해 각각 독립적인 GAT를 통해 수행된다.

### 3. Entity-Temporal Graph Attention Network (ET-GAT)

엔티티 간의 상관관계를 정적(Static) 측면과 동적(Dynamic) 측면에서 동시에 포착한다.

- **정적 그래프 $\mathcal{G}_S$**: 업종, 위치 등 정적 속성이 유사한 엔티티들을 연결한다.
- **동적 그래프 $\mathcal{G}_T$**: 시간적 패턴이 유사한 엔티티들을 연결한다.

두 그래프의 인접 행렬을 결합한 $\tilde{A}$를 사용하여 엔티티-시간 표현 $\tilde{z}_t$를 추출한다.

$$\tilde{A} = \begin{bmatrix} A_S & A_T \\ A_T & A_S \end{bmatrix}$$

최종적으로 현재 시점 $t$의 표현은 이전 시점 $t-1$과 현재 시점의 정보를 모두 참조하여 GAT를 통해 업데이트된다.

### 4. Anomaly Detection 및 학습 목표

분리된 두 성분(안정적, 변동성)에서 학습된 표현들을 결합(Concatenate)하여 MLP 층에 입력하고, 미래의 KPI 값 $\hat{y}$를 예측한다. 전체 손실 함수는 예측 오차를 줄이는 MSE 손실과 변동성 성분의 상호 정보량을 최소화하는 정규화 항의 합으로 정의된다.

$$\min \mathcal{L} = \mathcal{L}_{MSE} + \lambda I(x, x_v)$$

학습 데이터셋은 정상 데이터만을 사용하여 훈련하며, 테스트 시 예측값 $\hat{y}$와 실제값 $y$의 차이가 훈련 세트에서 관찰된 최대 차이(임계값)보다 클 경우 이상치로 판정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Ele.me의 실제 거래 데이터를 기반으로 구축한 EMTAD 데이터셋을 사용하였다. 총 114개의 속성이 포함되어 있으며, 30일간의 데이터를 통해 31일째의 상태를 예측하는 슬라이딩 윈도우 방식을 적용하였다.
- **비교 대상**: AE, VAE, Isolation Forest, DeepSVDD, COPOD를 베이스라인으로 설정하였다. GDN과 MTAD-GAT는 단일 엔티티 기반 모델로, 대규모 엔티티 집합에 적용하기에 시간 비용이 너무 크고 엔티티 간 분산으로 인해 성능 저하가 우려되어 제외하였다.
- **평가 지표**: Precision, Recall, F1-Score, AUC(ROC)를 사용하였다.

### 결과 분석

실험 결과, GraphAD는 Precision(75.86%), F1-Score(53.99%), AUC(70.58%) 지표에서 기존 방법론들을 압도하였다. 특히 Precision의 경우 기존 방법론 대비 최대 184.4%의 상대적 향상을 보였다. Recall은 다른 모델들에 비해 약간 낮거나 유사한 수준이었으나, 실제 산업 현장에서는 오탐(False Positive)을 줄여 사용자에게 정확한 신호를 주는 것이 더 중요하므로 Precision의 비약적인 향상이 매우 가치 있는 결과로 해석된다.

### Ablation Study

각 모듈의 기여도를 분석한 결과, 엔티티 간의 차이를 모델링하는 부분을 제거했을 때(`w/o entityGAT`) 성능 저하가 가장 심하게 나타났다. 이는 EMTAD 문제에서 엔티티별 특성을 반영하는 것이 모델 성능의 핵심임을 시사한다. 또한 K-Decom, A-GAT, temporalGAT를 제거했을 때도 모든 지표가 하락하여 각 구성 요소의 설계가 타당함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 다변량 시계열 이상 탐지에 GNN을 적용함에 있어, 단순히 속성 간의 관계를 보는 것을 넘어 **엔티티라는 상위 개념의 이질성**을 성공적으로 모델링하였다. 특히 KPI를 안정적 성분과 변동성 성분으로 분해하여 서로 다른 제약 조건(상호 정보량 최대화/최소화)을 부여한 점은 데이터의 특성을 깊게 이해한 설계라고 판단된다.

또한, 산업적 관점에서 Precision의 중요성을 강조한 점이 인상적이다. 많은 이상 탐지 모델들이 Recall을 높이는 데 집중하지만, 실제 비즈니스 어시스턴트에서는 잦은 오경보가 오히려 신뢰도를 떨어뜨리기 때문에 정밀도를 높인 GraphAD의 접근 방식이 실용적이다.

다만, 상호 정보량(Mutual Information)의 직접적인 계산이 어려워 CLUB라는 근사 방식을 사용한 점과, 임계값(Threshold)을 엔티티별로 다르게 설정해야 한다는 점이 실제 대규모 배포 시 관리 포인트가 될 수 있다.

## 📌 TL;DR

본 논문은 O2O 플랫폼의 소매업체별 특성을 반영한 **Entity-Wise Multivariate Time-Series Anomaly Detection (EMTAD)** 문제를 정의하고, 이를 해결하기 위한 GNN 기반 모델인 **GraphAD**를 제안하였다. KPI 분해 모듈과 속성 및 엔티티-시간 그래프 어텐션 네트워크를 통해 데이터의 복잡한 상관관계를 포착하였으며, Ele.me의 실제 데이터를 통해 기존 모델 대비 비약적인 Precision 향상을 입증하였다. 이 연구는 다수의 이질적인 엔티티가 공존하는 대규모 시계열 데이터 환경에서 매우 효과적인 이상 탐지 프레임워크를 제공한다는 점에서 향후 산업적 적용 가치가 매우 높다.
