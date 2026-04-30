# Federated Learning with Graph-Based Aggregation for Traffic Forecasting

Audri Banik, Glaucio Haroldo Silva de Carvalho, Renata Dividino (2025)

## 🧩 Problem to Solve

교통량 예측(Traffic Forecasting)의 핵심은 도로 네트워크 상의 센서들로부터 수집된 시계열 데이터를 활용하여 특정 지역이나 도로 구간의 교통 속도 및 흐름을 정확히 추정하는 것이다. 이러한 시스템에서 각 센서나 지역을 하나의 클라이언트로 간주하면, 원본 데이터를 공유하지 않고 모델을 학습시키는 연합 학습(Federated Learning, FL) 방식이 개인정보 보호 측면에서 매우 적합하다.

그러나 기존의 대표적인 연합 학습 알고리즘인 Federated Averaging (FedAvg)는 모든 클라이언트의 데이터가 독립적이고 동일하게 분포(IID, Independent and Identically Distributed)되어 있다고 가정한다. 교통 데이터의 경우, 인접한 도로 센서들 간의 공간적 관계(Spatial Relationship)가 예측 성능에 결정적인 영향을 미치므로, 이러한 IID 가정은 실제 환경과 맞지 않으며 성능 저하를 야기한다. 최근 이를 해결하기 위해 그래프 신경망(GNN)을 연합 학습에 결합한 연구들이 등장했으나, 서버 측에서 복잡한 GNN 모델을 학습시켜야 하므로 상당한 계산 오버헤드가 발생하는 문제가 있다.

따라서 본 논문의 목표는 FedAvg의 단순함과 그래프 학습의 공간적 관계 추출 능력을 결합하여, 계산 효율적이면서도 성능이 뛰어난 경량화된 그래프 인식 연합 학습(Lightweight Graph-aware FL) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 서버에서 복잡한 GNN 모델을 전체적으로 학습시키는 대신, 그래프 연결성(Graph Connectivity)에 기반한 단순한 이웃 집계(Neighbourhood Aggregation) 원리를 파라미터 업데이트 과정에 적용하는 것이다. 즉, 모델 파라미터 자체를 그래프의 노드 특징(Node Feature)으로 취급하고, 그래프 구조에 따라 가중치를 부여하여 집계함으로써 공간적 의존성을 효율적으로 캡처한다. 이를 통해 높은 계산 비용 없이도 클라이언트 간의 공간적 관계를 모델에 반영할 수 있는 실용적인 균형점을 제시한다.

## 📎 Related Works

교통 예측을 위한 연합 학습 시스템은 크게 서버 측 설정과 클라이언트 측 설정으로 나뉜다.

1.  **서버 측 설정 (Server-Side Setting):** 서버가 전역 그래프를 유지하며 집계 전략을 최적화한다. FedAGCN, FedASTA, FedDA, FedGM, CNFGNN 등이 이에 해당하며, 클러스터링, 어텐션 메커니즘, 메타 학습 등을 사용하여 공간적-시간적 관계를 모델링한다. 하지만 이러한 방식들은 복잡한 학습 파이프라인으로 인해 계산 비용이 높고 확장성이 떨어진다는 한계가 있다.
2.  **클라이언트 측 설정 (Client-Side Setting):** 서버는 단순한 FedAvg를 사용하고, 대신 클라이언트 내부의 모델 구조(예: Spatio-Temporal Networks)를 고도화하는 방식이다. pFedCTP, M3FGM, FedAGAT 등이 대표적이다. 이 방식은 서버 측 집계 알고리즘의 개선 가능성을 간과하고 있다는 점이 특징이다.
3.  **기타 시계열 작업:** EV 충전 수요 예측(PFGL)이나 스마트 홈 CIoT 장치(GFIoTL) 등에서도 그래프 기반 FL이 시도되었으며, 공간적 근접성이나 행동 유사성을 그래프로 모델링하여 집계에 활용하였다.

본 논문은 서버 측 설정에 집중하되, 기존의 무거운 GNN 기반 학습 대신 경량화된 가중 평균(Weighted Averaging) 방식을 제안함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 시스템은 $N$개의 교통 센서(클라이언트)와 하나의 중앙 서버로 구성된다. 도로 네트워크는 방향 그래프 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{W})$로 모델링되며, 인접 행렬 $\mathcal{A} \in \{0, 1\}^{N \times N}$가 도로의 연결성을 나타낸다. 각 클라이언트는 로컬 데이터를 사용하여 시계열 예측 모델을 학습하고, 서버는 수집된 모델 파라미터를 그래프 구조에 따라 집계하여 다시 배포한다.

### 로컬 모델 (Client-side)
각 클라이언트는 GRU 기반의 Encoder-Decoder 아키텍처를 사용하여 시간적 역학(Temporal Dynamics)을 모델링한다. 
- **Encoder:** 입력 시퀀스 $x_i$를 처리하여 문맥 벡터(Hidden state) $h_{c,i}$를 생성한다.
- **Decoder:** 생성된 $h_{c,i}$와 마지막 입력 프레임을 바탕으로 미래의 교통 상태 $\hat{y}_i$를 자기회귀(Auto-regressive) 방식으로 예측한다.
- **손실 함수:** 예측값과 실제값 사이의 Mean Squared Error (MSE)를 사용하여 로컬에서 학습한다.

### 서버 집계 방법 (Server-side Aggregation)
서버는 각 클라이언트의 모델 파라미터를 노드 특징 행렬 $X \in \mathbb{R}^{N \times P}$로 간주한다 ($P$는 파라미터 수). 본 논문에서는 두 가지 경량 집계 방법을 제안한다.

#### 1. Graph Neighbourhood-Aware Averaging (GraphFedAvg)
전통적인 FedAvg를 확장하여 그래프 구조를 반영한 방식이다. 먼저 자기 루프(Self-loop)가 추가된 인접 행렬 $\tilde{A} = A + I$와 차수 행렬(Degree Matrix) $\tilde{D}$를 정의한다.
파라미터 업데이트 식은 다음과 같다:
$$X^{(\ell+1)} = \tilde{D}^{-1} \tilde{A} X^{(\ell)}$$
여기서 $\ell$은 전파 단계이며, 이 연산을 $\mathcal{L}$번 반복하면 $\mathcal{L}$-hop 이웃의 정보가 전파된다. 개별 클라이언트 $i$ 관점에서의 업데이트 식은 다음과 같다:
$$X_i^{(\ell+1)} = \sum_{j=1}^{N} \frac{\tilde{A}_{ij}}{\tilde{D}_{ii}} X_j^{(\ell)}$$
이는 단순히 모든 클라이언트를 평균 내는 것이 아니라, 연결된 이웃들의 파라미터를 가중 평균하는 방식이다.

#### 2. Graph Message Passing-Aware Averaging (MPFedAvg)
Label Propagation 알고리즘에서 영감을 얻은 방식으로, 자신의 원래 파라미터와 이웃의 정보를 적절히 혼합하여 정교하게 업데이트한다.
업데이트 식은 다음과 같다:
$$X^{(\ell+1)} = \alpha \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X^{(\ell)} + (1-\alpha) X^{(\ell)}$$
여기서 $\alpha \in [0, 1]$는 하이퍼파라미터로, 로컬 파라미터 유지와 이웃 정보 수용 사이의 트레이드-오프를 조절한다. 개별 클라이언트 $i$에 대한 식으로 표현하면 다음과 같다:
$$X_i^{(\ell+1)} = \alpha \sum_{j=1}^{N} \frac{\tilde{A}_{ij}}{\sqrt{\tilde{D}_{ii}} \sqrt{\tilde{D}_{jj}}} X_j^{(\ell)} + (1-\alpha) X_i^{(\ell)}$$
이 방식은 정규화된 인접 행렬을 사용하여 파라미터를 전파함으로써, 다중 홉(Multi-hop) 이웃의 이점을 누리면서도 초기 로컬 모델의 특성을 유지할 수 있게 한다.

## 📊 Results

### 실험 설정
- **데이터셋:** PEMS-BAY (325개 센서), METR-LA (207개 센서).
- **평가 지표:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE).
- **비교 대상 (Baselines):** 
    - Centralized GRU (상한선), Local GRU, GRU + FedAvg, GRU + FMTL, CNFGNN.
- **하이퍼파라미터:** $\alpha=0.8$ (MPFedAvg), 로컬 에포크 3회, 클라이언트-서버 라운드 5회.

### 주요 결과 (RMSE 기준)
| Method | PEMS-BAY | METR-LA |
| :--- | :---: | :---: |
| GRU (Centralized) | 4.172 | 11.787 |
| GRU (Local) | 4.152 | 12.224 |
| GRU + FedAvg | 4.432 | 12.058 |
| CNFGNN | 3.822 | 11.487 |
| **GraphFedAvg (2L)** | **3.745** | **11.473** |
| **MPFedAvg (1L)** | **3.733** | **11.489** |

- **FedAvg의 한계:** PEMS-BAY 데이터셋에서 FedAvg는 오히려 로컬 학습보다 성능이 낮게 나타났다($4.432$ vs $4.152$). 이는 단순 평균 방식이 공간적 구조를 무시하기 때문임을 보여준다.
- **제안 방법의 우수성:** 제안된 GraphFedAvg와 MPFedAvg는 모든 베이스라인을 능가하였으며, 특히 가장 강력한 베이스라인인 CNFGNN 대비 약 $1.9\%$에서 $8.1\%$의 성능 향상을 보였다.
- **계층 수 영향:** 1층(1L)과 2층(2L) 변형 간의 성능 차이가 매우 미미하여, 단일 층의 그래프 연산만으로도 필수적인 공간 의존성을 충분히 캡처할 수 있음을 확인하였다. 이는 계산 효율성 측면에서 매우 큰 이점이다.

## 🧠 Insights & Discussion

본 연구는 서버 측 집계 단계에 그래프 구조를 도입하는 것만으로도 복잡한 GNN 학습 없이 상당한 성능 향상을 얻을 수 있음을 증명하였다. 특히 FedAvg가 교통 예측과 같은 공간 의존적 데이터에서 성능이 저하되는 문제를 정확히 짚어냈으며, 이를 경량화된 가중 평균 방식으로 해결하였다.

**강점:**
- 모델의 복잡도를 크게 높이지 않고도 서버 측 집계 알고리즘 개선만으로 성능을 높였다.
- 계산 비용이 매우 낮아 자원이 제한된 환경(Resource-constrained environments)에서도 적용 가능하다.

**한계 및 논의사항:**
- 본 연구는 정적인 그래프(Static Graph)를 가정하고 있다. 하지만 실제 교통 시스템에서는 사고나 도로 공사 등으로 인해 도로 간의 관계가 실시간으로 변하는 동적 그래프(Dynamic Graph) 특성이 강하다.
- $\alpha$와 같은 하이퍼파라미터에 대한 민감도 분석이 충분히 제시되지 않았으며, 최적의 $\alpha$ 값을 찾는 일반적인 방법론에 대한 논의가 필요하다.

## 📌 TL;DR

본 논문은 교통량 예측을 위한 연합 학습에서 기존 FedAvg의 IID 가정 한계를 극복하기 위해, **서버 측 집계 과정에 그래프 연결성을 반영한 경량화된 집계 알고리즘(GraphFedAvg, MPFedAvg)**을 제안한다. 복잡한 GNN 학습 없이 단순한 이웃 파라미터 가중 평균만으로도 기존 GNN 기반 FL 방법론보다 우수한 성능(RMSE 기준 최대 8.1% 향상)을 보였으며, 이는 계산 효율성과 예측 정확도 사이의 실용적인 타협점을 제시한 연구이다. 향후 동적 그래프로의 확장을 통해 실제 도로 환경에서의 적용 가능성을 높일 수 있을 것으로 기대된다.