# Multivariate Time-Series Anomaly Detection based on Enhancing Graph Attention Networks with Topological Analysis

Zhe Liu, Xiang Huang, Jingyun Zhang, Zhifeng Hao, Li Sun, Hao Peng (2024)

## 🧩 Problem to Solve

본 논문은 산업 현장에서 필수적인 다변량 시계열(Multivariate Time-Series) 데이터의 비지도 학습 기반 이상치 탐지(Unsupervised Anomaly Detection) 문제를 해결하고자 한다. 다변량 시계열 데이터는 수많은 센서에서 생성되는 특성(Feature) 차원과 시간(Temporal) 차원이 복합적으로 얽혀 있어 분석이 매우 까다롭다.

기존의 방법론들은 주로 RNN을 통해 시간적 의존성을 모델링하거나, GNN 및 Transformer를 통해 공간적(특성 간) 의존성을 분석한다. 그러나 이러한 방식들은 단일 차원에만 집중하거나 조립도가 낮은(coarse-grained) 특징 추출 방식을 사용하기 때문에, 복잡한 상호 관계와 역동적인 변화가 존재하는 대규모 데이터셋에서 성능이 저하되는 한계가 있다. 특히, 전역적 이상치(Global anomalies)뿐만 아니라 정상 범위 내에 있지만 패턴이 무너진 문맥적 이상치(Contextual anomalies)를 효과적으로 탐지하는 것이 주요 과제이다.

따라서 본 연구의 목표는 시간적 차원과 특성 간 차원을 모두 세밀한 관점(fine-grained perspective)에서 분석할 수 있는 새로운 프레임워크인 TopoGDN을 제안하여 이상치 탐지 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Graph Attention Network(GAT)에 위상 분석(Topological Analysis)을 결합하여 특성 간의 고차원적 구조 정보를 추출하는 것이다. 주요 기여 사항은 다음과 같다.

1. **TopoGDN 프레임워크 제안**: 시간적 의존성과 특성 간 의존성을 서로 다른 다중 스케일(multi-scale)에서 분리하여 추출하는 이상치 탐지 구조를 설계하였다.
2. **Multi-scale Temporal Convolution 모듈**: 다양한 크기의 합성곱 커널을 사용하여 긴 시간 윈도우 내에서 발생할 수 있는 국소적(local) 특징과 전역적(global) 특징 간의 충돌을 해결하고 세밀한 시간적 특징을 추출한다.
3. **위상 분석 기반의 강화된 GAT**: Persistent Homology(지속성 호몰로지)를 도입하여 그래프의 필터링 과정에서 발생하는 고차원 위상 특징을 추출하고, 이를 노드 특징에 통합함으로써 특성 간 의존성 모델링의 정확도를 크게 향상시켰다.

## 📎 Related Works

### 다변량 시계열 이상치 탐지

기존 방법론은 크게 예측 기반(Prediction-based)과 재구성 기반(Reconstruction-based)으로 나뉜다.

- **예측 기반**: 과거 데이터를 통해 미래 값을 예측하고 실제 값과의 차이를 통해 이상치를 판별한다. GDN은 그래프 구조 학습을 통해 특성 간 의존성을 모델링했으나 시간적 의존성 활용이 부족했다. Anomaly Transformer는 Transformer를 통해 이를 해결하려 했으나 이상치가 매우 드문(sparse) 상황에서 취약함을 보였다.
- **재구성 기반**: VAE나 GAN 같은 생성 모델을 사용하여 데이터를 재구성하고 재구성 오차를 이용한다. 하지만 GAN의 학습 불안정성이나 VAE의 잠재 공간 평활화로 인한 데이터 흐림(blurring) 현상이 한계로 지적된다.

### 그래프 신경망(GNN) 및 위상 분석(Topological Analysis)

GCN과 같은 기존 GNN은 메시지 확산을 통해 특징을 추출하지만, 층이 깊어질수록 노드 특징이 유사해지는 Over-smoothing 문제가 발생한다. GAT는 어텐션 메커니즘을 통해 동적으로 관계를 조정하여 이를 개선했다. 한편, 위상 데이터 분석(TDA)은 연결 성분(connected components)이나 구멍(cavities) 같은 구조적 특징을 캡처하며, 특히 Persistent Homology(PH)는 그래프 필터링을 통해 불변하는 구조적 특성을 벡터화하여 GNN의 표현력을 높이는 데 활용될 수 있다.

## 🛠️ Methodology

TopoGDN은 예측 기반의 이상치 탐지 모델로, 크게 네 가지 모듈로 구성된다.

### 1. Multi-scale Temporal Convolution (MSTCN) 모듈

시간 윈도우 내의 다양한 시간적 해상도를 캡처하기 위해 $1 \times 2, 1 \times 3, 1 \times 5, 1 \times 7$ 크기의 1차원 합성곱 커널을 사용한다. $p$번째 커널에 대한 연산은 다음과 같다.

$$y^{(p)}[t] = \sum_{j=0}^{d_p-1} x_i[t+j] \cdot f_p[j]$$

여기서 $f_p$는 커널의 가중치, $d_p$는 커널의 너비이다. 이후 모든 스케일의 출력값들을 평균 풀링(Average Pooling)하여 최종 시간적 특징 $Z_i[t]$를 생성한다.

### 2. Graph Structure Learning 모듈

센서 간의 내재적 구조가 없는 경우, 데이터 간의 유사도를 기반으로 그래프 구조를 학습한다. 각 센서(노드)에 학습 가능한 임베딩 $c_i \in \mathbb{R}^d$를 할당하고, 코사인 유사도를 통해 에지를 생성한다.

$$e_{ij} = \frac{c_i \cdot c_j}{\|c_i\| \cdot \|c_j\|}$$

인접 행렬 $A_{ij}$는 유사도 $e_{ij}$가 상위 $K$개(Top-K)에 해당하는 경우 1, 그렇지 않으면 0으로 설정되는 이진 행렬로 구성된다.

### 3. Topological Feature Attention Module

본 모듈은 GAT의 어텐션 메커니즘과 위상 그래프 풀링(Topological Graph Pooling)의 결합으로 이루어진다.

- **Graph Attention Mechanism**: 노드 특징 $Z_i$와 센서 임베딩 $c_i$를 결합하여 어텐션 계수 $\alpha_{ij}$를 계산하고, 이웃 노드의 정보를 집계하여 노드 특징 $z_i$를 갱신한다.
- **High-order Topological Graph Pooling**:
    1. GAT로 정제된 그래프를 $k$개의 서로 다른 관점(View)으로 변환한다.
    2. 임계값 기반의 필터링(Threshold Filtering)을 통해 그래프의 부분집합 시퀀스 $\emptyset = \mathcal{G}^{(0)} \subseteq \mathcal{G}^{(1)} \subseteq \dots \subseteq \mathcal{G}^{(n)} = \mathcal{G}$를 생성한다.
    3. Vietoris-Rips complex를 적용하여 0차원(연결 성분) 및 1차원(사이클) 호몰로지 특징을 추출하고, 이를 Persistence Diagram 및 Barcode로 시각화한다.
    4. 최종적으로 $\Psi$ 함수(Triangle, Gaussian 등)를 통해 위상 바코드를 Topological Vector로 변환하여 특징에 통합한다.

### 4. Anomaly Score Calculator

위상 특징이 통합된 최종 임베딩 $p^{(t)}$를 Feed-Forward 네트워크에 통과시켜 다음 시점의 예측값 $\hat{x}_t$를 도출한다. 이상치 점수는 실제값 $x_t$와 예측값 $\hat{x}_t$ 사이의 $L_1$ 손실을 정규화한 후, 센서 차원에서의 최댓값으로 결정한다.

$$\text{Anomaly Score} = \max_{i=1, \dots, n} \left( \text{Normalize} \left( |\hat{x}_{it} - x_{it}| \right) \right)$$

학습 시에는 정규 데이터만을 사용하여 MSE(Mean Square Error)를 최소화하며, 테스트 시에는 검증 세트의 최대 점수를 임계값으로 설정하여 이상 여부를 판별한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MSL, SWaT, SMD, WADI 등 4개의 벤치마크 데이터셋을 사용하였다. 특히 WADI는 데이터 규모가 가장 커서 가장 도전적인 환경이다.
- **비교 대상**: PCA, LSTM-VAE, MAD-GAN, OmniAnomaly, GDN, TranAD, ImDiffusion 등 7가지 모델과 비교하였다.
- **지표**: Precision, Recall, F1-Score를 측정하였다.

### 정량적 결과

TopoGDN은 모든 데이터셋에서 가장 높은 F1-Score를 달성하였다. 특히 SWaT와 SMD 데이터셋에서는 2위 모델보다 약 2~3% 높은 성능을 보였다. 규모가 큰 WADI 데이터셋에서도 다른 모델보다 우수한 정확도와 F1-Score를 기록하며 일반화 능력을 입증하였다.

### 효율성 분석

모델 파라미터 수와 연산량(FLOPs), 실행 시간 측면에서도 TopoGDN은 매우 효율적이다.

- **파라미터 수**: TopoGDN(592K) $\ll$ TranAD(619K) $\ll$ ImDiffusion(4966K).
- **실행 시간**: TopoGDN(1.2h) $\ll$ TranAD(2.3h) $\ll$ ImDiffusion(7.8h).
특히 Diffusion 모델 기반의 ImDiffusion은 WADI 데이터셋에서 실행 시간이 너무 길어 결과를 산출하지 못했으나, TopoGDN은 안정적으로 작동하였다.

### 절제 연구(Ablation Study)

GDN 기본 모델에 MSTCN과 TA(Topological Analysis) 모듈을 추가했을 때 F1-Score가 평균 7%에서 19%까지 향상되었다. 특히 TA 모듈은 GNN의 특징 추출 능력을 획기적으로 높여주는 역할을 하며, MSTCN은 SMD와 같이 시간적 특징이 풍부한 데이터셋에서 효과가 컸다.

## 🧠 Insights & Discussion

본 연구는 GNN의 고질적인 문제인 Over-smoothing을 위상 분석(TDA)을 통해 해결하려 했다는 점이 매우 인상적이다. 단순한 인접성 기반의 메시지 전달을 넘어, 그래프의 고차원적 구조(연결 성분, 루프 등)를 벡터화하여 입력함으로써 모델이 데이터의 기하학적 특성을 이해할 수 있게 하였다.

또한, 제안된 MSTCN과 TA 모듈이 "Plug-and-play" 방식으로 설계되어, TopoGDN뿐만 아니라 다른 GNN이나 시간적 네트워크에도 쉽게 통합될 수 있다는 점은 확장성 측면에서 큰 강점이다.

다만, 하이퍼파라미터 민감도 분석에서 Batch Size가 너무 크면 위상 특징이 불분명해지거나, 그래프 필터링 횟수가 과도하면($f=20$) 오히려 성능이 급격히 하락하는 현상이 관찰되었다. 이는 위상 분석이 데이터의 구조에 매우 민감하게 반응함을 시사하며, 실제 적용 시 최적의 필터링 수준을 찾는 과정이 필수적임을 의미한다.

## 📌 TL;DR

본 논문은 다변량 시계열 이상치 탐지를 위해 **Multi-scale Temporal Convolution**과 **위상 분석(Persistent Homology)**이 결합된 GAT 기반의 **TopoGDN** 프레임워크를 제안한다. 이 모델은 시간적/공간적 의존성을 다중 스케일에서 정밀하게 추출하며, 특히 그래프의 고차원 구조 정보를 활용하여 기존 GNN의 한계를 극복하였다. 실험 결과, 4개의 벤치마크 데이터셋에서 기존 SOTA 모델들보다 높은 F1-Score를 기록하였으며, 연산 효율성 또한 매우 뛰어나 대규모 산업 데이터셋에 적용 가능성이 높음을 입증하였다.
