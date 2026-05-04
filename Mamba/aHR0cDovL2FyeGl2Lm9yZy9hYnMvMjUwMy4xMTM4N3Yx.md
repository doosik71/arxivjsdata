# Hierarchical Information-Guided Spatio-Temporal Mamba for Stock Time Series Forecasting

Wenbo Yan, Shurui Wang, and Ying Tan (2025)

## 🧩 Problem to Solve

주식 시계열 예측은 투자 결정의 핵심 요소이지만, 극심한 변동성과 복잡한 상호 의존성으로 인해 매우 어려운 과제이다. 최근 Mamba 모델이 우수한 선택 메커니즘(selection mechanism)을 바탕으로 다양한 시계열 예측 작업에서 뛰어난 성능을 보였으나, 주식 시장에 적용할 때 다음과 같은 두 가지 주요 한계가 존재한다.

첫째, 기존 Mamba 기반 모델들은 개별 주식의 시계열을 독립적으로 모델링하는 경향이 있어, 동일 시장 내 다른 주식들이 개별 주식에 미치는 영향, 즉 상호 의존성(interdependence)을 충분히 캡처하지 못한다. 

둘째, Mamba의 핵심인 선택 메커니즘이 오직 과거의 시계열 데이터에만 의존한다. 이로 인해 시장 전체의 흐름을 나타내는 거시적 정보(macro information)를 통합하여 더 지능적인 선택을 내리는 능력이 부족하다.

따라서 본 논문의 목표는 주식 시장의 전반적인 역학(market dynamics)과 개별 주식 간의 복잡한 상호 관계를 모두 포착할 수 있는 계층적 구조의 Spatio-Temporal Mamba 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 주식 데이터를 **공통성(Commonality)**과 **특이성(Specificity)**으로 분리하고, 이를 기반으로 **개별 $\rightarrow$ 시간적 $\rightarrow$ 글로벌** 단계로 이어지는 계층적 모델링 구조를 구축하는 것이다.

1. **Index-Guided Frequency Filtering Decomposition**: 주식 지수(Index)를 가이드로 사용하여 주식 시계열을 거시적 흐름인 공통성과 개별 특성인 특이성으로 분해한다.
2. **Hierarchical Spatio-Temporal Structure**: Node Independent Mamba, TIGSTM, GIGSTM의 3단계 계층 구조를 통해 주식의 내재적 특성, 동적인 시간적 관계, 그리고 정적인 글로벌 관계를 순차적으로 학습한다.
3. **Information-Guided Mamba**: 거시적 정보(Macro information)를 Mamba의 선택 메커니즘(입력 및 출력 행렬 생성 과정)에 직접 통합하여, 시장 상황을 인식하는 의사결정이 가능하도록 설계하였다.

## 📎 Related Works

논문에서는 시계열 예측 모델을 세 가지 범주로 나누어 설명한다.

- **Temporal Models**: Transformer 기반의 Informer, FEDformer와 CNN 기반의 TimesNet, 그리고 최신 Mamba 기반의 TimeMachine 등이 있으며, 주로 단일 시퀀스의 패턴 추출에 집중한다.
- **Spatio-Temporal Models**: ASTGCN, MTGNN, DCRNN 등은 시간적 변화와 공간적(변수 간) 관계를 동시에 모델링하며, 주로 그래프 신경망(GNN)을 활용한다.
- **Stock Price Forecasting Models**: MTDNN이나 MASTER와 같이 주식 시장의 특수성을 반영한 모델들이 제안되었으나, Mamba의 효율적인 롱-시퀀스 모델링 능력을 주식 시장의 거시/미시 구조와 결합한 시도는 부족했다.

본 제안 방법은 기존 Spatio-Temporal 모델들이 정적인 그래프나 단순한 어텐션에 의존했던 것과 달리, 주식 지수를 이용한 주파수 도메인 분해와 Mamba의 선택 메커니즘 제어라는 차별점을 가진다.

## 🛠️ Methodology

### 1. Index-Guided Frequency Filtering Decomposition
주식 데이터에서 거시적 흐름(Commonality)과 개별 특성(Specificity)을 분리하기 위해 Fast Fourier Transform (FFT)를 사용한다. 주식 지수($I$)의 진폭(amplitude)을 이용하여 필터 파라미터를 학습하고, 이를 통해 주식 시계열 $X$를 다음과 같이 분해한다.

- **Commonality ($X_c$)**: 주식 지수의 진폭 정보를 이용해 필터링하여 시장 전체의 공통적 경향성을 추출한다.
  $$X^f_c = X^f \Theta \sigma(W^T_c I^f_{amp})$$
- **Specificity ($X_s$)**: 전체 신호에서 공통성을 제거하고 추가 필터링을 거쳐 개별 주식만의 고유한 특성을 추출한다.
  $$X^f_s = (X^f - X^f_c) \Theta (1 - \sigma(W^T_s I^f_{amp}))$$

최종적으로 Inverse FFT (iFFT)를 통해 시간 도메인으로 복원하여 $X_c$와 $X_s$를 얻는다.

### 2. Node Independent Mamba
이웃 노드의 정보가 유입되기 전, 각 주식의 내재적 특성을 먼저 학습하는 단계이다. 1D Convolution과 Linear Projection을 통해 Mamba의 파라미터 $\Delta, B, C$를 생성하고, State Space Model (SSM)을 통해 독립적인 특징을 추출한다. 이는 펀더멘털이 약한 주식은 시장 상황이 좋아도 상승하기 어렵다는 주관적 평가 과정을 모사한 것이다.

### 3. Temporal Information-Guided Spatio-temporal Mamba (TIGSTM)
시간 단계별로 변하는 동적인 관계와 거시 정보를 통합한다.
- **Sparse Neighbor Aggregation**: 분해된 특이성($X_s$)을 이용해 각 시간 단계마다 Attention 기반의 희소 그래프를 생성하며, 상위 30%의 이웃 노드 정보만 집계한다.
- **Information-Guided Selection**: 공통성($X_c$)에서 추출한 시간 단계별 거시 정보($M_S$)를 Mamba의 입력 행렬 $B_S$와 출력 행렬 $C_S$에 연결(concatenate)하여 시퀀스 선택 과정을 가이드한다.
  $$B_S = \langle W^T_{B,S} X_S, \text{Broadcast}(M_S) \rangle$$

### 4. Global Information-Guided Spatio-temporal Mamba (GIGSTM)
전체 기간에 걸친 정적인 관계와 글로벌 거시 정보를 학습한다.
- **Global Neighbor Aggregation**: 모든 시간 단계의 특이성 정보를 통합하여 완전 연결 그래프(Fully Connected Graph)를 구축하고 전역적인 이웃 정보를 집계한다.
- **Global Information-Guided Selection**: 시간 단계별 거시 정보($M_S$)를 다시 통합하여 글로벌 거시 정보($M_G$)를 생성하고, 이를 Mamba의 $B_G, C_G$ 행렬에 통합하여 최종 선택 과정을 가이드한다.

### 5. Prediction and Loss
최종 출력 $O_G$를 선형 층에 통과시켜 평균(mean)과 편차(dev)를 예측하는 mean-deviation 접근 방식을 사용하며, 최종 예측값은 $\hat{Y} = \text{mean} + e^{\text{dev}}$로 계산한다. 손실 함수로는 순위 분포 학습을 위해 Pearson 상관계수 손실($L_{pearson}$)을 사용한다.
$$L_{pearson} = -\frac{(Y - \bar{Y})^T (\hat{Y} - \bar{\hat{Y}})}{\sqrt{(Y - \bar{Y})^T (Y - \bar{Y})} \cdot \sqrt{(\hat{Y} - \bar{\hat{Y}})^T (\hat{Y} - \bar{\hat{Y}})}}$$

## 📊 Results

### 실험 설정
- **데이터셋**: CSI500, CSI800, CSI1000 (중국 주식 시장 데이터)
- **비교 모델**: ASTGCN, MTGNN, DCRNN (Spatio-Temporal), iTransformer, TimesNet (Temporal), MASTER (Stock specialized)
- **평가 지표**: IC (상관계수), PNL (손익), MAXD (최대 낙폭), SHARPE (샤프 지수), WINR (승률), PL (손익비)

### 주요 결과
HIGSTM은 모든 데이터셋에서 IC, PNL, SHARPE, WINR, PL 지표에서 SOTA(State-of-the-art) 성능을 달성하였다.
- **정량적 개선**: 평균적으로 IC는 11%, SHARPE는 10%, PNL 및 PL은 6% 이상 향상되었다.
- **리스크 분석**: CSI500 데이터셋에서 MAXD(최대 낙폭)가 Filternet보다 약간 높게 나타나 리스크가 소폭 증가했으나, SHARPE 지수가 15% 이상 크게 향상된 점으로 보아 리스크 대비 수익률이 압도적으로 높음을 알 수 있다.
- **Ablation Study**: TIGSTM, GIGSTM, Decomposition, Index-guided filtering 중 하나라도 제거했을 때 성능이 30% 이상 급격히 하락하여, 제안한 모든 구성 요소가 필수적임을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 분석을 통해 다음과 같은 통찰을 얻을 수 있다.

1. **특이성(Specificity)의 중요성**: 원본 시계열을 그대로 사용하여 그래프를 생성했을 때는 이웃 간의 차이가 뚜렷하지 않았으나, 공통성을 제거한 특이성 데이터를 사용했을 때 비로소 유의미하고 뚜렷한 상관관계를 가진 이웃 노드들이 식별되었다. 이는 시장의 공통 흐름이 개별 주식 간의 고유한 관계를 가리고 있었음을 의미한다.
2. **Temporal vs Global Graph**: 시간적 그래프(Temporal Graph)는 매우 희소하며 빠르게 변하는 관계를 포착하는 반면, 글로벌 그래프(Global Graph)는 더 다양하고 정적인 관계를 포착한다. 이 두 가지를 계층적으로 처리함으로써 동적/정적 시장 특성을 모두 반영할 수 있었다.
3. **Mamba 선택 메커니즘의 확장**: Mamba의 선택 메커니즘에 단순 시계열 외에 거시적 정보를 외부 주입(injection)하는 방식이 주식 예측과 같은 복잡한 도메인에서 매우 효과적임을 보여주었다.

한계점으로는, 수익률을 높이기 위해 리스크(MAXD)가 소폭 상승하는 경향이 있다는 점이 언급된다. 이는 고수익-고위험 전략의 특성일 수 있으나, 향후 리스크를 더욱 정교하게 제어하는 메커니즘이 추가될 필요가 있다.

## 📌 TL;DR

본 논문은 주식 시계열 예측을 위해 **주식 지수 기반의 주파수 분해(Decomposition)**와 **계층적 Spatio-Temporal Mamba 구조**를 제안한 HIGSTM 모델을 제시한다. 주식 데이터를 공통성과 특이성으로 분리하여 각각 거시 정보와 이웃 관계 추출에 활용하고, 이를 Mamba의 선택 메커니즘에 통합함으로써 시장 상황을 인식하는 예측이 가능하게 했다. 실험 결과, CSI500/800/1000 데이터셋 모두에서 기존 모델들을 압도하는 수익률과 예측 정확도를 달성하여, 향후 퀀트 투자 및 시계열 예측 연구에 중요한 기여를 할 것으로 보인다.