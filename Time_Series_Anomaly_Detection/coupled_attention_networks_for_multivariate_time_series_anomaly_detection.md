# Coupled Attention Networks for Multivariate Time Series Anomaly Detection

Feng Xia, Xin Chen, Shuo Yu, Mingliang Hou, Mujie Liu, and Linlin You (2023)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 이상 탐지(Multivariate Time Series Anomaly Detection, MTAD)에서 발생하는 변수 간의 복잡한 의존성 문제를 해결하고자 한다. 실제 환경에서 수집되는 다변량 시계열 데이터는 수많은 센서로부터 생성되며, 각 센서(변수) 사이에는 상호 의존성이 존재한다.

이러한 변수 간의 관계는 정적이지 않고 시간에 따라 동적으로 변화하는 특성을 가진다. 기존의 많은 모델들이 정적인 그래프 구조를 사용하거나, 혹은 동적인 관계만을 학습하여 전역적인 상관관계를 놓치는 한계가 있었다. 따라서 본 연구의 목표는 전역적(Global) 상관관계와 지역적(Local) 동적 상관관계를 동시에 포착할 수 있는 Coupled Attention Network(CAN) 프레임워크를 제안하여, 변수 간의 동적인 관계 변화가 심한 환경에서도 높은 이상 탐지 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전역-지역 그래프(Global-Local Graph)와 시간적 자기 주의 집중(Temporal Self-Attention) 메커니즘을 결합하여 변수 간의 복잡한 의존성과 시간적 의존성을 동시에 모델링하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Global-Local Graph Convolutional Network**: 센서의 내재적 특성을 반영하는 전역 그래프와 현재 입력 시퀀스에 따라 변하는 지역 그래프를 함께 구축하여, 정적 및 동적 관계를 모두 표현하는 구조를 설계하였다.
2. **Coupled Attention Network**: 시간적 의존성을 학습하는 Temporal Self-Attention 모듈과 변수 간 의존성을 학습하는 Global-Local Graph Convolutional 레이어를 결합한 Coupled Attention Module(CAM)을 제안하였다.
3. **Multilevel Encoder-Decoder Framework**: 예측(Prediction)과 재구성(Reconstruction) 작업을 동시에 수행하는 멀티레벨 구조를 통해 시계열 데이터의 표현력을 극대화하였으며, 특히 재구성 작업은 학습 단계에서 표현 학습을 돕는 보조 수단으로 활용하였다.

## 📎 Related Works

기존의 시계열 이상 탐지 연구는 크게 단변량(Univariate)과 다변량(Multivariate) 접근 방식으로 나뉜다.

* **단변량 접근 방식**: ARIMA와 같은 통계 모델이나 AE, VAE, LSTM 기반의 재구성 모델들이 사용되었으나, 변수 간의 상관관계를 고려하지 못한다는 한계가 있다.
* **다변량 접근 방식**:
  * **재구성 기반(Reconstruction-based)**: LSTM-VAE, OmniAnomaly 등이 있으며, 정상 데이터의 저차원 표현을 학습하여 재구성 오차를 통해 이상치를 탐지한다. 주로 전역적인 데이터 분포 포착에 유리하다.
  * **예측 기반(Prediction-based)**: LSTM-NDT 등이 있으며, 과거 데이터를 통해 미래 값을 예측하고 예측 오차를 이용한다. 특정 시점의 급격한 변화를 포착하는 데 유리하다.
  * **그래프 학습 기반(Graph Learning-based)**: GDN, MTAD-GAN, GTA 등이 변수 간 관계를 그래프로 모델링한다. 하지만 정적 그래프는 동적 변화를 반영하지 못하고, GAT와 같은 동적 그래프는 전역적인 상관관계를 무시하는 경향이 있다.

본 논문은 예측과 재구성 작업을 결합하고, 전역-지역 그래프를 동시에 사용하여 기존 모델들이 가진 정적/동적 관계 포착의 불균형 문제를 해결함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

CAN 프레임워크는 하나의 Encoder와 두 개의 Decoder(Prediction Decoder, Reconstruction Decoder)로 구성된다. Encoder는 입력 데이터의 시간적 및 변수 간 의존성을 추출하여 표현 벡터(Representation Vector)를 생성하고, 이를 두 Decoder에 전달한다.

### 2. Coupled Attention Module (CAM)

CAM은 모델의 핵심 레이어로, 다음 두 가지 구성 요소로 이루어져 있다.

**A. Temporal Self-attention Layer**
각 센서별로 병렬적인 self-attention 레이어를 적용하여 시간적 의존성을 학습한다. 입력 $H^n$에 대해 Query($Q^n$), Key($K^n$), Value($V^n$)를 생성하며, 다음과 같이 계산된다.
$$Q^n = H^n W_q, \quad K^n = H^n W_k, \quad V^n = H^n W_v$$
이후 Scaled Dot-Product Attention을 통해 시간적 관계를 업데이트하며, Multi-head attention을 적용하여 풍부한 정보를 추출한다.

**B. Global-Local Graph Convolutional Layer**
변수 간의 관계를 두 가지 그래프로 정의한다.

* **전역 그래프(Global Graph)**: 학습 가능한 임베딩 벡터 $E$를 사용하여 센서 간의 정적 유사도를 계산한다.
    $$A^g = \text{Relu}(E \times E^T)$$
* **지역 그래프(Local Graph)**: 현재 입력 데이터 $H$와 센서 특성을 결합하여 동적인 주의 계수 $a^l_{i,j}$를 계산함으로써 실시간 관계를 반영한다.
* **결합 및 컨볼루션**: 전역 그래프 $A^g$에서 상위 $K_m$개의 이웃만을 선택하여 마스크 행렬 $A^m$을 생성하고, 이를 통해 지역 그래프를 필터링한다. 최종 상태 업데이트는 다음과 같다.
    $$A^{gl} = A^m (A^l + \tilde{A}^g)$$
    $$H = (\beta H + (1-\beta) A^{gl} H) W$$
    여기서 $\beta$는 원래 상태를 유지하는 비율을 조절하는 하이퍼파라미터이다.

### 3. Multilevel Encoder-Decoder 및 학습 절차

* **Encoder**: 여러 층의 CAM을 쌓아 시퀀스 임베딩 $X^e$를 생성한다. $X^e$는 Autoencoder(AE)를 통해 저차원으로 투영되었다가 다시 복원되어 Decoder로 전달된다.
* **Decoders**: 두 Decoder 모두 Masked Self-Attention을 사용하여 단방향성을 유지한다.
  * **Prediction Decoder**: 다음 시점의 값 $Y_{:,K+1}$을 예측한다.
  * **Reconstruction Decoder**: 전체 입력 시퀀스 $Y_{:,:K}$를 재구성한다.
* **손실 함수**: 예측 손실($L_{pre}$)과 재구성 손실($L_{rec}$) 모두 RMSE(Root Mean Square Error)를 사용하며, 가중 합으로 최종 손실을 정의한다.
    $$L = \phi L_{pre} + \phi L_{rec} \quad (\phi + \phi = 1)$$
    학습 초기에는 재구성 작업($\phi$)에 더 높은 비중을 두고, 후기에는 예측 작업($\phi$)의 비중을 높여 학습을 최적화한다.

### 4. 추론 및 이상 점수 계산

추론 단계에서는 오버피팅 위험이 있는 재구성 모델을 제외하고 **예측 모델**만을 사용한다.

1. **예측 오차 계산**: $\text{Err}^n(t) = |X^n_{t} - Y^n_{t}|$
2. **정규화**: 센서별 특성을 고려하여 평균($\mu^n$)과 사분위수 범위($\text{IQR}^n$)를 이용해 정규화된 점수 $S^n(t)$를 산출한다.
3. **최종 점수**: 가장 큰 오차를 보이는 상위 $K_s$개 센서의 점수를 합산하여 최종 이상 점수 $s_t$를 구하며, 이를 임계값과 비교하여 이상 여부를 판단한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: SWaT(수처리 시스템), WADI(수분배 시스템), SMAP(위성 텔레메트리 데이터)의 3가지 실제 데이터셋을 사용하였다.
* **평가 지표**: Precision, Recall, F1-score를 사용하였으며, 연속적인 이상 구간 중 하나라도 탐지하면 정답으로 처리하는 Point-Adjust 전략을 적용하였다.
* **비교 대상**: PCA, KNN, AE, IF와 같은 전통적 모델부터 LSTM-VAE, OmniAnomaly, MTAD-GAN, GDN, GTA, DVGCRN 등 최신 딥러닝/그래프 모델들과 비교하였다.

### 2. 주요 결과

실험 결과, CAN은 모든 데이터셋에서 가장 높은 F1-score를 기록하며 state-of-the-art(SOTA) 성능을 보였다.

* **SWaT**: F1-score 0.9266 (GTA 대비 약 2% 향상)
* **WADI**: F1-score 0.8955 (DVGCRN 대비 약 6% 향상)
* **SMAP**: F1-score 0.9428

특히 센서 관계가 매우 복잡한 WADI 데이터셋에서 타 모델 대비 월등한 성능 향상을 보였는데, 이는 본 모델의 전역-지역 그래프 구조가 복잡한 상호 의존성을 효과적으로 포착했음을 시사한다.

### 3. 분석 및 절제 실험(Ablation Study)

* **지역 그래프 및 그래프 컨볼루션 제거**: 성능이 크게 하락하여, 변수 간 의존성 모델링이 MTAD에 필수적임을 확인하였다.
* **재구성 Decoder 제거**: 모델 성능이 유의미하게 하락하였다. 이는 재구성 작업이 Encoder가 더 좋은 시계열 표현을 학습하도록 돕는 보조 학습(Auxiliary Learning) 역할을 수행함을 의미한다.
* **파라미터 민감도**: 센서 임베딩 차원(10)과 추출 관계 수($K_m=10$), 레이어 수(3층)에서 최적의 성능을 보였다.

## 🧠 Insights & Discussion

본 논문의 강점은 단순히 예측이나 재구성 중 하나에 치중하지 않고, 두 작업을 결합하여 학습의 안정성과 표현력을 높였다는 점이다. 특히 전역 그래프를 통해 센서의 정적 타입/특성을 잡고, 지역 그래프로 실시간 상황 변화를 반영함으로써 그래프 모델의 고질적인 문제인 '정적 구조의 한계'와 '동적 구조의 불안정성'을 동시에 해결하였다.

다만, 재구성 모델이 오버피팅되기 쉽다는 이유로 추론 단계에서는 완전히 배제하고 학습 시에만 사용하는 방식을 취했다. 이는 재구성 작업이 일종의 사전 학습(Pre-training)과 유사한 역할만을 수행하고 있음을 보여준다.

비판적 관점에서 본다면, 예측과 재구성을 결합하는 방식이 단순히 손실 함수의 가중 합으로 이루어져 있어, 두 작업 간의 시너지를 극대화할 수 있는 보다 정교한 융합 메커니즘(예: 적대적 학습이나 상호 정보량 최대화 등)에 대한 탐구가 추가되었다면 더 좋은 결과가 있었을 것으로 생각된다.

## 📌 TL;DR

본 논문은 다변량 시계열 데이터의 동적 변수 관계를 포착하기 위해 **전역-지역 그래프 기반의 Coupled Attention Network(CAN)**를 제안하였다. 이 모델은 정적/동적 센서 관계와 시간적 의존성을 동시에 학습하며, **예측과 재구성 작업을 결합한 멀티레벨 인코더-디코더 구조**를 통해 강력한 표현력을 확보하였다. 실험을 통해 복잡한 센서 관계를 가진 환경(WADI 등)에서 기존 SOTA 모델들을 상회하는 성능을 입증하였으며, 향후 시계열 이상 탐지 연구에서 예측-재구성 작업의 효과적인 결합 방향성을 제시하였다.
