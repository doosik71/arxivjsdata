# Recurrent Auto-Encoder with Multi-Resolution Ensemble and Predictive Coding for Multivariate Time-Series Anomaly Detection

Heejeong Choi, Subin Kim, and Pilsung Kang (2022)

## 🧩 Problem to Solve

본 논문은 스마트 팩토리, 발전소, 사이버 보안과 같은 복잡한 시스템에서 발생하는 다변량 시계열 데이터(Multivariate Time-Series Data)의 이상치 탐지(Anomaly Detection) 문제를 다룬다. 실제 환경의 시계열 데이터는 매우 복잡한 비선형적 시간적 의존성(Nonlinear Temporal Dependencies)과 확률적 특성을 가지고 있어, 정상 상태의 행동 패턴을 정확하게 학습하는 것이 매우 어렵다.

특히, 실제 산업 현장에서는 이상치 데이터가 매우 드물게 발생하기 때문에 지도 학습(Supervised Learning)이 불가능하며, 따라서 정상 데이터만을 이용해 정상 패턴의 풍부한 표현(Rich Representation)을 학습하고 이를 통해 알려지지 않은 이상치를 식별해야 하는 비지도 학습(Unsupervised Learning) 관점의 접근이 필수적이다. 본 연구의 목표는 다중 해상도 앙상블(Multi-resolution Ensemble)과 예측 코딩(Predictive Coding)을 결합하여 시계열 데이터의 다중 척도 의존성을 효과적으로 포착하는 비지도 학습 기반의 이상치 탐지 모델인 RAE-MEPC를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 재구성(Reconstruction) 기반 방법론과 예측(Prediction) 기반 방법론의 장점을 결합하여 더 정보량이 많은 시계열 표현을 학습하는 것이다. 이를 위해 다음과 같은 설계를 도입하였다.

1. **Multi-resolution Ensemble Encoding**: 서로 다른 인코딩 길이를 가진 여러 개의 서브 인코더를 통해, 거시적인 글로벌 패턴(Global Patterns)과 미세한 지역적 특성(Local Characteristics)을 동시에 포착하고 이를 계층적으로 통합한다.
2. **Multi-resolution Ensemble Decoding**: 저해상도 정보가 고해상도 디코딩을 돕는 'Coarse-to-Fine' 융합 전략을 사용하여, 재구성 과정에서 발생하는 오차 누적 문제를 완화하고 재구성 성능을 높인다.
3. **Predictive Coding**: 재구성 작업 외에 미래 시점의 데이터를 예측하는 보조 작업(Auxiliary Task)을 추가함으로써, 인코더가 시간적 의존성을 더 깊게 학습하도록 유도하여 표현 학습의 품질을 향상시킨다.

## 📎 Related Works

논문에서는 기존의 시계열 이상치 탐지 방법을 크게 두 가지 범주로 분류하여 설명한다.

1. **예측 기반 방법(Prediction-based Methods)**: 과거 데이터를 통해 미래 값을 예측하고, 실제 값과의 차이가 임계치를 넘으면 이상치로 판단한다. LSTM-AD 등이 대표적이며, 주로 RNN 계열의 모델을 사용하여 시간적 특징을 학습한다.
2. **재구성 기반 방법(Reconstruction-based Methods)**: 데이터를 잠재 공간(Latent Space)으로 압축했다가 다시 복원하는 과정을 통해 정상 패턴을 학습한다. 정상 데이터로만 학습된 모델은 이상치 데이터를 제대로 복원하지 못해 재구성 오차가 커진다는 가정을 바탕으로 한다. RAE, LSTM-VAE, 그리고 GAN을 이용한 MAD-GAN 등이 이에 해당한다. 특히 GAN 기반 방법은 학습 과정의 불안정성(Mode Collapse 등)이라는 한계가 있다.

본 연구는 기존의 RAMED가 제안한 다중 해상도 디코딩 개념을 확장하여 인코딩 단계에도 다중 해상도 앙상블을 도입하였으며, 여기에 비디오 표현 학습 등에서 사용되는 Predictive Coding 개념을 시계열 데이터에 적용함으로써 기존의 단순 재구성 모델들보다 더 강력한 표현력을 갖추고자 하였다.

## 🛠️ Methodology

### 전체 시스템 구조

RAE-MEPC는 하나의 인코더(Encoder)와 두 개의 디코더(재구성 디코더 $RD$, 예측 디코더 $PD$)로 구성된다. 인코더는 입력 데이터를 다중 척도 의존성이 포함된 압축된 표현으로 매핑하며, 재구성 디코더는 입력을 복원하고, 예측 디코더는 미래 시점의 데이터를 예측한다.

### 1. Multi-Resolution Ensemble Encoding

인코더는 $K^{(E)}$개의 서브 인코더로 구성되며, 각 서브 인코더 $E_k$는 서로 다른 인코딩 길이 $T^{(E_k)}$를 가진다. 인코딩 길이는 다음과 같이 정의된다.
$$T^{(E_k)} = \lfloor \frac{1}{\tau^{k-1}} \times T \rfloor, 1 \le k \le K^{(E)}$$
여기서 $\tau$는 해상도를 결정하는 하이퍼파라미터이다. 각 서브 인코더는 원본 데이터 $X$를 다운샘플링한 하위 시퀀스 $X^{(E_k)}$를 입력으로 받아 LSTM을 통해 특징을 추출한다. 최종적으로 가장 낮은 해상도(거시적)부터 가장 높은 해상도(미세적)까지의 특징들을 MLP를 통해 계층적으로 통합하여 최종 표현 $h^{(E)}$를 생성한다.

### 2. Multi-Resolution Ensemble Decoding

재구성 디코더는 $K^{(RD)}$개의 서브 디코더로 구성되며, 인코더와 마찬가지로 서로 다른 디코딩 길이를 가진다. 이 모델은 **Coarse-to-Fine 융합 전략**을 사용하여, 더 낮은 해상도의 서브 디코더 $RD_{k+1}$의 정보를 더 높은 해상도의 서브 디코더 $RD_k$에 반영한다. 통합된 은닉 상태 $\hat{h}^{(RD_k)}_t$는 다음과 같이 계산된다.
$$\hat{h}^{(RD_k)}_t = \beta h^{(RD_k)}_{t+1} + (1-\beta) \text{MLP}^{(RD_k-RD_{k+1})}([h^{(RD_k)}_{t+1}; h^{(RD_{k+1})}_{[t/\tau]}])$$
여기서 $\beta$는 저해상도 정보의 반영 정도를 조절하는 파라미터이다. 최종 결과물은 가장 높은 해상도의 디코더 출력값을 역순으로 배치하여 얻는다.

### 3. Predictive Coding

예측 디코더 $PD$는 인코더의 최종 표현 $h^{(E)}$를 입력으로 받아, $T/2$ 시점 이후의 미래 시계열 $\vec{Y}_{pred}$를 예측하는 LSTM 구조이다. 이는 인코더가 단순히 데이터를 복제하는 것이 아니라, 데이터의 내재적인 시간적 흐름과 규칙을 학습하도록 강제하는 역할을 한다.

### 4. 손실 함수 및 학습 절차

모델은 다음 세 가지 손실 함수의 합을 최소화하는 방향으로 학습된다.
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda_{shape} \mathcal{L}_{shape} + \lambda_{pred} \mathcal{L}_{pred}$$

- **재구성 손실 ($\mathcal{L}_{recon}$)**: 입력과 재구성 출력 간의 MSE(Mean Squared Error)이다.
- **형태 강제 손실 ($\mathcal{L}_{shape}$)**: 서로 다른 해상도의 서브 디코더들이 입력 데이터와 일관된 시간적 경향성을 갖도록 유도한다. 미분 불가능한 DTW(Dynamic Time Warping) 대신 미분 가능한 **sDTW(smoothed DTW)**를 사용하여 계산한다.
- **예측 손실 ($\mathcal{L}_{pred}$)**: 예측된 미래 값과 실제 미래 값 사이의 MSE이다.

### 5. 이상치 탐지 절차

학습이 완료된 후, 검증 데이터셋(Validation Set)을 통해 정상 잔차(Residual)의 분포 $e_t \sim \mathcal{N}(\mu, \Sigma)$를 추정한다. 테스트 데이터에 대해 다음과 같이 Anomaly Score를 계산하며, 이 값이 정의된 임계치(THR)를 초과하면 이상치로 판단한다.
$$\text{Anomaly Score} = (e_t - \mu)^T \Sigma^{-1} (e_t - \mu)$$

## 📊 Results

### 실험 설정

- **데이터셋**: `power-demand` (전력 수요 데이터)와 `2D-gesture` (손동작 좌표 데이터) 두 가지 벤치마크 데이터셋을 사용하였다.
- **비교 모델**: 예측 기반(LSTM-AD), GAN 기반 재구성(MAD-GAN), 일반 재구성(EncDec-AD, RAMED) 모델들과 비교하였다.
- **평가 지표**: AUROC, AUPRC, Best F1-Score를 사용하여 성능을 측정하였다.

### 주요 결과

- **정량적 결과**: RAE-MEPC는 두 데이터셋 모두에서 대부분의 지표에서 베이스라인 모델들을 능가하였다. 특히 `2D-gesture` 데이터셋에서는 모든 지표에서 최고 성능을 기록하였으며, `power-demand` 데이터셋에서는 AUROC와 AUPRC에서 가장 우수한 성적을 보였다.
- **구성 요소 분석 (Ablation Study)**:
  - Multi-resolution ensemble encoding을 제거했을 때 모든 지표에서 성능이 하락하여, 다중 척도 특징 통합의 중요성이 입증되었다.
  - Predictive coding을 제거했을 때 역시 성능이 감소하여, 예측 작업이 풍부한 표현 학습에 기여함을 확인하였다.
- **하이퍼파라미터 민감도**: 해상도 파라미터 $\tau$가 4일 때 가장 좋은 성능을 보였으며, 예측 손실 가중치 $\lambda_{pred}$가 커질수록(최대 1까지) 성능이 향상되는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 재구성 기반의 오토인코더가 빠지기 쉬운 '단순 복제' 문제를 해결하기 위해 다중 해상도 앙상블과 예측 코딩을 도입하여 유의미한 성과를 거두었다. 특히 AUPRC 지표에서 큰 향상을 보인 것은 모델이 정밀도(Precision)와 재현율(Recall) 사이의 균형을 잘 맞추고 있으며, 오탐(False Positive)을 효과적으로 줄였음을 의미한다.

**강점**:

- 단순한 구조의 LSTM을 사용하면서도 아키텍처 설계(앙상블, 계층적 통합, 보조 작업)만으로 성능을 극대화하였다.
- sDTW를 도입하여 시계열의 전반적인 '형태'를 보존하도록 강제한 점이 효과적이었다.

**한계 및 비판적 해석**:

- 논문에서도 언급하였듯, 본 모델은 변수 간의 상관관계(Inter-correlation between variables)를 직접적으로 모델링하는 메커니즘이 부족하다. 다변량 시계열에서는 변수 간의 상호작용이 이상치 판단의 핵심인 경우가 많으므로, 향후 Graph Neural Networks(GNN) 등과의 결합이 필요해 보인다.
- 일부 구간에서 여전히 False Alarm이 발생하는 문제가 있으며, 이는 단순한 재구성 오차 기반의 탐지 방식이 갖는 근본적인 한계일 수 있다. 사후 처리(Post-processing) 단계의 개선이 필요하다.

## 📌 TL;DR

본 논문은 다변량 시계열 이상치 탐지를 위해 **다중 해상도 앙상블 인코딩/디코딩**과 **예측 코딩(Predictive Coding)**을 결합한 **RAE-MEPC** 모델을 제안한다. 이 모델은 데이터의 거시적/미세적 특징을 동시에 포착하고 미래 값을 예측하는 보조 작업을 통해 정상 패턴의 고도화된 표현을 학습하며, 실험 결과 기존의 예측 및 재구성 기반 모델들보다 우수한 탐지 성능(특히 AUPRC)을 보였다. 향후 변수 간 상관관계 모델링을 추가한다면 실제 산업 현장의 복잡한 다변량 시스템에 더욱 효과적으로 적용될 수 있을 것으로 기대된다.
