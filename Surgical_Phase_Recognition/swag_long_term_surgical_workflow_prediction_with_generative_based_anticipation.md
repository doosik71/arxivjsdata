# SWAG: Long-term Surgical Workflow Prediction with Generative-based Anticipation

Maxence Boels, Yang Liu, Prokar Dasgupta, Alejandro Granados, Sebastien Ourselin (2025)

## 🧩 Problem to Solve

본 논문은 수술 워크플로우 분석에서 기존의 수술 단계 인식(Surgical Phase Recognition)과 예측(Anticipation) 사이의 간극을 해결하고자 한다. 기존의 인식 모델들은 현재의 수술 단계나 동작을 식별하는 데에는 뛰어나지만, 수술 중 실시간으로 미래의 절차적 단계에 대한 가이드를 제공하는 능력은 부족하다.

또한, 기존의 예측 방법론들은 주로 단기적인 예측이나 단일 이벤트(예: 다음 단계의 발생 시점)를 예측하는 데 국한되어 있다. 그러나 실제 수술 워크플로우는 밀도가 높고, 반복적이며, 매우 긴 시퀀스로 이루어져 있다는 특징이 있다. 따라서 본 연구의 목표는 수술 단계 인식과 장기적인 미래 워크플로우 예측을 통합하여, 수술 중 동적인 계획 조정과 효율적인 운영실 협업을 가능하게 하는 생성 기반의 프레임워크인 SWAG(Surgical Workflow Anticipative Generation)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 수술 워크플로우의 인식과 장기 예측을 단일 생성 모델로 통합한 점에 있다. 주요 설계 아이디어는 다음과 같다.

1. **통합 생성 프레임워크**: 현재 단계 인식과 미래 시퀀스 예측을 동시에 수행하는 SWAG 모델을 제안하여, 수술 워크플로우의 시간적 연속성을 확보하였다.
2. **디코딩 전략의 비교 분석**: 단일 패스(Single-Pass, SP) 방식과 자기회귀(Auto-Regressive, AR) 방식의 두 가지 디코딩 방법론을 비교하고, 분류(Classification)와 회귀(Regression) 작업에 각각 적용하여 성능을 분석하였다.
3. **사전 지식 임베딩(Prior Knowledge Embedding)**: 훈련 데이터셋에서 추출한 클래스 전이 확률(Class Transition Probabilities)을 사용하여 미래 토큰 임베딩을 초기화함으로써 예측 정확도를 향상시켰다.
4. **R2C(Regression-to-Classification) 방법론**: 연속적인 시간 예측값(회귀 결과)을 이산적인 단계 시퀀스(분류 결과)로 변환하는 매핑 방식을 제안하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **수술 단계 인식**: CNN, RNN 및 최근의 Transformer 기반 모델(Trans-SVNet, LoViT, SKiT 등)이 개발되어 사후 비디오 분석에서 높은 성능을 보였다. 하지만 이러한 모델들은 '현재' 상태만 식별할 뿐, 미래의 이벤트를 예측하여 실시간 의사결정을 돕는 기능은 부족하다.
- **수술 워크플로우 예측**: 주로 수술 종료까지 남은 시간(RSD) 예측이나, 다음 도구/단계의 발생 시점(Next-occurrence)을 예측하는 회귀 문제로 접근하였다. 이러한 방식은 특정 시점의 단일 이벤트만 예측하므로, 고정된 시간 범위 내에서 발생하는 여러 이벤트의 시퀀스를 포착하지 못하는 맹점이 있다.

### 차별점

SWAG는 단순히 다음 이벤트의 발생 시간을 맞추는 것이 아니라, 생성 모델(Generative Model)을 통해 임의의 길이와 빈도를 가진 미래 단계 시퀀스를 직접 생성한다. 이를 통해 인식과 예측 작업을 통합하고, 수술 전체 과정에 대한 포괄적인 뷰를 제공한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

SWAG의 파이프라인은 **Vision Encoder $\rightarrow$ Windowed Self-Attention (WSA) $\rightarrow$ Compression and Pooling (CP) $\rightarrow$ Decoder** 순으로 구성된다.

1. **Vision Encoder**: 사전 학습된 ViT(Vision Transformer)를 AVT(Anticipative Video Transformer) 방식으로 미세 조정하여 각 프레임당 768차원의 임베딩을 추출한다.
2. **Windowed Self-Attention (WSA)**: 추출된 특징 시퀀스 $F_t$에 대해 너비 $W=20$의 슬라이딩 윈도우를 적용하여 지역적인 시간적 문맥을 캡처한다.
3. **Compression and Pooling (CP)**:
    - **Global Key-pooling**: 특징을 저차원 공간으로 투영한 후 누적 최대 풀링(Cumulative Max-pooling)을 통해 $M$개의 컨텍스트 토큰 $K_t$를 생성한다.
    - **Interval-pooling**: 60초 간격으로 최대 풀링을 수행하여 시간적 일관성을 유지하며 토큰을 압축한다. (주로 AR 디코더에서 사용)

### 디코더 및 예측 절차

디코더는 두 가지 방식으로 작동한다.

- **Single-Pass (SP)**: $N$개의 입력 토큰 $Q_t$를 한 번의 순전파(Forward pass)로 처리하여 $N$분 뒤까지의 미래 단계들을 동시에 예측한다.
- **Auto-Regressive (AR)**: GPT-2 구조를 사용하여 이전 예측값을 다시 입력으로 사용하는 반복적 생성 방식을 취한다.

### 핵심 방정식 및 기법

#### 1. 사전 지식 임베딩 (Prior Knowledge Embedding, SP*)

훈련 세트에서 현재 클래스 $i$가 주어졌을 때 $h_n$분 뒤에 클래스 $j$가 나타날 확률 $P(y_{t+h_n \cdot 60} = j | y_t = i)$를 계산하여 텐서 $P$를 구축한다. 미래 토큰 $q_t$는 다음과 같이 구성된다.
$$p'_t = W_p p_t + \text{bias}_p$$
$$q_t = \text{LayerNorm}(u_t + \alpha p'_t + \text{PositionalEncoding}(t))$$
여기서 $p_t$는 사전 계산된 전이 확률 벡터이며, 이를 통해 모델은 수술의 일반적인 흐름이라는 강력한 가이드라인을 가지고 예측을 시작한다.

#### 2. R2C (Regression-to-Classification)

회귀 모델이 예측한 각 클래스별 잔여 시간(Remaining Time)을 오름차순으로 정렬하고 빈(bin)으로 나누어, 이를 이산적인 단계 시퀀스 $\hat{Y}_t = \{\hat{y}_0, \hat{y}_1, \dots, \hat{y}_N\}$로 변환한다.

#### 3. 손실 함수

- **분류 작업**: Weighted Cross-Entropy Loss를 사용하여 클래스 불균형 문제를 해결한다.
- **회귀 작업**: Mean Squared Error (MSE) Loss를 사용하여 예측 시간과 실제 시간의 차이를 최소화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(C80), AutoLaparo21(AL21).
- **지표**:
  - **Weighted F1**: 클래스 불균형을 고려한 프레임 레벨 정확도.
  - **SegF1**: IoU(Intersection over Union) 임계값 $\tau=0.25$와 헝가리안 알고리즘을 사용하여 예측된 세그먼트와 실제 세그먼트의 일치도를 측정하는 세그먼트 레벨 지표.
  - **MAE (Mean Absolute Error)**: 회귀 작업에서 시간 예측 오차 측정.

### 주요 결과

1. **분류 성능**:
    - **AL21(복잡한 수술)**: SP* 모델이 F1 스코어 41.3%, SegF1 34.8%를 기록하며 가장 우수한 성능을 보였다. 복잡한 수술일수록 생성 모델의 정교한 모델링이 필수적임을 시사한다.
    - **Cholec80(정형화된 수술)**: 단순 확률 기반 베이스라인(Naive2)이 F1 39.5%로 높았으나, 세그먼트 레벨(SegF1)에서는 SWAG가 훨씬 더 일관된 시간적 연속성을 보였다.

2. **회귀 성능 (Remaining Time)**:
    - 2분 및 3분 horizon에서 SWAG-SP는 wMAE 각각 0.32분, 0.48분을 기록하며 기존의 Bayesian 및 IIA-Net 모델보다 뛰어난 성능을 보였다.
    - 특히 IIA-Net은 도구의 바운딩 박스와 세그멘테이션 맵 같은 추가 정보가 필요했지만, SWAG는 오직 단계 라벨만으로 더 좋은 성능을 냈다.

3. **RSD (Remaining Surgery Duration) 예측**:
    - 수술 전체 종료 시간 예측에서도 BD-Net을 제외한 대부분의 기존 방법론보다 우수하거나 경쟁력 있는 성능을 보였다.

## 🧠 Insights & Discussion

### 분석 및 해석

본 연구는 수술의 **'복잡도'**에 따라 최적의 예측 전략이 달라진다는 중요한 인사이트를 제공한다.

- **정형화된 워크플로우(Cholec80)**: 단계 순서가 일정하므로 단순한 전이 확률 기반 모델로도 높은 정확도를 얻을 수 있으며, 회귀 기반 접근법(R2C)이 효과적이다.
- **가변적인 워크플로우(AutoLaparo21)**: 수술자마다 절차가 다양하므로, 단순 확률보다는 복잡한 시간적 의존성을 학습할 수 있는 생성 모델(SP*)이 훨씬 더 강력한 성능을 발휘한다.

또한, 프레임 단위의 F1 스코어보다 SegF1 지표에서 SWAG가 우위에 있다는 점은, 본 모델이 단순한 라벨 맞추기를 넘어 **수술의 전체적인 흐름과 단계의 경계를 더 일관성 있게 예측**하고 있음을 의미한다.

### 한계 및 향후 과제

- **장기 예측의 어려움**: 예측 범위가 15~20분을 넘어가면 F1 스코어가 30% 미만으로 급격히 떨어진다. 이는 수술 중 발생하는 돌발 상황이나 외과 의사의 주관적 결정으로 인한 내재적 불확실성 때문이다.
- **결정론적 예측의 한계**: 현재 모델은 단일한 미래 경로만을 예측한다. 실제로는 여러 가능한 시나리오가 존재할 수 있으므로, 확률적 분포나 다중 경로를 예측하는 프레임워크로의 확장이 필요하다.

## 📌 TL;DR

본 논문은 수술 단계 인식과 장기적인 미래 워크플로우 예측을 통합한 생성 기반 프레임워크 **SWAG**를 제안하였다. 특히 **클래스 전이 확률을 이용한 사전 지식 임베딩**을 통해 예측 성능을 높였으며, 수술의 복잡도가 높을수록 단순한 확률 모델보다 생성 모델이 훨씬 유리함을 입증하였다. 이 연구는 수술 중 실시간 가이드 시스템 및 수술 로봇의 상황 인지 능력을 향상시키는 데 중요한 기초가 될 것으로 기대된다.
