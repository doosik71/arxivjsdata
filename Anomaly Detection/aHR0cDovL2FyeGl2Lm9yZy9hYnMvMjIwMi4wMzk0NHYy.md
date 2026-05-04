# Detecting Anomalies within Time Series using Local Neural Transformations

Tim Schneider, Chen Qiu, Marius Kloft, Decky Aspandi Latif, Steffen Staab, Stephan Mandt, Maja Rudolph (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 시계열 데이터 내에서 발생하는 **국소적 이상치 탐지(Local Anomaly Detection)**이다. 시계열 이상치 탐지는 크게 시계열 전체를 하나의 점수로 평가하는 글로벌 이상치 탐지와, 각 타임스텝별로 이상치 점수를 부여하는 국소적 이상치 탐지로 나뉜다.

이 문제는 자율주행 자동차, 금융, 마케팅, 의료 진단 및 역학 조사 등 다양한 도메인에서 필수적이다. 특히 수처리 시설이나 화학 공장과 같은 사이버 물리 시스템(Cyber-Physical Systems)에서 이상치를 탐지하지 못할 경우 수백만 명에게 심각한 피해를 줄 수 있기 때문에 매우 중요하다.

기존의 딥러닝 기반 이상치 탐지(AD)는 주로 이미지와 같은 고차원 데이터에서 큰 성과를 거두었으나, 시계열 데이터는 복잡한 시간적 의존성을 가지고 있으며 데이터의 다양성이 매우 높아 동일한 수준의 성능을 내기 어렵다. 특히, 이미지에서는 회전, 자르기 등 강력한 데이터 변형(Transformation) 기법을 사용할 수 있지만, 시계열 데이터에서는 이와 같이 효과적인 수동 설계 변형(Hand-crafted transformation)을 정의하기 어렵다는 점이 주요한 한계로 지적된다. 따라서 본 논문의 목표는 데이터로부터 직접 로컬 변형을 학습하여 시계열 내의 이상치를 효과적으로 탐지하는 **Local Neural Transformations (LNT)** 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **표현 학습(Representation Learning)**과 **변형 학습(Transformation Learning)**을 결합하여, 잠재 공간(Latent Space) 내에서 데이터의 다양한 뷰(View)를 생성하고 이를 통해 이상치를 탐지하는 것이다.

핵심 기여 사항은 다음과 같다:

1. **LNT 프레임워크 제안**: 시계열 표현 학습과 로컬 변형 학습을 통합한 새로운 자기지도 학습(Self-supervised learning) 방법론을 제시하였다.
2. **새로운 손실 함수 DDCL 도입**: 변형된 뷰들이 서로 다양하면서도 의미 있는 정보를 담도록 강제하는 Dynamic Deterministic Contrastive Loss (DDCL)를 제안하였다.
3. **이론적 분석**: DDCL만 사용할 경우 발생할 수 있는 Manifold Collapse(표현 붕괴) 문제를 수학적으로 증명하고, 이를 방지하기 위해 Contrastive Predictive Coding (CPC) 손실 함수가 필수적인 정규화 역할을 함을 입증하였다.
4. **실증적 검증**: 사이버 물리 시스템(SWaT, WaDi) 및 음성 데이터(LibriSpeech) 벤치마크를 통해 기존의 강력한 베이스라인 모델들보다 우수한 성능을 보임을 확인하였다.

## 📎 Related Works

논문에서는 시계열 이상치 탐지 방법론을 다음과 같이 분류하여 설명한다:

- **시퀀스 예측 기반**: 다음 타임스텝의 값을 예측하고, 실제 값과의 오차를 이상치 점수로 사용한다. (예: RNN, TCN 기반 예측)
- **오토인코더(Autoencoder) 기반**: 정상 데이터로 학습된 모델의 재구성 오차(Reconstruction Error)를 이용한다.
- **생성 모델 기반**: VAE, GAN 등을 사용하여 데이터의 분포를 학습하고, 생성 확률이나 판별자의 오차를 활용한다.
- **기타**: 그래프 기반 어텐션 메커니즘이나 하이퍼스피어 분류기(Hypersphere Classifier)를 사용하는 방식이 있다.

**기존 접근 방식과의 차별점**:
기존의 자기지도 학습 기반 AD는 이미지 분야에서 주로 발전하였으며, 수동으로 설계된 변형(Rotation 등)에 의존했다. 최근 시계열에서도 Neural Transformation을 학습하려는 시도가 있었으나, 이는 주로 전체 시퀀스를 이상치로 판별하는 글로벌 AD에 치중되어 있었다. LNT는 이를 국소적(sub-sequence level) 수준으로 확장하였으며, 데이터 레벨이 아닌 **잠재 공간(Latent Space)**에서 변형을 수행함으로써 로컬 세만틱과 컨텍스트 세만틱을 모두 포착한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

LNT는 크게 **특징 추출기(Encoder)**와 **특징 변형기(Local Neural Transformations)**의 두 가지 구성 요소로 이루어져 있다. 입력 시퀀스가 들어오면 인코더가 각 타임스텝의 임베딩을 생성하고, 학습된 신경망 기반 변형기들이 이 임베딩을 받아 서로 다른 잠재 뷰(Latent Views)를 생성한다.

### 주요 구성 요소 및 학습 절차

#### 1. 로컬 시계열 표현 학습 (CPC)

인코더 $g_{enc}$는 입력 데이터 $x_t$를 잠재 표현 $z_t$로 매핑하며, 오토레그레시브(Autoregressive) 모듈 $g_{ar}$는 이를 컨텍스트 벡터 $c_t$로 요약한다. 이때 **Contrastive Predictive Coding (CPC)** 손실 함수를 사용하여 $c_t$가 미래의 로컬 표현 $z_{t+k}$를 잘 예측하도록 학습한다.

$$L_{CPC} = -\mathbb{E}_{X \sim \mathcal{D}} \left[ \log \frac{\exp(z_{t+k}^T W_k c_t)}{\sum_{j \in X} \exp(z_j^T W_k c_t)} \right]$$

여기서 $W_k$는 $k$-스텝 미래 예측을 위한 선형 변환 행렬이다. 이는 인코더가 데이터의 로컬 세만틱을 잘 학습하도록 돕는다.

#### 2. 로컬 신경망 변형 및 DDCL

LNT는 $L$개의 신경망 변형기 $T_l(\cdot)$를 통해 $z_t$로부터 다양한 뷰 $z_t^{(l)} = T_l(z_t)$를 생성한다. 이 뷰들은 **Dynamic Deterministic Contrastive Loss (DDCL)**를 통해 학습된다.

DDCL의 개별 기여도는 다음과 같이 정의된다:
$$\ell_t^{(k,l)}(x_{\le t}) = -\log \frac{h(z_t^{(l)}, W_k c_{t-k})}{h(z_t^{(l)}, W_k c_{t-k}) + \sum_{m \neq l} h(z_t^{(l)}, z_t^{(m)})}$$
여기서 $h(z_i, z_j) = \exp \frac{z_i^T z_j}{\|z_i\| \|z_j\|}$는 지수화된 코사인 유사도이다.

**DDCL의 작동 원리 (Push & Pull)**:

- **Pull (분자)**: 변형된 뷰 $z_t^{(l)}$가 컨텍스트 정보 $W_k c_{t-k}$와 가까워지도록 하여, 뷰가 의미 있는(Semantic) 정보를 담게 한다.
- **Push (분모)**: 서로 다른 변형기들에 의해 생성된 뷰 $z_t^{(l)}$와 $z_t^{(m)}$가 서로 멀어지도록 하여, 생성된 뷰들의 다양성(Diversity)을 확보한다.

전체 손실 함수는 다음과 같이 결합된다:
$$L = L_{CPC} + \lambda \cdot L_{DDCL}$$

### 추론 및 이상치 점수 산출

학습이 완료된 후, 테스트 시퀀스에 대해 각 타임스텝 $t$의 이상치 점수 $\ell_t$를 다음과 같이 계산한다:
$$\ell_t(x_{\le t}) = \sum_{k=1}^K \sum_{l=1}^L \ell_t^{(k,l)}(x_{\le t})$$
점수가 높을수록 해당 시점의 데이터가 정상 범주에서 벗어난 이상치일 확률이 높다고 판단한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - **WaDi**: 수분배 네트워크 데이터 (112차원, 1Hz 샘플링)
  - **SWaT**: 수처리 공정 데이터 (51차원 $\rightarrow$ 45차원 필터링)
  - **LibriSpeech**: 음성 데이터 (인위적인 사인파 노이즈를 추가하여 이상치 생성)
- **비교 대상**: Isolation Forest, PCA, LSTM 예측 기반, GDN, DeepSVDD, DAGMM, MAD-GAN, BeatGAN, THOC 등 다양한 클래스의 AD 알고리즘.
- **측정 지표**: $F_1$ score, ROC-AUC.

### 주요 결과

1. **SWaT 데이터셋**: LNT는 $F_1$ score **88.65%**를 기록하며 비교 대상 중 가장 높은 성능을 보였다.
2. **WaDi 데이터셋**: $F_1$ score는 GDN 등과 유사하거나 낮을 수 있으나, **Recall(재현율)이 60.92%**로 가장 높게 나타났다. 이는 미션 크리티컬한 시스템에서 미탐지(False Negative)를 줄이는 것이 중요하다는 점에서 매우 유의미한 결과이다.
3. **LibriSpeech 데이터셋**: ROC-AUC **0.93**을 달성하며 LSTM 등 딥러닝 베이스라인을 압도하였다. 복잡한 시간적 역동성을 가진 음성 데이터에서도 LNT의 변형 학습이 효과적임을 입증하였다.
4. **CPC 변형 모델과의 비교**: 단순히 CPC 손실을 점수로 사용하거나 CPC 특징을 OC-SVM에 넣는 방식보다, LNT(CPC + DDCL)의 성능이 일관되게 우수하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 타당성

본 논문은 $L_{CPC}$와 $L_{DDCL}$의 결합이 단순한 성능 향상을 넘어 이론적으로 필수적임을 보였다. **Theorem 1**을 통해 DDCL만으로 학습할 경우 인코더가 상수를 출력하는 **Manifold Collapse**가 발생함을 증명하였으며, CPC가 이를 방지하는 정규화 장치 역할을 함을 명시하였다.

또한, LNT는 두 가지 세만틱을 모두 포착한다:

- **Local Semantics**: 현재 타임 윈도우 내의 신호 특성 (CPC가 담당).
- **Contextualized Semantics**: 현재 윈도우가 더 긴 시간 지평(Time Horizon)의 전체 흐름 속에서 어떻게 위치하는가 (DDCL이 담당).

### 해석 가능성

학습된 변형기들을 데이터 공간으로 디코딩하여 시각화한 결과, LNT가 단순한 노이즈를 만드는 것이 아니라 **신호의 지연(Delay)을 조절**하는 등의 의미 있는 변형을 학습했음을 확인하였다. 이는 도메인 지식 없이도 데이터로부터 유용한 증강(Augmentation) 방식을 스스로 찾아낼 수 있음을 시사한다.

### 한계 및 논의

현재의 해석은 매우 고수준(High-level)이며, 실제 응용 단계에서 각 변형이 정확히 어떤 물리적 의미를 갖는지 상세히 분석하는 데는 한계가 있다. 하지만 수동 설계된 변형이 없는 시계열 분야에서 학습 가능한 변형을 도입했다는 점은 큰 진전이다.

## 📌 TL;DR

본 논문은 시계열 데이터의 국소적 이상치 탐지를 위해, 표현 학습(CPC)과 잠재 공간 내 변형 학습(DDCL)을 통합한 **Local Neural Transformations (LNT)** 프레임워크를 제안한다. 이론적으로 Manifold Collapse를 방지하는 구조를 설계하였으며, 실험을 통해 특히 재현율(Recall)과 ROC-AUC 측면에서 기존 딥러닝 모델보다 우수한 성능을 입증하였다. 이 연구는 수동적인 데이터 증강 없이도 데이터로부터 직접 유용한 변형을 학습하여 복잡한 시계열 시스템의 이상치를 탐지할 수 있는 새로운 방향성을 제시한다.
