# Detecting Anomalies within Time Series using Local Neural Transformations

Tim Schneider, Chen Qiu, Marius Kloft, Decky Aspandi Latif, Steffen Staab, Stephan Mandt, Maja Rudolph (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 시계열 데이터 내에서 발생하는 **국소적 이상치(Local Anomalies)**를 탐지하는 것이다. 시계열 이상치 탐지는 전역적 이상치(Global Anomalies, 시계열 전체를 하나의 이상치로 간주)와 달리, 특정 타임스텝이나 짧은 구간 내에서 발생하는 이상징후를 포착해야 하므로 각 시점마다 이상치 점수(Anomaly Score)를 산출해야 하는 더 어려운 과제이다.

이 문제는 자율주행 자동차, 금융, 마케팅, 의료 진단 및 역학 조사 등 다양한 응용 분야에서 매우 중요하다. 특히 수처리 시설이나 화학 공장과 같은 사이버 물리 시스템(Cyber-Physical Systems)에서 이상치를 탐지하지 못할 경우 수백만 명에게 피해를 줄 수 있는 치명적인 상황이 발생할 수 있다.

논문의 목표는 이미지 분야에서 성공적으로 적용된 자기지도학습(Self-supervised Learning) 기반의 이상치 탐지 기법을 시계열 데이터에 맞게 확장하는 것이다. 특히 이미지와 달리 시계열 데이터는 정해진 데이터 변환(Transformation, 예: 회전, 크롭)을 정의하기 어렵다는 점을 극복하기 위해, 데이터로부터 직접 **국소적 신경망 변환(Local Neural Transformations, LNT)**을 학습하는 방법론을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시계열의 **국소적 의미론(Local Semantics)**과 **맥락적 의미론(Contextualized Semantics)**을 동시에 포착하기 위해, 표현 학습(Representation Learning)과 변환 학습(Transformation Learning)을 결합하는 것이다.

주요 기여 사항은 다음과 같다.

1. **LNT 프레임워크 제안**: 시계열 표현 학습과 국소적 변환 학습을 통합한 새로운 자기지도학습 기반의 이상치 탐지 방법론을 개발하였다.
2. **새로운 손실 함수 도입**: 표현 학습을 위한 CPC(Contrastive Predictive Coding) 손실과 변환 학습을 위한 DDCL(Dynamic Deterministic Contrastive Loss)을 결합하여 학습시킨다.
3. **이론적 분석**: DDCL 단독 학습 시 발생할 수 있는 매니폴드 붕괴(Manifold Collapse) 현상을 이론적으로 증명하였으며, 이를 방지하기 위해 CPC 손실이 정규화(Regularization) 역할을 수행해야 함을 입증하였다.
4. **성능 검증**: 사이버 물리 시스템 데이터셋(SWaT, WaDi)과 음성 데이터셋(LibriSpeech)을 통해 기존의 딥러닝 기반 이상치 탐지 방법론보다 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

시계열 이상치 탐지를 위한 기존의 딥러닝 접근 방식은 크게 네 가지로 분류된다.

- **시퀀스 예측(Sequence Forecasting)**: 과거 데이터를 통해 다음 스텝을 예측하고, 실제 값과의 예측 오차를 이상치 점수로 사용한다. (예: LSTM, TCN 기반 모델)
- **오토인코더(Autoencoders)**: 정상 데이터로 학습된 모델의 재구성 오차(Reconstruction Error)를 활용한다. (예: LSTM-AE, TCN-AE)
- **생성 모델(Deep Generative Models)**: VAE나 GAN을 사용하여 데이터의 분포를 학습하고, 생성 확률이나 판별자의 오차를 활용한다.
- **기타 방법**: One-class 분류기나 그래프 신경망(GDN) 등을 사용하여 특징 공간에서의 밀도를 측정하거나 변수 간 관계를 모델링한다.

최근에는 데이터 증강(Data Augmentation)을 이용한 자기지도학습 기반의 이상치 탐지가 주목받고 있다. 하지만 시계열 데이터는 이미지처럼 사람이 직접 효율적인 변환(Rotation 등)을 설계하기 어렵다. 기존 연구(Qiu et al., 2021)에서는 변환을 학습하는 방식을 제안했으나, 이는 시퀀스 전체의 이상 여부를 판단하는 전역적 탐지에 치중되어 있었으며, 이를 국소적 탐지에 적용할 경우 사소한(Trivial) 변환만을 학습하게 되어 성능이 저하되는 문제가 있었다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

LNT는 크게 **특징 추출기(Encoder)**와 **특징 변환기(Local Neural Transformations)**라는 두 가지 구성 요소로 이루어져 있다. 입력 시퀀스가 들어오면 인코더가 각 시점의 임베딩 $z_t$를 생성하고, 이를 다시 여러 개의 신경망 변환기 $T_l(\cdot)$에 통과시켜 다양한 잠재 뷰(Latent Views)를 생성한다.

### 2. 주요 구성 요소 및 학습 절차

#### (1) 국소 시계열 표현 (Local Time Series Representations)

인코더 $g_{\text{enc}}$는 입력 데이터를 국소 잠재 표현 $z_t$로 매핑하며, 이는 다시 자기회귀(Autoregressive) 모듈 $g_{\text{ar}}$을 통해 맥락 벡터(Context Vector) $c_t$로 요약된다. 이때 사용되는 **CPC(Contrastive Predictive Coding) 손실 함수**는 다음과 같다.

$$L_{\text{CPC}} = -\mathbb{E}_{X \sim D} \left[ \log \frac{\exp(z_{t+k}^T W_k c_t)}{\sum_{j \in X} \exp(z_j^T W_k c_t)} \right]$$

여기서 $W_k$는 $k$-스텝 미래 예측을 위한 선형 변환 행렬이다. 이 손실 함수는 맥락 표현 $c_t$가 주변의 국소 표현 $z_{t+k}$를 잘 예측하도록 유도하여, 데이터의 유용한 지역적 의미론을 학습하게 한다.

#### (2) 국소 신경망 변환 (Local Neural Transformations)

학습된 $z_t$는 $L$개의 신경망 $T_l(\cdot)$을 통해 서로 다른 뷰 $z_t^{(l)} = T_l(z_t)$로 변환된다. 이 변환된 뷰들은 **DDCL(Dynamic Deterministic Contrastive Loss)**을 통해 학습된다. DDCL은 각 뷰가 다양하면서도 의미 있는 정보를 담도록 다음과 같은 메커니즘을 가진다.

$$\ell^{(k,l)}_t (x_{\le t}) = -\log \frac{h(z_t^{(l)}, W_k c_{t-k})}{h(z_t^{(l)}, W_k c_{t-k}) + \sum_{m \neq l} h(z_t^{(l)}, z_t^{(m)})}$$

여기서 $h(z_i, z_j) = \exp \left( \frac{z_i^T z_j}{\|z_i\|\|z_j\|} \right)$는 지수화된 코사인 유사도이다.

- **Pull (분자)**: 변환된 뷰 $z_t^{(l)}$를 맥락 정보 $W_k c_{t-k}$에 가깝게 끌어당겨 맥락적 의미론을 갖게 한다.
- **Push (분모)**: 서로 다른 변환 뷰들 $z_t^{(l)}$과 $z_t^{(m)}$을 서로 밀어내어 뷰의 다양성을 확보한다.

최종 학습 목적 함수는 두 손실 함수의 가중 합으로 정의된다.
$$L = L_{\text{CPC}} + \lambda \cdot L_{\text{DDCL}}$$

### 3. 이상치 점수 산출 (Scoring)

학습이 완료된 후, 테스트 데이터에 대해 특정 시점 $t$에서의 DDCL 기여도를 계산하여 이상치 점수 $\ell_t$로 사용한다.

$$\ell_t (x_{\le t}) = \sum_{k=1}^K \sum_{l=1}^L \ell^{(k,l)}_t (x_{\le t})$$

점수가 높을수록 해당 시점의 데이터가 정상 패턴에서 벗어난 이상치일 가능성이 높음을 의미한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SWaT (수처리 시스템), WaDi (수분배 시스템), LibriSpeech (음성 데이터, 인위적 이상치 추가).
- **기준선(Baselines)**: Isolation Forest, PCA, LSTM-VAE, MAD-GAN, GDN, THOC 등 고전적 방법부터 최신 딥러닝 모델까지 광범위하게 비교하였다.
- **측정 지표**: $F_1\text{-score}$, ROC-AUC.

### 2. 정량적 결과

- **SWaT 데이터셋**: LNT는 $F_1\text{-score}$ 88.65%를 기록하며 비교 대상 중 가장 높은 성능을 보였다.
- **WaDi 데이터셋**: $F_1$ 점수와 정밀도(Precision) 면에서는 GDN과 유사하거나 약간 낮았으나, **재현율(Recall)** 면에서 60.92%로 가장 높은 수치를 기록하였다. 이는 미션 크리티컬한 시스템에서 False Negative를 줄이는 것이 중요하다는 점에서 매우 유의미한 결과이다.
- **LibriSpeech 데이터셋**: ROC-AUC 기준 0.93을 달성하여 LSTM(0.58)이나 THOC(0.82) 대비 압도적인 성능 향상을 보였다. 특히 복잡한 동적 특성을 가진 음성 데이터에서 LNT의 변환 기반 대조 학습이 효과적임을 입증하였다.

### 3. 정성적 결과 및 시각화

별도의 디코더를 학습시켜 잠재 공간의 변환 결과를 데이터 공간으로 복원하여 시각화한 결과, LNT가 특정 채널의 **지연(Delay)**을 조정하는 변환을 스스로 학습했음을 확인하였다. 이는 LNT가 단순한 노이즈가 아니라 데이터의 의미 있는 물리적 특성을 포착하고 있음을 보여준다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 타당성

본 논문의 가장 큰 강점은 표현 학습(CPC)과 변환 학습(DDCL)의 결합 필요성을 이론적으로 증명한 점이다. **Theorem 1**을 통해 DDCL만으로 학습할 경우 모든 입력을 동일한 값으로 매핑하는 상수 인코더로 수렴하는 **매니폴드 붕괴(Manifold Collapse)**가 발생함을 보였다. 따라서 CPC는 단순한 보조 작업이 아니라, 모델이 유의미한 특징 공간을 형성하도록 강제하는 필수적인 정규화 장치이다.

### 2. 맥락적 의미론의 중요성

LNT는 단순한 지역적 윈도우 내의 정보뿐만 아니라, 다양한 시간 간격($k$)의 맥락 벡터 $c_{t-k}$를 대조함으로써 **맥락적 의미론(Contextualized Semantics)**을 확보한다. 이는 윈도우 내에서는 정상이지만 긴 시간 흐름으로 보았을 때 이상한 패턴을 잡아낼 수 있게 하며, 이것이 CPC 기반의 단순 이상치 탐지보다 LNT가 우수한 성능을 보이는 이유이다.

### 3. 한계 및 비판적 해석

시각화 결과가 의미 있게 도출되었으나, 저자들도 인정하듯 이는 매우 고수준(High-level)의 해석이며 실제 도메인 지식과 결합하여 구체적인 이상 원인을 진단하기에는 여전히 한계가 있다. 또한, 변환기 $T_l$의 개수 $L$이나 하이퍼파라미터 $\lambda$에 대한 민감도 분석이 더 상세히 이루어졌다면 방법론의 강건성을 더 확신할 수 있었을 것이다.

## 📌 TL;DR

본 논문은 시계열 데이터의 국소적 이상치 탐지를 위해 **잠재 공간에서의 신경망 변환을 학습하는 LNT(Local Neural Transformations)** 방법을 제안한다. 국소적 표현을 학습하는 **CPC 손실**과 뷰의 다양성 및 맥락을 학습하는 **DDCL 손실**을 함께 사용하여 매니폴드 붕괴를 방지하고 성능을 극대화하였다. 실험 결과, 특히 재현율(Recall)과 복잡한 시계열(음성 등) 탐지에서 기존 모델들을 압도하는 성능을 보였으며, 이는 시계열 AD에서 표현 학습과 변환 학습의 결합이 필수적임을 시사한다.
