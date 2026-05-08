# NVAE-GAN Based Approach for Unsupervised Time Series Anomaly Detection

Liang Xu, Liying Zheng, Weijun Li, Zhenbo Chen, Weishun Song, Yue Deng, Yongzhe Chang, Jing Xiao, Bo Yuan (2021)

## 🧩 Problem to Solve

본 논문은 단변량 시계열(Univariate Time Series) 데이터에서 비지도 학습(Unsupervised Learning) 기반의 이상치 탐지(Anomaly Detection) 문제를 해결하고자 한다. 시계열 이상치 탐지는 네트워크 모니터링, 설비 유지보수, 정보 보안 등 다양한 산업 분야에서 매우 중요하며 필수적인 작업이다.

그러나 실제 환경에서 수집되는 시계열 데이터는 다음과 같은 이유로 높은 정확도로 이상치를 탐지하기 어렵다:

1. **데이터의 노이즈 및 복잡성**: 실제 세계의 데이터는 노이즈가 많고 이상 패턴이 매우 복잡하여, 정상 샘플과 이상 샘플을 구분하거나 효과적인 잠재 특징(Hidden features)을 추출하는 것이 어렵다.
2. **레이블링된 데이터의 부족**: 지도 학습(Supervised Learning)을 적용하려면 많은 양의 이상치 레이블이 필요하지만, 실제 시계열 시나리오에서는 레이블된 데이터가 매우 제한적이고 불균형하게 분포되어 있다.
3. **기존 모델의 한계**: 통계적 모델(S-H-ESD 등)은 주기성이 있는 데이터에만 국한되거나, One-Class SVM과 같이 특징 공학(Feature Engineering)에 과도하게 의존하는 경향이 있다. 또한, 얕은 구조의 딥러닝 모델은 복잡한 시간적 상관관계를 충분히 캡처하지 못한다.

따라서 본 논문의 목표는 레이블 없이도 복잡한 시간적 특징을 효과적으로 학습하여 높은 정확도로 이상치를 탐지할 수 있는 비지도 학습 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시계열 데이터를 이미지로 변환하여 2D 특징 추출 능력이 뛰어난 심층 계층적 VAE(Hierarchical VAE) 구조를 적용하는 것이다. 이를 위해 **T2IVAE (Time series to Image VAE)** 모델을 제안하며, 주요 기여 사항은 다음과 같다:

1. **1D 시계열의 2D 이미지 변환**: 입력 시계열을 2D 이미지로 변환함으로써, 단순한 1차원 데이터보다 더 풍부한 시간적 특징과 상관관계를 모델에 제공한다.
2. **NVAE (Nouveau VAE) 구조 채택**: 단순한 VAE 대신 깊은 계층적 VAE 구조인 NVAE를 적용하여 시계열 데이터의 재구축(Reconstruction) 성능을 향상시켰다.
3. **GAN 기반의 적대적 훈련 전략**: 훈련 단계에서 Generative Adversarial Networks (GAN) 기법을 도입하여 모델의 과적합(Overfitting)을 방지하고 데이터 분포를 더 정밀하게 학습하도록 하였다.

## 📎 Related Works

논문에서는 비지도 이상치 탐지를 위한 두 가지 주요 프레임워크인 VAE와 GAN을 소개한다.

- **VAE 기반 접근법 (예: DONUT)**: 재구축 확률(Reconstruction probability)을 통해 이상치를 탐지한다. 시계열 데이터에 적합하도록 수정된 ELBO 손실 함수와 결측치 주입(Missing data injection) 등의 기법을 사용한다.
- **GAN 기반 접근법 (예: TADGAN)**: LSTM 레이어를 기반으로 생성자와 판별자를 구성하여 시계열 분포의 시간적 상관관계를 캡처한다. Cycle consistency loss를 통해 재구축 성능을 높이며, 다양한 방식으로 이상치 점수를 계산한다.
- **NVAE**: 이미지 생성 분야에서 State-of-the-art 성능을 보인 모델로, Depth-wise separable convolutions와 정규 분포의 잔차 매개변수화(Residual parameterization)를 사용하는 깊은 계층적 구조가 특징이다.

**차별점**: 기존 모델들이 1D 시계열을 직접 입력으로 사용하는 것과 달리, T2IVAE는 시계열을 2D 이미지로 인코딩하여 NVAE의 강력한 2D 특징 추출 능력을 시계열 도메인에 활용한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. Sequence 2D Encoding

1D 시계열 $X = \{s_1, s_2, \dots, s_N\}$에서 특정 시점 $k$의 이상 여부를 판단하기 위해, 윈도우 크기 $w = 2d$인 부분 시퀀스 $x_k$를 추출한다. 이후 이를 다음 두 가지 방법을 통해 2D 이미지로 변환한다.

- **Gramian Angular Field (GAF)**: 시계열 데이터를 $[-1, 1]$ 범위로 정규화한 후 극좌표계(Polar coordinates)로 변환하여 시간 단계 간의 삼각함수 합을 계산한다. 이를 통해 시간적 상관관계를 2D 행렬로 표현하며, 이상치가 존재할 경우 이미지 상에 뚜렷한 차이가 나타난다.
- **Recurrence Plots (RP)**: 원래 시계열에서 추출한 궤적(Trajectories) 간의 거리를 계산하여 이미지로 표현한다. 이상 지점이 존재하면 다른 정상 지점들과의 거리가 매우 커지므로 이미지 상에서 쉽게 식별 가능하다.

최종 입력 $\check{x}_k$는 GAF 이미지와 RP 이미지를 채널 축을 따라 결합(Concatenate)하여 생성하며, 형태는 $[B, w, w, 2]$가 된다.

### 2. Model Architecture (NVAE)

T2IVAE는 계층적 다중 스케일 인코더-디코더 구조를 가진다. 잠재 변수 $z$를 $K$개의 불연속 그룹 $z = \{z_1, z_2, \dots, z_K\}$으로 나누어 모델링한다.

- **인코더 (Bottom-up)**: 입력 이미지 $\tilde{x}$로부터 결정론적 네트워크를 통해 표현을 추출하고, top-down 네트워크를 통해 잠재 변수 $z_i$를 자기회귀적(Auto-regressively)으로 샘플링한다.
- **디코더 (Top-down)**: 인코더의 top-down 네트워크를 재사용하여 $z_i$를 샘플링하고, 최종적으로 입력 이미지 $\hat{x}$를 재구축한다.
- **Residual Cells**: 다층 컨볼루션으로 수용 영역(Reception field)을 넓혀 장기적인 상관관계를 가진 데이터를 더 잘 모델링하도록 설계되었다.

### 3. Training Strategy

#### 기본 손실 함수

모델은 재구축 손실(Reconstruction loss)과 KL 정규화 항(KL regularization term)의 합으로 학습된다.
$$\mathcal{L} = \mathcal{L}_{rec}(\tilde{x}, \hat{x}) + \mathcal{L}_{KL}(\tilde{x}, z)$$
여기서 $\mathcal{L}_{rec}$는 평균 제곱 오차(MSE)로 계산된다:
$$\mathcal{L}_{rec}(\tilde{x}, \hat{x}) = \frac{1}{2} \sum_{b,i,j,c=1}^{B,w,w,C} (\tilde{x}_{b,i,j,c} - \hat{x}_{b,i,j,c})^2$$
$\mathcal{L}_{KL}$은 잠재 변수의 사후 분포와 사전 분포를 가깝게 만들어 모델의 강건성을 높인다.

#### 적대적 훈련 (Adversarial Training)

VAE가 이상치 데이터까지 학습하여 재구축해버리는 과적합 문제를 해결하기 위해, 훈련 후반부에 GAN 기법을 도입한다. 인코더를 판별자(Discriminator)로, 디코더를 생성자(Generator)로 설정한다.

- **판별자 목표**: 잠재 변수가 실제 데이터에서 온 것인지, 재구축된 데이터나 사전 분포에서 생성된 것인지 구분하여 $\mathcal{L}_{KL}$을 최대화/최소화한다.
- **생성자 목표**: 생성된 데이터의 잠재 변수가 실제 데이터의 잠재 변수 분포와 유사해지도록 하여 $\mathcal{L}_{KL}$을 최소화한다.

### 4. Detection Strategy

1. **이상치 점수 계산**: 입력 이미지 $\check{x}_k$와 재구축 이미지 $\hat{x}_k$ 사이의 MSE를 계산하여 시점 $k$의 이상치 점수로 사용한다.
2. **적응형 임계치 (Adaptive Thresholding)**: 전체 시점 점수의 평균($\text{mean}$)과 표준편차($\text{std}$)를 계산하여 임계치를 $\text{mean} + 2 \cdot \text{std}$로 설정한다. 점수가 이 값을 넘으면 이상치로 판단한다.
3. **가지치기 (Pruning)**: 허위 양성(False Positive)을 줄이기 위해 Hundman의 방법을 변형하여 적용한다. 검출된 이상 시퀀스들을 최대 점수 기준으로 정렬한 후, 특정 하강률(Descent rate) 및 조건($\rho_i < \tau$, $m_i < 4 \cdot \text{std}$ 등)을 만족하는 지점부터 그 이후의 시퀀스들은 정상으로 간주하여 제거한다.

## 📊 Results

### 실험 설정

- **데이터셋**: NAB (Numenta Anomaly Benchmark), NASA (MSL, SMAP), 그리고 자체 수집한 네트워크 스위치 데이터.
- **평가 지표**: Overlap F1 Score (예측된 이상 구간이 실제 이상 구간과 겹치면 TP로 기록).
- **비교 모델**: DONUT (VAE 기반), TADGAN (GAN 기반), Luminol (통계 기반) 및 ARIMA, LSTM, DeepAR 등.

### 주요 결과

- **NAB 데이터셋**: $\text{T2IVAE}_{\text{GAN}}$이 **0.639**의 F1 score를 기록하며 가장 우수한 성능을 보였다.
- **NASA 데이터셋**: $\text{T2IVAE}_{\text{GAN}}$이 **0.651**을 기록하여 TADGAN에 이어 두 번째로 높은 성능을 보였다.
- **실제 데이터셋**: $\text{T2IVAE}_{\text{GAN}}$이 **0.504**의 F1 score를 기록하며 비교 모델들(DONUT 0.481, TADGAN 0.424 등)보다 우수한 성능을 보였다.

| Model | NASA (Mean) | NAB (Mean) | Total Mean |
| :--- | :---: | :---: | :---: |
| $\text{T2IVAE}_{\text{GAN}}$ | 0.651 | 0.639 | **0.645** |
| TADGAN* | 0.661 | 0.600 | 0.630 |
| DONUT | 0.526 | 0.427 | 0.476 |

## 🧠 Insights & Discussion

**강점 및 분석**:

- **2D 변환의 유효성**: 동일한 VAE 기반인 DONUT보다 T2IVAE가 월등한 성능을 보인 것은, 시계열을 2D 이미지로 변환하여 시간적 상관관계를 명시적으로 제공하고, NVAE의 강력한 2D 특징 추출 능력을 활용했기 때문이다.
- **적대적 훈련의 효과**: $\text{T2IVAE}$보다 $\text{T2IVAE}_{\text{GAN}}$의 성능이 일관되게 높게 나타났다. 이는 GAN 기반의 훈련 전략이 VAE의 고질적인 문제인 과적합을 억제하고 데이터 분포를 더 정확하게 학습하도록 도왔음을 시사한다.
- **강건성**: 노이즈가 많은 실제 네트워크 데이터셋에서도 가장 높은 성능을 기록함으로써, 제안 방법론이 실제 환경의 복잡한 데이터에서도 강건하게 작동함을 입증하였다.

**한계 및 논의**:

- **계산 복잡도**: 1D 데이터를 2D 이미지로 확장하고 깊은 계층적 VAE 구조를 사용하므로, 단순한 모델에 비해 연산 비용과 메모리 사용량이 증가했을 가능성이 크다. 이에 대한 정량적 분석은 본문에 명시되지 않았다.
- **하이퍼파라미터 민감도**: GAN 훈련의 불안정성을 해결하기 위해 VAE 훈련 후 후반부에만 적대적 훈련을 적용했는데, 이 전환 시점이나 가중치 $\alpha, \beta$ 등의 설정이 성능에 미치는 영향에 대한 세부 분석이 부족하다.

## 📌 TL;DR

본 논문은 단변량 시계열 이상치 탐지를 위해 **시계열을 2D 이미지로 변환(GAF, RP)**하고, 이를 **계층적 VAE인 NVAE**로 재구축하여 이상치를 탐지하는 **T2IVAE** 모델을 제안한다. 특히 훈련 후반부에 **GAN 기반의 적대적 학습**을 적용하여 과적합을 방지함으로써, NAB(0.639), NASA(0.651), 실제 데이터(0.504)에서 기존 모델들을 상회하는 F1 score를 달성하였다. 이 연구는 적절한 특징 공학(Image Encoding)과 고도화된 VAE 구조의 결합이 시계열 이상치 탐지 성능을 크게 향상시킬 수 있음을 보여준다.
