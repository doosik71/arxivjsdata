# Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects

Kexin Zhang, Qingsong Wen, Chaoli Zhang, Rongyao Cai, Ming Jin, Yong Liu, James Y. Zhang, Yuxuan Liang, Guansong Pang, Dongjin Song, and Shirui Pan (2024)

## 🧩 Problem to Solve

본 논문은 시계열(Time Series) 데이터 분석을 위한 자기지도학습(Self-Supervised Learning, SSL)의 전반적인 체계를 정리하고 분석하는 것을 목표로 한다. 시계열 데이터는 인간 활동 인식, 산업 결함 진단, 의료 등 다양한 실세계 시나리오에서 방대하게 생성되지만, 이를 위한 고품질의 레이블링된 데이터를 확보하는 것은 매우 많은 시간과 비용이 소요되는 작업이다.

기존의 컴퓨터 비전(CV)이나 자연어 처리(NLP) 분야에서는 SSL이 비약적인 발전을 이루었으나, 이를 시계열 데이터에 직접 적용하는 데에는 다음과 같은 고유한 어려움이 존재한다:

1. **데이터 특성의 차이**: 시계열 데이터는 계절성(Seasonality), 추세(Trend), 주파수 영역 정보 등 CV/NLP와는 다른 독특한 세만틱 특성을 가진다.
2. **데이터 증강(Data Augmentation)의 난제**: 이미지의 회전이나 크롭(Crop)과 같은 기법을 시계열에 적용할 경우, 시계열의 핵심인 시간적 의존성(Temporal Dependency)이 파괴될 위험이 크다.
3. **다변량 특성**: 대부분의 시계열 데이터는 다변량(Multivariate) 구조를 가지며, 유용한 정보가 소수의 차원에만 집중되어 있어 유의미한 특징 추출이 어렵다.

따라서 본 논문은 이러한 간극을 메우기 위해 시계열 SSL 방법론을 체계적으로 분류하고, 최신 연구 동향과 데이터셋, 그리고 향후 연구 방향을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 시계열 SSL 방법론에 대한 포괄적이고 체계적인 **Taxonomy(분류 체계)**를 제안한 것이다. 저자들은 시계열 SSL을 크게 세 가지 학습 패러다임으로 분류하고, 이를 다시 10개의 세부 카테고리로 세분화하였다.

1. **Generative-based (생성 기반)**: 데이터를 생성하거나 재구성하는 과정을 통해 표현력을 학습한다.
   - Autoregressive-based forecasting, Autoencoder-based reconstruction, Diffusion-based generation으로 구분한다.
2. **Contrastive-based (대조 기반)**: 긍정 샘플(Positive sample)과 부정 샘플(Negative sample) 간의 거리를 조절하여 특징을 학습한다.
   - Sampling, Prediction, Augmentation, Prototype, Expert knowledge contrast의 5가지 방식으로 세분화한다.
3. **Adversarial-based (적대적 기반)**: GAN(Generative Adversarial Networks) 구조를 활용하여 데이터 생성 능력을 높이거나 표현력을 강화한다.
   - Generation and Imputation, Auxiliary representation enhancement로 구분한다.

또한, 시계열 분석의 4대 주요 작업(이상치 탐지, 예측, 분류, 군집화)에 사용되는 벤치마크 데이터셋을 정리하고, 각 방법론과 작업 간의 상관관계를 분석하였다.

## 📎 Related Works

논문은 기존의 SSL 및 시계열 관련 서베이 연구들을 검토하며 본 연구의 차별점을 명시한다.

- **기존 SSL 서베이**: CV나 NLP 분야의 서베이는 매우 방대하지만, 시계열에 특화된 포괄적인 리뷰는 부족한 실정이다.
- **기존 시계열 SSL 연구**: 일부 연구(Eldele et al., Deldari et al.)가 시계열 SSL을 다루었으나, 이는 주로 대조 학습(Contrastive Learning)의 일부에만 집중되어 있으며, 생성 기반이나 적대적 기반 방법론에 대한 포괄적인 분석이 부족하다.
- **차별점**: 본 논문은 단순한 방법론 나열을 넘어, 생성-대조-적대적 학습이라는 세 가지 관점에서 10개의 세부 카테고리를 설정하고, 각 방법론의 수학적 표현, 장단점, 그리고 실제 응용 데이터셋과의 연결 고리를 상세히 분석하였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 제안한 Taxonomy에 따라 시계열 SSL의 핵심 방법론들을 설명한다.

### 1. Generative-based Methods

데이터의 일부를 통해 전체를 복원하거나 미래를 예측하는 Pretext task를 수행한다.

- **Autoregressive-based Forecasting (ARF)**: 과거 윈도우 $x_{[1:t]}$를 통해 미래 윈도우 $\hat{x}_{[t+1:t+K]}$를 예측한다.
  - 목표 함수: $\hat{x}_{[t+1:t+K]} = f(x_{[1:t]})$
  - 손실 함수: 예측값과 실제값 사이의 MSE(Mean Square Error)를 최소화한다.
    $$L = \frac{1}{K} \sum_{k=1}^{K} (\hat{x}_{[t+k]} - x_{[t+k]})^2$$
- **Autoencoder-based Reconstruction**: 입력을 저차원 표현 $z$로 인코딩한 후 다시 원래 입력으로 복원한다.
  - 과정: $z = E(x), \tilde{x} = D(z)$
  - 손실 함수: $\text{L} = \lVert x - \tilde{x} \rVert^2$
  - 변형 모델: 잡음을 추가하는 DAE(Denoising AE), 일부를 마스킹하는 MAE(Masked AE), 확률 분포를 학습하는 VAE(Variational AE) 등이 있다.
- **Diffusion-based Generation**: 데이터에 점진적으로 노이즈를 추가하는 Forward process와 이를 다시 제거하여 데이터를 생성하는 Reverse process를 학습한다. 최근 CSDI, TimeGrad 등의 모델이 시계열 결측치 보간 및 예측에 적용되고 있다.

### 2. Contrastive-based Methods

유사한 샘플은 가깝게, 서로 다른 샘플은 멀게 배치하도록 학습한다.

- **Sampling Contrast**: 인접한 시간 윈도우는 유사하고, 멀리 떨어진 윈도우는 다르다는 가정을 기반으로 한다.
- **Prediction Contrast**: 현재 컨텍스트 $c_t$와 미래 샘플 $x_{t+k}$ 사이의 상호 정보량(Mutual Information)을 최대화한다. 주로 InfoNCE 손실 함수를 사용한다.
  $$L = -\mathbb{E} \left[ \log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)} \right]$$
- **Augmentation Contrast**: 동일 샘플에 서로 다른 증강(Jittering, Scaling 등)을 적용해 두 뷰(View)를 생성하고, 이들의 유사도를 높인다.
- **Prototype Contrast**: 개별 샘플 간의 대조가 아닌, 클러스터 중심(Prototype)과의 거리를 대조하여 군집 친화적인 표현을 학습한다.
- **Expert Knowledge Contrast**: DTW(Dynamic Time Warping) 거리와 같은 도메인 지식을 활용해 긍정/부정 샘플을 더 정확하게 선택한다.

### 3. Adversarial-based Methods

GAN 구조의 생성자($G$)와 판별자($D$)의 경쟁을 통해 학습한다.

- **Generation and Imputation**: 실제와 유사한 시계열 데이터를 생성하거나, 결측치를 자연스럽게 채우는 작업에 사용된다.
- **Auxiliary Representation Enhancement**: 기본 손실 함수 $L_{base}$에 적대적 손실 $L_{adv}$를 추가하여 표현력의 강건성을 높인다.
  $$L = L_{base} + L_{adv}$$

## 📊 Results

논문은 SSL 방법론이 적용되는 4가지 주요 작업과 벤치마크 데이터셋을 정리하여 정량적 성능을 비교 분석하였다.

### 실험 설정 및 지표

- **이상치 탐지 (Anomaly Detection)**: PSM, SMD, MSL, SMAP, SWaT 데이터셋 사용. 지표로 Precision(P), Recall(R), F1-score를 사용한다.
- **예측 (Forecasting)**: ETTh, ETTm, Electricity, Weather 데이터셋 사용. 지표로 MSE, MAE를 사용한다.
- **분류 및 군집화 (Classification & Clustering)**: HAR, UCR, UEA 데이터셋 사용. 지표로 Accuracy(Acc)를 사용한다.

### 주요 분석 결과

1. **작업별 적합 방법론**:
   - **생성 기반 SSL** $\rightarrow$ **이상치 탐지 및 예측**에 적합하다. 데이터 생성 프로세스가 두 작업의 본질과 일치하기 때문이다.
   - **대조 기반 SSL** $\rightarrow$ **분류 및 군집화**에 매우 강력하다. 인스턴스 식별(Instance Discrimination) 능력이 클래스 구분 능력과 직결되기 때문이다.
2. **정량적 성능**:
   - 이상치 탐지에서는 Skip-CPC, Dcdetector와 같은 대조 학습 기반 모델이 높은 성능을 보였다.
   - 예측 작업에서는 LaST, CoST와 같이 계절성과 추세를 분리하여 학습하는(Disentangled representation) 모델이 우수한 성능을 나타냈다.
   - 분류 작업에서는 TS2Vec 등이 강세를 보였으나, 데이터셋의 특성에 따라 최적의 모델이 다르므로 단일 모델로 모든 데이터를 해결하기는 어렵다는 점이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 시계열 SSL의 방대한 문헌을 '학습 패러다임'이라는 일관된 기준으로 정리함으로써, 연구자들이 자신의 문제 정의에 따라 어떤 SSL 전략을 선택해야 하는지에 대한 명확한 가이드라인을 제공한다. 특히, 단순한 성능 비교를 넘어 생성-대조-적대적 학습이 각각 시계열의 어떤 특성(시간적 의존성, 클래스 구분, 분포 학습 등)을 포착하는지를 이론적으로 연결한 점이 돋보인다.

### 한계 및 미해결 과제

- **데이터 증강의 불확실성**: 시계열에서 '어떤 증강 기법의 조합이 최적인가'에 대한 일반적인 해답이 아직 부족하다.
- **귀납적 편향(Inductive Bias)**: 현재의 SSL은 주로 데이터 주도(Data-driven) 방식이다. 시계열의 계절성, 주기성 등 도메인 지식을 모델 구조나 손실 함수에 더 효과적으로 반영할 방법이 필요하다.
- **비정형 데이터 처리**: 불규칙한 간격으로 측정되거나 매우 희소한(Sparse) 시계열 데이터에 대한 SSL 연구가 여전히 부족하다.

## 📌 TL;DR

본 논문은 시계열 데이터 분석을 위한 자기지도학습(SSL)의 전반적인 체계를 **생성 기반, 대조 기반, 적대적 기반**의 세 가지 패러다임과 10개의 세부 카테고리로 분류한 포괄적인 서베이 논문이다. 분석 결과, **생성 기반 SSL은 예측 및 이상치 탐지**에, **대조 기반 SSL은 분류 및 군집화** 작업에 더 적합함을 밝혀냈다. 향후 시계열 SSL 연구는 최적의 데이터 증강 조합 탐색, 도메인 지식(귀납적 편향)의 통합, 그리고 거대 시계열 모델(Large Time Series Models)의 구축 방향으로 나아갈 것으로 전망된다.
