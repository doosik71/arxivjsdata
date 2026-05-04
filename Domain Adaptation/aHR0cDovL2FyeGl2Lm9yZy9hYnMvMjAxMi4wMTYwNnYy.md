# Domain Adaptation with Incomplete Target Domains

Zhenpeng Li, Jianan Jiang, Yuhong Guo, Tiantian Tang, Chengxiang Zhuo, Jieping Ye (2023)

## 🧩 Problem to Solve

본 논문은 타겟 도메인(Target Domain)의 데이터가 불완전한 상태, 즉 일부 피처 값이 누락된 경우의 도메인 적응(Domain Adaptation, DA) 문제를 해결하고자 한다.

일반적인 도메인 적응 연구들은 소스 도메인(Source Domain)과 타겟 도메인의 데이터가 모두 완전히 관측되었다고 가정한다. 그러나 실제 환경에서는 데이터 수집의 어려움으로 인해 누락된 데이터(Missing Data)가 빈번하게 발생한다. 예를 들어, 서비스 플랫폼의 신규 사용자는 가입 과정에서 필수 정보만 입력하고 선택 항목을 건너뛰는 경우가 많으며, 이러한 데이터의 불완전성은 개인화 추천이나 광고 전략의 성능을 저하시키는 원인이 된다.

따라서 본 논문의 목표는 타겟 도메인의 데이터가 부분적으로만 관측된 상황에서, 레이블이 풍부한 소스 도메인의 지식을 효과적으로 전이하여 타겟 도메인에 최적화된 분류기(Classifier)를 학습시키는 것이다. 특히, 본 연구는 두 도메인의 피처 공간이 동일한 Homogeneous 설정뿐만 아니라 서로 다른 Heterogeneous 설정까지 모두 지원하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터 보간(Imputation)과 도메인 적응을 하나의 엔드-투-엔드(End-to-End) 네트워크 내에서 동시에 수행하는 **IDIAN (Incomplete Data Imputation based Adversarial Network)** 모델을 제안한 것이다.

단순히 누락된 값을 0으로 채우거나 무시하는 기존 방식 대신, 타겟 도메인의 관측된 피처 간 상관관계를 이용하여 누락된 값을 복원하는 Generator를 도입하였다. 또한, 보간된 데이터가 도메인 적응 과정에서 유효하게 작용하도록 Autoencoder 기반의 피처 공간 통합, 클래스 간 정렬을 위한 Contrastive Loss, 그리고 분포 차이를 줄이기 위한 Adversarial Learning을 유기적으로 결합하였다.

## 📎 Related Works

### 1. Learning with Incomplete Data

누락된 데이터를 처리하기 위해 초기에는 EM 알고리즘이나 MICE와 같은 통계적 방법이 사용되었으나, 이는 모델 구조에 대한 사전 지식이 필요하거나 계산 복잡도가 높다는 한계가 있다. 최근에는 GAIN, Ambient-GAN, MisGAN과 같은 GAN 기반의 딥러닝 보간 방법들이 제안되었지만, 이들은 주로 (준)지도 학습 환경에 집중되어 있으며 도메인 적응 상황을 고려하지 않았다.

### 2. Domain Adaptation

DANN, CDAN과 같은 적대적 도메인 적응 방식과 MMD 기반의 분포 정렬 방식들이 연구되어 왔다. 이러한 방법들은 소스 도메인의 레이블을 활용해 타겟 도메인과 피처 분포를 일치시켜 일반화 성능을 높인다. 그러나 앞서 언급했듯이, 기존의 모든 도메인 적응 방법들은 두 도메인의 데이터가 완전히 관측되었다는 가정을 전제로 하고 있어, 실제 데이터의 불완전성을 처리하지 못한다는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

IDIAN은 크게 세 가지 주요 모듈로 구성된다.

1. **Incomplete Data Generator ($G_i$):** 타겟 도메인의 누락된 피처 값을 채우는 역할.
2. **Domain-specific Autoencoders ($G_s, De_s, G_t, De_t$):** 소스와 타겟의 피처를 통합된 공간으로 매핑하고 정보 손실을 방지하는 역할.
3. **Adversarial Domain Adapter ($G, D, F$):** 통합된 공간에서 도메인 간 분포를 정렬하고 최종 분류를 수행하는 역할.

### 2. 상세 구성 요소 및 작동 원리

#### (1) 데이터 보간 (Incomplete Data Imputation)

타겟 도메인의 인스턴스 $x_t$, 마스크 벡터 $m_t$(관측됨: 1, 누락됨: 0), 그리고 노이즈 $\epsilon$을 입력으로 받아 누락된 부분을 채운다. 보간된 인스턴스 $\hat{x}_t$는 다음과 같이 계산된다.

$$\hat{x}_t = G_i(x_t, m_t, \epsilon) = x_t \odot m_t + \hat{G}_i(x_t \odot m_t + \epsilon \odot \bar{m}_t) \odot \bar{m}_t$$

여기서 $\bar{m}_t = 1 - m_t$이며, $\odot$은 Hadamard product를 의미한다. 이 식은 원래 관측된 값은 유지하면서 누락된 부분($\bar{m}_t$)만 보간 네트워크 $\hat{G}_i$를 통해 채우도록 보장한다.

#### (2) Autoencoder를 통한 피처 공간 통합

Heterogeneous 피처 공간을 처리하기 위해 각 도메인 전용 추출기 $G_s, G_t$와 디코더 $De_s, De_t$를 사용한다. 재구성 손실(Reconstruction Loss) $L_{AE}$를 최소화하여 피처 추출 과정에서 핵심 정보가 유지되도록 한다.

$$L_{AE} = \frac{1}{n_s} \sum_{i=1}^{n_s} \|De_s(G_s(x_{s_i})) - x_{s_i}\|^2 + \frac{1}{n_t} \sum_{i=1}^{n_t} \|De_t(G_t(\hat{x}_{t_i})) - \hat{x}_{t_i}\|^2$$

#### (3) 도메인 간 Contrastive Loss

동일한 클래스에 속하는 소스와 타겟 인스턴스가 통합 피처 공간에서 가깝게 위치하도록 유도한다. 클래스 레이블 $y$에 따라 거리 함수 $L_{dis}$를 다음과 같이 정의한다.

$$L_{dis} = \begin{cases} \|f_i - f_j\|^2 & \text{if } \delta(y_i, y_j) = 1 \\ \max(0, \rho - \|f_i - f_j\|^2) & \text{if } \delta(y_i, y_j) = 0 \end{cases}$$

이를 통해 클래스 내 거리는 좁히고, 클래스 간 거리는 마진 $\rho$ 이상으로 벌려 변별력을 높인다.

#### (4) Adversarial Feature Alignment

최종적으로 공통 피처 추출기 $G$, 도메인 판별기 $D$, 분류기 $F$를 통해 분포를 정렬한다. 판별기 $D$는 도메인을 구분하려 하고, $G$는 $D$를 속이도록 학습되어 두 도메인의 분포 차이를 줄인다.

$$L_{adv} = -\frac{1}{n_s} \sum_{i=1}^{n_s} \log D(G(f_{s_i})) - \frac{1}{n_t} \sum_{j=1}^{n_t} \log(1 - D(G(f_{t_j})))$$

분류기 $F$는 소스와 타겟의 레이블된 데이터를 사용하여 Cross-entropy 손실 $L_{cls}$를 최소화하도록 학습된다.

### 3. 전체 학습 목표

최종 목적 함수는 다음과 같이 정의되며, 이를 통해 엔드-투-엔드로 학습한다.

$$\min_{G_i, G_s, G_t, G, F, De_s, De_t} \max_{D} L(\Theta) = L_{cls} + \beta L_{AE} + \gamma L_{cont} - \lambda L_{adv}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋:** MNIST, MNIST-M, SVHN, SYN, USPS 등 5종의 숫자 인식 데이터셋을 조합하여 6가지 도메인 적응 태스크 구성.
- **시뮬레이션:** 타겟 도메인의 피처 값을 20%~80%까지 무작위로 누락시켜 불완전한 환경을 조성.
- **실제 데이터:** 라이드 헤일링(Ride-hailing) 서비스의 신규 사용자 예측 데이터 사용 (피처 누락률 약 89%, Heterogeneous 공간).
- **비교 대상:** Target only, DANN, CDAN.

### 2. 정량적 결과

- **숫자 인식 태스크:** 타겟 도메인의 데이터 누락률이 높아질수록 IDIAN의 성능 우위가 뚜렷하게 나타났다. 특히 누락률 40% 설정에서 IDIAN은 DANN과 CDAN보다 유의미하게 높은 정확도를 보였다 (예: MNIST $\to$ MNIST-M에서 IDIAN 0.213 vs CDAN 0.176).
- **실제 데이터 태스크:** 매우 높은 누락률(89%) 환경에서도 IDIAN이 AUC, ACC, F1-score 등 모든 지표에서 가장 우수한 성능을 기록했다. 특히 F1-score 기준 baseline 대비 3.8% 향상된 결과를 보였다.

### 3. Ablation Study

- **Imputation 제거:** 데이터 누락률이 높을수록 성능이 급격히 하락하여, 보간 모듈의 중요성이 입증되었다.
- **$L_{AE}$ 또는 $L_{cont}$ 제거:** 두 손실 함수를 제거했을 때 모두 성능이 저하되었으며, 이는 정보 보존과 클래스 간 정렬이 도메인 적응에 필수적임을 시사한다.

## 🧠 Insights & Discussion

본 논문은 이론적인 도메인 적응 연구와 실제의 '지저분한(dirty)' 데이터 사이의 간극을 메우려는 시도를 하였다. 특히 데이터 보간을 단순한 전처리 단계가 아닌, 적대적 학습 루프 내의 최적화 대상으로 포함시킨 점이 강점이다. 이를 통해 보간된 값이 단순히 통계적으로 그럴듯한 값이 아니라, 분류 성능과 도메인 정렬에 기여하는 방향으로 학습되도록 유도하였다.

다만, 본 모델은 타겟 도메인에 아주 적은 양이라도 레이블된 데이터($D_T^l$)가 존재해야 하는 Semi-supervised 설정을 가정하고 있다. 완전히 레이블이 없는 Unsupervised DA 환경에서 본 모델이 어떻게 작동할지에 대해서는 명시적으로 언급되지 않았다. 또한, Heterogeneous 공간 처리를 위해 별도의 추출기를 두는 방식은 유효하지만, 두 도메인 간의 피처 의미론적 관계(semantic relationship)에 대한 사전 정보 없이 학습에만 의존한다는 점이 한계로 작용할 수 있다.

## 📌 TL;DR

- **주요 기여:** 타겟 도메인 데이터가 누락된 상황을 해결하기 위해, 데이터 보간-피처 통합-분포 정렬을 통합한 **IDIAN** 프레임워크를 제안함.
- **핵심 메커니즘:** Generator를 통한 피처 보간 $\to$ Autoencoder 및 Contrastive Loss를 통한 공간 통합 $\to$ Adversarial Learning을 통한 도메인 정렬.
- **결과 및 의의:** 시뮬레이션 데이터와 실제 라이드 헤일링 데이터 모두에서 기존 DANN, CDAN보다 우수한 성능을 보였으며, 특히 데이터 누락이 심할수록 그 효과가 큼. 실제 환경의 불완전한 데이터를 다루는 도메인 적응 연구에 중요한 이정표를 제시함.
