# Deep Federated Anomaly Detection for Multivariate Time Series Data

Wei Zhu, Dongjin Song, Yuncong Chen, Wei Cheng, Bo Zong, Takehiko Mizoguchi, Cristian Lumezanu, Haifeng Chen, and Jiebo Luo (2022)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터에 대한 **연합 비지도 이상치 탐지(Federated Unsupervised Anomaly Detection, FedUAD)** 문제를 해결하고자 한다.

다변량 시계열 데이터의 이상치 탐지를 위해서는 대량의 정상 데이터를 수집하여 프로파일링하는 것이 중요하지만, 실제 IoT 환경에서는 다음과 같은 제약 사항이 존재한다:

1. **데이터 프라이버시 및 규제**: 데이터 공유가 금지되어 있어 데이터를 중앙 서버로 직접 전송하는 것이 불가능하거나 금지되어 있다.
2. **데이터의 이질성(Heterogeneity)**: 각 엣지 디바이스(Edge Device)가 수집하는 정상 상태의 모드(Mode)가 서로 다르다. 예를 들어, 고령자의 디바이스에는 '걷기'와 '앉기' 데이터만 있고, 젊은이의 디바이스에는 '달리기'와 '자전거 타기' 데이터가 있을 수 있다.
3. **부분적 데이터 커버리지**: 특정 디바이스가 전체 정상 공간의 일부만 커버하는 경우, 단순히 지역적으로 학습된 모델을 사용하면 다른 디바이스에서는 정상인 패턴을 이상치로 오탐지(False Positive)할 가능성이 매우 높다.

따라서 본 논문의 목표는 데이터 프라이버시를 보호하면서도, 여러 디바이스에 이질적으로 분산된 정상 패턴을 통합적으로 학습하여 정확하게 이상치를 탐지할 수 있는 연합 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 **Fed-ExDNN (Federated Exemplar-based Deep Neural Network)**이라는 프레임워크를 제안하며, 핵심 아이디어는 다음과 같다:

1. **Exemplar-based Local Learning**: 각 엣지 디바이스에서 정상 패턴의 대표값인 Exemplar(본보기)를 학습하는 **ExDNN**을 설계하였다. 이는 단순한 재구성 오차(Reconstruction Error) 기반 방식보다 이질적인 데이터 분포를 다루기에 더 적합하다.
2. **Deep Relation Preserving (DRP)**: 오토인코더(Auto-Encoder) 초기화 없이도 효율적인 표현 학습을 수행하기 위해, 원본 특징 공간의 지역적 유사성을 보존하는 DRP 기법을 도입하였다.
3. **Federated Constrained Clustering (FedCC)**: 중앙 서버에서 각 디바이스의 지역 Exemplar들을 단순 평균 내지 않고, 투영 함수(Projection Function)와 제약 조건이 있는 클러스터링을 통해 정렬(Align)하고 집계함으로써 통합된 글로벌 Exemplar 모듈을 생성한다.

## 📎 Related Works

### 비지도 이상치 탐지 (Unsupervised Anomaly Detection)

기존의 딥러닝 기반 이상치 탐지는 크게 네 가지로 분류된다:

- **One-class classification**: Deep SVDD, CVDD 등이 있으며, 데이터를 하나의 구(Sphere) 안에 모으는 방식이다.
- **Reconstruction-based**: LSTM-AE, USAD, BeatGAN 등이 있으며, 정상 데이터를 잘 재구성하도록 학습한 뒤 재구성 오차가 큰 데이터를 이상치로 간주한다.
- **Contrastive learning**: 최근 각광받는 방식으로, 데이터 간의 상대적 유사성을 학습한다.
- **Clustering-based**: DAGMM과 같이 특징 공간에서 클러스터링을 수행한다.

### 연합 학습 (Federated Learning)

FedAvg, FedProx 등이 대표적이며, 주로 지도 학습(Supervised Learning)을 위해 설계되었다. 하지만 비지도 이상치 탐지에 이를 그대로 적용할 경우, 각 모델이 정상 공간의 일부만 학습하여 글로벌 모델이 특정 모드로 붕괴(Collapse)하거나 이상치 공간까지 포함하게 되는 문제가 발생한다.

## 🛠️ Methodology

Fed-ExDNN은 엣지 디바이스의 **ExDNN**과 서버의 **FedCC**로 구성된다.

### 1. Local Device: ExDNN

ExDNN은 다변량 시계열 데이터를 입력받아 LSTM 기반 인코더 $f(\cdot; \theta)$를 통해 임베딩하고, 이를 $K$개의 학습 가능한 Exemplar 세트 $C = \{c_1, \dots, c_K\}$와 비교한다.

**학습 목표 및 손실 함수:**
ExDNN은 다음 세 가지 손실 함수의 합을 최적화한다:

1. **클러스터링 손실 (KL Divergence)**: 학습 데이터가 Exemplar에 가깝게 배치되도록 유도한다.
   - $q_{ij}$는 데이터 $i$가 Exemplar $j$에 할당될 확률이며, 코사인 유사도 $s(\cdot, \cdot)$를 사용하여 계산한다.
   - 이를 통해 각 Exemplar는 특정 정상 패턴의 중심점 역할을 하게 된다.
2. **균형 손실 (Balanced Loss)**: 특정 클러스터에만 샘플이 쏠리는 것을 방지하고, 노이즈 데이터에 대한 강건성을 높인다.
   $$\min_{\theta, C} -\alpha^T \log \left( \frac{1}{n} \sum_{i=1}^n q_i \right)$$
3. **Deep Relation Preserving (DRP) 손실**: 오토인코더 없이도 원본 공간의 유사성을 유지하도록 하며, KNN 그래프를 미니배치 내 샘플로 근사하여 계산한다.
4. **절대 점수 손실 (Absolute Score Loss)**: 클러스터링 결과뿐만 아니라 실제 이상치 점수(Anomaly Score)의 수치적 값을 직접 최적화하기 위해 도입되었다.
   $$\frac{1}{n} \sum_{i=1}^n \log (1 + \exp(-\gamma_3 (s(f(X_i), \bar{c}_i) - m)))$$
   여기서 $\bar{c}_i$는 소프트하게 근사된 가장 가까운 Exemplar이며, $m$은 마진(Margin) 값이다.

**이상치 점수 계산:**
테스트 데이터 $X$에 대해 가장 가까운 Exemplar와의 코사인 유사도의 음수 값을 점수로 사용한다:
$$\text{Score}(x) = -\max_j s(f(X), c_j)$$

### 2. Central Server: FedCC

서버는 각 디바이스로부터 $\theta^l$과 $\{c_1^l, \dots, c_K^l\}$를 전송받는다. $\theta^l$은 FedAvg로 평균내어 글로벌 인코더 $g$를 만들지만, Exemplar는 이질성 때문에 단순 평균 시 성능이 저하된다.

**FedCC 프로세스:**

1. **투영 및 정렬**: 투영 함수 $h_\phi$를 학습하여 지역 Exemplar들을 정렬된 임베딩 공간으로 보낸다.
2. **제약 조건 기반 클러스터링**: 동일한 초기값을 가졌던 Exemplar들이 서로 가깝게 유지되도록 하는 제약 조건 $R(c_i^l)$을 포함하여 클러스터링을 수행한다.
3. **글로벌 Exemplar 생성**: 클러스터링 지표 $q_{il,z}$를 가중치로 사용하여 각 지역 Exemplar들의 가중 평균으로 글로벌 Exemplar $u_z$를 계산한다:
   $$u_z = \frac{\sum_{i=1}^K \sum_{l=1}^L q_{il,z} c_i^l}{\sum_{i=1}^K \sum_{l=1}^L q_{il,z}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 2D Gesture, ECG5000, SWaT, HAR Laying, UWave, ArabicDigits 등 6개 다변량 시계열 데이터셋 사용.
- **비교 대상**: LSTM-AE, BeatGAN, MemAE, USAD, Deep SVDD, CVDD, DAGMM 및 연합 학습 베이스라인(FedAvgAE, FedProxAE 등).
- **지표**: AUC, F1-score, Precision, Recall.

### 주요 결과

1. **ExDNN의 성능 (Local)**:
   - 거의 모든 데이터셋에서 기존의 재구성 기반(Reconstruction-based) 및 One-class 방식보다 우수한 성능을 보였다. 특히 이질적인 정상 케이스가 많은 AerobicDigit 데이터셋에서 CVDD 대비 평균 AUC가 4.58% 향상되었다.
   - Ablation Study를 통해 절대 점수 손실(Absolute term)과 균형 손실(Balanced term)이 성능 향상 및 학습 안정화에 기여함을 확인하였다.
2. **Fed-ExDNN의 성능 (Federated)**:
   - 단순 FedAvg나 FedProx를 적용한 모델들보다 월등한 성능을 기록하였다.
   - 특히 FedKmsEx(K-means 기반 집계)보다 Fed-ExDNN이 우수한데, 이는 FedCC가 투영 함수를 통해 지역 Exemplar 간의 정렬(Alignment)을 먼저 수행하기 때문이다.
   - t-SNE 시각화 결과, FedAvg는 글로벌 Exemplar들이 한곳으로 뭉쳐버리는(Collapse) 현상이 나타났으나, Fed-ExDNN은 다양성을 유지하면서도 적절히 정렬된 모습을 보였다.

## 🧠 Insights & Discussion

### 강점

- **이질성 해결**: 단순한 가중치 평균이 아니라, Exemplar라는 대표값을 정의하고 이를 클러스터링 기반으로 집계함으로써 데이터 분포가 다른 디바이스 간의 지식을 효율적으로 통합하였다.
- **효율적인 표현 학습**: DRP를 도입함으로써 연합 학습 환경에서 통신 비용과 프라이버시 문제를 야기할 수 있는 오토인코더 사전 학습 단계를 제거하였다.
- **강건성**: Balanced Loss를 통해 학습 데이터에 일부 이상치가 섞여 있는 오염된(Contaminated) 상황에서도 강건한 성능을 유지함을 입증하였다.

### 한계 및 논의

- **하이퍼파라미터 민감도**: Exemplar의 개수 $K$가 성능에 영향을 미치며, 데이터셋의 복잡도에 따라 적절한 $K$를 찾는 과정이 필요하다.
- **가정**: 본 연구는 각 디바이스의 데이터가 비록 이질적일지라도, 전체 시스템 관점에서는 공통된 정상 공간의 부분 집합이라는 가정을 전제로 한다.

## 📌 TL;DR

본 논문은 데이터 프라이버시 보호와 데이터 이질성 문제를 동시에 해결하는 **연합 비지도 이상치 탐지 프레임워크(Fed-ExDNN)**를 제안한다. 지역적으로는 **ExDNN**을 통해 정상 패턴의 대표값(Exemplar)을 학습하고, 서버에서는 **FedCC**라는 제약 조건 기반 클러스터링을 통해 이들을 정렬 및 집계한다. 실험 결과, 기존의 재구성 기반 방식이나 단순 연합 평균 방식보다 다변량 시계열 데이터의 이상치 탐지 성능이 크게 향상되었으며, 특히 데이터 분포가 서로 다른 환경에서 매우 효과적임을 입증하였다. 이 연구는 프라이버시가 중요한 의료, IoT 보안 분야의 이상치 탐지 시스템 구축에 중요한 기여를 할 것으로 기대된다.
