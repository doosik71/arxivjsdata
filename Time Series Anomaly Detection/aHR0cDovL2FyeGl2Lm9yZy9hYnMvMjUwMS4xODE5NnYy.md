# GDformer: Going Beyond Subsequence Isolation for Multivariate Time Series Anomaly Detection

Qingxiang Liu, Chenghao Liu, Sheng Sun, Di Yao, Yuxuan Liang (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 다변량 시계열 이상치 탐지(Multivariate Time Series Anomaly Detection, MTS AD)에서 발생하는 **Subsequence Isolation(부분 시퀀스 고립)** 문제이다.

기존의 비지도 학습 기반 방법론들은 크게 재구성 오차(Reconstruction Error) 기반 방식과 연관성 발산(Association Divergence) 기반 방식으로 나뉜다. 그러나 이 두 방식 모두 전체 시계열을 짧은 길이의 부분 시퀀스(Subsequence)로 나누어 처리하는 특성이 있다. 이로 인해 다음과 같은 문제가 발생한다.

1. **제한된 컨텍스트**: 부분 시퀀스 내의 정보만을 활용하므로 전체 시계열 수준의 글로벌한 문맥 정보를 충분히 반영하지 못한다.
2. **일관성 없는 탐지 기준**: 각 부분 시퀀스마다 시계열의 변동성이나 이상치의 포함 여부가 다르기 때문에, 동일한 점수라도 시퀀스에 따라 그 의미가 달라진다. 이는 결국 전체 시계열에 적용할 통일된 탐지 임계값(Unified Detection Criterion)을 설정하기 어렵게 만들며, 결과적으로 False Positive(오탐)와 False Negative(미탐)를 증가시킨다.

따라서 본 논문의 목표는 부분 시퀀스 고립 문제를 해결하기 위해 전체 시계열 수준에서 공유되는 글로벌 표현(Global Representations)을 학습하고, 이를 통해 일관된 탐지 기준을 제공하는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Global Dictionary**와 **Prototypes**를 도입하여 Transformer의 Self-attention 구조를 개선하는 것이다.

1. **Global Dictionary-enhanced Transformer (GDformer)**: 모든 정상 데이터 포인트가 공유하는 글로벌 표현을 학습하기 위해 Key($K$)와 Value($V$) 벡터로 구성된 글로벌 딕셔너리를 도입하였다. 이를 통해 기존의 Self-attention이 가진 $O(T^2)$의 복잡도 문제를 해결함과 동시에, 전체 시계열의 맥락을 반영하는 글로벌 표현을 얻을 수 있다.
2. **Dictionary-based Cross Attention**: Query는 입력 시퀀스에서, Key와 Value는 글로벌 딕셔너리에서 가져오는 Cross-attention 메커니즘을 통해 포인트와 글로벌 표현 간의 상관관계 가중치를 산출한다.
3. **Similarity-based Criterion via Prototypes**: 정상 포인트들의 상관관계 가중치 분포를 캡처하는 프로토타입(Prototypes)을 도입하였다. 정상 데이터와 이상 데이터 간의 프로토타입 유사도 차이(Similarity Discrepancy)를 이용하여 더 정밀하고 압축된 탐지 경계(Detection Boundary)를 형성한다.

## 📎 Related Works

기존의 시계열 이상치 탐지 연구는 통계적 방법, 머신러닝 방법, 그리고 딥러닝 방법으로 분류된다.

- **통계 및 머신러닝 기반**: ARIMA, LOF, Deep-SVDD 등이 있으며, 주로 데이터의 밀도나 군집 중심으로부터의 거리 등을 기반으로 이상치를 판별한다. 하지만 복잡한 다변량 시계열의 패턴을 학습하는 데 한계가 있다.
- **재구성 기반(Reconstruction-based)**: LSTM-VAE, OmniAnomaly 등이 대표적이다. 정상 데이터를 잘 재구성하도록 학습시킨 후, 재구성 오차가 큰 지점을 이상치로 간주한다. 그러나 이상치의 희소성과 복잡한 패턴으로 인해 정상 데이터의 영향력이 너무 커져 변별력이 떨어지는 경우가 많다.
- **연관성 기반(Association-based)**: Anomaly Transformer와 DCdetector가 이에 해당한다. 이들은 포인트 간의 연관성(Association) 차이를 이용하지만, 앞서 언급한 **Subsequence Isolation** 문제로 인해 부분 시퀀스별로 탐지 기준이 달라지는 한계를 가진다.

GDformer는 이러한 기존 방식들과 달리, 딕셔너리 기반의 Cross-attention을 통해 부분 시퀀스 수준을 넘어 전체 시계열 수준의 정보를 통합함으로써 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 아키텍처

GDformer는 딕셔너리 기반의 Cross-attention 모듈과 Feed-forward 레이어를 교대로 쌓은 구조이다. 마지막에는 재구성을 위한 Projection 블록이 위치한다.

### 2. 주요 구성 요소 및 절차

#### 2.1 입력 임베딩 (Input Embedding)

입력 시퀀스 $X \in \mathbb{R}^{T \times d}$에 대해 $\alpha$의 확률로 랜덤 마스킹을 수행한 후, Instance Normalization을 적용하여 노이즈를 제거한다. 이후 선형 레이어를 통해 $D$ 차원의 임베딩 $X^0$로 변환한다.

#### 2.2 딕셔너리 기반 Cross Attention (Dictionary-Based Cross Attention)

기존 Self-attention 대신, 학습 가능한 글로벌 딕셔너리 $K^l \in \mathbb{R}^{N \times D}$와 $V^l \in \mathbb{R}^{N \times D}$를 사용한다. 여기서 $N$은 딕셔너리 크기이며 $N \ll T$이다.

- **Cross Attention 계산**:
  각 헤드 $h$에 대해 Query $Q^h_l$은 입력 표현에서 유도되며, 계산식은 다음과 같다.
  $$U^h_l = \text{Softmax}\left(\frac{Q^h_l {K^h_l}^\top}{\sqrt{D_h}}\right) V^h_l$$
  이 연산의 시간 복잡도는 $O(TN)$으로, 기존 Self-attention의 $O(T^2)$보다 효율적이다.

- **유사도 평가 (Similarity Evaluation)**:
  Cross-attention 맵 $M^h_l = \text{Softmax}(\frac{Q^h_l {K^h_l}^\top}{\sqrt{D_h}})$은 각 포인트와 글로벌 표현 간의 상관관계를 나타낸다. 이를 더 정밀하게 분석하기 위해 정상 분포를 대표하는 프로토타입 $E^l \in \mathbb{R}^{P \times N}$을 도입한다.
  $$S^h_l = M^h_l \text{Softmax}(E^l)^\top$$
  여기서 $S^h_l \in \mathbb{R}^{T \times P}$는 포인트별 상관관계 분포와 프로토타입 간의 유사도를 의미하며, 이를 합산하여 포인트별 유사도 강도 $\hat{S}^h_l$을 얻는다.

### 3. 학습 목표 및 손실 함수

모델은 재구성 손실($L_c$)과 유사도 발산 손실($L_s$)의 조합으로 학습된다.
$$L_{total} = L_c - \lambda L_s = \|X - \hat{X}\|^2_2 - \lambda \sum_l \sum_h \hat{S}^h_l$$
여기서 $\lambda > 0$는 두 손실 항의 균형을 맞추는 하이퍼파라미터이다. $L_s$를 최대화함으로써 프로토타입이 정상 데이터의 상관관계 패턴을 더 잘 학습하도록 유도한다.

### 4. 추론 및 이상치 판별

학습이 완료된 후, 이상치 점수(Anomaly Score)는 프로토타입과의 유사도의 역수로 계산된다.
$$\text{AnomalyScore}(X) = \text{Softmax}\left(-\sum_l \sum_h \hat{S}^h_l\right)$$
최종적으로 이 점수가 임계값 $\delta$보다 크면 이상치($y_i=1$)로, 작으면 정상($y_i=0$)으로 판별한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MSL, SMAP, SWaT, PSM 등 4개의 실제 벤치마크 데이터셋을 사용하였다.
- **지표**: Precision, Recall, F1-score를 측정하였다.
- **비교 대상**: OCSVM, IForest 등 고전적 방법부터 Anomaly Transformer, DCdetector 등 최신 딥러닝 모델까지 총 19개의 베이스라인과 비교하였다.

### 2. 주요 결과

- **정량적 성능**: GDformer는 모든 데이터셋에서 기존 모델들을 압도하며 SOTA(State-of-the-art) 성능을 달성하였다. 특히 AnomalyTrans와 DCdetector 대비 높은 F1-score를 기록하였다.
- **전이 학습(Transferability)**: 한 데이터셋에서 학습된 글로벌 딕셔너리와 프로토타입을 다른 데이터셋에 적용했을 때도 성능 저하가 적었다. 이는 서로 다른 도메인의 시계열 데이터라도 정상 상태의 공통된 템포럴 패턴이 존재함을 시사한다.
- **효율성**: $O(TN)$의 복잡도 덕분에 메모리 사용량이 현저히 낮았으며, 학습 시간 또한 AnomalyTrans 대비 88.8%, DCdetector 대비 94.7% 감소하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

GDformer의 우수한 성능은 **글로벌 문맥의 통합**과 **유사도 기반 기준**에서 기인한다.

- 기존 모델들이 부분 시퀀스 내의 상대적 연관성에 의존하여 임계값 설정에 어려움을 겪은 반면, GDformer는 전역적으로 학습된 딕셔너리를 통해 모든 포인트에 대해 통일된 기준을 적용할 수 있었다.
- 재구성 오차만 사용하는 것보다 프로토타입을 통한 유사도 분석을 병행하는 것이 훨씬 더 명확한 탐지 경계를 형성함을 확인하였다.

### 2. 한계 및 비판적 해석

- **이론적 분석 부족**: 논문에서는 글로벌 딕셔너리의 Key-Value 쌍이 구체적으로 어떤 기능을 수행하는지에 대한 수학적/이론적 분석이 부족하며, 이를 향후 과제로 남겨두었다.
- **하이퍼파라미터 민감도**: 실험 결과, $\lambda, P, N$ 등의 설정에 따라 성능 변화가 있으며, 특히 SMAP 데이터셋에서 이러한 민감도가 더 크게 나타났다. 이는 최적의 하이퍼파라미터를 찾는 과정이 필수적임을 의미한다.

## 📌 TL;DR

GDformer는 다변량 시계열 이상치 탐지에서 기존의 **Subsequence Isolation(부분 시퀀스 고립)** 문제를 해결하기 위해 **글로벌 딕셔너리 기반의 Cross-attention**과 **프로토타입 유사도 메커니즘**을 도입한 모델이다. 이를 통해 전체 시계열 수준의 정상 표현을 학습하여 일관된 탐지 기준을 제공하며, 계산 복잡도를 $O(T^2)$에서 $O(TN)$으로 낮춰 효율성을 극대화하였다. 이 연구는 시계열 데이터 간의 공통된 정상 패턴을 학습할 수 있음을 보여주었으며, 향후 시계열 이상치 탐지를 위한 파운데이션 모델(Foundation Model) 구축의 가능성을 제시하였다.
