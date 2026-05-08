# MELODY: Robust Semi-Supervised Hybrid Model for Entity-Level Online Anomaly Detection with Multivariate Time Series

Jingchao Ni, Gauthier Guinet, Peihong Jiang, Laurent Callot, Andrey Kan (2024)

## 🧩 Problem to Solve

본 논문은 대규모 클라우드 네이티브 시스템에서 소프트웨어 배포(deployment) 과정 중 발생하는 이상 징후를 실시간으로 탐지하는 문제를 다룬다. 소프트웨어 업데이트 과정에서 발생하는 결함 있는 코드 변경은 서비스 성능 저하 및 연쇄적인 장애를 초래할 수 있으므로, 이를 빠르게 탐지하여 롤백(rollback)하는 것이 매우 중요하다.

기존의 다변량 시계열(Multivariate Time Series, MTS) 이상 탐지 연구들은 주로 단일 엔티티 내의 특정 시점에서 발생하는 '포인트 레벨(point-level)' 이상 탐지에 집중해 왔다. 그러나 실제 운영 환경에서의 배포 모니터링은 '엔티티 레벨(entity-level)'의 탐지가 필요하며, 여기에는 다음과 같은 네 가지 독특한 어려움이 존재한다.

1. **엔티티의 이질성(Heterogeneity):** 배포되는 서비스마다 메트릭의 스케일, 시계열의 길이, 그리고 포함된 변수의 수가 서로 다르다.
2. **낮은 지연 시간 허용치(Low Latency Tolerance):** 배포 엔티티는 수분 내외로 짧게 유지되는 경우가 많아, 엔티티마다 개별 모델을 학습시키는 것이 불가능하며 사전 학습된 모델을 공유해야 한다.
3. **모호한 이상 정의(Ambiguous Anomaly Definition):** 단순한 수치 변화가 반드시 이상을 의미하지는 않으며, 도메인 전문가의 감독 신호(supervision signal)가 필수적이다.
4. **제한된 지도 데이터(Limited Supervision):** 데이터 라벨이 특정 시점이 아닌 엔티티 전체에 대해 부여되며, 라벨링 비용이 높고 노이즈가 포함될 가능성이 크다.

따라서 본 논문의 목표는 이질적인 엔티티들 간에 공유 가능하면서도, 제한된 라벨을 효율적으로 활용하여 정확하게 이상 배포 엔티티를 식별하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **온라인 특징 추출기(Online Feature Extractor, OFE)**를 통해 서로 다른 MTS를 동일한 특징 공간으로 투영하고, 이를 **반지도 학습 기반의 하이브리드 모델(SemiAD)**로 분석하는 것이다.

가장 중추적인 설계 아이디어는 다음과 같다.

- **특징 정렬:** 서로 다른 스케일과 차원을 가진 MTS를 '정상 이력 대비 편차'라는 비교 가능한 형태의 특징으로 변환하여 엔티티 간 공통 모델 사용이 가능하게 한다.
- **SemiDOC (Semi-supervised Deep One-Class model):** 대량의 무라벨 데이터와 소량의 라벨 데이터를 동시에 활용하며, 특히 Negative Sampling을 통해 정상 데이터의 경계(boundary)를 명시적으로 좁혀 탐지 성능을 높인다.
- **하이브리드 앙상블:** 반지도 학습 모델(SemiDOC)과 지도 학습 모델(LightGBM)을 결합하되, 특히 SemiDOC가 먼저 정상 엔티티를 필터링하고 남은 데이터를 LightGBM가 정밀 판별하는 순차적(Sequential) 구조를 제안하여 오탐지(False Positive)를 줄인다.

## 📎 Related Works

기존 연구는 크게 MTS 기반 방법론과 일반적인 이상 탐지(General AD) 방법론으로 나뉜다.

1. **MTS 기반 방법론:** OmniAnomaly, AnomalyTransformer, TranAD 등은 주로 재구성(reconstruction)이나 예측(forecasting) 기반의 비지도 학습을 사용한다. 그러나 이들은 포인트 레벨 탐지에 특화되어 있어, 엔티티 레벨의 라벨만을 가진 본 문제에 직접 적용하기 어렵고 엔티티 간 모델 공유가 어렵다는 한계가 있다.
2. **일반 AD 방법론:** DeepSVDD나 DeepSAD와 같은 모델은 벡터 형태의 데이터에서 이상치를 탐지하는 데 효과적이다. 하지만 이들은 시계열 데이터의 특성을 반영하지 못하며, MTS를 벡터화하는 전처리 과정 없이는 사용이 불가능하다.

MELODY는 OFE를 통해 MTS를 벡터화함으로써 일반 AD 모델의 이점을 취하는 동시에, 반지도 학습 구조를 통해 기존 비지도 학습 모델보다 적은 라벨로도 더 높은 정밀도를 달성한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. Online Feature Extractor (OFE)

OFE는 이질적인 MTS를 고정된 차원의 특징 벡터 $\mathbf{z} \in \mathbb{R}^{134}$로 변환한다.

- **Featurizer:** 원시 값을 직접 사용하지 않고, 과거 이력 대비 현재 값의 편차(anomalous degree)를 계산하여 비교 가능하게 만든다.
  - **Rule-based:** 통계 기반(SbF), 임계치 기반(TbF), 누락 값 카운트 기반(CbF) 특징을 추출한다.
  - **Algorithm-based:** Subsequence-based Nearest Neighbor (SubNN)와 Median Forecast (MD) 알고리즘을 사용하여 포인트 레벨의 이상 점수를 생성한다.
  - **Meta-Data:** 배포 설정과 관련된 8개의 정적 특징을 추가한다.
- **Time Pooling Layer:** 배포마다 길이가 다른 문제를 해결하기 위해, 현재 시점까지의 특징값들에 대해 $\text{MaxPool}$과 $\text{MeanPool}$을 적용하여 엔티티 레벨의 특징을 생성한다.
    $$\hat{B}_{8,9}^C = \text{MaxPool}([B_{8,9}^1, \dots, B_{8,9}^C]), \quad \bar{B}_{8,9}^C = \text{MeanPool}([B_{8,9}^1, \dots, B_{8,9}^C])$$
- **Feature Aggregator:** 여러 서비스가 동일한 메트릭을 가질 경우 $\text{MaxPool}$로 통합하고, 특정 메트릭이 누락된 경우 학습 데이터의 평균값으로 보간(imputation)하여 최종적으로 134차원의 벡터 $\mathbf{z}$를 생성한다.

### 2. Semi-Supervised Anomaly Detection (SemiAD)

SemiAD는 두 가지 모델의 하이브리드 구조로 구성된다.

#### (1) Semi-Supervised Deep One-Class Model (SemiDOC)

DeepSVDD를 확장하여, 소량의 라벨된 이상치를 이용해 정상 데이터의 초구(hypersphere) 경계를 좁히는 모델이다. 신경망 $q(\cdot; \theta)$를 통해 데이터를 임베딩하며, 학습 목표는 다음과 같다.
$$\min_{\theta} \frac{1}{W} \sum_{8=1}^{W} \left( \text{dist}(q(\mathbf{z}_{@8}; \theta), \mathbf{c}) + \ell(\mathbf{z}_{@8}, \mathbf{z}_{=8}; \theta) \right) + \lambda \|\theta\|^2$$
여기서 $\ell(\cdot)$은 정상 샘플 $\mathbf{z}_{@8}$과 이상 샘플 $\mathbf{z}_{=8}$ 사이의 거리를 일정 임계치 $X$ 이상으로 벌리는 힌지 손실(Hinge Loss)이다.
$$\ell(\mathbf{z}_{@8}, \mathbf{z}_{=8}; \theta) = \max(X - \text{dist}(q(\mathbf{z}_{@8}; \theta), q(\mathbf{z}_{=8}; \theta)), 0)$$
추론 시에는 학습된 반경 $R'$를 기준으로 $\text{Clip}(\frac{\text{dist}(q(\mathbf{z}_{new}; \theta), \mathbf{c})}{R'}, 0, 1)$를 통해 이상 점수를 계산한다.

#### (2) Supervised Anomaly Detector

노이즈가 포함된 무라벨 데이터로 인한 편향을 방지하기 위해, LightGBM 기반의 지도 학습 모델을 병렬로 운영하여 클래스 경계를 보완한다.

#### (3) Hybrid Combining Strategy

- **MELODY-M (Mean):** 두 모델의 점수를 평균 내어 최종 점수를 산출한다.
- **MELODY-S (Sequential):** SemiDOC가 먼저 낮은 점수의 정상 엔티티를 필터링하고, 확신이 낮은(점수가 높은) 샘플들에 대해서만 LightGBM가 최종 판별을 내린다.

## 📊 Results

### 실험 설정

- **데이터셋:** Amazon AWS의 실제 배포 데이터 (약 30K개 엔티티, 1.2M개 시계열).
- **비교 대상:** MTS 기반 방법(OmniAnomaly, AnomalyTransformer, TranAD) 및 일반 AD 방법(DeepSVDD, DeepSAD, RF, LightGBM).
- **지표:** F1-score, Precision, Recall, FPR (False Positive Rate).

### 주요 결과

- **정량적 성능:** MELODY-S는 모든 라벨 설정(Hard, Soft, Naive)에서 가장 높은 F1-score를 기록하였다. 특히 Hard Labels 데이터셋에서 기존 SOTA 대비 F1-score가 최대 $56.5\%$ 상대적으로 향상되었다.
- **오탐지 감소:** MELODY-S는 Sequential 전략 덕분에 Precision이 높고 FPR이 낮게 나타났다. 이는 실제 롤백 시스템에서 불필요한 롤백을 줄이는 데 매우 중요하다.
- **실제 적용 결과:** A/B 테스트 결과, 기존 LightGBM 단일 모델의 Precision이 $29.7\%$인 반면, MELODY 기반 모델은 $35.9\%$를 기록하여 실무적인 유효성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순히 복잡한 딥러닝 모델을 사용하는 것보다, **데이터의 특성에 맞는 특징 엔지니어링(OFE)과 효율적인 모델 결합 전략**이 실무 환경에서 훨씬 중요하다는 점을 시사한다.

1. **특징 정렬의 중요성:** MTS 기반의 최신 모델들이 성능이 낮게 나온 이유는, 그들이 엔티티 간의 이질성을 극복하지 못했기 때문이다. OFE를 통해 데이터를 '비교 가능한 공간'으로 보낸 것이 성능 향상의 핵심이었다.
2. **반지도 학습의 효용성:** DeepSVDD와 같은 완전 비지도 모델보다, Negative Sampling을 도입한 SemiDOC가 정상 데이터의 경계를 훨씬 더 정교하게 정의함을 t-SNE 시각화를 통해 확인하였다.
3. **순차적 앙상블의 논리:** One-class 모델은 '정상과 다른 것'을 찾고, 지도 학습 모델은 '이상징후의 패턴'을 찾는다. 이 둘을 순차적으로 배치함으로써 "확실한 정상 $\rightarrow$ 의심 샘플 $\rightarrow$ 정밀 판별"의 파이프라인을 구축하여 정밀도를 극대화하였다.

한계점으로는, 특징 추출 과정에서 사용된 도메인 전문가의 룰(Rule-based features)에 의존성이 있다는 점이 있으나, 이는 실무 환경에서 어느 정도 수용 가능한 범위로 판단된다.

## 📌 TL;DR

MELODY는 서로 다른 성격의 다변량 시계열(MTS) 데이터를 가진 배포 엔티티들을 동일한 특징 공간으로 정렬(OFE)하고, 반지도 학습 모델(SemiDOC)과 지도 학습 모델(LightGBM)을 순차적으로 결합하여 이상 배포를 탐지하는 프레임워크이다. 특히 소량의 라벨과 대량의 무라벨 데이터를 모두 활용해 오탐지를 획기적으로 줄였으며, 실제 아마존 AWS 환경에서 롤백 시스템의 정밀도를 향상시킴을 입증하였다. 이 연구는 실무적인 제약 조건(이질적 데이터, 라벨 부족, 저지연 요구) 하에서의 엔티티 레벨 이상 탐지에 중요한 가이드라인을 제시한다.
