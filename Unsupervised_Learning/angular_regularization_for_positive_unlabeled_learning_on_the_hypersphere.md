# Angular Regularization for Positive-Unlabeled Learning on the Hypersphere

Vasileios Sevetlidis, George Pavlidis, and Antonios Gasteratos (2025)

## 🧩 Problem to Solve

본 논문은 Positive-Unlabeled (PU) 학습 문제를 해결하고자 한다. PU 학습은 오직 일부의 긍정(positive) 샘플에만 레이블이 지정되어 있고, 나머지 데이터는 긍정 또는 부정(negative) 샘플이 섞여 있는 unlabeled 상태인 분류 문제이다. 즉, 명시적인 부정 샘플의 감독(negative supervision)이 불가능한 상황에서 분류기를 학습시켜야 한다.

이러한 설정은 의료 진단, 감성 분석, 이상 탐지(anomaly detection)와 같이 부정 샘플을 수집하는 비용이 매우 높거나, 정의가 모호하거나, 윤리적 제약으로 인해 획득하기 어려운 도메인에서 매우 중요하다. 본 연구의 목표는 명시적인 부정 샘플 없이도 긍정 클래스와 부정 클래스를 효과적으로 분리할 수 있는 기하학적 기반의 학습 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 학습 공간을 단위 하이퍼스피어(unit hypersphere)로 설정하고, 긍정 클래스를 하나의 학습 가능한 프로토타입 벡터(prototype vector)로 표현하는 '기하학 우선(geometry-first)' 접근 방식이다.

주요 기여 사항은 다음과 같다.

1. **하이퍼스피어 기반 PU 모델링**: 입력 데이터를 단위 하이퍼스피어로 매핑하고, 학습된 긍정 프로토타입과의 코사인 유사도(cosine similarity)를 통해 클래스를 판별하는 단순하고 해석 가능한 구조를 제안한다.
2. **Unlabeled-only 각도 정규화(Angular Regularization)**: unlabeled 샘플들이 긍정 프로토타입 근처로 뭉치는 현상(false positive clustering)을 방지하기 위해, unlabeled 세트가 하이퍼스피어 상에서 고르게 분산되도록 유도하는 정규화 항을 도입하였다.
3. **단순성 및 안정성**: 기존의 EM 스타일 반복 학습, 모멘텀 큐(momentum queue), 의사 레이블링(pseudo-labeling) 또는 클래스 사전 확률(class-prior) 추정 없이 단일 단계(single-stage)의 end-to-end 최적화를 통해 학습 안정성을 확보하였다.

## 📎 Related Works

기존의 PU 학습 방법론은 크게 세 가지로 나뉜다.

1. **리스 기반 접근법 (Risk-based approaches)**: $uPU$, $nnPU$ 등이 대표적이며, 분류 리스크를 긍정 및 unlabeled 데이터 항으로 분해하여 추정한다. 하지만 이러한 방법들은 클래스 사전 확률 $\pi$ 추정에 매우 민감하며, 표현 학습의 품질이 낮을 때 성능이 급격히 저하된다.
2. **표현 학습 기반 접근법 (Representation-learning approaches)**: $WConPU$, $PiCO$ 등은 대비 학습(contrastive learning)이나 의사 레이블링을 통해 부정 샘플을 생성한다. 그러나 의사 레이블링은 초기 오분류가 전파되는 확인 편향(confirmation bias) 문제에 취약하며, 복잡한 보조 메커니즘이 필요하다.
3. **경계/이상치 기반 접근법 (Boundary/Anomaly-based methods)**: $Dense-PU$ 등은 긍정 집합의 밀도나 볼록 껍질(convex hull)을 이용해 부정 샘플을 식별한다. 이는 긍정 데이터의 매니폴드가 복잡하거나 비볼록(non-convex)한 경우 실패할 가능성이 높다.

본 논문의 제안 방법은 명시적인 부정 모델링이나 반복적인 의사 레이블 생성 과정을 완전히 배제하고, 하이퍼스피어 상의 기하학적 구조와 정규화를 통해 문제를 해결함으로써 기존 방법들의 불안정성을 극복한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 입력 데이터를 단위 하이퍼스피어 $\mathbb{S}^{d-1}$로 매핑하는 신경망 인코더 $f_\theta: \mathcal{X} \to \mathbb{S}^{d-1}$를 학습한다. 긍정 클래스는 학습 가능한 단위 벡터인 프로토타입 $\mu \in \mathbb{S}^{d-1}$로 정의되며, 임베딩 벡터 $z = f_\theta(x)$와 $\mu$ 사이의 코사인 유사도를 기반으로 점수를 계산한다.

### 핵심 방정식 및 점수 함수

분류를 위한 방향성 점수(directional score) $s(z)$는 다음과 같이 정의된다.
$$s(z) = \kappa \mu^\top z$$
여기서 $\kappa > 0$는 점수의 날카로움(sharpness)을 조절하는 고정 스케일링 인자이며, $\mu^\top z$는 두 벡터의 코사인 유사도이다.

### 손실 함수 및 학습 절차

최종 손실 함수 $L$은 세 가지 항의 합으로 구성된다:
$$L = L_{pos} + L_{unlab} + \lambda L_{reg}$$

1. **긍정 정렬 손실 ($L_{pos}$)**: 긍정 샘플 $P$가 프로토타입 $\mu$와 일치하도록 유도한다.
   $$L_{pos} = -\frac{1}{|P|} \sum_{i \in P} \kappa \mu^\top z_i$$
2. **Unlabeled 중립 손실 ($L_{unlab}$)**: unlabeled 샘플 $U$에 대해 최대 불확실성(target = 0.5)을 부여하는 대칭적 이진 교차 엔트로피(BCE) 손실을 적용한다.
   $$L_{unlab} = -\frac{1}{|U|} \sum_{j \in U} [0.5 \log \sigma(\ell_j) + 0.5 \log(1 - \sigma(\ell_j))]$$
   단, $\ell_j = \kappa \mu^\top z_j$이며, $\sigma$는 시그모이드 함수이다.
3. **각도 분산 정규화 ($L_{reg}$)**: unlabeled 임베딩들이 서로 멀어지도록 유도하여 하이퍼스피어 상의 균일성(uniformity)을 높인다.
   $$L_{reg} = \log \left( \frac{1}{|U|(|U|-1)} \sum_{i \neq j} e^{t z_i^\top z_j} \right)$$
   여기서 $t > 0$는 온도 하이퍼파라미터이다.

### 가중치 적용 및 추론

불안정한 unlabeled 샘플의 영향을 조절하기 위해 소프트 가중치 $w(z)$를 도입한다.
$$w(z) = \sigma(\alpha(\mu^\top z - m))$$
여기서 $m$은 각도 마진(angular margin)이며, 프로토타입에 가까운(즉, $\mu^\top z > m$) 샘플에 더 높은 가중치를 부여하여 $L_{unlab}$에 반영함으로써 가짜 긍정(false positive)의 붕괴를 방지한다.

추론 시에는 $s(z) \ge \tau$인 경우 긍정으로 분류하며, 임계값 $\tau$는 검증 세트에서 $F1$ score를 최대화하도록 설정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-10, STL-10, SVHN, ADNI, MedMNIST (OCTMNIST, PathMNIST)
- **평가 지표**: F1 Score, Accuracy, Precision, Recall, AUC, Average Precision (AP)
- **기준선(Baselines)**: $uPU, nnPU, WConPU, PiCO, Dense-PU, ImbPU$ 등 다양한 PU 학습 방법론

### 주요 결과

1. **정량적 성능**: AngularPU는 대부분의 데이터셋에서 매우 경쟁력 있는 성능을 보였으며, 특히 STL-10(F1: 99.24)과 ADNI(F1: 79.74)에서 SOTA 수준의 성능을 달성하였다.
2. **재현율(Recall)의 강점**: Accuracy는 일부 대비 학습(contrastive) 방법론보다 약간 낮을 수 있으나, Recall과 F1 score에서 우위를 보인다. 이는 PU 학습의 특성상 숨겨진 긍정 샘플을 찾아내는 것이 중요하며, 제안 방법의 기하학적 설계가 이를 효과적으로 수행함을 시사한다.
3. **의료 영상 데이터 (ADNI & MedMNIST)**: 레이블이 극도로 적은 ADNI 데이터셋에서 모든 지표(F1, AUC, AP)에서 최적의 성능을 기록하였다. 또한 MedMNIST 실험에서도 높은 AUC와 AP를 기록하며 실제 의료 스크리닝 시나리오에서의 유용성을 입증하였다.

### 분석 결과

- **Ablation Study**: 각도 정규화($L_{reg}$)와 적응형 가중치(weights)를 제거했을 때 AUC와 AP가 눈에 띄게 하락하였다. 특히 $L_{reg}$는 unlabeled 샘플의 무분별한 뭉침을 방지하는 핵심 역할을 한다.
- **기하학적 비교**: 유클리드 거리 기반 모델보다 코사인 유사도 기반의 각도 모델이 모든 데이터셋에서 압도적으로 높은 AP를 기록하였다. 이는 하이퍼스피어 상의 스케일 불변성(scale-invariance)이 PU 학습에 더 적합한 귀납적 편향(inductive bias)을 제공함을 의미한다.

## 🧠 Insights & Discussion

### 강점

본 연구는 복잡한 의사 레이블링 파이프라인이나 사전 확률 추정 없이, 오직 긍정 클래스의 방향성 모델링과 unlabeled 세트의 분산 정규화만으로 고성능의 PU 분류기를 구현하였다. 특히 고차원 임베딩 공간에서 발생하기 쉬운 붕괴 현상을 기하학적으로 해결한 점이 돋보인다.

### 한계 및 가설

논문에서는 긍정 클래스가 하이퍼스피어 상의 단일 방향 모드(single dominant directional mode)를 형성한다고 가정한다. 만약 긍정 클래스가 매우 다양한 여러 개의 클러스터로 나뉘어 있는 다중 모드(multi-modal) 분포를 가진다면, 단일 프로토타입 $\mu$만으로는 한계가 있을 수 있다. 저자들 또한 향후 연구로 vMF 혼합 모델(vMF mixtures)의 도입을 언급하고 있다.

### 비판적 해석

본 방법론은 Precision보다 Recall을 우선시하는 경향을 보인다. 이는 의료 스크리닝과 같이 "긍정 샘플을 놓치지 않는 것"이 중요한 도메인에서는 매우 큰 장점이지만, 정밀도가 극도로 중요한 시스템에서는 임계값 $\tau$의 정교한 튜닝이 필수적일 것이다.

## 📌 TL;DR

본 논문은 하이퍼스피어 임베딩 공간에서 **긍정 프로토타입 기반의 방향성 점수**와 **unlabeled 샘플의 각도 분산 정규화**를 결합한 **AngularPU** 프레임워크를 제안한다. 이 방법은 복잡한 의사 레이블링이나 사전 확률 추정 없이 end-to-end로 학습 가능하며, 특히 레이블이 부족한 고차원 데이터셋(의료 영상 등)에서 높은 재현율(Recall)과 F1 score를 달성한다. 향후 PU 학습 연구에서 복잡한 파이프라인 대신 기하학적 정규화를 통해 안정성을 확보하는 새로운 방향성을 제시하였다.
