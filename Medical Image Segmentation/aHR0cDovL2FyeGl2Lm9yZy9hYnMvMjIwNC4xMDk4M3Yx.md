# Federated Contrastive Learning for Volumetric Medical Image Segmentation

Yawen Wu, Dewen Zeng, Zhepeng Wang, Yiyu Shi, and Jingtong Hu (2022)

## 🧩 Problem to Solve

본 논문은 라벨링된 데이터가 매우 부족한 환경에서 3차원 의료 영상 분할(Volumetric Medical Image Segmentation) 성능을 높이기 위한 방법을 다룬다. 딥러닝 기반의 의료 영상 분석은 대규모의 라벨링된 데이터셋을 필요로 하지만, 의료 데이터의 특성상 각 의료 기관(site)이 보유한 데이터의 양이 제한적이며, 전문가의 수작업 라벨링 비용이 매우 높다는 문제가 있다.

Federated Learning(FL)은 데이터를 중앙으로 수집하지 않고 로컬에서 학습함으로써 프라이버시를 보호하며 협력 학습을 가능하게 하지만, 기존의 FL 방식은 기본적으로 모든 데이터에 라벨이 있다는 전제의 지도 학습(Supervised Learning)에 의존한다. 이를 해결하기 위해 라벨이 없는 데이터로 사전 학습을 수행하는 Contrastive Learning(CL)을 FL에 접목하려는 시도가 있으나, 두 가지 핵심적인 한계가 존재한다. 첫째, 개별 클라이언트가 보유한 데이터의 다양성(Diversity)이 부족하여 CL이 효과적인 표현(Representation)을 학습하지 못한다. 둘째, 각 클라이언트가 독립적인 특성 공간(Feature Space)을 학습하게 되어, 모델 통합 시 특성 공간의 불일치로 인해 성능이 저하된다.

따라서 본 연구의 목표는 데이터 프라이버시를 유지하면서도, 제한된 라벨만으로 높은 분할 성능을 달성할 수 있는 Federated Contrastive Learning(FCL) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 로컬 데이터의 다양성 부족과 특성 공간의 불일치 문제를 해결하기 위해 **Feature Exchange(FE)**와 **Global Structural Matching(GSM)**이라는 두 가지 메커니즘을 제안한 것이다.

1. **Feature Exchange (FE):** 원본 이미지 데이터를 공유하는 대신, 인코더를 통해 추출된 저차원 특성 벡터(Feature Vectors)를 클라이언트 간에 교환한다. 이를 통해 각 사이트는 다른 사이트의 데이터 특성을 참조할 수 있게 되어, 로컬 학습 시 더 다양한 대조군(Contrastive data)을 확보함으로써 데이터 다양성 문제를 해결한다.
2. **Global Structural Matching (GSM):** 3D 의료 영상의 해부학적 구조적 유사성을 활용하여 서로 다른 클라이언트 간의 특성 공간을 정렬한다. 서로 다른 환자라도 동일한 해부학적 영역은 유사한 특성을 가질 것이라는 직관에 기반하여, 로컬 특성을 다른 클라이언트의 동일 영역 특성에 맞춤으로써 통합된 특성 공간(Unified Feature Space)을 학습한다.

## 📎 Related Works

**Federated Learning (FL)**은 분산된 클라이언트들이 원본 데이터를 공유하지 않고 모델 파라미터만을 교환하여 공유 모델을 학습하는 방식이다. 하지만 대부분의 기존 연구는 모든 데이터에 라벨이 필요하다는 한계가 있어, 라벨링 비용이 높은 의료 분야에 적용하기 어렵다.

**Contrastive Learning (CL)**은 유사한 샘플은 가깝게, 서로 다른 샘플은 멀게 배치하도록 학습하는 자기지도 학습(Self-supervised Learning) 기법이다. 그러나 기존 CL은 대규모의 중앙 집중식 데이터셋을 가정하고 설계되었으며, FL 환경처럼 데이터가 파편화되고 다양성이 낮은 상황에서는 성능이 크게 저하된다.

**Federated Unsupervised Pre-training** 분야의 기존 연구 중 FedCA와 같은 방식은 클라이언트 간에 공유 데이터셋을 사용하지만, 이는 의료 데이터의 프라이버시 정책상 현실적으로 불가능하다. 본 논문은 원본 데이터를 공유하지 않고 특성 벡터만을 교환함으로써 프라이버시를 유지하면서도 CL의 효과를 극대화한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

본 논문이 제안하는 프레임워크는 크게 **FCL 기반의 사전 학습(Pre-training)** 단계와 **제한된 라벨을 이용한 미세 조정(Fine-tuning)** 단계로 구성된다.

### 1. 전체 파이프라인

먼저 분산된 클라이언트들이 라벨이 없는 3D 볼륨 데이터에서 2D 슬라이스를 샘플링하고, FCL을 통해 공유 인코더(Shared Encoder)를 학습한다. 이렇게 학습된 인코더는 U-Net의 인코더 부분으로 사용되며, 이후 소량의 라벨링된 데이터를 사용하여 지도 학습 방식으로 미세 조정을 수행한다.

### 2. Feature Exchange (FE) 기반의 Contrastive Learning

로컬 CL의 성능을 높이기 위해 MoCo(Momentum Contrast) 아키텍처를 채택하여 메인 인코더와 모멘텀 인코더를 운영한다.

- **Negative Samples 구성:** 각 클라이언트는 자신의 로컬 특성 뱅크 $Q_{l,c}$와 다른 클라이언트로부터 전달받은 원격 특성 뱅크 $\{Q_{l,i}\}$를 합쳐 통합 메모리 뱅크 $Q$를 구성한다.
  $$Q = Q_{l,c} \cup \{Q_{l,i} | 1 \le i \le |C|, i \neq c\}$$
  여기서 $|C|$는 클라이언트의 수이다. 너무 많은 Negative 샘플은 학습 효율을 떨어뜨리므로, $Q$에서 무작위로 $K$개의 샘플을 추출하여 $Q'$를 구성한다.
  $$Q' = \{Q_i | i \sim U(|Q|, K)\}$$
- **Local Positives 구성:** 볼륨 데이터를 $S$개의 파티션으로 나누어, 서로 다른 환자(볼륨)의 동일 파티션 내 이미지들을 Positive 쌍으로 정의한다.
- **Local Contrastive Loss:** 로컬 특성 $q$와 로컬 Positive $P(q)$, 그리고 샘플링된 Negative $Q'$ 사이의 손실 함수는 다음과 같다.
  $$L_{local} = -\frac{1}{|P(q)|} \sum_{k^+ \in P(q)} \log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{n \in Q'} \exp(q \cdot n / \tau)}$$

### 3. Global Structural Matching (GSM)

특성 공간의 불일치를 해결하기 위해 원격 Positive $\Lambda(q)$를 정의한다. $\Lambda(q)$는 $Q'$ 내에서 $q$와 동일한 파티션(해부학적 영역)에 속하는 특성들의 집합이다.
$$\Lambda(q) = \{p | p \in Q', \text{partition}(p) = \text{partition}(q)\}$$
최종 손실 함수는 로컬 정렬과 글로벌 정렬을 모두 고려하여 다음과 같이 정의된다.
$$L_q = L_{remote} + L_{local}$$
여기서 $L_{remote}$는 $L_{local}$ 식에서 $P(q)$를 $\Lambda(q)$로 대체하여 계산한 값이다.

## 📊 Results

### 실험 설정

- **데이터셋:** ACDC MICCAI 2017 챌린지 데이터셋 (100명의 환자, 3D 심장 MRI 영상).
- **설정:** 10개의 클라이언트로 나누어 각 클라이언트당 10명의 환자 데이터 할당.
- **평가 지표:** Dice Similarity Coefficient (DSC).
- **비교 대상:** Random init, Local CL, Rotation, SimCLR, SwAV, FedRotation, FedSimCLR, FedSwAV, FedCA.
- **시나리오:** 각 클라이언트당 라벨링된 환자 수를 $N \in \{1, 2, 4, 8\}$명으로 설정하여 로컬 미세 조정 및 연합 미세 조정 성능을 비교.

### 주요 결과

1. **로컬 미세 조정 결과:** 제안 방법은 모든 라벨 수 설정에서 베이스라인을 압도하였다. 특히 $N=1$일 때 DSC 0.506을 기록하여, 가장 성능이 좋은 베이스라인이 $N=2$일 때 기록한 성능(0.508)과 유사한 수준을 보였다. 이는 라벨링 효율성을 약 2배 향상시켰음을 의미한다.
2. **연합 미세 조정 결과:** 협력 학습을 통한 미세 조정에서도 제안 방법이 가장 높은 성능을 보였으며, $N=1$일 때 DSC 0.646을 달성하였다. 또한 로컬 미세 조정보다 전반적으로 높은 성능을 보여, 사전 학습된 인코더가 연합 학습 환경에서도 효과적으로 작동함을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 의료 영상 데이터의 고질적인 문제인 '데이터 부족'과 '프라이버시 보호'라는 두 마리 토끼를 잡기 위해 특성 벡터 교환이라는 전략을 사용하였다. 특히 단순한 CL의 적용을 넘어 해부학적 구조 정보(Structural similarity)를 활용한 GSM을 통해 분산된 환경에서도 일관된 특성 공간을 학습할 수 있음을 보여주었다. 이는 매우 적은 양의 라벨만으로도 고성능의 세그멘테이션 모델을 구축할 수 있다는 점에서 실용적 가치가 매우 높다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, 특성 벡터를 교환하는 과정에서 추가적인 통신 비용이 발생하며, 이는 클라이언트 수가 늘어날수록 부담이 될 수 있다. 둘째, 비록 원본 이미지를 공유하지 않더라도 추출된 특성 벡터로부터 원본 이미지를 복원하려는 Inversion Attack에 취약할 가능성이 있다. 이에 대해 저자들은 향후 통신 비용 최적화와 보안 강화 방안(DeepObfuscator 등)을 연구하겠다고 명시하였다.

## 📌 TL;DR

본 논문은 라벨링된 데이터가 부족한 의료 영상 분할 문제를 해결하기 위해, 특성 벡터를 교환하여 데이터 다양성을 높이고 해부학적 구조 유사성을 이용해 특성 공간을 정렬하는 **Federated Contrastive Learning (FCL)** 프레임워크를 제안한다. 실험 결과, 제안 방법은 기존의 연합 학습 및 자기지도 학습 방식보다 훨씬 적은 라벨만으로도 월등한 분할 성능을 보였으며, 이는 실제 의료 현장에서 라벨링 비용을 획기적으로 줄이면서도 프라이버시를 보호하는 모델을 구축하는 데 기여할 가능성이 크다.
