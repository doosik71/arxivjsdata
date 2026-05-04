# Domain Generalization for Person Re-identification: A Survey Towards Domain-Agnostic Person Matching

Hyeonseo Lee, Juhyun Park, Jihyong Oh, Chanho Eom (2025)

## 🧩 Problem to Solve

본 논문은 Person Re-identification (ReID) 분야에서 발생하는 **Domain Shift** 문제를 해결하기 위한 **Domain Generalization (DG-ReID)** 기술을 체계적으로 분석한다.

일반적인 ReID 시스템은 학습 데이터와 테스트 데이터가 유사한 특성을 가진다고 가정하지만, 실제 환경에서는 카메라의 시점(viewpoint), 배경(background), 조명 조건(lighting) 등의 변화로 인해 모델의 성능이 급격히 저하되는 문제가 발생한다. 이를 해결하기 위해 기존에는 Target domain의 레이블 없는 데이터를 학습에 활용하는 Domain-Adaptive ReID (DA-ReID)가 제안되었으나, 이는 배포 전 target 데이터에 접근해야 한다는 제약이 있어 확장성이 떨어진다.

따라서 본 논문은 Target domain의 데이터에 전혀 접근하지 않고도, 여러 Source domain에서 학습한 지식을 바탕으로 처음 보는(unseen) 환경에서도 강건하게 작동하는 **Domain-Agnostic Person Matching**을 달성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 DG-ReID 분야의 첫 번째 체계적인 서베이(Survey)를 제공한다는 점에 있다. 주요 기여 사항은 다음과 같다.

- **DG-ReID 방법론의 분류 체계(Taxonomy) 정립**: 기존 연구들을 Normalization, Mixture-of-Experts, Memory, Meta-learning, Data-driven, CLIP-based, 그리고 기타 방법론의 7가지 범주로 분류하고 분석하였다.
- **시스템 아키텍처 분석**: Backbone network(ResNet, MobileNet, ViT)의 특성과 다중 소스 입력 구성(Multi-source input configuration) 전략을 상세히 검토하였다.
- **교차 작업 관점의 분석(Cross-Task Perspective)**: DG-ReID와 개념적 유사성이 높은 Visible-Infrared ReID (VI-ReID)와의 비교 분석을 통해 도메인 갭을 극복하기 위한 공통적 전략과 차이점을 고찰하였다.
- **실험 프로토콜 및 벤치마크 정리**: DG-ReID의 성능을 객관적으로 평가하기 위한 3가지 평가 프로토콜과 주요 데이터셋의 통계 및 최신 SOTA(State-of-the-art) 성능 비교를 제시하였다.

## 📎 Related Works

논문은 ReID의 발전 과정을 세 단계로 구분하여 설명하며 DG-ReID의 필요성을 역설한다.

1. **Traditional ReID**: 학습과 테스트 도메인이 동일하다고 가정하며, 주어진 도메인 내에서 변별력 있는 특징(discriminative features)을 학습하는 데 집중한다. 그러나 도메인 시프트에 매우 취약하다.
2. **Domain-Adaptive ReID (DA-ReID)**: Target domain의 데이터를 학습 과정에 포함하여 Source와 Target 간의 특징 분포를 정렬(align)한다. 하지만 특정 Target domain에 과적합(overfit)될 위험이 있으며, 타겟 데이터 접근이 필수적이라는 한계가 있다.
3. **Domain-Generalizable ReID (DG-ReID)**: Target domain 데이터 없이 여러 Source domain에서 **Domain-invariant(도메인 불변)** 특징을 학습하여, 완전히 새로운 환경에서도 안정적인 성능을 내는 것을 지향한다.

## 🛠️ Methodology

본 논문은 DG-ReID를 구현하기 위한 방법론을 모듈 중심으로 분석한다.

### 1. 시스템 아키텍처

- **Backbone**: 가장 널리 쓰이는 ResNet-50 외에도, 연산 효율성과 과적합 방지를 위한 MobileNetV2, 그리고 전역 문맥 파악 및 강력한 일반화 성능을 위한 ViT (Vision Transformer)가 사용된다.
- **입력 구성**: 여러 소스 데이터셋을 하나의 풀로 합쳐 학습하거나, 각 도메인별로 별도의 네트워크/어댑터를 사용하는 방식을 취한다.

### 2. 핵심 DG 모듈

#### ① Normalization-based

Batch Normalization (BN)은 도메인 특유의 통계량을 캡처하는 경향이 있고, Instance Normalization (IN)은 스타일 변형을 제거하여 도메인 불변 특징을 추출하는 경향이 있다.

- **Fixed Combination**: IBN-Net과 같이 BN과 IN을 고정된 비율로 섞어 사용한다.
- **Learnable Combination**: MetaBIN과 같이 채널별로 학습 가능한 가중치 $\alpha$를 도입하여 BN과 IN의 기여도를 동적으로 조절한다.
$$y = \alpha(\gamma_{BN} \cdot \hat{x}_{BN} + \beta_{BN}) + (1-\alpha)(\gamma_{IN} \cdot \hat{x}_{IN} + \beta_{IN})$$

#### ② Mixture-of-Experts (MoE)-based

여러 전문 네트워크(Expert)를 두고 게이팅 메커니즘을 통해 최적의 전문가를 선택적으로 활용한다.

- **Independent Experts**: 도메인별 독립 전문가를 두고, 쿼리 특징과 클래스 프로토타입 간의 유사도 $\omega_j$를 기반으로 가중 합산한다.
$$\tilde{f} = \sum_{j \neq k} \sigma(\omega_j) \cdot f_j$$
- **Shared Experts**: 파라미터를 공유하되 BN 레이어만 분리하여 효율성과 일반화 성능을 동시에 잡는다.

#### ③ Memory-based

특징 맵이나 분류기 가중치를 메모리 뱅크에 저장하여 참조함으로써 정밀한 매칭을 수행한다.

- **Momentum Update**: 새로운 특징을 저장할 때 지수 이동 평균(EMA)을 사용하여 점진적으로 업데이트한다.
$$w_{j,i}^p \leftarrow (1-m) \cdot w_{j,i}^p + m \cdot f(x_{j,i}^p)$$

#### ④ Meta-learning-based

학습 과정을 에피소드 형태로 구성하여 '학습하는 법을 학습'함으로써 unseen domain에 대한 적응력을 높인다.

- **Domain-level Learning**: 소스 도메인을 Meta-train과 Meta-test 세트로 나누어 시뮬레이션한다.
- **Loss Function**: 기본적으로 Cross-entropy loss와 함께, 앵커($f_a$), 긍정($f_p$), 부정($f_n$) 샘플 간의 거리를 조절하는 Triplet loss를 사용한다.
$$L_{triplet} = \max(0, \|f_a - f_p\| - \|f_a - f_n\| + \text{margin})$$

#### ⑤ Data-driven Learning-based

- **Synthetic Data**: 3D 메시 기반 생성기나 GAN을 통해 다양한 포즈, 조명, 배경의 가상 데이터를 생성하여 데이터 다양성을 확보한다.
- **Unlabeled Data**: 레이블 없는 비디오 데이터에서 Bipartite matching을 통해 의사-긍정(pseudo-positive) 쌍을 찾아 학습에 활용한다.
- **Augmentation**: 정렬(Alignment)과 균일성(Uniformity)을 동시에 최적화하여 특징 공간의 붕괴를 막고 일반화를 돕는다.

#### ⑥ CLIP-based

Vision-Language Model의 텍스트-이미지 정렬 능력을 활용한다.

- **Text-Guided Learning**: 도메인 불변 프롬프트와 도메인 관련 프롬프트를 설계하고, 이미지 특징이 불변 프롬프트에 더 가깝게 위치하도록 유도한다.
$$L_{apn} = \max(0, \text{Sim}(f_a^I, f_p^T) - \text{Sim}(f_a^I, f_n^T) + m)$$

#### ⑦ 기타 방법론

- **Gradient-Alignment**: 주 작업(ID 분류)과 보조 작업(Saliency detection)의 그래디언트를 직교하게 만들어 도메인 노이즈를 제거한다.
- **Federated Stylization**: 연합 학습 환경에서 데이터 프라이버시를 지키며 스타일 변형 데이터를 생성해 일반화 성능을 높인다.

## 📊 Results

### 1. 평가 데이터셋 및 지표

- **데이터셋**: Market-1501, MSMT17 등의 대규모 데이터셋을 학습에 사용하고, PRID2011, VIPeR, iLIDS, GRID 등의 소규모 데이터셋을 unseen target으로 사용하여 일반화 능력을 평가한다.
- **지표**: Rank-1 Accuracy(최상위 결과가 정답일 확률)와 mAP(평균 정밀도)를 사용한다.

### 2. 평가 프로토콜

- **Protocol-1**: 소스 도메인의 모든 데이터를 학습에 사용하고 4개의 소규모 데이터셋에서 테스트한다.
- **Protocol-2**: 4개의 주요 데이터셋 중 하나를 Target으로 남겨두고 나머지 3개로 학습한다 (가장 엄격한 설정).
- **Protocol-3**: Protocol-2와 유사하나, 소스 도메인의 테스트 세트까지 학습에 모두 포함시켜 현실적인 상한선을 측정한다.

### 3. 주요 결과 분석

- **SOTA 성능**: 최근 제안된 **ReNorm** 방법론이 Protocol-1에서 평균 mAP와 Rank-1 모두에서 가장 우수한 성능을 보였다.
- **도메인별 난이도**: MSMT17 데이터셋이 특히 어려운 것으로 나타났다. Attention Map 분석 결과, MetaBIN은 배경 노이즈에 민감하게 반응하는 반면, ACL은 인물 중심의 안정적인 attention을 보였으나 세부 특징(가방 등)을 놓치는 경향이 있었다. 이는 너무 광범위하거나 너무 좁은 attention 모두 일반화에 방해가 될 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 논문은 DG-ReID의 현주소를 분석하며 다음과 같은 통찰을 제시한다.

- **강점**: Normalization의 동적 조절, Meta-learning을 통한 도메인 시뮬레이션, CLIP의 시맨틱 가이드 등이 도메인 갭을 줄이는 데 효과적임을 확인하였다.
- **한계 및 미해결 과제**:
  - **Attention 최적화**: 인물 중심의 특징을 추출하면서도 동시에 식별 가능한 세부 정보를 놓치지 않는 정교한 attention 메커니즘이 필요하다.
  - **데이터 의존성**: 합성 데이터나 레이블 없는 데이터의 활용도가 높으나, 여전히 실제-가상 간의 간극(sim-to-real gap)이 존재한다.
- **비판적 해석**: 현재의 DG-ReID 연구는 주로 정지 영상(Still image)에 치중되어 있다. 실제 감시 시스템은 비디오 스트림 기반이므로, 시간적(temporal) 일관성을 활용한 DG 연구로의 확장이 필수적이다.

## 📌 TL;DR

본 논문은 Target domain 데이터 없이도 강건한 Person Re-identification을 수행하는 **Domain Generalization (DG-ReID)** 기술을 총망라한 최초의 체계적 서베이 보고서이다. 7가지 핵심 방법론(Normalization, MoE, Memory, Meta-learning, Data-driven, CLIP, 기타)으로 기술을 분류하고, SOTA 모델인 ReNorm 등의 성능을 분석하였다. 특히 비디오 기반 DG, 카메라 인식(Camera-aware) 학습, 주파수 도메인 분석, 그리고 Person Search로의 확장 가능성을 제시함으로써 향후 연구 방향을 명확히 정의하였다. 이 연구는 실무 환경에서 데이터 수집 제약 없이 배포 가능한 ReID 시스템 구축을 위한 이론적 토대를 제공한다.
