# SAM-CP: Marrying SAM with Composable Prompts for Versatile Segmentation

Pengfei Chen et al. (2025)

## 🧩 Problem to Solve

본 논문은 Vision Foundation Model인 Segment Anything Model (SAM)을 시맨틱 인식 기반의 세그멘테이션(Semantic-aware segmentation) 작업에 적용할 때 발생하는 주요 문제들을 해결하고자 한다. SAM은 이미지 픽셀을 패치(patch) 형태로 그룹화하는 뛰어난 일반화 능력을 갖추고 있으나, 다음과 같은 두 가지 핵심적인 한계가 존재한다.

첫째, SAM은 각 패치에 대한 시맨틱 레이블(semantic label) 정보를 제공하지 않는다. 이로 인해 단순한 픽셀 그룹화를 넘어 특정 클래스를 식별하는 시맨틱 세그멘테이션을 수행하기 위해서는 추가적인 메커니즘이 필요하다.

둘째, SAM은 하나의 인스턴스를 여러 개의 하위 패치로 과분할(over-segmentation)하는 경향이 있다. 기존 연구들은 단순히 각 패치에 레이블을 할당하려 했으나, 이렇게 분할된 패치들 중 어떤 것들이 동일한 인스턴스에 속하는지를 판단하는 문제는 여전히 까다로운 과제로 남아 있다.

따라서 본 논문의 목표는 SAM이 생성한 패치들을 효과적으로 레이블링하고 병합함으로써, 하나의 모델로 시맨틱, 인스턴스, 그리고 파놉틱 세그멘테이션(panoptic segmentation)을 모두 수행할 수 있는 다재다능한 프레임워크인 SAM-CP를 구축하는 것이다.

## ✨ Key Contributions

SAM-CP의 핵심 아이디어는 SAM의 결과물 위에 두 가지 유형의 **Composable Prompts (조합 가능한 프롬프트)**를 도입하여 세그멘테이션 작업을 계층적으로 해결하는 것이다.

1. **Prompt I (Semantic Labeling):** 주어진 텍스트 레이블 $T$와 SAM 패치 $P$가 주어졌을 때, 해당 패치가 해당 텍스트 클래스에 부합하는지를 판단한다.
2. **Prompt II (Instance Merging):** 동일한 텍스트 레이블로 분류된 두 개의 패치 $P_1$과 $P_2$가 있을 때, 이들이 동일한 인스턴스에 속하는지를 판단한다.

이 두 프롬프트를 조합함으로써, Prompt I만으로는 시맨틱 세그멘테이션을, Prompt I과 II를 모두 사용하면 인스턴스 및 파놉틱 세그멘테이션을 달성할 수 있다. 또한, 단순한 나열식 계산으로 인한 연산 복잡도($O(N^2)$) 문제를 해결하기 위해 **Unified Affinity Framework (통합 어피니티 프레임워크)**를 제안하여 쿼리 기반으로 패치들을 효율적으로 병합한다.

## 📎 Related Works

최근 SAM과 같은 강력한 기반 모델을 다양한 도메인에 적용하려는 시도가 많았다. 관련 연구는 크게 두 갈래로 나뉜다.

첫째, Grounded-SAM과 같이 별도의 독립적인 모델(예: Grounding-DINO)을 사용하여 제안 영역(proposal)을 생성하고, SAM은 이를 정교화(refinement)하는 용도로만 사용하는 방식이다. 이는 SAM 자체의 기반 모델로서의 기능을 약화시킨다는 단점이 있다.

둘째, SSAM, Semantic-SAM, SAM-CLIP과 같이 SAM이 생성한 각 패치에 시맨틱 레이블을 할당하려는 시도이다. 그러나 이들은 앞서 언급한 과분할 문제, 즉 여러 패치가 하나의 인스턴스를 구성하는 상황을 효과적으로 처리하지 못하는 한계가 있다.

SAM-CP는 단순한 레이블 할당을 넘어 '인스턴스 병합'이라는 관점을 도입하고, 이를 효율적인 어피니티 전파 메커니즘으로 구현함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

SAM-CP는 입력 이미지에서 SAM이 추출한 패치 집합 $P = \{P_1, P_2, ..., P_N\}$을 입력으로 받는다. 이후 **Patch Encoder**와 **Unified Affinity Decoder**를 거쳐 최종 마스크를 생성한다.

### 1. Patch Encoder

각 패치 $P_n$에 대해 ResNet50 또는 Swin-L 백본과 RoIAlign 연산자를 적용하여 기본 특징 벡터 $\tilde{f}_n$을 추출한다. 특히 배경 영역을 마스킹하여 더 정확한 특징을 뽑아내는 **MaskRoI** 연산자를 사용하며, 이후 MLP와 6개의 Multi-head Self-Attention 레이어를 통해 최종 패치 특징 $f_n$을 생성한다.

### 2. Unified Affinity Decoder

이 모듈은 쿼리와 키(key) 간의 어피니티(affinity, 유사도/친밀도)를 계산하여 패치들을 병합한다.

- **Queries:** 두 종류의 쿼리를 설정한다.
  - **Semantic Queries ($e^S_c$):** CLIP 텍스트 인코더를 통해 클래스 텍스트 레이블을 벡터화한 것이다.
  - **Instance Queries ($e^I_n$):** 각 패치가 하나의 인스턴스가 될 수 있다고 가정하는 'Patch-as-Query (PasQ)' 방식을 사용하여 패치 특징 $f_n$으로 초기화한다.
- **Affinity Propagation:** 쿼리 $Q$와 패치 특징 $K, V$ 사이의 어피니티 행렬 $A$ (크기 $M \times N$)를 계산한다. 여기서 $A_{m,n}$은 패치 $P_n$이 쿼리 $Q_m$에 속할 확률을 의미한다.
- **핵심 메커니즘:**
  - **Dynamic Cross-Attention (DCA):** 어피니티 행렬 $A$를 동적 마스크로 사용하여 높은 어피니티를 가진 패치의 특징만 추출한다.
  - **Affinity Refinement (AR):** 코사인 유사도를 이용하여 어피니티 행렬 $A$를 점진적으로 업데이트한다.
  - **Query Enhancement (QE):** 쿼리의 특징과 해당 쿼리가 가진 높은 어피니티 영역의 RoI 특징을 융합하여 쿼리 임베딩을 강화한다.

### 3. 학습 목표 및 손실 함수

모델은 시맨틱 수준과 인스턴스 수준의 감독 신호를 모두 받는다.

**시맨틱 수준 감독 (Semantic-level Supervision):**
쿼리 $Q_m$과 텍스트 임베딩 $e_c$ 사이의 유사도 $S^{cls}_{m,c}$를 계산하며, Focal Loss를 사용하여 분류 오차를 학습한다.
$$S^{cls}_{m,c} = \frac{1}{s} \cdot \hat{Q}_m^\top \cdot \hat{e}_c + b$$
$$L_{cls} = \frac{1}{M} \sum_{c=1}^{C} \sum_{m=1}^{M} FL(\sigma(S^{cls}_{m,c}), I[c^*_m = c])$$

**인스턴스 수준 감독 (Instance-level Supervision):**
헝가리안 알고리즘(Hungarian algorithm)을 통해 예측 쿼리와 실제 인스턴스를 매칭한다. 매칭 결과에 따라 정답 어피니티 행렬 $B$를 구성하고, Mask Focal Loss($L_{mfl}$)와 Dice Loss($L_{dice}$)를 적용한다.
$$L_{mfl} = \frac{1}{M^*} \sum_{m=1}^{M} \frac{\epsilon_m}{\max(|B_m|_0, 1)} \cdot \sum_{n=1}^{N} FL(A_{m,n}, B_{m,n})$$
$$L_{dice} = \frac{1}{M^*} \sum_{m=1}^{M} \epsilon_m \cdot Dice(A_m, B_m)$$

최종 손실 함수는 다음과 같다.
$$L_{all} = \lambda_{cls} L_{cls} + \lambda_{mfl} L_{mfl} + \lambda_{dice} L_{dice}$$

## 📊 Results

### 실험 설정

- **데이터셋:** COCO-Panoptic, ADE20K, Cityscapes.
- **평가 지표:** PQ (Panoptic Quality), SQ (Segmentation Quality), RQ (Recognition Quality), AP (Average Precision), mIoU.
- **Open-Vocabulary 설정:** 학습 시 보지 못한 클래스에 대해 대응하기 위해 Frozen CLIP 이미지 인코더를 사용하며, 닫힌 도메인 점수와 CLIP 점수를 $\kappa=0.4$ 비율로 결합하여 최종 점수를 산출한다.

### 주요 결과

1. **Open-Vocabulary Segmentation:** Table 1에 따르면, SAM-CP는 COCO$\rightarrow$ADE20K 및 ADE20K$\rightarrow$COCO 실험에서 이전 SOTA 모델인 FC-CLIP과 FrozenSeg를 모든 지표(PQ, SQ, RQ, AP)에서 능가하였다. 특히 Cityscapes 데이터셋에서 매우 높은 인스턴스 세그멘테이션 성능을 보였다.
2. **Closed-Domain Segmentation:** Table 2 결과, SAM-CP는 ResNet-50 및 Swin-L 백본 모두에서 경쟁력 있는 성능을 보였으며, 특히 인스턴스 레벨 인식(PQ, AP)에서 MaskFormer 등보다 우수한 성능을 기록하였다.
3. **특징 분석:** t-SNE 시각화를 통해 SAM-CP가 학습한 특징들이 SAM의 원본 특징보다 훨씬 더 시맨틱하게 구분(discriminative)되어 군집화됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점

SAM-CP는 SAM의 강력한 '픽셀 그룹화' 능력과 제안된 '조합 가능한 프롬프트'의 '시맨틱 인식' 능력을 성공적으로 분리하여 결합하였다. 이를 통해 하나의 통합된 파이프라인으로 시맨틱, 인스턴스, 파놉틱 세그멘테이션을 모두 수행할 수 있는 범용성을 확보하였다.

### 한계 및 비판적 해석

본 논문에서 명시한 가장 큰 한계는 **SAM의 제안 영역(proposal) 품질에 대한 의존성**이다. SAM-CP는 SAM이 생성한 패치들을 병합하는 방식이므로, 만약 SAM이 객체를 아예 찾지 못했거나(missing rate) 처음부터 잘못 병합한 경우, 이후의 프롬프트 과정만으로는 이를 회복할 수 없다. 실제로 Mask DINO와 비교했을 때 SAM의 missing rate가 더 높으며, 이것이 closed-domain에서 SOTA 성능 달성에 걸림돌이 됨을 분석하였다.

또한, 추론 속도가 SAM의 속도에 종속된다는 점이 한계로 지적된다. 하지만 저자들은 SAM2와 같은 더 빠르고 강력한 모델이 나오면 프레임워크를 그대로 이식할 수 있는 모듈형 구조임을 강조하며, 실제로 SAM2-CP를 통해 성능 향상과 속도 개선 가능성을 보여주었다.

## 📌 TL;DR

SAM-CP는 SAM의 패치들을 시맨틱하게 분류하고(Prompt I), 동일 인스턴스로 병합하는(Prompt II) 두 가지 조합 가능한 프롬프트를 도입한 프레임워크이다. 이를 효율적으로 구현하기 위해 쿼리 기반의 통합 어피니티 프레임워크를 사용하여 연산 효율성을 높였으며, 특히 **Open-Vocabulary Panoptic Segmentation에서 SOTA 성능을 달성**하였다. 비록 SAM의 초기 패치 생성 품질이라는 상한선(upper bound)이 존재하지만, Vision Foundation Model에 다중 입도의 시맨틱 인식 능력을 부여하는 새로운 방법론을 제시했다는 점에서 가치가 높다.
