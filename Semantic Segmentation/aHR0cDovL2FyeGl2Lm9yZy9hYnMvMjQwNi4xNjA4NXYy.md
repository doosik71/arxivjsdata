# A Simple Framework for Open-Vocabulary Zero-Shot Segmentation

Thomas Stegmüller, Tim Lebailly, Nikola Dukic, Behzad Bozorgtabar, Tinne Tuytelaars, Jean-Philippe Thiran

## 🧩 Problem to Solve

기존의 시맨틱 세분화(Semantic Segmentation) 방법은 방대한 양의 세밀한 주석을 요구하며, 미리 정의된 클래스에만 작동하는 폐쇄형 어휘(closed-vocabulary) 방식이라 새로운 클래스에 대한 일반화 능력이 떨어집니다. 비전-언어 대조 학습(vision-language contrastive learning) 모델(예: CLIP)은 제로샷 분류(zero-shot classification)에는 뛰어나지만, 캡션에 지역화(localization) 단서가 부족하고 이미지 표현 학습과 교차 모달 정렬(cross-modality alignment)이 얽혀 있어 제로샷 개방형 어휘 세분화(open-vocabulary zero-shot segmentation)와 같은 고밀도 작업에는 취약합니다.

## ✨ Key Contributions

* **단순성 및 호환성**: SimZSS는 최소한의 하이퍼파라미터로 설계되어 소규모의 잘 정리된 데이터셋과 대규모의 노이즈가 많은 데이터셋 모두와 높은 호환성을 가집니다.
* **강건성**: 다양한 사전 학습된 비전 타워(vision tower)를 지원하며, 지도 학습 및 자기 지도 학습 기반의 비전 백본 사전 학습 방식을 모두 수용합니다.
* **고효율성**: 시각적 표현 학습을 교차 모달 개념 수준 정렬(cross-modality concept-level alignment)로부터 분리함으로써 높은 계산 및 데이터 효율성을 달성합니다.
* **최고 성능**: COCO Captions 데이터셋으로 8개의 GPU에서 15분 이내에 학습했을 때, 8개 벤치마크 데이터셋 중 7개에서 최신 기술(state-of-the-art) 결과를 달성했습니다.

## 📎 Related Works

* **개방형 어휘 학습 (Open-vocabulary learning)**:
  * CLIP (Radford et al., 2021) 및 LiT (Zhai et al., 2022)와 같은 비전-언어 대조 학습 모델은 대규모 이미지-캡션 쌍을 활용하여 분류 능력은 우수하지만, 밀도 높은(dense) 작업에는 성능이 부족합니다.
  * LiT는 사전 학습되고 고정된 비전 모델에 텍스트 인코더를 정렬하여 계산 및 데이터 효율성을 높였습니다.
* **개방형 어휘 세분화 (Open-vocabulary segmentation)**:
  * **픽셀 수준 주석 활용**: Li et al. (2022), Ghiasi et al. (2022) 등은 세분화 마스크를 학습하고 사전 학습된 텍스트 인코더를 사용하여 일반화 가능한 분류기의 가중치를 제공하는 방법을 제안했지만, 픽셀 수준 주석 요구 사항을 완전히 해소하지 못했습니다.
  * **마스크 없는 접근 방식**:
    * PACL (Mukhoti et al., 2023)은 두 모달리티의 인터페이스에서 투영(projection)만을 학습하여 패치 수준 교차 모달 유사성(patch-level cross-modal similarities)을 부트스트랩합니다.
    * TCL (Cha et al., 2023)은 기존의 세밀한 유사성을 활용하여 텍스트 기반 마스크를 학습하고 영역-텍스트 정렬을 강화합니다.
    * GroupViT (Xu et al., 2022)는 시각적 토큰의 계층적 그룹화를 수행합니다.
    * ReCo (Shin et al., 2022)는 객체의 동시 발생(co-occurrence)을 활용하여 지역화 능력을 향상시켰습니다.
    * Wysoczańska et al. (2024), Rewatbowornwong et al. (2023)은 CLIP의 비전-언어 정렬 능력과 DINO와 같은 자기 지도 학습 비전 트랜스포머의 공간 인식을 결합합니다.

## 🛠️ Methodology

SimZSS는 주로 두 가지 원칙에 기반합니다: i) 공간 인식 능력을 가진 고정된 비전 전용 모델을 활용하고 텍스트 인코더만 정렬하며, ii) 텍스트의 이산적 특성과 언어 지식을 활용하여 캡션 내의 로컬 개념을 식별합니다.

1. **텍스트 개념 식별 (Textual Concept Identification)**:
    * 주어진 이미지-캡션 쌍 $(x_v, x_t)$에서, 언어의 문법적 규칙을 활용하여 캡션 내의 주요 주어와 목적어를 식별합니다. 이는 SpaCy의 `en_core_web_trf` 모델을 POS(Part-of-Speech) 태거로 사용하여 명사구(Noun Phrases, NPs)를 추출하는 방식으로 이루어집니다.
    * 노이즈를 제거하고 이미지에 나타날 가능성이 있는 NP만 남기기 위해, 명사구를 명사와 첫 번째 복합어(compound)로 제한하고, 사전 정의된 '개념 은행(concept bank)'에 없는 개념은 폐기합니다.
    * 각 개념에 해당하는 토큰 인덱스 $S_l$를 추적하고, 해당 인덱스의 지역 표현(local representation) $z^i_t \in R^{d_t}$를 평균하여 개념 표현 $c^l_t \in R^{d_t}$를 얻습니다:
        $$c^l_t = \frac{1}{|S_l|}\sum_{i \in S_l} z^i_t$$
    * 텍스트 개념 $c^l_t$는 선형 투영 $g: R^{d_t} \to R^{d_v}$를 통해 시각적 공간으로 매핑됩니다:
        $$\tilde{c}^l_t = g(c^l_t)$$

2. **시각 개념 식별 (Visual Concept Identification)**:
    * 식별된 텍스트 개념 $\tilde{c}^l_t \in R^{d_v}$를 사용하여 조밀한 시각적 표현(dense visual representation) $z_v \in R^{n_v \times d_v}$ (고정된 비전 인코더 $f_v$로부터 얻음)를 쿼리하여 해당 시각 개념 $c^l_v \in R^{d_v}$를 얻습니다.
    * 각 지역 시각 표현과 쿼리 텍스트 개념 간의 유사성을 계산합니다:
        $$s = \text{softmax} \left( \frac{z_v \tilde{c}^l_t}{\tau} \right)$$
        여기서 $\tau$는 온도(temperature) 파라미터입니다.
    * 유사성 기반 풀링(similarity-based pooling)을 통해 시각 개념을 얻습니다:
        $$c^l_v = z^T_v s$$

3. **교차 모달 일관성 (Cross-Modality Consistency)**:
    * **전역 일관성 (Global Consistency)**: 이미지와 캡션 간의 전역적 유사성을 보장하기 위해 CLIP과 동일한 대조 목표(contrastive objective) $L_g$를 사용합니다.
        $$L_g = -\frac{1}{2b} \sum_i \log \left( \frac{\exp(\bar{s}_{ii})}{\sum_j \exp(\bar{s}_{ij})} \right) - \frac{1}{2b} \sum_j \log \left( \frac{\exp(\bar{s}_{jj})}{\sum_i \exp(\bar{s}_{ij})} \right)$$
    * **개념 수준 일관성 (Concept-level Consistency)**:
        * 개념 수준에서 일관성을 강화하기 위해, 배치 내의 모든 텍스트 개념 $C_t \in R^{\tilde{b} \times d_t}$와 시각 개념 $C_v \in R^{\tilde{b} \times d_v}$를 사용합니다.
        * 선형 분류기 가중치 $h \in R^{k \times d_v}$는 동일한 개념의 표현을 합산하여 계산됩니다:
            $$h_i = \sum_j \mathbb{1}_{\{q_j=i\}} g(C_t)_j$$
        * $h$와 $C_v$의 열을 L2 정규화한 후, 시각 개념에 대한 확률 분포를 도출합니다:
            $$p = \text{softmax} (C_v h^T)$$
        * 쿼리 텍스트 개념과 검색된 시각 개념 간의 일관성을 보장하기 위해 교차 엔트로피 손실 $L_l$이 사용됩니다:
            $$L_l = \frac{1}{\tilde{b}} \sum_i \sum_j -\mathbb{1}_{\{q_i=j\}} \log(p_{ij})$$
    * **총 손실**: SimZSS의 전체 목표 $L_{tot}$는 전역 및 지역 일관성 목표의 가중 합입니다:
        $$L_{tot} = L_g + \lambda L_l$$
        여기서 $\lambda$는 가중치 매개변수입니다.

## 📊 Results

* **제로샷 전경 세분화 (Zero-shot Foreground Segmentation)**: SimZSS는 Pascal VOC, Pascal Context, COCO-Stuff, Cityscapes, ADE20K를 포함한 5개 표준 세분화 데이터셋에서 SOTA 결과를 달성했습니다. 특히 COCO Captions로 학습된 ViT-B 모델은 Pascal VOC에서 90.3 mIoU를 기록하며 LiT를 크게 능가했습니다.
* **제로샷 세분화 (배경 클래스 포함)**: 배경 클래스를 포함한 세분화 작업에서도 SimZSS는 LiT 대비 우수한 성능을 보였습니다. 배경 감지 메커니즘이 간단함에도 불구하고 Pascal Context, COCO-Object, Pascal VOC에서 경쟁력 있는 결과를 얻었습니다.
* **제로샷 분류 (Zero-shot Classification)**: SimZSS는 ImageNet-1k에서 LiT와 유사한 분류 성능을 유지하며, 저데이터(low-data) 환경에서는 약간의 개선을 보였습니다. 세분화 작업과는 달리 분류 작업에서는 LAION-400M과 같은 대규모 데이터셋이 COCO Captions와 같은 잘 정리된 데이터셋보다 더 유리하다는 점을 시사합니다.
* **확장성 및 효율성**: SimZSS는 LAION-400M과 같은 대규모 데이터셋에 대한 뛰어난 확장성을 보여주며, SimZSS 고유의 연산은 전체 실행 시간의 1% 미만을 차지하여 LiT와 유사한 런타임을 가지며 CLIP보다 학습 비용이 훨씬 저렴합니다.
* **개념 은행의 영향**: SimZSS는 개념 은행에 포함되지 않은 클래스(예: Pascal VOC 클래스 제외)에 대해서도 LiT보다 뛰어난 성능을 보였습니다. 이는 제안된 지역 일관성 목표가 개념 은행 내 클래스에만 국한되지 않고 일반적인 로컬 특징과 정렬되도록 텍스트 인코더를 효과적으로 훈련시킨다는 것을 나타냅니다.

## 🧠 Insights & Discussion

* **핵심 원리**: SimZSS의 성공은 시각적 표현 학습과 교차 모달 개념 수준 정렬을 분리하고, 고정된 자기 지도 학습 비전 트랜스포머의 공간 인식 능력을 활용하며, 텍스트의 이산적 특성을 활용하여 로컬 개념을 효과적으로 식별한 데 있습니다.
* **의미 수준 일관성**: 개념 수준에서 인스턴스 기반의 대조 학습 대신 의미 수준의 일관성 목표($L_l$)를 사용한 것이 고밀도 세분화 작업 성능 향상에 결정적인 역할을 했습니다.
* **데이터셋 특성**: 세분화 작업에서는 COCO Captions와 같은 잘 정리된 데이터셋이 더 효과적이며, 분류 작업에서는 LAION-400M과 같은 대규모 데이터셋이 더 유리하다는 점이 밝혀졌습니다. 이는 작업 유형에 따른 데이터셋의 특성을 이해하고 활용하는 중요성을 강조합니다.
* **일반화 능력**: SimZSS는 명시적으로 개념 은행에 포함되지 않은 클래스에 대해서도 뛰어난 일반화 능력을 보여주며, 이는 모델이 광범위한 로컬 시각적 특징과 텍스트 개념을 정렬하는 방법을 학습했음을 시사합니다.
* **제한 사항**: 현재 SimZSS의 배경 감지 메커니즘은 비교적 간단하며, 이는 추가적인 연구를 통해 개선될 수 있는 부분입니다.

## 📌 TL;DR

* **문제**: 기존 제로샷 개방형 어휘 세분화 방법은 주석 부담이 크고, 비전-언어 모델은 분류에 강하지만 지역화 단서 부족으로 세분화에 취약합니다.
* **해결책**: SimZSS는 공간 인식이 뛰어난 **사전 학습되고 고정된 비전 모델**을 활용하고, **텍스트 인코더만을 학습**시킵니다. 캡션에서 명사구를 추출하여 **로컬 개념을 식별**하고, 이 텍스트 개념과 이미지 패치 간의 **교차 모달 일관성**을 전역 및 개념 수준에서 강화합니다.
* **성과**: SimZSS는 계산 및 데이터 효율성이 매우 높으며, 표준 제로샷 세분화 벤치마크에서 **최신 기술(SOTA) 성능**을 달성합니다. 이 방법은 다양한 백본과 데이터셋에 강건하며, 심지어 **개념 은행에 없는 클래스에 대해서도 뛰어난 일반화 능력**을 보여줍니다.
