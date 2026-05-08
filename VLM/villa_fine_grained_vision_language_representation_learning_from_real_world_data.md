# ViLLA: Fine-Grained Vision-Language Representation Learning from Real-World Data

Maya Varma, Jean-Benoit Delbrouck, Sarah Hooper, Akshay Chaudhari, Curtis Langlotz (2023)

## 🧩 Problem to Solve

기존의 Vision-Language Models (VLMs), 예를 들어 CLIP이나 ALIGN은 주로 웹에서 수집된 이미지-캡션 쌍을 통해 학습된다. 이러한 데이터셋은 일반적으로 이미지 전체와 이를 묘사하는 간결한 텍스트 간의 일대일(one-to-one) 매핑을 학습하는 구조를 가진다. 그러나 의료 데이터나 제품 데이터베이스와 같은 실제 세계의 멀티모달 데이터셋은 훨씬 더 복잡한 특성을 보인다.

하나의 이미지(예: X-ray)가 여러 개의 미세 영역(fine-grained regions)과 그 영역들에 대응하는 다양한 텍스트 속성(attributes)을 포함하는 경우가 많다. 저자들은 이를 **Pairwise Complexity(쌍별 복잡도)**가 높다고 정의한다. 즉, 하나의 이미지-텍스트 쌍이 수많은 '영역-속성' 쌍으로 분해될 수 있는 상태를 의미한다.

본 논문의 목표는 이러한 높은 Pairwise Complexity를 가진 데이터셋에서 표준적인 일대일 VLM이 미세한 영역-속성 관계를 학습하는 데 겪는 어려움을 정량적으로 분석하고, 이를 해결하여 미세한 추론 능력을 향상시킬 수 있는 새로운 표현 학습 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 복잡한 이미지-텍스트 샘플을 직접 학습하는 대신, 이를 **미세한 '영역-속성' 쌍으로 분해하여 표준 VLM에 제공**하는 것이다. 이를 위해 저자들은 다음 두 가지 핵심 기여를 제시한다.

1. **Pairwise Complexity의 영향 분석**: 합성 데이터셋인 `DocMNIST`를 구축하여, 훈련 데이터의 복잡도가 증가함에 따라 표준 VLM의 미세 추론 성능이 최대 37%까지 하락함을 체계적으로 증명하였다.
2. **ViLLA 방법론 제안**: 복잡한 데이터셋에서 미세한 영역-속성 관계를 캡처하기 위한 self-supervised 학습 프레임워크인 ViLLA를 제안한다. 이는 경량화된 mapping model을 통해 데이터를 분해하고, 이를 통해 표준 VLM을 학습시키는 2단계 파이프라인으로 구성된다.

## 📎 Related Works

**One-to-One VLMs**
CLIP과 같은 모델들은 이미지 전체 임베딩과 텍스트 전체 임베딩을 대조 학습(Contrastive Learning)을 통해 정렬한다. 하지만 이러한 방식은 이미지 내의 국소적인 영역과 특정 단어 간의 세밀한 관계를 학습하는 데 한계가 있다.

**Fine-Grained Representation Learning**
미세한 정보를 학습하기 위해 사람이 직접 라벨링한 영역-텍스트 쌍을 사용하는 연구들이 있었으나, 이는 비용이 매우 많이 들며 의료와 같은 전문 분야로 확장하기 어렵다. `RegionCLIP`과 같이 사전 학습된 CLIP을 zero-shot으로 활용해 pseudo-label을 생성하는 방법이 제안되었으나, 이는 CLIP 자체의 국소화(localization) 성능 한계와 도메인 외 데이터(out-of-domain data)에 대한 취약성이라는 문제를 안고 있다.

**Learning from Real-World Multimodal Data**
의료(ConVIRT, BioViL)나 제품 데이터셋을 위한 VLM 연구들이 진행되어 왔으나, 본 논문은 특히 '데이터의 복잡도'라는 관점에서 접근하여 다양한 도메인에 적용 가능한 범용적인 해결책을 제시한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

ViLLA는 크게 두 단계의 파이프라인으로 구성된다.

### Stage 1: Mapping Image Regions to Attributes

이미지-텍스트 샘플을 미세한 영역-속성 쌍으로 분해하는 과정이다.

1. **데이터 분해**:
    * 이미지 $x_i$를 $r_i$개의 영역 $\{x_0^i, \dots, x_{r_i}^i\}$으로 분해한다. (RPN 또는 그리드 분할 방식 사용)
    * 텍스트 $t_i$를 $a_i$개의 속성 $\{t_0^i, \dots, t_{a_i}^i\}$으로 분해한다.
2. **Mapping Model 구조**:
    * **Image Encoder**: 도메인별 사전 학습된 가중치(CLIP 또는 ConVIRT)를 사용하며, RoIAlign을 통해 영역 임베딩 $e_i \in \mathbb{R}^{r_i \times d}$를 추출한다.
    * **Projection Heads**: 각 속성 $k$에 대응하는 별도의 투영 헤드 $P^k$를 둔다. 이 헤드는 $\text{Linear} \rightarrow \text{ReLU} \rightarrow \text{Linear}$ 구조이며, 특정 속성에 특화된 영역 임베딩을 생성한다.
    * **Text Encoder**: 사전 학습된 SBERT 또는 CLIP을 사용하여 속성 임베딩 $h_i^k \in \mathbb{R}^{1 \times d}$를 생성한다.
3. **학습 절차 및 손실 함수**:
    인코더들은 고정(freeze)하고 투영 헤드 $P^k$만 학습시킨다. 다음과 같은 Contrastive Loss를 사용하여 영역과 속성을 정렬한다.
    $$L(x_i, t_i) = -\sum_{k \in t_i} \log \frac{\sigma(P^k(e_i), h_i^k)}{\sigma(P^k(e_i), h_i^k) + \sum_{j=1, k \notin t_j}^{|B|} \sigma(P^k(e_j), h_i^k)}$$
    여기서 $\sigma(a, b) = \exp(\max(\langle a, b \rangle / \tau))$이다. 이 식은 이미지 $x_i$ 내의 적어도 한 영역이 속성 $k$와 높은 유사도를 가지도록 유도하며, 속성 $k$가 없는 이미지 $x_j$와의 유사도는 낮추도록 설계되었다.

### Stage 2: Learning Vision-Language Representations

Stage 1에서 학습된 모델을 사용하여 실제 VLM을 학습시키는 단계이다.

1. **영역-속성 쌍 생성**:
    투영 헤드 $P^k$와 속성 임베딩 $h_i^k$의 내적을 통해 점수 벡터 $v \in \mathbb{R}^{r_i \times 1}$를 계산하고, $\max(v) - \epsilon$ 보다 큰 점수를 가진 모든 영역에 속성 $k$를 할당한다.
2. **VLM 학습**:
    이렇게 생성된 '영역-속성' 쌍을 기존의 '이미지-텍스트' 쌍과 함께 데이터셋에 추가(augmentation)한다. 이후 표준적인 일대일 VLM 구조를 사용하여 양방향 대조 학습(Bidirectional Contrastive Loss)으로 최종 표현을 학습한다.

## 📊 Results

### 실험 설정

* **데이터셋**: `DocMNIST`(합성), `DeepFashion`(제품), `MIMIC-CXR`(의료), `COCO`(자연어 이미지)의 4가지 도메인을 사용하였다.
* **평가 작업**: Zero-shot Object Detection, Text $\rightarrow$ Region Retrieval, Region $\rightarrow$ Text Retrieval.
* **지표**: $AP_{50}$ (Detection), R-Precision, Precision@k (Retrieval).

### 주요 결과

1. **Pairwise Complexity 분석 (DocMNIST)**:
    훈련 데이터의 복잡도($c$)가 $5.0 \rightarrow 29.4$로 증가할 때, 표준 VLM의 Text $\rightarrow$ Region retrieval 성능은 **36.9%**, Region $\rightarrow$ Text retrieval 성능은 **20.5% 하락**하였다. 이는 데이터가 복잡해질수록 표준 VLM이 미세 관계를 학습하지 못함을 시사한다.
2. **Zero-shot Object Detection**:
    COCO 데이터셋에서 ViLLA는 CLIP 대비 $AP_{50}$ 기준 8.1포인트, RegionCLIP 대비 3.6포인트 향상된 성능을 보였다. 특히 작은 객체(Small objects)에 대한 성능 향상이 두드러졌다.
3. **Retrieval 성능**:
    DocMNIST에서 ViLLA는 기존 최고 baseline(CLIP-FT-Img)보다 R-Precision 기준 14.2포인트(Text $\rightarrow$ Region) 및 7.8포인트(Region $\rightarrow$ Text) 높게 나타났다. 복잡도가 높은 데이터셋일수록 ViLLA의 성능 향상 폭이 더 컸다.
4. **의료 데이터 (CheXpert 5x200)**:
    MIMIC-CXR로 학습한 ViLLA는 CheXpert 벤치마크에서 GLoRIA보다 3.8포인트 높은 정확도를 기록하였으며, 일대일 VLM인 ConVIRT나 BioViL을 크게 상회하였다.

## 🧠 Insights & Discussion

**강점 및 해석**
ViLLA는 복잡한 멀티모달 데이터를 단순히 하나의 쌍으로 처리하지 않고, self-supervised 방식으로 세분화하여 학습함으로써 표준 VLM의 효율성을 유지하면서도 미세 추론 능력을 획기적으로 끌어올렸다. 특히, 정답 라벨(ground-truth) 없이도 투영 헤드를 통한 매핑 모델을 학습시켜 pseudo-label을 생성한 점이 실용적이다. 실험 결과, 복잡도가 높은 데이터셋에서 성능 향상이 더 뚜렷하게 나타난 것은 본 모델이 해결하고자 한 'Pairwise Complexity' 문제가 실제 성능 저하의 핵심 원인이었음을 뒷받침한다.

**한계 및 미해결 과제**

1. **데이터 모달리티의 제한**: 현재 연구는 이미지-텍스트 쌍에 국한되어 있으며, 오디오, 비디오, 시계열 데이터 등 다른 모달리티로의 확장이 필요하다.
2. **정답 라벨의 부재**: MIMIC-CXR과 같은 실제 의료 데이터는 정밀한 영역-속성 정답지가 없어, 매핑 정확도를 완벽하게 검증하는 데 한계가 있다. (현재는 일부 엔티티에 대해서만 검증 수행)
3. **하이퍼파라미터 의존성**: 영역-속성 할당 시 사용되는 임계값 $\epsilon$의 설정이 매핑의 정밀도(Precision)와 재현율(Recall) 사이의 트레이드오프를 결정하며, 이는 검증 셋에 의존적인 부분이 있다.

## 📌 TL;DR

본 논문은 실제 세계의 복잡한 데이터셋(높은 Pairwise Complexity)에서 표준 VLM이 미세한 영역-속성 관계를 학습하지 못한다는 문제를 제기하고, 이를 해결하기 위한 **ViLLA** 프레임워크를 제안한다. ViLLA는 **(1) 경량 매핑 모델을 통한 영역-속성 쌍 분해 $\rightarrow$ (2) 분해된 쌍을 이용한 표준 VLM 학습**의 2단계 구조를 가지며, 이를 통해 의료, 제품, 자연어 이미지 등 다양한 도메인에서 Zero-shot Detection 및 Retrieval 성능을 크게 향상시켰다. 이 연구는 향후 정밀한 국소화가 필요한 전문 도메인의 VLM 구축에 중요한 방법론적 토대를 제공할 것으로 기대된다.
