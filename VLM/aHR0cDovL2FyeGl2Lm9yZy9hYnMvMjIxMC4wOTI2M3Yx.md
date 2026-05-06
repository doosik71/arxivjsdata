# Vision-Language Pre-training: Basics, Recent Advances, and Future Trends

Zhe Gan, Linjie Li, Chunyuan Li, Lijuan Wang, Zicheng Liu, Jianfeng Gao (2022)

## 🧩 Problem to Solve

본 논문은 최근 몇 년간 급격히 발전한 Vision-Language Pre-training (VLP) 분야의 전반적인 기술 동향을 체계적으로 정리하고 분석하는 것을 목표로 한다.

전통적인 컴퓨터 비전 및 자연어 처리 모델은 특정 작업(Task-specific)에 최적화된 설계를 통해 성능을 높였으나, 이는 데이터 효율성이 낮고 범용성이 떨어진다는 한계가 있었다. 특히 시각 정보(Vision)와 언어 정보(Language)라는 서로 다른 모달리티를 효과적으로 정렬(Alignment)하고 융합(Fusion)하여, 인간과 유사한 멀티모달 지능(Multimodal Intelligence)을 구현하는 것이 핵심적인 문제이다.

논문의 구체적인 목표는 다음과 같다.

- 이미지-텍스트, 핵심 컴퓨터 비전 작업, 비디오-텍스트라는 세 가지 주요 카테고리로 VLP 방법론을 분류하고 리뷰한다.
- 모델 아키텍처, 학습 목표(Objectives), 데이터셋의 진화 과정을 상세히 분석한다.
- 거대 파운데이션 모델(Foundation Models), 통합 모델링(Unified Modeling), In-context Few-shot Learning 등 최신 연구 주제와 산업계 적용 사례 및 한계를 논의한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 VLP의 전체 랜드스케이프를 세 가지 관점에서 통합적으로 조망했다는 점이다.

1. **VLP의 범주 확장**: VLP를 단순히 이미지-텍스트 매칭 작업에 국한하지 않고, 이미지 분류, 객체 탐지, 세그멘테이션과 같은 핵심 컴퓨터 비전 작업(Core CV tasks) 및 비디오-텍스트 작업으로 확장하여 분석하였다.
2. **기술적 진화 단계 정립**: VL 모델의 발전을 '소규모 작업 특화 모델 $\rightarrow$ 중규모 사전 학습 모델 $\rightarrow$ 대규모 파운데이션 모델'의 세 단계로 구분하여, 성능 향상의 핵심 동인이 아키텍처의 변화와 데이터 규모의 확대에 있음을 밝혔다.
3. **통합적 프레임워크 제시**: Dual Encoder와 Fusion Encoder라는 두 가지 핵심 구조를 중심으로, 다양한 사전 학습 목표(MLM, ITM, ITC, MIM)와 이들이 각각의 다운스트림 태스크에 미치는 영향을 체계적으로 정리하였다.

## 📎 Related Works

논문은 VLP 이전의 초기 VL 모델들(Early VL Models)과 최근의 VLP 모델들을 비교하며 그 차이점을 설명한다.

- **초기 작업 특화 모델 (Task-specific Methods)**: ResNet, Faster R-CNN 등으로 추출한 고정된 시각 특징과 GLoVe 등으로 추출한 텍스트 특징을 단순 융합하거나, 특정 어텐션 메커니즘(Inter-modality/Intra-modality Attention)을 설계하여 사용하였다. 이러한 방식은 특정 데이터셋에는 강력하나, 새로운 도메인으로의 전이 능력이 매우 낮았다.
- **기존 VLP 연구의 한계**: 기존의 일부 서베이 논문들은 이미지-텍스트 작업에만 집중하거나, 비디오-텍스트 작업만을 따로 다루는 경향이 있었다. 본 논문은 이를 통합하여 Core CV task까지 포함함으로써 VLP의 적용 범위를 극대화하여 다룬다.
- **차별점**: 본 연구는 단순한 모델 나열이 아니라, 'Computer Vision in the Wild'라는 개념을 도입하여 실제 환경에서의 범용적인 시각-언어 시스템 구축이라는 관점에서 최신 트렌드를 분석한다.

## 🛠️ Methodology

논문은 VLP의 방법론을 아키텍처, 학습 목표, 데이터셋의 세 가지 핵심 요소로 나누어 설명한다.

### 1. 모델 아키텍처 (Model Architectures)

VLP 모델은 크게 두 가지 구조로 분류된다.

- **Dual Encoder**: 이미지와 텍스트를 각각 독립적인 인코더로 처리하고, 최종적으로 코사인 유사도(Cosine Similarity)와 같은 단순한 연산으로 정렬한다. (예: CLIP, ALIGN)
- **Fusion Encoder**: 두 모달리티의 특징을 추출한 후, Transformer 층을 통해 깊은 상호작용(Deep Interaction)을 모델링한다. (예: UNITER, VinVL)
  - **Merged Attention**: 텍스트와 시각 특징을 단순히 연결(Concatenate)하여 하나의 Transformer에 입력한다.
  - **Co-attention**: 각각의 모달리티를 처리하는 별도의 Transformer를 두고, Cross-attention을 통해 정보를 교환한다.

### 2. 사전 학습 목표 (Pre-training Objectives)

모델이 멀티모달 표현을 학습하기 위해 사용하는 주요 손실 함수는 다음과 같다.

- **Masked Language Modeling (MLM)**: 텍스트의 일부를 $[MASK]$ 토큰으로 가리고, 이미지와 주변 텍스트를 이용해 이를 예측한다.
$$L_{MLM}(\theta) = -\mathbb{E}_{(\tilde{w}, \tilde{v})\sim D} \log P_\theta(\tilde{w}_m | \tilde{w}_{\setminus m}, \tilde{v})$$
- **Image-Text Matching (ITM)**: 이미지와 텍스트 쌍이 서로 일치하는지(Matched/Mismatched)를 이진 분류한다.
- **Image-Text Contrastive Learning (ITC)**: 배치 내에서 정답 쌍의 유사도는 높이고, 오답 쌍의 유사도는 낮추도록 학습한다.
$$L_{i2t}^{ITC}(\theta) = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(s_{i,i}/\sigma)}{\sum_{j=1}^N \exp(s_{i,j}/\sigma)}$$
- **Masked Image Modeling (MIM)**: 이미지의 일부 패치나 영역을 마스킹하고, 텍스트 정보를 이용해 이를 복구한다.

### 3. 비디오-텍스트 확장 (Video-Text VLP)

비디오 데이터의 특성인 시간적 동역학(Temporal Dynamics)을 처리하기 위해 다음과 같은 추가 기법이 사용된다.

- **Frame Order Modeling (FOM)**: 무작위로 섞인 프레임의 원래 순서를 예측함으로써 시간적 순서를 학습한다.
- **Masked Video Modeling (MVM)**: 비디오 프레임의 일부를 마스킹하고 이를 복구하는 과제를 수행한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 VLP의 발전 과정에 따른 성능 변화 추이를 분석한다.

- **VQA 성능의 비약적 향상**: VQA(Visual Question Answering) 태스크를 사례로 들었을 때, 2017년 작업 특화 모델 시대(약 66%) $\rightarrow$ 중규모 VLP 시대(약 78%) $\rightarrow$ 대규모 파운데이션 모델 시대(약 84%)로 정확도가 지속적으로 상승했음을 보여준다.
- **데이터 규모의 영향**: 학습 데이터셋의 규모가 수백만(M) 단위에서 수십억(B) 단위로 확장됨에 따라, Zero-shot 전이 능력이 비약적으로 향상되었음을 확인하였다. (예: CLIP의 400M 데이터 학습 결과)
- **핵심 비전 작업으로의 전이**: VLP 모델을 통해 이미지 분류를 '리트리벌(Retrieval)' 문제로 재정의함으로써, 기존의 Close-set 분류의 한계를 넘어 Open-vocabulary 인식이 가능해졌음을 정량적/정성적으로 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

- **패러다임의 전환**: 본 논문은 시각 인식 문제를 '분류'에서 '검색(Retrieval)'의 관점으로 바꾼 것이 Open-set 인식의 핵심임을 정확히 짚어냈다.
- **North Star의 제시**: 저자들은 좋은 VLP 모델의 기준을 '광범위한 태스크 수행 능력'과 '최소한의 비용으로 새로운 태스크에 적응하는 능력'으로 정의하며, 연구 커뮤니티가 나아가야 할 방향을 명확히 제시하였다.

### 한계 및 비판적 해석

- **데이터 오염 및 편향**: 웹에서 수집한 대규모 데이터셋(LAION 등)에는 인종차별, 성고정관념 등 유해한 콘텐츠가 포함되어 있으며, 이를 그대로 학습한 모델이 편향된 결과를 출력할 위험이 크다.
- **계산 비용의 문제**: 모델 크기가 5B, 70B 등으로 커짐에 따라 학습 및 추론 비용이 기하급수적으로 증가하며, 이는 학계보다는 산업계 중심의 연구로 치우치게 만드는 요인이 된다.
- **시간적 추론의 부족**: 비디오 VLP의 경우, 여전히 많은 모델이 단일 프레임의 정보에 의존하는 'Single-frame bias' 문제가 있으며, 진정한 의미의 시간적 추론(Temporal Reasoning) 능력은 아직 부족한 상태이다.

## 📌 TL;DR

본 논문은 이미지-텍스트, 핵심 비전 작업, 비디오-텍스트를 아우르는 **Vision-Language Pre-training (VLP)의 포괄적인 서베이 보고서**이다.

**핵심 요약:**

- **아키텍처**: Dual Encoder(빠른 검색) $\rightarrow$ Fusion Encoder(깊은 이해) $\rightarrow$ 통합 모델(Unified Model)로 진화 중이다.
- **학습 목표**: MLM, ITM, ITC, MIM 등의 조합을 통해 모달리티 간 정렬을 수행한다.
- **핵심 가치**: 대규모 사전 학습을 통해 특정 도메인에 국한되지 않는 **Open-vocabulary** 인식 능력을 확보하였으며, 이는 'Computer Vision in the Wild'를 가능케 한다.

이 연구는 향후 모든 시각/언어 작업을 하나의 가중치로 처리하는 **General-Purpose Multimodal Foundation Model (Generalist Agent)** 구축을 위한 이론적/기술적 토대를 제공하며, 특히 모델 스케일링과 효율적인 적응(Efficient Adaptation) 연구의 중요성을 시사한다.
