# Vision-and-Language Pretrained Models: A Survey

Siqu Long, Feiqi Cao, Soyeon Caren Han, and Haiqin Yang (2022)

## 🧩 Problem to Solve

컴퓨터 비전(Computer Vision, CV)과 자연어 처리(Natural Language Processing, NLP) 분야에서는 각각 사전 학습된 모델(Pretrained Models)이 비약적인 발전을 이루었다. 이러한 흐름에 따라 두 모달리티(Modality)의 데이터를 통합하여 공동 표현(Joint Representation)을 학습하는 Visual-Language Pretrained Models(VLPMs) 연구가 활발히 진행되고 있다.

본 논문이 해결하고자 하는 문제는 VLPM 분야의 급격한 성장으로 인해 파편화된 연구 결과들을 체계적으로 정리하고 분석할 수 있는 포괄적인 가이드라인이 부족하다는 점이다. 따라서 본 연구의 목표는 VLPM의 일반적인 아키텍처, 데이터 인코딩 방법, 상호작용 모델, 사전 학습 및 미세 조정(Fine-tuning) 전략을 종합적으로 분석하여 CV 및 NLP 연구자들에게 학술적인 통찰을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 VLPM의 전체 파이프라인을 네 가지 주요 구성 요소로 정의하고, 이를 중심으로 기존 모델들을 체계적으로 분류한 것이다.

1. **범용 아키텍처 정의**: VLPM의 구조를 $\text{Raw Input Data} \rightarrow \text{V/L Representation} \rightarrow \text{V-L Interaction Model} \rightarrow \text{V-L Representation}$의 단계로 정형화하였다.
2. **인코딩 방법론 분석**: 언어 및 시각 데이터가 어떻게 토큰화되고 임베딩되는지, 그리고 각 모달리티 내부의 정보를 처리하는 Intra-modality processing의 역할을 분석하였다.
3. **상호작용 모델의 분류**: Self-attention, Co-attention, 그리고 VSE(Visual-Semantic Embedding) 기반 모델로 나누어 각 구조의 특성과 효율성을 비교하였다.
4. **학습 전략 및 평가 체계 정리**: 사용되는 데이터셋의 성격(In-domain vs Out-of-domain)과 사전 학습 작업(Pretraining Tasks), 그리고 이를 평가하기 위한 다양한 다운스트림 태스크를 집대성하였다.

## 📎 Related Works

기존의 관련 연구들 또한 시각-언어 학습을 다루었으나, 본 논문은 다음과 같은 차별점을 가진다.

- **기존 연구의 한계**: 이전의 서베이 논문들은 일부 특정 태스크만을 부분적으로 검토하거나, 시스템적인 분석에만 치중하여 전체적인 VLPM의 생태계를 포괄하지 못했다.
- **본 논문의 차별점**: 본 연구는 단순한 태스크 나열을 넘어, 입력 데이터의 인코딩부터 최종 표현 생성에 이르기까지의 전체 흐름을 분석하며, 특히 사전 학습 전략과 전이 학습(Transfer Learning)의 관계를 심도 있게 다룬 최초의 포괄적 리뷰 논문임을 주장한다.

## 🛠️ Methodology

논문에서 제시하는 VLPM의 일반적인 시스템 구조와 방법론은 다음과 같다.

### 1. 입력 데이터 인코딩 (Input Data Encoding)

- **Language Encoding**: 대부분 BERT 형식의 입력 표현을 사용한다. 이는 $\text{Token Embedding} + \text{Position Embedding} + \text{Segment(Modality) Embedding}$의 합으로 구성된다. 일부 모델은 시각적 객체 태그를 텍스트에 추가하는 Early-fusion 전략을 사용하기도 한다.
- **Vision Encoding**: 이미지를 시퀀스 형태로 변환하여 BERT 스타일의 표현으로 만든다.
  - **RoI-based**: Faster R-CNN 등을 사용하여 객체 영역(Region of Interest)의 특징을 추출한다.
  - **Patch/Pixel-based**: 이미지를 격자 형태의 패치나 픽셀 단위로 분할하여 처리하며, 최근에는 ViT와 같이 단순 선형 투영(Linear Projection)을 사용하는 경향이 있다.
  - **Spatial Position Embedding**: 시각 토큰의 공간적 관계를 정의하기 위해 2D 좌표 기반의 임베딩을 추가한다.

### 2. 시각-언어 상호작용 모델 (V-L Interaction Model, V-LIM)

두 모달리티의 정보를 융합하는 방식에 따라 세 가지로 분류한다.

- **Self-attention-based**: 시각-언어 표현을 단순히 연결(Concatenate)하여 하나의 Transformer 블록에 입력한다. 단일 스트림(Single-stream) 구조가 많으며, $\text{[CLS]}$ 토큰의 출력을 전체의 공동 표현으로 사용한다.
- **Co-attention-based**: 각 모달리티를 위한 별도의 스트림을 유지하며, 특정 Cross-attention 층에서만 정보를 교환한다. 이는 이중 스트림(Double-stream) 구조에 해당하며 모달리티별 특성을 더 잘 유지할 수 있다.
- **VSE-based (Dual-encoder)**: 두 모달리티를 완전히 독립적인 인코더로 처리한 후, 공유된 임베딩 공간(Shared VSE space)에서 유사도를 측정한다. 계산 비용이 매우 낮아 대규모 검색(Retrieval) 작업에 유리하다.

### 3. 사전 학습 전략 (Pretraining)

사전 학습은 주로 다음 세 가지 핵심 작업(Objective)을 통해 수행된다.

- **Cross-modal Masked Language Modeling (CMLM)**: 텍스트의 일부 토큰을 마스킹하고 시각 정보와 주변 텍스트를 이용해 예측한다. 손실 함수로는 Negative Log-Likelihood (NLL)가 사용된다.
- **Cross-modal Masked Region Modeling (CMRM)**: 시각 토큰을 마스킹하고 이를 복원한다. 방법론에 따라 객체 클래스를 예측하는 $\text{CMRM}_C$, 확률 분포를 근사하는 $\text{CMRM}_D$, 특징 벡터를 직접 회귀하는 $\text{CMRM}_R$으로 나뉜다.
- **Cross-modal Alignment (CA)**: 이미지-텍스트 쌍이 일치하는지를 판단한다. 융합 기반 모델은 이진 분류(Binary Classification) 문제를 풀며, Dual-encoder 모델은 대조 학습(Contrastive Learning)을 통해 랭킹 문제를 해결한다.

## 📊 Results

본 논문은 서베이 논문으로서 개별 모델의 수치적 결과보다는, 다양한 VLPM들이 적용된 다운스트림 태스크의 경향성과 성능 요인을 분석한다.

### 1. 평가 데이터셋 및 지표

- **V-L Understanding**: VQA(Visual Question Answering), CMR(Cross Modal Retrieval), VE(Visual Entailment), VCR(Visual Commonsense Reasoning), REC(Referring Expression Comprehension) 등이 포함된다.
- **V-L Generation**: Image Captioning (IC), Multi-modal Machine Translation (MMT) 등이 있으며, 주로 NLL loss를 최소화하는 방식으로 평가한다.

### 2. 주요 분석 결과

- **데이터셋의 영향**: 웹에서 수집한 대규모 데이터(Out-of-domain, 예: CC, SBU)와 특정 도메인 데이터(In-domain, 예: COCO, VG)를 함께 사용할 때 전이 학습 성능이 향상된다.
- **구조적 효율성**: Self-attention 기반 모델이 Co-attention 기반 모델보다 VQA 등에서 더 나은 성능을 보이는 경향이 있다.
- **인코딩의 영향**: RoI 기반 인코딩보다 패치/픽셀 기반 인코딩이 추론 속도 면에서 훨씬 유리하며, 최근의 트렌드로 자리 잡고 있다.

## 🧠 Insights & Discussion

### 강점 및 가치

본 논문은 복잡한 VLPM의 아키텍처를 '데이터 $\rightarrow$ 표현 $\rightarrow$ 상호작용 $\rightarrow$ 결과'라는 명확한 파이프라인으로 정리함으로써, 새로운 모델을 설계하려는 연구자들에게 체계적인 체크리스트를 제공한다. 특히 단순한 모델 나열이 아니라 인코딩 수준부터 상호작용 방식까지 계층적으로 분석한 점이 우수하다.

### 한계 및 비판적 해석

- **정량적 비교의 부족**: 다양한 모델을 분류하였으나, 동일한 조건에서의 벤치마크 성능 비교표가 부족하여 어떤 구조가 절대적으로 우월한지 판단하기 어렵다.
- **최신 트렌드의 반영**: 논문 작성 시점의 최신 모델들은 반영되었으나, 이후 등장한 초거대 다중모달 모델(LMM)의 특성을 완전히 포괄하기에는 구조적 분석이 다소 전통적인 Transformer 기반에 치중되어 있다.

### 향후 연구 방향

논문은 다음 세 가지 방향을 제시한다.

1. **Fine-grained Alignment**: 단순한 마스킹을 넘어 임베딩 수준에서 시각-언어 특징을 명시적으로 정렬하는 방법 연구가 필요하다.
2. **Multi-task Synergy**: 다양한 사전 학습 작업 간의 시너지 효과와 최적의 작업 조합/순서에 대한 체계적인 분석이 필요하다.
3. **Training Metrics**: 다운스트림 태스크에서만 성능을 확인하는 현재의 방식은 비용이 너무 크므로, 학습 과정 중에 성능을 예측할 수 있는 새로운 지표(예: Perplexity)의 도입이 필요하다.

## 📌 TL;DR

본 논문은 시각-언어 사전 학습 모델(VLPM)의 전체 생태계를 분석한 포괄적인 서베이 보고서이다. 입력 인코딩, 상호작용 모델(Self/Co-attention, VSE), 사전 학습 작업(CMLM, CMRM, CA)으로 이어지는 표준 프레임워크를 제시하였다. 이 연구는 향후 다중모달 모델을 설계할 때 어떤 구성 요소를 선택해야 하는지에 대한 이론적 기반을 제공하며, 특히 세밀한 정렬(Fine-grained alignment)과 효율적인 학습 지표 개발의 중요성을 시사한다.
