# Vision Language Transformers: A Survey

Clayton Fields, Casey Kennington (2023)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전(Computer Vision, CV)과 자연어 처리(Natural Language Processing, NLP)가 교차하는 영역인 시각-언어(Vision-Language, VL) 모델링의 복잡성과 어려움을 해결하고자 한다. 이미지에 대한 질문에 답하는 시각적 질의응답(Visual Question Answering, VQA)이나 이미지 캡셔닝(Image Captioning)과 같은 VL 작업은 인간에게는 매우 쉬우나 컴퓨터에게는 역사적으로 매우 도전적인 과제였다.

기존의 VL 딥러닝 모델들은 개념적으로 매우 복잡하고 좁은 범위의 용도에만 국한되는 경향이 있었다. 본 논문의 목표는 최근 Vaswani et al. (2017)이 제안한 Transformer 아키텍처를 VL 모델링에 적용하여 성능과 범용성을 크게 향상시킨 다양한 VL Transformer 모델들을 종합적으로 분석하고, 이들의 강점, 한계 및 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 파편화되어 있는 수많은 VL Transformer 모델들을 체계적으로 분류하고 분석한 광범위한 합성(Synthesis) 보고서를 제공하는 것이다. 특히 다음과 같은 설계 관점에서의 분석을 제공한다.

- **임베딩 전략 분석**: 텍스트와 시각 데이터를 모델의 특징 공간(Feature Space)으로 투영하는 다양한 방법론(Region, Grid, Patch)을 비교한다.
- **아키텍처 분류**: 시각 및 언어 모달리티 간의 상호작용 방식에 따라 모델을 Dual Encoder, Fusion Encoder(Single-Tower, Two-Tower), Combination Encoder, Encoder-Decoder 구조로 세분화하여 분석한다.
- **사전 학습(Pretraining) 태스크 정의**: Masked Language Modeling(MLM)부터 Contrastive Learning에 이르기까지, 모델이 시각-언어 정렬을 학습하기 위해 사용하는 다양한 목적 함수를 정리한다.
- **데이터셋 및 하위 작업 분석**: 모델이 사용한 데이터의 규모와 소스, 그리고 이를 통해 달성한 하위 작업(Downstream Tasks)의 성능 및 특성을 분석한다.

## 📎 Related Works

본 논문은 VL Transformer의 근간이 되는 다음과 같은 관련 연구들을 다룬다.

- **Transformer**: Vaswani et al. (2017)의 Attention 메커니즘을 기반으로 하며, RNN을 대체하여 NLP와 CV의 표준 모델이 된 배경을 설명한다.
- **NLP Transformers**: BERT와 GPT-3와 같은 사전 학습 모델들이 대규모 비라벨링 데이터를 통해 성능을 높이고 미세 조정(Fine-tuning)을 통해 전이 학습(Transfer Learning)을 수행하는 방식을 언급한다.
- **CV Transformers**: ViT(Vision Transformer)와 BEiT 등이 CNN의 유도 편향(Inductive Bias) 없이도 대규모 데이터셋을 통해 CNN에 경쟁하는 성능을 낼 수 있음을 설명한다.
- **기존 VL 모델의 한계**: Transformer 이전의 태스크 전용 모델들(DGAF, MAttNet 등)은 구조가 매우 복잡하고 범용성이 떨어져, 새로운 작업에 적용하려면 모델을 처음부터 다시 설계해야 했다는 점을 지적하며 VL Transformer의 차별성을 강조한다.

## 🛠️ Methodology

### 1. Transformer 기본 구조 및 수학적 원리

VL Transformer의 기초가 되는 Attention 메커니즘은 쿼리($Q$), 키($K$), 값($V$) 벡터를 사용하여 가중 합을 계산한다. Scaled Dot-Product Attention의 수식은 다음과 같다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-Head Attention(MHA)은 서로 다른 학습된 선형 투영을 통해 여러 개의 attention head를 병렬로 처리하여 다양한 표현 하위 공간의 정보를 동시에 포착한다.

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$

### 2. 임베딩 전략 (Embedding Strategies)

- **Region Features**: Fast R-CNN이나 YOLO와 같은 객체 탐지 네트워크를 사용하여 이미지 내 특정 객체 영역을 추출한다. 객체의 위치 정보와 특징 벡터를 결합하여 사용하지만, 객체 탐지 모델이 학습한 카테고리로 인식이 제한되고 계산 비용이 높다는 단점이 있다.
- **Grid Features**: CNN의 특징 맵(Feature Map)을 격자 형태로 추출하여 평탄화(Flatten)한다. 객체 카테고리의 제한은 없으나, 여전히 pretrained CNN에 의존하며 전처리에 많은 시간이 소요된다.
- **Patch Embeddings**: 이미지를 고정된 크기의 패치($P \times P$)로 나누고 선형 투영을 통해 임베딩한다. CNN 없이 직접 처리하므로 계산 효율성이 매우 높으며 ViT 기반 모델들(ViLT 등)에서 주로 사용된다.

### 3. 모델 아키텍처 (Model Architecture)

- **Dual Encoders**: 시각 및 텍스트 인코더가 분리되어 있으며, 최종 출력 단계에서 코사인 유사도(Cosine Similarity)와 같은 단순한 방식으로 상호작용한다 (예: CLIP, ALIGN).
- **Fusion Encoders**:
  - **Single-Tower**: 두 모달리티의 임베딩을 단순히 연결(Concatenate)하여 하나의 Transformer 스택에 입력한다 (예: ViLT, UNITER).
  - **Two-Tower**: 별도의 스택에서 처리 후 Cross-Attention 메커니즘을 통해 상호작용한다 (예: ViLBERT, LXMERT).
- **Combination Encoders**: Dual Encoder의 효율성과 Fusion Encoder의 심층적 상호작용을 모두 결합한 형태이다 (예: VLMo, ALBEF).
- **Encoder-Decoder**: 인코더가 특징을 추출하고 디코더가 텍스트를 생성하는 구조로, 이미지 캡셔닝과 같은 생성 작업에 유리하다 (예: VL-T5, OFA).

### 4. 사전 학습 태스크 (Pretraining Tasks)

- **Masked Language Modeling (MLM)**: 텍스트 일부를 $[MASK]$ 토큰으로 가리고, 시각 정보를 참고하여 이를 예측한다.
  $$L_{MLM}(\theta) = -\mathbb{E}_{(t,v) \sim D} \log P_\theta(t_m | t_{\setminus m}, v)$$
- **Masked Image Modeling (MIM)**: 이미지 패치나 영역을 마스킹하고 해당 영역의 카테고리나 특징을 예측한다.
- **Contrastive Learning**: 이미지-텍스트 쌍의 코사인 유사도를 최대화하고, 맞지 않는 쌍의 유사도는 최소화한다.
- **Image-Text Matching (ITM)**: 두 입력이 서로 일치하는지 여부를 이진 분류(Binary Classification)하는 작업이다.

## 📊 Results

본 논문은 개별 실험 결과보다는 기존 모델들의 벤치마크 성능 경향성을 분석한다.

- **VL-Alignment (Retrieval)**: CLIP, ALIGN과 같은 Dual Encoder 모델들이 압도적인 효율성과 성능을 보인다. 특히 이미지 특징을 캐싱할 수 있어 대규모 검색 작업에 최적화되어 있다.
- **VL-Understanding (VQA, NLVR2)**: 심층적인 상호작용이 가능한 Fusion Encoder와 Encoder-Decoder 모델들이 우수한 성능을 보이며, Dual Encoder는 복잡한 추론 작업에서 성능이 떨어진다.
- **VL-Text Generation**: 디코더를 갖춘 모델(CoCa, SimVLM 등)이 이미지 캡셔닝 작업에서 강점을 보이며, 최근에는 OFA나 DaVinci처럼 다양한 작업을 하나의 시퀀스-투-시퀀스(Seq2Seq) 프레임워크로 통합한 모델들이 높은 범용성을 입증했다.
- **Visual Grounding**: mDETR, Referring Transformer 등 특화 모델들이 RefCOCO와 같은 벤치마크에서 정밀한 객체-텍스트 매칭 능력을 보여준다.

## 🧠 Insights & Discussion

### 강점

사전 학습된 VL Transformer는 기존의 태스크 전용 모델들에 비해 **범용적인 표현(Generalized Representations)**을 학습한다. 이를 통해 최소한의 아키텍처 변경과 미세 조정만으로도 다양한 하위 작업에 적용할 수 있는 전이 학습 능력이 비약적으로 향상되었다.

### 한계 및 비판적 해석

- **자원 소모의 극심함**: 모델의 규모가 커짐에 따라(예: Flamingo의 80B 파라미터) 사전 학습과 추론에 필요한 계산 비용과 데이터 양이 기하급수적으로 증가한다.
- **시각-언어 정렬의 모호성**: MLM이나 ITM 같은 태스크가 실제로 시각 정보와 언어 정보의 '깊은 정렬'을 유도하는지, 아니면 단순히 언어 모델의 강력한 성능에 의존하는 것인지에 대한 의문이 남아 있다.
- **임베딩 전략의 트레이드오프**: Patch Embedding은 효율적이지만, Region Feature가 제공하는 명시적인 객체 정보(Object-level semantics)가 부족하여 객체 식별 작업에서 성능 저하가 발생할 수 있다.

### 향후 연구 방향

- **고품질 데이터 생성**: 단순 웹 크롤링 데이터의 노이즈를 줄인 고품질의 VL 데이터셋 구축이 시급하다.
- **심층적 정렬 태스크 개발**: 단순 분류를 넘어 시각적 근거(Visual Grounding)를 명확히 요구하는 사전 학습 태스크의 도입이 필요하다.
- **다양한 모달리티 확장**: 시각-언어를 넘어 오디오(Audio)나 로봇 조작(Embodied AI)을 포함한 멀티모달 모델로의 확장이 기대된다.

## 📌 TL;DR

본 논문은 수많은 VL Transformer 모델들을 임베딩 전략, 아키텍처, 사전 학습 방법론, 데이터셋 기준으로 체계적으로 정리한 종합 서베이 논문이다. VL Transformer는 전이 학습을 통해 범용성과 성능을 획기적으로 높였으나, 막대한 계산 비용과 데이터 의존성이라는 한계가 있다. 향후 연구는 단순한 모델 구조 변경보다는 고품질 데이터 확보와 더 정교한 시각-언어 정렬 방법론 개발에 집중되어야 함을 시사한다.
