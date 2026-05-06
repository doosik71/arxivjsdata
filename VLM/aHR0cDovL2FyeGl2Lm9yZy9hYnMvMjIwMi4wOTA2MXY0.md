# VLP: A Survey on Vision-Language Pre-training

Feilong Chen, Duzhen Zhang, Minglun Han, Xiuyi Chen, Jing Shi, Shuang Xu, Bo Xu (2022)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전(CV)과 자연어 처리(NLP)의 통합 영역인 Vision-Language Pre-training(VLP) 분야의 전반적인 연구 동향을 체계적으로 분석하고 정리하는 것을 목표로 한다.

전통적으로 멀티모달 학습은 다량의 고품질 레이블링 데이터가 필요하다는 한계가 있었다. 하지만 단일 모달리티(uni-modal) 분야에서 Transformer 기반의 사전 학습 모델(pre-training models)이 거대한 규모의 비정형 데이터로부터 보편적인 표현(universal representations)을 학습하여 다운스트림 태스크의 성능을 비약적으로 향상시킨 사례가 많았다. 이에 따라 저자들은 이러한 사전 학습 패러다임을 멀티모달 태스크에 적용하여, 데이터 부족 문제를 해결하고 모델이 시각적 정보와 언어적 정보 사이의 의미적 대응 관계(semantic correspondence)를 스스로 학습하게 하는 VLP의 메커니즘을 분석하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 VLP 분야의 방대한 연구들을 다섯 가지 핵심 관점(특징 추출, 모델 구조, 사전 학습 목표, 데이터셋, 다운스트림 태스크)으로 분류하여 체계적인 프레임워크를 제시했다는 점이다. 특히 이미지-텍스트(Image-Text)뿐만 아니라 비디오-텍스트(Video-Text) 사전 학습까지 포괄하여 분석함으로써, VLP의 최신 기술 상태(SOTA)를 한눈에 파악할 수 있는 종합적인 가이드라인을 제공한다.

## 📎 Related Works

기존에는 사전 학습된 언어 모델(PLMs)이나 비전 모델에 대한 개별적인 서베이 논문들은 존재하였으나, 시각과 언어의 결합에 집중한 VLP 전용 서베이는 부족한 상태였다. VLP는 단순히 두 모델을 결합하는 것이 아니라, 두 모달리티 간의 정렬(alignment)과 융합(fusion)을 어떻게 최적화할 것인가라는 독특한 도전 과제를 가지고 있다. 본 논문은 기존의 단일 모달리티 사전 학습 방식에서 나아가, 멀티모달 환경에서의 상호작용을 극대화하는 VLP 모델들의 차별점을 분석한다.

## 🛠️ Methodology

본 논문은 VLP의 전체 파이프라인을 다음과 같은 구성 요소로 나누어 설명한다.

### 1. Feature Extraction (특징 추출)

시각 및 텍스트 데이터를 모델이 처리할 수 있는 벡터 형태로 변환하는 과정이다.

- **Image Feature**: 객체 검출기(Object Detector) 기반의 지역 특징(OD-RFs), CNN 기반의 그리드 특징(CNN-GFs), 그리고 ViT 기반의 패치 특징(ViT-PFs)으로 구분된다.
- **Video Feature**: 비디오를 여러 프레임의 이미지 집합으로 보고, 각 프레임에 대해 CNN-GFs 또는 ViT-PFs를 적용하여 추출한다.
- **Text Feature**: BERT와 같은 언어 모델을 따라 텍스트를 서브워드(subwords) 단위로 분절하고, Word, Position, Type embedding을 합산하여 표현한다.

### 2. Model Architecture (모델 구조)

모달리티 간의 융합 방식과 전체 설계를 기준으로 분류한다.

- **Single-stream vs Dual-stream**:
  - **Single-stream**은 텍스트와 시각 특징을 단순히 연결(concatenation)하여 하나의 Transformer 블록에 입력하며, 파라미터 효율성이 높다.
  - **Dual-stream**은 각 모달리티를 독립적인 Transformer 블록에서 처리한 후, Cross-Attention 메커니즘을 통해 상호작용을 유도한다.
- **Encoder-only vs Encoder-decoder**: 단순 표현 학습을 위한 인코더 구조와 텍스트 생성을 위한 디코더가 포함된 구조로 나뉜다.

### 3. Pre-training Objectives (사전 학습 목표)

모델이 두 모달리티의 관계를 학습하게 하는 핵심 손실 함수 및 목표들이다.

- **Completion (완성)**: 마스킹된 요소를 나머지 정보를 통해 복원하는 방식이다.
  - **Masked Language Modeling (MLM)**: 시각 정보 $v$와 나머지 텍스트 $w_{\setminus m}$을 이용해 마스킹된 텍스트 $w_m$을 예측한다.
    $$L_{MLM} = -\mathbb{E}_{(v,w) \sim D} \log P(w_m | w_{\setminus m}, v)$$
  - **Masked Vision Modeling (MVM)**: 텍스트와 나머지 시각 특징을 통해 마스킹된 시각 특징을 복원하며, L2 회귀(Regression) 또는 객체 클래스 분류(Classification) 방식을 사용한다.

- **Matching (매칭)**: 두 모달리티를 공통된 잠재 공간으로 투영하여 정렬하는 방식이다.
  - **Vision-Language Matching (VLM)**: 두 입력이 서로 일치하는지 여부를 0과 1 사이의 점수로 예측하는 이진 분류 문제로 정의한다.
  - **Vision-Language Contrastive Learning (VLC)**: 배치 내의 여러 쌍 중에서 정답 쌍의 유사도를 높이고 오답 쌍의 유사도를 낮추는 대조 학습을 수행한다.
    $$L_{VLC} = \frac{1}{2} \mathbb{E}_{(I,T) \sim D} [CE(y_{v2t}, p_{v2t}(I)) + CE(y_{t2v}, p_{t2v}(T))]$$
  - **Word-Region Alignment (WRA)**: 단어와 시각적 지역(region) 간의 최적 운송(Optimal Transport) 거리를 최소화하여 세밀한 정렬을 학습한다.

- **Temporal (시간적 순서)**: 비디오 데이터에서 프레임의 순서를 섞은 후 원래 위치를 예측하는 Frame Order Modeling (FOM)을 통해 시간적 맥락을 학습한다.

## 📊 Results

본 논문은 실험적 수치보다는 기존 모델들의 특성을 분석한 종합적인 결과표(Table 2, 3)를 제시한다.

- **Image-Text VLP**: VisualBERT, ViLBERT, UNITER, CLIP 등이 대표적이며, 대부분 COCO, Conceptual Captions (CC3M) 데이터셋을 활용하여 VQA, VLR(Retrieval) 등의 태스크에서 성능을 검증한다. 특히 UNITER와 같은 모델은 단일 스트림 구조와 다양한 사전 학습 목표를 결합하여 범용적인 성능을 보였다.
- **Video-Text VLP**: VideoBERT, CLIP4Clip 등이 있으며, HowTo100M과 같은 대규모 비디오 데이터셋을 사용한다. 비디오 모델들은 특히 시간적 정보의 처리와 고해상도 프레임 추출의 효율성이 중요한 지표가 된다.
- **Downstream Tasks**: VLP 모델들은 Visual Question Answering (VQA), Visual Entailment (VE), Vision-Language Retrieval (VLR) 등 매우 다양한 태스크에서 전이 학습(transfer learning)의 효용성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 VLP의 현재 상태를 분석하며 향후 연구 방향으로 다음과 같은 통찰을 제시한다.

1. **오디오 정보의 통합**: 현재 VLP는 시각과 언어에 집중되어 있으나, 실제 환경에서는 오디오가 제공하는 감정 및 경계 정보가 중요하므로 시각-언어-오디오의 3중 모달리티 학습이 필요하다.
2. **지식 기반 인지 학습**: 단순한 데이터 피팅을 넘어, 외부의 상식(common sense)이나 상황적 지식을 모델에 주입하는 Knowledge-guided pre-training이 필요하다.
3. **Prompt Tuning의 도입**: 거대 모델의 파라미터를 모두 튜닝하는 것은 비효율적이므로, NLP에서 성공적이었던 Prompt Tuning을 멀티모달 영역으로 확장하여 파라미터 효율적인 전이 학습을 구현해야 한다.
4. **모델 압축 및 가속화**: 실시간 배포를 위해 지식 증류(Knowledge Distillation), 가지치기(Pruning), 양자화(Quantization) 등의 기법을 VLP 모델에 적용하는 연구가 더 필요하다.
5. **도메인 외 사전 학습 (Out-of-domain Pre-training)**: 학습 데이터와 실제 테스트 데이터의 분포 차이를 극복하기 위한 인과 추론(Causality) 기반의 학습 방법 등이 논의되어야 한다.

## 📌 TL;DR

본 논문은 시각-언어 사전 학습(VLP)의 전 과정을 특징 추출, 구조, 학습 목표, 데이터, 태스크의 5가지 관점에서 집대성한 최초의 종합 서베이 논문이다. 특히 MLM, MVM, VLC와 같은 핵심 학습 메커니즘을 수식과 함께 상세히 설명하며, 단순한 성능 비교를 넘어 향후 VLP가 나아가야 할 방향(오디오 통합, 프롬프트 튜닝, 모델 압축 등)을 구체적으로 제시함으로써 향후 멀티모달 연구의 이정표 역할을 할 것으로 기대된다.
