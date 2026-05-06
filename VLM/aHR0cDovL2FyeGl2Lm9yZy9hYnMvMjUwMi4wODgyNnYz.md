# Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation

Mohammad Mahdi Abootorabi et al. (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)이 가진 고유한 한계점인 환각(hallucinations) 현상과 학습 데이터의 정적 특성으로 인한 지식의 최신성 부족 문제를 해결하고자 한다. 이를 완화하기 위해 외부의 동적인 정보를 통합하는 Retrieval-Augmented Generation (RAG) 기술이 도입되었으나, 기존의 RAG는 주로 텍스트라는 단일 모달리티에 의존한다는 한계가 있다.

현실 세계의 정보는 텍스트뿐만 아니라 이미지, 오디오, 비디오 등 다양한 형태로 존재하며, 이러한 멀티모달 데이터를 통합적으로 활용할 때 더 정확하고 풍부한 답변 생성이 가능하다. 따라서 본 논문은 텍스트, 이미지, 오디오, 비디오 등 여러 모달리티를 통합하여 생성 결과물을 향상시키는 Multimodal RAG (M-RAG) 시스템의 전반적인 구조를 분석하고, 이를 체계화된 분류 체계(Taxonomy)로 제시하는 것을 목표로 한다. 특히 모달리티 간의 정렬(Alignment)과 추론(Reasoning) 과정에서 발생하는 고유한 도전 과제들을 심도 있게 다룬다.

## ✨ Key Contributions

본 논문의 핵심 기여는 빠르게 발전하고 있는 Multimodal RAG 분야를 위해 다음과 같은 체계적인 분석 프레임워크를 제공한 것이다.

1. **포괄적인 M-RAG 분석**: 데이터셋, 벤치마크, 평가 지표부터 검색(Retrieval), 융합(Fusion), 증강(Augmentation), 생성(Generation)에 이르는 전체 파이프라인과 학습 전략, 손실 함수, 에이전트 기반 접근 방식까지 통합적으로 리뷰하였다.
2. **구조적 분류 체계(Taxonomy) 제안**: 최신 M-RAG 모델들을 핵심 기여도에 따라 분류한 체계적인 텍스노미를 제안하여, 방법론적 발전 방향과 최신 트렌드를 한눈에 파악할 수 있게 하였다.
3. **오픈 소스 리소스 제공**: 연구자들이 활용할 수 있도록 관련 데이터셋, 벤치마크 및 구현 상세 내용을 포함한 공개 리소스를 구축하였다.
4. **미래 연구 방향 제시**: 현재 기술의 공백(Research gaps)을 식별하고, 향후 연구가 나아가야 할 방향(예: Any-to-Any 모달리티 지원, Embodied AI로의 확장 등)을 제시하였다.

## 📎 Related Works

기존의 RAG 관련 서베이들은 주로 Agentic RAG와 같은 특정 구조나 단일 모달리티 환경에 집중되어 있었다. 논문에서 언급된 기존의 멀티모달 RAG 관련 연구(Zhao et al., 2023a)는 주로 애플리케이션과 모달리티별 분류에 치중했다는 한계가 있다.

본 논문은 이와 달리 '혁신 중심의 관점(innovation-driven perspective)'을 채택한다. 즉, 단순히 어떤 분야에 쓰였는가가 아니라, 검색 효율을 어떻게 높였는지, 모달리티 융합을 어떤 아키텍처로 해결했는지 등 방법론적 혁신을 중심으로 100편 이상의 최신 논문을 분석함으로써 기존 서베이들과 차별점을 둔다.

## 🛠️ Methodology

### 1. 수학적 정식화 (Mathematical Formulation)

Multimodal RAG의 목표는 멀티모달 쿼리 $q$가 주어졌을 때 멀티모달 응답 $r$을 생성하는 것이다.

- **코퍼스 정의**: 멀티모달 코퍼스를 $D = \{d_1, d_2, \dots, d_n\}$이라 하며, 각 문서 $d_i$는 특정 모달리티 $M_{d_i}$와 연결된다.
- **인코딩**: 각 문서는 해당 모달리티 전용 인코더 $\text{Enc}_{M_{d_i}}$를 통해 벡터 $z_i = \text{Enc}_{M_{d_i}}(d_i)$로 변환되어 공유 시맨틱 공간(Shared semantic space)에 투영된다.
- **검색**: 쿼리 $q$의 인코딩 표현 $e_q$와 각 문서 표현 $z_i$ 사이의 관련성 점수 $s(e_q, z_i)$를 계산한다.
- **컨텍스트 구성**: 모달리티별 임계값 $\tau_{M_{d_i}}$를 기준으로 관련 문서 집합 $X$를 구성한다.
    $$X = \{d_i \in D \mid s(e_q, z_i) \geq \tau_{M_{d_i}}\}$$
- **생성**: 생성 모델 $G$는 쿼리 $q$와 검색된 컨텍스트 $X$를 조건으로 응답 $r$을 생성한다.
    $$r = G(q, X)$$

### 2. 시스템 파이프라인 및 구성 요소

#### A. 검색 전략 (Retrieval Strategy)

- **효율적 검색**: Maximum Inner Product Search (MIPS) 변형 기법과 TPU-KNN, ScaNN 등을 사용하여 대규모 임베딩 공간에서 빠르게 유사도를 계산한다.
- **모달리티 중심 검색**: 텍스트 중심(BM25, ColBERT), 비전 중심(EchoSight, ImgRet), 비디오 중심(iRAG, VideoRAG), 오디오 중심(WavRAG, SEAL) 등 각 데이터의 특성에 맞는 전용 리트리버를 사용한다.
- **재순위화(Re-ranking)**: 검색된 후보군을 다시 정렬하여 정밀도를 높이며, 여기에는 최적화된 예시 선택 및 필터링 메커니즘이 포함된다.

#### B. 융합 메커니즘 (Fusion Mechanisms)

- **점수 융합 및 정렬**: CLIP Score fusion이나 프로토타입 기반 임베딩을 통해 서로 다른 모달리티의 점수를 통합한다.
- **어텐션 기반 메커니즘**: Cross-attention 또는 Co-attention Transformer를 사용하여 모달리티 간의 상호작용을 동적으로 조절한다.
- **통합 프레임워크**: Dense2Sparse 투영과 같이 밀집 임베딩을 희소 벡터로 변환하여 저장 효율성과 해석 가능성을 높이는 방식을 사용한다.

#### C. 증강 기술 (Augmentation Techniques)

- **컨텍스트 풍부화**: 검색된 데이터에 엔티티 관계나 시맨틱 설명을 추가하여 생성 모델에 더 풍부한 근거를 제공한다.
- **적응형 및 반복적 검색**: 쿼리의 복잡도에 따라 검색 횟수를 조절하거나, 1차 검색 결과를 바탕으로 쿼리를 수정하여 다시 검색하는 Coarse-to-fine 과정을 거친다.

#### D. 생성 기술 (Generation Techniques)

- **In-Context Learning (ICL)**: 검색된 내용을 퓨샷 예시로 활용하여 모델의 재학습 없이 추론 능력을 높인다.
- **추론(Reasoning)**: Chain-of-Thought (CoT)를 적용하여 복잡한 멀티모달 추론 과정을 단계별로 수행한다.
- **출처 표기(Source Attribution)**: 생성된 답변이 어떤 이미지나 텍스트 구간에서 왔는지 명시적으로 인용하여 신뢰성을 높인다.

### 3. 학습 전략 및 손실 함수 (Training & Loss Functions)

- **정렬(Alignment)**: InfoNCE loss를 사용하여 긍정 쌍(positive pair)은 가깝게, 부정 쌍(negative pair)은 멀게 배치하는 Contrastive Learning을 수행한다.
    $$L_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{K} \exp(\text{sim}(z_i, z_k)/\tau)}$$
- **강건성(Robustness)**: 학습 과정에서 무관한 결과를 의도적으로 주입하거나(Noise injection), Query Dropout을 통해 일반화 성능을 향상시킨다.
- **기타 손실 함수**: 거리 기반 학습을 위한 Triplet Loss와 생성물 품질 향상을 위한 GAN Loss 등이 사용된다.

## 📊 Results

본 논문은 서베이 논문이므로 개별 모델의 성능 수치보다는 M-RAG 생태계 전반의 실험적 기반을 분석하였다.

### 1. 데이터셋 및 벤치마크

- **데이터셋**: 일반 이미지-텍스트(LAION-5B, MS-COCO), 비디오-텍스트(HowTo100M), 오디오-텍스트(AudioSet), 의료(MIMIC-CXR), 패션(DeepFashion) 등 광범위한 데이터셋이 사용됨을 확인하였다.
- **벤치마크**: $M^2RAG$, MRAG-Bench, Dyn-VQA 등이 제안되었으며, 이는 단순한 정확도를 넘어 멀티홉 추론과 동적 검색 능력을 평가한다.

### 2. 평가 지표

- **검색 성능**: Top-K Accuracy, Recall@K, MRR (Mean Reciprocal Rank)을 사용하여 측정한다.
- **텍스트 생성**: BLEU, ROUGE, METEOR 등을 사용하여 참조 텍스트와의 유사도를 평가한다.
- **이미지/비디오 품질**: FID (Fréchet Inception Distance), CLIP Score, CIDEr 등을 통해 시각적 정렬도와 품질을 측정한다.
- **효율성**: FLOPs, 쿼리당 응답 시간 및 검색 시간을 측정한다.

## 🧠 Insights & Discussion

### 1. 강점 및 가능성

M-RAG는 텍스트만으로는 해결할 수 없는 복잡한 현실 세계의 문제(예: 의료 영상 분석, 자율 주행 설명, 패션 디자인 자동화)를 해결할 수 있는 강력한 잠재력을 가지고 있다. 특히 에이전트 기반의 상호작용 시스템으로 발전함에 따라 단순한 답변 생성을 넘어 능동적인 정보 탐색이 가능해질 것으로 보인다.

### 2. 한계 및 비판적 해석

- **모달리티 편향(Modality Bias)**: 많은 M-RAG 시스템이 겉으로는 멀티모달을 표방하지만, 실제로는 텍스트 정보에 과하게 의존하는 경향이 있다.
- **설명 가능성 부족**: 응답의 근거를 제시할 때 정밀한 포인트가 아닌 문서 전체나 넓은 이미지 영역을 인용하는 등 세밀한 Attribution이 부족하다.
- **취약성**: Knowledge Poisoning 공격(악의적인 정보 주입)에 취약하며, 적은 수의 적대적 샘플만으로도 전체 생성 결과가 왜곡될 수 있음이 지적되었다.

### 3. 향후 연구 방향

- **Any-to-Any 모달리티**: 특정 모달리티의 입력을 다른 모달리티의 출력으로 자유롭게 변환하고 검색하는 통합 임베딩 공간 구축이 필요하다.
- **Embodied AI 확장**: 물리적 환경의 센서 데이터와 RAG를 결합하여 로보틱스나 내비게이션 등 실제 물리적 상호작용이 가능한 지능형 시스템으로 진화해야 한다.

## 📌 TL;DR

본 논문은 멀티모달 데이터를 활용하는 RAG 시스템의 전체 파이프라인(검색 $\rightarrow$ 융합 $\rightarrow$ 증강 $\rightarrow$ 생성 $\rightarrow$ 학습)을 체계적으로 정리한 종합 서베이 보고서이다. 특히 단순한 사례 나열이 아니라 수학적 정식화와 상세한 분류 체계(Taxonomy)를 통해 M-RAG의 기술적 구조를 정의하였다. 이 연구는 향후 멀티모달 지식 베이스를 활용한 더 신뢰할 수 있고 강력한 AI 시스템을 구축하는 데 있어 핵심적인 가이드라인 역할을 할 것으로 기대된다.
