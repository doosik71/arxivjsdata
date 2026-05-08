# UNITER: UNiversal Image-TExt Representation Learning

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu

## 🧩 Problem to Solve

대부분의 Vision-and-Language (V+L) 태스크는 시각 및 텍스트 단서 간의 의미론적 간극을 메우기 위해 공동 다중 모달 임베딩에 의존합니다. 하지만 이러한 임베딩은 특정 태스크에 맞춰져 있어 다른 태스크에 일반화하기 어렵습니다. 이 논문은 모든 V+L 태스크에 적용할 수 있는 **범용적인 이미지-텍스트 표현을 학습하는 것**을 목표로 합니다.

## ✨ Key Contributions

- **UNITER (UNiversal Image-TExt Representation)** 모델을 제안하여 다양한 V+L 태스크를 위한 강력한 다중 모달 표현을 학습했습니다.
- **Conditional Masking** 기법을 Masked Language Modeling (MLM) 및 Masked Region Modeling (MRM)에 도입하고, 새로운 **Optimal Transport (OT) 기반 Word-Region Alignment (WRA)** 태스크를 사전 학습에 활용했습니다. Conditional Masking은 한 모달리티만 마스킹하고 다른 모달리티는 온전히 관찰하여 더 나은 정렬 학습을 유도합니다.
- 광범위한 V+L 벤치마크(9개 데이터셋의 6개 태스크)에서 새로운 **최첨단(state-of-the-art, SOTA) 성능**을 달성했으며, 기존 다중 모달 사전 학습 방법들을 크게 능가했습니다.
- 각 사전 학습 태스크 및 데이터셋의 효과에 대한 **심층적인 실험 및 분석**을 통해 유용한 통찰력을 제공했습니다.

## 📎 Related Works

- **자기 지도 학습 (Self-supervised learning):** 이미지 색칠, 직소 퍼즐 풀기, 인페인팅, 회전 예측 등 컴퓨터 비전 태스크에 적용되어 원본 데이터를 감독 신호로 활용합니다.
- **사전 학습 언어 모델 (Pre-trained language models):** ELMo, BERT, GPT2, XLNet, RoBERTa, ALBERT와 같은 모델이 대규모 언어 말뭉치와 Transformer를 활용하여 NLP 태스크에서 큰 발전을 이뤘습니다.
- **다중 모달 사전 학습 (Multimodal pre-training):**
  - **비디오-텍스트: 비디오**BERT, CBT는 비디오 프레임 특징과 언어 토큰의 공동 분포 학습에 BERT를 적용했습니다.
  - **이미지-텍스트:**
    - **Two-stream 아키텍처:** ViLBERT, LXMERT는 이미지와 텍스트에 독립적인 두 개의 Transformer를 사용한 후 세 번째 Transformer로 융합합니다.
    - **Single-stream 아키텍처:** B2T2, VisualBERT, Unicoder-VL, VL-BERT는 하나의 Transformer를 이미지와 텍스트 모두에 적용합니다.
    - VLP는 이미지 캡셔닝 및 VQA에 사전 학습 모델을 적용했습니다.
    - VALUE는 사전 학습 모델 이해를 위한 probing 태스크를 개발했습니다.
- **UNITER의 차별점:** 이전 연구들이 양 모달리티에 무작위 마스킹을 적용한 것과 달리, UNITER는 **Conditional Masking**을 사용하여 한 모달리티만 마스킹합니다. 또한, 기존 연구에서 간접적으로만 학습되던 단어-영역 정렬을 **Optimal Transport 기반 WRA**를 통해 명시적으로 강화합니다.

## 🛠️ Methodology

UNITER는 이미지 임베더(Image Embedder), 텍스트 임베더(Text Embedder) 및 다층 트랜스포머(multi-layer Transformer)로 구성됩니다.

1. **모델 아키텍처:**

   - **Image Embedder:** Faster R-CNN으로 시각 특징(pooled ROI features)과 7차원 위치 특징(바운딩 박스 좌표, 너비, 높이, 면적)을 추출한 후 FC 레이어를 거쳐 동일한 임베딩 공간으로 투영하고 LN을 적용합니다.
   - **Text Embedder:** BERT를 따라 WordPieces로 토큰화하고, 단어 임베딩과 위치 임베딩을 합산한 후 LN을 적용합니다. (텍스트/시각 입력을 구분하기 위한 모달리티 임베딩도 사용)
   - **Transformer:** Image Embedder와 Text Embedder의 출력을 받아 교차 모달리티(cross-modality) 맥락화된 임베딩을 학습합니다.

2. **사전 학습 태스크:** 각 미니 배치마다 하나의 태스크를 무작위로 샘플링하여 학습합니다.

   - **Masked Language Modeling (MLM):**
     - 입력 단어 중 15%를 무작위로 마스킹하고 `[MASK]` 토큰으로 대체합니다.
     - **Conditional Masking:** 마스킹된 단어를 나머지 단어와 **모든 이미지 영역**을 기반으로 예측합니다.
     - 목표: 마스킹된 단어를 예측하는 손실을 최소화합니다.
       $$L_{MLM}(\theta) = -\mathbb{E}_{(w,v)\sim D} \log P_{\theta}(w_m|w_{\setminus m},v)$$
   - **Image-Text Matching (ITM):**
     - `[CLS]` 토큰의 표현을 사용하여 전체 이미지-텍스트 쌍이 일치하는지(0 또는 1) 예측하는 이진 분류 태스크입니다.
     - 긍정/부정 쌍을 샘플링하여 학습하며, 부정 쌍은 이미지 또는 텍스트를 무작위로 대체하여 생성합니다.
     - 손실: 이진 교차 엔트로피 손실.
       $$L_{ITM}(\theta) = -\mathbb{E}_{(w,v)\sim D} [y\log s_{\theta}(w,v) + (1-y) \log(1-s_{\theta}(w,v))]$$
   - **Word-Region Alignment (WRA):**
     - Optimal Transport (OT)를 사용하여 단어와 이미지 영역 간의 **세밀한 정렬(fine-grained alignment)**을 명시적으로 학습합니다.
     - OT는 한 분포를 다른 분포로 변환하는 최소 비용을 계산하여 분포 매칭을 최적화합니다. 여기서는 이미지 영역 임베딩을 단어 임베딩으로 변환하는 비용을 최소화하여 교차 모달 정렬을 최적화합니다.
     - 비용 함수 $c(w_i, v_j)$로 코사인 거리를 사용하며, IPOT 알고리즘으로 수송 계획 T를 근사하여 OT 거리를 계산합니다.
     - 손실: OT 거리.
       $$L_{WRA}(\theta) = D_{ot}(\mu,\nu) = \min_{T\in\Pi(a,b)} \sum_{i=1}^T \sum_{j=1}^K T_{ij} \cdot c(w_i,v_j)$$
   - **Masked Region Modeling (MRM):**
     - 이미지 영역 중 15%를 샘플링하고 시각 특징을 0으로 마스킹합니다.
     - **Conditional Masking:** 마스킹된 영역을 나머지 영역과 **모든 단어**를 기반으로 재구성하도록 학습합니다.
     - 세 가지 변형:
       - **Masked Region Feature Regression (MRFR):** 마스킹된 영역의 Transformer 출력을 원본 시각 특징으로 회귀합니다 (L2 손실).
       - **Masked Region Classification (MRC):** 마스킹된 영역의 객체 의미 클래스를 예측합니다 (Faster R-CNN의 가장 높은 확신도를 가진 객체 카테고리를 hard label로 사용, 교차 엔트로피 손실).
       - **Masked Region Classification with KL-Divergence (MRC-kl):** MRC와 유사하나, 객체 탐지기의 원본 출력 분포(soft label)를 사용하여 지식 증류(knowledge distillation) 방식으로 KL 발산 손실을 최소화합니다.

3. **사전 학습 데이터셋:**
   - **In-domain 데이터:** COCO, Visual Genome. (대부분의 V+L 태스크가 이 데이터셋 기반)
   - **Out-of-domain 데이터:** Conceptual Captions, SBU Captions.
   - 다운스트림 평가 이미지와 Flickr30K 이미지의 중복을 제거하여 클린한 데이터셋을 구축했으며, 모든 데이터를 결합하여 학습합니다.

## 📊 Results

- **사전 학습 태스크 분석 (UNITER-base):**
  - 사전 학습 없이 (L1) 또는 텍스트 MLM만으로 (L2) 학습한 것보다 UNITER의 사전 학습 태스크가 Meta-Sum 점수를 크게 향상시켰습니다.
  - MRM 변형 중 **MRC-kl이 가장 좋은 성능**을 보였으며, MRC-kl과 MRFR을 결합할 때 상호 보완적으로 작용하여 성능이 향상되었습니다.
  - **WRA 추가 시 VQA 및 RefCOCO+에서 특히 큰 성능 향상**을 보였는데, 이는 WRA가 학습하는 세밀한 단어-영역 정렬이 영역 수준의 인식 및 추론 태스크에 도움이 됨을 시사합니다.
  - **Conditional Masking이 Joint Random Masking보다 더 효과적**이며, 더 빠른 수렴과 높은 최종 정확도를 보였습니다.
  - In-domain 데이터에 Out-of-domain 데이터를 추가하여 학습할 때 성능이 더욱 향상되었습니다.
- **다운스트림 태스크 성능 (UNITER-large):**
  - UNITER-large 모델은 VQA, VCR, NLVR$_{2}$, Visual Entailment, Image-Text Retrieval, Referring Expression Comprehension 등 **모든 6개 V+L 태스크(9개 데이터셋)에서 새로운 SOTA를 달성**했습니다.
  - UNITER-base 모델 또한 VQA를 제외한 모든 태스크에서 기존 모델들을 크게 능가했습니다.
  - 특히, UNITER는 ViLBERT나 LXMERT와 같은 two-stream 모델보다 **더 적은 파라미터(UNITER-base: 86M vs. LXMERT: 183M, ViLBERT: 221M)로 더 높은 성능**을 달성하며 single-stream 모델의 잠재력을 입증했습니다.
  - **VCR에 대한 2단계 사전 학습:** 사전 학습 데이터셋과 매우 다른 데이터로 구성된 VCR 데이터셋에 대해 2단계 사전 학습을 적용했을 때 모델 성능이 크게 향상되었습니다.
  - **NLVR$_{2}$에 대한 구조 적응:** UNITER는 단일 이미지-텍스트 쌍으로 사전 학습되었음에도 불구하고, NLVR$_{2}$처럼 두 이미지를 입력으로 받는 태스크에 Pair-biattn 설정을 통해 성공적으로 적응하여 뛰어난 성능을 보였습니다.

## 🧠 Insights & Discussion

- UNITER는 V+L 태스크를 위한 **범용적인 표현 학습의 중요성**을 입증했습니다.
- **Conditional Masking**은 마스킹된 모달리티의 정보를 온전히 관찰되는 다른 모달리티로부터 추론하게 함으로써, 이미지와 텍스트 간의 **더 나은 잠재적 정렬 학습**에 기여합니다.
- **Optimal Transport 기반 WRA**는 단어와 이미지 영역 간의 세밀한 정렬을 명시적으로 촉진하여, 특히 영역 수준의 추론이 필요한 태스크에서 성능 향상에 큰 영향을 미칩니다.
- Single-stream Transformer 아키텍처가 적절한 사전 학습 전략과 결합될 때, 복잡한 two-stream 아키텍처보다 더 효율적이고 강력한 성능을 발휘할 수 있음을 보여주었습니다.
- 사전 학습 데이터셋의 양과 도메인 유사성 모두 모델 성능에 중요하며, In-domain 및 Out-of-domain 데이터를 모두 활용하는 것이 가장 효과적입니다.
- UNITER는 다양한 다운스트림 태스크에 **높은 일반화 능력과 적응력**을 보여주었으며, 입력 구조가 사전 학습과 다른 경우에도 최소한의 수정으로 높은 성능을 달성할 수 있음을 입증했습니다.
- 어텐션 맵 시각화는 UNITER 모델이 단어와 이미지 영역 간의 **교차 모달 정렬**을 효과적으로 학습함을 보여주었습니다.

## 📌 TL;DR

이 논문은 이미지-텍스트 공동 이해를 위한 **범용적인 이미지-텍스트 표현(UNITER)**을 학습하는 문제를 다룹니다. UNITER는 단일 스트림(single-stream) 트랜스포머 아키텍처를 기반으로, **Conditional Masking** (다른 모달리티를 온전히 관찰하며 한 모달리티만 마스킹)과 **Optimal Transport 기반 Word-Region Alignment (WRA)**를 포함한 4가지 독창적인 사전 학습 태스크를 제안합니다. 광범위한 실험을 통해 UNITER는 VQA, VCR, 이미지-텍스트 검색 등 6개 주요 V+L 태스크에서 새로운 최첨단 성능을 달성했으며, 특히 Conditional Masking과 WRA가 교차 모달 정렬 학습에 크게 기여함을 입증했습니다. UNITER는 더 적은 파라미터로도 기존 two-stream 모델들을 능가하는 효율적이고 강력한 다중 모달 표현 학습 모델입니다.
