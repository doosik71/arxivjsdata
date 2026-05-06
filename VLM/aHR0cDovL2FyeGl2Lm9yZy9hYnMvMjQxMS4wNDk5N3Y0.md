# LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation

Weiquan Huang, Aoqi Wu, Yifan Yang, Xufang Luo, Yuqing Yang, Liang Hu, Qi Dai, Chunyu Wang, Xiyang Dai, Dongdong Chen, Chong Luo, Lili Qiu (2024/2025)

## 🧩 Problem to Solve

본 논문은 기초 멀티모달 모델인 CLIP의 시각적 표현(visual representation) 능력을 대규모 언어 모델(LLM)의 강력한 텍스트 이해 능력과 방대한 오픈 월드 지식을 활용해 향상시키는 것을 목표로 한다.

기존 CLIP 모델은 웹 규모의 이미지-텍스트 쌍을 통해 정렬되었으나, 다음과 같은 근본적인 한계가 존재한다.

1. **긴 텍스트 처리 능력 부족**: CLIP의 텍스트 인코더는 입력 토큰 길이가 77개로 제한되어 있으며, 길고 복잡한 이미지 캡션(dense captions)을 처리하는 데 어려움이 있다.
2. **LLM 특징 공간의 낮은 변별력**: LLM을 단순히 CLIP의 텍스트 인코더로 대체하려고 시도할 경우, LLM의 출력 특징들이 서로 유사한 텍스트 간의 변별력이 낮아(poor separability) 대조 학습(contrastive learning)을 통한 효율적인 시각 인코더 학습이 어렵다. 실제로 Llama 3-8B와 같은 모델은 동일 이미지에 대한 서로 다른 캡션을 구분하는 성능이 매우 낮게 나타난다.

결과적으로 본 연구는 LLM의 지식을 CLIP의 시각 인코더에 효과적으로 전이하여, 더 풍부하고 고차원적인 텍스트 감독(textual supervision)을 통해 시각적 표현력을 극대화하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LLM을 CLIP의 '개인 튜터(private tutor)'로 활용하여 시각 인코더를 고도화하는 **2단계 포스트 트레이닝 전략**이다.

1. **Caption Contrastive (CC) Fine-tuning**: LLM의 출력 공간을 조정하여 이미지 캡션 간의 변별력을 높이는 단계이다. 이를 통해 LLM이 대조 학습에 적합한 텍스트 임베딩 모델로 기능하게 한다.
2. **LLM2CLIP Post-training**: CC-파인튜닝된 LLM을 고정(freeze)시킨 채, 이를 텍스트 인코더로 사용하여 CLIP의 시각 인코더(Vision Encoder)만을 효율적으로 학습시키는 단계이다.
3. **효율적인 학습 파이프라인**: LLM의 그래디언트를 고정하고 텍스트 특징을 오프라인으로 미리 계산하여 저장하는 방식을 통해, 기존 LoRA 기반 방법보다 학습 속도를 약 4배 높이면서도 더 우수한 성능을 달성하였다.

## 📎 Related Works

### 1. LLM과 CLIP의 통합 시도

- **Jina-CLIP**: BERT 변형 모델을 사용하여 더 긴 텍스트를 지원하지만, Llama-3와 같은 최신 LLM에 비해 파라미터 수가 매우 적어 LLM의 잠재력을 충분히 활용하지 못한다.
- **E5-V**: MLLM 레이어에서 LLM 특징을 집계하고 ViT 그래디언트를 고정한다. 그러나 시각 인코더 자체가 가진 복잡한 특징 추출 능력의 한계를 해결하지 못해 성능이 낮다.
- **MATE**: 학습 가능한 어댑터를 통해 CLIP과 LLM의 간극을 메우려 했으나, 본 논문에서 지적한 'LLM 특징 공간의 낮은 변별력' 문제를 간과하였다.

### 2. 캡션 확장 연구 (Longer Captions)

- **DCI, LaCLIP, DreamLIP**: 사람이 직접 작성하거나 ChatGPT, InstructBLIP 등을 이용해 캡션을 확장하여 학습시키는 방식이다. 하지만 이들은 긴 텍스트를 처리하기 위해 텍스트를 요약하거나 분할하는 타협안을 사용했다. 반면 LLM2CLIP은 LLM을 인코더로 직접 활용하여 긴 텍스트의 포괄적인 이해가 가능하다.

## 🛠️ Methodology

### Stage 1: LLM Caption Contrastive Fine-tuning

LLM을 생성 모델에서 변별력 있는 텍스트 인코더로 변환하는 단계이다.

**1. 모델 구조 조정**

- **Sentence Representation**: 문장 전체를 대표하는 특징을 추출하기 위해 모든 출력 토큰의 평균 풀링(Average Pooling)을 사용한다.
- **Bidirectional Attention**: 생성 능력이 필요 없으므로 인과적 마스크(causal mask)를 제거하여 양방향 텍스트 관계 모델링이 가능하게 한다.
- **PEFT**: 효율적인 학습을 위해 LoRA(Low-Rank Adaptation)를 적용한다.

**2. 학습 방법 및 손실 함수**

- **Supervised SimCSE**: 감독 학습 기반의 대조 학습 손실 함수를 사용한다. 동일한 이미지에서 추출된 서로 다른 두 캡션을 양성 쌍(positive pair)으로 설정하여 거리를 좁히고, 나머지 캡션들은 음성 쌍(negative pair)으로 설정하여 거리를 멀게 한다.
- **수식적 직관**: 텍스트 특징 $\mathbf{z}$에 대해, 동일 이미지 캡션 쌍 $(\mathbf{z}_i, \mathbf{z}_j)$의 유사도를 최대화하고 다른 캡션과의 유사도는 최소화하는 방향으로 학습한다.

### Stage 2: LLM2CLIP Post Fine-tuning

CC-파인튜닝된 LLM을 활용해 CLIP의 시각 인코더를 최적화하는 단계이다.

**1. 전체 파이프라인**

- **LLM Frozen**: LLM의 파라미터는 고정하여 메모리 사용량을 줄이고 연산 효율을 높인다.
- **Linear Adaptor**: LLM의 출력층 뒤에 4개의 선형 레이어로 구성된 단순한 인버티드 보틀넥(inverted bottleneck) MLP 구조의 어댑터를 추가하여 시각 공간과의 정렬을 돕는다.
- **ViT Training**: 시각 인코더(ViT)의 모든 그래디언트를 열어 LLM의 풍부한 지식을 시각적 표현에 반영하도록 학습한다.

**2. 학습 절차 및 데이터**

- **Contrastive Learning**: LLM과 시각 인코더 간의 교차 모달 대조 학습을 수행한다.
- **데이터 전략**: 실제 캡션(Real captions)과 MLLM이 생성한 정밀 캡션(Dense MLLM captions)을 50:50 비율로 혼합하여 사용한다. 이는 짧은 텍스트와 긴 텍스트 성능 사이의 균형을 맞추기 위함이다.

## 📊 Results

### 1. 제로샷 이미지-텍스트 검색 (Zero-Shot Retrieval)

- **단문/장문 검색**: CLIP, EVA02, SigLip2 등 SOTA 모델에 LLM2CLIP을 적용했을 때 모든 지표에서 성능이 향상되었다. 특히 장문 검색(ShareGPT4V, Urban1K, DOCCI)에서 매우 큰 폭의 향상($+14.8 \sim +15.8$ 포인트)을 보였다.
- **데이터 규모**: 학습 데이터 양이 3M $\rightarrow$ 15M $\rightarrow$ 60M으로 증가함에 따라 성능이 일관되게 향상되는 경향을 보였다.

### 2. 다국어 검색 및 기타 성능

- **교차 언어 검색**: 영어 데이터로만 학습했음에도 불구하고, LLM의 내재적 다국어 능력이 전이되어 EVA02와 SigLip2의 다국어 검색 성능이 크게 개선되었다.
- **ImageNet 분류**: 제로샷 분류 성능은 약간 하락했으나, Linear Probe 실험에서는 성능이 향상되었다. 이는 LLM2CLIP이 더 우수한 시각적 특징(visual features)을 추출함을 의미한다.
- **VLLM 성능 향상**: Llava 1.5의 시각 인코더를 LLM2CLIP으로 교체했을 때, 벤치마크의 87.5% 이상에서 성능 향상이 관찰되어 복잡한 이미지 추론 능력이 개선됨을 입증했다.

### 3. 학습 효율성 (Efficiency)

- **오프라인 로딩**: LLM 특징을 미리 계산하여 저장하는 방식을 통해 배치 사이즈를 10배 키울 수 있었으며, 학습 시간을 4배 단축했다 (LoRA 대비).

## 🧠 Insights & Discussion

**강점 및 성과**

- 본 연구는 LLM을 단순히 텍스트 인코더로 쓰는 것이 아니라, **'특징 공간의 변별력'**이라는 핵심 문제를 해결함으로써 CLIP의 시각 인코더를 성공적으로 고도화했다.
- 특히 긴 텍스트 처리 능력의 획기적 개선은 기존 CLIP 계열 모델들이 가졌던 가장 큰 약점을 극복한 점이라 평가할 수 있다.

**한계 및 논의사항**

- **제로샷 분류 성능 하락**: ImageNet 제로샷 성능이 약간 떨어진 이유는 LLM이 매우 짧은 단어 수준의 캡션을 구분하는 데 최적화되지 않았거나, 데이터 정렬이 충분하지 않았기 때문으로 추정된다. 이를 해결하기 위해 더 큰 규모의 데이터셋 확장이 필요할 수 있다.
- **데이터 비율의 Trade-off**: 실험 결과, Dense 캡션의 비율이 높으면 장문 검색 성능이 올라가고, 실제 짧은 캡션의 비율이 높으면 단문 검색 성능이 올라가는 경향이 있다. 1:1 비율이 최적의 균형점임을 확인했다.

## 📌 TL;DR

LLM2CLIP은 LLM의 강력한 언어 이해 능력을 CLIP의 시각 인코더에 주입하기 위한 2단계 학습 전략을 제안한다. 먼저 **Caption Contrastive Fine-tuning**을 통해 LLM의 텍스트 변별력을 높인 후, 이를 고정된 텍스트 인코더로 사용하여 **시각 인코더를 포스트 트레이닝**한다. 이 방법은 특히 긴 텍스트 검색과 다국어 검색에서 압도적인 성능 향상을 보였으며, 학습 효율성 또한 극대화하였다. 결과적으로 LLM2CLIP은 일반적인 시각-언어 정렬 모델뿐만 아니라 LMM(Large Multimodal Model)의 시각적 기반을 강화하는 데 중요한 역할을 할 가능성이 높다.
