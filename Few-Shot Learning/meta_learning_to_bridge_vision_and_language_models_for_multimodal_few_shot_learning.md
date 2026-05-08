# METALEARNING TO BRIDGE VISION AND LANGUAGE MODELS FOR MULTIMODAL FEW-SHOT LEARNING

Ivona Najdenkoska, Xiantong Zhen, Marcel Worring (2023)

## 🧩 Problem to Solve

본 논문은 시각(Vision)과 언어(Language)라는 서로 다른 두 모달리티 사이의 거대한 도메인 간극(domain gap)으로 인해 발생하는 Multimodal few-shot learning의 어려움을 해결하고자 한다. 기존의 방법론들은 주로 고정된(frozen) 언어 모델에 시각적 개념을 프롬프트 형태로 전달하여 해결하려 했으나, 이는 가설 공간(hypothesis space)을 줄이기 위해 사람이 직접 설계한 '작업 유도(task induction)' 문구에 의존한다는 한계가 있다.

특히, 단순한 이진 분류 작업에서는 이러한 수동 설계가 작동할 수 있지만, 더 복잡한 작업으로 넘어갈수록 매번 새로운 task induction을 설계해야 하는 번거로움과 비효율성이 발생한다. 따라서 본 연구의 목표는 사람이 직접 개입하여 작업을 정의하는 대신, 데이터로부터 직접 작업을 유도하는 완전히 학습 가능한(learnable) 방식의 multimodal meta-learning 접근법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모의 사전 학습된 시각 모델과 언어 모델을 그대로 유지(frozen)하면서, 그 사이를 연결하는 가벼운 **Meta-mapper** 네트워크를 통해 메타 지식을 학습시키는 것이다.

중심적인 설계 직관은 다음과 같다.

1. **학습 가능한 브릿지 설계**: 시각적 표현을 언어 모델의 잠재 공간(latent space)으로 매핑하는 Meta-mapper를 도입하여, 두 모달리티 간의 간극을 메운다.
2. **메타 학습의 도입**: 다양한 multimodal few-shot task를 순차적으로 관찰하며 공유된 메타 지식을 축적함으로써, 새로운 작업이 주어졌을 때 단 몇 번의 그래디언트 업데이트만으로도 빠르게 적응(fast adaptation)할 수 있도록 한다.
3. **데이터 기반 작업 유도**: 수동으로 작성된 텍스트 지시문 없이, support set의 데이터를 통해 모델이 스스로 작업을 파악하게 함으로써 유연성을 높인다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별점을 제시한다.

- **Large-scale Language Models & Prompting**: GPT-3와 같은 모델들은 In-context learning을 통해 적은 예시만으로 작업을 수행한다. 최근에는 시각-언어 모델에서도 프롬프팅 기법이 사용되고 있으며, 본 논문은 고정된 텍스트 프롬프트 대신 학습 가능한 **Visual prefix**를 최적화하는 방향을 취한다.
- **Meta-learning for Few-shot Learning**: 기존의 메타 학습은 주로 단일 모달리티(예: 이미지 분류)에 집중되어 왔으며, 크게 Metric-based, Memory-based, Optimization-based 접근법으로 나뉜다. 본 연구는 유연성과 모달리티 독립성이 높은 **Optimization-based meta-learning** 방식을 채택한다.
- **Multimodal Few-shot Learning**: 'Frozen' 모델은 언어 모델의 In-context learning 능력을 활용해 multimodal few-shot learning을 시도했으나, 수동적인 task induction에 의존했다. 'Flamingo'는 거대한 파라미터 규모로 이를 해결하려 했으나, 본 논문은 매우 적은 학습 파라미터(200만 개 미만)만으로도 효율적인 적응이 가능함을 보여줌으로써 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 **Frozen Vision Encoder**, **Trainable Meta-mapper**, **Frozen Language Model**의 세 가지 모듈로 구성된다. 전체 파이프라인은 시각 특징을 추출하여 언어 모델이 이해할 수 있는 Visual prefix로 변환하고, 이를 기반으로 텍스트를 생성하는 autoregressive 방식으로 동작한다.

### 주요 구성 요소 및 역할

1. **Vision Encoder ($v_\phi$):** CLIP (ViT/B-32)를 사용하며, 입력 이미지 $x$로부터 시각적 특징 $v_\phi(x) = x_1, \dots, x_n$을 추출한다. 이 모델의 파라미터 $\phi$는 고정된다.
2. **Meta-mapper ($f_\theta$):** 시각적 특징을 언어 모델의 임베딩 공간으로 매핑하는 핵심 모듈이다. Set multi-head attention block을 사용하여 시각적 특징과 학습 가능한 파라미터 $p$를 결합해 최적의 **Visual prefix** ($p^*_1 \dots p^*_l$)를 생성한다.
3. **Language Model ($g_\omega$):** GPT-2를 사용하며, Meta-mapper가 생성한 Visual prefix와 텍스트 토큰 임베딩을 입력받아 다음 토큰을 예측하는 autoregressive 생성을 수행한다. 이 모델의 파라미터 $\omega$ 역시 고정된다.

### 주요 방정식 및 학습 절차

**1. Meta-mapper의 동작:**
Meta-mapper는 self-attention 메커니즘을 통해 특징 간의 유사도를 측정하고 가중치를 부여한다.
$$\text{MetaMap}_\theta(Q, K, V) = \sigma(QK^T) * V$$
여기서 $Q, K, V$는 모두 시각적 특징과 학습 가능한 prefix의 결합체 $[p_1 \dots p_l, x_1, \dots, x_n]$이다. 최종적으로 생성된 Visual prefix는 다음과 같이 정의된다.
$$p^*_1 \dots p^*_l = \text{MetaMap}_\theta([p_1 \dots p_l, x_1, \dots, x_n])$$

**2. 메타 학습 과정 (Optimization-based):**
학습은 Inner-loop와 Outer-loop의 두 단계로 이루어진다. 각 태스크 $T_i$에 대해 support set $\mathcal{D}_{tr}^i$와 query set $\mathcal{D}_{ts}^i$가 주어진다.

- **Inner-loop (Task Adaptation):**
  특정 태스크 $T_i$에 대해 support set을 사용하여 메타 파라미터 $\theta$를 태스크 전용 파라미터 $\theta'_i$로 빠르게 업데이트한다.
  $$\theta'_i = \theta - \alpha \nabla_\theta L_{T_i}(f_\theta)$$
  여기서 $L_{T_i}$는 Cross-entropy 손실 함수이다.

- **Outer-loop (Meta-update):**
  업데이트된 $\theta'_i$를 사용하여 query set $\mathcal{D}_{ts}^i$에서의 성능을 평가하고, 이를 바탕으로 전체 메타 파라미터 $\theta$를 최적화한다.
  $$\min_\theta \sum_{x_j, y_j \in \mathcal{D}_{ts}^i} L_{T_i}(f_{\theta'_i})$$
  최종적으로 SGD를 통해 $\theta \leftarrow \theta - \beta \nabla_\theta \sum L_{T_i}(f_{\theta'_i})$ 형태로 업데이트가 수행된다.

**3. 추론 (Inference):**
새로운 태스크가 주어지면, support set을 통해 $\theta$를 빠르게 파이프라인에 맞게 미세 조정(fine-tuning)한 뒤, query 이미지에 대해 top-k nucleus sampling을 통해 텍스트 답변을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 학습에는 COCO2017을 사용해 태스크 구조로 재구성하였고, 테스트에는 Real-Name miniImageNet, Real-Fast VQA, Open-Ended miniImageNet, Fast-VQA의 4가지 벤치마크를 사용하였다.
- **비교 대상**: Frozen (Tsimpoukelli et al., 2021) 모델을 주요 베이스라인으로 설정하였으며, 상한선(upper-bound) 확인을 위해 판별 모델인 ANIL을 참고하였다.
- **지표**: Accuracy (%)를 측정하였다.

### 주요 결과

1. **성능 향상**: 제안된 모델은 수동적인 task induction 없이도 Frozen 베이스라인을 크게 상회하는 성능을 보였다. 특히 Real-Name miniImageNet 2-way 5-shot 설정에서 높은 정확도를 기록하였다.
2. **에피소딕 학습의 효과**: 단순한 미니배치 학습(non-episodic)보다 태스크 단위로 학습하는 에피소딕 학습(episodic training)이 성능을 유의미하게 향상시켰다.
3. **데이터 효율성**: Support set에서 동일한 샘플을 반복해서 보여주는 것(repeats)보다, 서로 다른 샘플(shots)의 수를 늘리는 것이 성능 향상에 훨씬 효과적임을 확인하였다.
4. **VQA 성능**: Real-Fast VQA 및 Fast-VQA에서도 Frozen 모델 대비 우수한 성능을 보였으며, 이는 모델이 시각적 개념을 단어에 바인딩하고 복잡한 쿼리에 대해 추론하는 능력을 갖췄음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 큰 성과는 **가벼운 Meta-mapper**만으로 대규모 frozen 모델들의 능력을 효과적으로 결합했다는 점이다. 학습 파라미터가 약 200만 개에 불과하여 계산 효율성이 매우 높으며(GTX 1080Ti에서 2시간 이내 학습), 사람이 일일이 지시문을 작성할 필요가 없는 데이터 주도적(data-driven) 방식이라는 점에서 실용성이 높다.

### 한계 및 논의사항

1. **평가 지표의 한계**: 본 논문은 정답 단어와 정확히 일치하는지만을 측정하는 accuracy를 사용하였다. 하지만 생성 모델의 특성상 정답과 의미는 같으나 단어가 다른 '패러프레이징'된 답변이 나올 수 있으며, 이는 정량적으로는 오답 처리되지만 정성적으로는 더 훌륭한 답변일 수 있다.
2. **가설 공간의 크기**: 생성 방식(Open-ended)은 분류 방식(Closed-set)보다 훨씬 넓은 가설 공간(전체 어휘집)을 가지므로, ANIL과 같은 분류기 모델과 직접적인 공정한 비교가 어렵다.
3. **도메인 전이 문제**: COCO2017으로 학습하고 miniImageNet으로 테스트하는 cross-domain 설정에서 정량적 수치는 낮아졌으나, 생성된 문장의 질은 더 상세해지는 경향을 보였다.

## 📌 TL;DR

본 논문은 시각-언어 모델 간의 간극을 메우기 위해 **Frozen Vision/Language Encoder** 사이에 학습 가능한 **Meta-mapper**를 도입한 **Multimodal Meta-learning** 프레임워크를 제안한다. 이 모델은 수동적인 작업 지시문 없이도 support set의 데이터를 통해 빠르게 새로운 태스크에 적응하며, 매우 적은 파라미터와 계산 비용으로 기존 Frozen 모델보다 우수한 성능을 달성하였다. 이는 향후 다양한 모달리티를 결합한 효율적인 few-shot 학습 연구에 중요한 기초가 될 것으로 보인다.
