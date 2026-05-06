# A Zero-shot Learning Method Based on Large Language Models for Multi-modal Knowledge Graph Embedding

Bingchen Liu, Jingchen Li, Yuanyuan Fang, Xin Li (2025)

## 🧩 Problem to Solve

본 논문은 Multi-modal Knowledge Graph(MMKG) 임베딩 표현 학습에서 발생하는 Zero-shot Learning(ZSL)의 한계를 해결하고자 한다. MMKG는 텍스트, 이미지, 오디오 등 다양한 모달리티를 통합하여 엔티티와 관계를 설명하는 구조이지만, 실제 환경에서는 학습 데이터에 포함되지 않은 새로운 관계나 엔티티, 즉 Unseen categories가 빈번하게 등장한다.

기존의 ZSL 방법론들은 Seen classes에서 Unseen classes로의 매핑 관계를 구축하는 데 집중했으나, MMKG 환경에서는 Unseen categories의 시맨틱 정보가 효과적으로 전달되지 않아 추론 정확도가 현저히 떨어진다는 문제가 있다. 따라서 본 연구의 목표는 Large Language Models(LLMs)의 강력한 추론 및 생성 능력을 활용하여 Unseen categories의 시맨틱 정보 전이를 최적화하고, 이를 통해 MMKG의 Zero-shot 임베딩 성능을 향상시키는 ZSLLM 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LLM을 단순한 분류기가 아니라, Unseen categories에 대한 보조 특징(Auxiliary features)을 생성하고 지식을 전이하는 도구로 활용하는 것이다.

1. **LLM 기반 보조 특징 생성**: ChatGPT를 통해 Unseen classes의 텍스트 임베딩을 얻고, Dall-E를 통해 Unseen classes와 유사한 가상 이미지를 생성함으로써 데이터 부족 문제를 해결하고 Cross-modal 시맨틱 정렬을 달성한다.
2. **Knowledge Distillation(KD) 도입**: LLM의 방대한 지식이 특정 태스크에 최적화되지 않았거나 분산되어 있을 수 있다는 점을 고려하여, Teacher 모델에서 Student 모델로 지식을 전이하는 지식 증류 기법을 적용해 Zero-shot 시나리오에서의 성능을 강화한다.
3. **MMKG 구조 활용**: 생성된 특징들을 Multi-modal Knowledge Graph로 구성하고 Graph Convolutional Network(GCN)를 통해 최종 분류를 수행함으로써, 엔티티 간의 구조적 관계를 학습에 반영한다.

## 📎 Related Works

### 1. Knowledge Graph Embeddings (KGE)

KGE는 엔티티와 관계를 저차원 벡터 공간으로 매핑하는 기술로, 크게 세 가지로 분류된다.

- **Translation-based**: $\text{head} + \text{relation} \approx \text{tail}$ 형태로 모델링 (예: TransH). 구조가 단순하여 복잡한 시나리오 대응에 한계가 있다.
- **Bilinear-based**: 곱셈 형태의 scoring function을 사용하여 세밀한 상호작용을 캡처 (예: HOLE).
- **Neural network-based**: 신경망을 통해 트리플렛의 점수를 평가한다. 최근에는 이미지 등 추가 모달리티를 결합한 Multi-modal KGE(예: DFMKE)가 제안되고 있다.

### 2. Zero-shot Learning (ZSL)

ZSL은 학습 시 보지 못한 클래스를 추론하는 패러다임으로, Inductive ZL과 Transductive ZL로 나뉜다.

- **Inductive ZL**: 학습 데이터만 사용하며, 보조 정보(속성, 텍스트 설명)를 통해 간극을 메운다.
- **Transductive ZL**: 추론 단계에서 Unseen class 데이터의 분포 정보를 활용한다. 본 논문의 ZSLLM은 Unseen classes의 텍스트 정보를 이용해 이미지 분포를 추론하여 학습에 활용하므로 Transductive ZL에 해당한다.

### 3. Large Language Models (LLMs)

범용 모델(General-purpose, 예: LLaMA, DeepSeek)과 특정 도메인에 최적화된 전문 모델(Specialized)로 구분된다. 본 연구는 범용 LLM의 일반화 능력을 활용하여 Zero-shot 성능을 높이는 방식을 취한다.

## 🛠️ Methodology

ZSLLM 프레임워크는 크게 세 가지 모듈로 구성된다.

### 1. Seen Class Learning Module

Seen classes의 텍스트 특징($W_s$)과 시각적 특징($V_s$)을 결합하여 기본 인식 능력을 학습한다.

- **특징 결합**: $x = \text{concat}(W_s, V_s)$를 통해 벡터를 생성하고, 이를 $\text{ReLU}(\text{Conv}(\dots))$와 Pooling 층을 거쳐 특징을 추출한다.
- **Knowledge Distillation**: Teacher 모델이 학습한 지식을 Student 모델로 전이한다. 손실 함수는 다음과 같이 정의된다.
  $$\text{loss} = \alpha \cdot \text{student loss} + (1 - \alpha) \cdot \text{distillation loss}$$
  여기서 $\text{student loss}$는 Cross Entropy Loss를 사용하며, $\text{distillation loss}$는 다음과 같이 KL-Divergence를 사용한다.
  $$\text{distillation loss} = \text{KLDivLoss}(\log(\text{softmax}(\text{student preds}/\text{temp})), \text{softmax}(\text{teacher preds}/\text{temp}))$$

### 2. Unseen Class Recognition Module

Unseen classes에 대한 보조 정보를 생성하여 시맨틱 간극을 메운다.

- **텍스트 및 구조 정보**: ChatGPT로 Unseen classes의 단어 특징($w_u$)을 생성하고, 이를 MMKG 구조에 입력하여 업데이트된 특징 $\hat{F}_{s+u}$를 얻는다.
  $$\hat{F}_{s+u} = \sigma(D^{-1}A\sigma(D^{-1}AF_{s+u}\theta)\theta)$$
- **시각적 특징 생성**: Dall-E를 통해 Unseen class의 가상 이미지를 생성하고, CNN을 통해 특징 벡터 $V_g$를 추출한다. 이때 Softmax를 적용한 $\text{SoftLoss}$와 일반 $\text{HardLoss}$를 조합하여 학습한다.
  $$L = \alpha \cdot \text{CE}(V_g, V_c) + (1 - \alpha) \cdot \text{CE}(V''_g, V_c)$$

### 3. Unseen Class Classification Module

앞선 모듈에서 얻은 모든 특징을 결합하여 최종 분류를 수행한다.

- **특징 통합**: Seen/Unseen의 텍스트 및 시각적 특징을 결합하여 행렬 $M_c$를 생성한다.
- **GCN 분류**: Self-loop가 추가된 인접 행렬 $\tilde{A}$와 차수 행렬 $\tilde{D}$를 이용하여 GCN 레이어를 통해 최종 클래스를 예측한다.
  $$H^{(1)} = \text{ReLU}(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(0)} W^{(0)})$$
  최종 결과는 $\text{Softmax}(H^{(1)})$를 통해 출력된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: ImageNet, AWA2, aPY (Attribute Pascal and Yahoo).
- **평가 지표**: $\text{Hits@n}$ (예측 결과 중 정답이 상위 $n$개 안에 포함될 확률).
- **비교 대상**: ConSE, SYNC, EXEM, GCNZ, SGCN, DGP 등 기존 ZSL 및 KGE 모델.

### 2. 정량적 결과

- **ImageNet**: 1-hop Top-1 정확도에서 ZSLLM은 $33.46\%$를 기록하여, 가장 성능이 좋았던 DGP($26.16\%$)보다 크게 앞섰다. 2-hop, 3-hop으로 갈수록 성능 차이는 유지되거나 증가하는 경향을 보였다.
- **AWA2 및 aPY**: AWA2에서는 $85.32\%$, aPY에서는 $58.4\%$의 Top-1 정확도를 달성하여 모든 baseline 모델을 능가하였다.

### 3. 분석 결과

- **Ablation Study**: Seen class 컴포넌트를 제거했을 때 성능이 $1.22\%$p 하락했으며, Knowledge Distillation을 제거했을 때도 성능 저하가 확인되어 각 구성 요소의 필요성이 입증되었다.
- **하이퍼파라미터 분석**:
  - **Distillation Temperature**: $3.0$에서 최고 성능을 보였으며, 너무 높으면 클래스 구분 정보가 손실된다.
  - **GCN Layers**: $3000$ 레이어에서 정점을 찍고 이후 하락하는 양상을 보인다. 저자들은 이를 Over-smoothing 현상으로 설명한다.

## 🧠 Insights & Discussion

본 논문은 LLM의 생성 능력(ChatGPT, Dall-E)을 데이터 증강의 수단으로 활용하여 Zero-shot Learning의 고질적인 문제인 '학습 데이터 부재'를 영리하게 해결하였다. 특히 단순한 특징 추출을 넘어 Knowledge Distillation을 통해 LLM의 범용 지식을 특정 MMKG 태스크로 전이시킨 점이 성능 향상의 주요 요인으로 분석된다.

다만, 실험 결과 중 **GCN 레이어 수를 1,000에서 5,000까지 변화시키며 3,000에서 최적값을 찾았다는 기술**은 일반적인 GCN 연구(보통 2~3개 레이어에서 Over-smoothing 발생)와 비교했을 때 매우 이례적이다. 이에 대한 이론적 근거가 부족하며, 일반적인 GCN 구조에서 수천 개의 레이어를 쌓았을 때의 수렴 가능성에 대해 비판적인 검토가 필요하다. 또한, 외부 API(ChatGPT, Dall-E)에 의존하고 있어 모델의 버전 업데이트에 따라 결과가 달라질 수 있는 재현성 문제가 존재한다.

## 📌 TL;DR

본 연구는 LLM을 활용하여 Multi-modal Knowledge Graph의 Unseen categories를 위한 텍스트 및 이미지 특징을 생성하고, Knowledge Distillation을 통해 이를 학습시키는 ZSLLM 프레임워크를 제안하였다. ImageNet, AWA2, aPY 데이터셋에서 기존 SOTA 모델들을 압도하는 성능을 보였으며, 이는 LLM의 생성 능력과 GCN의 구조적 학습이 결합되어 Zero-shot 시나리오에서의 시맨틱 전이 효율을 극대화했기 때문이다. 향후 다른 KG 시나리오로의 확장 가능성이 높다.
