# NOTES ON THE MATHEMATICAL STRUCTURE OF GPT LLM ARCHITECTURES

Spencer Becker-Kahn (2024)

## 🧩 Problem to Solve

본 논문은 현대의 대규모 언어 모델(LLM), 특히 GPT-3 스타일의 트랜스포머(Transformer) 아키텍처가 수학적으로 어떻게 구성되어 있는지를 엄밀하게 정의하는 것을 목표로 한다. 저자는 많은 주요 LLM 관련 논문들이 모델의 세부 구조를 명확한 수학적 언어나 완전한 의사코드(pseudo-code)로 기술하지 않고 모호하게 설명하는 경향이 있다는 점에 주목한다. 따라서 본 연구의 목적은 컴퓨터 비전이나 딥러닝의 구현 관점이 아닌, 순수 수학적 관점에서 LLM의 아키텍처를 함수들의 합성으로 정의하여 그 구조를 투명하게 공개하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 GPT-3 스타일의 LLM을 유클리드 공간 사이의 매핑(map)으로 간주하고, 이를 구성하는 각 단계를 수학적으로 정형화한 것이다. 구체적으로는 텍스트 데이터가 토큰화(Tokenization)를 거쳐 임베딩(Embedding)되고, 디코더 스택(Decoder Stack)의 어텐션 레이어(Attention Layer)와 피드포워드 레이어(Feedforward Layer)를 통과하여 최종적으로 다음 토큰의 확률 분포를 예측하기까지의 전 과정을 함수적 체인으로 정의하였다.

## 📎 Related Works

저자는 Phuong과 Hutter의 연구[1]를 언급하며, 트랜스포머에 대한 형식적 알고리즘(Formal algorithms)의 필요성이 제기되었음을 밝힌다. 기존의 LLM 관련 논문들이 공학적인 구현에 치중하여 수학적 세부 사항을 생략했다는 점을 한계로 지적하며, 본 논문은 이를 보완하기 위해 아키텍처의 수학적 구조를 명시적으로 기술하는 데 집중한다.

## 🛠️ Methodology

본 논문은 GPT-3 스타일 LLM의 전체 파이프라인을 다음과 같은 단계로 설명한다.

### 1. 토큰화, 인코딩 및 임베딩 (Tokenization, Encodings and Embeddings)
텍스트를 수치 데이터로 변환하는 과정은 다음과 같은 단계를 거친다.
- **Vocabulary 구성**: 말뭉치(Corpus) $C$에서 UTF-8 인코딩을 통해 기본 어휘집 $V_0$를 생성하고, Byte Pair Encoding(BPE) 알고리즘을 반복 적용하여 최종 어휘집 $V$ (크기 $n_{vocab}$)를 구축한다.
- **One-Hot Encoding**: 어휘집 $V$의 각 토큰을 $\mathbb{R}^{n_{vocab}}$ 공간의 표준 정규직교 기저 벡터로 매핑하는 전단사 함수 $\sigma: V \rightarrow \{e_j\}_{j=1}^{n_{vocab}}$를 정의한다.
- **Embedding**: 원-핫 벡터를 낮은 차원의 벡터 공간 $\mathbb{R}^d$로 투영하기 위해 학습 가능한 행렬 $W_E \in \mathbb{R}^{d \times n_{vocab}}$를 사용한다. 입력 행렬 $t \in \mathbb{R}^{n_{ctx} \times n_{vocab}}$에 대해 임베딩 결과 $X$는 다음과 같이 계산된다.
  $$X = tW_E^T$$
- **Unembedding**: 모델의 최종 출력을 다시 어휘집 공간으로 되돌리기 위해 학습 가능한 행렬 $W_U \in \mathbb{R}^{n_{vocab} \times d}$를 사용하여 $XW_U^T$를 계산한다.

### 2. 피드포워드 레이어 (Feedforward Layers)
피드포워드 레이어는 다층 퍼셉트론(MLP)을 기반으로 하며, 각 행(토큰)에 독립적으로 적용된다.
- **MLP 구조**: 입력 $x \in \mathbb{R}^d$에 대해 가중치 행렬 $W^{(l)}$와 편향 $b^{(l)}$을 이용하여 다음과 같이 층을 쌓는다.
  $$z^{(l+1)} = b^{(l+1)} + W^{(l+1)}\sigma(z^{(l)})$$
- **Feedforward Layer 정의**: MLP $m$이 있을 때, 피드포워드 레이어 $\text{ff}_m$은 잔차 연결(Residual Connection)을 포함하여 다음과 같이 정의된다.
  $$\text{ff}_m(X) = X + m(X)$$

### 3. 어텐션 레이어 (Attention Layers)
어텐션 레이어는 행 간의 상호작용을 계산하며, 자기회귀(Autoregressive) 특성을 유지하기 위해 마스킹을 적용한다.
- **Autoregressive Masking**: $\text{softmax}^*$ 함수는 $i < j$인 경우 값을 $0$으로 만들어 미래 토큰을 참조하지 못하게 한다.
- **Attention Head**: 하나의 헤드 $h$는 쿼리(Query), 키(Key), 값(Value), 출력(Output) 행렬을 사용하여 다음과 같이 계산된다.
  $$h(X) = \left( \text{softmax}^*\left( X W_{qk}^h X^T \right) \otimes W_{ov}^h \right) X$$
  여기서 $W_{qk}^h = (W_q^h)^T W_k^h$ 이고 $W_{ov}^h = W_o^h W_v^h$ 로 정의되어 저차원 투영이 이루어진다.
- **Attention Layer**: 여러 개의 어텐션 헤드 집합 $H$에 대해 잔차 연결을 적용한다.
  $$\text{attn}_H(X) = X + \sum_{h \in H} h(X)$$

### 4. 전체 트랜스포머 구조 (The Full Transformer)
- **Residual Block**: 어텐션 레이어와 피드포워드 레이어의 합성으로 정의된다.
  $$B(H, m) = \text{ff}_m \circ \text{attn}_H$$
- **Decoder Stack**: $n$개의 잔차 블록을 순차적으로 합성한 함수 $D$이다.
  $$D = B_n \circ B_{n-1} \circ \dots \circ B_1$$
- **전체 파이프라인**: 트랜스포머 $T$는 다음과 같은 함수 합성으로 표현된다.
  $$T = (\text{Unembedding}) \circ (\text{Decoder Stack } D) \circ (\text{Embedding})$$

### 5. 다음 토큰 예측 (Logits and Predictions)
입력 문장 $S$에 대해 $T(t(S))$의 마지막 행을 로짓(logits) $l(S) \in \mathbb{R}^{n_{vocab}}$로 정의하며, 이에 소프트맥스 함수를 적용하여 다음 토큰 $i$가 나타날 확률 $P_S(i)$를 구한다.
$$P_S(i) = (\text{softmax}(l(S)))_i$$

## 📊 Results

본 논문은 특정 데이터셋을 이용한 성능 측정이나 실험적 결과(정량적/정성적 결과)를 제시하지 않는다. 이는 본 논문의 목적이 새로운 모델을 제안하거나 성능을 개선하는 것이 아니라, 기존 GPT-3 스타일 아키텍처의 **수학적 구조를 엄밀하게 정의하는 것**에 있기 때문이다. 따라서 실험 섹션은 존재하지 않는다.

## 🧠 Insights & Discussion

본 논문은 모호하게 기술되던 LLM의 아키텍처를 함수 해석학적 관점에서 명확히 정의했다는 점에서 강점을 가진다. 특히 어텐션 헤드의 저차원 투영 구조와 잔차 연결을 포함한 블록의 합성을 수학적으로 명시함으로써, 모델의 연산 흐름을 정확히 파악할 수 있게 한다.

다만, 다음과 같은 한계와 가정들이 존재한다.
- **위치 인코딩(Positional Encoding)**: 저자는 위치 인코딩의 다양한 방식이 존재함을 언급하며, 단순하게 학습 가능한 행렬 $P$를 더하는 방식만을 예시로 들고 세부 논의에서는 제외하였다.
- **학습 알고리즘 부재**: 본 보고서는 오직 '아키텍처'에만 집중하고 있으며, 손실 함수(Loss Function)나 최적화 알고리즘(Optimizer) 등 실제 모델을 학습시키기 위한 절차는 전혀 다루지 않았다.
- **단순화된 구조**: 실제 GPT-3 등의 모델에서 사용되는 Layer Normalization과 같은 정규화 과정이 수학적 정의에서 생략되어 있어, 실제 구현체와는 약간의 간극이 있을 수 있다.

## 📌 TL;DR

본 논문은 GPT-3 스타일의 LLM 아키텍처를 순수 수학적 함수들의 합성으로 정의한 이론적 노트이다. 토큰화 $\rightarrow$ 임베딩 $\rightarrow$ 디코더 스택(어텐션 $\rightarrow$ 피드포워드) $\rightarrow$ 언임베딩 $\rightarrow$ 확률 예측으로 이어지는 전 과정을 엄밀한 수식으로 정형화하였다. 이 연구는 LLM의 내부 동작 원리를 수학적으로 분석하려는 향후 연구나, 모델의 형식적 검증(Formal Verification)을 위한 기초 자료로 활용될 가능성이 높다.