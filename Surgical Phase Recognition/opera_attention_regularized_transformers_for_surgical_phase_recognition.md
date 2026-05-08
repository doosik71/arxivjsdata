# OperA: Attention-Regularized Transformers for Surgical Phase Recognition

Tobias Czempiel et al. (2021)

## 🧩 Problem to Solve

본 논문은 복강경 담낭 절제술(laparoscopic cholecystectomy)과 같은 긴 비디오 시퀀스에서 수술 단계(surgical phase)를 정확하게 인식하는 문제를 해결하고자 한다. 수술 단계 인식은 미래의 스마트 수술실(Operating Room)을 구축하는 데 있어 핵심적인 요소이며, 이를 통해 수술 중 실시간 피드백 제공, 오류나 이상 이벤트 발생 시 알림, 그리고 수술 교육을 위한 자동화된 수술 요약 및 분석 등의 기반을 마련할 수 있다.

그러나 수술 영상 분석에는 다음과 같은 어려움이 존재한다. 첫째, 환자의 해부학적 구조의 다양성과 집도의의 수술 스타일 차이로 인해 데이터의 변동성이 크다. 둘째, 학습에 사용할 수 있는 고품질의 데이터셋이 제한적이다. 기존의 LSTM 기반 모델들은 긴 시퀀스를 처리할 때 이전 정보를 망각하는 경향이 있어, 수술과 같은 장시간의 시퀀스 모델링에 한계가 있었다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Transformer 아키텍처를 수술 단계 인식에 도입하고, 모델이 학습 과정에서 고품질의 프레임에 집중하도록 유도하는 **Attention Regularization** 손실 함수를 제안한 것이다. 주요 기여 사항은 다음과 같다.

1. 수술 단계 인식 분야에 Transformer 기반 모델을 성공적으로 적용하여 기존의 temporal refinement 방법들보다 우수한 성능을 달성하였다.
2. CNN 백본의 예측 신뢰도가 높은 프레임에 더 많은 가중치를 두도록 강제하는 새로운 Attention Regularizer를 제안하여 특징 추출의 품질을 높였다.
3. 학습된 Attention weight를 활용하여 각 수술 단계의 특징을 가장 잘 나타내는 대표 프레임(Characteristic frames)을 추출하고 시각화하는 방법을 제시하였다.
4. 두 가지 서로 다른 수술 비디오 데이터셋(Cholec80, MitiSW)을 통해 제안 방법의 성능을 정밀하게 검증하였다.

## 📎 Related Works

기존의 수술 단계 인식 연구는 다음과 같이 발전해 왔다. 초기에는 평균적인 수술 흐름과 비교하는 이진 수술 신호(binary surgical signals) 방식이나 Hidden Markov Models (HMM)이 사용되었다. 이후 CNN을 이용해 이미지 특징을 추출하고 이를 직접 분류하는 EndoNet이 등장했으며, 이후 LSTM을 결합한 EndoLSTM이나 ResLSTM 등이 temporal refinement를 통해 성능을 개선하였다. 최근에는 Temporal Convolutional Networks (TCN)를 결합한 TeCNO와 같은 모델들이 제안되었다.

본 논문은 이러한 기존 방식들이 가진 한계, 특히 LSTM의 망각 문제와 긴 시퀀스 처리의 어려움을 극복하기 위해 self-attention 메커니즘을 사용하는 Transformer를 도입하였다. 특히 기존 연구들이 Transformer를 주로 수술 도구 분류에만 사용했던 것과 달리, 본 논문은 이를 수술 단계 인식이라는 더 넓은 범위의 시퀀스 분석 문제에 적용하고, attention weight를 통한 설명 가능성(explainability)을 부여했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

OperA 모델은 크게 이미지 특징을 추출하는 **CNN Backbone**과 이를 시퀀스로 처리하는 **Sequential Transformer Network**로 구성된다. 전체 구조는 $\text{ResNet-50} \rightarrow \text{Transformer Layers} \rightarrow \text{Linear/Softmax}$ 순으로 이어진다.

### 상세 구성 요소 및 절차

**1. Feature Extraction Backbone**

- ResNet-50을 사용하여 프레임별로 이미지 특징 $F \in \mathbb{R}^{2048}$와 클래스 확률 $p(F) \in [0,1]^c$를 추출한다.
- 이 백본은 수술 단계 인식 및 도구 검출 작업으로 사전 학습되며, Transformer 학습 시에는 가중치가 고정(frozen)된다.

**2. Sequential Transformer Network**

- 11개의 Transformer 레이어를 쌓아 올린 구조이다. 각 레이어는 linear layer, scaled dot-product attention, layer normalization, 그리고 residual connection으로 구성된다.
- **Causal Masking**: 실시간(online) 예측을 가능하게 하기 위해, 현재 프레임이 미래의 프레임 정보를 참조하지 못하도록 binary mask $M \in \{0,1\}$을 적용한다.
- attention 메커니즘의 수식은 다음과 같다.
  $$\text{AttentionWeights}(Q, K) = \text{softmax}\left(\text{mask}\left(\frac{QK^T}{\sqrt{d}}\right)\right)$$
  $$\text{Attention}(Q, K, V) = \text{AttentionWeights}(Q, K)V$$
- 최종 출력층에서는 linear layer와 softmax를 통해 각 프레임의 단계 확률 $y$를 예측하며, median frequency balanced cross-entropy loss $L_c$를 사용하여 학습한다.

**3. Normalized Frame-Wise Attention**

- Causal mask로 인해 앞쪽 프레임은 여러 번 참조되지만 뒤쪽 프레임은 적게 참조되는 불균형이 발생한다. 이를 해결하기 위해 프레임별 attention 가중치를 정규화한 $n$을 계산한다.
  $$n_j = \frac{\sum_i A_{ij}}{\sum_i M_{ij}}$$
  여기서 $A_{ij}$는 attention weight이며, $M_{ij}$는 mask 값이다.

**4. Attention Regularization**

- CNN 백본이 잘못 예측한 프레임(낮은 신뢰도)에 Transformer가 과하게 의존하는 것을 방지하기 위해, CNN의 예측 에러와 attention weight를 결합한 regularization loss를 도입한다.
  $$L_{reg} = \langle n, \text{CEE}(p(F), y) \rangle$$
  여기서 $\text{CEE}$는 CNN 예측값 $p(F)$와 정답 $y$ 사이의 Cross Entropy Evaluation 값이다. 즉, CNN의 확신이 낮은(에러가 큰) 프레임에 높은 attention $n$이 할당되면 패널티를 준다.
- 최종 손실 함수는 다음과 같다.
  $$L = L_c + \lambda \cdot L_{reg}$$

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80 (80개 비디오, 7개 단계) 및 MitiSW (85개 비디오, 8개 단계)를 사용하였다. 두 데이터셋 모두 1fps로 샘플링되었다.
- **지표**: 비디오 수준의 Accuracy(Acc)와 F1-score를 측정하였으며, 5-fold cross validation을 수행하였다.
- **비교 대상**: ResNet-50 (Baseline), ResLSTM, MTRCNet-CL, TeCNO.

### 정량적 결과

- **성능 향상**: OperA는 두 데이터셋 모두에서 기존의 temporal refinement 모델들보다 우수한 성능을 보였다. 특히 Cholec80에서 Accuracy 기준 baseline 대비 약 2~6% 향상된 결과를 보였다.
- **Ablation Study**:
  - 레이어 수: 6개 레이어보다 11개 레이어를 사용했을 때 Accuracy와 F1-score가 약 1% 향상되었다.
  - Regularization 효과: Attention Regularization을 적용했을 때 성능이 추가적으로 약 1% 향상됨을 확인하였다.
  - Positional Encoding (PE): 흥미롭게도 PE를 추가했을 때 성능이 약간 저하되는 경향이 있었는데, 이는 수술 영상의 시퀀스 길이가 NLP 태스크보다 매우 길기 때문으로 추측된다.

### 정성적 결과

- **예측 일관성**: OperA의 예측 결과는 단순 CNN 결과보다 훨씬 매끄럽고(smoother) 일관성 있게 나타났다.
- **Attention 분석**: Regularization을 적용한 모델의 경우, 높은 attention을 받는 프레임(HA frames)이 수술 단계의 특징을 잘 나타내는 핵심 프레임(예: 특정 수술 도구가 명확히 보이는 프레임)과 일치함을 확인하였다. 반면, 낮은 attention(LA frames)은 CNN 예측이 틀렸거나 정보량이 적은 프레임에서 나타났다.

## 🧠 Insights & Discussion

본 논문은 Transformer의 self-attention이 수술 영상과 같은 긴 시퀀스에서 long-term dependency를 모델링하는 데 매우 효과적임을 입증하였다. 특히 주목할 점은 **Attention Regularization**의 역할이다. 단순히 모델의 성능을 높이는 것을 넘어, 모델이 "어떤 프레임을 믿어야 하는지"를 학습하게 함으로써 attention weight의 해석 가능성을 높였다.

하지만 몇 가지 논의할 점이 있다. 첫째, Positional Encoding이 성능을 저하시킨 점은 수술 영상의 시간적 특성이 일반적인 텍스트와는 다르다는 것을 시사하며, 이에 적합한 새로운 positional embedding 방식에 대한 연구가 필요함을 보여준다. 둘째, CNN 백본을 고정(frozen)하고 학습했으므로, 백본과 Transformer를 end-to-end로 미세 조정(fine-tuning)했을 때의 성능 향상 가능성이 남아 있다.

결론적으로, 본 연구는 attention weight를 통해 수술의 핵심 장면을 자동으로 추출할 수 있음을 보여주었으며, 이는 향후 수술 비디오 요약(Surgery Summarization) 시스템으로 확장될 가능성이 매우 높다.

## 📌 TL;DR

본 논문은 수술 단계 인식을 위해 Transformer 기반의 **OperA** 모델을 제안하며, CNN의 예측 신뢰도에 따라 attention을 조절하는 **Attention Regularization**을 도입하여 성능과 해석 가능성을 동시에 잡았다. 실험 결과, 기존 LSTM 및 TCN 기반 모델들을 압도하는 성능을 보였으며, 특히 수술의 핵심 프레임을 정확하게 식별해냄으로써 향후 수술 영상 자동 요약 기술에 기여할 수 있는 기반을 마련하였다.
