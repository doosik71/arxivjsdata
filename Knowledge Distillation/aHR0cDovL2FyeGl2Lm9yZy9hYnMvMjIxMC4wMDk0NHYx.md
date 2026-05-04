# Attention Distillation: self-supervised vision transformer students need more guidance

Kai Wang, Fei Yang, Joost van de Weijer (2022)

## 🧩 Problem to Solve

본 논문은 Self-Supervised Learning(SSL)으로 학습된 Vision Transformer(ViT)의 거대한 파라미터 규모와 높은 계산 비용 문제를 해결하고자 한다. ViT는 뛰어난 성능을 보이지만, 메모리와 계산 자원이 제한된 디바이스에서 사용하기에는 부적합하다. 이를 위해 지식 증류(Knowledge Distillation, KD)를 통해 거대한 Teacher 모델의 지식을 작은 Student 모델로 이전하는 것이 필요하다.

기존의 Self-Supervised Knowledge Distillation(SSKD) 연구들은 주로 ConvNet 아키텍처에 집중되어 있었으며, ViT의 핵심인 Attention 메커니즘을 적절히 활용하지 못해 Student 모델과 Teacher 모델 간의 성능 격차가 크게 발생하는 한계가 있었다. 따라서 본 논문의 목표는 SSL 기반의 ViT를 위한 효율적인 지식 증류 방법론을 제안하여, 작은 규모의 ViT(특히 ViT-T)에서도 SSL 수준의 고성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Teacher 모델이 이미지의 어느 영역을 중요하게 여기는지에 대한 '이유(Why)'를 Student 모델에게 전달하는 **Attention Distillation**이다. 단순히 Teacher의 최종 출력 결과(What)만을 모방하는 기존 방식에서 벗어나, ViT의 핵심 구성 요소인 Self-Attention Map을 직접 증류함으로써 Student 모델이 Teacher와 유사한 시각적 집중 영역을 갖도록 유도한다.

주요 기여 사항은 다음과 같다.
1. **AttnDistill 프레임워크 제안**: Projector Alignment(PA)와 Attention Guidance(AG)라는 두 가지 모듈을 통해 SSL 기반 ViT의 지식 증류를 수행한다.
2. **범용적 Attention 증류 메커니즘**: Teacher와 Student의 Head 수나 Patch 수가 서로 다르더라도 적용 가능한 유연한 Attention 정렬 방식을 제안하였다.
3. **최소 규모 ViT-T의 SSL 학습 성공**: 지식 증류를 통해 SSL 기반의 ViT-T 모델을 최초로 성공적으로 학습시켰으며, 이는 지도 학습(Supervised learning) 기반의 ViT-T 성능에 근접하는 결과를 보였다.

## 📎 Related Works

### Self-Supervised Learning (SSL)
SSL은 레이블이 없는 데이터에서 고품질의 특징 표현을 학습하는 방법으로, 크게 Contrastive Learning(예: MoCo, SimCLR)과 Masked Image Encoding(예: MAE) 두 가지 흐름으로 나뉜다. 최근에는 ConvNet에서 ViT로 백본 아키텍처가 확장되고 있으며, ViT는 ConvNet보다 Inductive Bias가 적어 대규모 데이터셋에서 더 큰 잠재력을 가진다.

### Knowledge Distillation for SSL
기존의 SSKD 연구들은 주로 ConvNet을 대상으로 하였으며, Pseudo labels를 활용하거나(CC), Memory bank를 통해 인스턴스 수준의 유사도 분포를 정렬하는(SEED, CompRess) 방식이 사용되었다. 또한, SimReg와 같이 Projector를 통해 특징 공간을 정렬하는 방식이 제안되었으나, ViT와 같이 아키텍처 차이가 큰 경우에는 단순한 Projector 회귀만으로는 지식 전이가 불충분하다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인
AttnDistill은 Teacher-Student 쌍의 ViT 모델을 구성하고, 두 가지 손실 함수인 Projector Alignment loss($L_c$)와 Attention Guidance loss($L_a$)를 결합하여 Student 모델을 최적화한다.

### 1. Projector Alignment (PA)
Teacher($V_t$)와 Student($V_s$) 모델은 서로 다른 특징 차원(Feature dimension)을 가지는 경우가 많다. 이를 해결하기 위해 선형 매핑 함수인 Projector $P$를 도입하여 Student의 특징 공간을 Teacher의 공간으로 투영한다. 특히 ViT에서 이미지 전체의 대표성을 띄는 Class token($E^c$)에 집중하여 다음과 같은 MSE 손실 함수를 사용한다.

$$L_c = ||E_t^c - P(E_s^c)||^2$$

### 2. Attention Guidance (AG)
단순한 특징 정렬은 '무엇(What)'이 있는지에 대한 정보만 제공하므로, '왜(Why)' 그러한 결과가 나왔는지에 대한 가이드를 제공하기 위해 Self-Attention Map을 증류한다. 

ViT의 Attention Map $A_{l,h}$는 다음과 같이 계산된다.
$$A_{l,h} = \text{Softmax}(Q_{l,h} \cdot (K_{l,h})^T / \sqrt{d})$$

본 논문에서는 특히 Class token이 다른 모든 토큰에 주는 영향력(Attention probabilities)을 추출하여, Student가 Teacher와 유사한 영역을 주목하도록 Kullback-Leibler(KL) Divergence를 사용하여 정렬한다. 아키텍처 차이에 따라 다음 네 가지 케이스로 나누어 처리한다.

- **Case (a) Head 및 Patch 수가 동일한 경우**: 각 Head별로 직접 KL Divergence를 적용한다.
  $$L_a = \sum_{h \in [1,H]} \text{KL}(A_t^h || A_s^h)$$
- **Case (b) Head 수는 같으나 Patch 수가 다른 경우**: Teacher의 Attention Map을 Bicubic interpolation을 통해 Student의 크기로 맞춘 후, 정규화(NR)를 거쳐 KL Divergence를 적용한다.
- **Case (c) Patch 수는 같으나 Head 수가 다른 경우**: Log-summation을 통해 여러 Head의 Attention 확률을 하나로 통합(Aggregation)한 뒤 정렬한다.
  $$a_j = \frac{1}{T} \cdot \sum_{h \in [1,H]} \log(a_j^h)$$
  $$A = \text{Softmax}([a_0, a_1, \dots, a_N])$$
- **Case (d) Head와 Patch 수가 모두 다른 경우**: Interpolation과 Aggregation을 순차적으로 적용한다.

### 훈련 목표 및 최종 손실 함수
최종적으로 Student 모델은 다음의 통합 손실 함수를 통해 학습된다.
$$L = L_c + \lambda \cdot L_a$$
여기서 $\lambda$는 Attention Guidance의 영향력을 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정
- **데이터셋**: ImageNet-Subset (Ablation 및 SSKD 비교용), ImageNet-1K (SSL 성능 비교용)
- **모델 쌍**: 
    - Mugs(ViT-S/16) $\rightarrow$ ViT-T/16
    - Mugs(ViT-B/16) $\rightarrow$ ViT-S/16
    - DINO(ViT-S/8) $\rightarrow$ ViT-S/16
- **지표**: k-NN Accuracy, Linear Probing (LP.), Fine-tuning Accuracy

### 주요 결과
1. **SSKD 성능**: ImageNet-Subset 실험에서 AttnDistill은 SEED, CompRess, SimReg 등 기존 ConvNet 기반 SSKD 방법론보다 월등히 높은 성능을 보였다. 특히 Student 모델의 규모가 작아질수록(예: 8-Layer, 3-Head) Attention Guidance의 효과가 극명하게 나타났다.
2. **SOTA 달성**: ImageNet-1K에서 ViT-T/16 모델을 증류했을 때, 기존의 SSL 기반 방법론들을 제치고 SOTA k-NN 및 LP 성능을 기록하였다. 이는 지도 학습 기반의 ViT-T 성능과 거의 대등한 수준이다.
3. **효율성**: DINO(ViT-S/8)를 Teacher로 하여 ViT-S/16을 학습시킨 경우, Patch 수를 줄임으로써 계산 비용을 75% 감소시키면서도 매우 높은 성능을 유지하였다.
4. **다운스트림 작업**: 반지도 학습(Semi-supervised learning) 설정(데이터 1% 사용 시)에서 Mugs(ViT-S/16) 대비 약 3.9%의 성능 향상을 보였으며, CIFAR-10/100 전이 학습에서도 기존 지도 학습 기반 소형 모델보다 우수한 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 ViT의 지식 증류에서 단순한 특징(Feature) 정렬보다 **Attention Map의 정렬이 훨씬 강력한 가이드**가 된다는 것을 입증하였다. 특히 "Attention Drift" 현상(Student의 시선이 Teacher와 달라지는 현상)을 방지함으로써, 작은 모델이 거대 모델의 시각적 추론 과정을 효과적으로 모방하게 만들었다.

### 한계 및 비판적 해석
- **아키텍처 의존성**: 제안된 방법은 Transformer의 Attention 메커니즘을 전제로 하므로, ConvNet과 ViT 간의 교차 증류(Cross-architecture distillation)에는 추가적인 계산 비용과 정의가 필요하다는 한계가 있다.
- **단일 Teacher 가정**: 현재의 프레임워크는 하나의 Teacher-Student 쌍만을 고려한다. 다수의 Teacher 모델로부터 지식을 통합하여 전이하는 방법(Multi-teacher distillation)에 대해서는 논의되지 않았다.
- **Finetuning 성능**: k-NN과 Linear Probing에서는 SOTA를 달성했으나, 전체 모델을 Fine-tuning 했을 때는 SOTA 모델과 약간의 성능 차이가 존재한다. 이는 특징 추출 단계의 정렬이 반드시 최종 분류 단계의 최적화로 이어지지는 않음을 시사한다.

## 📌 TL;DR

본 논문은 SSL 기반의 Vision Transformer를 효율적으로 압축하기 위해, 클래스 토큰의 정렬(PA)과 Attention Map의 정렬(AG)을 결합한 **AttnDistill** 방법론을 제안한다. 특히 서로 다른 규모의 ViT 간에도 적용 가능한 유연한 Attention 정렬 방식을 도입하여, **최초로 SSL 기반의 초소형 ViT-T 모델 학습에 성공**하였으며, k-NN 및 Linear Probing 지표에서 SOTA 성능을 달성하였다. 이 연구는 향후 더욱 거대한 SSL 기반 ViT 모델이 등장했을 때, 이를 실용적인 크기의 모델로 경량화하는 데 중요한 기반 기술이 될 것으로 보인다.