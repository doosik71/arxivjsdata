# EfficientVLM: Fast and Accurate Vision-Language Models via Knowledge Distillation and Modal-adaptive Pruning

Tiannan Wang, Wangchunshu Zhou, Yan Zeng, Xinsong Zhang (2022)

## 🧩 Problem to Solve

최근 대규모 사전 학습된 Vision-Language Models(VLMs)는 다양한 시각-언어 작업에서 뛰어난 성능을 보이고 있다. 그러나 대부분의 VLMs는 수억 개의 파라미터로 구성되어 있어, 실제 환경에 배포할 때 공간, 메모리 및 지연 시간(latency) 제약으로 인해 파인튜닝과 배포에 큰 어려움이 따른다.

기존의 모델 압축 시도(예: MiniVLM, DistilVLM)는 주로 Object-feature 기반의 VLM에 국한되어 있었다. 이러한 방식은 시각 특징 추출기(vision feature extractor)를 Transformer 모델과 함께 end-to-end 방식으로 증류(distill)할 수 없다는 한계가 있으며, 결과적으로 일반적인 크기의 VLM에 비해 성능이 크게 떨어지는 문제가 있었다. 본 논문의 목표는 fully Transformer-based VLM을 효율적으로 압축하여, 성능 하락을 최소화하면서도 속도와 효율성을 극대화한 소형 VLM을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'증류 후 가지치기(distilling then pruning)'** 프레임워크이다. 이는 단순히 모델의 크기를 줄이는 것이 아니라, 사전 학습 단계에서 일반적인 능력을 보존하는 증류를 수행하고, 이후 파인튜닝 단계에서 작업별로 모달리티의 중요도를 다르게 적용하는 적응형 가지치기를 수행하는 2단계 전략이다.

1. **Task-agnostic Compression**: 사전 학습 단계에서 Knowledge Distillation(KD)을 통해 교사 모델(Teacher)의 지식을 학생 모델(Student)에게 전달하여, 특정 작업에 종속되지 않은 콤팩트한 VLM을 생성한다.
2. **Modal-adaptive Pruning**: VLM의 각 모달리티(시각, 텍스트, 교차 모달리티)가 작업마다 가지는 중요도가 다르다는 점에 착안하여, 미분 가능한 $L_0$-norm 근사화를 통해 모달리티별 중요도를 자동으로 추론하고 중복 구조를 제거하는 적응형 가지치기 알고리즘을 제안한다.

## 📎 Related Works

**Vision-Language Pre-training (VLP)**
기존의 VLP 연구는 크게 두 가지 방향으로 나뉜다. 첫째는 Object Detection에 의존하여 이미지를 객체 중심 특징으로 표현하는 방식이다. 이는 고해상도 입력이 필요하고 시간이 많이 소요되며, end-to-end 최적화가 어렵다는 단점이 있다. 둘째는 CNN이나 Vision Transformer(ViT)를 사용하여 이미지를 인코딩하는 방식으로, 추론 속도는 빠르나 일부 정교한 시각-언어 정렬(fine-grained alignment) 능력이 부족할 수 있다.

**Pre-trained Model Compression**
BERT와 같은 언어 모델 압축을 위해 Knowledge Distillation, Pruning, Quantization 등이 연구되어 왔다. 하지만 VLM 분야에서는 이러한 압축 기법의 적용이 적었으며, 기존의 DistilVLM 등은 앞서 언급한 object-feature 기반 모델의 한계로 인해 최신 SOTA 모델들에 비해 성능이 낮았다. 본 연구는 fully Transformer-based VLM을 대상으로 KD와 Pruning을 결합한 첫 번째 시도라는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 모델 구조 (Model Overview)

EfficientVLM은 SOTA 모델인 X-VLM을 기반으로 하며, 구조를 절반으로 축소하였다.

- **Vision Encoder**: 12개 층 $\rightarrow$ 6개 층
- **Text Encoder**: 6개 층 $\rightarrow$ 3개 층
- **Cross-modal Encoder**: 6개 층 $\rightarrow$ 3개 층
- **전체 파라미터**: 약 93M (X-VLM의 44.3%)

### 2. Knowledge Distillation을 이용한 사전 학습

학생 모델을 X-VLM의 짝수 번째 층만 유지하여 초기화한 후, 다음과 같은 세 가지 KD 목적 함수를 사용하여 사전 학습을 진행한다.

**Attention Distillation**
교사 모델과 학생 모델의 self-attention 행렬 사이의 MSE(Mean Square Error)를 최소화한다.
$$L_{attn} = \frac{1}{h} \sum_{j=1}^{L} \sum_{i=1}^{h} \text{MSE}(A^S_{i,j}, A^T_{i,2j})$$
여기서 $A^S$와 $A^T$는 각각 학생과 교사 모델의 attention 행렬을 의미한다.

**Hidden States Distillation**
중간 층의 은닉 상태(hidden states)를 모방하도록 유도한다.
$$L_{hid} = \sum_{i=1}^{L} \text{MSE}(H^S_i, H^T_{2i})$$

**Logits Distillation**
최종 예측값(logits)의 분포를 맞추기 위해 KL-Divergence를 사용한다.

최종 사전 학습 손실 함수는 기존 VLP 손실($L_{VLP}$)과 KD 손실($L_{KD}$)의 결합으로 정의된다.
$$L_{pretrain} = \lambda L_{VLP} + (1-\lambda) L_{KD}$$

### 3. Modal-adaptive Pruning을 이용한 파인튜닝

본 논문은 모든 모달리티가 모든 작업에서 동일하게 중요하지 않다는 점을 발견하였다. 예를 들어, 이미지-텍스트 검색(ITR)에서는 시각 인코더가 중요하지만, NLVR2 작업에서는 텍스트 인코더의 중요도가 상대적으로 높다.

**알고리즘 흐름**
미분 가능한 $L_0$-norm 정규화의 근사치인 Hard-Concrete distribution을 사용하여 파라미터 $\theta$에 마스크 $z \in \{0, 1\}$를 적용한다.
$$\tilde{\theta} = \theta \cdot z$$
최적화 목표는 다음과 같이 학습 손실과 모델 크기에 대한 정규화 항을 결합하는 것이다.
$$\mathbb{E}_z \left[ \frac{1}{D} \sum_{i=1}^{D} L(x_i, y_i; \tilde{\theta}) + \lambda \|\tilde{\theta}\|_0 \right]$$

실제 구현에서는 타겟 모델 크기를 강제하기 위해 Lagrangian multiplier를 도입한 손실 함수 $L_{Lgr}$를 사용하며, 최종 파인튜닝 목적 함수는 다음과 같다.
$$L_{ft} = \lambda L_{VL} + (1-\lambda) L_{KD} + L_{Lgr}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: COCO, Visual Genome(인도메인), SBU Captions, Conceptual Captions(아웃오브도메인).
- **평가 작업**: Image-Text Retrieval (ITR), VQA 2.0, NLVR2, COCO Caption Generation.
- **비교 대상**: MiniVLM, DistilVLM, ViLT, X-VLM small 등.

### 2. 주요 결과

- **성능 회복**: EfficientVLM은 교사 모델(X-VLM) 성능의 **98.4%를 유지**하면서 파라미터 수는 44.3%로 줄였다.
- **속도 향상**: 추론 속도를 GPU에서 **1.9배**, CPU에서 **2.2배** 가속화하였다.
- **정량적 지표**:
  - **ITR**: R@1(Text Retrieval) 78.7%, R@1(Image Retrieval) 60.6% 달성. 기존 효율적 VLM들보다 절대적으로 높은 성능을 보였다.
  - **NLVR2**: Accuracy 81.83%(val) / 81.72%(test)로 기존 모델들을 크게 상회하였다.
  - **COCO Caption**: CIDEr 점수 127.3을 기록하였다.

### 3. 절제 연구 (Ablation Study)

- **KD의 효과**: Logits, Hidden states, Attention distillation을 각각 추가할 때마다 성능이 점진적으로 향상됨을 확인하였다.
- **Pruning의 효과**: 단순히 파인튜닝하는 것보다 Modal-adaptive pruning과 KD를 함께 적용했을 때 가장 높은 성능이 나타났다.
- **Sparsity 분석**: 모델의 40%~50%를 가지치기 하더라도 교사 모델 성능의 95% 이상을 유지하는 강인함을 보였다.

## 🧠 Insights & Discussion

**강점**
본 연구는 fully Transformer-based VLM을 대상으로 end-to-end 압축 파이프라인을 성공적으로 구축하였다. 특히 모달리티별 중요도가 작업에 따라 다르다는 통찰을 바탕으로 '적응형 가지치기'를 도입하여, 단순한 일괄 압축보다 훨씬 효율적인 성능-속도 트레이드오프를 달성하였다.

**한계 및 논의**

- **범용성**: 본 프레임워크는 X-VLM에 적용되었으며, 다른 최신 VLM 구조에서도 동일한 효과가 나타날지에 대한 추가 검증이 필요하다.
- **압축 기법의 확장**: Knowledge Distillation과 Pruning 외에, 양자화(Quantization)나 행렬 분해(Matrix Decomposition)와 같은 추가적인 압축 기법을 결합한다면 더 극단적인 효율화를 이룰 수 있을 것으로 보인다.
- **데이터 효율성**: EfficientVLM은 4M의 이미지-텍스트 쌍으로 학습되었음에도 7M을 사용한 DistilVLM보다 뛰어난 성능을 보였는데, 이는 제안한 증류 및 가지치기 전략이 데이터 효율성을 높이는 데 기여했음을 시사한다.

## 📌 TL;DR

본 논문은 대규모 VLM의 배포 문제를 해결하기 위해 **'증류 후 가지치기(distilling then pruning)'** 프레임워크를 제안하였다. 사전 학습 단계에서는 KD를 통해 일반적인 능력을 전이하고, 파인튜닝 단계에서는 작업별 모달리티 중요도를 자동으로 추론하는 **Modal-adaptive Pruning**을 적용하였다. 그 결과, 파라미터를 44.3%로 줄이면서도 교사 모델 성능의 98.4%를 보존하고 추론 속도를 2.2배 향상시킨 **EfficientVLM**을 개발하였으며, 이는 기존의 소형 VLM들보다 월등한 성능을 보여준다는 것을 입증하였다.
