# Exploring Vision-Language Models for Imbalanced Learning

Yidong Wang, Zhuohao Yu, Jindong Wang, Qiang Heng, Hao Chen, Wei Ye, Rui Xie, Xing Xie, Shikun Zhang (2023)

## 🧩 Problem to Solve

본 연구는 Vision-Language Models (VLMs)가 데이터 분포가 불균형한 imbalanced dataset, 특히 long-tailed recognition 상황에서 겪는 성능 저하 문제를 해결하고자 한다. 일반적인 VLM은 대규모 데이터셋을 통한 contrastive language-image pre-training을 통해 우수한 zero-shot classification 성능을 보이지만, 학습 데이터의 클래스 분포가 편향된 경우 다수 클래스(head classes)에 비해 소수 클래스(tail classes)의 예측 성능이 현저히 떨어진다는 한계가 있다.

이 문제의 중요성은 자율 주행이나 의료 진단과 같이 소수 클래스의 정확한 예측이 안전 및 생명과 직결되는 응용 분야에서 매우 크다. 실제로 CLIP과 같은 모델이 iNaturalist18 데이터셋에서 단 5%의 정확도를 기록했다는 점은 VLM이 불균형 데이터셋에 취약함을 보여준다. 따라서 본 논문의 목표는 VLM에 적절한 구조적 수정과 imbalanced learning 알고리즘을 결합하여, 특히 tail 클래스에 대한 인식 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VLM의 frozen image encoder 뒤에 가벼운 decoder를 추가하여 tail 클래스의 미세한 특징(nuanced features)을 캡처하고, 이를 기존의 imbalanced learning 알고리즘과 결합하는 것이다.

주요 기여 사항은 다음과 같다.

1. **VLM과 Imbalanced Learning의 결합 탐색**: VLM을 불균형 분류 문제에 적용하는 체계적인 방법론을 최초로 제시하고 광범위한 실험적 분석을 수행하였다.
2. **Lightweight Decoder 제안**: 대규모 클래스 수로 인한 OOM(Out-Of-Memory) 문제를 방지하고, pre-training 단계에서 부족했던 소수 클래스의 표현력을 보완하기 위해 가벼운 decoder 구조를 도입하였다.
3. **다양한 imbalanced 알고리즘 적용**: Focal Loss, Balanced SoftMax, Distribution Alignment 등 기존의 손실 함수 엔지니어링 및 two-stage 학습 방법을 VLM 파이프라인에 통합하여 성능 향상을 입증하였다.
4. **분석적 통찰 제공**: pre-training 데이터의 크기, backbone 모델의 크기, 학습 비용(메모리 및 탄소 배출량)이 불균형 학습 성능에 미치는 영향을 심층 분석하였다.

## 📎 Related Works

### Vision Foundation Models

CLIP, BLIP과 같은 VLM은 자연어 텍스트를 통해 시각적 개념을 학습함으로써 추가 학습 없이도 새로운 클래스를 인식하는 zero-shot 능력을 갖추었다. 최근 LAION-5B와 같은 거대 데이터셋의 등장으로 성능이 더욱 향상되었으나, 이러한 모델들의 학습 목적 함수는 주로 contrastive learning에 집중되어 있어, 클래스 간 불균형이 심한 downstream task에서의 최적화 문제는 충분히 다루어지지 않았다.

### Imbalanced Learning

기존의 불균형 학습 접근 방식은 크게 네 가지로 분류된다.

- **Loss function engineering**: 손실 함수에 가중치를 부여하거나 logits을 조정하여 gradient의 균형을 맞춘다.
- **Two-stage Decision boundary adjustment**: 먼저 일반적인 학습으로 표현(representation)을 학습한 뒤, classifier head만을 재조정하는 방식이다.
- **Task-specific architecture design**: 불균형 문제 해결을 위한 특수 구조를 설계한다.
- **Transfer learning 및 기타**: 외부 데이터를 활용하거나 도메인 적응 기법을 사용한다.

본 논문은 이러한 기존의 imbalanced learning 기법들을 VLM의 강력한 feature extraction 능력과 결합하여 시너지를 내고자 하며, 특히 단순한 linear probing나 full fine-tuning보다 효율적인 decoder 기반의 접근 방식을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서 제안하는 파이프라인은 $\text{Frozen Image Encoder} \rightarrow \text{Lightweight Decoder} \rightarrow \text{Classifier}$ 순으로 구성된다. VLM의 text encoder는 imbalanced learning 알고리즘과의 결합이 어렵고 연산 비용이 높기 때문에 사용하지 않는다.

### 주요 구성 요소 및 역할

1. **Frozen Image Encoder**: CLIP 등의 모델에서 제공하는 ViT-L14 등을 사용하며, 일반적인 이미지 표현을 추출하는 역할을 한다. 가중치는 고정(frozen) 상태로 유지된다.
2. **Lightweight Decoder**: ViT의 블록과 유사한 3개의 Transformer block으로 구성된다. 각 블록은 multi-head attention과 MLP로 이루어져 있으며, encoder가 놓친 소수 클래스의 세부 특징을 학습한다.
3. **Classifier**: Decoder의 출력을 받아 최종 클래스를 분류하는 linear layer이다.

### 학습 절차 및 손실 함수

학습은 크게 두 가지 단계 또는 단일 단계로 진행될 수 있다.

#### 1. Standard Training (One-stage)

가장 기본적인 방식은 instance-balanced sampling과 Cross-Entropy (CE) loss를 사용하는 것이다.
$$\ell(D_s; \tilde{\theta}_e, \theta_d, \theta_c) = -\frac{1}{s} \sum_{i=1}^{s} \log \frac{\exp(\eta_{y_i})}{\sum_{j=1}^{K} \exp(\eta_j)}$$
여기서 $\eta_j$는 클래스 $j$에 대한 classification score(logit)이다.

#### 2. Class-specific Loss (Loss Engineering)

클래스별 가중치 $w_{y_i}$와 logit bias $\delta_j$를 도입하여 gradient의 불균형을 해소한다.
$$\ell(D_s; \tilde{\theta}_e, \theta_d, \theta_c) = -\frac{1}{s} \sum_{i=1}^{s} w_{y_i} \log \frac{\exp(\eta_{y_i} + \delta_{y_i})}{\sum_{j=1}^{K} \exp(\eta_j + \delta_j)}$$

#### 3. Two-stage Training

표현 학습과 분류기 학습을 분리하는 전략이다.

- **1단계**: 일반적인 CE loss로 encoder와 decoder를 학습시켜 양질의 representation을 얻는다.
- **2단계**: backbone(encoder, decoder)을 고정하고, class-balanced sampling 등을 통해 classifier $\theta_c$ 또는 추가 조정 파라미터 $\theta_a$만을 재학습(calibrate)한다.
$$\ell(D_s; \tilde{\theta}_e, \tilde{\theta}_d, \tilde{\theta}_c, \theta_a) = -\frac{1}{s} \sum_{i=1}^{s} w_{y_i} \log \frac{\exp(\eta_{y_i} + \delta_{y_i})}{\sum_{j=1}^{K} \exp(\eta_j + \delta_j)}$$

## 📊 Results

### 실험 설정

- **데이터셋**: ImageNet-LT, Places-LT, iNaturalist18 (테스트셋은 평가의 공정성을 위해 balanced 상태로 구성)
- **기준 모델**: CLIP (ViT-L14 backbone)
- **비교 대상**: Zero-shot, Linear probing, Full fine-tuning, Prompt tuning (CoOp)
- **평가 지표**: Overall Accuracy, Many/Medium/Few-shot Accuracy, P-R-F1 score

### 정량적 결과

실험 결과, 제안된 Decoder + Imbalanced Learning 조합이 모든 데이터셋에서 zero-shot 성능을 압도하였다.

- **ImageNet-LT**: Zero-shot(70.54%) 대비 평균 6.58% 향상되어 최고 79.44% (LADE Loss) 달성.
- **iNaturalist18**: Zero-shot(5.45%) 대비 **69.82%라는 극적인 향상**을 보이며 최고 73.24% (CRT) 달성.
- **Places-LT**: Zero-shot(37.69%) 대비 평균 6.17% 향상되어 최고 48.66% (MARC) 달성.

### 주요 분석 결과

1. **Decoder의 필수성**: Linear probing나 full fine-tuning보다 Decoder를 추가한 방식이 훨씬 우수했다. 특히 iNaturalist18에서 linear probing의 정확도는 10.03%에 불과했지만, decoder 기반 imbalanced 방법들은 70%대를 기록했다.
2. **Prompt Tuning과의 비교**: CoOp과 같은 prompt tuning은 클래스 수가 많아질 때 연산 비용(VRAM)이 급증하여 iNaturalist18과 같은 데이터셋에서는 적용이 불가능했으나, 제안 방법은 메모리 효율적으로 작동하며 성능 또한 더 높았다.
3. **Backbone 크기의 영향**: ViT-B16보다 ViT-L14를 사용할 때 성능이 유의미하게 향상되었다.
4. **Pre-training 데이터의 영향**: 400M 데이터로 학습된 CLIP보다 2B 데이터로 학습된 Laion-CLIP이 zero-shot에서는 유리할 수 있으나, downstream tuning을 거친 후에는 CLIP 기반 모델이 더 좋은 성능을 보이기도 했다. 이는 단순한 데이터 양의 증가보다 타겟 도메인에 맞는 tuning이 더 중요함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 통찰

본 연구는 VLM이 제공하는 강력한 일반적 특징(general features)을 유지하면서도, lightweight decoder를 통해 특정 imbalanced task에 필요한 세부 특징을 효과적으로 학습할 수 있음을 증명하였다. 특히, 매우 많은 클래스를 가진 iNaturalist18에서의 성능 향상은 VLM이 가진 잠재력이 적절한 imbalanced learning 알고리즘과 결합되었을 때 극대화될 수 있음을 보여준다. 또한, 하드웨어 제약(VRAM)을 고려한 실용적인 구조를 제안함으로써 실제 적용 가능성을 높였다.

### 한계 및 비판적 해석

1. **모델 다양성 부족**: CLIP과 Laion-CLIP 위주로 실험이 진행되어, 다른 VLM 아키텍처에서도 동일한 결과가 나올지는 미지수이다.
2. **텍스트 인코더 배제**: imbalanced learning 알고리즘과의 통합을 위해 텍스트 인코더를 사용하지 않았는데, 이는 VLM의 핵심인 '멀티모달 정렬' 능력을 충분히 활용하지 못한 설계일 수 있다. 향후 텍스트 모달리티를 불균형 학습에 어떻게 통합할지가 중요한 연구 과제가 될 것이다.
3. **ResNet 결과의 부진**: Decoder 구조가 ViT 기반으로 설계되어 ResNet backbone과의 결합 시 성능이 낮게 나타났다. 이는 제안된 decoder가 특정 아키텍처에 의존적임을 의미한다.

## 📌 TL;DR

본 논문은 VLM이 불균형 데이터셋(long-tailed)에서 취약하다는 문제를 지적하고, 이를 해결하기 위해 **frozen VLM encoder 뒤에 lightweight decoder를 추가하고 기존의 imbalanced learning 알고리즘(Balanced SoftMax, MARC 등)을 결합**하는 방법론을 제안한다. 이 방법은 특히 클래스 수가 많은 iNaturalist18 데이터셋에서 zero-shot 대비 정확도를 5%에서 73%까지 끌어올리는 괄목할 만한 성과를 거두었으며, 메모리 효율성 또한 확보하였다. 이 연구는 거대 모델의 pre-training 데이터 양보다 downstream에서의 적절한 불균형 최적화 기법이 성능 향상에 더 결정적일 수 있음을 시사하며, 향후 VLM 기반의 특수 목적 분류기 설계에 중요한 기준을 제공한다.
