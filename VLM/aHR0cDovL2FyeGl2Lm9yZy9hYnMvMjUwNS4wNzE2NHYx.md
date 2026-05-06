# EmoVLM-KD: Fusing Distilled Expertise with Vision-Language Models for Visual Emotion Analysis

SangEun Lee, Yubeen Lee, Eunil Park (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 이미지에서 지배적인 감정을 예측하는 Visual Emotion Analysis (VEA)의 성능 향상이다. VEA는 객관적인 정보만을 인식하는 일반적인 컴퓨터 비전 작업과 달리, 주관적이고 모호한 감정적 정보를 캡처해야 하므로 매우 도전적인 과제이다.

최근 Vision-Language Model (VLM)의 등장으로 성능 향상이 있었으나, 저자들은 Instruction-tuned VLM과 기존의 Vision Model(CNN 또는 ViT 기반)이 서로 상보적인 강점을 가지고 있다는 점을 발견하였다. VLM은 사전 학습된 방대한 언어적 지식을 활용하여 감정을 분석하는 반면, 기존의 Vision Model은 순수하게 시각적 특징에 의존한다. 이로 인해 두 모델은 동일한 이미지에 대해 서로 다른 예측 결과를 내놓는 경우가 많다.

따라서 본 논문의 목표는 두 모델의 강점을 결합하여 VEA 성능을 극대화하는 것이다. 하지만 두 거대 모델을 동시에 사용하는 Ensemble 방식은 계산 비용이 너무 높기 때문에, 기존 Vision Model의 전문 지식을 VLM으로 효율적으로 전이하는 가벼운 프레임워크를 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Knowledge Distillation (KD) 프레임워크를 사용하여 기존 Vision Model의 예측 패턴을 VLM의 시각 인코더(Visual Encoder)에 부착된 매우 가벼운 모듈로 전이하는 것이다.

단순히 두 모델을 합치는 것이 아니라, VLM을 먼저 감정 특화 데이터로 Instruction Tuning한 뒤, 기존 Vision Model(Teacher)의 지식을 작은 증류 모듈(Student)에 학습시켜 VLM의 파라미터 증가를 최소화하면서도(약 0.003% 증가) 시각적 전문성을 확보하는 설계 구조를 제안하였다. 최종적으로는 Gate Module을 통해 VLM의 출력과 증류 모듈의 출력을 동적으로 조절하여 최적의 결과를 도출한다.

## 📎 Related Works

VEA 분야의 기존 연구들은 다음과 같은 단계로 발전해 왔다.

- **초기 연구:** 색상, 질감, 모양과 같은 Low-level visual features나 handcrafted features에 의존하였으나, 복잡한 장면의 세부적인 감정 형성 과정을 설명하는 데 한계가 있었다.
- **중기 연구:** Multiple Instance Learning이나 Probabilistic Latent Semantic Analysis 등을 도입하여 Mid-level representation을 활용해 지역적 감정을 분석하려는 시도가 있었다.
- **딥러닝 기반 연구:** CNN 및 Transformer 기반 모델들이 등장하며 Global feature 학습이 가능해졌으며, Attention Mechanism을 통해 감정적 영역을 식별하는 방법(예: OSANet, MASANet) 등이 제안되었다.
- **VLM 기반 연구:** 최근에는 EmoVIT와 같이 Visual Instruction Tuning을 통해 VLM의 성능을 높이려는 연구가 진행되었다.

본 연구는 기존 VLM 기반 접근 방식이 언어적 지식에 치우쳐 시각적 특징 추출 능력이 부족할 수 있다는 점에 주목하며, 도메인 특화 Vision Model의 지식을 증류(Distillation)하여 VLM과 결합함으로써 기존의 단일 모델 접근 방식 및 단순 VLM 튜닝 방식과 차별점을 둔다.

## 🛠️ Methodology

EmoVLM-KD의 전체 파이프라인은 다음의 3단계로 구성된다.

### 1. Instruction Tuning VLM

먼저 VLM(본 논문에서는 Qwen2-VL-7b 사용)이 이미지의 감정을 인간처럼 해석할 수 있도록 Instruction Tuning을 수행한다. GPT-4를 이용해 생성한 세 가지 유형의 데이터셋을 사용한다.

- **Categorical:** 주어진 선택지 중 가장 적절한 감정 카테고리를 선택하게 하는 데이터.
- **Conversation:** 이미지의 주요 요소에 대해 묻고 답하는 상호작용 데이터.
- **Reasoning:** 이미지가 특정 감정을 전달하는 추론 과정을 설명하게 하는 데이터.
학습 시에는 QLoRA 기법을 적용하여 언어 모델의 Query, Key, Value projection layer만을 튜닝함으로써 기존 지식을 보존하며 감정 특화 작업을 학습시킨다.

### 2. Knowledge Distillation (KD)

계산 효율성을 위해 VLM의 Visual Encoder 상단에 가벼운 Projection Module을 추가하고, 이를 기존의 Vision Model(Teacher, 여기서는 ViT)로부터 학습시킨다. 이때 VLM의 Visual Encoder와 LLM 부분은 모두 Frozen 상태로 유지하며, 오직 증류 모듈만 학습한다.

손실 함수는 Teacher 모델의 확률 분포를 따르게 하는 $\text{KL Divergence Loss}$($L_{KD}$)와 정답 레이블을 맞추게 하는 $\text{Cross-Entropy Loss}$($L_{CE}$)의 가중합으로 정의된다.

$$L_{KD} = \tau^2 \sum_{i} p_{t}^{\tau}(i) \log \frac{p_{t}^{\tau}(i)}{p_{s}^{\tau}(i)}$$

$$L_{CE} = -\sum_{i} y_{i} \log p_{s}(i)$$

$$L_{total} = \alpha L_{KD} + (1-\alpha) L_{CE}$$

여기서 $\tau$는 확률 분포를 부드럽게 만드는 Temperature 파라미터이며, $\alpha$는 두 손실의 균형을 조절하는 가중치(본 논문에서는 0.5 사용)이다.

### 3. Gate Module

최종 단계에서는 Instruction-tuned VLM의 예측 결과와 증류 모듈의 예측 결과를 통합하는 Gate Module을 학습시킨다.

- VLM의 출력은 원-핫 인코딩(one-hot encoded) 벡터로 변환된다.
- 증류 모듈의 출력은 감정 카테고리에 대한 확률 분포이다.
두 벡터를 Concatenation 하여 선형 레이어(Linear Layer)에 통과시켜 최종 로짓(logits) $\hat{y}$를 생성한다.

$$\hat{y} = W \cdot h + b$$

여기서 $h$는 결합된 벡터, $W$는 가중치 행렬, $b$는 편향(bias)이다. 이 과정에서도 VLM과 증류 모듈은 Frozen 상태이며, 오직 Gate Module만 학습된다.

## 📊 Results

### 실험 설정

- **데이터셋:** EmoSet, FI (8개 감정), Emotion6 (6개 감정), Flickr, Instagram (이진 감정) 총 5개의 벤치마크를 사용하였다.
- **비교 대상:** 기존 Vision Models (VGG, ResNet, ViT), VEA 전용 모델 (WSCNet, MDAN, PDANet 등), Zero-shot VLMs (InstructBLIP, LLaVA-Next 등), Instruction-tuned VLM (EmoVIT).

### 주요 결과

EmoVLM-KD는 대부분의 데이터셋에서 State-of-the-art (SOTA) 성능을 달성하였다.

- **FI 데이터셋:** ViT(68.86%)와 MDAN(76.42%)을 제치고 79.51%의 정확도를 기록하였다.
- **Emotion6 데이터셋:** 기존 VEA 전용 모델들의 최고 성능(61.66%)을 크게 상회하는 73.91%를 달성하였다.
- **이진 분류(Flickr, Instagram):** 각각 88.90%, 89.59%의 높은 정확도를 보이며 강건함을 입증하였다.

### Ablation Study

- **증류 모듈의 효과:** 단순 VLM(71.19%) 대비 EmoVLM-KD(79.51%)는 약 8%의 성능 향상을 보였으며, 이는 전체 파라미터의 0.003%라는 매우 적은 비용으로 달성되었다.
- **깊이 분석:** 증류 모듈의 층이 깊어질수록 오히려 성능이 하락하는 경향을 보였으며, 단일 hidden layer(1024 dim) 구성이 가장 우수하였다.
- **하이퍼파라미터:** $\alpha=0.5$일 때 KL Loss와 정확도 사이의 최적의 트레이드-오프가 이루어짐을 확인하였다.
- **Gate 구조:** 다양한 게이팅 방법(MoE, Bilinear pooling 등) 중 'Concat & Linear' 방식이 가장 높은 성능을 보였다.

## 🧠 Insights & Discussion

본 연구는 VLM의 광범위한 추론 능력과 도메인 특화 Vision Model의 정교한 시각적 특징 추출 능력이 서로 보완적이라는 점을 실험적으로 증명하였다. 특히 흥미로운 점은 **증류된 모듈(Student)의 성능이 Teacher 모델인 ViT보다 높게 나타났다**는 점이다. 이는 증류 과정에서 Teacher의 예측 패턴을 배우는 동시에 Ground-truth 레이블을 통한 직접적인 감독(Direct Supervision)을 함께 받았기 때문으로 분석된다.

또한, 정성적 분석(Qualitative Examples)을 통해 VLM이 틀리고 증류 모듈이 맞히는 경우, 혹은 그 반대의 경우에 Gate Module이 적절하게 가중치를 조절하여 정답을 선택하는 과정을 확인하였다. 이는 단순히 모델을 결합한 것이 아니라, 각 모델의 신뢰도를 동적으로 판단하는 메커니즘이 효과적으로 작동하고 있음을 시사한다.

한계점으로는 특정 데이터셋(Emoset)에서는 최고 성능에 도달하지 못했다는 점이 있으나, 전반적인 데이터셋에서 일관된 우위를 보였다는 점에서 범용성이 높다고 평가할 수 있다.

## 📌 TL;DR

본 논문은 VLM과 기존 Vision Model의 상보적 특성을 활용하여, Vision Model의 전문성을 가벼운 모듈 형태로 VLM에 주입하는 **EmoVLM-KD**를 제안하였다. Knowledge Distillation과 Gate Module을 통해 계산 비용을 최소화하면서도(파라미터 0.003% 추가) 5개의 VEA 벤치마크에서 SOTA 성능을 달성하였다. 이 연구는 거대 모델의 효율적인 도메인 특화 전이 학습 방법론을 제시했다는 점에서 향후 다양한 멀티모달 감정 분석 연구에 중요한 기초가 될 것으로 보인다.
