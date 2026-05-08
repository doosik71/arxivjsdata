# Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences

Hyojin Bahng, Caroline Chan, Fredo Durand, Phillip Isola (2025)

## 🧩 Problem to Solve

본 논문은 시각 정보(Image)와 언어 정보(Text) 사이의 정렬(Alignment)을 학습하는 과정에서 발생하는 고비용의 데이터 수집 문제를 해결하고자 한다. 특히, 최근의 멀티모달 데이터가 매우 상세하고 복잡해짐에 따라 생성 모델들이 이미지에 없는 내용을 생성하는 환각(Hallucination) 현상이나, 텍스트의 세부 속성 및 관계를 제대로 반영하지 못하는 정렬 불량 문제가 빈번하게 발생하고 있다.

기존의 정렬 개선 방법론들은 주로 RLHF(Reinforcement Learning from Human Feedback)나 DPO(Direct Preference Optimization)와 같이 인간의 선호도(Human Preferences) 데이터나 GPT-4V와 같은 고비용의 AI 피드백에 의존한다. 그러나 이러한 방식은 데이터 수집 비용이 매우 높고 시간이 많이 소요된다는 한계가 있다. 또한, 기존의 정렬 측정 지표들은 주로 짧은 캡션에 치중되어 있어, 밀도 높고 상세한 묘사(Dense Alignment)를 평가하는 데 한계가 있다. 따라서 본 연구의 목표는 인간의 개입 없이도 확장 가능하며 효율적으로 이미지-텍스트 정렬을 학습할 수 있는 보상 모델(Reward Model)을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Cycle Consistency(사이클 일관성)**를 보상 신호로 활용하여 인간의 선호도 데이터를 대체하는 것이다. 즉, "이미지 $\rightarrow$ 텍스트 $\rightarrow$ 이미지" 또는 "텍스트 $\rightarrow$ 이미지 $\rightarrow$ 텍스트"의 경로를 거쳐 원래의 입력값으로 얼마나 잘 복원되는지를 측정하여, 복원 성능이 높을수록 정렬이 잘 된 결과물이라고 판단하는 직관에 기반한다.

주요 기여 사항은 다음과 같다:

1. **CyclePrefDB 구축**: 이미지-텍스트 및 텍스트-이미지 생성 작업에 대해 사이클 일관성을 기반으로 구축한 866K 개의 비교 쌍(Comparison Pairs)으로 구성된 대규모 선호도 데이터셋을 제안한다.
2. **CycleReward 모델**: CyclePrefDB를 통해 학습된 보상 모델로, 정렬 측정 지표 및 Best-of-N 샘플링의 검증기(Verifier)로 활용 가능하며, 인간의 선호도 없이도 높은 성능을 보인다.
3. **DPO 및 Diffusion DPO 적용**: 구축된 데이터셋을 사용하여 시각-언어 모델(VLM)과 확산 모델(Diffusion Model)의 성능을 인간의 감독 없이 향상시킬 수 있음을 입증하였다.

## 📎 Related Works

기존의 이미지-텍스트 정렬 지표는 크게 두 가지로 나뉜다:

- **Reference-based metrics**: BLEU, CIDEr, METEOR와 같이 정답(Ground Truth) 텍스트와의 언어적 유사성을 측정한다. 하지만 스타일이나 구문이 다를 경우 일반화 성능이 떨어지며, 정답 데이터가 필수적이어서 목적 함수(Objective function)로 사용하기 어렵다.
- **Reference-free metrics**: CLIPScore와 같이 임베딩 유사도를 측정하거나, HPSv2, PickScore와 같이 인간의 선호도를 학습한 보상 모델을 사용한다. 최근에는 GPT-4V와 같은 대규모 모델을 쿼리하여 평가하기도 하지만, 여전히 상세하고 긴 캡션을 평가하는 데는 한계가 있다.

본 연구와 유사하게 Image2Text2Image [32]나 DDPO [4]가 사이클 일관성 개념을 도입했으나, 이들은 추론 시마다 대형 모델을 통해 실시간으로 점수를 계산하므로 속도가 매우 느리고 미분 불가능하다는 단점이 있다. 반면, 본 논문은 이를 통해 학습된 가벼운 보상 모델을 제안함으로써 추론 속도와 미분 가능성을 확보하고 성능을 높였다.

## 🛠️ Methodology

### 1. Cycle Consistency as Preferences

본 논문은 인간의 주석 없이 선호도를 도출하기 위해 다음과 같은 사이클 일관성 점수를 정의한다.

**이미지 $\rightarrow$ 텍스트 정렬 평가:**
이미지 $x$에 대해 생성된 텍스트 $y$가 있을 때, 텍스트-이미지 모델 $G$를 통해 다시 이미지 공간으로 매핑하여 원래 이미지 $x$와의 유사도를 측정한다.
$$s(x \to y) := d_{img}(x, G(y))$$
여기서 $d_{img}$는 DreamSim을 사용하여 이미지 간의 시각적 유사도를 계산한다.

**텍스트 $\rightarrow$ 이미지 정렬 평가:**
텍스트 $y$에 대해 생성된 이미지 $x$가 있을 때, 이미지-텍스트 모델 $F$를 통해 다시 텍스트 공간으로 매핑하여 원래 텍스트 $y$와의 유사도를 측정한다.
$$s(y \to x) := d_{text}(y, F(x))$$
여기서 $d_{text}$는 SBERT를 사용하여 텍스트 간의 유사도를 계산한다.

이 점수를 기반으로 두 후보 $y_i, y_j$에 대해 다음과 같이 쌍별 선호도(Pairwise Preference)를 정의한다:
$$y_i \succ y_j \quad \text{if} \quad s(x \to y_i) > s(x \to y_j)$$

### 2. Dataset Generation (CyclePrefDB)

상세한 정렬(Dense Alignment)을 포착하기 위해 DCI(Densely Captioned Images) 데이터셋을 사용하였다.

- **Image-to-Text**: 11개의 다양한 이미지-텍스트 모델(BLIP2, LLaVA 시리즈, InternVL2 등)을 사용하여 다양한 품질의 캡션을 생성하고, LLaVA-1.5-13B를 역매핑 모델 $F$로 사용하여 점수를 매겼다.
- **Text-to-Image**: 4개의 텍스트-이미지 모델(SD 1.5, SDXL, SD3, FLUX)을 사용하여 이미지를 생성하고, SD3를 역매핑 모델 $G$로 사용하여 점수를 매겼다.

### 3. Reward Modeling

학습된 보상 모델 $r_\theta$는 주어진 쌍 $(x, y_i, y_j)$에서 선호되는 샘플에 더 높은 점수를 부여하도록 학습된다. 손실 함수는 다음과 같은 Bradley-Terry loss를 사용한다.

- **이미지-텍스트 손실**: $\mathcal{L}_{img} := -\mathbb{E}_{(x, y_i, y_j) \sim \mathcal{D}} [\log \sigma(r_\theta(x, y_i) - r_\theta(x, y_j))]$
- **텍스트-이미지 손실**: $\mathcal{L}_{text} := -\mathbb{E}_{(y, x_i, x_j) \sim \mathcal{D}} [\log \sigma(r_\theta(x_i, y) - r_\theta(x_j, y))]$
- **최종 손실**: $\mathcal{L} = \mathcal{L}_{text} + \lambda \mathcal{L}_{img} \quad (\lambda=1)$

모델 아키텍처는 BLIP을 백본으로 하며, ViT-L/16 인코더와 BERT-base 텍스트 인코더, 그리고 5계층의 MLP로 구성된다.

## 📊 Results

### 1. 인간 선호도와의 일치도

Table 2에 따르면, `CycleReward-Combo`는 인간의 선호도와 평균 65%의 높은 일치율을 보였다. 특히 GPT-4o는 텍스트-이미지 생성 작업에서 일치율이 24.8%까지 떨어지는 반면, 사이클 일관성 기반 모델은 두 작업 모두에서 일관된 성능을 보였다.

### 2. 정렬 측정 지표로서의 성능

DetailCaps-4870(상세 캡셔닝 평가)에서 `CycleReward-Combo`는 60.50%의 정확도를 기록하여, 인간 선호도로 학습된 ImageReward(50.70%)나 훨씬 거대한 모델인 VQAScore-11B(50.24%)를 크게 상회하였다.

### 3. Best-of-N (BoN) 샘플링

보상 모델을 사용하여 $N$개의 후보 중 최적의 결과물을 선택하는 BoN 전략을 적용한 결과, 특히 상세 캡셔닝 작업(LLaVA-W, DeCapBench)에서 기존 지표들보다 훨씬 큰 성능 향상을 이끌어냈다. 이는 `CycleReward`가 단순한 정확성을 넘어 세부 묘사의 풍부함과 정확성 사이의 균형을 잘 잡고 있음을 시사한다.

### 4. DPO 및 Diffusion DPO 결과

- **Image-to-Text**: Qwen-VL-Chat에 `CyclePrefDB-I2T`를 이용해 DPO를 적용한 결과, 상세 캡셔닝뿐만 아니라 지각, 추론, 환각 감소 등 일반적인 VLM 작업 성능이 전반적으로 향상되었다. 이는 캡셔닝 데이터만으로 학습했음에도 불구하고 일반적인 정렬 능력이 개선되었음을 보여준다.
- **Text-to-Image**: SD 1.5에 `CyclePrefDB-T2I`를 이용해 Diffusion DPO를 적용한 결과, 특히 복잡한 프롬프트에 대해 인간 선호도 데이터로 학습한 Pick-A-Pic 모델과 대등하거나 더 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 인간의 주석 없이도 사이클 일관성이라는 자기지도 학습(Self-supervised) 신호를 통해 대규모 선호도 데이터셋을 구축하고, 이를 통해 고성능의 보상 모델을 만들 수 있음을 증명하였다. 특히, 데이터 수집의 확장성(Scalability) 면에서 압도적인 이점이 있으며, 학습된 보상 모델이 원본 사이클 일관성 점수를 그대로 사용하는 것보다 더 정확한 판단을 내린다는 점(Distillation 효과)이 인상적이다.

### 한계 및 비판적 해석

1. **복원 모델에 대한 의존성**: 사이클 일관성의 품질은 사용된 $F$와 $G$ 모델의 성능에 전적으로 의존한다. 예를 들어, 텍스트-이미지 모델이 잘못된 이미지를 생성하면 잘못된 선호도 신호가 생성될 수 있다(Figure 5의 실패 사례).
2. **모델의 편향(Bias)**: 역매핑에 사용된 모델들의 내재적 편향이 그대로 보상 모델에 전이될 수 있다.
3. **토큰 제한**: SD3의 77토큰 제한으로 인해 매우 긴 텍스트에 대한 정렬을 평가하는 데 한계가 있다.
4. **심미적 요소의 부재**: 사이클 일관성은 정보의 보존(Information Preservation)에 집중하므로, 인간이 중요하게 생각하는 예술적 스타일이나 심미성(Aesthetics)은 반영하지 못한다.

## 📌 TL;DR

본 논문은 인간의 선호도 데이터 없이 **"이미지 $\leftrightarrow$ 텍스트"의 복원 가능성(Cycle Consistency)**을 통해 정렬 수준을 측정하고, 이를 기반으로 대규모 선호도 데이터셋(`CyclePrefDB`)과 보상 모델(`CycleReward`)을 제안한다. 이 모델은 상세 캡셔닝 및 텍스트-이미지 생성 작업에서 기존의 인간 피드백 기반 모델과 대등하거나 더 뛰어난 성능을 보였으며, DPO를 통해 모델의 전반적인 정렬 능력을 향상시켰다. 이 연구는 향후 인간의 개입 없이 멀티모달 모델을 정렬시키는 확장 가능한 프레임워크를 제공한다는 점에서 매우 중요하다.
