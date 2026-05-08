# Multimodal LLMs as Customized Reward Models for Text-to-Image Generation

Shijie Zhou, Ruiyi Zhang, Huaisheng Zhu, Branislav Kveton, Yufan Zhou, Jiuxiang Gu, Jian Chen, Changyou Chen (2025)

## 🧩 Problem to Solve

텍스트-이미지 생성(Text-to-Image, T2I) 모델의 성능을 평가하고 인간의 선호도에 맞게 정렬(Alignment)하는 것은 매우 중요하다. 그러나 기존의 보상 모델(Reward Model)들은 다음과 같은 한계점을 가지고 있다.

1. **CLIP 기반 모델의 한계**: CLIPScore, HPSv2, ImageReward 등은 CLIP을 미세 조정하여 사용하지만, CLIP이 텍스트를 단어들의 집합(Bag-of-words)으로 처리하는 경향이 있어 복잡한 이미지-텍스트 관계를 추론하는 능력이 부족하고 일반화 성능이 떨어진다.
2. **MLLM 기반 VQA 방식의 비효율성**: 최근의 Multimodal Large Language Models(MLLMs)를 이용한 평가 방식은 상세한 시스템 프롬프트를 통해 답변을 유도하는 VQA(Visual Question Answering) 방식을 사용한다. 이는 추론 시 시간이 많이 소요되며, 훈련 과정 또한 복잡하고 비효율적이다.
3. **토큰 확률 기반 방식의 정밀도 부족**: 특정 토큰(예: "yes", "good")의 생성 확률을 점수로 사용하는 방식은 이산적인(discrete) 레이블에 의존하므로, 미세한 품질 차이를 가진 샘플 간의 상대적 선호도를 학습하기 어렵고 편향된 점수를 생성할 가능성이 크다.

본 논문의 목표는 복잡한 프롬프트 없이도 효율적으로 작동하며, 다양한 관점(정렬, 충실도, 안전성 등)에서 인간의 선호도를 정확하게 반영하는 다목적 MLLM 기반 보상 모델인 **LLaVA-Reward**를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 MLLM의 텍스트 생성 능력을 포기하는 대신, **MLLM의 은닉 상태(Hidden States)를 직접 활용하여 보상 값을 예측**하도록 설계한 것이다.

- **효율적인 보상 구조**: 길고 복잡한 시스템 프롬프트 없이 이미지-텍스트 쌍을 입력하면 MLLM의 마지막 층 은닉 상태를 통해 즉각적으로 보상 점수를 산출한다.
- **Skip-connection Cross Attention (SkipCA) 도입**: 디코더 전용(Decoder-only) MLLM 구조에서는 인과적 어텐션(Causal Attention)으로 인해 시각적 토큰이 이후의 텍스트 토큰에 영향을 받지 못하는 문제가 있다. 이를 해결하기 위해 초기 단계의 시각적 특징($e_v$)과 깊은 층의 은닉 표현($e_h$)을 직접 연결하는 SkipCA 모듈을 설계하여 시각-텍스트 간의 양방향 상호작용 및 추론 능력을 강화하였다.
- **다양한 관점의 맞춤형 평가**: LoRA(Low-Rank Adaptation) 어댑터를 활용하여 텍스트-이미지 정렬(Alignment), 충실도/아티팩트(Fidelity/Artifact), 안전성(Safety) 등 각 평가 관점별로 최적화된 어댑터를 교체하며 사용할 수 있도록 구현하였다.

## 📎 Related Works

논문은 기존의 T2I 평가 방식을 세 가지 범주로 구분하여 설명한다.

1. **CLIP 기반 점수 방식**: CLIP의 유사도를 이용하며 구현이 간단하지만, 복잡한 시맨틱 관계를 파악하지 못하는 한계가 있다.
2. **VQA 기반 평가 방식**: 상세한 지침을 통해 MLLM이 텍스트 답변을 생성하게 한다. 설명 가능성은 높으나 계산 비용이 매우 크며 RLHF(Reinforcement Learning from Human Feedback) 훈련에 적용하기에는 실시간성이 떨어진다.
3. **특수 토큰 확률 기반 방식**: 특정 판단 토큰의 확률값을 점수로 활용한다. VQA 방식보다 효율적이지만, 이산적인 레이블로 인해 연속적인 선호도 학습이 어렵다.

**LLaVA-Reward와의 차별점**: LLaVA-Reward는 토큰 생성이나 확률 예측이 아닌, **연속적인 값의 보상 모델링**을 위해 Bradley-Terry 랭킹 손실 함수를 사용하며, SkipCA를 통해 시각적 인지 능력을 극대화하여 기존 방식보다 정밀하고 효율적인 평가가 가능하다.

## 🛠️ Methodology

### 1. 전체 아키텍처

LLaVA-Reward는 **Phi-3.5-vision 4.2B**를 기본 모델로 사용한다. 전체 구조는 다음과 같다.

- **입력**: 이미지 $i$와 텍스트 $t$의 쌍.
- **백본**: MLLM (Phi-3.5-vision) + LoRA 어댑터.
- **보상 헤드 (Reward Head)**: SkipCA 모듈 $\rightarrow$ 선형 투영 층($g$).

### 2. Skip-connection Cross Attention (SkipCA)

디코더 전용 모델의 단방향 흐름을 극복하기 위해 도입되었다.

- **Query ($Q$)**: MLLM의 깊은 층(보통 마지막 층)의 EOS(End-of-Sentence) 토큰 은닉 상태 $e_h$.
- **Key ($K$) & Value ($V$)**: 시각적 프로젝터(Visual Projector) 직후의 투영된 시각적 토큰 $e_v$.
- **작동 원리**: 시각적 정보가 깊은 층으로 갈수록 희석되는 경향이 있으므로, 가장 순수한 시각 특징 $e_v$를 추출하여 마지막 층의 텍스트 표현 $e_h$와 교차 어텐션을 수행한다.
- **수식**:
$$r_{\theta}(i, t) = g(f_{SCA}(e_h, e_v))$$
여기서 $f_{SCA}$는 Cross-Attention 연산이며, $g$는 최종적으로 스칼라 보상 값이나 벡터 값을 생성하는 선형 층이다.

### 3. 학습 목표 및 손실 함수

데이터의 성격에 따라 두 가지 손실 함수를 사용한다.

**A. 쌍 기반 선호도 데이터 (Paired Preference Data)**
선택된 이미지 $i_c$와 거부된 이미지 $i_r$이 주어졌을 때, Bradley-Terry (BT) 모델을 기반으로 한 랭킹 손실을 사용한다.
$$\mathcal{L}_{rank} = \mathbb{E}_{(i_c, i_r, t) \sim D^p} [-\log \sigma (\frac{s_{\theta}^p(i_c, t) - s_{\theta}^p(i_r, t)}{T})]$$
여기서 $s_{\theta}^p$는 모델이 예측한 보상 점수이며, $T$는 온도 파라미터이다.

더 정밀한 모델링을 위해 **GPM(General Preference Model)** 방식을 적용하여 보상을 벡터 공간으로 확장하고 내적(Inner product)을 통해 선호도를 계산하는 방식도 제안하였다.

**B. 비쌍 기반/이진 데이터 (Unpaired/Binary Data)**
안전성 평가와 같이 안전/불안전의 이진 레이블이 있는 경우 Cross-Entropy(CE) 손실을 사용한다.
$$\mathcal{L}_{CE} = -\mathbb{E}_{(i_c, i_r, t) \sim D^p} [\log \sigma(s_{\theta}^p(i_c, t)) + \log(1 - \sigma(s_{\theta}^p(i_r, t)))]$$

### 4. 학습 절차

- 시각적 인코더와 내부 언어 모델의 대부분은 동결(Freeze)한다.
- **시각적 프로젝터, SkipCA 모듈, LoRA 어댑터**만 학습시킨다 (전체 파라미터의 약 8%만 업데이트).

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MJ-Bench (정렬, 안전성, 아티팩트 평가), TIFA 160 (정렬), Unsafe Diffusion & SMID (안전성).
- **비교 대상**: CLIP 기반 모델(HPSv2, ImageReward), 오픈소스 MLLM, 폐쇄형 MLLM(GPT-4o, Gemini Ultra), MLLM 기반 보상 모델(VQAScore, LLaVA-score).

### 2. 주요 결과

- **자동 평가 성능**: MJ-Bench에서 LLaVA-Reward는 정렬, 안전성, 아티팩트의 모든 지표에서 기존 CLIP 기반 모델 및 일반 MLLM보다 우수한 성능을 보였다. 특히 폐쇄형 모델인 GPT-4o와 경쟁 가능한 수준의 정확도를 달성하였다.
- **추론 효율성**: 표 8에 따르면, VQA 기반 방식(Evalalign 등)이 평가당 4~7초가 소요되는 반면, LLaVA-Reward는 **0.35초** 만에 평가를 완료하여 매우 높은 효율성을 입증하였다.
- **T2I 생성 품질 향상 (Inference-time Scaling)**: LLaVA-Reward를 FK steering(Feynman Kac steering)의 보상 모델로 사용했을 때, ImageReward나 CLIPScore를 사용했을 때보다 텍스트 정렬도가 훨씬 높은 이미지가 생성됨을 확인하였다 (예: "purple train" 등의 복잡한 속성 반영 능력이 뛰어남).

### 3. Ablation Study 결과

- **SkipCA의 효과**: SkipCA를 제거하고 단순 MLP를 사용했을 때 성능이 하락하였으며, 특히 안전성 평가에서 그 차이가 두드러졌다. 이는 깊은 층에서 희석된 시각 정보를 다시 복구하는 것이 중요함을 의미한다.
- **부정 샘플(Negative Samples)의 중요성**: 하드 부정 샘플 70k개를 추가하여 학습했을 때 정렬 정확도가 약 1.6% 상승하여, 정밀한 구분이 가능해졌다.
- **은닉 상태 추출 위치**: 마지막 층(Layer 32)의 은닉 상태를 사용하는 것이 중간 층을 사용하는 것보다 성능이 좋았다.

## 🧠 Insights & Discussion

**강점**

- 본 연구는 MLLM을 단순히 "판단 도구(Judge)"로 쓰는 것을 넘어, 효율적인 "보상 함수(Reward Function)"로 변환시켰다.
- 특히 SkipCA 모듈은 Decoder-only 모델이 가진 구조적 한계(시각 토큰의 정보 희석)를 효과적으로 해결하는 실용적인 방법론을 제시하였다.
- LoRA를 통해 단일 모델로 여러 평가 관점을 유연하게 처리할 수 있다는 점이 매우 효율적이다.

**한계 및 논의사항**

- **데이터 의존성**: 보상 모델의 성능이 학습 데이터의 품질에 크게 의존하며, 고품질의 인간 선호도 데이터가 부족할 경우 Reward Hacking(보상 값만 높이고 실제 품질은 낮은 현상)이 발생할 위험이 있다.
- **가정**: 본 논문은 Phi-3.5-vision을 기본 모델로 사용했으나, 다른 MLLM에서도 동일한 효과가 나타날지에 대한 광범위한 검증은 추가로 필요하다.
- **비판적 해석**: 추론 속도는 매우 빠르지만, VQA 방식이 제공하는 "이유(Rationale)"를 제공하지 못한다는 점은 디버깅 관점에서 아쉬운 부분이다. 하지만 보상 모델로서의 목적(최적화 및 빠른 평가)에는 더 부합하는 설계라고 판단된다.

## 📌 TL;DR

LLaVA-Reward는 MLLM의 은닉 상태를 직접 활용하고 **SkipCA(Skip-connection Cross Attention)** 모듈을 통해 시각-텍스트 추론 능력을 강화한 효율적인 T2I 보상 모델이다. 복잡한 프롬프트 없이도 정렬, 안전성, 충실도를 정밀하게 평가할 수 있으며, 추론 속도가 매우 빨라 실시간 평가 및 확산 모델의 추론 시점 스케일링(Inference-time scaling)에 적용하여 이미지 품질을 획기적으로 높일 수 있다. 이 연구는 향후 인간의 선호도에 완전히 정렬된 T2I 생성 모델을 구축하는 데 핵심적인 평가 도구로 활용될 가능성이 크다.
