# Parrot: Multilingual Visual Instruction Tuning

Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, De-Chuan Zhan, Han-Jia Ye (2025)

## 🧩 Problem to Solve

본 논문은 Multimodal Large Language Models(MLLMs)에서 발생하는 **Multilingual Erosion(다국어 침식)** 문제를 해결하고자 한다. 기존의 MLLM들은 주로 영어 중심의 Supervised Fine-Tuning(SFT) 데이터셋을 사용하여 학습되는데, 이로 인해 학습이 진행될수록 영어 이외의 언어를 처리하는 능력이 저하되는 현상이 발생한다.

이 문제의 중요성은 기술적 접근성의 형평성과 관련이 있으며, 특히 입력 언어가 비영어권임에도 불구하고 모델이 영어로 답변하거나 비영어권 언어의 시각적 토큰 정렬(Visual Token Alignment)에 실패하는 문제가 핵심이다. 따라서 본 연구의 목표는 최소한의 다국어 데이터를 사용하여 MLLM의 다국어 능력을 향상시키고, 시각적 특징을 입력 언어에 맞게 동적으로 변환하는 메커니즘을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Textual Guidance(텍스트 가이드)**를 통해 시각적 토큰을 언어 수준에서 정렬하는 것이다. 구체적으로는 다음과 같은 설계 아이디어를 제안한다.

1. **PARROT 아키텍처**: 영어에 편향된(English-biased) 시각적 특징을 입력 텍스트의 언어 정보에 따라 해당 언어 전용 임베딩으로 변환하는 **Mixture-of-Experts(MoE)** 모듈을 도입하였다.
2. **MMMB 벤치마크 제안**: 기존 벤치마크의 한계(언어 수 부족, 평가 표준 미비 등)를 극복하기 위해 6개 언어, 15개 카테고리, 총 12,000개의 질문으로 구성된 **Massive Multilingual Multimodal Benchmark(MMMB)**를 구축하였다.

## 📎 Related Works

기존의 MLLM 연구들은 주로 LLaVA와 같이 Vision Encoder와 LLM을 MLP Projector나 Q-Former로 연결하는 방식을 사용한다. 다국어 능력을 확장하기 위한 기존 접근 방식은 다음과 같은 한계가 있다.

- **데이터 의존성**: 많은 모델이 방대한 양의 번역된 말뭉치(Translated Corpus)에 의존하며, 이는 데이터 확보 비용이 높고 번역 노이즈가 포함될 가능성이 크다.
- **언어의 제한**: 대부분의 벤치마크나 모델이 영어와 중국어 수준에 머물러 있어, 다양한 언어 가족(Language Family)에 대한 범용적인 평가와 성능 확보가 부족하다.
- **단순 번역 기반 방식**: 입력 질문을 영어로 번역해 처리하고 다시 대상 언어로 번역하는 방식은 문화적 맥락을 놓치고 특정 언어 성능이 올라가면 다른 언어가 떨어지는 '시소 효과(Seesaw effect)'를 유발한다.

PARROT은 이러한 대규모 데이터 의존성에서 벗어나, MoE 구조를 통해 적은 양의 데이터로도 언어별 시각 토큰 정렬을 수행함으로써 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

PARROT은 Vision Encoder를 통해 추출된 시각적 특징을 Projector를 통해 언어 임베딩 공간으로 투영한 뒤, 이를 MoE 모듈을 통해 입력 언어에 최적화된 표현으로 변환하여 LLM에 전달하는 구조를 가진다.

### 주요 구성 요소 및 작동 원리

#### 1. Textual Guidance 및 Cross-Attention

입력된 텍스트 임베딩 $H_t \in \mathbb{R}^{N \times C}$와 시각적 특징의 $[CLS]$ 토큰 $H_{cls}^v$ 사이의 cross-attention을 계산하여, 현재 입력 언어의 맥락이 반영된 특징 $H'_v$를 생성한다.

$$H'_v = \text{Attention}(Q, K, V) = \text{Softmax} \left( \frac{H_{cls}^v H_t^T}{\sqrt{C}} \right) H_t$$

여기서 $Q = H_{cls}^v$이며, $K$와 $V$는 모두 $H_t$와 동일하다. 이 과정은 시각적 특징을 텍스트 가이드에 따라 동적으로 조정하는 역할을 한다.

#### 2. Multilingual MoE (Mixture-of-Experts)

영어에 편향된 시각 토큰 $H_v$를 각 언어에 특화된 임베딩으로 변환하기 위해 MoE 모듈을 사용한다.

- **Router**: 앞서 계산된 $H'_v$를 입력으로 받아 어떤 전문가(Expert)를 활성화할지 확률 분포 $P$를 생성한다.
    $$P = \text{Softmax}(\text{Linear}(H'_v))$$
- **Experts**: 각 언어에 대응하는 MLP 형태의 전문가 집합 $E = [e_1, e_2, \dots, e_E]$가 존재하며, 선택된 전문가들이 $H_v$를 변환한다.
    $$\text{MoE}(H_v) = \sum_{i=1}^{k} P[i] \cdot E(H_v)_i$$

#### 3. MoE Reweighting

훈련 안정성을 높이고 원본 시각-의미 정보의 변동성을 줄이기 위해, 최종 시각 임베딩 $G_v$는 다음과 같이 잔차 연결(Residual Connection) 형태로 계산된다.

$$G_v = H_v + \alpha \text{MoE}(x)$$

여기서 $\alpha$는 trade-off 파라미터이다.

### 학습 절차 (Training Stage)

학습은 총 2단계로 진행된다.

- **Stage 1: Modality Alignment (양식 정렬)**: Vision Encoder와 LLM의 가중치를 동결하고 Projector만을 학습시킨다. 이 단계에서는 MoE 모듈을 우회(Bypass)하여 대규모 이미지-텍스트 쌍을 통해 기본적인 모달리티 간 간극을 메운다.
- **Stage 2: Instruction Tuning for Multilingual Alignment (다국어 정렬 지시 튜닝)**: Vision Encoder는 동결한 채 Projector, LLM, MoE 모듈을 학습시킨다. 이 단계에서 다국어 지시 튜닝 데이터를 사용하여 MoE가 언어별로 시각 토큰을 올바르게 변환하도록 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 새로 구축한 MMMB(6개 언어, 12,000개 샘플)와 기존 MMBench를 사용하였다.
- **비교 모델**: Qwen2-VL, LLaVA-OneVision, Idefics3 등 최신 MLLM들과 비교하였다.
- **평가 지표**: 4지선다형 질문에 대해 'Yes/No' 형식으로 변환하여 평가하는 **Circular Validation** 전략을 사용하여 무작위 추측으로 인한 성능 왜곡을 방지하였다.

### 주요 결과

- **정량적 성과**: PARROT-Qwen2-7B 모델은 MMMB와 MMBench의 6개 언어 중 4개 언어에서 SOTA(State-of-the-art) 성능을 달성하였으며, 특히 터키어(Turkish)와 아랍어(Arabic)에서 괄목할 만한 향상을 보였다.
- **범용 성능**: MME, ScienceQA, SEED-Bench 등 일반적인 다중모달 벤치마크에서도 경쟁력 있는 성능을 유지하여, 다국어 능력 향상이 일반 성능을 저하시키지 않음을 입증하였다.
- **데이터 효율성**: 다른 다국어 MLLM들에 비해 사용된 학습 데이터의 양이 1% 미만임에도 불구하고 유사하거나 더 뛰어난 성능을 보였다.
- **시각화 분석**: t-SNE 분석 결과, LLaVA는 사용된 Vision Encoder(OpenAI-CLIP vs Chinese-CLIP)에 따라 언어 정렬 성능이 크게 갈렸으나, PARROT은 어떤 인코더를 사용하더라도 텍스트 가이드를 통해 시각-텍스트 특징 간의 거리를 효과적으로 좁혔음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 단순히 다국어 데이터를 많이 붓는 방식이 아니라, **MoE를 이용한 언어별 시각 토큰 변환**이라는 구조적 접근을 통해 데이터 효율성을 극대화하였다. 특히 저자원 언어(Low-resource languages)에서 발생할 수 있는 '다국어의 저주(Curse of Multilingualism)'—즉, 저자원 언어를 학습시키려다 고자원 언어의 성능이 떨어지는 현상—를 MoE의 전문가 분리를 통해 효과적으로 회피하였다.

### 한계 및 비판적 해석

- **환각 현상 (Hallucinations)**: 정성적 결과 분석에서 특정 차량 모델(Xiaomi SU7 $\rightarrow$ Porsche Taycan)을 오인하는 등 여전히 시각적 환각 문제가 존재한다.
- **벤치마크 구축의 의존성**: MMMB 구축 과정에서 GPT-4를 사용한 번역 및 정제 과정을 거쳤다. 비록 전문가의 수동 교정(Manual Calibration)을 통해 품질을 높였으나, 기본적으로 GPT-4의 언어적 편향이 벤치마크에 반영되었을 가능성을 배제할 수 없다.
- **가정**: 본 연구는 시각적 특징이 기본적으로 영어에 편향되어 있다는 전제하에 MoE를 설계하였다. 만약 완전한 언어 중립적(Language-agnostic) 인코더가 등장한다면 MoE의 필요성이 낮아질 수 있다.

## 📌 TL;DR

PARROT은 MLLM에서 발생하는 다국어 성능 저하(Multilingual Erosion) 문제를 해결하기 위해, **텍스트 가이드 기반의 MoE 모듈**을 도입하여 영어 편향적인 시각 토큰을 입력 언어에 맞는 특정 임베딩으로 변환하는 모델이다. 또한 6개 언어를 포괄하는 **MMMB 벤치마크**를 통해 그 성능을 검증하였으며, 매우 적은 양의 다국어 데이터만으로도 SOTA 수준의 다국어 시각 이해 능력을 확보하였다. 이 연구는 향후 저자원 언어의 멀티모달 접근성을 높이는 데 중요한 기여를 할 것으로 보인다.
