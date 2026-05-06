# LLaVA-Gemma: Accelerating Multimodal Foundation Models with a Compact Language Model

Musashi Hinck, Matthew L. Olson, David Cobbley, Shao-Yen Tseng, Vasudev Lal (2024)

## 🧩 Problem to Solve

본 연구는 거대 다중모드 모델(Large Multimodal Models, LMMs)의 높은 계산 비용 문제를 해결하기 위해, 작지만 강력한 언어 모델을 기반으로 한 소규모 다중모드 기초 모델(Multimodal Foundation Models, MMFM)의 가능성을 탐색한다. 특히, 모델의 파라미터 크기, 시각적 인코더(Vision Encoder)의 성능, 그리고 학습 단계의 구성이 모델의 전체적인 성능과 효율성에 어떠한 영향을 미치는지 분석하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

- **LLaVA-Gemma 모델 제품군 제안**: 최근 공개된 Gemma-2B 및 Gemma-7B 언어 모델을 기반으로 하여 효율적인 다중모드 상호작용이 가능한 LLaVA-Gemma 모델들을 구축하였다.
- **설계 요소에 대한 광범위한 절제 연구(Ablation Study)**: 커넥터(Connector)의 사전 학습 여부, 시각적 백본(Vision Backbone)의 종류, 언어 모델의 크기라는 세 가지 핵심 설계 변수가 성능에 미치는 영향을 심층적으로 분석하였다.
- **시각적 주의 집중 분석**: Relevancy Map을 통해 모델 크기에 따른 시각적 토큰에 대한 Attention 패턴의 차이를 시각화하여 분석하였다.
- **학습 및 추론 효율성 측정**: 모델 크기에 따른 학습 시간 및 추론 속도의 트레이드오프를 정량적으로 제시하였다.

## 📎 Related Works

본 연구는 시각적 명령어 튜닝(Visual Instruction Tuning)의 대표적인 프레임워크인 LLaVA를 기반으로 한다. 기존의 LLaVA는 주로 Llama-2와 같은 대형 모델을 사용하였으며, 최근에는 LLaVA-Phi와 같이 더 작은 언어 모델을 사용하는 시도들이 있었다.

기존 연구들 중 일부(예: Prismatic VLMs)는 초기 사전 학습 단계를 건너뛰는 것이 다운스트림 성능을 향상시킨다고 주장하였으나, 본 논문은 LLaVA-Gemma 실험을 통해 이러한 가설을 검증하고 상충하는 결과를 제시함으로써 기존 접근 방식과의 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

LLaVA-Gemma는 전형적인 LLaVA 구조를 따른다. 시스템은 크게 세 가지 구성 요소로 이루어져 있다.

1. **Vision Encoder**: 이미지를 특징 벡터로 변환하는 인코더로, CLIP과 DINOv2 두 가지 옵션을 사용한다.
2. **MLP Connector**: 시각적 특징 공간을 언어 모델의 임베딩 공간으로 투영(Projection)하는 다층 퍼셉트론(Multi-Layer Perceptron)이다.
3. **Language Model (LM)**: 텍스트 생성 및 추론을 담당하는 백본으로, Gemma-2B 또는 Gemma-7B를 사용한다. Gemma 모델은 256k라는 매우 큰 토큰 세트를 가지고 있어, 더 다양한 임베딩 공간을 활용할 수 있는 특징이 있다.

### 학습 절차

학습은 총 2단계로 진행된다.

- **Stage 1 (Pretraining)**: 시각 인코더와 언어 모델을 동결(Freeze)시킨 상태에서 MLP 커넥터만을 학습시킨다. CC3M에서 필터링된 595k 개의 샘플 데이터를 사용하여 시각적 특징과 텍스트 임베딩 간의 정렬(Alignment)을 수행한다.
- **Stage 2 (Instruction Tuning)**: 언어 모델과 커넥터를 함께 미세 조정(Joint Fine-tuning)한다. GQA, TextCaps 및 합성 데이터(Synthetic data)를 포함한 665k 개의 다중모드 명령어 튜닝 데이터셋을 사용한다.

## 📊 Results

### 실험 설정

- **벤치마크**: GQA, MME, MM-Vet, POPE, VQAv2, MMVP, ScienceQA 등 7가지 평가 지표를 사용하였다.
- **비교 대상**: LLaVA-Phi (2B), LLaVA-v1.5 (Llama-2-7B)의 보고된 성능과 비교하였다.

### 주요 결과

1. **시각 인코더의 영향**:
   - Gemma-2B 모델의 경우, CLIP보다 DINOv2를 사용했을 때 대부분의 벤치마크에서 성능이 향상되었다.
   - 반면, Gemma-7B 모델에서는 DINOv2의 효과가 불분명하거나 일부 지표(MM-Vet, POPE 등)에서 오히려 성능이 하락하는 경향을 보였다.
2. **사전 학습(Pretraining)의 효과**:
   - 커넥터의 사전 학습을 생략했을 때 거의 모든 경우에서 성능이 감소하였다. 이는 사전 학습 생략이 유리하다는 일부 기존 연구의 결과와 상충한다.
3. **언어 모델 크기의 영향**:
   - 모델 크기가 커진다고 해서 모든 작업의 성능이 향상되지는 않았다. 특히 ScienceQA와 같이 일반 지식이 많이 필요한 작업에서는 7B 모델이 우수했으나, GQA, MME, POPE 등에서는 오히려 2B 모델이나 다른 베이스라인보다 낮은 성능을 보이기도 했다.
4. **효율성**:
   - Intel Gaudi 2 가속기 기준, Gemma-2B 모델의 학습 시간은 4시간이었으나, Gemma-7B 모델은 16시간이 소요되었다. 학습 및 추론 속도 면에서 7B 모델은 2B 모델 대비 약 $0.25\times$ 수준의 속도를 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 단순히 모델을 제안하는 것에 그치지 않고, Relevancy Map을 통해 모델의 내부 동작을 분석하였다. 실험 결과, Gemma-2B 모델은 시각적 입력에 대한 Attention이 분산되어 있어 이미지 해석에 실패하는 반면, Gemma-7B 모델은 객체 간의 경계(예: 오리와 물의 경계)에 더 집중하는 경향을 보였다. 이는 더 강력한 LLM 백본이 시각적 토큰에 대한 Attention 능력을 향상시킬 수 있음을 시사한다.

### 한계 및 비판적 해석

LLaVA-Gemma는 전반적으로 준수한 성능을 보였으나, 동일 규모의 SOTA(State-of-the-art) 모델들을 뛰어넘지는 못했다. 특히 7B 모델이 특정 벤치마크에서 2B 모델보다 낮은 성능을 보이는 현상은 단순한 파라미터 수의 증가가 다중모드 이해 능력의 선형적 증가로 이어지지 않음을 보여준다. 이는 모델의 크기보다 데이터의 질, 혹은 시각 인코더와 언어 모델 간의 정렬(Alignment) 최적화가 더 중요할 수 있음을 의미한다.

## 📌 TL;DR

본 연구는 Gemma-2B/7B를 기반으로 한 소규모 다중모드 모델인 LLaVA-Gemma를 제안하고, 시각 인코더 종류와 사전 학습 여부, 모델 크기가 성능에 미치는 영향을 분석하였다. 분석 결과, 2B 모델에서는 DINOv2 인코더가 효과적이었으며, 커넥터의 사전 학습이 필수적임을 확인하였다. 또한, 모델 크기가 커질수록 일반 지식 능력과 시각적 Attention 집중력은 향상되지만, 모든 VQA 작업에서 성능이 비례하여 상승하지는 않는다는 점을 밝혀냈다. 이 연구는 향후 효율적인 소형 VLM 설계 및 최적화를 위한 기초 자료로 활용될 가능성이 높다.
