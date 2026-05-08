# NVLM: Open Frontier-Class Multimodal LLMs

Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping (2024)

## 🧩 Problem to Solve

본 논문은 기존의 open-access Multimodal Large Language Models (MLLMs)가 가진 세 가지 주요 한계점을 해결하고자 한다.

첫째, 모델 아키텍처의 비교 분석 부족이다. 현재 MLLM은 Decoder-only 구조(예: LLaVA)와 Cross-attention 기반 구조(예: Flamingo)로 나뉘어 있으나, 동일한 LLM backbone과 데이터셋을 사용하여 두 구조를 공정하게(apples-to-apples) 비교한 연구가 부족하며, 특히 proprietary 모델들의 구조는 공개되지 않았다.

둘째, 고해상도 이미지 처리와 추론 능력 간의 트레이드-오프 문제이다. Dynamic high-resolution 메커니즘은 OCR 관련 작업의 성능을 높이지만, 때때로 MMMU와 같은 복잡한 시각적 추론 작업의 정확도를 떨어뜨리는 경향이 있다.

셋째, Multimodal 학습 후 발생하는 텍스트 전용(text-only) 성능 저하 문제이다. 많은 open-access MLLM들이 시각-언어 작업 성능은 높였으나, 정작 LLM 본연의 텍스트 처리 능력이 하락하는 'catastrophic forgetting' 현상을 겪는다.

따라서 본 논문의 목표는 시각-언어 작업에서 최첨단(SOTA) 성능을 달성함과 동시에, 텍스트 전용 성능을 유지하거나 오히려 향상시키는 'Production-grade multimodality'를 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 요약할 수 있다.

1. **세 가지 아키텍처 제안 및 비교**: Decoder-only (NVLM-D), Cross-attention (NVLM-X), 그리고 이 둘의 장점을 결합한 Hybrid (NVLM-H) 아키텍처를 제안하고, 동일 조건에서 각 구조의 효율성과 성능을 분석하였다.
2. **1-D Tile-tagging 설계**: 고해상도 이미지의 Dynamic tiling 과정에서 각 타일의 위치 정보를 텍스트 기반의 1-D 태그($<tile\_k>$)로 삽입하여, OCR 및 시각적 추론 성능을 동시에 강화하였다.
3. **데이터 큐레이션 전략**: 데이터의 규모보다 품질과 작업의 다양성이 더 중요하다는 점을 입증하였으며, 정교하게 큐레이션된 pretraining 및 SFT 데이터셋을 구축하였다.
4. **텍스트 성능 보존 및 향상**: 고품질의 텍스트 전용 SFT 데이터를 multimodal 학습 과정에 통합함으로써, 텍스트 전용 성능의 저하를 막고 오히려 수학 및 코딩 능력을 향상시키는 결과를 얻었다.

## 📎 Related Works

기존 MLLM은 크게 두 가지 설계 방향을 가진다.

* **Decoder-only MLLMs**: LLaVA, InternVL 등이 대표적이며, 이미지 토큰을 projector(예: MLP)를 통해 텍스트 임베딩 공간으로 투영하여 LLM의 self-attention 층에서 함께 처리한다. 구조가 단순하고 통합적인 추론이 가능하지만, 고해상도 이미지 처리 시 시퀀스 길이가 매우 길어져 계산 효율성이 떨어진다.
* **Cross-attention-based MLLMs**: Flamingo와 그 파생 모델들이 해당하며, LLM의 self-attention 층 사이에 gated cross-attention 층을 삽입하여 이미지 토큰을 읽어온다. 이미지 토큰을 LLM 디코더에 직접 나열하지 않으므로 계산 효율성이 높으나, 구현이 복잡하고 대규모 pretraining 데이터가 필수적이다.

본 연구는 이러한 기존 접근 방식들이 LLM backbone의 동결 여부나 데이터셋 구성의 차이로 인해 직접적인 비교가 어려웠음을 지적하며, 통제된 실험 환경에서의 분석을 통해 차별점을 둔다.

## 🛠️ Methodology

### 1. Shared Vision Pathway

모든 NVLM 모델은 **InternViT-6B-448px-V1-5**를 Vision Encoder로 사용하며, 학습 과정 내내 frozen 상태로 유지한다. 고해상도 처리를 위해 Dynamic High-Resolution (DHR) 방식을 채택한다.

* **Dynamic Tiling**: 입력 이미지를 최대 6개의 타일과 1개의 전체 요약 썸네일(thumbnail) 타일로 분할한다.
* **Token Reduction**: 각 타일에서 생성된 1,024개의 토큰은 Pixel shuffle 연산을 통해 256개로 downsampling 되어 LLM의 연산 부담을 줄인다.

### 2. Model Architectures

#### NVLM-D (Decoder-only)

Vision encoder와 LLM을 2-layer MLP projector로 연결한다. 모든 이미지 토큰이 텍스트 토큰과 함께 LLM의 self-attention 층으로 입력된다. 특히, 타일 간의 경계를 구분하기 위해 텍스트 기반의 **1-D tile tag**($<tile\_1>, \dots, <tile\_k>, <tile\_global>$)를 이미지 토큰 앞에 삽입한다.

#### NVLM-X (Cross-attention)

Gated cross-attention 층을 LLM 층 사이에 삽입한다. Flamingo와 달리 **Perceiver resampler를 제거**하였는데, 이는 Perceiver가 이미지 토큰을 섞어 공간적 관계를 파괴함으로써 OCR 성능을 저하시키기 때문이다. 대신, tile tag를 cross-attention 마스크와 연동하여 LLM이 각 태그에 해당하는 타일 토큰만 참조하도록 설계하였다.

#### NVLM-H (Hybrid)

두 방식의 장점을 결합한 구조이다.

* **Thumbnail tokens**: LLM의 self-attention 층으로 직접 입력되어 텍스트와 함께 통합적인 multimodal reasoning을 수행한다.
* **Regular tile tokens**: Gated cross-attention 층을 통해 처리되어 세부적인 시각 정보를 효율적으로 캡처한다.

### 3. Training Procedure & Loss Function

학습은 두 단계로 진행된다.

1. **Pretraining**: LLM과 Vision encoder를 frozen 상태로 두고, Modality-alignment 모듈(MLP 또는 Cross-attn layers)만 학습시킨다.
2. **Supervised Fine-Tuning (SFT)**: Vision encoder는 frozen 상태로 유지하되, **LLM backbone과 alignment 모듈을 함께 학습(unfrozen)**시킨다. 이때 catastrophic forgetting을 방지하기 위해 고품질의 **Text-only SFT 데이터**를 multimodal 데이터와 함께 학습시킨다.

## 📊 Results

### 1. 실험 설정

* **Backbone**: Qwen2-72B-Instruct
* **Vision Encoder**: InternViT-6B
* **벤치마크**: MMMU, MathVista, VQAv2, AI2D, TextVQA, ChartQA, DocVQA, RealWorldQA, OCRBench (시각-언어), MMLU, GSM8K, MATH, HumanEval (텍스트 전용)

### 2. 주요 결과

* **시각-언어 성능**:
  * **NVLM-D 1.0 72B**는 OCRBench(853)와 VQAv2(85.4)에서 proprietary 모델을 포함한 모든 모델 중 최고 성능을 기록하였다.
  * **NVLM-H 1.0 72B**는 Open-access 모델 중 가장 높은 MMMU(60.2) 및 MathVista(66.6) 점수를 획득하여 강력한 multimodal reasoning 능력을 입증하였다.
* **텍스트 전용 성능**:
  * 기존 open-access MLLM(예: InternVL-2, LLaVA-OneVision)은 학습 후 텍스트 성능이 평균 6~7점 하락하는 경향을 보였다.
  * 반면, NVLM-D 1.0 72B는 backbone LLM 대비 텍스트 전용 벤치마크 평균 점수가 **4.3점 향상**되는 결과를 보였으며, 특히 수학 및 코딩 분야에서 뚜렷한 성능 향상이 관찰되었다.

### 3. 아키텍처 분석

* **효율성**: NVLM-X와 NVLM-H는 이미지 토큰을 디코더에 모두 나열하지 않으므로, NVLM-D보다 학습 및 추론 throughput이 훨씬 높다.
* **추론 능력**: NVLM-D는 통합 처리를 통해 reasoning 능력이 우수하지만 시퀀스 길이가 길어지는 단점이 있으며, NVLM-H는 이를 보완하여 효율성과 성능의 균형을 맞추었다.

## 🧠 Insights & Discussion

본 연구는 MLLM 구축에 있어 다음과 같은 중요한 인사이트를 제공한다.

첫째, **데이터의 양보다 질이 우선**이다. 무분별하게 큰 scale의 데이터보다 정교하게 큐레이션된 작업 중심의 데이터셋이 pretraining 단계부터 성능에 더 큰 영향을 미친다.

둘째, **텍스트 성능 저하 해결 방법**이다. 단순히 LLM을 frozen 상태로 두는 것(Llama 3-V 방식)은 텍스트 성능은 보존하지만 시각-언어 성능의 잠재력을 제한한다. 대신, LLM을 unfrozen 상태로 학습시키되 매우 높은 품질의 텍스트 전용 SFT 데이터를 함께 학습시키는 것이 두 영역의 성능을 모두 잡는 더 나은 전략임을 보여주었다.

셋째, **위치 정보 제공의 중요성**이다. 단순히 타일을 이어 붙이는 것보다 1-D tile tag를 통해 모델에게 타일의 구조적 정보를 명시적으로 제공하는 것이 OCR 및 복잡한 문서 이해 작업에서 결정적인 성능 향상을 가져온다.

## 📌 TL;DR

NVLM-1.0은 Decoder-only, Cross-attention, Hybrid의 세 가지 아키텍처를 제안하며, 고해상도 이미지 처리를 위한 1-D tile-tagging 설계와 고품질 데이터 큐레이션을 통해 GPT-4o 수준의 시각-언어 성능을 달성한 모델이다. 특히, 텍스트 전용 SFT 데이터를 전략적으로 활용하여 기존 MLLM들의 고질적 문제였던 텍스트 성능 저하를 극복하고 오히려 향상시켰다는 점에서 큰 의미가 있으며, 이는 향후 범용 multimodal AI 개발의 중요한 가이드라인이 될 것으로 보인다.
