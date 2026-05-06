# LARGELANGUAGEMODELSMEETCOMPUTERVISION; A BRIEF SURVEY

Raby Hamadi(2023)

## 🧩 Problem to Solve

본 논문은 최근 인공지능 분야의 핵심인 Large Language Models (LLMs)와 Computer Vision (CV)의 융합이라는 최신 연구 흐름을 체계적으로 분석하는 것을 목표로 한다.

전통적으로 자연어 처리(NLP)와 컴퓨터 비전(CV)은 서로 다른 영역으로 다뤄져 왔으나, Transformer 아키텍처가 두 분야 모두에서 핵심 백본(backbone)으로 자리 잡으면서 이들의 경계가 허물어지고 있다. 특히, 텍스트를 처리하도록 설계된 LLM이 시각적 데이터를 해석하고 이해하는 능력을 확장함으로써 인간과 유사하게 세상을 보고, 이해하고, 소통하는 통합 AI 모델을 구축하는 것이 매우 중요하다.

본 논문의 구체적인 목표는 다음과 같다:

1. Transformer와 그 후속 아키텍처(예: RetNet)의 진화를 분석하여 Vision Transformer (ViTs)와 LLMs의 발전 가능성을 제시한다.
2. 유료 및 오픈소스 LLM들의 성능 지표를 비교 분석하여 각 모델의 강점과 개선점을 밝힌다.
3. LLM이 비전 관련 태스크를 해결하기 위해 어떻게 활용되고 있는지에 대한 문헌 연구를 수행한다.
4. LLM 및 VLM 학습에 사용되는 다양한 데이터셋을 정리하여 제공한다.
5. 향후 연구 방향과 해결해야 할 과제들을 제시한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 LLM과 CV의 교차점을 포괄적으로 조망한 서베이(Survey)를 제공했다는 점이다. 구체적인 설계 아이디어와 분석 내용은 다음과 같다.

- **아키텍처의 진화 분석**: 기존 Transformer의 연산 효율성 문제(quadratic complexity)를 해결하기 위한 Retentive Network (RetNet)를 소개하고, 이를 2D 공간으로 확장하여 이미지 데이터를 처리하는 RetNet-based ViT의 메커니즘을 설명한다.
- **실용적 LLM 벤치마크 제공**: LlamaIndex 기능을 기준으로 GPT-4, Claude-2와 같은 유료 모델과 Llama2, Mistral-7B와 같은 오픈소스 모델의 실제 작동 가능 여부를 비교 분석한 표를 제시한다.
- **다양한 CV 응용 사례 분석**: 단순한 이미지 캡셔닝을 넘어 비디오 내레이션(LAVILA), 3D 공간 이해(3D-LLM), 포인트 클라우드 데이터 처리(LAMM), 전자상거래 상품 이해(PUMGPT) 등 최신 LLM-CV 융합 사례들을 상세히 다룬다.
- **방대한 데이터셋 아카이브**: 텍스트 전용 데이터셋뿐만 아니라 수억 개에서 수십억 개의 이미지-텍스트 쌍으로 구성된 대규모 시각-언어 데이터셋들을 체계적으로 정리하여 제시한다.

## 📎 Related Works

논문은 기존의 LLM 및 VLM 관련 서베이들의 한계를 지적하며 본 연구의 차별성을 강조한다.

- **LLM 중심 서베이 ([10], [11])**: LLM의 스케일링 법칙(scaling laws), 창발적 능력(emergent capabilities), 다양한 산업 적용 사례를 다루고 있으나, 정작 이러한 LLM들이 컴퓨터 비전 분야에서 어떻게 구체적으로 사용될 수 있는지에 대한 설명이 부족하다.
- **평가 방법론 서베이 ([12])**: LLM의 평가 환경과 벤치마크, 성공 및 실패 사례를 분석하지만, 비전 모델과의 통합 관점에서의 평가는 다루지 않는다.
- **VLM 프리트레이닝 서베이 ([13])**: 시각-언어 사전 학습 모델(VLPMs)의 구조와 학습 전략을 상세히 설명하지만, 매우 빠르게 발전하고 있는 최신 Transformer 후속 모델들이나 최신 NLP 기법들을 반영하지 못하는 한계가 있다.

본 논문은 이러한 공백을 메우기 위해 RetNet과 같은 최신 아키텍처를 포함하고, LLM이 실제 비전 태스크에 어떻게 투영되는지를 최신 문헌을 통해 상세히 분석한다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 새로운 알고리즘을 제안하기보다, 기존 모델들의 핵심 방법론을 분석하여 설명한다.

### 1. Transformer 및 RetNet

Transformer는 Self-attention 메커니즘을 통해 시퀀스 내 데이터의 중요도를 가중치로 계산한다. 하지만 시퀀스 길이가 길어질수록 연산량이 제곱으로 증가하는 문제가 있다. 이를 해결하기 위한 **RetNet (Retentive Network)**은 다음과 같은 세 가지 계산 패러다임을 통합한다.

- **Parallel representation**: 학습 시 병렬 처리를 가능하게 하여 효율성을 높인다.
- **Recurrent representation**: 추론 시 낮은 비용으로 빠른 디코딩(decoding)을 가능하게 한다.
- **Chunkwise recurrent representation**: 긴 시퀀스를 청크 단위로 나누어 선형 복잡도로 처리한다.

RetNet의 1D 입력 시퀀스에 대한 연산 식은 다음과 같다.
$$o_n = \sum_{m=1}^{n} \gamma^{n-m} (Q_n e^{in\theta})(K_m e^{im\theta})^\dagger v_m$$
여기서 $\dagger$는 conjugate transpose를 의미한다.

### 2. RetNet-Based Vision Transformers

RetNet은 기본적으로 1D 데이터용이므로, 이를 이미지와 같은 2D 공간에 적용하기 위해 **Bidirectional extension**과 **Manhattan distance** 개념을 도입한다.

- **Bidirectional Retention**: 토큰 간의 방향성에 구애받지 않고 양방향으로 정보를 처리한다.
- **2D Extension**: 이미지 내 토큰 $n$과 $m$의 좌표를 $(x_n, y_n)$과 $(x_m, y_m)$이라고 할 때, 거리 행렬 $D^{2d}$를 다음과 같이 정의한다.
$$D^{2d}_{nm} = \gamma^{|x_n - x_m| + |y_n - y_m|}$$
최종적인 2D retention 연산은 다음과 같이 표현된다.
$$\text{ReSA}(x) = (\text{Softmax}(QK^T) \odot D^{2d})V$$

### 3. Vision-Language Models (VLMs) 구조

VLM은 기본적으로 세 가지 핵심 구성 요소로 이루어진다.

- **Image Encoder**: 이미지를 특징 벡터로 추출한다.
- **Text Encoder**: 텍스트를 임베딩 공간으로 변환한다.
- **Fusion Mechanism**: 두 도메인의 정보를 통합한다.

주요 프리트레이닝 전략으로는 Contrastive Learning(이미지와 텍스트의 정렬), PrefixLM(이미지를 언어 모델의 접두사로 사용), Multi-modal Fusing(Cross Attention을 통한 통합) 등이 사용된다.

## 📊 Results

본 논문은 정량적인 단일 실험 결과보다는 여러 최신 모델들의 벤치마크 결과와 비교 분석 표를 제시한다.

### 1. LLM 성능 비교 (LlamaIndex 기준)

- **Paid LLMs**: GPT-4와 GPT-3.5-turbo는 대부분의 기능(Basic Query, Router, Text2SQL, Data Agents 등)을 완벽하게 지원하며 가장 신뢰도가 높다.
- **Open-Source LLMs**: Zephyr-7b-beta와 Llama2-70b-chat 등이 우수한 성능을 보이지만, 모델 크기가 작을수록(예: Mistral-7B 4bit) 일부 고급 기능(Pydantic Programs, Data Agents) 지원이 불충분하거나 부분적으로만 지원($\triangle$)된다.

### 2. 특정 모델의 성과

- **VisionLLM**: 이미지를 하나의 '외국어'로 처리하는 프레임워크를 통해, 객체 탐지(Detection) 전용 모델과 경쟁 가능한 수준인 COCO 데이터셋 mAP 60% 이상을 달성하였다.
- **PUMGPT**: Layer-wise Adapters (LA)를 통해 시각-텍스트 표현을 효율적으로 정렬하여, 상품 캡셔닝 및 속성 추출 등 이커머스 관련 태스크에서 우수한 성능을 보였다.
- **LAVILA**: LLM을 비디오 입력에 조건화하여 자동 비디오 내레이터로 활용함으로써 기존 SOTA 모델보다 높은 분류 및 검색 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 기회

본 논문은 단순한 아키텍처 설명을 넘어, LLM이 어떻게 CV의 다양한 도메인(3D, 비디오, 이커머스)으로 확장될 수 있는지 구체적인 사례를 통해 보여주었다. 특히 RetNet과 같은 차세대 아키텍처가 Transformer의 효율성 문제를 해결함으로써 더 거대한 시각-언어 모델의 등장을 가능케 할 것임을 시사한다.

### 한계 및 향후 과제

저자는 LLM과 VLM의 발전을 위해 다음과 같은 비판적 시각과 해결 과제를 제시한다.

1. **평가 체계의 한계**: 현재의 정적인 벤치마크는 모델이 데이터를 암기하는 현상을 걸러내지 못한다. 따라서 인간의 지능과 유사한 적응성, 추상적 추론, 윤리적 판단 등을 측정할 수 있는 **AGI 벤치마크**와 **동적 평가 프로토콜**이 필요하다.
2. **VLM 학습 전략의 비효율성**: 현재 VLM은 주로 다운스트림 태스크(downstream tasks)에서만 평가되는데, 이는 학습 후반부에야 결함을 발견하게 되어 자원 낭비가 심하다. NLP의 Perplexity와 유사한 **중간 지표(Intermediate metrics)**를 개발하여 학습 과정 중에 성능을 예측할 수 있어야 한다.
3. **미세한 정렬(Alignment) 문제**: 단순히 마스킹을 통해 학습하는 방식으로는 이미지의 세부 특징과 텍스트 특징 사이의 정교한 정렬을 보장하기 어렵다. 이를 해결하기 위해 임베딩 레벨에서의 마스킹(embedding-level masking) 도입을 제안한다.

## 📌 TL;DR

본 논문은 **LLM과 컴퓨터 비전의 융합**을 다루는 종합 서베이로, Transformer에서 RetNet으로 이어지는 아키텍처의 진화가 비전 태스크의 효율성을 어떻게 높이는지 분석한다. 또한, 유료/오픈소스 LLM의 실용적 비교와 함께 3D, 비디오 등 다양한 CV 응용 사례를 정리하였다. 특히, 단순한 성능 향상을 넘어 AGI 수준의 평가 체계 구축과 VLM의 효율적인 프리트레이닝 전략의 필요성을 강조하며, 향후 통합 AI 모델이 나아갈 방향을 제시하고 있다.
