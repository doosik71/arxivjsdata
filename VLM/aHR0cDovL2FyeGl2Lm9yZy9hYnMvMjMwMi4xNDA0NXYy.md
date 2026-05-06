# Language Is Not All You Need: Aligning Perception with Language Models

Shaohan Huang et al. (2023)

## 🧩 Problem to Solve

현재의 거대 언어 모델(Large Language Models, LLMs)은 텍스트 기반의 다양한 자연어 처리 작업에서 범용 인터페이스로서 성공적으로 작동하고 있다. 그러나 LLM은 이미지나 오디오와 같은 멀티모달 데이터를 네이티브하게 처리하는 능력에 한계가 있으며, 이는 실제 세계에 대한 지식 습득과 접지(grounding) 측면에서 인공 일반 지능(Artificial General Intelligence, AGI)을 달성하는 데 큰 장애물이 된다.

본 논문의 목표는 지각 능력(perception)을 LLM과 정렬(align)하여, 모델이 시각적 정보를 이해하고 이를 바탕으로 대화할 수 있는 멀티모달 거대 언어 모델(Multimodal Large Language Model, MLLM)인 $\text{KOSMOS-1}$을 구축하는 것이다. 이를 통해 로보틱스, 문서 지능 등 더 높은 가치를 지닌 영역으로 언어 모델의 응용 범위를 확장하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 언어 모델을 범용 작업 레이어(general-purpose interface)로 간주하고, 여기에 지각 모듈을 결합하여 텍스트와 이미지가 임의로 섞여 있는 웹 규모의 거대 데이터셋으로 학습시키는 것이다.

주요 기여 사항은 다음과 같다.

- **MLLM의 구축**: 텍스트, 이미지-캡션 쌍, 그리고 텍스트와 이미지가 교차 배치된(interleaved) 대규모 코퍼스를 사용하여 $\text{KOSMOS-1}$을 처음부터 학습시켰다.
- **범용 인터페이스로서의 언어 모델**: $\text{KOSMOS-1}$은 별도의 그래디언트 업데이트나 미세 조정(finetuning) 없이 제로샷(zero-shot) 및 퓨샷(few-shot) 설정에서 멀티모달 작업을 수행할 수 있다.
- **비언어적 추론 능력 검증**: MLLM의 비언어적 추론 능력을 진단하기 위해 Raven IQ 테스트 벤치마크를 도입하고 이를 평가하였다.
- **교차 모달 전이(Cross-modal Transfer)**: 언어에서 멀티모달로, 혹은 멀티모달에서 언어로 지식이 전이되는 현상을 확인하여 지각 능력의 결합이 상식 추론 성능을 향상시킴을 입증하였다.

## 📎 Related Works

본 연구는 언어 모델을 범용 인터페이스로 활용한다는 $\text{METALM}$의 철학을 계승한다. 기존의 멀티모달 접근 방식들은 특정 작업에 특화된 아키텍처를 사용하거나 제한된 데이터셋으로 학습되는 경향이 있었다.

$\text{KOSMOS-1}$은 $\text{Flamingo}$와 같은 기존 모델과 비교하여 더 작은 모델 크기(1.6B)로도 제로샷 이미지 캡셔닝 등에서 우수한 성능을 보인다. 특히 $\text{Flamingo}$가 제로샷 설정에서 텍스트 기반의 예시를 사용하는 것과 달리, $\text{KOSMOS-1}$은 진정한 의미의 제로샷 학습 능력을 보여준다는 점과 웹 규모의 인터리브(interleaved) 데이터를 통해 인컨텍스트 학습(in-context learning) 능력을 극대화했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

$\text{KOSMOS-1}$은 기본적으로 Transformer 기반의 Causal Language Model을 백본으로 사용한다. 텍스트 외의 모달리티는 임베딩 모듈을 통해 벡터로 변환되어 언어 모델의 입력으로 들어간다.

1. **입력 표현(Input Representation)**: 입력 데이터는 특수 토큰($\langle s \rangle$, $\langle /s \rangle$, $\langle image \rangle$, $\langle /image \rangle$)으로 구분된 시퀀스로 평탄화된다.
2. **비전 인코더 및 리샘플러(Vision Encoder & Resampler)**: 이미지는 사전 학습된 $\text{CLIP ViT-L/14}$ 모델을 통해 인코딩된다. 이후 $\text{Resampler}$라 불리는 attentive pooling 메커니즘을 사용하여 이미지 임베딩의 수를 줄여 Transformer 디코더에 입력한다.
3. **백본 아키텍처**:
   - **$\text{MAGNETO}$**: 학습 안정성과 성능 향상을 위해 각 서브레이어에 추가적인 $\text{LayerNorm}$을 도입한 $\text{MAGNETO}$ 변형 Transformer를 사용한다.
   - **$\text{XPOS}$**: 긴 컨텍스트 모델링 및 길이 외삽(extrapolation) 능력을 위해 $\text{XPOS}$ 상대 위치 인코딩을 적용하였다.

### 학습 절차 및 목표

모델은 다음 토큰 예측(next-token prediction) 작업을 통해 학습된다. 학습 목표는 주어진 컨텍스트 하에서 다음 토큰의 로그 가능도(log-likelihood)를 최대화하는 것이다.

$$ \mathcal{L} = \sum_{t} \log P(x_t | x_{<t}) $$

학습 시 손실 함수는 텍스트와 같은 이산 토큰(discrete tokens)에 대해서만 계산된다. 학습 데이터는 다음과 같이 구성된다.

- **단일 모달 데이터**: $\text{The Pile}$, $\text{Common Crawl}$ 등의 텍스트 코퍼스.
- **교차 모달 쌍 데이터**: $\text{LAION-2B}$, $\text{COYO-700M}$ 등 이미지-캡션 쌍.
- **인터리브 데이터**: 웹페이지에서 추출한 텍스트와 이미지가 섞여 있는 7,100만 개의 문서.

### 언어 전용 지시어 튜닝(Language-Only Instruction Tuning)

인간의 지시를 더 잘 따르게 하기 위해 $\text{Unnatural Instructions}$와 $\text{FLANv2}$ 데이터셋을 사용하여 언어 전용 지시어 튜닝을 수행하였다. 흥미로운 점은 언어 영역에서 학습된 지시어 수행 능력이 멀티모달 영역으로 전이된다는 것이다.

## 📊 Results

### 실험 설정 및 지표

- **이미지 캡셔닝**: $\text{MS COCO}$, $\text{Flickr30k}$ 데이터셋에서 $\text{CIDEr}$ 및 $\text{SPICE}$ 지표 측정.
- **VQA**: $\text{VQAv2}$, $\text{VizWiz}$ 데이터셋에서 정확도(Accuracy) 측정.
- **비언어적 추론**: $\text{Raven IQ Test}$의 50개 예제에 대해 정확도 측정.
- **OCR-free 이해**: $\text{Rendered SST-2}$, $\text{HatefulMemes}$에서 정확도 및 $\text{ROC AUC}$ 측정.
- **이미지 분류**: $\text{ImageNet}$의 $\text{Top-1}$ 정확도 측정.

### 주요 결과

1. **지각-언어 작업**: $\text{KOSMOS-1}$은 1.6B의 작은 크기임에도 불구하고 $\text{Flickr30k}$ 제로샷 캡셔닝에서 $\text{CIDEr}$ 67.1을 기록하며 $\text{Flamingo-9B}$보다 우수한 성능을 보였다. VQA에서도 $\text{VizWiz}$ 데이터셋에서 높은 강건성을 보였다.
2. **비언어적 추론**: $\text{Raven IQ Test}$에서 무작위 선택(17%)보다 높은 22%의 정확도를 기록하였으며, 지시어 튜닝 시 26%까지 향상되어 MLLM이 추상적 패턴을 인식할 가능성을 보여주었다.
3. **OCR-free 이해**: 외부 OCR 도구 없이 $\text{Rendered SST-2}$에서 67.1%의 정확도를 기록하여 이미지 내 텍스트를 직접 읽고 이해하는 능력을 입증하였다.
4. **웹페이지 QA**: $\text{WebSRC}$ 작업에서 텍스트만 사용한 $\text{LLM}$보다 높은 성능을 보여, 이미지의 레이아웃과 스타일 정보가 이해에 도움을 줌을 확인하였다.
5. **교차 모달 전이**:
   - **언어 $\rightarrow$ 멀티모달**: 언어 전용 지시어 튜닝이 VQA 및 캡셔닝 성능을 유의미하게 향상시켰다.
   - **멀티모달 $\rightarrow$ 언어**: $\text{KOSMOS-1}$은 텍스트 전용 $\text{LLM}$보다 물체의 크기나 색상에 관한 상식 추론($\text{RelativeSize}$, $\text{MemoryColor}$)에서 월등한 성능을 보였다. 이는 시각적 지식이 언어 작업으로 전이되었음을 의미한다.

## 🧠 Insights & Discussion

$\text{KOSMOS-1}$의 가장 큰 강점은 단순한 모달리티 결합을 넘어, 지각 능력이 모델의 전반적인 지능(특히 상식 추론)을 향상시킨다는 점을 밝혀낸 것이다. 특히 $\text{LLM}$이 텍스트 정보만으로는 알기 어려운 물리적 특성(크기, 색상 등)을 $\text{MLLM}$은 시각적 학습을 통해 습득하고 이를 언어 작업에 활용한다는 점이 인상적이다.

또한, 멀티모달 $\text{Chain-of-Thought (CoT)}$ 프롬프팅을 통해 복잡한 문제 해결 능력을 높일 수 있음을 보여주었으며, 이는 향후 MLLM의 추론 능력을 확장하는 핵심 방법론이 될 수 있다.

한계점으로는 $\text{Raven IQ Test}$와 같은 고도의 비언어적 추론 작업에서 인간 수준에 비해 여전히 성능 차이가 크다는 점이 있다. 또한 본 논문은 주로 이미지와 텍스트에 집중하고 있어, 오디오 등 다른 모달리티의 통합에 대해서는 향후 과제로 남겨두고 있다.

## 📌 TL;DR

본 논문은 웹 규모의 대규모 멀티모달 데이터를 사용하여 처음부터 학습된 $\text{KOSMOS-1}$ 모델을 제안한다. 이 모델은 제로샷 및 퓨샷 설정에서 이미지 캡셔닝, VQA, OCR-free 텍스트 이해 등 광범위한 작업을 수행하며, 특히 비언어적 추론(IQ 테스트) 능력을 보여준 최초의 MLLM 중 하나이다. 특히 시각적 지각 능력이 언어 모델의 상식 추론 능력을 향상시키는 '교차 모달 전이' 현상을 입증함으로써, 진정한 AGI를 위해서는 언어뿐만 아니라 지각 능력의 정렬이 필수적임을 시사한다.
