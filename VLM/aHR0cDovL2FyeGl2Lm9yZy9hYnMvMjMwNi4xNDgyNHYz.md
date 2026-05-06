# KOSMOS-2: Grounding Multimodal Large Language Models to the World

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, Furu Wei (2023)

## 🧩 Problem to Solve

기존의 Multimodal Large Language Models (MLLMs)는 텍스트, 이미지, 오디오 등 다양한 모달리티를 인식하고 자유 형식의 텍스트를 생성하는 데 성공적인 성과를 거두어 왔다. 그러나 이러한 모델들은 텍스트를 시각적 세계의 구체적인 영역과 연결하는 **Grounding** 능력이 부족하다는 한계가 있다.

Grounding 능력이 결여되면 사용자가 이미지의 특정 객체를 지칭하기 위해 매우 상세한 텍스트 설명을 입력해야 하며, 모델 역시 텍스트로만 답변을 제공하므로 지칭 대상의 모호성(coreference ambiguity)을 완전히 해결하기 어렵다. 본 논문의 목표는 MLLM에 Grounding 능력을 부여하여, 사용자가 바운딩 박스(bounding boxes)를 통해 이미지 영역을 직접 지정하거나, 모델이 시각적 답변(바운딩 박스)을 통해 텍스트를 이미지 영역에 연결할 수 있도록 하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 시각적 위치 정보를 텍스트 토큰과 동일한 방식으로 처리할 수 있도록 **이산화된 위치 토큰(discrete location tokens)**으로 변환하고, 이를 마크다운(Markdown)의 하이퍼링크 구조와 유사하게 텍스트 스팬과 연결하는 것이다.

구체적으로, 본 논문은 다음과 같은 기여를 한다:

1. **GRIT (Grounded Image-Text pairs) 데이터셋 구축**: 웹 스케일의 이미지-텍스트 쌍에서 명사구와 지칭 표현을 추출하고 이를 이미지 내 바운딩 박스와 연결한 대규모 데이터셋을 구축하였다.
2. **Grounding 및 Referring 능력 통합**: KOSMOS-1의 아키텍처를 기반으로, 바운딩 박스를 텍스트 시퀀스로 표현함으로써 모델이 시각적 영역을 인식하고 텍스트를 시각 세계에 Grounding할 수 있게 하였다.
3. **범용적 인터페이스 확장**: 단순한 텍스트 생성을 넘어, Grounded Image Captioning, Grounded VQA, Referring Expression Comprehension/Generation 등 고도의 시각-언어 작업을 수행할 수 있는 기반을 마련하였다.

## 📎 Related Works

본 연구는 이전 모델인 KOSMOS-1을 계승하며, 대규모 이미지-텍스트 데이터셋인 LAION-2B와 COYO-700M을 활용한다. 기존의 MLLM들은 주로 전역적인 이미지 특징을 이해하고 텍스트를 생성하는 데 집중하였으나, KOSMOS-2는 객체 수준의 세밀한 위치 정보(location tokens)를 학습 과정에 도입함으로써 기존 접근 방식과 차별화된다. 또한, 외부 detector에 의존하는 GRILL과 같은 모델과 달리, 모델 자체가 위치 토큰을 생성하도록 학습되어 더 통합적인 추론이 가능하다.

## 🛠️ Methodology

### 1. GRIT 데이터셋 구축 파이프라인

웹 스케일의 데이터셋을 구축하기 위해 다음의 2단계 파이프라인을 거친다.

- **Step-1: 명사구-바운딩 박스 쌍 생성**: `spaCy`를 사용하여 캡션에서 명사구(noun chunks)를 추출하고, 사전 학습된 grounding 모델인 `GLIP`을 통해 이미지 내의 바운딩 박스를 획득한다. 신뢰도 점수가 0.65 이상인 박스만 유지하며, NMS(Non-maximum suppression)를 통해 중복을 제거한다.
- **Step-2: 지칭 표현-바운딩 박스 쌍 생성**: 단순 명사구를 더 복잡한 지칭 표현(referring expressions)으로 확장한다. `spaCy`의 의존성 트리(dependency tree)를 재귀적으로 탐색하여 자식 토큰들을 결합함으로써 "a dog"를 "a dog in a field of flowers"와 같이 확장하고, 해당 명사구의 바운딩 박스를 확장된 표현에 할당한다.

### 2. Grounded 입력 표현 및 아키텍처

KOSMOS-2는 Transformer 기반의 Causal Language Model을 사용하며, 다음의 방식으로 시각적 위치를 인코딩한다.

- **위치 토큰의 이산화**: 이미지의 너비($W$)와 높이($H$)를 각각 $P$개의 세그먼트로 나누어 $P \times P$개의 빈(bin)을 생성한다. 본 논문에서는 $P=32$를 사용하여 총 1,024개의 위치 토큰을 정의하고 이를 단어 사전(vocabulary)에 추가한다.
- **바운딩 박스 표현**: 하나의 바운딩 박스는 좌상단 좌표 $(\text{loc}_1)$와 우하단 좌표 $(\text{loc}_2)$를 사용하여 다음과 같이 표현된다:
  $$\text{<box>}\text{<loc}_1\text{>}\text{<loc}_2\text{>}\text{</box>}$$
  여러 개의 박스가 있을 경우 $\text{<delim>}$ 토큰으로 구분한다.
- **하이퍼링크 포맷**: 텍스트 스팬과 위치 토큰을 연결하기 위해 마크다운 스타일의 포맷을 사용한다:
  $$\text{<p>text span</p><box><loc}_1\text{><loc}_2\text{></box>}$$
  여기서 $\text{<p>}$와 $\text{</p>}$는 텍스트 스팬의 시작과 끝을 알리는 특수 토큰이다.

### 3. 학습 절차

- **훈련 목표**: 다음 토큰 예측(next-token prediction) 작업을 통해 학습한다.
- **데이터 구성**: 텍스트 코퍼스, 이미지-캡션 쌍, 인터리브(interleaved) 데이터와 더불어 새롭게 구축한 GRIT 데이터를 함께 사용한다.
- **Instruction Tuning**: 모델을 인간의 지시사항에 맞게 정렬하기 위해 LLaVA-Instruct, Unnatural Instructions, FLANv2 및 GRIT 기반의 Grounded 지시 데이터를 사용하여 튜닝한다.

## 📊 Results

### 1. Multimodal Grounding

- **Phrase Grounding (Flickr30k Entities)**: Zero-shot 설정에서 외부 detector를 사용하는 GRILL보다 훨씬 뛰어난 성능을 보였으며, 기존의 fine-tuned 모델인 VisualBert보다 R@1 지표에서 약 7.4% 높은 성능을 달성하였다.
- **Referring Expression Comprehension (RefCOCO, RefCOCO+, RefCOCOg)**: Zero-shot 성능이 전반적으로 유망하며, 특히 RefCOCOg에서 이전 zero-shot 모델들을 크게 상회하였다. 다만, 짧은 표현이 많은 RefCOCO/RefCOCO+에서는 fine-tuned 모델보다 약간 낮은 성능을 보였다.

### 2. Multimodal Referring

- **Referring Expression Generation (RefCOCOg)**: 바운딩 박스가 주어졌을 때 이를 설명하는 텍스트를 생성하는 작업에서, Zero-shot임에도 불구하고 fine-tuned 모델인 SLR보다 CIDEr 점수 기준 1.1점 높게 나타났다. 또한, Few-shot 설정(k=4)에서 성능이 더욱 향상되어 in-context learning 능력을 입증하였다.

### 3. Perception-Language 및 Language Tasks

- **이미지 캡셔닝 및 VQA**: KOSMOS-1과 비교했을 때 Flickr30k(캡셔닝)에서는 약간의 성능 향상이 있었고, VQAv2에서는 미세하게 감소하였으나 전반적으로 경쟁력 있는 수준을 유지하였다.
- **언어 작업**: SuperGLUE 등 8가지 언어 벤치마크에서 KOSMOS-1과 유사한 성능을 보여, Grounding 능력을 추가했음에도 기존의 언어 이해 능력이 훼손되지 않았음을 확인하였다.

## 🧠 Insights & Discussion

KOSMOS-2는 MLLM에 Grounding 능력을 통합함으로써 단순한 텍스트 생성기를 넘어 시각적 세계와 상호작용할 수 있는 모델로 진화하였다. 특히 Zero-shot 환경에서도 강력한 성능을 보인다는 점은 대규모 Grounded 데이터셋(GRIT)의 구축이 모델의 일반화 능력에 결정적인 역할을 했음을 시사한다.

**한계점 및 논의**:

- **표현의 다양성 부족**: RefCOCO/RefCOCO+ 데이터셋에서 성능이 상대적으로 낮은 이유는 해당 데이터셋들이 "left bottom"과 같은 매우 짧고 단순한 공간적 지칭 표현을 주로 사용하기 때문으로 분석된다. 이는 모델이 인간의 다양한 지칭 방식(short-hand expressions)을 더 정교하게 이해해야 함을 의미한다.
- **실제 적용 가능성**: 바운딩 박스를 통해 입출력을 처리하는 방식은 Embodied AI(로봇 공학 등)에서 특정 객체를 지정하고 조작하는 인터페이스로 확장될 가능성이 매우 크다.

## 📌 TL;DR

KOSMOS-2는 바운딩 박스를 이산화된 위치 토큰으로 변환하여 텍스트와 연결하는 방식을 통해 **MLLM에 Grounding 능력을 부여한 모델**이다. 대규모 GRIT 데이터셋을 통해 학습되었으며, Zero-shot 설정에서도 텍스트를 이미지 영역에 매핑하거나 이미지 영역을 텍스트로 설명하는 작업에서 뛰어난 성능을 보인다. 이 연구는 언어, 지각, 행동이 통합되는 AGI로 가는 핵심 단계인 '시각적 세계로의 접지(Grounding)'를 성공적으로 구현하였다.
