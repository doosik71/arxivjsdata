# How Vision-Language Tasks Benefit from Large Pre-trained Models: A Survey

Yayun Qi, Hongxi Li, Yiqi Song, Xinxiao Wu, Jiebo Luo (2024)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전(Computer Vision)과 자연어 처리(NLP)의 접점에 있는 Vision-Language(VL) 작업들이 직면한 고질적인 문제들을 정의하고, 이를 해결하기 위해 최근 급격히 발전한 대규모 사전 학습 모델(Large Pre-trained Models)을 어떻게 활용할 수 있는지를 체계적으로 분석한다.

해결하고자 하는 핵심 문제는 다음과 같은 네 가지 클래식 챌린지로 요약된다:

1. **데이터 부족(Data Scarcity):** VL 작업의 학습 데이터 생성에는 많은 수작업이 필요하며, 단일 모달리티와 달리 자기지도 학습(Self-supervised learning)을 통한 데이터 구축이 매우 어렵다.
2. **추론 복잡성 증가(Escalating Reasoning Complexity):** 단순한 인식(Perception)을 넘어, 다단계 추론(Multi-hop reasoning), 상식 추론(Commonsense reasoning) 및 인과 관계 파악이 필요한 고난도 작업이 증가하고 있다.
3. **새로운 샘플에 대한 일반화(Generalization to Novel Samples):** 학습 데이터셋에 포함되지 않은 새로운 객체나 장면을 마주했을 때, 모델이 보유한 교차 모달 지식이 부족하여 예측 정확도가 떨어진다.
4. **작업의 다양성(Task Diversity):** 이미지 캡셔닝, VQA, 이미지 편집 등 작업마다 입력-출력 워크플로우와 추론 과정이 매우 달라, 단일 모델로 여러 작업을 수행하는 범용성 확보가 어렵다.

논문의 목표는 이러한 챌린지들을 해결하기 위해 LLM(Large Language Models)과 VLM(Vision-Language Models)이 도입된 최신 방법론들을 분류하고, 그 효과와 한계를 분석하여 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 기존의 서베이들이 모델의 구조(Architecture)나 사전 학습 전략(Pre-training strategy)에 집중했던 것과 달리, **'해결하고자 하는 문제(Challenge)'를 기준으로 방법론을 분류**했다는 점이다.

핵심적인 설계 아이디어는 다음과 같다:

- **LLM과 VLM의 시너지 활용:** LLM의 방대한 언어적 지식(Language Priors)과 VLM의 시각-언어 정렬(Vision-Language Alignment) 능력을 결합하여 기존의 감독 학습 기반 모델이 가진 한계를 극복한다.
- **패러다임의 전환:** 단순한 '학습 및 추론' 구조에서 벗어나, '사전 학습 $\rightarrow$ 프롬프팅 $\rightarrow$ 예측' 또는 '플래너(Planner) $\rightarrow$ 도구(Tool) 호출'과 같은 새로운 실행 패러다임을 제시한다.
- **위험 요소 분석:** 사전 학습 모델의 도입이 가져오는 성능 향상뿐만 아니라, 환각(Hallucination), 지식의 노후화(Outdated Knowledge) 등 내재적 위험성을 함께 논의하여 균형 잡힌 시각을 제공한다.

## 📎 Related Works

논문에서는 기존의 사전 학습 모델 관련 서베이들을 두 가지 부류로 나눈다:

1. **기초 정보 요약형 서베이:** 모델의 구조, 학습 데이터, 사전 학습 목표 등 '어떻게 작동하는가(How they work)'에 집중한 연구들이다.
2. **특정 작업 적용형 서베이:** 특정 하위 작업(예: 이미지 편집, OOD 탐지)에 사전 학습 모델을 어떻게 적용했는지 요약한 연구들이다.

**본 논문의 차별점:**
기존 연구들이 특정 모델이나 특정 작업에 국한된 반면, 본 논문은 이미지와 비디오를 모두 포함하는 광범위한 VL 작업 전반을 다루며, 특히 **'챌린지 중심의 분류 체계(Taxonomy)'**를 도입하여 연구자들이 직면한 구체적인 문제에 따라 어떤 모델과 전략을 선택해야 하는지 가이드라인을 제공한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

본 논문은 네 가지 주요 챌린지별로 사전 학습 모델을 활용한 해결 방법론을 다음과 같이 제시한다.

### 1. 데이터 부족 해결 (Solutions to Data Scarcity)

- **테스트 샘플 직접 추론(Direct Inference):** 학습 데이터 없이 LLM의 언어 사전 지식과 CLIP의 이미지-텍스트 유사도를 결합한다. 예를 들어, LLM이 캡션 후보를 생성하면 CLIP이 이미지와의 코사인 유사도를 계산하여 최적의 문장을 선택한다.
- **단일 모달리티 무라벨 데이터 학습:** CLIP의 공통 임베딩 공간을 활용하여, 텍스트 데이터만으로 학습한 뒤 테스트 시점에 이미지 임베딩으로 대체하는 방식을 취한다.
- **의사 쌍 데이터 생성(Pseudo Paired Data Generation):** Stable Diffusion과 같은 생성형 모델을 이용하여 텍스트에 맞는 가상 이미지를 생성하거나, VLM을 이용해 이미지에 대한 의사 캡션을 생성하여 학습 데이터로 활용한다.

### 2. 추론 복잡성 해결 (Solutions to Escalating Reasoning Complexity)

- **분할 정복(Divide-and-Conquer):** 복잡한 질문을 LLM을 통해 여러 개의 단순한 하위 질문(Sub-questions)으로 분해한다. 각 하위 질문에 대해 VLM이 시각적 세부 정보를 답하고, LLM이 이를 통합하여 최종 답을 내놓는다.
- **생각의 사슬(Chain-of-Thought, CoT):** 예측 과정을 일련의 중간 추론 단계(Rationales)로 분해한다. 텍스트 기반의 논리적 단계를 생성하거나, 최근에는 중간 단계로 이미지를 생성하는 '시각적 CoT' 방식이 도입되었다.

### 3. 새로운 샘플 일반화 해결 (Solutions to Generalization to Novel Samples)

- **LLM으로부터 세만틱 컨텍스트 추출:** LLM에게 특정 클래스의 상세 묘사(Descriptor)를 생성하게 하여 이를 CLIP의 프롬프트로 사용함으로써, 학습되지 않은 새로운 클래스에 대한 인식 능력을 높인다.
- **VLM의 교사 지식 증류(Knowledge Distillation):** 광범위한 지식을 가진 VLM을 교사 모델(Teacher)로, 특정 작업에 최적화된 모델을 학생 모델(Student)로 설정하여 VLM의 일반화 능력을 전이시킨다.

### 4. 작업 다양성 해결 (Solutions to Task Diversity)

- **지속 학습(Continual Learning):** 프롬프트 학습(Prompt Learning)이나 지시어 튜닝(Instruction Tuning)을 통해 하나의 VLM이 기존 지식을 잊지 않으면서 새로운 작업을 순차적으로 배울 수 있게 한다.
- **자연어/코드 기반 플래닝(Planning):** LLM을 '플래너'로 설정한다. LLM은 사용자의 지시를 받고 이를 수행하기 위한 일련의 계획(자연어 시퀀스 또는 Python 코드)을 생성하며, 계획에 따라 적절한 외부 도구(VLM, OCR, 검색 엔진 등)를 호출하여 결과를 도출한다.

## 📊 Results

논문은 각 방법론의 성능을 입증하기 위해 다양한 벤치마크 결과(Table III, IV, V, VI)를 제시한다.

- **이미지 캡셔닝 (데이터 부족 문제):** COCO 및 Flickr30K 데이터셋에서 실험한 결과, 의사 쌍 데이터를 생성하여 학습시킨 방법이 완전 감독 학습(Fully Supervised) 모델에 가장 근접한 성능을 보였으며, 단일 모달리티 학습 방법이 그 뒤를 이었다.
- **복잡한 시각 추론 (추론 복잡성 문제):** OK-VQA, A-OKVQA, VCR 등의 데이터셋에서 분할 정복 및 CoT 기반 방법론들이 MiniGPT-4와 같은 단일 단계 추론(One-step reasoning) 모델보다 일관되게 높은 정확도를 기록하였다.
- **개방형 어휘 인식 (일반화 문제):** ImageNet, CUB 등 6개 데이터셋에서 LLM의 세만틱 컨텍스트를 활용한 방법이 기본 CLIP 모델보다 높은 정확도를 보였으며, LVIS/COCO 객체 탐지 작업에서도 VLM 지식 증류 방식이 기존 베이스라인을 상회하는 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 유용성

사전 학습 모델의 도입은 단순한 성능 향상을 넘어, 모델의 **해석 가능성(Interpretability)**을 크게 높였다. 특히 CoT나 플래닝 기반 시스템은 모델이 어떤 논리적 단계를 거쳐 답을 냈는지 투명하게 보여준다.

### 잠재적 위험 및 한계 (Critical Analysis)

논문은 사전 학습 모델 사용 시 다음과 같은 네 가지 위험성을 경고한다:

1. **환각(Hallucination):** VLM의 시각적 환상이나 LLM의 사실 관계 오류가 발생할 수 있으며, 특히 다단계 추론 과정에서 이 오류가 누적되는 '스노우볼 효과(Snowball effect)'가 나타날 수 있다.
2. **지식의 노후화(Outdated Knowledge):** 모델의 파라미터는 정적이므로, 학습 이후의 최신 정보는 반영하지 못한다. 이를 위해 RAG(Retrieval-Augmented Generation)나 지식 편집(Knowledge Editing)의 도입이 필요하다.
3. **개념 연관 편향(Concept Association Bias):** CLIP과 같은 모델은 문장을 '단어의 집합(Bag-of-words)'으로 처리하는 경향이 있어, 문법 구조를 무시하고 단순히 이미지 내 객체 존재 여부에만 의존해 유사도를 계산하는 편향이 존재한다.
4. **조합적 개념 혼동(Compositional Concept Confusion):** 형용사, 숫자, 전치사 등 세부적인 수식어의 차이를 구분하지 못하는 문제가 있으며, 이는 정밀한 시각적 속성 구분 작업에서 치명적인 한계가 된다.

## 📌 TL;DR

본 논문은 Vision-Language 작업의 4대 난제(**데이터 부족, 추론 복잡성, 일반화, 작업 다양성**)를 해결하기 위해 LLM과 VLM을 통합하는 최신 전략들을 체계적으로 정리한 서베이 보고서이다. 특히 **'분할 정복', '생각의 사슬(CoT)', '세만틱 컨텍스트 추출', '코드 기반 플래닝'**과 같은 핵심 패러다임을 제시하며, 단순한 성능 지표를 넘어 사전 학습 모델이 가진 **내재적 편향과 환각 문제**라는 현실적인 위험 요소까지 심도 있게 논의하였다. 이 연구는 향후 더 정교한 다중 모달 시스템을 설계하려는 연구자들에게 방법론 선택의 기준과 주의 사항을 제공하는 중요한 이정표 역할을 할 것으로 보인다.
