# Gemini in Reasoning: Unveiling Commonsense in Multimodal Large Language Models

Yuqing Wang, Yun Zhao (2023)

## 🧩 Problem to Solve

본 논문은 최근 공개된 구글의 Multimodal Large Language Model(MLLM)인 Gemini의 상식 추론(Commonsense Reasoning) 능력을 종합적으로 평가하는 것을 목표로 한다.

상식 추론은 인간이 일상생활에서 당연하게 받아들이는 암시적 지식과 믿음을 바탕으로 세상을 이해하고 해석하는 능력이다. 기존의 LLM과 MLLM들은 이러한 내재적 상식이 부족하여 데이터를 일관되게 맥락화하는 데 어려움을 겪어왔다. 특히 Gemini의 경우, 초기 벤치마크(HellaSWAG 등)에서 GPT 시리즈보다 상식 추론 능력이 떨어진다는 결과가 보고되었으나, 이는 매우 제한적인 데이터셋에 기반한 평가였다.

따라서 본 연구는 Gemini가 다양한 도메인(일반, 물리, 사회, 시간 등)에서 실제로 어느 정도의 상식 추론 잠재력을 가지고 있는지, 그리고 다른 최신 모델(GPT-4 Turbo, GPT-3.5 Turbo, Llama-2 등)과 비교했을 때 어떤 강점과 약점을 보이는지 정밀하게 분석하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Gemini의 상식 추론 능력에 대한 최초의 포괄적 평가**: 언어 기반 데이터셋 11개와 멀티모달 데이터셋 1개를 포함하여 총 12개의 다양한 데이터셋을 통해 Gemini Pro 및 Gemini Pro Vision의 효용성을 검증하였다.
2. **상대적 성능 위치 파악**: Gemini Pro가 언어 기반 상식 추론에서 GPT-3.5 Turbo와 유사한 성능을 보이며, GPT-4 Turbo보다는 낮다는 점을 정량적으로 확인하였다.
3. **취약 도메인 및 오류 유형 식별**: 시간적(Temporal) 추론, 사회적(Social) 추론, 그리고 이미지 내 정서 인식(Emotion Recognition) 분야에서 Gemini가 겪는 구체적인 한계를 분석하고 이를 정성적 사례 연구와 오류 분석을 통해 제시하였다.

## 📎 Related Works

본 논문은 상식 추론과 관련된 세 가지 주요 연구 흐름을 언급한다.

1. **NLP에서의 상식 추론**: LLM의 발전에도 불구하고 상식 지식의 이해와 추론 능력에 대한 우려가 지속되고 있다. 이를 해결하기 위해 대규모 지식 그래프(Knowledge Graphs)를 활용하거나 상식 지식 전이(Knowledge Transfer) 방법을 사용하는 연구들이 진행되어 왔다.
2. **LLM의 훈련 패러다임**: 대규모 텍스트 데이터의 사전 학습(Pre-training) 이후, 특정 태스크를 위한 미세 조정(Fine-tuning)에서 제로샷(Zero-shot) 및 퓨샷(Few-shot) 학습으로 중심이 이동하였다. 특히 Chain-of-Thought(CoT)와 같은 프롬프팅 기법이 추론 능력을 향상시키는 핵심 수단으로 활용되고 있다.
3. **MLLM 평가**: GPT-4V 출시 이후 의료 영상, VQA(Visual Question Answering) 등의 분야에서 평가가 이루어졌으나, Gemini의 상식 추론 능력을 종합적으로 다룬 연구는 부족한 상태였다.

## 🛠️ Methodology

### 1. 실험 설계 및 데이터셋

본 연구는 상식 추론을 12가지 도메인(General, Contextual, Abductive, Event, Temporal, Numerical, Physical, Science, Riddle, Social, Moral, Visual)으로 정의하고, 이에 해당하는 데이터셋을 구성하였다.

- **언어 기반 데이터셋 (11개)**:
  - 일반 및 맥락 추론: $\text{CommonsenseQA, Cosmos QA, } \alpha\text{NLI, HellaSWAG}$
  - 전문 지식 추론: $\text{TRAM (시간), NumerSense (수치), PIQA (물리), QASC (과학), RiddleSense (수수께끼)}$
  - 사회 및 윤리 추론: $\text{Social IQa, ETHICS}$
- **멀티모달 데이터셋 (1개)**: $\text{VCR (Visual Commonsense Reasoning)}$ (시각적 맥락과 상식을 결합하여 답변 및 근거를 생성하는 태스크)

### 2. 평가 모델 및 설정

- **언어 모델 (LLMs)**: $\text{Llama-2-70b-chat, Gemini Pro, GPT-3.5 Turbo, GPT-4 Turbo}$
- **멀티모달 모델 (MLLMs)**: $\text{Gemini Pro Vision, GPT-4V}$
- **프롬프팅 기법**:
  - **Zero-shot Standard Prompting (SP)**: 모델의 내재적인 상식 능력을 측정하기 위해 사용한다.
  - **Few-shot Chain-of-Thought (CoT)**: 몇 개의 예시와 추론 과정을 제공하여 성능 향상 여부를 관찰한다.
- **측정 지표**: 모든 데이터셋에 대해 $\text{Accuracy}$ (정확도)를 사용하였으며, 디코딩 시 $\text{temperature} = 0$ (Greedy Decoding)을 적용하였다.

### 3. 추론 정당성 분석 (Reasoning Justification)

단순히 정답 여부만 확인하는 것이 아니라, 모델이 제시한 근거(Rationale)의 논리적 타당성을 검토하였다. 각 모델별로 정답과 오답 샘플을 30개씩 추출하여 "답변의 근거는 무엇인가?"라는 질문을 던지고, 이에 대한 응답을 사람이 직접 검토하여 $\text{True}$ 또는 $\text{False}$로 분류하였다.

## 📊 Results

### 1. 언어 기반 상식 추론 성능

- **전체 성능**: $\text{GPT-4 Turbo}$가 모든 데이터셋에서 압도적인 1위를 차지하였다. $\text{Gemini Pro}$는 $\text{GPT-3.5 Turbo}$와 매우 유사한 성능을 보였으며, 평균적으로 제로샷에서는 $1.3\%$, 퓨샷 CoT에서는 $1.5\%$ 더 높은 정확도를 기록하였다. $\text{Llama-2-70b}$보다는 확연히 우수한 성능을 보였다.
- **프롬프팅 효과**: CoT 기법은 모든 모델에서 성능을 향상시켰으며, 특히 $\text{CommonsenseQA, TRAM, Social IQa}$에서 그 효과가 두드러졌다.
- **취약점**: 모든 모델이 $\text{TRAM (시간)}$과 $\text{Social IQa (사회)}$ 도메인에서 상대적으로 낮은 성능을 보였다.

### 2. 멀티모달 상식 추론 성능 (VCR 데이터셋)

- **정량 결과**: $\text{GPT-4V}$가 모든 하위 태스크($\text{Q} \to \text{A, QA} \to \text{R, Q} \to \text{AR}$)에서 $\text{Gemini Pro Vision}$보다 우수한 성능을 보였다.
- **특이 사항**: 다만, 질문 유형별 분석 결과 **시간 관련(Temporal)** 질문에서는 $\text{Gemini Pro Vision}$이 $\text{GPT-4V}$를 능가하는 모습을 보였다.

### 3. 추론 과정 및 오류 분석

- **추론 정당성**: $\text{GPT-4 Turbo}$는 정답을 맞히지 못한 경우에도 논리적 일관성을 유지하는 경향이 강했다. MLLM의 경우, 정답은 맞혔으나 근거는 틀린 경우가 약 $24\text{--}26\%$ 존재하여, 단순한 확률적 추측으로 정답을 맞히는 경우가 있음을 시사한다.
- **오류 유형 (LLM)**:
  - 제로샷 설정에서는 $\text{Context Misinterpretation}$ (맥락 오해, $28.6\%$)이 가장 많았다.
  - 퓨샷 CoT 설정에서는 $\text{Knowledge Errors}$ (지식 오류, $29.3\%$)가 크게 증가하였는데, 이는 제공된 예시에 과하게 의존(Overfitting)하여 발생한 것으로 해석된다.
- **오류 유형 (MLLM)**:
  - $\text{Emotion Recognition Errors}$ (정서 인식 오류)가 가장 빈번하게 발생하였다 ($\text{GPT-4V}: 30.1\%, \text{Gemini Pro Vision}: 31.3\%$).
  - 이어 $\text{Spatial Perception Errors}$ (공간 지각 오류)가 주요 오류 원인으로 나타났다.

## 🧠 Insights & Discussion

### 강점 및 한계

- **강점**: $\text{Gemini Pro}$는 전반적으로 $\text{GPT-3.5 Turbo}$ 수준의 강력한 상식 추론 능력을 갖추고 있으며, 특히 멀티모달 환경에서 시간적 맥락을 파악하는 능력이 뛰어나다.
- **한계**:
  - **복잡한 사회적 역학**: 인간의 정서적 반응이나 복잡한 사회적 상호작용을 예측하는 데 어려움이 있다.
  - **추상적 추론**: 수수께끼(Riddles)나 복잡한 시간 순서가 얽힌 추론에서 성능 저하가 뚜렷하다.
  - **시각적 정서 파악**: 이미지 속 인물의 감정 상태를 정확히 읽어내는 능력이 부족하여, 이를 기반으로 하는 상식 추론에서 오류가 잦다.

### 비판적 해석

본 논문은 Gemini가 GPT-4에 비해 상식 추론 능력이 부족하다는 초기 평가를 반박하며, 실제로는 경쟁력 있는 수준임을 입증하였다. 그러나 분석 결과, 모델들이 '정답'은 맞히더라도 '논리적 근거'는 제시하지 못하는 현상이 발견되었다. 이는 모델이 진정한 의미의 상식적 추론을 수행하는 것이 아니라, 훈련 데이터의 패턴을 통해 정답을 도출하는 '지식 암기' 방식에 의존하고 있을 가능성을 시사한다. 따라서 향후 연구에서는 정확도(Accuracy)뿐만 아니라 추론 과정의 정합성을 평가할 수 있는 더 정교한 메트릭이 도입되어야 한다.

## 📌 TL;DR

본 연구는 $\text{Gemini Pro}$와 $\text{Gemini Pro Vision}$의 상식 추론 능력을 12개 데이터셋을 통해 종합 분석하였다. 실험 결과, $\text{Gemini Pro}$는 $\text{GPT-3.5 Turbo}$와 대등한 수준의 능력을 갖추었으나 $\text{GPT-4 Turbo}$에는 미치지 못했다. 특히 시간적/사회적 추론과 이미지 내 정서 인식에서 한계를 보였다. 이 연구는 MLLM이 단순한 정답 도출을 넘어 논리적 근거를 일관되게 제시하는 능력을 향상시켜야 함을 시사하며, 향후 AGI로 나아가기 위한 핵심 과제로 상식 추론의 고도화를 제시한다.
