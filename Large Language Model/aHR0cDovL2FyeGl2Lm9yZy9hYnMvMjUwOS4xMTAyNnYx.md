# Rethinking Human Preference Evaluation of LLM Rationales

Ziang Li, Manasi Ganti, Zixian Ma, Helena Vasconcelos, Qijia He, Ranjay Krishna (2025)

## 🧩 Problem to Solve

본 연구는 대규모 언어 모델(LLM)이 생성하는 자연어 rationale(추론 과정 및 근거)의 평가 방식에 내재된 한계를 해결하고자 한다. LLM이 생성하는 free-form 형태의 rationale은 복잡한 추론 작업의 성능을 향상시키고 모델의 내부 추론 과정을 인간이 이해할 수 있게 돕는 중요한 역할을 한다. 하지만 기존의 rationale 평가 방식은 주로 인간이나 LLM 판별자가 두 응답 중 하나를 선택하는 이진 선호도 판정(binary preference judgment)에 의존해 왔다.

이러한 이진 방식의 평가는 다음과 같은 치명적인 문제점을 가진다. 첫째, 인간의 선호도가 구체적으로 어떤 요소에 의해 결정되는지 알 수 없는 불투명성(opacity)이 존재한다. 둘째, 단순히 '승리' 또는 '패배'로 구분되는 거친 입도(coarse-grained)의 결과만 제공하므로, 특정 rationale이 왜 더 우수한지에 대한 세밀한 통찰을 제공하지 못한다. 따라서 본 논문은 좋은 rationale을 정의하는 속성(attribute)을 규명하고, 이러한 세부 속성이 인간의 선호도를 어떻게 설명하며, 이를 통해 더 정밀한 모델 평가 체계를 구축할 수 있는지를 탐구하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단순한 이진 선호도 평가를 넘어, 세부 속성 기반의 분석 프레임워크를 통해 LLM rationale 평가의 투명성과 정밀도를 높인 점에 있다. 중심적인 설계 아이디어는 다음과 같다.

1. **세부 속성 체계 구축**: 기존 문헌을 바탕으로 rationale의 품질을 결정하는 12가지 핵심 속성을 정의하고, 이를 측정하기 위한 다각적 방법론(자동화 지표, LLM 판별자, 인간 주석)을 제안한다.
2. **선호도 결정 요인 분석**: SHAP(SHapley Additive exPlanations) 분석과 LightGBM 모델을 결합하여, 어떤 세부 속성이 인간의 선호도 판정에 가장 큰 영향을 미치는지 정량적으로 분석한다.
3. **속성별 ELO rating 도입**: 기존의 통합 ELO 점수 대신, 개별 속성별로 ELO 점수를 산출하는 프레임워크를 도입하여 모델 간의 강점과 약점을 다각도에서 비교 분석한다.

## 📎 Related Works

본 논문은 크게 두 가지 관련 연구 흐름을 다룬다.

첫째, **인간 선호도 평가(Human Preference Evaluation)**이다. Chatbot Arena와 같은 플랫폼은 인간의 선호도를 바탕으로 LLM의 성능을 랭킹화하며, 이는 모델을 인간의 가치에 정렬(alignment)시키는 데 널리 사용되어 왔다. 그러나 이러한 방식은 결과론적인 순위만 제공할 뿐, 그 이유에 대한 설명이 부족하다는 한계가 있다.

둘째, **Rationale 평가(Rationale Evaluation)**이다. 기존 연구들은 consistency, faithfulness, clarity 등 개별적인 속성을 측정하려 시도했다. 특히 ROSCOE와 같은 도구는 단계별 추론(step-by-step reasoning)을 평가하기 위한 자동화된 지표 세트를 제공한다. 본 연구는 이러한 개별 속성 측정 도구들을 통합하여 인간의 전반적인 '선호도'라는 모호한 신호와 연결하려 한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. Rationale 속성 정의 및 측정
연구진은 고품질 rationale을 정의하기 위해 12가지 속성을 선정하였다. 주요 속성은 다음과 같다.
- **Faithfulness**: rationale이 모델의 실제 계산 과정이나 제공된 증거에 기반하는가.
- **Plausibility**: 진위 여부와 관계없이 논리적으로 그럴듯하게 들리는가.
- **Correctness**: 모든 단계와 최종 답변이 객관적으로 정확한가.
- **Self-Consistency**: rationale 내부에서 논리적 모순이 없는가.
- 기타 속성으로 Hallucination, Repetition, Informativeness, Source Consistency, Grammar, Arithmetic Accuracy, Conciseness, Completeness 등이 포함된다.

이 속성들은 세 가지 방식으로 측정된다.
- **Automated Heuristics**: ROSCOE 지표를 사용하여 정량적으로 측정한다.
- **LLM Judges**: GPT-4o, Gemini 2.5-Flash (0~1 척도) 및 OLMo 32B (0~10 척도)를 사용하여 평가한다.
- **Human Annotations**: 전문가 3인이 직접 샘플을 주석 처리한다.

### 2. 인간 선호도 설명 모델링 (SHAP 분석)
인간의 선호도가 어떤 속성에 의해 결정되는지 분석하기 위해 다음과 같은 절차를 거친다.
1. **데이터셋**: MT-Bench 및 Chatbot Arena에서 수학적/논리적 질문을 필터링하여 사용한다.
2. **예측 모델 학습**: 입력 변수 $X$를 12가지 세부 속성 점수로, 타겟 변수 $y$를 인간의 선호도 결과(chosen/rejected)로 설정하여 LightGBM 모델을 학습시킨다.
3. **SHAP 분석**: 학습된 모델에 SHAP를 적용하여 각 속성이 예측 결과에 기여하는 정도를 산출함으로써, 인간의 선호도를 결정짓는 지배적인 속성을 식별한다.

### 3. 속성별 ELO rating 산출
전통적인 ELO rating은 두 모델의 승패 기록을 바탕으로 단일 점수를 부여한다. 본 연구에서는 이를 확장하여 **속성별 ELO 점수**를 계산한다.
- 각 속성에 대해 LLM 판별자들이 부여한 점수를 바탕으로 승패를 결정한다.
- 이 승패 기록을 사용하여 속성별 ELO 점수를 독립적으로 계산하고, 이를 통해 모델별 '능력 프로필(Capability Profile)'을 구성한다.

## 📊 Results

### 1. 인간 선호도의 결정 요인 (Q2 결과)
SHAP 분석 결과, 모델과 데이터셋에 관계없이 인간의 선호도에 가장 큰 영향을 미치는 핵심 속성은 **Correctness(정확성), Plausibility(그럴듯함), Completeness(완전성)** 순으로 나타났다. 이는 인간 평가자가 rationale을 평가할 때 사실적 정확성과 논리적 완결성을 가장 중요하게 고려함을 시사한다.

### 2. 모델별 세부 성능 비교 (Q3 결과)
속성별 ELO rating을 통해 모델을 재평가한 결과, 통합 선호도 순위에서는 보이지 않던 세밀한 차이가 드러났다.
- **전반적 상위 모델**: GPT-4, GPT-3.5-Turbo, Claude-v1이 최상위권을 유지했다.
- **모델별 특이점**:
    - **Claude-v1**: 전반적인 성능은 높으나, Repetition(반복성) 속성에서 낮은 점수를 기록하며 동일한 내용을 반복하는 경향이 확인되었다.
    - **GPT-3.5-Turbo**: 놀랍게도 Arithmetic Accuracy(산술 정확도)와 Self-Consistency(자기 일관성) 항목에서는 GPT-4보다 더 높은 성능을 보이는 경우가 발견되었다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구는 모호한 '선호도'라는 개념을 분해 가능한 '속성'의 집합으로 치환함으로써, LLM 평가의 해석 가능성을 획기적으로 높였다. 특히 통합 점수로는 알 수 없었던 모델 간의 trade-off(예: GPT-4의 종합 성능 vs GPT-3.5의 특정 산술 능력)를 밝혀낸 점이 고무적이다.

### 한계 및 비판적 해석
1. **LLM 판별자의 신뢰성**: 평가 과정에서 LLM-as-a-judge 방식에 의존했다. 논문에서도 언급되었듯이, GPT-4o가 정답이 틀린 rationale에 만점(1.0)을 주는 사례가 발견되는 등 판별 모델 자체의 factual error와 편향 문제가 존재한다.
2. **데이터의 범위**: 분석 대상이 수학 및 논리 추론 작업에 국한되어 있다. 창의적 글쓰기나 상식 추론과 같은 다른 도메인에서도 동일한 속성 중요도가 나타날지는 미지수이다.
3. **인간 주석의 규모**: 인간 주석 작업이 논문의 공저자 3인에 의해 수행되었다. 전문가 수준의 평가라는 장점은 있으나, 더 다양하고 많은 수의 일반 사용자 데이터를 통해 검증할 필요가 있다.

## 📌 TL;DR

본 논문은 LLM rationale 평가 시 사용하는 단순한 이진 선호도(Win/Loss) 방식의 한계를 지적하고, 12가지 세부 속성 기반의 정밀 평가 프레임워크를 제안한다. SHAP 분석을 통해 인간의 선호도가 주로 **정확성, 그럴듯함, 완전성**에 의해 결정됨을 밝혔으며, 속성별 ELO rating을 통해 GPT-4와 GPT-3.5-Turbo 간의 세부적인 성능 역전 현상 등을 발견했다. 이 연구는 향후 LLM 평가가 단일 지표에서 벗어나 다각적인 속성 분석 체계로 나아가야 함을 시사하며, 특히 특정 속성을 타겟팅한 모델 미세 조정(Fine-tuning)의 가능성을 열어주었다.