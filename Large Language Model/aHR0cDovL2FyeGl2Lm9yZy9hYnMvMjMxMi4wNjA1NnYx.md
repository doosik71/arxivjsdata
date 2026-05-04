# METAL: Metamorphic Testing Framework for Analyzing Large-Language Model Qualities

Sangwon Hyun, Mingyu Guo, M. Ali Babar (2023)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large-Language Models, LLMs)의 출력 품질을 체계적으로 분석하고 검증하는 문제에 집중한다. LLM은 내부 동작 방식이 공개되지 않은 Black-box 특성과 확률적(Probabilistic) 특성을 가지고 있어, 다양한 응용 분야에서 출력 결과의 품질을 보장하기 어렵다.

기존의 LLM 품질 평가 연구들은 주로 적대적 입력 텍스트(Adversarial input texts)를 생성하여 강건성(Robustness)이나 공정성(Fairness)과 같은 품질 속성(Quality Attributes, QAs)을 테스트해 왔다. 그러나 이러한 기존 접근 방식은 다음과 같은 세 가지 한계점을 가진다.
1. **제한된 커버리지**: 평가 대상이 되는 QA와 LLM 태스크의 범위가 좁으며, 새로운 시나리오로 확장하기 어렵다.
2. **단일 지표 의존성**: 테스트의 효과성을 측정하기 위해 오직 Attack Success Rate (ASR)라는 단일 지표만을 사용하며, 이는 텍스트 섭동(Perturbation)의 질적 수준을 반영하지 못한다.
3. **데이터 의존성**: 정답 라벨이 포함된 테스트 데이터베이스에 의존하는 경우가 많아 비용이 많이 들고 확장성이 떨어진다.

따라서 본 논문의 목표는 Metamorphic Testing (MT) 기법을 적용하여 다양한 QA와 태스크를 체계적으로 테스트하고, 섭동의 품질을 고려한 새로운 측정 지표를 통해 LLM의 품질 리스크를 정확하게 분석할 수 있는 METAL 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Metamorphic Relations (MRs)**를 정의하여 LLM의 품질 평가를 모듈화하고 자동화하는 것이다. MR은 원본 입력과 섭동이 가해진 입력 사이의 관계를 정의함으로써, 정답 라벨(Oracle) 없이도 모델의 일관성과 품질을 검증할 수 있게 한다.

주요 기여 사항은 다음과 같다.
- **MR 템플릿 기반 자동 생성**: Robustness, Fairness, Non-determinism, Efficiency의 4가지 핵심 QA와 6가지 주요 LLM 태스크를 포괄하는 5가지 MR 템플릿을 정의하고, 이를 통해 수백 개의 MR을 자동으로 생성하는 프로세스를 구축하였다.
- **새로운 효과성 지표 제안**: 단순히 ASR만을 사용하는 것이 아니라, 텍스트의 의미적/구조적 유사성을 결합한 새로운 지표를 도입하여 MR의 유효성을 정밀하게 측정하였다.
- **LLM 기반 상호 검증**: LLM 스스로 또는 다른 LLM을 통해 섭동 텍스트를 생성하는 self/cross-examination 방식의 가능성을 실험적으로 검증하였다.

## 📎 Related Works

논문은 관련 연구를 세 가지 범주로 나누어 설명하며 기존 방식의 한계를 지적한다.

1. **Adversarial test datasets**: AdvGLUE, AdversarialSQuAD 등이 있으며, 주로 모델의 정확도(Accuracy)와 강건성을 측정한다. 하지만 이러한 데이터셋은 사람이 직접 제작하는 경우가 많아 양이 제한적이며, LLM의 광범위한 활용 시나리오를 모두 커버하기 어렵다.
2. **Adversarial attack generators**: 사전 학습된 NLP 모델을 이용해 텍스트를 변조하여 강건성을 테스트하는 방식이다. 그러나 대부분 특정 유형의 섭동에만 집중되어 있어 확장성이 낮으며, 여전히 ASR 지표에만 의존하여 섭동 자체의 품질을 고려하지 않는다.
3. **QA analysis on NLP models**: HELM과 같은 총체적 평가 방법론이 존재하지만, 이는 주로 기존 벤치마크를 기반으로 한 서베이 성격이 강하며, 실행 가능한(Executable) 프레임워크 형태로는 제공되지 않는 경우가 많다.

METAL 프레임워크는 라벨링되지 않은 무제한의 입력 소스를 사용할 수 있다는 점과, 다양한 QA와 태스크를 모듈화된 MRT(MR Template)를 통해 한 번에 평가할 수 있다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조
METAL 프레임워크는 크게 **Execution(실행)** 모듈과 **Evaluation(평가)** 모듈로 구성된다.
- **Execution 모듈**: 입력 텍스트에 대해 정의된 섭동 함수나 LLM 기반 프롬프트를 사용하여 섭동 텍스트를 생성하고, 타겟 LLM에 입력하여 그 출력값을 수집(Log)한다.
- **Evaluation 모듈**: 수집된 결과물을 MRT에 기반하여 검증하며, MR 만족 여부를 이진 값(1 또는 0)으로 반환한다.

### 2. Metamorphic Relation Templates (MRTs)
논문은 QA와 태스크의 상관관계를 분석하여 5가지 MRT를 정의하였다. 여기서 $M$은 LLM, $input$은 입력 텍스트, $prompt$는 태스크 지시문, $P$는 섭동 함수, $D$는 거리 함수, $\alpha$는 임계값을 의미한다.

- **Equivalence MRT**: 원본과 섭동 입력의 출력이 동일해야 함 (주로 분류 태스크의 Robustness 평가)
  $$\text{EquivalenceMRT} \implies M(input, prompt) = M(P(input), prompt)$$
- **Discrepancy MRT**: 의미가 변하는 섭동이 가해졌을 때 출력이 달라져야 함
  $$\text{DiscrepancyMRT} \implies M(input, prompt) \neq M(P(input), prompt)$$
- **Set Equivalence MRT**: 여러 섭동(예: 다양한 인구통계학적 그룹)에 대해 출력이 모두 동일해야 함 (Fairness, Non-determinism 평가)
  $$\text{Set EquivalenceMRT} \implies \forall o \in O: M(input, prompt) = o$$
- **Distance MRT**: 출력 간의 거리가 일정 임계값 $\alpha$ 이내여야 함 (생성 태스크의 Robustness 및 Efficiency 평가)
  $$\text{Distance MRT} \implies D(M(input, prompt), M(P(input), prompt)) \leq \alpha$$
- **Set Distance MRT**: 동일 입력에 대해 반복 생성된 출력들의 변동성이 $\alpha$ 이내여야 함 (생성 태스크의 Non-determinism 평가)
  $$\text{Set Distance MRT} \implies \forall o \in O: D(M(input, prompt), o) \leq \alpha$$

### 3. 섭동 함수 및 생성 방법
총 13가지의 섭동 함수($P$)를 구현하였으며, 이를 **의미 유지(Semantic-preserving)**와 **의미 변경(Semantic-altering)**으로 구분하였다.
- **Character-level**: 글자 교체, 삭제, 추가, 셔플, l33t 포맷 변환, 공백 추가 등.
- **Word-level**: 유의어 교체(의미 유지), 반의어 교체(의미 변경), 랜덤 단어 추가 등.
- **Sentence-level**: 문장 제거, 문장 교체, 인구통계학적 그룹 지정(Fairness 평가용) 등.

또한, 함수 기반 생성 외에도 LLM에 특정 프롬프트를 주어 섭동 텍스트를 생성하게 하는 방식을 함께 사용하였다.

## 📊 Results

### 1. 실험 설정
- **대상 모델**: Google PaLM, OpenAI GPT-3.5-Turbo, Meta Llama-2-7b-chat.
- **대상 태스크**: 분류(감성 분석, 뉴스 분류, 독성 탐지) 및 생성(Q&A, 텍스트 요약, 정보 검색).
- **데이터셋**: 웹 소스에서 수집한 900개의 입력 텍스트 (15 ~ 4K 토큰 길이).

### 2. 주요 결과 분석
- **품질 속성 평가 (RQ1)**:
    - **Robustness**: 분류 태스크보다 생성 태스크에서 ASR이 높게 나타나 더 취약함을 보였다. Llama2는 IR을 제외한 모든 태스크에서 가장 높은 ASR을 기록하여 가장 취약했다.
    - **Fairness**: Llama2가 모든 태스크에서 가장 높은 ASR을 보여 공정성 리스크가 가장 컸으며, PaLM과 ChatGPT는 상대적으로 안정적이었다.
    - **Non-determinism**: PaLM은 IR 태스크를 제외하고는 동일 입력에 대해 거의 동일한 출력을 생성하여 매우 낮은 변동성을 보였다.
    - **Efficiency**: PaLM이 원본과 섭동 입력 간의 추론 시간 차이가 가장 적어 가장 안정적인 효율성을 보였다.
- **MR의 효과성 측정 (RQ2)**:
    - 단순 ASR의 한계를 극복하기 위해 $EFM = M\text{-}ASR \times PerturbQuality$ 라는 지표를 도입하였다. 여기서 $PerturbQuality$는 원본과 섭동 텍스트 간의 의미적 유사도를 측정하여, 너무 망가진 텍스트로 인해 ASR이 높아지는 현상을 방지한다.
    - 분석 결과, 독성 탐지(TD)와 뉴스 분류(NC)에서는 `ConvertToL33tFormat` 섭동이 가장 효과적이었으며, Q&A와 IR에서는 단어 수준의 유의어 교체가 더 효과적이었다.
- **상호 검증 결과 (RQ3)**:
    - ChatGPT가 생성한 MR이 다른 모든 모델을 테스트할 때 가장 높은 $EFM$을 기록하여, LLM을 이용한 섭동 생성의 유효성이 입증되었다.

## 🧠 Insights & Discussion

### 강점
- **체계적 프레임워크**: 단순한 적대적 공격을 넘어, MRT라는 정형화된 템플릿을 통해 다양한 QA를 통합적으로 평가할 수 있는 체계를 구축하였다.
- **정답 라벨 불필요**: MT 기법을 적용함으로써 고비용의 라벨링 작업 없이도 대규모 데이터에 대해 모델의 일관성을 검증할 수 있다.
- **현실적인 지표 도입**: $EFM$ 지표를 통해 '공격의 성공률'과 '섭동의 질' 사이의 트레이드오프를 정량적으로 분석하여, 실제로 어떤 섭동이 모델의 취약점을 효과적으로 드러내는지 찾아냈다.

### 한계 및 비판적 해석
- **LLM 섭동의 품질 저하**: LLM을 이용해 섭동을 생성할 때, 사람이 이해할 수 없는 텍스트가 생성되는 경우가 발생하였다. 비록 $EFM$으로 이를 필터링하려 했으나, 생성 프로세스 자체의 정밀도를 높이는 추가 연구가 필요하다.
- **프롬프트 의존성**: 실험에 사용된 프롬프트가 모델의 결과에 큰 영향을 미칠 수 있으나, 본 논문에서는 제한된 수의 프롬프트만을 사용하였다. 프롬프트 자체에 대한 섭동(Prompt Perturbation)까지 확장한다면 더 포괄적인 분석이 가능할 것이다.
- **모델 규모의 차이**: Llama2의 경우 7B 모델을 사용하였는데, 이는 PaLM이나 GPT-3.5-Turbo와 같은 거대 모델과 직접 비교하기에는 파라미터 규모의 차이가 커서, Llama2의 낮은 성능이 단순히 모델 크기 때문인지 아니면 구조적 취약점 때문인지 명확하지 않다.

## 📌 TL;DR

본 논문은 LLM의 품질 속성(강건성, 공정성, 비결정성, 효율성)을 체계적으로 평가하기 위한 **METAL 프레임워크**를 제안한다. 이 프레임워크는 5가지 **Metamorphic Relation Templates (MRTs)**를 통해 정답 라벨 없이도 다양한 태스크에 대해 자동으로 테스트 케이스를 생성하고 검증한다. 특히, 섭동의 질을 반영한 $EFM$ 지표를 통해 단순 ASR의 맹점을 해결하였으며, ChatGPT가 다른 LLM의 취약점을 찾는 효과적인 MR 생성기로 활용될 수 있음을 보였다. 이 연구는 향후 미세 조정(Fine-tuning)된 도메인 특화 LLM의 품질을 검증하는 오픈 플랫폼으로 활용될 가능성이 높다.