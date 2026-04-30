# Measuring and Narrowing the Compositionality Gap in Language Models

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, Mike Lewis (2023)

## 🧩 Problem to Solve

본 연구는 언어 모델(Language Models, LMs)이 개별적인 하위 문제의 답을 조합하여 전체 문제의 해답을 도출하는 **compositional reasoning**(구성적 추론) 능력의 한계를 분석한다. 

많은 언어 모델이 방대한 데이터를 통해 개별적인 사실들을 암기하고 있지만, 이를 논리적으로 조합하여 이전에 본 적 없는 새로운 지식을 추론하는 능력은 별개의 문제이다. 특히, 모델의 규모가 커짐에 따라 단순 암기 능력은 향상되지만, 이러한 사실들을 결합하는 추론 능력이 동일한 속도로 향상되는지는 불분명하다.

따라서 본 논문의 목표는 다음과 같다:
1. 모델이 하위 질문에는 답할 수 있음에도 전체 질문에는 답하지 못하는 비율인 **compositionality gap**(구성성 격차)을 정량적으로 측정한다.
2. 모델 규모의 확장(scaling)이 이 격차를 줄이는지 확인한다.
3. 이 격차를 줄이기 위한 효과적인 프롬프팅 기법을 제안하고 검증한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Compositionality Gap의 정의 및 측정**: 모델이 구성 요소가 되는 하위 질문들을 모두 맞혔음에도 불구하고, 최종적인 구성 질문(compositional question)에 실패한 비율을 'compositionality gap'으로 정의하고 이를 측정하는 방법론을 제시하였다.
2. **Scaling Law의 한계 발견**: GPT-3 모델 제품군을 분석한 결과, 모델의 크기가 커질수록 단일 홉(single-hop) 질문에 대한 성능은 빠르게 향상되지만, 다중 홉(multi-hop) 질문 성능의 향상 속도는 그보다 느려 결과적으로 **compositionality gap이 줄어들지 않고 일정하게 유지됨**을 발견하였다.
3. **Self-Ask 프롬프팅 제안**: 기존의 Chain of Thought(CoT)보다 진보된 **Self-Ask** 기법을 제안하였다. 이는 모델이 스스로 하위 질문을 생성하고 그에 답한 뒤 최종 답을 내는 구조화된 방식을 취한다.
4. **외부 검색 엔진과의 결합**: Self-Ask의 구조적 특성을 활용하여, 하위 질문에 대한 답변을 모델 내부 지식이 아닌 외부 검색 엔진에서 가져오게 함으로써 성능을 더욱 극대화하는 **Self-Ask + Search Engine(SA+SE)** 구조를 구현하였다.
5. **새로운 데이터셋 구축**: 추론 능력을 정밀하게 측정하기 위해 자동으로 생성된 **Compositional Celebrities (CC)** 데이터셋과 수동으로 구축된 **Bamboogle** 데이터셋을 제안하였다.

## 📎 Related Works

본 논문은 복잡한 문제를 단순한 하위 문제로 분해하는 기존의 접근 방식들을 다룬다.

- **Chain of Thought (CoT) 및 Scratchpad**: 모델이 최종 답을 내기 전 중간 추론 과정을 생성하게 하여 성능을 높이는 방식이다. 하지만 CoT는 추론 과정이 비구조적이며, 모델이 하위 문제를 명확히 정의하지 않은 채 답변을 생성하는 경향이 있다.
- **Least-to-Most Prompting**: 복잡한 문제를 하위 문제로 분해하여 해결하는 방식이나, 이는 여러 번의 forward pass와 서로 다른 프롬프트가 필요하여 연산 비용이 높다는 한계가 있다.
- **Retrieval-Augmented Generation (RAG)**: 외부 지식을 검색하여 답변에 활용하는 방식이다. 기존 연구들은 주로 모델의 파인튜닝이나 특수한 쿼리 언어를 필요로 했으나, 본 논문의 Self-Ask는 별도의 튜닝 없이 프롬프팅만으로 검색 엔진을 통합한다.

## 🛠️ Methodology

### 1. Compositionality Gap의 측정
연구진은 다음과 같은 수식적 개념으로 격차를 측정한다.

$$\text{Compositionality Gap} = \frac{\text{Count}(\text{Sub-questions Correct} \land \text{Compositional Question Wrong})}{\text{Count}(\text{Sub-questions Correct})}$$

즉, 하위 지식을 모두 알고 있음에도 불구하고 이를 조합하지 못한 비율을 의미하며, 이는 순수하게 '조합 능력'의 결여를 측정하기 위함이다.

### 2. Self-Ask 프롬프팅
Self-Ask는 모델이 다음과 같은 단계로 사고하도록 유도하는 프롬프팅 기법이다.
- **단계 1**: 복잡한 질문이 주어지면 "Are follow up questions needed here: Yes"라고 판단한다.
- **단계 2**: "Follow up: [하위 질문]"을 생성하고, 곧바로 "Intermediate answer: [하위 질문에 대한 답]"을 작성한다.
- **단계 3**: 필요한 모든 하위 질문과 답이 도출될 때까지 이를 반복한다.
- **단계 4**: "So the final answer is: [최종 답]"을 출력한다.

이 방식은 추론 과정(decomposition)과 답변 과정(answering)을 명확히 분리하여, CoT보다 더 구조적인 추론을 가능하게 한다.

### 3. Self-Ask + Search Engine (SA+SE)
Self-Ask의 구조를 활용하여, 모델이 생성한 `Follow up` 질문을 외부 검색 엔진 API에 전달한다.
- 모델이 $\text{Follow up: } Q_{sub}$를 생성하면, 생성 프로세스를 일시 중단한다.
- $Q_{sub}$를 검색 엔진에 쿼리로 보내고, 반환된 검색 결과를 $\text{Intermediate answer: } A_{sub}$ 형태로 프롬프트에 삽입한다.
- 모델은 다시 이 정보를 바탕으로 다음 하위 질문을 생성하거나 최종 답을 도출한다.

## 📊 Results

### 1. 실험 설정
- **모델**: GPT-3 및 InstructGPT 제품군 (Ada, Babbage, Curie, Davinci)
- **데이터셋**: 
    - **CC**: 8.6k개의 2-hop 질문 (자동 생성)
    - **Bamboogle**: 125개의 2-hop 질문 (수동 생성, 검색 엔진이 오답을 낼 정도로 까다로운 질문들)
    - **2WikiMultiHopQA**, **Musique**: 기존 오픈 도메인 다중 홉 QA 데이터셋.
- **지표**: Exact Match (EM), F1, Cover-EM.

### 2. 주요 결과
- **Scaling의 한계**: 모델 크기가 증가함에 따라 하위 질문 정답률은 크게 상승하지만, 전체 질문 정답률은 그만큼 상승하지 않는다. 결과적으로 **compositionality gap은 모델 크기에 관계없이 약 40% 수준에서 유지**되는 경향을 보였다.
- **프롬프팅 성능 비교 (Bamboogle 기준)**:
    - Direct Prompting: 17.6%
    - Chain of Thought: 46.4%
    - **Self-Ask**: 57.6%
    - **Self-Ask + Search**: 60.0%
- **추론 효율성**: Self-Ask는 Least-to-Most 방식보다 성능이 비슷하거나 우수하면서도, 단일 forward pass 내에서 해결 가능하여 실행 속도가 약 30% 더 빨랐다.

## 🧠 Insights & Discussion

### 강점 및 발견
- **지식의 양 $\neq$ 추론 능력**: 모델 규모를 키우는 것이 사실적 지식의 암기량은 늘려주지만, 그 지식들을 조합하는 '추론 알고리즘' 자체를 개선하는 것은 아님을 시사한다.
- **확신도(Confidence)와의 상관관계**: 모델이 하위 질문의 답에 대해 가지는 Perplexity가 낮을수록(즉, 더 확신할수록) 이를 조합하여 정답을 맞힐 확률이 비약적으로 높아짐을 발견하였다. 이는 단순히 '맞혔느냐'보다 '얼마나 확신하며 맞혔느냐'가 구성적 추론의 핵심임을 보여준다.
- **구조적 프롬프팅의 효과**: Self-Ask의 엄격한 스캐폴딩(scaffolding)은 모델이 중간 단계에서 길을 잃지 않게 하며, 특히 Bamboogle과 같이 변동성이 큰 데이터셋에서 CoT보다 강력한 성능을 보였다.

### 한계 및 비판적 해석
- **데이터셋의 범위**: 대부분의 실험이 영어 2-hop 질문에 집중되어 있어, 더 복잡한 n-hop 추론이나 타 언어에서의 일반화 가능성은 추가 검증이 필요하다.
- **모델 크기의 제한**: 175B 이상의 초거대 모델(예: GPT-4)에서는 이 gap이 줄어드는지 확인할 수 없었다. 다만, 저자들은 GPT-4에서도 여전히 gap이 존재함을 관찰하였다고 언급하였다.

## 📌 TL;DR

본 논문은 언어 모델이 개별 사실은 알면서도 이를 조합해 답을 내지 못하는 **'compositionality gap'**을 정의하고, 이것이 모델의 크기를 키우는 것(Scaling)만으로는 해결되지 않음을 증명하였다. 이를 해결하기 위해 질문을 스스로 분해하고 답변하는 **Self-Ask** 프롬프팅을 제안하였으며, 여기에 **외부 검색 엔진**을 결합하여 추론 성능을 극대화하였다. 이 연구는 LLM의 성능 향상을 위해 단순한 파라미터 확장보다 추론 프로세스의 구조화가 더 중요함을 시사한다.