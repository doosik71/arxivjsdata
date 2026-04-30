# Can GPT Redefine Medical Understanding? Evaluating GPT on Biomedical Machine Reading Comprehension

Shubham Vatsal, Ayush Singh (2024)

## 🧩 Problem to Solve

본 연구는 생의학(Biomedical) 도메인에서의 문맥적 기계 독해(Contextual Machine Reading Comprehension, MRC) 작업에 대해 거대 언어 모델(LLM), 특히 GPT의 성능을 심층적으로 평가하고 최적의 프롬프팅 전략을 찾는 것을 목표로 한다.

문맥적 MRC는 모델이 외부 지식에 의존하지 않고 오직 주어진 문맥(Context)만을 사용하여 질문에 답해야 하는 작업이다. 생의학 도메인은 복잡하고 전문적인 어휘(In-domain vocabulary)가 많고, 전역적 지식(Global knowledge)에 대한 의존도가 높아 일반 도메인보다 난이도가 훨씬 높다. 기존의 지도 학습(Supervised) 모델들은 인간의 정답 수준(Gold standard)에 도달하는 데 어려움을 겪어 왔으며, 최신 LLM들이 생의학 작업에서 뛰어난 성능을 보이고 있음에도 불구하고, 특히 '문맥적 MRC' 설정에서의 성능 평가는 충분히 이루어지지 않은 상태였다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **생의학 문맥적 MRC 벤치마크 평가**: 네 가지 표준 생의학 MRC 데이터셋(ProcessBank, BioMRC, MASH-QA, CliCR)을 통해 GPT의 성능을 평가하고, 일부 데이터셋에서 새로운 State-of-the-Art(SoTA) 결과를 달성하였다.
2.  **Implicit RAG 프롬프팅 기법 제안**: 전통적인 Retrieval Augmented Generation(RAG)과 달리, 벡터 데이터베이스(Vector Database)를 통한 외부 임베딩 검색 과정 없이, LLM이 스스로 문맥 내에서 질문과 관련된 섹션을 먼저 추출하고 이를 바탕으로 답변하게 하는 'Implicit RAG' 기법을 제안하였다.
3.  **정성적 인간 평가 수행**: 자동화된 평가 지표의 한계를 극복하기 위해, 생의학 전문가를 통한 정성적 선호도 분석을 수행하여 Implicit RAG의 출력물이 실제 인간의 판단과 일치함을 확인하였다.

## 📎 Related Works

기존의 MRC 연구는 Cloze-style, Multiple-choice, Extractive, Generative QA 등으로 다양하게 발전해 왔다. 최근 LLM의 성능을 극대화하기 위해 Chain-of-Thought(CoT)나 Analogical Reasoning(AR)과 같은 프롬프팅 기법이 도입되었다.

특히, 방대한 문맥을 처리하기 위해 RAG(Retrieval Augmented Generation)가 널리 사용되고 있다. 전통적인 RAG는 전체 말뭉치(Corpus)를 임베딩하여 벡터 DB에 저장하고, 쿼리와의 시맨틱 유사도(Semantic similarity)를 기반으로 관련 청크(Chunk)를 검색한다. 그러나 이러한 방식은 임베딩 생성 및 DB 저장이라는 추가적인 연산 비용과 인프라 overhead가 발생한다. 본 논문은 이러한 외부 시스템 없이 LLM의 추론 능력만으로 검색 기능을 수행하는 Implicit RAG를 통해 차별성을 둔다.

## 🛠️ Methodology

본 연구에서는 GPT-4(32k context window 버전)를 사용하였으며, Zero-shot 설정에서 다음 네 가지 프롬프팅 기법을 비교 분석하였다.

### 1. Prompting Strategies

-   **Basic Prompting**: 모델에게 특정 전문가 역할(예: 생물학자, 의료 전문가)을 부여하고, 주어진 문맥과 질문을 제공하여 직접적으로 답을 구하는 가장 단순한 형태이다.
-   **Chain-of-Thought (CoT)**: Basic 프롬프트에 "Think step by step"이라는 문구를 추가하여, 모델이 최종 답안을 내기 전 중간 추론 과정을 거치도록 유도한다.
-   **Analogical Reasoning (AR)**: 모델의 전역 지식에 의존하는 기존 AR과 달리, 주어진 문맥 내에서 유사한 QA 쌍을 먼저 생성하게 한 뒤, 이를 참고하여 원래의 질문에 답하게 하는 변형된 방식을 사용한다.
-   **Implicit RAG**: 본 논문에서 제안한 기법으로, 다음과 같은 2단계 절차를 거친다.
    1.  **추출 단계**: 모델이 주어진 문맥에서 질문에 답하는 데 가장 도움이 될 만한 $N$개의 관련 섹션이나 텍스트 추출물을 먼저 식별한다. 이때 각 섹션의 길이는 지정된 단어 수 범위(예: 50~200단어) 내로 제한한다.
    2.  **답변 단계**: 위에서 스스로 추출한 텍스트만을 활용하여 최종 답변을 도출한다.

### 2. Implicit RAG의 작동 원리 및 설정

Implicit RAG는 벡터 DB 없이 LLM 내부의 주의 집중(Attention) 메커니즘을 활용하여 검색을 수행한다. 하이퍼파라미터로는 추출할 섹션의 개수(`number_of_sections`)와 각 섹션의 길이 제한(`lower_limit_length`, `upper_limit_length`)이 사용된다. 예를 들어 CliCR 데이터셋의 경우 3개의 섹션을 추출하도록 설정하였다.

## 📊 Results

### 1. 실험 설정
-   **모델**: GPT-4 (32k context window), Temperature=0, Frequency/Presence penalty=0.
-   **데이터셋**: ProcessBank, BioMRC, MASH-QA, CliCR (각기 다른 통계적 특성과 문제 유형을 가짐).

### 2. 데이터셋별 결과

-   **ProcessBank**: 모든 프롬프팅 기법이 기존 지도 학습 모델들을 압도하며 새로운 SoTA를 달성하였다. 특히 **Implicit RAG**가 0.97의 정확도로 가장 높은 성능을 보였다. 이는 전체 문맥의 광범위한 분석이 필요한 시간적/참-거짓 질문이 많기 때문으로 분석된다.
-   **BioMRC**: **Basic** 프롬프팅이 가장 우수(0.87)했으며, Implicit RAG가 그 뒤를 이었다. 다만, 데이터셋 자체의 구조적 결함(엔티티 ID 매핑 불일치 등)으로 인해 생성 모델인 GPT가 정답과 의미적으로 동일함에도 오답 처리되는 경우가 많았음을 지적하였다.
-   **MASH-QA**: **Basic** 프롬프팅이 가장 좋은 성능을 보였으며, Implicit RAG는 2위를 기록하였다. 이 데이터셋은 정답이 문맥 내 여러 곳에 흩어져 있는 Long-span 형태가 많아, 섹션을 나누어 추출하는 Implicit RAG가 오히려 문맥적 연속성을 해치는 경향이 발견되었다.
-   **CliCR**: **Implicit RAG**와 **AR**이 가장 높은 F1 스코어를 기록하였다. 특히 GPT는 기존 SoTA 모델을 능가했을 뿐만 아니라, 인간 전문가(Human Expert) 수준의 성능에 근접하였으며 인간 초보자(Human Novice)의 성능을 뛰어넘었다.

### 3. 정성적 분석 (Qualitative Analysis)
Implicit RAG가 추출한 섹션들이 실제로 질문과 관련이 있는지를 분석한 결과, ProcessBank와 BioMRC에서는 매우 높은 비율로 관련 섹션을 정확히 추출해 냈으며, CliCR에서도 81%의 높은 유효 추출률을 보였다.

## 🧠 Insights & Discussion

### 1. 주요 강점 및 발견
-   **Zero-shot의 위력**: 방대한 데이터를 학습한 LLM은 특정 도메인의 지도 학습 모델보다 Zero-shot 설정에서도 더 뛰어난 성능을 보일 수 있음을 입증하였다.
-   **Implicit RAG의 효용성**: 문맥의 길이가 길수록(예: CliCR), 그리고 여러 섹션의 정보를 조합해야 하는 'Bridging'이나 'Tracking' 능력이 필요할수록 Implicit RAG가 효과적이다.
-   **확장 가능성**: 32k 토큰 제한을 넘어서는 초거대 문맥의 경우, 문맥을 청크로 나누어 Implicit RAG를 반복 호출함으로써 관련 섹션들을 먼저 수집한 뒤 최종 답변을 내는 방식으로 확장 적용이 가능하다.

### 2. 한계 및 비판적 해석
-   **평가 지표의 한계**: Exact Match(EM)나 F1 스코어 같은 엄격한 매칭 지표는 생성 모델인 GPT의 특성을 반영하지 못한다. 의미적으로는 정답이지만 표기법(약어 vs 풀네임)이 달라 오답 처리되는 경우가 많으므로, Embedding 기반의 유사도 측정 지표 도입이 필요하다.
-   **데이터셋 의존성**: MASH-QA 사례에서 보듯, 정답이 분산되어 있는 경우 섹션 추출 방식이 오히려 독이 될 수 있다. 즉, 모든 MRC 작업에 Implicit RAG가 만능은 아니며, 데이터셋의 정답 분포 특성에 따라 프롬프팅 전략을 선택해야 한다.
-   **비용 문제**: GPT API 비용으로 인해 일부 데이터셋의 경우 전체가 아닌 15% 샘플 데이터로 프롬프트를 최적화한 후 전체 테스트를 수행하였는데, 이 과정에서 샘플링 편향이 발생했을 가능성이 있다.

## 📌 TL;DR

본 논문은 생의학 문맥적 MRC 작업에서 GPT-4의 성능을 평가하고, 외부 벡터 DB 없이 모델 내부적으로 관련 문맥을 먼저 추출하여 답을 내는 **Implicit RAG** 기법을 제안하였다. 실험 결과, Implicit RAG는 특히 문맥이 길고 복잡한 추론이 필요한 작업에서 매우 효과적이었으며, 일부 벤치마크에서 인간 전문가 수준의 성능 및 새로운 SoTA를 달성하였다. 이 연구는 LLM이 전문 도메인의 복잡한 독해 작업에서도 충분히 활용 가능함을 시사하며, 향후 RAG 시스템을 단순화하고 효율화하는 새로운 방향성을 제시한다.