# Attacks on Third-Party APIs of Large Language Models

Wanru Zhao, Vidit Khazanchi, Haodi Xing, Xuanli He, Qiongkai Xu, Nicholas Donald Lane (2024)

## 🧩 Problem to Solve

최근 대규모 언어 모델(Large Language Models, LLMs)은 외부의 세 번째 파티 API(Third-party API)를 호출하는 플러그인 생태계를 통해 실시간 정보 접근, 복잡한 계산, 특정 도메인의 전문 작업 수행 등 그 능력을 확장하고 있다. 그러나 이러한 통합 과정에서 LLM 서비스 플랫폼은 외부 API가 제공하는 데이터의 신뢰성을 완전히 보장할 수 없다는 심각한 보안 취약점을 안게 된다.

본 논문은 제3자 API 서비스가 악의적으로 조작되었을 때, LLM이 이를 필터링하지 못하고 그대로 수용하여 사용자에게 잘못된 정보를 제공하는 보안 취약점을 분석하는 것을 목표로 한다. 특히, 사용자가 인지하지 못하는 사이에 LLM의 출력이 미세하게 변경되는 공격 가능성을 탐색하고, 이에 따른 생태계의 안전성 문제를 제기한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 LLM의 입력 프롬프트를 직접 공격하는 기존의 Prompt Injection 방식에서 벗어나, LLM이 신뢰하고 사용하는 **외부 데이터 소스(API 응답 값)를 조작**하여 모델을 기만하는 공격 프레임워크를 제안한 것이다.

공격자는 API 응답의 JSON 형식 데이터 내의 특정 필드를 조작함으로써, LLM이 최종 답변을 생성하는 과정에서 잘못된 정보를 생성하도록 유도한다. 이는 LLM이 외부 도구를 사용하여 답변을 생성하는 파이프라인의 구조적 맹점을 이용한 것으로, 사용자와 LLM 모두가 데이터의 위조 여부를 파악하기 어렵다는 점이 핵심이다.

## 📎 Related Works

기존의 LLM 보안 연구는 주로 모델 자체의 취약점이나 사용자의 입력값(Prompt)을 조작하여 모델의 가드레일을 무너뜨리는 Prompt Injection 공격에 집중해 왔다. 또한 Toolformer, ToolLLM, API-Bank와 같은 연구들은 LLM이 어떻게 하면 더 효율적으로 외부 API를 사용하여 기능을 확장할 수 있는지에 초점을 맞추었다.

본 논문은 이러한 기존 연구들과 달리, LLM과 API 간의 상호작용 지점에서 발생하는 데이터 무결성 문제를 다룬다. 즉, 도구 사용의 효율성이 아닌 **도구의 신뢰성** 문제를 제기하며, API 응답 데이터가 오염되었을 때 LLM이 이를 어떻게 처리하는지를 분석함으로써 기존의 프롬프트 기반 공격과는 차별화된 새로운 공격 벡터를 제시한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
시스템의 기본 워크플로우는 다음과 같다:
1. **사용자 요청**: 사용자가 자연어로 질문을 입력한다.
2. **쿼리 생성**: LLM이 질문을 분석하고 해당 API를 호출하기 위한 API 형식의 쿼리를 생성한다.
3. **API 호출 및 응답**: 제3자 API 서버가 쿼리를 처리하여 JSON 형식의 데이터를 반환한다.
4. **최종 답변 생성**: LLM이 수신한 JSON 데이터를 기반으로 다시 자연어 형태의 답변을 구성하여 사용자에게 제공한다.

공격자는 이 과정 중 **3단계(API 응답)**에서 JSON 데이터를 조작하여 LLM에게 오염된 정보를 전달한다.

### 공격 방법론 (Threat Model)
본 논문은 API 콘텐츠를 조작하는 세 가지 구체적인 방법론을 제안한다.

- **삽입 기반 공격 (Insertion-based Attack)**: API 응답에 적대적인 콘텐츠를 추가하여 LLM이 부정확하거나 편향된, 혹은 해로운 출력을 생성하게 만든다.
- **삭제 기반 공격 (Deletion-based Attack)**: API 응답에서 핵심적인 정보를 누락시켜 LLM이 불완전하거나 잘못된 답변을 내놓게 한다.
- **치환 기반 공격 (Substitution-based Attack)**: 핵심 데이터를 삭제한 후 거짓된 내용으로 교체하는 방식으로, 삭제와 삽입의 결합 형태이다. 이는 LLM의 신뢰성을 가장 심각하게 훼손한다.

### 대상 API 및 조작 규칙
실험을 위해 WeatherAPI, MediaWiki API, NewsAPI 세 가지를 사용하였으며, 각 API별 타겟 엔티티는 다음과 같다.
- **WeatherAPI**: `location`(위치) 및 `temperature`(온도) 필드를 조작한다.
- **MediaWiki API**: `DATE`(날짜) 엔티티를 타겟으로 하며, spaCy를 통해 날짜를 식별 후 조작한다.
- **NewsAPI**: `PERSON`(인물), `ORG`(조직), `GPE`(지정학적 엔티티)를 타겟으로 조작한다.

### 평가 지표 (Attack Success Rate, ASR)
공격의 성공률을 측정하기 위해 다음과 같은 ASR 수식을 사용한다.

- **삽입 ASR**:
$$\text{ASR}_{\text{Insertion}} = \frac{\text{# of Successful Insertions}}{\text{# of Valid Instances}}$$
- **삭제 ASR**:
$$\text{ASR}_{\text{Deletion}} = \frac{\text{# of Successful Deletions}}{\text{# of Valid Instances}}$$
- **치환 ASR**: 삽입과 삭제의 균형 잡힌 평가를 위해 조화 평균(Harmonic Mean)을 사용한다.
$$\text{ASR}_{\text{Substitution}} = \frac{2 \times \text{InsertASR} \times \text{DeleteASR}}{\text{InsertASR} + \text{DeleteASR}}$$

## 📊 Results

### 실험 설정
- **대상 모델**: GPT-3.5-turbo (v0125), Gemini
- **데이터셋**: WikiQA (MediaWiki, WeatherAPI용), NewsQA (NewsAPI용)
- **측정 지표**: 각 공격 유형별 ASR

### 주요 결과
1. **공격 유형별 효과**: 전반적으로 **치환 공격(Substitution)**이 가장 높은 성공률을 보였으며, 그 다음으로 삭제 공격, 삽입 공격 순으로 효과적이었다. 이는 LLM이 정보의 부재보다 잘못된 정보가 포함된 데이터를 처리할 때 더 취약함을 시사한다.
2. **모델별 취약성**: WeatherAPI 실험 결과, Gemini가 GPT-3.5-turbo보다 모든 공격 유형에서 더 높은 ASR을 보여 상대적으로 더 취약한 것으로 나타났다.
3. **API별 결과**: NewsAPI의 경우 삽입 공격의 효율이 매우 낮았으나, 삭제 및 치환 공격은 여전히 높은 성공률을 유지했다. MediaWiki API에서는 치환 공격이 가장 위협적인 것으로 분석되었다.

## 🧠 Insights & Discussion

본 논문은 공격의 성공 여부에 영향을 미치는 세 가지 주요 요인을 분석하였다.

첫째, **충돌하는 지식의 주입(Conflicting Knowledge Injection)**이다. 조작된 정보가 LLM이 이미 내부적으로 보유하고 있는 지식과 강하게 충돌할 경우, 모델은 공격을 거부하는 경향이 있다. 반면, 내부 지식이 부족한 영역일수록 API의 조작된 정보를 그대로 믿고 수용할 가능성이 높다.

둘째, **추론 능력(Reasoning Capabilities)**이다. 추론 능력이 뛰어난 모델일수록 데이터 간의 불일치를 식별하여 무시할 가능성이 높다. 예를 들어, 온도 데이터를 완전히 무작위로 삽입하는 것보다 미세하게 조정하여 치환하는 것이 더 성공률이 높았다.

셋째, **공격의 품질(Attack Quality)**이다. Named Entity Recognition(NER) 기술의 정확도가 공격의 정밀도를 결정하며, API가 제공하는 정보의 양이 너무 방대할 경우 공격자가 정밀하게 타겟팅하여 조작하는 것이 어려울 수 있다는 점이 논의되었다.

## 📌 TL;DR

본 연구는 LLM이 외부 API를 통해 기능을 확장하는 과정에서, **신뢰할 수 없는 제3자 API가 제공하는 JSON 응답 값을 조작함으로써 LLM의 출력을 오염시킬 수 있음**을 입증하였다. 특히 치환 공격이 가장 치명적이며, 모델의 내부 지식이나 추론 능력에 따라 취약성이 달라짐을 확인하였다. 이는 향후 LLM 생태계에서 API 응답 데이터에 대한 검증 메커니즘과 보안 프로토콜 도입이 필수적임을 시사한다.