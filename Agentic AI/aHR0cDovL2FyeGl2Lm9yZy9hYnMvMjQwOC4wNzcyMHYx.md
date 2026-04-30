# Re-Thinking Process Mining in the AI-Based Agents Era

Alessandro Berti, Mayssa Maatallah, Urszula Jessen, Michal Sroka, Sonia Ayachi Ghannouchi (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)을 프로세스 마이닝(Process Mining, PM) 작업에 적용할 때 발생하는 한계를 해결하고자 한다. 현재 LLM을 PM에 활용하는 방식은 크게 두 가지이다. 첫째는 프로세스 마이닝 결과물을 텍스트 형태로 추상화하여 LLM에 제공하고 통찰을 얻는 방식이며, 둘째는 원본 데이터에서 실행 가능한 코드(Python, SQL 등)를 생성하는 방식이다.

그러나 이러한 접근 방식은 복잡한 추론 능력이 요구되는 시나리오에서 한계를 보인다. 특히 인간 분석가가 여러 단계로 나누어 수행하는 복합 작업(Composite Tasks)의 경우, LLM이 스스로 작업을 적절히 분해하고 각 단계를 정확하게 실행하는 데 어려움을 겪는다. 예를 들어, 이벤트 로그에서의 불공정성(Unfairness)을 측정하려면 보호 집단을 식별하고 두 집단 간의 행동을 비교하는 단계적 과정이 필요하지만, 단일 LLM 프롬프트만으로는 이러한 복잡한 파이프라인을 완벽히 수행하기 어렵다. 따라서 본 연구의 목표는 LLM의 시맨틱 능력과 결정론적 도구(Deterministic Tools)를 결합하여 복잡한 PM 작업을 효율적으로 수행할 수 있는 AI 기반 에이전트 워크플로우(AI-Based Agents Workflow, AgWf) 패러다임을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 '분할 정복(Divide-et-Impera)' 원칙을 프로세스 마이닝에 적용하는 것이다. 복잡하고 거대한 작업을 LLM이 처리 가능한 수준의 작은 단위 작업으로 분해하고, 각 작업에 최적화된 전문 에이전트를 배치하는 AI 기반 에이전트 워크플로우(AgWf)를 설계하였다.

특히, LLM의 비결정론적인(Non-deterministic) 추론 능력과 기존 PM 라이브러리(예: pm4py)의 결정론적인(Deterministic) 계산 능력을 통합함으로써, 계산의 정확성과 해석의 유연성을 동시에 확보하고자 하였다. 이는 LLM이 모든 것을 직접 계산하게 하는 대신, 필요한 도구를 선택하고 그 결과물을 해석하는 '오케스트레이터' 및 '분석가'의 역할을 수행하게 함으로써 전체적인 결과물의 품질을 높이는 전략이다.

## 📎 Related Works

기존의 LLM 기반 프로세스 마이닝 연구들은 주로 텍스트 추상화를 통한 질의응답이나 SQL 생성 및 오류 수정 메커니즘에 집중하였다. 또한 일부 연구에서는 세만틱 이상 탐지나 루트 코즈 분석(Root Cause Analysis)과 같은 특정 PM 작업에 LLM을 적용하는 가능성을 탐색하였다.

한편, 전통적인 PM 워크플로우(예: RapidMiner, Knime 기반)는 실험의 재현성(Reproducibility)과 표준화를 목적으로 설계되었다. 그러나 본 논문에서 제안하는 AgWf는 재현성보다는 '실행 가능성(Feasibility)'과 '최종 출력의 품질'에 초점을 맞춘다. 전통적인 워크플로우가 동일한 입력에 대해 항상 동일한 출력을 내놓는 결정론적 특성을 가진다면, AgWf는 LLM의 특성상 비결정론적인 성격을 띠며, 상황에 따라 최적의 도구를 선택하고 결과를 조합하는 유연한 접근 방식을 취한다는 점에서 차별화된다.

## 🛠️ Methodology

### AI-Based Agents Workflow (AgWf) 정의

본 논문은 AgWf를 다음과 같은 튜플로 정의한다.
$$\text{AgWf} = (F, T, \text{tools}, \text{selector}, \text{prec}, t_1, t_f)$$

각 구성 요소의 역할은 다음과 같다.
- $F \subseteq (\Sigma^\cup \to \Sigma^\cup)$: 문자열을 입력받아 문자열로 출력하는 결정론적 도구(Tools)의 집합이다.
- $T \subseteq (\Sigma^\cup \rightsquigarrow \Sigma^\cup)$: 비결정론적인 AI 기반 작업(Tasks)의 집합이다. 여기서 $\rightsquigarrow$ 기호는 동일한 입력에 대해 서로 다른 출력이 가능함을 의미한다.
- $\text{tools}: T \to \mathcal{P}(F)$: 특정 작업 $T$에 할당된 도구들의 집합을 매핑하는 함수이다.
- $\text{selector}: \Sigma^\cup \times \mathcal{P}(F) \rightsquigarrow F$: 주어진 질의와 사용 가능한 도구 집합 중에서 최적의 도구를 선택하는 비결정론적 함수이다.
- $\text{prec}: T \to \mathcal{P}(T)$: 각 작업의 선행 작업(Preceding tasks) 관계를 정의한다.
- $t_1, t_f$: 워크플로우의 시작 작업과 최종 작업을 의미한다.

### AgWf의 실행 절차 (Sequential Execution)

워크플로우의 실행은 상태 시퀀스 $\langle \sigma_0, \sigma_1, \dots, \sigma_f \rangle$로 표현된다. $\sigma_0$는 사용자의 초기 질의이며, 각 단계 $i$에서의 상태 $\sigma_i$는 다음과 같이 결정된다.

1. **도구가 없는 작업 ($\text{tools}(t_i) = \emptyset$)인 경우:**
   $$\sigma_i = \sigma_{i-1} \oplus t_i(\sigma_{i-1})$$
   이전 상태에 AI 작업의 결과물을 단순히 결합한다.

2. **도구가 있는 작업 ($\text{tools}(t_i) \neq \emptyset$)인 경우:**
   $$\sigma_i = \sigma_{i-1} \oplus t_i(\sigma_{i-1} \oplus \text{selector}(\sigma_{i-1}, \text{tools}(t_i))(\sigma_{i-1}))$$
   먼저 $\text{selector}$를 통해 최적의 도구를 선택하고, 해당 도구를 실행하여 얻은 결과물을 AI 작업의 입력으로 함께 제공하여 최종 응답을 생성한다. 여기서 $\oplus$는 문자열의 결합(Concatenation)을 의미한다.

### AI 기반 작업의 유형

효과적인 PM 파이프라인 구축을 위해 다음과 같은 작업 유형을 제안한다.
- **Prompt Optimizers**: 사용자의 모호한 질의를 AI 에이전트가 이해하기 쉬운 최적의 언어로 변환한다.
- **Ensembles**: 여러 작업에서 도출된 다양한 관점의 통찰들을 취합하여 하나의 일관된 보고서로 요약한다.
- **Routers**: 질의의 성격에 따라 세만틱 분석(LLM 직접 처리)으로 보낼지, 데이터 계산(코드 생성 및 실행)으로 보낼지 경로를 결정한다.
- **Evaluators**: 이전 작업의 출력 품질을 평가하고 점수를 부여하며, 필요시 루프를 통해 재작업을 요청한다.
- **Output Improvers**: 생성된 결과물에 대해 '제2의 의견(Second opinion)'을 제시하거나 코드의 보안 및 품질을 개선한다.

## 📊 Results

본 논문은 정량적인 벤치마크 결과보다는 CrewAI 프레임워크를 이용한 구현 사례를 통해 AgWf의 효용성을 입증한다.

### 1. 편향 탐지 (Bias Detection) 워크플로우
작성자는 세 가지 구현 수준을 비교한다.
- **단일 작업 방식**: 하나의 LLM 작업이 모든 도구를 사용하여 결과를 내는 방식이나, 복잡한 파이프라인(집단 분리 $\to$ 비교 $\to$ 분석)을 수행하지 못해 효과가 가장 낮다.
- **단순 분해 방식**: 집단 식별과 비교 작업을 분리하였으나, 도구 선택 과정에서 누락이 발생할 수 있다.
- **다중 관점 분해 방식**: 보호 집단 식별 후, DFG(Directly-Follows Graph) 관점과 Process Variants 관점에서 각각 비교 분석을 수행하고 이를 Ensemble 작업으로 통합한다. 이 방식이 가장 정교한 불공정성 통찰을 제공함을 확인하였다.

### 2. 루트 코즈 분석 (Root Cause Analysis) 워크플로우
다음의 단계로 구성된 파이프라인을 구현하였다.
- **T1 (Analysis)**: DFG 추상화를 바탕으로 잠재적 근본 원인 목록을 생성한다.
- **T2 (Grading)**: 각 인사이트에 대해 $1.0 \sim 10.0$ 사이의 신뢰도 점수를 부여한다.
- **T3 (Reasoning)**: 가장 점수가 높은 인사이트에 대해 Chain-of-Thought(CoT)를 적용하여 상세 추론 과정을 제시한다.

이 과정에서 Qwen 2.0 8B와 같은 상대적으로 작은 모델을 사용하더라도, 작업이 충분히 세분화되어 있다면 효율적으로 수행 가능함을 보였다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 연구는 LLM을 단순한 챗봇이 아닌, 전문 도구를 사용하는 '에이전트'들의 협업 체계로 재정의하였다. 특히 PM 분야에서 LLM의 고질적인 문제인 '환각(Hallucination)'과 '복잡한 계산 능력 부족'을 결정론적 도구(pm4py 등)와의 결합을 통해 해결하려 한 점이 돋보인다.

### 한계 및 향후 과제
- **워크플로우 설계의 수동성**: 현재 AgWf의 구조는 인간이 직접 설계한다. 이를 자동화하는 '오케스트레이팅 LLM'의 도입이 필요하다.
- **인간 참여(Human-in-the-Loop)**: 매우 일반적인 질의의 경우 AI가 스스로 최적화하기 어렵기 때문에, 중간 단계에서 사용자에게 명확한 의도를 묻는 메커니즘이 보완되어야 한다.
- **평가 프레임워크**: LLM-as-a-Judge 방식을 도입하여 각 개별 에이전트의 성능을 정밀하게 측정하고, 에이전트 간의 협업 효율성을 평가하는 연구가 필요하다.
- **도구의 성숙도**: CrewAI, LangGraph, AutoGen 등 다양한 프레임워크가 존재하지만, 여전히 GUI 부족이나 버전 업데이트에 따른 불안정성 등의 문제가 남아 있다.

## 📌 TL;DR

본 논문은 LLM이 복잡한 프로세스 마이닝(PM) 작업에서 겪는 추론 한계를 극복하기 위해, 작업을 세분화하고 결정론적 도구와 LLM을 결합한 **AI-Based Agents Workflow (AgWf)** 패러다임을 제안한다. 분할 정복(Divide-et-Impera) 원칙에 따라 Prompt Optimizer, Router, Ensemble 등 전문화된 에이전트를 배치함으로써 분석의 정확도와 품질을 높일 수 있음을 사례 연구(편향 탐지, 루트 코즈 분석)를 통해 보여주었다. 이는 향후 LLM 기반의 자율적 PM 분석 시스템 구축을 위한 중요한 설계 방향을 제시한다.