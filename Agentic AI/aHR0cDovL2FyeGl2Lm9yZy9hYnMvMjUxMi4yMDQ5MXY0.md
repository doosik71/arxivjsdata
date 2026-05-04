# Step-DeepResearch Technical Report

Agent-Team, StepFun (2025)

## 🧩 Problem to Solve

본 논문은 기존의 AI 에이전트들이 수행하는 '검색(Search)'과 실제 전문가가 수행하는 '연구(Research)' 사이의 근본적인 차이에서 발생하는 문제를 해결하고자 한다. 일반적인 검색 중심의 에이전트들은 정해진 정답이 있는 Multi-hop QA나 단순한 정보 추출(Retrieval) 정확도에 최적화되어 있으며, 이는 실제 세계의 개방형 연구(Open-ended research) 요구사항을 충족시키지 못한다.

실제 연구 과정은 단순한 정보 수집을 넘어 잠재적 의도 인식, 장기적 의사결정(Long-horizon decision-making), 다회차 도구 사용, 논리적 구조화, 그리고 교차 소스 검증(Cross-source verification)과 같은 복합적인 능력을 요구한다. 기존 시스템들은 주로 외부 워크플로우(Workflow)를 하드코딩하여 이를 해결하려 했으나, 이는 시스템 복잡도를 높이고 특정 시나리오에서의 깊이 있는 연구 능력을 보장하지 못한다. 따라서 본 연구의 목표는 모델이 전문가 수준의 인지 루프를 내재화하여, 외부 프레임워크에 의존하지 않고도 강력한 자율 연구 능력을 갖춘 비용 효율적인 엔드투엔드(End-to-end) 에이전트 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Deep Research 능력을 구성하는 '원자적 능력(Atomic Capabilities)'으로 분해하고, 이를 단계적으로 학습시키는 것이다. 단순히 다음 토큰을 예측하는 학습에서 벗어나, 다음 '원자적 행동(Atomic Action)'을 결정하도록 학습 목표를 재설정하였다.

주요 기여 사항은 다음과 같다.

1. **원자적 능력 기반 데이터 합성(Atomic-capability data synthesis):** 계획 수립, 정보 탐색, 성찰 및 검증, 보고서 작성이라는 네 가지 핵심 능력에 특화된 데이터 합성 파이프라인을 구축하여 모델의 기초 체력을 강화하였다.
2. **점진적 학습 파이프라인(Progressive training pipeline):** Agentic Mid-training $\rightarrow$ Supervised Fine-tuning (SFT) $\rightarrow$ Reinforcement Learning (RL)으로 이어지는 체계적인 최적화 경로를 제안하였다.
3. **ADR-Bench 구축:** 실제 산업 현장의 수요를 반영한 중국어 기반의 Deep Research 벤치마크인 ADR-Bench를 구축하여, 단순 정답 맞추기가 아닌 실질적인 유용성을 평가할 수 있는 체계를 마련하였다.

## 📎 Related Works

기존의 Deep Research 접근 방식은 크게 두 가지로 나뉜다. 첫째는 OpenAI DeepResearch나 Gemini DeepResearch와 같이 거대 모델과 복잡한 외부 워크플로우 오케스트레이션(Orchestration)을 결합한 방식이다. 이러한 방식은 높은 성능을 보이지만 시스템 복잡도가 매우 높고 하드코딩된 패턴에 의존하는 경향이 있다. 둘째는 RL 등을 통해 능력을 내재화하려는 엔드투엔드 최적화 방식(예: Kimi-Researcher)이다.

그러나 기존의 엔드투엔드 연구들은 주로 '검색 효율성'이나 '쿼리 최적화'에 집중했을 뿐, 장기적 논리 추론, 다중 소스 교차 검증, 고품질 보고서 구성과 같은 '원자적 핵심 능력'을 체계적으로 구축하는 전략은 부족했다. Step-DeepResearch는 이러한 공백을 메우기 위해 원자적 능력의 분해와 단계적 학습이라는 접근 방식을 취함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 원자적 능력 데이터 전략 (Data Strategy for Atomic Capabilities)

모델이 복잡한 행동 공간에서 길을 잃지 않도록, 학습 목표를 $\mathcal{A}_{\text{token}}$에서 더 작은 부분 집합인 $\mathcal{A}_{\text{atomic}}$으로 제한한다.

- **계획 및 과업 분해 (Planning & Task Decomposition):** 실제 고품질 기술 보고서나 학술 논문을 역공학(Reverse Engineering)하여, 해당 결과물이 나오기 위해 필요했을 '프로젝트 과업'과 '계획' 데이터를 생성한다.
- **심층 정보 탐색 (Deep Information Seeking):** Wikidata 및 CN-DBpedia와 같은 지식 그래프에서 서브그래프를 샘플링하여 Multi-hop 추론이 필요한 질문을 생성하거나, Wiki-doc의 하이퍼링크 구조를 따라가는 Topology walk를 통해 연상 검색 능력을 강화한다.
- **성찰, 검증 및 교차 검증 (Reflection, Verification & Cross-Validation):** '전문가 모델 생성 $\rightarrow$ 결과 검증 $\rightarrow$ 다회차 성찰'의 폐쇄 루프(Closed-loop)를 통해 오류를 스스로 수정하는 궤적 데이터를 생성한다.
- **보고서 생성 (Report Generation):** 전문가의 작성 스타일을 배우는 Mid-training 단계와 지침 준수 및 포맷팅을 배우는 SFT 단계로 나누어 학습시킨다.

### 2. 점진적 학습 파이프라인 (Training Pipeline)

32B 파라미터 규모의 Qwen2.5-32B-Base 모델을 기반으로 3단계 학습을 진행한다.

- **Stage 1: Agentic Mid-training:** 32K 컨텍스트에서 기초 인지 패턴을 학습시킨 후, 128K 컨텍스트로 확장하며 도구 호출(Tool use) 능력을 주입한다.
- **Stage 2: Post-training SFT:** 원자적 능력들을 조합하여 엔드투엔드 과업을 수행하는 궤적을 학습한다. 이때 '정확하면서도 가장 짧은' 경로를 선택하도록 최적화하여 효율성을 높인다.
- **Stage 3: Reinforcement Learning (RL):** 실시간 도구 환경에서 PPO(Proximal Policy Optimization) 알고리즘을 사용하여 정책을 최적화한다.

### 3. 보상 설계 및 RL 알고리즘

본 논문은 Rubric(평가 기준) 기반의 보상 시스템을 도입하였다.

- **Rubrics Judge:** 강한 모델(Teacher model)이 매긴 점수와 설명을 학습한 별도의 판별 모델을 구축하여 보상 신호를 제공한다.
- **Strict Reward Mapping:** '부분적 만족'이라는 모호한 구간을 제거하고, 긍정적 루브릭은 '완전 만족'일 때만 보상을 주고, 부정적 루브릭은 '불만족'일 때만 보상을 주는 이진 매핑(Binary mapping)을 적용하여 신호의 변별력을 높였다.
- **PPO objective:** 다음과 같은 Clipped PPO 목적 함수를 사용하여 정책 $\pi_\theta$를 업데이트한다.
$$\max_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^{T} \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$
여기서 중요도 비율 $r_t(\theta)$는 다음과 같다.
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$
우선순위 추정(Advantage Estimation)을 위해 GAE(Generalized Advantage Estimation)를 사용하며, $\gamma=1, \lambda=1$로 설정하여 희소 보상 환경에서의 신용 할당(Credit assignment)을 단순화하였다.

### 4. 시스템 아키텍처

ReAct 패러다임을 따르는 단일 에이전트 구조를 사용하며, `batch_web_surfer`, `todo`, `shell` 등의 도구 셋을 활용한다. 특히 컨텍스트 오버플로우를 방지하기 위해 도구 결과가 임계치를 넘으면 요약본만 컨텍스트에 넣고 원문은 로컬 파일에 저장하는 'Implicit Context Management' 전략을 사용한다.

## 📊 Results

### 1. 실험 설정

- **벤치마크:** ResearchRubrics (LLM 기반 평가) 및 ADR-Bench (인간 및 전문가 평가).
- **비교 대상:** 상용 에이전트 시스템(OpenAI DeepResearch, Gemini DeepResearch 등) 및 ReAct 기반 에이전트(DeepSeek-V3.2, GLM-4.6 등).

### 2. 정량적 결과 (ResearchRubrics)

Step-DeepResearch는 **61.42점**을 기록하며 단일 에이전트(ReAct Agent) 카테고리에서 1위를 차지하였다. 이는 OpenAI DeepResearch(60.67)를 앞선 수치이며, 최상위권인 Gemini DeepResearch(63.69)에 근접한 성능이다.

### 3. 인간 평가 및 비용 효율성 (ADR-Bench & Cost)

- **ADR-Bench:** 인간 전문가의 Elo 레이팅 결과, Step-DeepResearch는 대다수의 상용 시스템보다 높은 승률을 보였으며, 특히 일반 도메인에서 높은 유용성을 입증하였다.
- **비용 효율성:** Gemini나 OpenAI의 시스템이 보고서당 수 RMB의 비용을 소모하는 반면, Step-DeepResearch는 단일 호출 비용을 **0.50 RMB 미만**으로 유지하면서도 유사한 성능을 내어 극도로 높은 비용 효율성을 달성하였다.

### 4. 세부 분석

- **차원별 분석:** '명시적 기준(Explicit Criteria)'과 '인용 품질(Citation Quality)'에서 매우 높은 점수를 기록하여 사실적 근거에 기반한 작성이 가능함을 보였다.
- **도메인별 분석:** AI & ML, 역사 분석, 기술 문서 작성 분야에서 Gemini와 대등한 수준의 성능을 보였으나, STEM 및 철학 분야에서는 고차원 추론의 한계로 인해 Gemini에 밀리는 모습을 보였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 성과는 **모델의 규모(32B)가 상대적으로 작음에도 불구하고, 체계적인 데이터 전략과 점진적 학습을 통해 거대 상용 모델에 필적하는 연구 능력을 구현**했다는 점이다. 특히 '원자적 능력'을 먼저 학습시키고 이를 조합하는 방식이 단순한 데이터 증강보다 훨씬 효과적임을 입증하였다.

또한, 루브릭 기반의 RL이 단순 모방 학습(SFT)의 한계를 넘어 에이전트가 스스로 최적의 경로를 탐색하게 함으로써 실질적인 성능 향상을 이끌어냈다는 점이 주목할 만하다.

**한계점 및 논의 사항:**

1. **도구 사용의 견고성:** API 응답의 변동성이나 매우 복잡한 도구 조합 시나리오에서는 여전히 취약한 모습을 보인다.
2. **사실성 보장:** 정보 노이즈가 심한 환경에서 '그럴듯하지만 증명 불가능한(plausible but unprovable)' 추론을 내놓는 경우가 존재한다.
3. **보고서 가독성:** 체크리스트 기반 평가에서는 높은 점수를 얻더라도, 실제 인간이 느끼기에 분석의 깊이가 부족하고 정보가 파편화되어 나열되는 경향이 발견되었다.

## 📌 TL;DR

Step-DeepResearch는 32B 파라미터의 중형 모델임에도 불구하고, **원자적 능력 기반의 데이터 합성 $\rightarrow$ 점진적 학습(Mid-training $\rightarrow$ SFT $\rightarrow$ RL) $\rightarrow$ 루브릭 기반 보상 최적화** 과정을 통해 전문가 수준의 Deep Research 능력을 구현한 모델이다. ResearchRubrics와 ADR-Bench에서 최상위권 성능을 기록했으며, 특히 상용 서비스 대비 압도적으로 낮은 추론 비용으로 고성능을 구현하여 산업적 적용 가능성을 극대화하였다. 향후 멀티 에이전트 협업 및 사실성 강화 학습을 통해 더욱 정교한 연구 에이전트로 발전할 가능성이 높다.
