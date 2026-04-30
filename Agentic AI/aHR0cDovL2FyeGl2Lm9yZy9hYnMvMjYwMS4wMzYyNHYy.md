# Architecting Agentic Communities using Design Patterns: A Framework Grounded in ODP Enterprise Language Formalism

Zoran Milosevic and Fethi Rabhi (2026)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLM)의 급격한 발전으로 LLM Agent 및 Agentic AI 기술이 빠르게 등장하고 있으나, 이를 실제 산업 현장에 적용 가능한 수준의 정교한 생산 단계(production-grade) 시스템으로 구축하기 위한 체계적인 아키텍처 가이드라인이 부족한 상태이다.

기존의 디자인 패턴(예: ReAct, Tool Use, Planning 등)은 주로 개별 LLM의 프롬프팅 기법이나 단순한 작업 수행에 집중되어 있으며, 기업 환경에서 요구하는 엄격한 책임 추적성(accountability), 거버넌스 및 규제 준수(compliance) 문제를 해결하기에는 정밀함이 부족하다. 특히, 자율적인 AI 에이전트와 인간 참여자가 공존하는 복잡한 멀티 에이전트 시스템에서 어떻게 상호작용을 설계하고, 의사결정의 권한을 정의하며, 이를 공식적으로 검증할 수 있을 것인가에 대한 체계적인 방법론이 부재하다. 따라서 본 논문은 기업 및 산업 응용 분야에서 필수적인 거버넌스 구조를 갖춘 에이전트 커뮤니티를 설계하기 위한 공식적인 프레임워크를 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LLM 기반 엔티티를 세 가지 계층으로 분류하고, 이를 ISO 표준인 ODP(Open Distributed Processing) Enterprise Language(EL)의 공식주의(formalism)에 기반하여 설계하는 것이다.

1.  **3계층 분류 체계 제안**: 자율성과 조정 특성에 따라 LLM Agents(작업 중심), Agentic AI(목표 중심), Agentic Communities(조직/거버넌스 중심)로 구분하여 아키텍처 설계의 명확한 기준을 제시한다.
2.  **디자인 패턴 카탈로그 구축**: 총 46개의 디자인 패턴을 12개의 테마 카테고리로 분류하여 제공함으로써, 실무자가 요구사항에 따라 적절한 패턴을 선택하고 조합할 수 있게 한다.
3.  **ODP-EL 기반의 공식적 거버넌스**: Agentic Community를 ODP-EL의 커뮤니티 모델로 정의하여, 역할(Role), 규범적 제약(Normative Constraints), 계약(Contract)을 통해 AI와 인간의 협업을 공식적으로 명시하고 검증 가능하게 만든다.
4.  **단계적 설계 방법론 제시**: '특성 평가 $\rightarrow$ 패턴 조합 $\rightarrow$ 범위 설정'으로 이어지는 3단계 설계 프로세스를 통해 복잡한 에이전트 시스템을 체계적으로 구축하는 경로를 제안한다.

## 📎 Related Works

기존의 LLM 에이전트 연구들은 주로 ReAct, Reflexion과 같은 추론 및 행동 패턴이나, 메모리 증강, 도구 사용과 같은 개별 에이전트의 기능 향상에 집중해 왔다. 그러나 이러한 접근 방식은 다음과 같은 한계를 가진다.

- **정밀도 부족**: 대부분의 패턴이 서술적인 가이드라인 형태이며, 규제 산업(의료, 금융 등)에서 요구하는 수준의 공식적 검증(formal verification) 수단을 제공하지 않는다.
- **인간-AI 협업의 부재**: AI 에이전트 간의 상호작용은 다루지만, 실제 기업 환경에서 필수적인 '인간의 감독'과 '법적 책임'을 아키텍처 수준에서 어떻게 통합할지에 대한 논의가 부족하다.
- **거버넌스 결여**: 자율적 에이전트가 증가함에 따라 발생하는 예측 불가능성과 창발적 행동(emergent behavior)을 제어할 수 있는 공식적인 거버넌스 프레임워크가 부족하다.

본 논문은 이러한 한계를 극복하기 위해 항공(DO-178C)이나 의료(IEC 62304) 소프트웨어 공학에서 사용하는 엄격한 검증 체계를 벤치마킹하여, ODP-EL이라는 표준화된 언어를 통해 에이전트 커뮤니티의 거버넌스를 공식화하였다.

## 🛠️ Methodology

### 1. Three-Tier Classification Framework
본 논문은 LLM 기반 엔티티를 자율성(Autonomy)과 에이전시(Agency)의 수준에 따라 다음과 같이 정의한다.

- **LLM Agent**: 제어된 환경 내에서 특정 작업을 수행하는 엔티티이다. 스스로 목표를 설정하지 못하며 부여된 작업만을 수행하는 '작업 중심'의 자율성을 가진다.
- **Agentic AI**: 진정한 의미의 에이전시(Agency)를 가진 엔티티이다. 스스로 목표를 설정하고, 환경을 인식하며, 전략적 계획을 수립하고 적응적으로 행동하는 '목표 중심'의 자율성을 가진다.
- **Agentic Community**: LLM Agent, Agentic AI, 그리고 인간 참여자가 구조화된 프로토콜을 통해 협력하는 조정 프레임워크이다. 개별 에이전트의 능력을 넘어서는 창발적 지능을 구현하며, 공식적인 거버넌스 구조가 필수적이다.

### 2. ODP-EL Formal Foundation
Agentic Community의 거버넌스를 위해 ISO/IEC 15414 (ODP-EL) 표준을 적용한다. 핵심 구성 요소는 다음과 같다.

- **Roles (역할)**: LLM Agent, Agentic AI, 또는 인간이 채울 수 있는 행동적 플레이스홀더이다.
- **Deontic Tokens (의무 토큰)**: 거버넌스를 구현하는 핵심 메커니즘으로, 세 가지 토큰을 통해 권한과 의무를 제어한다.
    - $\text{Burden}$: 반드시 수행해야 하는 의무(Obligation)
    - $\text{Permit}$: 수행 가능한 허가(Permission)
    - $\text{Embargo}$: 수행해서는 안 되는 금지(Prohibition)
- **Contracts (계약)**: 역할 간의 규범적 관계를 정의하며, 상호 의무와 협력 프로토콜을 명시한다.

### 3. Design Pattern Methodology
아키텍처 설계를 위해 다음 3단계 프로세스를 거친다.

**Step 1: 특성 평가 (Assess)**
- 자율성 요구사항, 규제 환경, 데이터 특성, 학습 요구사항의 4가지 차원에서 유스케이스를 분석하여 후보 패턴을 식별한다.

**Step 2: 패턴 조합 (Compose)**
- 식별된 패턴을 세 가지 관계(Layered, Complementary, Alternative)를 바탕으로 조합한다.
- **Vertical Composition**: 단순 기능에서 복잡한 기능으로 쌓아 올리는 방식 (예: Tool-Using $\rightarrow$ ReAct $\rightarrow$ Community).
- **Horizontal Composition**: 동일 계층 내에서 서로 보완적인 패턴을 결합하는 방식.
- **Cross-Cutting Composition**: 거버넌스 패턴(Audit Trail 등)을 전체 시스템에 오버레이하는 방식.

**Step 3: 범위 설정 (Scope)**
- 구현 리소스와 리스크 허용도에 따라 복잡도 티어(Simple Automation $\rightarrow$ Departmental $\rightarrow$ Enterprise-Wide)를 결정하고 구현 순서를 정한다.

## 📊 Results

### 임상시험 매칭 시스템 (Clinical Trial Matching Community) 사례 연구
본 논문은 제안한 방법론을 의료 분야의 임상시험 환자 매칭 시스템에 적용하여 검증하였다.

**1. 적용 아키텍처 구조**
시스템은 총 15개의 패턴을 사용하며 3개의 논리적 계층으로 구성되었다.
- **Layer 1 (FHIR Foundation)**: FHIR 표준 기반 데이터 접근 및 거버넌스. $\text{Structured Extraction}$, $\text{Access Control}$, $\text{Audit Trail}$ 패턴 적용.
- **Layer 2 (Matching Workflow)**: 자율적 추론 및 매칭 핵심 로직. $\text{ReAct}$, $\text{Hierarchical Planning}$, $\text{Human-in-the-Loop}$ 패턴 적용.
- **Layer 3 (Conversational Negotiation)**: 외부 기관과의 동적 협상. $\text{Negotiation}$, $\text{Semantic Bridge}$, $\text{Orchestration}$ 패턴 적용.

**2. 공식적 검증 (Formal Verification)**
ODP-EL의 Deontic 토큰을 통해 다음과 같은 안전성 및 권한 속성을 런타임에 검증할 수 있음을 보였다.
- **Safety Property**: "동의 없이 환자 데이터에 접근할 수 없다."
    - $\forall a : \text{permit}(\text{access data}, a, p) \rightarrow \exists c : \text{burden}(\text{consent}, p) \text{ DISCHARGED}$
- **Authority Property**: "최종 등록 결정은 반드시 의사가 내려야 한다."
    - $\forall \text{enrollment} : \text{burden}(\text{makedecision}, \text{Physician}) \text{ REQUIRED}$
- **Prohibition Property**: "AI는 환자를 자동으로 등록할 수 없다."
    - $\forall ai \in \text{AIAGENTS} : \text{embargo}(\text{finaldecision}, ai) \text{ HOLDS}$

**3. 결과의 의의**
이러한 접근 방식은 단순히 테스트 기반의 보증이 아니라, 공식적인 증명을 통해 규제 산업(HIPAA 등)에서 요구하는 수준의 책임 추적성과 안전성을 확보할 수 있음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구는 파편화되어 있던 LLM 에이전트 패턴들을 체계적으로 분류했을 뿐만 아니라, 이를 엔터프라이즈 수준의 표준(ODP-EL)과 결합하여 '검증 가능한 AI 아키텍처'라는 실무적 해결책을 제시하였다. 특히 AI의 자율성과 인간의 통제권 사이의 균형을 $\text{Burden}$, $\text{Permit}$, $\text{Embargo}$라는 명확한 토큰 메커니즘으로 풀어낸 점이 돋보인다.

### 비판적 해석 및 논의
논문에서 언급된 **의도(Intent)**와 **의무(Obligation)**의 구분은 매우 중요하다. 저자들은 내면의 인지 상태인 $\text{Intent}$는 전송 불가능하지만, 외적으로 관찰 가능한 $\text{Obligation}$은 전송 가능하다고 주장한다. 이는 AI 에이전트에게 책임을 전가하는 것이 아니라, '수행해야 할 일'을 위임하고 최종 책임은 인간이나 법인(Party)이 지는 구조를 공식화한 것이다.

다만, 본 논문은 아키텍처 프레임워크에 집중하고 있어, 실제 LLM의 확률적 특성으로 인해 발생하는 '환각(Hallucination)'이나 '예상치 못한 프롬프트 주입'과 같은 하위 레벨의 보안 취약점이 공식적 거버넌스 계층에서 어떻게 실시간으로 완전히 차단될 수 있는지에 대한 구체적인 메커니즘은 더 보완될 필요가 있다.

## 📌 TL;DR

본 논문은 LLM 에이전트를 **LLM Agent $\rightarrow$ Agentic AI $\rightarrow$ Agentic Community**의 3계층으로 분류하고, ISO 표준인 **ODP-EL**을 도입하여 기업급 AI 시스템의 거버넌스와 책임 추적성을 공식적으로 설계하는 방법론을 제안한다. 46개의 디자인 패턴 카탈로그와 3단계 설계 프로세스를 제공하며, 임상시험 매칭 사례를 통해 **의무-허가-금지 토큰** 기반의 거버넌스가 실제 규제 환경에서 어떻게 검증 가능한 안전성을 제공하는지 입증하였다. 이 연구는 AI 에이전트가 단순한 프로토타입을 넘어 의료, 금융과 같은 안전 필수(safety-critical) 산업으로 진출하기 위한 아키텍처적 기반을 마련했다는 점에서 매우 중요하다.