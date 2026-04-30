# PROV-AGENT: Unified Provenance for Tracking AI Agent Interactions in Agentic Workflows

Renan Souza, Amal Gueroudji, Stephen DeWitt, Daniel Rosendo, Tirthankar Ghosal, Robert Ross, Prasanna Balaprakash, Rafael Ferreira da Silva (2025)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLM) 및 파운데이션 모델 기반의 AI 에이전트가 복잡한 워크플로우의 핵심으로 통합되고 있다. 이러한 에이전틱 워크플로우(Agentic Workflows) 내에서 에이전트들은 작업을 계획하고, 인간 및 다른 에이전트와 상호작용하며, 과학적 결과에 영향을 미친다. 그러나 AI 에이전트는 환각(Hallucination)을 일으키거나 잘못된 추론을 할 가능성이 있으며, 한 에이전트의 출력이 다른 에이전트의 입력이 되는 구조에서 이러한 오류는 워크플로우 전체로 전파되어 결과의 신뢰성을 심각하게 훼손할 수 있다.

전통적인 프로비넌스(Provenance, 데이터 기원 및 이력 추적) 기술은 데이터 흐름과 작업 실행 이력을 기록하는 데 유용하지만, AI 에이전트 특유의 메타데이터인 프롬프트(Prompt), 응답(Response), 추론 과정 및 결정 내역을 워크플로우의 광범위한 맥락과 연결하여 캡처하는 데 한계가 있다. 따라서 에이전트의 행동을 투명하고 추적 가능하며 재현 가능하게 만들어, 환각 리스크를 평가하고 오류의 근본 원인을 분석할 수 있는 새로운 프로비넌스 모델이 필요하다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 W3C PROV 표준을 확장하여 AI 에이전트의 상호작용을 워크플로우 프로비넌스 그래프의 '일급 시민(First-class components)'으로 통합하는 것이다.

1. **PROV-AGENT 모델 제안**: W3C PROV 표준을 확장하고 Model Context Protocol(MCP) 개념을 도입하여, 에이전트의 활동, 모델 호출, 프롬프트, 응답을 전통적인 워크플로우 작업 및 데이터와 단일 그래프 내에서 연결하는 프로비넌스 모델을 설계하였다.
2. **오픈소스 구현 시스템**: 데이터 관측성(Data Observability)과 런타임 에이전틱 프로비넌스 캡처를 위한 시스템을 구현하여 실시간에 가까운 추적이 가능하게 하였다.
3. **교차 시설(Cross-facility) 평가**: 엣지(Edge), 클라우드(Cloud), HPC(고성능 컴퓨팅) 환경을 아우르는 적층 제조(Additive Manufacturing) 워크플로우에 적용하여, 에이전트의 신뢰성 분석 및 복잡한 프로비넌스 쿼리가 가능함을 입증하였다.

## 📎 Related Works

기존의 에이전트 프레임워크인 LangChain, AutoGen, LangGraph, CrewAI 등은 멀티 에이전트 시스템의 상호작용과 MCP 기반의 도구(Tool) 호출, RAG(Retrieval-Augmented Generation) 등을 지원한다. 그러나 이러한 프레임워크들이 기록하는 프롬프트와 응답 데이터는 대개 워크플로우의 나머지 부분과 격리되어 저장되므로, 에이전트의 결정이 다운스트림 작업에 미치는 영향을 전체 맥락에서 파악하기 어렵다.

프로비넌스 분야에서는 W3C PROV 표준을 기반으로 PROV-DfA(인간 주도 워크플로우), ProvONE(워크플로우 메타데이터), PROV-ML(ML 모델 훈련 및 평가) 등의 확장이 있었다. 또한 FAIR4ML은 모델 중심의 재현성을 강조한다. 하지만 이러한 연구들은 워크플로우를 제어하는 '능동적 AI 에이전트'의 동적 의사결정 과정과 추론 경로를 캡처하는 데 초점을 맞추지 않았으며, 본 논문은 바로 이 지점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 모델
PROV-AGENT는 W3C PROV의 세 가지 핵심 클래스인 $\text{Entity}$, $\text{Activity}$, $\text{Agent}$를 확장하여 설계되었다.

1. **클래스 계층 구조**:
   - $\text{AIAgent}$: $\text{PROV Agent}$의 하위 클래스로, 워크플로우를 제어하는 AI 에이전트를 나타낸다.
   - $\text{AgentTool}$: $\text{PROV Activity}$의 하위 클래스로, 에이전트가 실행하는 도구의 실행 단위를 의미한다.
   - $\text{AIModelInvocation}$: $\text{PROV Activity}$의 하위 클래스로, 실제 AI 모델에 대한 호출 과정을 나타낸다.
   - $\text{Prompt}, \text{ResponseData}$: $\text{DataObject}$ (Entity의 하위 클래스)의 하위 클래스로, 모델에 입력된 프롬프트와 생성된 응답을 저장한다.
   - $\text{AIModel}$: 사용된 모델의 이름, 제공자, 온도(Temperature) 등 메타데이터를 가진 $\text{Entity}$이다.

2. **관계 정의 (Relationships)**:
   - $\text{AgentTool}$은 $\text{AIAgent}$에 의해 실행되며, $\text{AIModelInvocation}$에 의해 정보를 제공받는다 ($\text{wasInformedBy}$).
   - $\text{AIModelInvocation}$은 $\text{Prompt}$와 $\text{AIModel}$을 사용하며 ($\text{used}$), $\text{ResponseData}$를 생성한다 ($\text{generated}$).
   - $\text{ResponseData}$는 해당 $\text{AIAgent}$에게 귀속된다 ($\text{wasAttributedTo}$).

### 구현 상세
본 시스템은 분산 프로비넌스 프레임워크인 Flowcept를 확장하여 구현되었다.

- **캡처 메커니즘**: Python 데코레이터인 `@flowcept_agent_tool`을 사용하여 도구 실행 시 입력, 출력, 텔레메트리 데이터를 자동으로 캡처한다.
- **LLM 래퍼**: `FlowceptLLM`이라는 래퍼 클래스를 통해 OpenAI, LangChain 등 다양한 인터페이스의 LLM 호출 시 프롬프트, 응답, 모델 메타데이터를 가로채어 프로비넌스 데이터베이스에 저장한다.
- **인프라 통합**: 엣지-클라우드-HPC로 이어지는 분산 환경에서 발생하는 데이터를 브로커 기반 모델로 수집하여 중앙의 W3C PROV 확장 모델 그래프로 통합한다.

## 📊 Results

### 실험 설정 및 유스케이스
연구진은 오크리지 국립연구소(ORNL)에서 개발 중인 **자율 적층 제조(Autonomous Additive Manufacturing)** 워크플로우를 대상으로 평가를 진행하였다.

- **워크플로우 구성**:
    - **Edge**: 금속 3D 프린터의 센서 드라이버가 레이어별 데이터를 실시간 스트리밍한다.
    - **HPC**: 물리 기반 모델이 센서 데이터를 처리하여 제어 결과와 점수(Scores)를 생성한다.
    - **Cloud**: $\text{gpt-4o}$ 기반의 분석 에이전트가 이 점수들을 분석하여 최적의 제어 결정을 내린다. 이 결정은 다음 레이어의 결정에 영향을 주는 피드백 루프 구조를 가진다.

### 주요 분석 결과 (Enabled Queries)
PROV-AGENT를 통해 다음과 같은 핵심 쿼리가 가능함을 확인하였다.

1. **전체 계보 추적 (Full Lineage)**: 특정 에이전트의 결정($\text{Agent\_Decision}_i$)부터 최초의 입력 데이터($\text{Sensor\_Data}_i$)까지의 모든 경로를 역추적할 수 있다.
2. **의사결정 근거 확인**: 특정 레이어에서 에이전트가 내린 결정, 당시 가용했던 점수 옵션, 그리고 LLM이 생성한 추론 과정($\text{Response}$)을 함께 조회할 수 있다.
3. **환각 지점 식별**: 예상치 못한 결정(Hallucination)이 발견되었을 때, 해당 결정과 연결된 $\text{Prompt}$와 $\text{Response}$를 즉각 추출하여 모델의 어떤 부분이 잘못되었는지 분석할 수 있다.
4. **영향도 분석**: 특정 시점의 에이전트 결정이 이후의 워크플로우 활동 및 최종 결과물에 어떻게 전파되었는지 전방 추적(Forward tracing)할 수 있다.
5. **오류 전파 경로 파악**: 잘못된 데이터가 어디서 기원했는지 역추적하고, 그 데이터가 어떤 에이전트의 결정을 통해 전파되었는지 경로를 시각화할 수 있다.

## 🧠 Insights & Discussion

본 논문은 AI 에이전트가 단순한 도구를 넘어 워크플로우의 제어권을 갖게 된 '에이전틱 워크플로우' 시대에 필수적인 **책임감 있는 AI(Responsible AI)** 프레임워크를 제시하였다.

**강점**:
- W3C PROV라는 표준을 확장함으로써 상호운용성을 확보하였고, MCP라는 최신 프로토콜을 수용하여 실용성을 높였다.
- 단순한 로그 저장이 아니라, 데이터-작업-에이전트를 연결한 그래프 구조를 통해 복잡한 피드백 루프 내에서의 오류 전파를 정밀하게 추적할 수 있다.
- 엣지-클라우드-HPC라는 극단적인 이기종 환경에서의 실무적인 적용 가능성을 보여주었다.

**한계 및 논의사항**:
- 현재 구현은 주로 LLM에 집중되어 있으나, 제안된 모델은 모달리티에 무관하도록 설계되어 향후 비전이나 오디오 모델로 확장이 가능하다.
- 프로비넌스 데이터의 양이 방대해질 경우, 실시간 쿼리 성능을 유지하기 위한 최적화 방안에 대한 논의가 추가로 필요할 것으로 보인다.
- 본 논문은 추적 가능성을 제공하지만, 환각을 '감지'하거나 '자동 수정'하는 메커니즘은 포함되어 있지 않다. 다만, 구축된 프로비넌스 그래프가 향후 이러한 자동화 도구의 기반 데이터로 활용될 가능성이 크다.

## 📌 TL;DR

PROV-AGENT는 AI 에이전트의 프롬프트, 응답, 추론 과정을 전통적인 워크플로우 이력(Provenance)과 통합하는 W3C PROV 확장 모델이다. 이를 통해 에이전트의 비결정적 행동으로 인한 환각이나 오류가 워크플로우 전체에 어떻게 전파되는지 정밀하게 추적할 수 있으며, 이는 고신뢰성이 요구되는 과학적/산업적 AI 워크플로우의 디버깅, 투명성 확보 및 모델 튜닝에 핵심적인 역할을 할 것으로 기대된다.