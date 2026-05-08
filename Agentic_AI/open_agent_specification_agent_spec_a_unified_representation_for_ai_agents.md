# Open Agent Specification (Agent Spec): A Unified Representation for AI Agents

Soufiane Amini et al. (2025)

## 🧩 Problem to Solve

현재 AI 에이전트 생태계는 다양한 에이전트 프레임워크(예: LangGraph, AutoGen, CrewAI 등)의 급격한 확산으로 인해 심각한 파편화(Fragmentation) 문제를 겪고 있다. 각 프레임워크는 고유한 추상화 계층, 데이터 흐름 시맨틱, 도구 통합 방식을 정의하고 있어, 특정 프레임워크에서 개발된 에이전트를 다른 환경으로 이식하거나 재사용하는 것이 매우 어렵다.

이러한 파편화는 단순히 개발의 불편함을 넘어, 서로 다른 스택 간에 에이전트의 동작을 일관되게 평가하거나 워크플로우를 재현하는 것을 방해한다. 특히 기업 환경에서는 거버넌스 관리와 프로토타이핑에서 배포까지의 시간을 늦추는 핵심 요인이 된다. 본 논문은 모델 중심(Model-centric)의 표준화를 넘어, 에이전트의 행동과 실행 시맨틱을 표준화하는 에이전트 중심(Agent-centric)의 통합 표현 방식의 필요성을 제기하며, 이를 해결하기 위한 **Open Agent Specification (Agent Spec)**을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 머신러닝 모델의 상호 운용성을 위해 ONNX가 수행한 역할과 유사하게, AI 에이전트와 워크플로우를 위한 **선언적(Declarative)이고 프레임워크에 구애받지 않는(Framework-agnostic) 표준 설정 언어**를 구축하는 것이다.

주요 기여 사항은 다음과 같다:

- **Agent Spec 정의**: 에이전트와 워크플로우를 고충실도(High fidelity)로 정의할 수 있는 선언적 언어를 제안하여, "한 번 정의하고 어디서든 실행(Define-once, run-anywhere)"할 수 있는 이식성을 제공한다.
- **핵심 컴포넌트 및 시맨틱 공식화**: Agents, Flows, Nodes, Tools 및 제어/데이터 흐름 엣지(Edges)에 대한 표준 세트를 정의하여 시스템의 신뢰성과 재사용성을 높였다.
- **지원 도구셋 제공**: 프로그래밍 방식의 작성 및 직렬화를 위한 Python SDK(`PyAgentSpec`), 레퍼런스 런타임(`WayFlow`), 그리고 주요 프레임워크(LangGraph, AutoGen, CrewAI)를 위한 런타임 어댑터(Runtime Adapters)를 함께 제공한다.
- **평가 하네스(Evaluation Harness)로서의 활용**: 동일한 Agent Spec 설정을 서로 다른 런타임에서 실행함으로써, 프레임워크 간의 성능, 견고성, 효율성을 일관되게 비교할 수 있는 환경을 구축하고 이를 통해 벤치마크 결과를 제시하였다.

## 📎 Related Works

기존의 AI 에이전트 프레임워크들은 각각 뚜렷한 강점을 가지고 있다. LangGraph는 상태 기반의 유향 그래프(Directed Graphs)를 통한 명시적 제어 흐름에 강점이 있고, AutoGen은 멀티 에이전트 간의 대화 및 도구 사용에 특화되어 있으며, CrewAI는 역할 기반의 협업에 집중한다.

또한, 에이전트 생태계의 표준화를 위한 몇 가지 시도가 있었으나, 본 논문이 제안하는 Agent Spec과는 초점이 다르다:

- **Anthropic의 MCP (Model Context Protocol)**: 리소스 및 데이터 제공(Provisioning)의 표준화에 집중한다.
- **Google의 Agent2Agent 및 BeeAI의 ACP**: 에이전트 간의 통신 메시지 교환 표준화에 집중한다.
- **Agntcy**: 에이전트 발견(Discovery) 및 거버넌스 표준을 지향한다.

기존의 노력들이 주로 리소스 제공이나 통신 레이어의 상호 운용성에 집중했다면, Agent Spec은 **에이전트의 행동(Behavior)과 실행 시맨틱(Execution Semantics)**이라는 더 깊은 수준의 표준화를 다룬다는 점에서 차별점을 가진다.

## 🛠️ Methodology

Agent Spec은 에이전트 시스템을 구성하는 개념적 객체들을 **컴포넌트(Component)**로 정의하고, 이를 JSON 형식을 통해 직렬화하여 런타임에 전달하는 구조를 가진다.

### 1. 전체 시스템 구조

전체 파이프라인은 `Agent Spec 정의 $\rightarrow$ JSON 직렬화 $\rightarrow$ 런타임 어댑터 $\rightarrow$ 특정 프레임워크 실행` 순으로 진행된다. 런타임 어댑터는 일종의 컴파일러 역할을 하여, 선언적인 Agent Spec 설정을 타겟 프레임워크의 구체적인 실행 프리미티브로 변환한다.

### 2. 주요 구성 요소 및 역할

- **Base Component**: 모든 컴포넌트의 기본 단위로, 메타데이터와 속성을 가진다. 다른 컴포넌트를 참조할 때는 `$component_ref:{COMPONENT_ID}` 형태의 심볼릭 참조를 사용한다.
- **Agent**: 최상위 컴포넌트로, 대화 메모리와 도구와 같은 공유 리소스를 보유하며 시스템의 진입점 역할을 한다.
- **LLM**: 생성적 컴포넌트로, 모델 식별자 및 생성 설정(Generation settings)을 포함하는 `LLMConfig`를 통해 구성된다.
- **Tool**: 에이전트가 실행할 수 있는 절차적 함수이다. 실행 위치에 따라 $\text{ServerTools}$, $\text{ClientTools}$, $\text{RemoteTools}$, $\text{MCPTools}$의 네 가지 유형으로 분류한다.
- **Flow**: 유향 그래프 형태의 워크플로우로, 결정론적인 실행 경로를 제공한다. $\text{StartNode}$에서 시작하여 $\text{EndNode}$에서 종료된다.
- **Node**: Flow 내의 정점으로, $\text{LLMNode}$, $\text{APINode}$, $\text{AgentNode}$, $\text{FlowNode}$, $\text{BranchingNode}$, $\text{ToolNode}$ 등이 정의되어 있다.

### 3. 실행 및 데이터 흐름 제어

Agent Spec은 실행의 모호성을 제거하기 위해 제어 흐름과 데이터 흐름을 엄격히 구분한다.

- **ControlFlowEdge (제어 흐름 엣지)**: 실행 순서(Execution Order)를 결정한다. 노드의 특정 브랜치에서 다음 노드로의 전이 가능성을 정의하며, 런타임에 실제 경로가 결정된다.
- **DataFlowEdge (데이터 흐름 엣지)**: 한 노드의 출력 속성(Output property)이 다른 노드의 입력 속성(Input property)으로 어떻게 매핑되는지를 정의한다.
- **I/O 스키마**: 모든 컴포넌트는 입력과 출력을 명시적으로 선언해야 하며, `{{property_name}}` 형태의 플레이스홀더를 사용하여 런타임에 값을 동적으로 바인딩한다.

## 📊 Results

본 연구는 Agent Spec의 이식성과 재사용성을 증명하기 위해 4개의 런타임(WayFlow, LangGraph, AutoGen, CrewAI)을 사용하여 3가지 벤치마크 데이터셋에서 실험을 수행하였다.

### 1. 실험 설정

- **데이터셋 및 지표**:
  - **SimpleQA Verified**: 사실 관계 확인 작업 ($F1\text{-score}$ 측정)
  - **BIRD-SQL**: 자연어-to-SQL 작업 ($\text{EX\%}$ 측정)
  - **$\tau^2\text{-Bench}$**: 고객 서비스 시뮬레이션 환경 ($\text{Pass}@k$ 측정)
- **비교 대상**: 동일한 Agent Spec 설정을 기반으로 구현된 서로 다른 프레임워크의 에이전트들.

### 2. 주요 결과

- **SimpleQA Verified**:
  - ReAct 스타일 에이전트의 경우, CrewAI가 가장 높은 $F1\text{-score}$를 기록했으나 응답 시간이 다른 프레임워크보다 55%에서 140%까지 느렸다.
  - Flow 기반의 Agentic RAG 솔루션(WayFlow, LangGraph)이 단순 ReAct보다 훨씬 높은 정확도를 보였으며, 이는 문제 분해(Decomposition)와 자기 성찰(Self-reflection)의 효과임을 입증했다.
- **BIRD-SQL**:
  - 프레임워크 간의 성능 차이가 상대적으로 적었으나, '계획-생성-성찰(Plan-Generate-Reflect)' flow가 단순 LLM 호출 베이스라인보다 일관되게 높은 성능을 보였다.
- **$\tau^2\text{-Bench}$**:
  - 복잡한 추론 작업에서 AutoGen이 상대적으로 우수한 성능을 보였으나, 지연 시간(Latency)이 증가하는 경향을 보였다.
  - CrewAI는 단일 작업 해결 중심의 설계 특성상, 대화형 상호작용이 중요한 이 벤치마크에서 가장 낮은 점수를 기록했다.

## 🧠 Insights & Discussion

### 강점 및 유효성

본 논문은 동일한 선언적 명세(Agent Spec)가 서로 다른 런타임에서도 기능적인 에이전트를 성공적으로 인스턴스화할 수 있음을 보여줌으로써, 프레임워크 독립적인 이식성을 실증하였다. 특히, 동일한 설정을 사용함에도 불구하고 런타임에 따라 성능과 지연 시간이 달라지는 것을 통해, 에이전트의 성능이 단순히 프롬프트뿐만 아니라 프레임워크의 실행 계층(Execution layer) 및 내부 구현 방식에 의해 영향을 받는다는 점을 명확히 드러냈다.

### 한계 및 미해결 질문

- **컴포넌트의 범위**: 현재 버전에서는 메모리(Memory) 컴포넌트나 고급 플래닝 모듈(Planning modules)에 대한 표준 표현이 부족하며, 이는 향후 확장 과제로 남아 있다.
- **런타임 최적화**: 현재의 어댑터 방식은 단순 변환에 가깝다. 논문에서 언급된 것처럼 연속적인 $\text{LLMNode}$를 융합(Fusing)하는 등의 컴파일러 수준의 최적화가 이루어진다면 더 높은 효율성을 얻을 수 있을 것이다.

### 비판적 해석

Agent Spec은 에이전트 개발의 '표준 인터페이스'를 제공함으로써 파편화를 해결하려 하지만, 실험 결과에서 보듯 런타임 간의 성능 편차가 크다는 점은 시사하는 바가 크다. 이는 표준화된 언어로 정의하더라도 각 프레임워크가 이를 해석하고 실행하는 '시맨틱'의 미묘한 차이가 결과에 큰 영향을 미친다는 것을 의미한다. 따라서 향후에는 단순한 구조적 표준화를 넘어, 실행 결과의 일관성을 보장하는 'Conformance Test Suite'의 정밀도를 높이는 것이 핵심이 될 것이다.

## 📌 TL;DR

본 논문은 AI 에이전트 프레임워크의 파편화 문제를 해결하기 위해, 에이전트와 워크플로우를 정의하는 통합 선언적 언어인 **Agent Spec**을 제안한다. 이는 ML 모델의 ONNX와 유사한 역할을 하며, 개발자가 한 번 정의한 에이전트를 LangGraph, AutoGen, CrewAI 등 다양한 환경에서 수정 없이 실행할 수 있게 한다. 실험을 통해 이식성을 검증하였으며, 동일한 명세 하에서도 프레임워크별 성능/효율 차이가 발생함을 확인하여 일관된 에이전트 평가를 위한 기반을 마련하였다. 향후 에이전트 설계의 표준으로 자리 잡는다면, 에이전트 시스템의 재사용성과 상호 운용성을 획기적으로 향상시킬 가능성이 크다.
