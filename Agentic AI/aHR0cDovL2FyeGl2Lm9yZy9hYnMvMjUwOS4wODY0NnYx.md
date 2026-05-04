# Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations

Ron F. Del Rosario, Klaudia Krawiecka, Christian Schroeder de Witt (2025)

## 🧩 Problem to Solve

최근 Large Language Model (LLM) 에이전트가 복잡한 다단계 작업을 자동화하는 능력이 향상됨에 따라, 시스템의 견고함(Robustness), 보안성(Security), 그리고 예측 가능성(Predictability)을 보장하는 아키텍처 패턴의 필요성이 증대되었다.

특히, 외부 데이터 소스(웹페이지, PDF, 이메일 등)에 숨겨진 악의적인 지시사항이 에이전트의 동작을 조작하는 **Indirect Prompt Injection** 공격은 기존의 반응형(Reactive) 에이전트 구조에서 매우 치명적인 취약점으로 작용한다. 또한, 에이전트가 복잡한 작업을 수행할 때 목적을 상실하고 루프에 빠지거나, 비효율적인 경로를 선택하는 등 추론의 일관성이 떨어지는 문제 또한 해결해야 할 과제이다.

본 논문의 목표는 전략적 계획(Strategic Planning)과 전술적 실행(Tactical Execution)을 분리하는 **Plan-then-Execute (P-t-E)** 패턴을 제안하고, 이를 통해 보안성과 예측 가능성을 극대화하며 실제 프로덕션 환경에 적용 가능한 상세 구현 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM 에이전트의 보안과 신뢰성을 높이기 위해 '설계에 의한 보안(Security by Design)' 관점에서 P-t-E 패턴을 정의하고 구체화한 것에 있다.

1. **P-t-E 아키텍처의 정립**: 계획 수립자(Planner)와 실행자(Executor)를 명확히 분리하여, 외부 데이터에 노출되기 전 제어 흐름(Control-flow)을 확정함으로써 제어 흐름 무결성(Control-flow Integrity)을 확보한다.
2. **심층 방어(Defense-in-Depth) 전략 제시**: P-t-E 패턴만으로는 불충분하며, 최소 권한 원칙(Principle of Least Privilege), 태스크 기반 도구 범위 제한(Task-scoped tool access), 샌드박스 기반 코드 실행(Sandboxed code execution)을 결합한 다층 보안 모델을 제안한다.
3. **프레임워크별 구현 블루프린트 제공**: LangGraph, CrewAI, AutoGen이라는 세 가지 주요 프레임워크에서 P-t-E 패턴을 어떻게 안전하게 구현할 수 있는지 상세한 구조와 코드 레퍼런스를 제공한다.
4. **고급 패턴 및 최적화 방안 제시**: 동적 재계획(Dynamic re-planning), DAG(Directed Acyclic Graph)를 이용한 병렬 실행, 그리고 인간 개입 검증(Human-in-the-Loop, HITL)을 포함한 $\text{Plan-Validate-Execute (P-V-E)}$ 모델을 제안한다.

## 📎 Related Works

논문은 P-t-E 패턴을 설명하기 위해 가장 널리 쓰이는 **ReAct (Reason + Act)** 패턴과 비교 분석한다.

- **ReAct 패턴**: `Thought $\rightarrow$ Action $\rightarrow$ Observation`의 타이트한 반복 루프를 통해 동작한다. 매 단계마다 LLM이 다음 행동을 결정하므로 매우 유연하고 적응력이 높지만, 다음과 같은 한계가 있다.
  - **단기적 사고 (Short-term thinking)**: 전체 작업에 대한 총체적 관점이 부족하여 복잡한 의존성이 있는 작업에서 비효율적인 경로를 선택할 가능성이 크다.
  - **보안 취약성**: 도구 실행 결과(Observation)가 다시 LLM의 입력으로 들어가기 때문에, 외부 데이터에 포함된 악의적 프롬프트가 에이전트의 다음 판단을 즉각적으로 하이재킹할 수 있다.
  - **비용 및 지연 시간**: 매 단계마다 LLM 호출이 필요하므로 작업 단계가 많아질수록 API 비용과 지연 시간이 선형적으로 증가한다.

반면, **P-t-E 패턴**은 실행 전 전체 계획을 수립하므로 경로가 예측 가능하고, 고비용의 추론 모델(Planner)은 초기에 한 번만 사용하고 실행 단계에서는 경량 모델(Executor)을 사용할 수 있어 비용 효율적이다.

## 🛠️ Methodology

### 전체 시스템 구조

P-t-E 패턴은 크게 다음과 같은 구성 요소로 이루어진다.

1. **Planner (계획 수립자)**: 사용자의 모호한 요청을 분석하여 구체적이고 실행 가능한 하위 작업의 시퀀스로 분해한다. 출력물은 단순 텍스트가 아닌 JSON이나 DAG와 같은 구조화된 형태의 '계획 아티팩트'이다.
2. **Executor (실행자)**: Planner가 생성한 계획을 한 단계씩 수행한다. Planner보다 단순하고 비용이 저렴한 모델을 사용할 수 있으며, 경우에 따라 결정론적인 코드나 단순한 ReAct 에이전트로 구성될 수 있다.
3. **Verifier & Refiner (검증 및 수정자, 선택 사항)**: 실행 전 계획의 논리적 타당성과 보안 준수 여부를 검토하며, 오류가 발견될 경우 이를 수정한다.

### 보안 구현 메커니즘

본 논문은 단순한 패턴 적용을 넘어 다음과 같은 보안 통제 장치를 강조한다.

- **제어 흐름 무결성 (Control-Flow Integrity)**: 외부 데이터를 읽기 전에 계획을 확정(Lock-in)하여, 도구 출력값이 계획 자체를 변경하지 못하도록 차단한다.
- **최소 권한 원칙 (Least Privilege)**: 도구를 에이전트 전체에 부여하는 것이 아니라, 특정 단계(Step)나 태스크(Task) 단위로 동적 할당한다. 예를 들어, 계산 단계에서는 $\text{calculator\_tool}$만 접근 가능하고 $\text{send\_email\_tool}$은 접근 불가능하게 설정한다.
- **격리된 실행 환경 (Sandboxing)**: 코드 실행 능력이 있는 에이전트의 경우, 반드시 Docker 컨테이너와 같은 ephemeral한 환경에서 코드를 실행하고 결과를 반환받은 뒤 컨테이너를 파기함으로써 호스트 시스템으로의 RCE(Remote Code Execution) 공격을 방지한다.

### 프레임워크별 구현 특징

- **LangGraph**: 상태 머신(State Machine) 기반의 그래프 구조로 구현한다. `planner_node` $\rightarrow$ `executor_node` $\rightarrow$ `replan_node` 순으로 노드를 구성하며, 조건부 엣지(Conditional Edge)를 통해 실행 상태에 따라 재계획 루프로 진입하거나 종료한다.
- **CrewAI**: 계층적 프로세스($\text{Process.hierarchical}$)를 사용한다. $\text{Manager Agent}$가 Planner 역할을 수행하며, 하위 $\text{Worker Agents}$에게 태스크를 위임한다. 특히 $\text{Task.tools}$가 $\text{Agent.tools}$보다 우선순위가 높다는 점을 이용하여 태스크 수준의 정밀한 도구 제어를 수행한다.
- **AutoGen**: 대화형 에이전트 구조를 활용한다. `GroupChat` 내에서 사용자 정의 화자 선택 함수(`speaker_selection_method`)를 통해 $\text{Planner} \rightarrow \text{Coder} \rightarrow \text{Executor}$ 순의 결정론적 상태 머신을 구축하며, `use_docker=True` 설정을 통해 기본적으로 샌드박싱을 제공한다.

## 📊 Results

본 논문은 특정 벤치마크 데이터셋에 대한 실험 결과보다는 아키텍처 가이드라인과 비교 분석에 집중한다. 주요 결과 및 분석 내용은 다음과 같다.

1. **정성적 보안 분석**: P-t-E 패턴은 ReAct 대비 Indirect Prompt Injection에 대해 훨씬 강력한 저항력을 가진다. 계획이 이미 수립된 상태에서 데이터를 읽기 때문에, 데이터 내의 지시문이 새로운 도구 호출을 생성하거나 실행 흐름을 바꿀 수 없다.
2. **정량적 효율성 (참조)**: DAG 기반 병렬 실행을 도입할 경우, I/O 바운드 작업(웹 검색, API 호출 등)에서 최대 $3.6\times$의 속도 향상을 얻을 수 있음을 언급한다.
3. **비교 분석 결과 (Table 1, 3)**:
    - **예측 가능성**: P-t-E $\gg$ ReAct
    - **보안성(간접 주입 저항력)**: P-t-E $\gg$ ReAct
    - **초기 응답 속도(Latency)**: ReAct $\gg$ P-t-E (P-t-E는 upfront planning으로 인해 초기 지연이 발생함)
    - **비용(복잡한 작업 시)**: P-t-E가 효율적 (고비용 모델 호출 횟수 감소)

## 🧠 Insights & Discussion

### 강점 및 통찰

본 논문의 가장 큰 통찰은 LLM 보안의 패러다임을 **행동적 억제(Behavioral Containment)**에서 **아키텍처적 격리(Architectural Containment)**로 전환해야 한다는 점이다. 시스템 프롬프트를 통해 "악의적인 지시를 따르지 마라"고 지시하는 것은 취약하지만, P-t-E와 같이 구조적으로 제어 흐름을 고정하고 샌드박스를 적용하는 것은 확률적인 LLM의 특성과 무관하게 하드웨어/소프트웨어 수준의 강제성을 부여할 수 있다.

### 한계 및 논의사항

1. **초기 지연 시간 (Time-to-First-Action)**: 전체 계획을 수립해야 하므로 사용자가 첫 번째 액션을 보기까지의 시간이 길어진다. 이는 실시간성이 중요한 챗봇 서비스에서는 단점이 될 수 있다.
2. **토큰 소비량**: 복잡한 작업을 위해 상세한 계획을 세울 경우, 초기 계획 단계에서 수천 개의 토큰이 한 번에 소비될 수 있다. 이를 해결하기 위해 본 논문은 하위 계획 수립자(Sub-planners)를 두는 계층적 계획 구조를 대안으로 제시한다.
3. **신뢰의 보정 (Calibration of Trust)**: LLM은 "그럴듯하게 틀린(Convincingly wrong)" 계획을 세울 수 있다. 따라서 단순히 자동화를 하는 것이 아니라, 고위험 작업의 경우 $\text{Plan} \rightarrow \text{Validate} \rightarrow \text{Execute}$ 순서로 인간이 계획 단계에서 개입하는 것이 실행 단계에서 개입하는 것보다 훨씬 효과적임을 강조한다.

## 📌 TL;DR

본 논문은 LLM 에이전트의 보안과 신뢰성을 확보하기 위해 **전략적 계획과 전술적 실행을 분리하는 P-t-E(Plan-then-Execute) 패턴**을 제안한다. P-t-E는 계획을 먼저 확정함으로써 외부 데이터에 의한 제어 흐름 하이재킹(Indirect Prompt Injection)을 방어하며, 여기에 **최소 권한 원칙(Task-scoped tools)**과 **Docker 샌드박싱**을 결합한 심층 방어 전략을 구축할 것을 권고한다. LangGraph, CrewAI, AutoGen 등 주요 프레임워크의 구현 방법을 제시하며, 결과적으로 LLM 에이전트의 보안은 모델 튜닝이 아닌 **시스템 아키텍처 설계의 문제**임을 역설한다.
