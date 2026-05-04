# AgentGuard: Runtime Verification of AI Agents

Roham Koohestani (2025)

## 🧩 Problem to Solve

최근의 AI 시스템은 단순한 콘텐츠 생성 수준의 Generative AI에서 스스로 계획을 세우고 도구를 사용하는 Agentic AI로 빠르게 진화하고 있다. 그러나 이러한 자율적 에이전트 시스템은 다음과 같은 고유한 위험 요소를 내포하고 있다.

- **비결정론적 동작과 예측 불가능성**: LLM 기반 에이전트는 확률적 프로세스에 기반하므로 출력값이 일정하지 않으며, 다단계 워크플로우가 체인 형태로 연결될 때 예상 경로에서 기하급수적으로 벗어날 가능성이 크다.
- **환각(Hallucinations)**: 사실과 다르거나 논리적으로 결함이 있는 출력을 생성하며, 이는 에이전트가 잘못된 믿음을 바탕으로 행동하게 하여 시스템 실패로 이어진다.
- **창발적 행동(Emergent Behaviors)**: 특히 다중 에이전트 시스템(MAS)에서 개별 에이전트 분석만으로는 예측할 수 없는 의도치 않은 행동(예: 기만, 목표 전도)이 나타날 수 있다.
- **새로운 취약점**: 프롬프트 인젝션(Prompt Injection)이나 에이전트 간 전파되는 프롬프트 감염과 같은 새로운 공격 표면이 존재한다.

기존의 소프트웨어 검증 방식은 결정론적 논리와 관리 가능한 상태 공간을 가정하므로, 이러한 확률적이고 복잡한 Agentic AI 시스템에는 부적합하다. 따라서 본 논문은 "시스템이 실패할 것인가?"라는 이분법적 질문에서 벗어나, **"주어진 제약 조건 내에서 실패/성공할 확률은 얼마인가?"**라는 확률적 보장(Probabilistic Guarantees)의 문제로 패러다임을 전환하여 해결하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Dynamic Probabilistic Assurance (DPA)**라는 새로운 패러다임을 제시하고, 이를 구현한 **AgentGuard** 프레임워크를 제안한 것이다.

중심 아이디어는 에이전트를 정적으로 분석하는 대신, 실행 중에 발생하는 입출력 트레이스를 실시간으로 관찰하여 에이전트의 행동을 모델링하는 **Runtime Verification (RV)** 방식을 도입하는 것이다. 이를 통해 상태 공간 폭발 문제를 피하면서, 실제로 발생한 실행 경로에 대해 강력한 정량적 보장을 제공하고 미래 행동을 예측할 수 있는 '디지털 트윈' 모델을 동적으로 구축한다.

## 📎 Related Works

논문에서는 AI 신뢰성 확보를 위한 검증 스택을 다음과 같이 분류하고 각각의 한계를 지적한다.

1. **신경망 공식 검증 (Formal Verification of NNs)**: SMT 솔버 등을 사용하여 네트워크 자체의 견고함을 증명하려 하지만, 현대 LLM의 거대한 규모로 인해 계산적으로 불가능(NP-hard)하다.
2. **블랙박스 테스트 및 사후 검증 (Post-Hoc Verification)**: LLM이 생성한 코드나 계획을 외부 도구(Lean 4, SMT 솔버 등)로 검증하는 방식이다. 유용하지만 자연어 요구사항을 정밀한 명세로 변환하는 과정이 어렵다.
3. **프로세스 수준 검증 (Process-level Verification)**: CFG(문맥 자유 문법)나 PDA(푸시다운 오토마타), FSM(유한 상태 머신)을 사용하여 에이전트가 정해진 규칙(예: Thought $\rightarrow$ Action $\rightarrow$ Observation)을 따르는지 감시하는 방식이다.

**기존 연구와의 차별점**: 기존의 프로세스 수준 검증은 단순히 에이전트가 정의된 규칙을 준수하는지 확인하는 **준수성(Conformance)** 확인에 집중한다. 반면 AgentGuard는 규칙 준수를 넘어, 실행 과정에서 나타나는 **창발적인 확률적 행동(Emergent Probabilistic Behavior)** 자체를 분석하여 정량적인 예측 지표를 제공한다는 점에서 차별화된다.

## 🛠️ Methodology

AgentGuard는 기존 에이전트 프레임워크(AutoGen, LangGraph 등) 위에 얹혀지는 검사 계층(Inspection Layer)으로 작동하며, 크게 세 가지 기술(MDP, Online Learning, PMC)을 결합한다.

### 1. 에이전트의 Markov Decision Process (MDP) 모델링

에이전트의 행동을 다음과 같은 튜플 $(S, A, P, R, \gamma)$로 정의하는 **Agentic MDP (AMDP)**로 정형화한다.

- **상태 (States, $S$)**: 에이전트의 진행 상황과 문맥의 스냅샷이다. 이전 도구 호출 내역뿐만 아니라, 테스트 통과 여부, 메모리/스크래치패드 내용, 대화 기록 등이 포함된다.
- **행동 (Actions, $A$)**: 에이전트가 수행하는 고수준 작업이다. 예를 들어 `run_compiler()`, `execute_test_suite()`와 같은 도구 호출이 이에 해당한다.
- **전이 확률 (Transition Probability, $P$)**: 행동 후 다음 상태로 전이될 확률이다. 예를 들어 `execute_test_suite` 행동을 취했을 때, 결과가 '성공', '실패', '타임아웃' 상태로 전이될 확률을 캡처한다.
- **보상 함수 (Reward, $R$)**: 목표 달성 정도를 명시한다. 새로운 테스트 케이스 통과 시 양수 보상($+3$), 회귀 오류 발생 시 음수 보상($-5$), 최종 작업 완료 시 큰 보상을 부여한다.
- **감쇠 요인 (Discount Factor, $\gamma$)**: 미래 보상의 중요도를 결정한다.

### 2. 전체 시스템 아키텍처

AgentGuard의 파이프라인은 다음과 같이 구성된다.

1. **Trace Monitor & Event Abstrator**: 에이전트의 원시 I/O(LLM 호출, 도구 호출 등)를 캡처하여 상태 모델의 전이에 해당하는 정형 이벤트(예: $S_{stateA} \rightarrow A_{action1} \rightarrow S_{stateB}$)로 추상화한다.
2. **Online Model Learner**: 추상화된 이벤트 스트림을 바탕으로 에이전트의 행동을 나타내는 MDP를 실시간으로 업데이트하며 전이 확률을 유지한다.
3. **Probabilistic Model Checker (PMC)**: 학습된 MDP를 바탕으로 정량적 속성을 검증한다. 이때 PCTL(Probabilistic Computation Tree Logic)과 같은 확률적 시간 논리를 사용하여 질문을 던진다. (예: "제한 시간 내에 성공 상태에 도달할 최대 확률은 얼마인가?")
4. **Assurance Dashboard & Actuator**: 검증 결과를 시각화하여 관리자에게 제공하며, 안전 임계값이 무너질 경우 경고를 보내거나 자동 응답을 트리거한다.

### 3. 구현 세부사항 (Proof-Of-Concept)

- **언어 및 도구**: Python 기반으로 구현되었으며, 모델 체커로는 **Storm** (stormpy 바인딩 사용)을 채택하였다.
- **구성**: `AgentGuardLogger`가 이벤트 큐를 관리하고, `AnalyzerThread`라는 백그라운드 스레드가 모델 업데이트와 PMC 호출을 주기적으로 수행한다.
- **설정**: 검증 로직과 에이전트 코드를 분리하기 위해 `yaml` 설정 파일에서 상태, 행동, 검증 속성을 정의한다.

## 📊 Results

본 논문은 AgentGuard의 가능성을 입증하기 위해 자율 소프트웨어 수정 에이전트인 **RepairAgent**에 적용한 사례를 제시한다.

### 실험 설정

- **대상 시스템**: RepairAgent (버그 수정을 위해 정보 수집, 가설 수립, 패치 적용 등의 행동을 수행하는 LLM 에이전트).
- **정의된 상태**: `Understand the bug` $\rightarrow$ `Collect information` $\rightarrow$ `Try to fix the bug` $\rightarrow$ `Fix_Success` / `Fix_Failed`.
- **정의된 행동**: `express_hypothesis`, `read_range`, `discard_hypothesis` 등.

### 분석 결과 및 지표

AgentGuard를 통해 RepairAgent의 전략을 AMDP로 학습시킨 결과, 다음과 같은 정량적 지표를 도출할 수 있었다.

- **성공 확률 ($P_{max}$)**: 현재 상태에서 최종 수정 성공 상태에 도달할 확률을 예측하여, 확률이 낮을 경우 리소스 재배치나 인간의 개입을 결정하는 근거로 활용한다.
- **기대 사이클 수 ($E_{min}$)**: 완료까지 예상되는 단계 수이다. 이 값이 지나치게 높으면 에이전트가 무한 루프에 빠졌거나 무의미한 탐색을 하고 있다고 판단하여 자동 종료시킬 수 있다.
- **미수정 확률**: $\text{P}_{max} = ? [ \neg \text{"write\_fix"} ]$ 와 같은 속성을 통해 수정 시도조차 하지 못할 확률을 계산하여 문제 해결 프로세스를 개선한다.

## 🧠 Insights & Discussion

### 강점

본 연구는 AI 에이전트 검증의 관점을 '정적 규칙 준수'에서 '동적 확률 분석'으로 확장하였다. 특히 런타임 검증(RV)을 도입함으로써 모델 체킹의 고질적인 문제인 상태 공간 폭발을 피하고, 실제 실행 경로에 기반한 실시간 보장을 제공한다는 점이 매우 강력하다.

### 한계 및 향후 과제

- **상태 공간 정의의 수동성**: 현재는 개발자가 상태 공간을 직접 정의해야 한다. 이를 위해 POMDP(부분 관찰 마르코프 결정 과정) 등을 활용한 반자동/자동 상태 추상화 기법이 필요하다.
- **계산 오버헤드**: 에이전트가 복잡해질수록 전체 모델을 주기적으로 재검증하는 데 발생하는 오버헤드가 커질 수 있다. 증분 검증(Incremental Verification) 알고리즘 도입이 필요하다.
- **다중 에이전트 확장성**: 현재는 단일 에이전트 중심의 MDP를 사용한다. 다중 에이전트 시스템(MAS) 분석을 위해 Stochastic Games 이론과 PRISM-games 같은 프레임워크의 통합이 필요하다.

## 📌 TL;DR

AgentGuard는 자율 AI 에이전트의 예측 불가능한 창발적 행동을 실시간으로 감시하고 정량적으로 분석하는 **런타임 검증(Runtime Verification)** 프레임워크이다. 에이전트의 행동을 **MDP(마르코프 결정 과정)**로 모델링하고 **확률적 모델 체킹(PMC)**을 적용하여, 성공 확률이나 예상 소요 시간과 같은 수학적 보장을 실시간으로 제공한다. 이 연구는 AI 안전성 확보의 패러다임을 정적 검증에서 동적/확률적 보장으로 전환시켰으며, 향후 자율 에이전트의 신뢰성 및 비용 관리에 중요한 기반 기술이 될 가능성이 높다.
