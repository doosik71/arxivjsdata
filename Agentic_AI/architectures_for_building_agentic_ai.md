# Architectures for Building Agentic AI

Slawomir Nowaczyk (2025)

## 🧩 Problem to Solve

본 논문(책의 챕터)은 Agentic AI 및 Generative AI 시스템에서 발생하는 낮은 신뢰성(Reliability) 문제를 해결하고자 한다. 현재 많은 시스템이 최신 거대 언어 모델(LLM)의 성능에만 의존하고 있으나, 모델 자체의 성능만으로는 일관성 없는 동작, 감사(Audit)의 불가능성, 그리고 새로운 상황에 대한 취약성을 완전히 극복할 수 없다.

연구의 핵심 목표는 AI 에이전트의 신뢰성이 단순히 모델의 능력이 아닌, 시스템의 **아키텍처적 속성(Architectural property)**임을 논증하는 것이다. 즉, 모델이 제안하는 내용을 어떻게 검증하고, 제약하며, 실행하는지에 대한 구조적 설계가 신뢰성을 결정짓는다는 점을 강조한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 "모델은 제안하고, 아키텍처는 결정한다(Models propose, architectures dispose)"는 철학으로 요약된다. 신뢰성 있는 에이전트를 구축하기 위해 다음과 같은 설계 원칙을 제시한다.

1. **원칙적 컴포넌트화(Principled Componentisation):** 시스템을 목표 관리자(Goal manager), 플래너(Planner), 도구 라우터(Tool-router), 실행기(Executor), 메모리(Memory), 검증기(Verifier), 안전 모니터(Safety monitor) 등으로 분리하여 결함의 영향 범위(Blast radius)를 제한한다.
2. **규율 있는 인터페이스(Disciplined Interfaces):** 자유 형식의 모델 출력을 타입화된 스키마(Typed schemas)와 검증된 계약(Contracts)을 통해 예측 가능하고 감사 가능한 동작으로 변환한다.
3. **명시적 제어 및 보증 루프(Explicit Control and Assurance Loops):** 추론과 실행 사이에 검증기, 비판자(Critic), 감독자(Supervisor)를 배치하여 작은 추론 오류가 치명적인 사고로 이어지는 것을 방지한다.

## 📎 Related Works

### 고전적 에이전트 아키텍처

- **Reactive Architecture:** 지각을 행동으로 즉시 매핑하며 지연 시간이 짧지만, 장기적 계획이나 추론이 필요한 작업에서 취약하다.
- **Deliberative Architecture:** 세계 모델을 유지하고 탐색/계획을 통해 행동을 선택하며, 설명 가능성이 높지만 지연 시간이 길고 모델 불일치 문제가 발생할 수 있다.
- **Hybrid Architecture:** 위 두 방식을 결합하여 빠른 안전 제어(Reactive)와 느린 추론(Deliberative)을 분리한다.
- **BDI (Belief-Desire-Intention) 모델:** 신념(Beliefs), 욕구(Desires), 의도(Intentions)를 통해 에이전트의 행동을 구조화한다. 본 논문은 현대의 GenAI 에이전트가 BDI의 구조(세계 상태 $\rightarrow$ 목표 $\rightarrow$ 계획)를 계승하면서도 추론 부분을 신경망 모델로 대체했음을 지적한다.

### 현대적 에이전트 패턴

- **Tool-using agents:** Toolformer, ReAct, ReWOO 등 LLM이 외부 도구를 오케스트레이션하는 방식이다.
- **Memory-augmented agents:** RAG, MemGPT 등 외부 저장소를 통해 문맥 창의 한계를 극복하는 방식이다.
- **Planning agents:** ToT, GoT, PAL, Reflexion 등 탐색과 자가 수정을 통해 추론 능력을 강화하는 방식이다.

## 🛠️ Methodology

### 전체 시스템 파이프라인

신뢰성 있는 에이전트는 다음과 같은 논리적 흐름을 따른다:
$\text{Goal Manager} \rightarrow \text{Planner} \rightarrow \text{Tool Router} \rightarrow \text{Execution Gateway} \rightarrow \text{Verifier/Critic} \rightarrow \text{Actuation}$

1. **Goal Manager:** 사용자 의도와 제약 조건을 정규화하여 구체적인 작업 정의를 생성한다.
2. **Planner:** 생성형 모델을 사용하여 가설을 세우고 행동 후보군(Plan)을 도출한다.
3. **Tool Router:** 추상적인 행동을 구체적인 도구 매핑으로 변환하며, 타입 스키마를 통해 인자를 채운다.
4. **Execution Gateway:** 실행 전 스키마 검증, 전제 조건 확인, 샌드박스 내 시뮬레이션을 수행한다.
5. **Verifier/Critic:** 제안된 계획이 정책 및 안전 규칙에 부합하는지 검토한다.
6. **Safety Supervisor:** 예산(Budget), 종료 조건, 에스컬레이션(인간 개입) 규칙을 강제한다.

### 아키텍처 패밀리별 상세 설계

#### 1. Tool-using Agents

- **MRKL:** LLM을 포맨(Foreman)으로 활용하여 특정 작업을 전문 모듈(계산기, DB 쿼리 등)로 라우팅한다.
- **ReAct:** $\text{Thought} \rightarrow \text{Action} \rightarrow \text{Observation}$ 과정을 교차 반복하여 환각을 줄인다.
- **ReWOO:** 계획 생성과 실행을 분리하여 토큰 효율성을 높이고 계획 자체를 감사 가능한 객체로 만든다.

#### 2. Memory-augmented Agents

- **계층적 메모리:** Working Memory(단기/스크래치패드), Episodic Memory(경험 로그), Semantic Memory(RAG/지식 베이스)로 구분한다.
- **신뢰성 확보 방안:** 모든 메모리 항목에 출처(Provenance)를 기록하고, 유효 기간(TTL)을 설정하며, 신뢰 등급(Trust tiers)을 부여하여 오염을 방지한다.

#### 3. Planning & Self-improvement Agents

- **ToT (Tree of Thoughts) & GoT (Graph of Thoughts):** 추론 과정을 트리나 그래프 형태로 확장하고 스코어링 모델을 통해 최적의 경로를 탐색한다.
- **PAL (Program-Aided Language models):** 자연어 추론 대신 실행 가능한 코드(Python 등)를 생성하여 외부 인터프리터에서 실행함으로써 결정론적 계산을 보장한다.
- **Reflexion:** 실행 결과에 대한 언어적 반성(Reflection)을 통해 다음 시도에서 오류를 수정한다.

#### 4. Multi-agent Systems

- **구조:** Supervisor-Worker(중앙 통제형) 또는 Peer-to-Peer(협력형) 구조를 가진다.
- **제어 메커니즘:** 메시지 스키마 강제, 대화 라운드 제한(Termination), 중재자(Arbiter)를 통한 의견 충돌 해결 등을 통해 "채팅 폭풍"과 같은 무한 루프를 방지한다.

#### 5. Embodied & Web Agents

- **Simulate-before-actuate:** 물리적/웹 환경에서 실행 전 디지털 트윈이나 스냅샷 DOM에서 시뮬레이션을 선행한다.
- **최소 권한 원칙:** 도구 사용 권한을 매우 좁게 제한하고, 가역적인(Reversible) 명령만 허용하며, 물리적 안전 레이어(Interlocks)를 최하단에 배치하여 모델의 오류가 물리적 사고로 이어지지 않게 차단한다.

## 📊 Results

본 논문은 특정 실험 데이터를 제시하는 연구 논문이 아니라, 아키텍처 설계 가이드를 제공하는 서베이 성격의 챕터이다. 다만, 언급된 각 방법론의 정성적/정량적 이점은 다음과 같이 설명된다.

- **PAL:** 자유 형식의 CoT보다 알고리즘 및 수학적 작업에서 정밀도가 높으며, 예외 처리를 통해 실패 모드를 명확히 한다.
- **ReWOO:** ReAct 대비 토큰 사용량과 지연 시간을 크게 줄이면서도 도구 실패 상황에서 더 강건한 성능을 보인다.
- **ToT/GoT:** 단순 선형 추론보다 복잡한 문제(Game of 24 등)에서 정답률이 향상되며, 추론 경로의 가시성을 제공한다.
- **Multi-agent:** 역할 분담과 상호 비판 과정을 통해 단일 에이전트보다 정확도가 향상되지만, 합의 편향(Sycophancy) 위험이 존재함을 명시한다.

## 🧠 Insights & Discussion

### 강점

본 논문은 LLM 에이전트의 신뢰성을 '모델 튜닝'의 영역에서 '시스템 공학'의 영역으로 확장했다는 점에서 매우 중요하다. 특히 BDI와 같은 고전적 AI 이론을 현대의 LLM 구조에 접목하여, 단순한 프롬프트 엔지니어링을 넘어선 견고한 프레임워크를 제시하였다.

### 한계 및 논의사항

- **설계 복잡도 증가:** 제안된 모든 검증 레이어와 샌드박스를 구현할 경우 시스템의 복잡도가 기하급수적으로 증가하며, 이는 개발 비용과 유지보수의 어려움으로 이어질 수 있다.
- **성능과 신뢰성의 트레이드-오프:** 엄격한 스키마 검증과 시뮬레이션 단계는 신뢰성을 높이지만, 에이전트의 응답 속도(Latency)를 늦추고 창의적인 문제 해결 능력을 제한할 가능성이 있다.
- **결정론적 가드레일의 한계:** 모든 도구 호출을 스키마로 제약하더라도, 모델이 논리적으로 잘못된(그러나 형식적으로는 맞는) 요청을 보낼 경우를 완벽히 막기 위해서는 더 정교한 세만틱 검증기가 필요하다.

## 📌 TL;DR

본 논문은 Agentic AI의 신뢰성이 모델의 성능이 아닌 **아키텍처 설계**에서 온다고 주장한다. 이를 위해 **컴포넌트화, 타입화된 인터페이스, 명시적 보증 루프**라는 세 가지 핵심 축을 제시하며, 도구 사용, 메모리 증강, 계획 수립, 다중 에이전트, Embodied AI 등 5가지 아키텍처 패밀리에 대한 신뢰성 확보 전략을 상세히 다룬다. 이 연구는 향후 LLM 기반 자율 시스템을 구축할 때 "모델은 제안하고, 아키텍처가 검증 및 결정한다"는 구조적 가이드라인을 제공함으로써, 예측 불가능한 AI 행동을 제어 가능한 시스템으로 변환하는 데 중요한 역할을 할 것으로 보인다.
