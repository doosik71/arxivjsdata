# AUTOGENSTUDIO: A No-Code Developer Tool for Building and Debugging Multi-Agent Systems

Victor Dibia, Jingya Chen, Gagan Bansal, Suff Syed, Adam Fourney, Erkang Zhu, Chi Wang, Saleema Amershi (2024)

## 🧩 Problem to Solve

최근 여러 개의 에이전트(Generative AI 모델과 도구의 결합체)가 협력하는 Multi-Agent System(MAS)이 복잡하고 장기적인 작업을 해결하는 효과적인 패턴으로 부상하고 있다. 그러나 이러한 시스템을 구축하기 위해서는 사용 모델, 도구(Tools), 오케스트레이션 메커니즘(Orchestration mechanisms) 등 수많은 파라미터를 정밀하게 설정해야 하며, 에이전트 간의 복잡한 상호작용을 디버깅하는 과정이 매우 까다롭다.

특히 기존의 AutoGen, CAMEL, TaskWeaver와 같은 프레임워크들은 주로 Python 코드를 통한 'Code-first' 방식을 제공한다. 이는 전문 개발자가 아닌 사용자에게는 진입 장벽이 높으며, 프로토타이핑 속도를 늦추고 설정 오류를 발생시키기 쉬운 구조이다. 따라서 본 논문은 Multi-Agent 워크플로우를 신속하게 프로토타이핑하고, 디버깅하며, 평가할 수 있는 No-code 개발 도구의 필요성을 제기한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AutoGen 프레임워크를 기반으로 한 No-code 개발 도구인 **AUTOGENSTUDIO**의 제안과 구현이다. 주요 설계 아이디어는 다음과 같다.

1. **선언적 명세(Declarative Specification):** 에이전트와 워크플로우를 JSON 기반의 선언적 형식으로 정의하여, 코드 작성 없이도 시스템을 구축할 수 있게 한다.
2. **드래그 앤 드롭(Drag-and-Drop) UI:** 모델, 스킬(Skills), 메모리 컴포넌트를 에이전트에 연결하고, 에이전트를 다시 워크플로우에 배치하는 직관적인 인터페이스를 제공한다.
3. **시각적 프로파일링 및 디버깅:** 에이전트 간의 메시지 흐름, 토큰 비용, 도구 호출 성공 여부 등의 메트릭을 시각화하여 시스템의 병목 지점과 오류를 쉽게 파악할 수 있도록 한다.
4. **재사용 가능한 템플릿 갤러리:** 검증된 에이전트 구성 요소와 워크플로우 템플릿을 공유하고 재사용할 수 있는 생태계를 구축한다.

## 📎 Related Works

### 1. Agents (LLMs + Tools)

LLM의 환각(Hallucination)과 추론 능력의 한계를 극복하기 위해 ReAct와 같이 모델이 도구를 사용하여 행동하는 Agentic 구현 방식이 연구되었다. LangChain이나 LIDA와 같은 프레임워크는 모델과 도구를 엮은 고정된 파이프라인(Prescriptive pipelines)을 제공하지만, 동적인 문제 공간에 적응하는 데 한계가 있다.

### 2. Multi-Agent Frameworks

AutoGen, CAMEL, OS-Copilot 등은 에이전트 간의 자율적인 협력을 가능하게 하는 추상화 계층을 제공한다. 그러나 이들 대부분은 코드 중심의 구현 방식을 취하고 있으며, 다음과 같은 한계가 존재한다.

- **높은 진입 장벽:** 빠른 프로토타이핑을 방해하는 코드 기반 설정.
- **디버깅 도구 부족:** 에이전트 상호작용을 분석하고 평가할 수 있는 시각적 도구 및 메트릭 부재.
- **재사용성 결여:** 구조화된 템플릿의 부재로 인해 매번 유사한 설정을 반복해야 함.

AUTOGENSTUDIO는 이러한 한계를 해결하기 위해 시각적 인터페이스와 선언적 정의 방식을 도입하여 차별화를 꾀한다.

## 🛠️ Methodology

### 전체 시스템 아키텍처

AUTOGENSTUDIO는 크게 **Frontend UI**와 **Backend API**로 구성된다.

#### 1. Frontend User Interface (React 기반)

UI는 사용자의 목적에 따라 세 가지 주요 뷰(View)를 제공한다.

- **Build View:** 저수준 컴포넌트(모델, 스킬, 메모리)를 정의하고, 이를 에이전트에 할당한 뒤, 최종적으로 워크플로우에 배치하는 'Define-and-Compose' 경험을 제공한다. 드래그 앤 드롭 방식으로 구성 요소를 연결할 수 있다.
- **Playground View:** 정의된 워크플로우를 실제로 실행하고 테스트하는 공간이다. 세션을 생성하여 태스크를 수행하고, 그 결과를 실시간으로 관찰할 수 있다.
- **Gallery View:** 다른 사용자가 만들어 공유한 에이전트 아티팩트(JSON 형태)를 가져와 자신의 워크플로우에 적용할 수 있는 저장소이다.

#### 2. Backend API (FastAPI 기반)

백엔드는 REST 및 WebSocket 엔드포인트를 통해 프론트엔드와 통신하며, 다음의 핵심 클래스들이 역할을 수행한다.

- $\text{DBManager}$: 스킬, 모델, 에이전트, 워크플로우, 세션 등의 엔티티에 대해 CRUD 작업을 수행한다.
- $\text{WorkflowManager}$: JSON 형태의 선언적 명세를 읽어 실제 AutoGen 에이전트 객체로 변환(Hydration)하고 태스크를 실행한다.
- $\text{Profiler}$: 에이전트 간의 메시지를 파싱하여 비용 및 성능 메트릭을 계산한다.

### 주요 구성 요소 및 워크플로우 정의

시스템 내에서 다루는 핵심 개념은 다음과 같다.

- **Model:** 에이전트의 지능을 담당하는 LLM.
- **Skills/Tools:** 특정 작업을 수행하는 Python 함수 또는 API.
- **Memory:** 정보를 저장하고 회상하는 단기/장기 기억 장치(예: Vector DB).
- **Agent:** 모델, 스킬, 메모리가 결합된 설정 단위.
- **Workflow:** 에이전트들이 협력하는 방식(예: Autonomous Chat, Sequential Chat)과 종료 조건 등을 정의한 구성.

### 추론 및 배포 절차

구축된 워크플로우는 JSON 설정 파일로 내보낼 수 있으며, 이는 다음과 같이 활용된다.

1. **Python API:** `WorkflowManager("workflow.json").run(message="...")` 형태의 코드로 간단히 통합 가능.
2. **CLI/API Endpoint:** `autogenstudio serve` 명령어를 통해 API 서버로 구동.
3. **Docker:** 컨테이너화하여 클라우드 환경(Azure, GCP, AWS 등)에 대규모 배포 가능.

## 📊 Results

### 평가 방법 및 정량적 결과

본 연구는 정교한 벤치마크 테스트 대신, 실제 오픈소스 릴리스 후 사용자 피드백을 기반으로 하는 'In-situ, Iterative Evaluation' 방식을 채택하였다.

- **채택률:** 출시 후 5개월 동안 20만 회 이상의 다운로드 수를 기록하며 광범위한 사용성을 입증하였다.
- **사용자 피드백 분석:** GitHub에 접수된 135개 이상의 이슈를 분석하였다. 연구진은 OpenAI의 `text-embedding-3-large` 모델로 이슈 텍스트를 임베딩하고, UMAP과 K-Means 클러스터링을 통해 사용자 페인 포인트(Pain points)를 8개의 그룹으로 분류하였다.
- **개선 사항:** 분석 결과 $\rightarrow$ (a) 컴포넌트 영속성 문제 $\rightarrow$ DB 레이어 도입, (b) 도구 작성의 어려움 $\rightarrow$ 자동 도구 생성 및 IDE 통합, (c) 엔드투엔드 테스트 실패 $\rightarrow$ Build 뷰 내 테스트 버튼 추가 등의 개선이 이루어졌다.

### 정성적 사례 연구 (Persona Analysis)

'소프트웨어 엔지니어 Jack'이라는 페르소나를 통해 책 생성 자동화 워크플로우를 구축하는 과정을 시뮬레이션하였다.

- **초기 단계:** 단순한 $\text{UserProxy} \rightarrow \text{AssistantAgent}$ 구조로 시작했으나 내용이 너무 짧은 문제 발생.
- **반복 개선:** Profiler를 통해 문제를 진단하고, 역할을 분리하여 $\text{ContentAssistant}, \text{QAAssistant}, \text{ImageGeneratorAssistant}$를 포함한 $\text{GroupChat}$ 구조로 변경함으로써 품질을 높였다.
- **최종 단계:** 완성된 워크플로우를 JSON으로 내보내 API 엔드포인트로 배포하는 과정을 통해 No-code 도구가 실제 개발 사이클을 얼마나 단축시키는지 보여주었다.

## 🧠 Insights & Discussion

### 도출된 설계 패턴

연구진은 AUTOGENSTUDIO의 운영 경험을 통해 No-code MAS 도구가 갖춰야 할 4가지 설계 패턴을 제시한다.

1. **Define-and-Compose:** 복잡한 파라미터를 한 번에 설정하기보다, 개별 컴포넌트를 먼저 정의하고 이를 나중에 조합하는 방식이 사용자 경험에 유리하다.
2. **Debugging & Sensemaking:** MAS는 매우 취약(Brittle)하므로, 단순 로그가 아닌 시각적 프로파일러를 통해 에이전트의 행동을 해석할 수 있는 도구가 필수적이다.
3. **Seamless Export:** No-code로 시작하더라도 결국 실제 서비스에 통합하기 위한 코드/API 형태의 내보내기 기능이 반드시 필요하다.
4. **Collaboration & Sharing:** 템플릿 갤러리를 통해 베스트 프랙티스를 공유하는 것이 커뮤니티의 혁신 속도를 높인다.

### 한계 및 비판적 해석

- **생산 환경 준비 부족:** 논문에서도 언급되었듯이, 본 도구는 프로토타이핑에 최적화되어 있으며 인증(Authentication)이나 세밀한 보안 설정 등 실제 Production 환경에 필요한 기능은 부족하다.
- **평가의 주관성:** 정량적인 성능 향상 지표(예: 기존 코드 방식 대비 개발 시간 단축 수치 등)보다는 다운로드 수와 이슈 분석이라는 간접적인 지표에 의존하고 있어, 도구의 효율성을 객관적으로 측정했다는 점에서는 다소 미흡하다.

## 📌 TL;DR

본 논문은 Multi-Agent System 구축의 높은 진입 장벽과 디버깅의 어려움을 해결하기 위해 No-code 도구인 **AUTOGENSTUDIO**를 제안한다. 이 도구는 드래그 앤 드롭 UI, JSON 기반 선언적 명세, 시각적 프로파일링 기능을 제공하여 개발자가 코딩 없이도 빠르게 MAS를 설계하고 검증하며 배포할 수 있게 한다. 20만 회 이상의 다운로드와 사용자 피드백 분석을 통해 '정의 후 조합(Define-and-Compose)' 및 '시각적 디버깅'의 중요성을 입증하였으며, 이는 향후 자율형 에이전트 시스템의 개발 패러다임을 가속화하는 데 중요한 역할을 할 것으로 기대된다.
