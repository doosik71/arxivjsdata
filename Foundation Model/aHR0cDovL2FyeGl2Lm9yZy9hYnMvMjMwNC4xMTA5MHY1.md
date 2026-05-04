# A REFERENCE ARCHITECTURE FOR DESIGNING FOUNDATION MODEL BASED SYSTEMS

Qinghua Lu, Liming Zhu, Xiwei Xu, Zhenchang Xing, Jon Whittle (2024)

## 🧩 Problem to Solve

본 논문은 Foundation Model (FM) 기반 시스템을 설계할 때 체계적인 가이드라인이 부족하다는 문제를 해결하고자 한다. 특히 다음과 같은 세 가지 핵심 도전 과제에 집중한다.

첫째, **이동하는 경계(Moving Boundary)와 인터페이스 진화**의 문제이다. FM의 능력이 급격히 확장됨에 따라, 기존에 외부 컴포넌트로 존재하던 기능들이 FM 내부로 흡수되는 현상이 발생한다. 이는 시스템 아키텍처의 경계를 불분명하게 만들며 설계의 지속 가능성을 위협한다.

둘째, **책임감 있고 안전한 AI(Responsible and Safe AI)**의 구현 문제이다. FM의 불투명한 특성과 급격한 지능 발달로 인해 책임 소재(Accountability)의 복잡성, 신뢰성(Trustworthiness) 확보, 그리고 오남용 방지를 위한 지속적인 위험 평가가 필수적이지만, 이를 아키텍처 수준에서 어떻게 구현할지에 대한 구체적인 방법론이 부족하다.

셋째, **시스템의 적응성(Adaptability)과 수정 가능성(Modifiability)**의 확보이다. FM의 버전 업데이트나 새로운 모델의 등장에 따라 시스템을 효율적으로 변경하고 유지보수할 수 있는 구조적 설계가 필요하다.

## ✨ Key Contributions

본 논문의 핵심 기여는 FM 기반 시스템의 진화 경로를 정의하고, 이를 바탕으로 **패턴 지향적 참조 아키텍처(Pattern-oriented Reference Architecture)**를 제안한 것이다.

중심적인 직관은 FM이 단순한 도구를 넘어 시스템의 중심축이 되는 과정을 세 단계의 진화 과정으로 모델링하고, 각 단계에서 발생할 수 있는 설계 결정 사항을 패턴화하여 제공함으로써 '설계에 의한 책임감 있는 AI(Responsible-AI-by-design)'를 달성하는 것이다. 이를 위해 시스템 층(System Layer), 운영 층(Operation Layer), 공급망 층(Supply Chain Layer)으로 구성된 3계층 구조의 참조 아키텍처를 제시한다.

## 📎 Related Works

논문은 특정 관련 연구 섹션을 별도로 두지는 않았으나, 다음과 같은 기존 접근 방식과 모델들을 언급하며 본 연구의 차별점을 제시한다.

- **기존 AI 시스템**: 다수의 Narrow AI 모델과 비-AI 컴포넌트가 공존하는 형태로, 특정 작업에 특화된 모델들을 조합하여 사용하였다.
- **Socratic Models 및 PaLM-E**: FM의 결합 방식에 따라 모듈형(Chain of FMs) 구조와 단일 거대 모델(Ultra-large FM) 구조의 가능성을 보여준 사례로 언급된다.
- **HuggingGPT**: FM이 외부 컴포넌트를 연결하는 커넥터 역할을 수행하는 초기 단계의 사례로 제시된다.

본 논문은 이러한 개별 모델이나 단편적인 구현 사례를 넘어, 소프트웨어 공학적 관점에서 아키텍처의 진화 방향을 제시하고 이를 체계적으로 관리하기 위한 참조 모델을 제안한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. AI 시스템의 아키텍처 진화 (Architecture Evolution)

논문은 FM 기반 시스템이 다음과 같은 세 단계를 거쳐 진화한다고 분석한다.

- **현재 (Architecture now)**: 많은 수의 Narrow AI 모델과 Non-AI 컴포넌트가 상호작용하는 구조이다.
- **5년 후 (FM-as-a-connector)**: 하나의 FM이 중심이 되어 다른 Narrow 모델이나 Non-AI 컴포넌트를 연결하는 커넥터 역할을 수행한다. (통신, 조정, 변환, 촉진 커넥터 기능 수행)
- **10년 후 (Alternative 1 & 2)**:
  - **Alternative 1**: 소수의 FM들이 체인 형태로 연결된 모듈형 구조이며, Prompt Engineering이 핵심이 된다.
  - **Alternative 2**: 모든 기능이 통합된 하나의 Ultra-large FM이 지배하는 단일 구조(Monolithic Architecture)로 진화한다.

### 2. 주요 설계 결정 사항 (Architectural Design Decisions)

개발자가 FM 기반 시스템을 설계할 때 고려해야 할 7가지 결정 지점을 정의한다.

- **FM 소싱**: 자체 개발(Sovereign), 외부 도입(External), 또는 외부 모델의 맞춤형 튜닝(Customized) 중 선택한다.
- **구조 선택**: 모델 체인(Chain of FMs)을 통한 유지보수성 확보와 단일 거대 모델(Ultra-large FM)을 통한 성능 확보 사이의 트레이드오프를 결정한다.
- **책임 분리**: 컴포넌트의 책임을 작게 쪼개어 FM에 흡수되더라도 영향도를 최소화하는 전략을 취한다.
- **검증 체계**: 자동 응답(Automatic response)과 검증자(Verifier) 도입 간의 선택을 통해 신뢰성을 조절한다.
- **상호작용 방식**: 사용자의 입력을 기다리는 수동적 상호작용(Passive)과 멀티모달 컨텍스트를 분석해 제안하는 능동적 상호작용(Proactive)을 결정한다.
- **에이전트 구성**: 단일 에이전트와 다중 에이전트(Multi-agents) 구성 중 복잡도와 성능을 고려해 선택한다.
- **투명성 수준**: 사고 과정을 공개하는 'Think aloud'와 내부적으로 처리하는 'Think silently' 방식을 선택한다.

### 3. 패턴 지향적 참조 아키텍처 (Reference Architecture)

제안하는 아키텍처는 세 개의 레이어로 구성된다.

#### (1) 시스템 층 (System Layer)

실제 배포된 시스템의 구성 요소이다.

- **Interaction components**: 멀티모달 컨텍스트 엔지니어링과 Prompt Optimiser를 통해 사용자 의도를 파악하고 최적의 프롬프트를 생성한다.
- **FMs**: 다양한 형태의 FM(Sovereign, External, Fine-tuned 등)이 배치된다.
- **Adaptability Patterns**: FM에 의한 기능 흡수 문제(Moving boundary)를 해결하기 위해 **Microkernel 패턴**과 **Adapter 패턴**을 적용하여 구성 요소 간의 결합도를 낮춘다.
- **Agents**: Coordinator와 Worker 역할로 나뉜 에이전트들이 목표를 수행하며, API 기반의 거버넌스를 통해 오남용을 방지한다.
- **Data**: RAG(Retrieval Augmented Generation)를 위한 벡터 데이터베이스와 기밀 유지를 위한 Federated RAG를 포함한다.

#### (2) 운영 층 (Operation Layer)

책임감 있는 AI를 위한 모니터링 및 관리 도구이다.

- **Verifier & Guardrails**: 전처리, 중간 처리, 후처리 단계에서 입력과 출력의 신뢰성을 검증하고 유해한 콘텐츠를 필터링한다.
- **Continuous Risk Assessor**: 실시간으로 AI 리스크 메트릭을 모니터링하여 위험을 평가하고 프롬프트를 수정하거나 거절한다.
- **AgentOps**: 모든 입력, 출력 및 중간 단계 데이터를 기록하는 로그 저장소로, 추적 가능성(Traceability)을 보장한다.

#### (3) 공급망 층 (Supply Chain Layer)

AI 컴포넌트의 개발 및 조달 과정이다.

- **Training/Tuning**: RLHF(Reinforcement Learning from Human Feedback)와 LoRA와 같은 매개변수 효율적 미세 조정(PEFT) 기술을 통해 모델의 성능과 책임감을 높인다.
- **Registry**: 도구/모델 레지스트리와 AIBOM(AI Bill of Materials) 레지스트리를 통해 공급망의 투명성을 확보하고, 검증된 컴포넌트만 사용하도록 제어한다.
- **Co-versioning**: 모델, 데이터셋, 아티팩트 간의 버전을 함께 관리하여 감사 가능성(Auditability)을 높인다.

## 📊 Results

본 논문은 제안한 참조 아키텍처의 완전성과 유용성을 평가하기 위해 실제 구축된 **RAI(Responsible AI) 챗봇** 시스템에 매핑하는 실험을 진행하였다.

- **대상 시스템**: 과학자들이 AI 프로젝트의 잠재적 리스크를 평가할 수 있도록 돕는 챗봇.
- **매핑 결과**:
  - **System Layer**: GPT-4(External FM) 사용, LlamaIndex를 통한 RAG 구현, Think-aloud 패턴 적용, 다중 에이전트 구성.
  - **Operation Layer**: 인간 전문가에 의한 Verifier 패턴 구현, 블랙박스 기록 장치를 통한 데이터 캡처.
  - **Supply Chain Layer**: AIBOM 레지스트리 구축 및 FM 미세 조정 논의 단계.

실험 결과, RAI 챗봇의 모든 구성 요소가 제안된 참조 아키텍처의 레이어 및 패턴과 성공적으로 매핑되었으며, 이를 통해 본 아키텍처가 FM 기반 시스템의 설계 지침으로서 완전하고 사용 가능하다는 것을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 FM 기반 시스템 설계에서 **적응성(Adaptability)**과 **수정 가능성(Modifiability)**이라는 소프트웨어 품질 속성이 핵심임을 역설한다. FM의 기능 확장으로 인해 외부 컴포넌트가 모델 내부로 흡수되는 'Moving Boundary' 현상은 피할 수 없는 흐름이며, 이를 해결하기 위해 Microkernel이나 Adapter와 같은 전통적인 소프트웨어 공학 패턴을 AI 아키텍처에 도입한 점이 매우 고무적이다.

또한, 책임감 있는 AI를 단순한 윤리적 가이드라인이 아니라 **운영 층(Operation Layer)**과 **공급망 층(Supply Chain Layer)**이라는 구체적인 아키텍처 컴포넌트(Guardrails, AIBOM, Verifier 등)로 구체화하여 '설계에 의한 책임감(RAI-by-design)'을 구현하려 한 점이 돋보인다.

다만, 본 논문은 정량적인 성능 지표보다는 아키텍처의 구조적 타당성과 매핑 가능성에 집중하고 있다. 실제 이러한 아키텍처를 적용했을 때 유지보수 비용이 얼마나 절감되는지, 혹은 리스크 탐지율이 얼마나 향상되는지에 대한 정량적 분석이 추가된다면 더 강력한 설득력을 가질 수 있을 것이다.

## 📌 TL;DR

본 논문은 급격히 진화하는 Foundation Model(FM) 기반 시스템의 설계 혼란을 해결하기 위해, **시스템-운영-공급망의 3계층 구조로 이루어진 패턴 지향적 참조 아키텍처**를 제안한다. 특히 FM의 기능 확장에 따른 아키텍처 경계 변화 문제를 소프트웨어 공학 패턴으로 해결하고, 전 과정에 걸친 Guardrails와 AIBOM 등을 통해 책임감 있는 AI 시스템을 설계할 수 있는 구체적인 가이드를 제공한다. 이 연구는 향후 복잡한 AI 에이전트 시스템의 표준 설계 템플릿으로 활용될 가능성이 높다.
