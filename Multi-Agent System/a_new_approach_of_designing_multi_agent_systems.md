# A new approach of designing Multi-Agent Systems With a practical sample

Sara Maalal, Malika Addou (2011)

## 🧩 Problem to Solve

본 논문은 복잡한 분산 애플리케이션을 구현하기 위한 소프트웨어 패러다임인 Multi-Agent Systems (MAS)의 설계 및 구현 과정에서 발생하는 복잡성 문제를 해결하고자 한다. MAS는 다수의 에이전트가 상호작용하며 환경에 영향을 미치는 구조를 가지므로, 시스템의 여러 부분을 서로 다른 관점에서 접근해야 하며 이를 하나의 일관된 시스템으로 통합하는 과정이 매우 어렵다.

기존의 소프트웨어 공학 방법론들은 분석 단계에 치중되어 있으며, 설계에서 실제 구현 단계로 넘어가는 과정에서 상당한 간극(gap)이 존재한다. 특히, 조직적 개념을 모델링하는 도구가 부족하거나 특정 에이전트 모델에 종속되는 경향이 있어, 범용적이고 확장 가능한 구현 방법론이 절실한 상황이다. 따라서 본 논문의 목표는 Model Driven Architecture (MDA)와 Agent Unified Modeling Language (AUML)를 결합하여, 모델로부터 소스 코드를 자동으로 생성할 수 있는 범용적인 MAS 설계 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 AUML을 기반으로 한 범용 클래스 메타 모델(Generic Class Meta-model)을 설계하고, 이를 MDA 접근 방식을 통해 실제 구현 코드로 변환하는 파이프라인을 구축하는 것이다.

가장 중점적인 기여는 에이전트의 유형(Reactive, Cognitive, Intentional 등)과 환경(Environment) 간의 관계를 체계화한 계층적 클래스 다이어그램을 제안한 점이다. 이를 통해 설계자는 세부적인 기술적 구현에 매몰되지 않고 고수준의 모델링에 집중할 수 있으며, AndroMDA와 같은 도구를 사용하여 UML 모델을 Java 소스 코드로 자동 변환함으로써 개발 시간과 비용을 획기적으로 줄이고 모듈성과 재사용성을 높일 수 있다.

## 📎 Related Works

논문에서는 MAS 분석 및 설계를 위한 다양한 기존 방법론들을 소개하며 그 한계를 지적한다.

- **AAII:** BDI(Belief, Desire, Intention) 시스템 구축 경험을 바탕으로 템플릿을 제공한다.
- **Gaia:** 에이전트를 조직 사회의 구성원으로 모델링하며 역할(Role)과 프로토콜을 정의하지만, 지원하는 에이전트 아키텍처가 제한적이라는 한계가 있다.
- **MESSAGE 및 INGENIAS:** 조직, 에이전트, 목표, 상호작용, 환경의 5가지 관점에서 MAS를 정의한다. INGENIAS는 코드 생성 도구(IDK)를 제공하지만, 사회적 규범이나 동적인 조직 변화(에이전트의 가입/탈퇴 등)를 명시적으로 모델링하지 못한다.
- **MaSE:** 분석부터 구현까지 포괄하는 end-to-end 방법론으로 특정 아키텍처에 의존하지 않는 것을 목표로 한다.
- **AUML 및 AML:** UML을 확장하여 에이전트 지향 모델링을 지원하는 표준 및 시각적 언어이다.
- **ASPECS:** Holonic 관점을 도입하여 복잡한 시스템을 계층적 구조의 Holons로 설계한다.

저자는 이러한 기존 방법론들이 주로 분석 단계에 집중되어 있어, 실제 구현 단계와의 연결 고리가 부족하며 실세계 애플리케이션에 적용된 사례가 여전히 적다는 점을 비판적으로 분석하였다.

## 🛠️ Methodology

본 논문에서 제안하는 방법론은 MDA의 프레임워크를 따르며, AUML을 통해 설계된 메타 모델을 UML로 변환하고 최종적으로 소스 코드를 생성하는 흐름을 가진다.

### 1. MDA (Model Driven Architecture) 접근 방식

MDA는 기술적 세부 사항에 구애받지 않고 기능과 동작에 집중하는 세 단계의 모델 변환 과정을 거친다.

- **CIM (Computation Independent Model):** 자동화와 독립적인 비즈니스 프로세스를 기술한다.
- **PIM (Platform Independent Model):** 기술 아키텍처와 독립적인 세부 기능 분석 모델이다.
- **PSM (Platform Specific Model):** PIM을 특정 타겟 플랫폼(예: Java)에 투영하여 얻은 설계 모델이며, 이를 통해 코드가 생성된다.

### 2. AUML 기반 범용 클래스 다이어그램 구조

제안된 메타 모델은 크게 세 가지 계층의 관계 모델로 구성된다.

**첫 번째 계층: 에이전트와 환경의 상호작용**

- **Environment:** 시스템 전체에 영향을 미치며, 속성(Deterministic/Non-deterministic, Static/Dynamic, Continuous/Discrete)과 인지(Perception) 섹션으로 구성된다. 주요 함수로 $\text{Run}()$, $\text{Perceive}()$, $\text{ModifState}()$를 가진다.
- **Agent:** 역할(Roles), 속성(Attributes), 인지(Perception)를 가지며, $\text{Run}()$, $\text{Perceive}()$, $\text{Act}()$ 함수를 수행한다.
- **관계 클래스:** 에이전트와 환경 사이의 $\text{Action}$, 에이전트 간의 $\text{Interaction}$ 클래스를 정의하여 $\text{getInformation}()$ 및 $\text{inform}()$ 등의 상호작용을 처리한다.

**두 번째 계층: 에이전트의 전문화 (Specialization)**

- **Reactive Agent:** 단순한 자극-반응 구조를 가지며 환경을 인지하고 행동한다.
- **Cognitive Agent:** 지식의 상징적 표현을 가지며, 목표에 따라 행동 여부를 결정하는 $\text{Decide}()$ 함수를 포함한다.
- **Communicative Agent:** 정보를 전달하는 데 특화되어 있으며 $\text{Communicate}()$ 함수를 사용한다.

**세 번째 계층: 인지 에이전트의 세부 전문화**

- **Adaptive Agent:** 환경 변화에 따라 목표와 지식 베이스를 변경하는 $\text{Change\_information}()$ 함수를 가진다.
- **Intentional Agent (BDI):** 신념(Beliefs), 욕구(Desires), 의도(Intentions) 모델을 따른다. $\text{Revise\_beliefs}() \rightarrow \text{Generate\_desires}() \rightarrow \text{Filter}() \rightarrow \text{Actions\_selection}()$ 의 과정을 통해 행동을 결정한다.
- **Rational Agent:** 성능 측정 함수 $\text{Mesure\_performance}(\text{Percept, Belief})$를 통해 가장 효율적인 행동을 선택한다.

### 3. 구현 파이프라인

AUML 다이어그램 $\rightarrow$ UML 클래스 다이어그램 $\rightarrow$ AndroMDA $\rightarrow$ Java 코드 생성의 순서로 진행된다. AndroMDA는 UML 모델을 입력받아 Spring Framework 기반의 비즈니스 레이어, Hibernate 기반의 데이터 액세스 레이어 등을 자동으로 생성한다.

## 📊 Results

본 방법론의 타당성을 검증하기 위해 '채팅 애플리케이션(Chat Application)'을 예제로 구현하였다.

- **설계 내용:** 3개의 Reactive Agent(채팅 사용자)를 설정하였다. 이 에이전트들은 사용자가 수신자 이름을 지정할 때까지 반응하지 않으며, 이름과 메시지를 수신하여 적절한 대상에게 전달하고 인터페이스를 갱신하는 역할을 수행한다.
- **실험 절차:**
    1. MagicDraw 17을 사용하여 제안된 메타 모델 기반의 UML 클래스 다이어그램을 작성한다.
    2. 작성된 모델을 EMF-UML2 형식으로 내보낸다.
    3. AndroMDA 및 Maven 플러그인을 사용하여 코드를 생성한다. (`mvn org.andromda.maven.plugins:andromdaapp-maven-plugin:generate` 명령 실행)
- **결과물:**
  - `ChatAgents`라는 마스터 프로젝트 아래에 `mda` (UML 모델 및 설정), `common` (공유 리소스), `core` (Spring/Hibernate 기반 서비스), `web` (프레젠테이션 레이어), `app` (.ear 번들) 등의 서브 프로젝트가 자동으로 생성되었다.
  - 특히 `Chat.java`와 같은 핵심 클래스 파일이 생성되어, 개발자가 생성된 코드 내에서 세부 비즈니스 로직을 쉽게 구현할 수 있음을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 MDA와 AUML의 결합을 통해 MAS 설계의 추상화 수준을 높이고 구현 자동화를 달성했다는 점에서 강점을 가진다. 특히, 에이전트의 이론적 분류(BDI, Rational 등)를 클래스 계층 구조로 명확히 정의하여, 설계자가 이론적 배경을 바탕으로 실제 코드를 생성할 수 있는 가이드라인을 제공하였다.

그러나 다음과 같은 한계점과 논의 사항이 존재한다.
첫째, AndroMDA가 생성하는 코드의 품질과 복잡성 문제이다. 저자 역시 모델 설계 단계에서 매우 높은 신뢰성이 확보되지 않으면 생성된 코드에서 오류가 발생할 가능성이 크다고 언급하였다.
둘째, 제시된 예제인 채팅 애플리케이션은 가장 단순한 형태인 Reactive Agent만을 사용하였다. 제안한 메타 모델의 진정한 효용성을 입증하기 위해서는 BDI나 Adaptive Agent와 같은 고수준의 인지 능력이 필요한 복잡한 시나리오에서의 검증이 추가로 필요하다.
셋째, Java 외에 C++나 웹 서비스 등 다른 플랫폼으로의 확장 가능성에 대한 구체적인 방법론은 제시되지 않았으며, 이는 향후 과제로 남아 있다.

## 📌 TL;DR

본 논문은 MAS 설계 시 발생하는 분석과 구현 사이의 간극을 줄이기 위해, AUML 기반의 범용 에이전트 메타 모델을 제안하고 이를 MDA(Model Driven Architecture) 프레임워크 및 AndroMDA 도구를 통해 Java 코드로 자동 변환하는 방법론을 제시하였다. 채팅 애플리케이션 사례를 통해 모델로부터의 코드 생성 가능성을 입증하였으며, 이는 향후 복잡한 분산 에이전트 시스템의 개발 비용 절감과 재사용성 향상에 기여할 수 있을 것으로 기대된다.
