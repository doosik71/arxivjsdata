# SuperCoder2.0: Technical Report on Exploring the feasibility of LLMs as Autonomous Programmer

Anmol Gautam, Kishore Kumar, Adarsh Jha, Mukunda NS, Ishaan Bhola (SuperAGI Research)

## 🧩 Problem to Solve

본 논문은 대규모 소프트웨어 코드베이스에서 인공지능이 인간의 개입 없이 독립적으로 버그를 수정하고 기능을 구현하는 '자율 프로그래머(Autonomous Programmer)'의 실현 가능성을 탐구한다. 

기존의 AI 기반 코딩 도구들은 다음과 같은 핵심적인 문제들에 직면해 있다. 첫째, 방대한 코드베이스 내에서 문제의 근본 원인이 되는 정확한 위치를 찾아내는 '버그 로컬라이제이션(Bug Localization)'의 어려움이다. 둘째, LLM이 생성한 코드를 기존 코드에 삽입할 때, 특히 파이썬(Python)과 같이 들여쓰기에 민감한 언어에서 발생하는 린팅(Linting) 문제와 구문 오류이다. 셋째, 단순한 코드 생성을 넘어 실제 저장소 수준의 테스트 케이스를 통과할 수 있는 신뢰성 있는 솔루션을 도출하는 프로세스의 부재이다.

따라서 본 연구의 목표는 효율적인 코드베이스 탐색 전략과 구조적 코드 수정 방식을 결합하여, 실제 소프트웨어 엔지니어링 작업(SWE-bench Lite 기준)을 자율적으로 해결할 수 있는 시스템인 SuperCoder2.0을 구축하는 것이다.

## ✨ Key Contributions

SuperCoder2.0의 핵심 아이디어는 **'계층적 탐색 공간 축소(Hierarchical Search Space Reduction)'**와 **'구조 기반의 코드 재작성(Structure-aware Code Rewriting)'**이다. 

단순히 특정 라인 번호를 찾는 대신 메서드나 클래스 단위의 '관련 위치(Relevant Location)'를 식별함으로써 추상화 수준을 높여 문제 해결 능력을 향상시켰다. 또한, 부분적인 코드 삽입 대신 추상 구문 트리(Abstract Syntax Tree, AST) 파싱을 통해 메서드나 클래스 전체를 교체하는 방식을 채택하여 코드의 무결성을 유지하고 린팅 오류를 최소화하였다. 마지막으로, 다양한 온도(Temperature) 설정으로 여러 후보 솔루션을 생성하고 이를 저장소 수준의 테스트 케이스로 검증하는 피드백 루프를 도입하여 솔루션의 강건성을 확보하였다.

## 📎 Related Works

본 논문은 LLM 기반의 자율 소프트웨어 엔지니어링 분야의 최신 흐름을 다룬다. GPT-4와 Claude와 같은 강력한 LLM의 등장과 더불어, ReACT(Reasoning and Acting) 및 Chain-of-Thought(COT)와 같은 추론 기법이 복잡한 코딩 작업의 정확도를 높이는 데 기여했음을 언급한다.

특히, OpenDevin이나 Agentless와 같은 최신 도구들과의 차별점을 강조한다. 기존의 에이전트 기반 접근 방식(Agent-based approach)은 복잡한 상호작용으로 인해 오버헤드가 크고 효율성이 떨어지는 경향이 있다. 반면, SuperCoder2.0은 Agentless 팀의 발견을 수용하여, 반복적인 '추론-행동-관찰' 루프보다는 명확한 컨텍스트와 타겟 질문을 통해 단일 LLM 호출로 효율적인 결과를 얻는 전략을 취한다. 또한, 단순한 시맨틱 검색을 넘어 저장소의 구조적 맵(Repository Map)을 결합하여 검색의 정확도를 높인 점이 기존 접근 방식과의 차별점이다.

## 🛠️ Methodology

SuperCoder2.0의 전체 파이프라인은 크게 **탐색(Search)** 단계와 **수정(Edit)** 단계로 구성된다.

### 1. Search Phase (계층적 탐색)
문제의 근본 원인이 되는 '관련 위치'를 찾기 위해 세 단계의 계층적 축소 전략을 사용한다.

1.  **후보 파일 식별**: 
    - **RAG (Retrieval Augmented Generation)**: Jina Code Embeddings를 사용하여 메서드 이름, 시그니처, 리턴 문, Docstring을 벡터화하여 저장한다. 쿼리 생성 모듈이 생성한 $N$개의 쿼리로 관련 파일을 검색한다.
    - **Repository File Level Map**: 전체 코드베이스의 파일 구조를 재귀적으로 파싱하여 맵을 생성하고, 이를 통해 상위 $M$개의 파일을 식별한다. 
    - 최종 후보 파일 집합은 위 두 방식의 합집합(Union)으로 결정된다.
2.  **파일 우선순위 선정**: 
    - 후보 파일들에 대해 클래스 이름, 메서드, 인자, 데코레이터 등을 포함한 **File Level Schematic Map**을 생성한다. 
    - PreAssimilation 모듈이 이 맵을 분석하여 가장 관련성이 높은 상위 $L$개(SWE-bench Lite에서는 최대 2개)의 파일을 최종 선택한다.
3.  **관련 위치(Relevant Location) 추출**: 
    - CoderParser 모듈이 파일 전체 내용과 문제 정의를 분석하여 수정이 필요한 위치를 **Top-Level, Class-Level, Method/Function-Level**의 세 단계 중 하나로 지정하고 상세 수정 계획을 수립한다.

### 2. Edit Phase (코드 수정 및 삽입)
식별된 위치에 실제로 코드를 적용하는 단계로, `CodeGeneration` 모듈과 `CodeEditing` 모듈로 나뉜다.

- **CodeGeneration**: LLM을 사용하여 수정된 코드 세그먼트를 생성한다. 이때 다양한 온도(Temperature) 값을 설정하여 $k$개의 서로 다른 후보 솔루션을 생성함으로써 해결책의 다양성을 확보한다.
- **CodeEditing**: LLM 없이 Python의 **AST(Abstract Syntax Tree)** 라이브러리를 사용하여 실제 코드를 조작한다.
    - **Method/Class 수준**: 해당 메서드나 클래스의 전체 본문을 새로 생성된 코드로 완전히 교체한다. 이는 부분 수정 시 발생하는 들여쓰기 오류를 원천적으로 차단한다.
    - **Top-Level 수준**: 지정된 시작 및 종료 라인 사이의 코드를 통째로 교체한다.

### 3. Iterative Feedback and Refinement
솔루션의 신뢰성을 높이기 위해 다음과 같은 루프를 수행한다.
1. **Baseline 측정**: 수정 전 모든 저장소 테스트 케이스를 실행하여 통과/실패 상태를 기록한다.
2. **Post-Evaluation**: 생성된 $k$개의 솔루션을 각각 적용한 후 다시 모든 테스트를 실행한다. 기존에 통과했던 테스트가 실패로 변한 솔루션은 즉시 제외한다.
3. **Refinement**: 테스트 실패 시, 실패한 테스트 케이스의 정보와 관련 코드 세그먼트를 다시 LLM에 제공하여 단 한 번의 추가 수정을 시도하는 피드백 루프를 가동한다.

## 📊 Results

### 실험 설정
- **데이터셋**: SWE-bench Lite (실제 GitHub 이슈 300개로 구성된 부분 집합)
- **사용 모델**: GPT-4, Claude 3.5 Sonnet
- **측정 지표**: 해결률(Resolution Rate), 파일 레벨 로컬라이제이션 정확도(File-Level Localization Accuracy)

### 주요 결과
- **해결률 (Resolution Rate)**: 전체 인스턴스의 **34%**를 성공적으로 해결하였다. 이는 Table I에 제시된 바와 같이 Bytedance MarsCode Agent와 대등하며, AutoCodeRover(30.67%)나 Amazon Q Developer Agent(29.67%)보다 높은 성능이다.
- **로컬라이제이션 정확도**: 상위 5개 후보 파일 내에 정답 파일이 포함될 확률이 **84.33%**에 달해, RAG와 Repository Map의 결합이 매우 효과적임을 입증하였다.
- **정성적 분석**: 라인 단위의 수정보다 메서드 전체를 교체하는 방식이 파이썬의 린팅 문제를 해결하는 데 훨씬 효율적임을 확인하였다.

## 🧠 Insights & Discussion

SuperCoder2.0의 성과는 LLM이 코드의 구조적 패턴을 이해하는 능력이 뛰어나다는 점을 활용한 결과이다. 특히, 정밀한 라인 번호 추적보다 **메서드 단위의 추상화된 접근**이 복잡한 버그 해결에 더 유리하다는 인사이트를 제공한다. 이는 LLM의 생성 토큰 제한(약 4,000 토큰) 내에서 대부분의 표준 메서드가 포함될 수 있다는 실용적인 가설과도 일치한다.

또한, 단순한 시맨틱 검색(RAG)만으로는 대규모 저장소의 맥락을 모두 파악하기 어렵지만, 이를 구조적 맵(Repository Map)과 결합했을 때 로컬라이제이션 성능이 비약적으로 상승함을 보여주었다.

**한계 및 논의사항**:
- 현재 피드백 루프가 단 1회로 제한되어 있어, 매우 복잡한 의존성을 가진 버그의 경우 추가적인 반복 수정이 필요할 수 있다.
- 파이썬 외의 다른 프로그래밍 언어에 대해서는 AST 기반 교체 전략이 어떻게 작동할지에 대한 검증이 이루어지지 않았다.
- 솔루션 선택 과정에서 여전히 LLM의 판단에 의존하는 부분이 있어, 결정론적인 검증 메커니즘의 추가 도입이 필요할 것으로 보인다.

## 📌 TL;DR

SuperCoder2.0은 **계층적 탐색(RAG $\rightarrow$ Schematic Map $\rightarrow$ Location)**과 **AST 기반의 메서드 단위 코드 교체**를 통해 자율적으로 소프트웨어 버그를 수정하는 시스템이다. SWE-bench Lite에서 **34%의 해결률**과 **84.33%의 파일 탐색 정확도(Top-5)**를 기록하며 글로벌 리더보드 상위권의 성능을 보였다. 이 연구는 자율 프로그래밍 시스템에서 '정밀한 위치 찾기'보다 '구조적 단위의 교체'가 더 안정적이고 효율적임을 입증했으며, 향후 AI 기반의 자율 소프트웨어 엔지니어링 도구 개발에 중요한 기준을 제시한다.