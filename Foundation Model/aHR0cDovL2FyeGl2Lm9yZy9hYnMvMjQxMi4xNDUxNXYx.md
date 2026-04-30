# Relational Programming with Foundation Models

Ziyang Li, Jiani Huang, Jason Liu, Felix Zhu, Eric Zhao, William Dodds, Neelay Velingker, Rajeev Alur, Mayur Naik (2024)

## 🧩 Problem to Solve

본 논문은 Foundation Models(FM)가 가진 강력한 능력에도 불구하고, 이를 실제 AI 애플리케이션에 통합하여 사용할 때 발생하는 근본적인 한계점들을 해결하고자 한다. 구체적인 문제는 다음과 같다.

첫째, 언어 모델(LM)의 고질적인 문제인 Hallucination(환각) 현상으로 인해 사실과 다른 주장을 하거나 잘못된 추론 체인을 생성하는 경우가 빈번하다. 둘째, 현대 데이터베이스의 주류 형태인 정형 데이터(Structured Data)를 신뢰성 있게 통합하고 처리하는 능력이 부족하다. 셋째, 서로 다른 데이터 모달리티(Modality)를 복잡한 패턴으로 조합하여 다중 모달(Multi-modal) 애플리케이션을 구축하는 것이 여전히 어려운 과제로 남아 있다.

따라서 본 연구의 목표는 이러한 개별적인 보완 메커니즘(In-context learning, 정보 검색, 코드 해석 등)을 하나의 일반적인 솔루션으로 통합하는 선언적 프레임워크인 VIEIRA를 제안하는 것이다.

## ✨ Key Contributions

VIEIRA의 핵심 아이디어는 Foundation Models를 **입력과 출력이 관계형(Relational)인 상태 없는 함수(Stateless Functions)**로 취급하는 것이다. 

중심적인 설계 직관은 관계형 프로그래밍 패러다임을 추상화 계층으로 사용하여, 신경망 기반의 Foundation Models와 기호적(Symbolic) 논리 프로그램을 매끄럽게 결합하는 것이다. 이를 통해 복잡한 추론은 논리 프로그램이 담당하고, 개별적인 인식이나 텍스트 생성은 FM이 담당하게 함으로써 Hallucination을 줄이고 정밀한 제어를 가능하게 한다. 또한, 확장 가능한 플러그인 라이브러리를 통해 다양한 FM을 쉽게 통합할 수 있는 구조를 설계하였다.

## 📎 Related Works

논문은 기존의 접근 방식을 세 가지 범주로 나누어 설명한다.

1.  **Neuro-symbolic methods**: DeepProbLog나 SCALLOP과 같이 신경망 학습과 기호적 추론을 결합한 기존 연구들이 존재한다. 그러나 이들은 주로 신경망 모델의 학습이나 미세 조정(Fine-tuning)에 집중하는 반면, VIEIRA는 Zero-shot 또는 Few-shot 설정에서 이미 학습된 Foundation Models를 활용하여 애플리케이션을 구축하는 데 중점을 둔다.
2.  **Foundation models**: Chain-of-Thought(CoT)나 ReAct와 같은 프롬프팅 기법들이 추론 능력을 향상시켰으나, VIEIRA는 이러한 기법들과는 독립적으로 작동하며, 이를 논리적 프레임워크 내에서 통합하여 견고성을 높이는 상위 계층의 솔루션을 제공한다.
3.  **Tools aiding language models**: PAL, Toolformer, AutoGPT 등 LM이 외부 도구나 API를 호출하게 하는 시도들이 있었다. VIEIRA는 이러한 전략들을 통합하여 다중 모달 FM을 조합하기 위한 '접착 언어(Glue language)' 역할을 수행한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
VIEIRA는 Datalog 기반의 선언적 논리 프로그래밍 언어를 사용하며, SCALLOP 컴파일러를 확장하여 Foundation Models를 플러그인 형태로 지원하는 외국어 인터페이스(Foreign Interface)를 구축하였다.

### 핵심 구성 요소 및 언어 특징
1.  **관계 및 데이터 타입**: 기본적으로 정적 타입의 튜플들로 구성된 집합 기반의 관계(Relation)를 사용한다. 특히 FM과의 통합을 위해 `Tensor` 타입과 `Algebraic Data Types (ADTs)`를 도입하였다. ADT는 VQA(Visual Question Answering)와 같은 도메인 특화 언어(DSL)를 정의하는 데 사용된다.
2.  **논리 추론 (Logical Reasoning)**: Horn rules를 지원하여 연접(Conjunction), 이접(Disjunction), 재귀(Recursion), 층화 부정(Stratified Negation) 및 집계(Aggregation)를 수행할 수 있다.
3.  **확률적 소프트 논리 (Probabilistic Soft Logic)**: 각 튜플에 확률 값을 부여할 수 있다. 특히 `Tensor` 타입에 대해 코사인 유사도를 계산하는 소프트 등가 연산자 $\tilde{=}$ (soft-eq)를 도입하여, 시맨틱 검색과 같은 Soft-join 연산을 가능하게 한다.

### 외국어 인터페이스 (Foreign Interface)
FM을 VIEIRA에 통합하기 위해 두 가지 핵심 구조를 사용한다.

-   **Foreign Predicate (FP)**: 외부 함수를 호출하여 사실(Facts)을 생성하는 술어이다. 입력으로 Bounded arguments를 받고, 결과로 Free arguments의 리스트를 반환한다.
-   **Foreign Attribute (FA)**: 술어 선언을 장식하는 고차 함수이다. 모델의 설정값이나 프롬프트 템플릿을 숨기고 단순화된 인터페이스를 제공한다. (예: `@gpt`, `@clip`)

### 학습 및 추론 절차
본 프레임워크는 별도의 추가 학습을 필요로 하지 않는 **No-training** 방식을 지향한다. 추론 과정은 다음과 같다.
1.  **데이터 추출**: FM(예: GPT-4)을 사용하여 비정형 텍스트나 이미지에서 정형 관계(Relation)를 추출한다.
2.  **기호적 추론**: 추출된 관계들을 바탕으로 VIEIRA의 논리 규칙(Rules)에 따라 결론을 도출한다.
3.  **결과 반환**: 최종적으로 쿼리에 해당하는 답을 생성하거나, 이미지 생성 모델의 경우 최종 텐서를 출력한다.

## 📊 Results

### 실험 설정
-   **데이터셋 및 작업**: Date Reasoning(DR), Tracking Shuffled Objects(TSO), Kinship Reasoning(KR), Math Reasoning(MR), HotpotQA, Amazon ESCI Product Search, GQA, CLEVR, VQAR, OFCP, IGP20 등 총 9가지 벤치마크 작업을 수행하였다.
-   **사용 모델**: GPT-4, CLIP, SAM, OWL-ViT, ViLT, Stable Diffusion 등 12종의 모델을 플러그인으로 사용하였다.
-   **평가 지표**: Exact Match(EM), nDCG, Recall@k, Manual Inspection(MI) 등을 사용하였다.

### 주요 결과
1.  **자연어 추론**: TSO 작업에서 100% 정확도를 달성하였으며, DR과 KR에서도 GPT-4의 Zero-shot CoT보다 우수한 성능을 보였다. 특히 추론 체인의 길이($k$)가 길어지거나 추적 대상 객체 수($n$)가 늘어나도 성능이 일정하게 유지되는 강한 일반화 능력을 보였다.
2.  **정보 검색 및 검색**: HotpotQA에서 미세 조정된 모델(Fine-tuned)과 유사한 성능을 보였으며, Amazon ESCI에서는 단순 임베딩 기반 검색(MIPS)보다 높은 nDCG를 기록하였다.
3.  **다중 모달 추론 (VQA)**: GQA와 CLEVR 데이터셋에서 ViLT-VQA 및 PNP-VQA 대비 월등한 성능 향상을 보였다. 특히 카운팅이나 수치 비교와 같은 복잡한 논리 연산에서 강점을 나타냈다.
4.  **이미지 생성 및 편집**: IGP20 데이터셋에서 InstructPix2Pix보다 지시 사항을 더 정확하게 준수하는 결과를 보였으며, OFCP 이미지 편집에서 74%의 시맨틱 정확도를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
VIEIRA의 가장 큰 강점은 **신경망의 인식 능력과 기호적 시스템의 추론 능력을 명확히 분리**했다는 점이다. 
-   **Hallucination 억제**: LLM이 직접 답을 내게 하지 않고, 사실 추출(Extraction) 단계만 담당하게 한 뒤 실제 계산과 추론은 논리 엔진이 수행함으로써 환각 현상을 획기적으로 줄였다.
-   **해석 가능성 및 디버깅**: 모든 중간 단계의 관계(Intermediate relations)를 확인할 수 있어, 어느 단계에서 오류가 발생했는지 체계적으로 분석할 수 있다.
-   **사용 편의성**: 논리 프로그래밍 배경이 없는 학부생들이 대부분의 솔루션을 구현했을 정도로 사용성이 높으며, 코드의 양(LoC)이 매우 간결하다.

### 한계 및 논의사항
-   **시맨틱 파싱 의존성**: GSM8K와 같은 수학 추론 작업에서는 성능이 약간 낮게 나타났는데, 이는 LLM이 정형화된 계산 단계(Structured computation steps)를 추출하는 과정에서 발생하는 오류에 민감하기 때문이다.
-   **프롬프트 설계**: 여전히 모델의 성능이 추출 단계의 프롬프트 설계와 Few-shot 예제에 의존하는 경향이 있다.

## 📌 TL;DR

본 논문은 Foundation Models를 관계형 함수로 정의하고 이를 논리 프로그램과 결합한 선언적 프레임워크 **VIEIRA**를 제안한다. 이 연구는 LLM의 환각 문제를 기호적 추론으로 해결하고, 다중 모달 모델들을 하나의 언어(Datalog 확장판)로 통합하여 제어할 수 있음을 입증하였다. 9가지의 다양한 벤치마크에서 기존 신경망 전용 모델이나 단순 프롬프팅 기법보다 우수하거나 대등한 성능을 보였으며, 특히 복잡한 논리 구조를 가진 작업에서 뛰어난 일반화 능력을 보여 향후 복잡한 AI 에이전트 설계 및 신경-기호 통합 시스템 연구에 중요한 기여를 할 것으로 평가된다.