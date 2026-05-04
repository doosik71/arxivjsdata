# Combining TSL and LLM to Automate REST API Testing: A Comparative Study

Thiago Barradas, Aline Paes, Vânia de Oliveira Neves (2025)

## 🧩 Problem to Solve

본 연구는 REST API의 통합 테스트(Integration Testing)를 자동화하는 과정에서 발생하는 어려움을 해결하고자 한다. 현대의 분산 시스템은 그 복잡성이 매우 높고 가능한 시나리오가 방대하여, 모든 입력 조합을 테스트하는 전수 테스트(Exhaustive Testing)가 사실상 불가능하다. 이로 인해 개발 팀은 제한된 시간 내에 수동으로 테스트를 설계해야 하며, 이는 결과적으로 테스트 커버리지의 부족, 잠재적 결함 미검출, 그리고 막대한 수동 노력이라는 문제로 이어진다.

특히 REST API 테스트에서는 비즈니스 규칙과 시스템 계약에 부합하는 유효하고 일관된 입력 데이터를 생성하는 것이 매우 까다롭다. 기존의 자동화 도구들은 단순한 스키마 기반의 퍼징(Fuzzing)에 의존하는 경우가 많아, 복잡한 비즈니스 로직이나 실제 운영 환경의 통합 경로를 충분히 검증하지 못하는 한계가 있다. 따라서 본 논문의 목표는 OpenAPI 명세서를 기반으로 LLM(Large Language Models)을 활용하여 비즈니스 로직을 반영한 통합 테스트 케이스를 자동으로 생성하는 **RestTSLLM** 접근 방식을 제안하고, 다양한 LLM의 성능을 비교 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Test Specification Language (TSL)**를 중간 단계로 도입하여 LLM의 추론 과정과 구현 과정을 분리한 **RestTSLLM** 방법론을 제안한 것이다.

핵심 아이디어는 '분해된 프롬프팅(Decomposed Prompting)'에 있다. LLM이 한 번에 OpenAPI 명세에서 실행 가능한 코드까지 생성하게 하는 대신, 다음과 같은 단계적 구조를 취한다:
1. **추론 단계**: OpenAPI 명세 $\rightarrow$ TSL (비즈니스 시나리오 및 테스트 케이스 정의)
2. **구현 단계**: TSL $\rightarrow$ 실행 가능한 테스트 코드 (예: .NET xUnit)

이러한 설계는 LLM이 코드 문법이나 구조에 신경 쓰지 않고 오직 비즈니스 규칙 이해와 시나리오 설계에만 집중하게 함으로써, 최종 생성되는 테스트 코드의 품질과 일관성을 높이는 효과를 준다. 또한, 다양한 최신 LLM들을 동일한 파이프라인에서 평가하여 어떤 모델이 REST API 테스트 생성에 가장 적합한지 정량적으로 제시하였다.

## 📎 Related Works

논문에서는 LLM을 소프트웨어 공학 및 테스트 자동화에 적용한 기존 연구들을 검토한다. 

- **기존 LLM 연구**: 많은 연구가 유닛 테스트(Unit Testing) 생성에 집중하고 있으나, 더 많은 문맥 정보가 필요한 통합 테스트에 대한 연구는 상대적으로 부족하다.
- **REST API 전용 도구**: RESTler, RESTest와 같은 전통적인 블랙박스 도구들은 커버리지는 높일 수 있으나, 자연어로 작성된 명세서의 비즈니스 로직을 깊이 있게 이해하는 데 한계가 있다.
- **최근 LLM 기반 접근**: RESTGPT는 OpenAPI 명세서를 풍부하게 만들어 다른 도구가 사용할 수 있게 돕고, LlamaRestTest는 강화 학습을 통해 동적으로 입력을 생성한다.

**RestTSLLM의 차별점**은 특정 모델의 파인튜닝(Fine-tuning) 없이 일반 목적의 LLM과 프롬프트 엔지니어링만을 사용한다는 점이다. 또한, 동적 실행보다는 재사용 가능한 테스트 아티팩트(TSL 및 코드)를 생성하는 데 집중하여 개발자의 일상적인 워크플로우에 더 쉽게 통합될 수 있도록 설계되었다.

## 🛠️ Methodology

### 전체 파이프라인 (RestTSLLM)
RestTSLLM은 **Behavior $\rightarrow$ Examples $\rightarrow$ Action**의 3단계 프롬프트 엔지니어링 구조를 가진다.

1. **Behavior (행동 정의)**: LLM에게 숙련된 개발자 및 테스터의 역할을 부여한다. OpenAPI 명세 해석, TSL 변환, AAA(Arrange, Act, Assert) 패턴 적용 등의 지침을 시스템 프롬프트로 제공하여 출력 형식을 제한한다.
2. **Examples (예시 제공)**: Few-shot 및 Decomposed Prompting을 적용한다.
   - **Prompt 1**: (OpenAPI $\rightarrow$ TSL) 변환 예시를 통해 비즈니스 규칙 추출 방법을 학습시킨다.
   - **Prompt 2**: (TSL $\rightarrow$ xUnit 코드) 변환 예시를 통해 추상적 시나리오를 실제 코드로 구현하는 방법을 학습시킨다.
3. **Action (실행)**: 학습된 로직을 실제 대상 프로젝트의 OpenAPI 명세에 적용하여 TSL을 먼저 생성하고, 이를 다시 최종 테스트 코드로 변환한다.

### 평가 지표 및 계산 방식
본 연구는 생성된 테스트의 품질을 측정하기 위해 세 가지 지표를 사용한다:
- **Success Rate ($\text{SR}$)**: 생성된 테스트가 실패 없이 성공적으로 실행되는 비율.
- **Branch Coverage ($\text{C}$)**: 테스트 실행 중 제어 흐름의 분기점이 얼마나 실행되었는지 측정하는 지표.
- **Mutation Score ($\text{MS}$)**: 코드에 인위적인 결함(Mutant)을 주입했을 때 테스트가 이를 얼마나 잘 감지하는지 측정하는 지표.

최종 성능 평가를 위해 **TOPSIS** (Technique for Order Preference by Similarity to Ideal Solution) 기법을 적용하여 가중치 합산 점수인 $\text{Calculated Score} (S)$를 산출한다. 각 지표에 동일한 가중치 $w = 33.33\%$를 부여한 공식은 다음과 같다:

$$S = w \cdot \text{SuccessRate} + w \cdot \text{Coverage} + w \cdot \text{MutationScore}$$

### 실험 설정
- **대상 LLM**: Claude 3.5 Sonnet, Deepseek R1, Qwen 2.5 32b, Sabiá 3, LLaMA 3.2 90b, GPT 4o, Gemini 1.5 Pro, Mistral Large (총 8종).
- **대상 프로젝트**: .NET 기반의 오픈소스 REST API 프로젝트 6개.
- **도구**: 테스트 실행은 Visual Studio 2022, 뮤테이션 테스트는 Stryker.NET을 사용하였다.

## 📊 Results

### 정량적 결과
실험 결과, **Claude 3.5 Sonnet**이 모든 지표에서 1위를 차지하며 가장 효과적인 모델임이 입증되었다.

| Model | Calculated Score ($S$) | Success Rate ($\text{SR}$) | Coverage ($\text{C}$) | Mutation Score ($\text{MS}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Claude 3.5 Sonnet** | **70.9%** | **100%** | **71.7%** | **40.8%** |
| Deepseek R1 | 67.1% | 97.0% | 67.5% | 36.9% |
| Qwen 2.5 32b | 65.8% | 95.5% | 68.7% | 33.3% |
| Sabiá 3 | 65.5% | 97.5% | 64.3% | 34.7% |

- **상위권 모델**: Claude 3.5 Sonnet, Deepseek R1, Qwen 2.5, Sabiá 3가 우수한 성능을 보였으며, 이들 간의 점수 차이는 최대 7.5% 내외로 비교적 근소하였다.
- **실행 성공률**: 모든 모델이 평균 95.5% 이상의 매우 높은 성공률을 기록하였다. 특히 Claude 3.5 Sonnet은 단 하나의 실패 테스트도 생성하지 않았다.
- **비용 효율성**: 모든 모델의 실행 비용이 매우 낮았으며(프로젝트당 1달러 미만), 특히 LLaMA, Sabiá, Qwen은 0.09달러 미만의 매우 낮은 비용으로 경쟁력 있는 결과를 냈다.

### 테스트 실패 분석
전체 생성된 1,635개 테스트 중 39개(2.38%)가 실패하였으며, 주요 원인은 다음과 같다:
- **Property Length (15건)**: 허용 범위를 벗어난 경계값 검증 실패.
- **Misinterpretation (7건)**: API 명세에 대한 오해 또는 잘못된 로직 적용.
- **Authentication (5건)**: 인증 정보 누락 또는 잘못된 사용.

## 🧠 Insights & Discussion

### 강점 및 유효성
본 연구는 TSL이라는 중간 언어를 도입함으로써 LLM의 인지 부하를 줄이고, 비즈니스 추론과 코드 구현을 분리하는 전략이 실제로 효과적임을 보여주었다. 특히, 정성적 분석 결과 생성된 테스트 케이스들이 OpenAPI 명세의 비즈니스 규칙(인증 기준, 상태 코드 등)을 정확히 반영하고 있었으며, 기존에 수동으로 작성된 테스트보다 더 나은 성과를 보인 경우도 있었다.

### 한계 및 비판적 해석
1. **OpenAPI 의존성**: 본 방법론은 고품질의 OpenAPI 명세서가 존재한다는 가정하에 작동한다. 명세서가 부실하거나 오래된 레거시 시스템에서는 성능이 급격히 저하될 가능성이 크다.
2. **결정론적 결과의 부족**: LLM의 온도(Temperature) 설정을 1로 유지했기에 결과의 무작위성이 존재하며, 이는 재현성 측면에서 한계가 있다.
3. **뮤테이션 스코어의 한계**: 블랙박스 테스트 생성 특성상 내부 구현 세부 사항을 알 수 없으므로, 뮤테이션 스코어가 상대적으로 낮게 나타나는 경향이 있다. 이는 명세서와 실제 구현체 사이의 간극에서 발생하는 문제로 해석된다.
4. **주관적 평가**: RQ1의 정성적 분석이 제1저자 1인에 의해 수행되어 평가의 주관성이 개입되었을 가능성이 있다.

## 📌 TL;DR

본 논문은 OpenAPI 명세서로부터 REST API 통합 테스트를 자동 생성하기 위해 **TSL(Test Specification Language)**을 중간 단계로 활용하는 **RestTSLLM** 접근 방식을 제안한다. 8종의 LLM을 비교 분석한 결과, **Claude 3.5 Sonnet**이 가장 높은 정확도와 커버리지를 보이며 최적의 모델로 선정되었다. 이 연구는 LLM을 활용한 테스트 자동화가 매우 저렴한 비용으로 가능하며, 특히 추론과 구현을 분리하는 '분해된 프롬프팅' 전략이 테스트 품질 향상에 핵심적임을 시사한다. 향후 이 방법론은 마이크로서비스 및 이벤트 기반 아키텍처와 같은 더 복잡한 시스템으로 확장될 가능성이 높다.