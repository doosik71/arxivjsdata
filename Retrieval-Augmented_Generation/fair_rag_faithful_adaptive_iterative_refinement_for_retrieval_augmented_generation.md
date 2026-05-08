# FAIR-RAG: Faithful Adaptive Iterative Refinement for Retrieval-Augmented Generation

Mohammad Aghajani Asl, Majid Asgari-Bidhendi, Behrooz Minaei-Bidgoli (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 고질적인 문제인 환각(hallucination)과 지식의 정체성(knowledge staleness)을 해결하기 위한 Retrieval-Augmented Generation (RAG) 프레임워크의 한계를 다룬다.

기존의 표준적인 "Retrieve-then-Read" 방식의 RAG는 단일 단계의 검색만으로 답을 찾을 수 없는 복잡한 Multi-hop 쿼리(여러 단계의 추론과 정보 결합이 필요한 질문)에서 성능이 급격히 저하된다. 또한, 기존의 반복적(Iterative) 또는 적응형(Adaptive) RAG 방법론들은 검색된 증거 내의 정보 공백(evidence gaps)을 체계적으로 식별하고 메우는 메커니즘이 부족하여, 노이즈를 전파하거나 불충분한 컨텍스트를 바탕으로 답변을 생성하는 경향이 있다.

따라서 본 연구의 목표는 복잡한 쿼리에 대해 증거 기반의 동적 추론 프로세스를 구축하여, 정보 공백을 명시적으로 분석하고 이를 통해 정교하게 정제된 컨텍스트를 확보함으로써 최종 답변의 신뢰성과 충실도(faithfulness)를 극대화하는 FAIR-RAG 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

FAIR-RAG의 핵심 아이디어는 단순히 답변을 생성하고 수정하는 것이 아니라, **'구조화된 증거 평가(Structured Evidence Assessment, SEA)'**라는 분석적 게이팅 메커니즘을 통해 정보의 충분성을 검증하고 부족한 부분을 정밀하게 타격하여 검색하는 반복 루프를 설계한 것이다.

주요 기여 사항은 다음과 같다.

1. **SEA 중심의 반복 정제 루프:** 쿼리를 필수 확인 항목(checklist)으로 분해하고, 수집된 증거를 이에 대조하여 '정보 공백(Remaining Gaps)'을 명시적으로 식별하는 프로세스를 도입하였다.
2. **2단계 적응형 쿼리 전략:** 초기 시맨틱 분해(semantic decomposition)와 SEA의 분석 결과에 기반한 적응형 쿼리 정제(Adaptive Query Refinement)를 통해 검색 효율을 높였다.
3. **동적 자원 할당(Dynamic Resource Allocation):** 쿼리의 복잡도에 따라 적절한 크기의 LLM(Small, Large, Reasoning model)을 동적으로 할당하여 비용, 지연 시간, 품질 간의 균형을 최적화하였다.
4. **이중 충실도 보장 체계:** 생성 전 SEA를 통한 최종 검증과 생성 시 엄격한 제약 조건(인용 강제, 외부 지식 배제)을 적용하여 환각을 최소화하였다.

## 📎 Related Works

본 논문은 RAG의 발전 과정을 세 가지 흐름으로 구분하여 설명하며, FAIR-RAG의 차별점을 제시한다.

1. **표준 RAG:** 단일 단계 검색 방식은 Single-hop 쿼리에는 효과적이나, 다수의 문서에서 정보를 합성해야 하는 Multi-hop 작업에서는 한계가 명확하다.
2. **반복적 및 다단계 RAG:**
    - **Self-Ask, SuRe:** 쿼리 분해를 사용하지만, 정적인 계획에 의존하거나 이전 답변에만 의존하여 동적인 증거 변화에 유연하게 대응하지 못한다.
    - **ReAct, IRCoT:** 추론 단계마다 국소적인 검색을 수행하지만, 전체 증거 풀에 대한 총체적 분석이 부족하여 비순차적인 정보 공백을 놓칠 수 있다.
    - **ITER-RETGEN:** 이전 생성 결과 전체를 쿼리로 사용하므로 노이즈가 포함될 가능성이 크다.
    - **FAIR-RAG의 차별점:** 답변이 아닌 '정보 공백'이라는 명시적 신호를 기반으로 쿼리를 생성함으로써 훨씬 더 집중적이고 효율적인 증거 수집이 가능하다.
3. **적응형 및 충실도 중심 RAG:**
    - **Adaptive-RAG:** 초기 단계에서 쿼리 복잡도를 분류하여 경로를 결정하지만, 루프 내부에서의 동적 적응력은 부족하다.
    - **SELF-RAG:** Reflection 토큰을 위해 모델을 미세 조정(fine-tuning)해야 하므로 범용 모델 적용이 어렵고, 평가가 전술적(step-by-step) 수준에 머문다.
    - **FAIR-RAG의 차별점:** 미세 조정 없이 모듈형 SEA를 통해 전략적(holistic) 관점에서 증거의 충분성을 평가한다.

## 🛠️ Methodology

FAIR-RAG는 크게 네 가지 단계로 구성된 파이프라인을 가진다.

### 1. 초기 쿼리 분석 및 적응형 라우팅 (Adaptive Routing)

입력 쿼리 $x$가 들어오면 $A_{router}$ 에이전트가 복잡도를 분석하여 네 가지 카테고리로 분류하고, 최종 답변 생성에 사용할 LLM $G_{selected}$를 할당한다.

- **OBVIOUS:** 파라미터 지식으로 충분 $\rightarrow$ RAG 파이프라인 우회, 대형 LLM으로 즉시 답변.
- **SMALL:** 단순 사실 확인 $\rightarrow$ 효율적인 소형 LLM 할당.
- **LARGE:** 정보 합성 필요 $\rightarrow$ 성능이 좋은 대형 LLM 할당.
- **REASONING:** 다단계 추론 필요 $\rightarrow$ 최신 추론 특화 LLM 할당.

### 2. 반복적 검색 및 정제 사이클 (Iterative Retrieval and Refinement Cycle)

최대 3회 반복되며, 다음의 세부 단계를 거친다.

**가. 적응형 쿼리 생성 (Adaptive Query Generation)**

- 1회차: $A_{decompose}$가 쿼리를 최대 4개의 독립적인 하위 쿼리로 분해한다.
- 2회차 이후: $A_{refine}$이 SEA에서 식별된 '정보 공백'과 '확인된 사실'을 바탕으로 타겟팅된 정제 쿼리를 생성한다.

**나. 하이브리드 검색 및 재순위화 (Hybrid Retrieval and Reranking)**

- Dense Vector Search(시맨틱 유사도)와 Sparse Search(키워드 일치)를 동시에 수행한다.
- 검색 결과는 Reciprocal Rank Fusion (RRF) 알고리즘을 통해 통합되어 상위 5개 문서가 후보군으로 선정된다.

**다. 증거 필터링 (Evidence Filtering)**

- $A_{filter}$ 에이전트가 각 문서의 유용성을 평가하여 불필요한 노이즈를 제거한다.

**라. 구조화된 증거 평가 (Structured Evidence Assessment, SEA)**

- 본 프레임워크의 핵심 모듈로, '전략 정보 분석가(Strategic Intelligence Analyst)' 역할을 수행한다.
- **프로세스:** $\text{쿼리 분해(Checklist)} \rightarrow \text{증거 대조(Audit)} \rightarrow \text{사실 확인(Confirmed)} \text{ 및 공백 식별(Remaining Gaps)} \rightarrow \text{충분성 판단(Sufficient: Yes/No)}$.
- 모든 체크리스트 항목이 충족되어야만 `Yes`를 반환하며, 그렇지 않으면 식별된 공백이 다음 루프의 쿼리 생성 신호로 사용된다.

### 3. 충실한 답변 생성 (Faithful Answer Generation)

SEA가 `Yes`를 반환하면, 선정된 $G_{selected}$가 답변을 생성한다. 이때 다음과 같은 엄격한 제약 조건이 부여된다.

- 제공된 증거에만 기반하여 작성할 것.
- 외부 지식이나 파라미터 지식을 절대 도입하지 말 것.
- 모든 주장에 대해 소스 문서 인덱스(예: [1], [2])를 인용할 것.
- 증거가 여전히 부족하다면 추측하지 말고 부족함을 명시할 것.

### 4. 동적 자원 할당 (Dynamic Resource Allocation)

작업의 난이도에 따라 서로 다른 모델을 배치한다.

- **Llama-3-8B-Instruct:** 쿼리 분해, SEA 등 비교적 단순한 분석 작업.
- **Llama-3.1-70B-Instruct:** 증거 필터링, 쿼리 정제, 최종 답변 생성 등 고도의 인지 능력이 필요한 작업.
- **DeepSeek-R1:** 매우 깊은 추론이 필요한 특수 케이스.

## 📊 Results

### 실험 설정

- **데이터셋:** Multi-hop QA (HotpotQA, 2WikiMultiHopQA, Musique) 및 Open-domain QA (TriviaQA). 각 1,000개 샘플 사용.
- **비교 대상:** Standard RAG, Iter-Retgen, IRCoT, ReAct, Self-RAG, SuRe, Adaptive-RAG.
- **평가 지표:** Exact Match (EM), F1-Score, script-based Accuracy (ACC), LLM-as-Judge Accuracy ($\text{ACC}_{\text{LLM}}$).

### 주요 결과

1. **정량적 성능:**
    - **HotpotQA:** F1-score 0.453을 기록하며, 가장 강력한 반복적 베이스라인인 Iter-Retgen(0.370) 대비 **8.3포인트의 절대적 향상**을 보이며 SOTA를 달성하였다.
    - **2WikiMultiHopQA / Musique:** 각각 F1 0.320, 0.264를 기록하여 Self-RAG 및 Iter-Retgen을 크게 상회하였다.
    - **TriviaQA:** F1 0.731로 단순 사실 쿼리에서도 범용적인 성능을 입증하였다.
2. **반복 횟수의 영향:**
    - 1회 $\rightarrow$ 2회 $\rightarrow$ 3회로 증가함에 따라 성능이 일관되게 향상되었다 (예: HotpotQA F1 0.398 $\rightarrow$ 0.447).
    - 그러나 4회차부터는 오히려 성능이 저하되는 경향이 나타났는데, 이는 과도한 검색이 노이즈를 유입시켜 최종 합성을 방해하기 때문인 것으로 분석된다.
3. **구성 요소 분석 (Ablation):**
    - 쿼리 분해 및 정제 모듈이 매우 높은 품질 점수(평균 4.1~4.5/5.0)를 기록하여 효과적임을 확인하였다.
    - SEA 모듈은 특히 복잡한 Multi-hop 데이터셋에서 80% 이상의 높은 정확도로 충분성을 판단하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

FAIR-RAG의 성공 요인은 **'답변 중심'이 아닌 '증거 중심'의 루프**에 있다. 기존 방식들이 "지금까지 생성한 답변을 보고 다음 검색어를 정하는" 방식이었다면, FAIR-RAG는 "질문에 답하기 위해 필요한 필수 정보 목록(checklist) 중 무엇이 빠졌는가"를 먼저 분석한다. 이러한 전략적 접근은 검색의 목적성을 명확히 하여 Multi-hop 추론에서 발생하는 정보 누락 문제를 효과적으로 해결한다.

### 한계 및 비판적 논의

1. **기초 모델 의존성:** 본 시스템은 LLM의 추론 능력과 프롬프트 엔지니어링에 크게 의존한다. SEA 모듈이 잘못된 충분성 판단(False Positive)을 내릴 경우 루프가 조기에 종료되어 오답을 낼 위험이 있다.
2. **비용 및 지연 시간:** 반복적인 루프와 다수의 에이전트 호출로 인해 Standard RAG 대비 계산 비용과 추론 시간이 크게 증가한다. 이는 실시간 서비스 적용 시 병목이 될 수 있다.
3. **에러 전파:** 파이프라인의 단계가 많아 초기 단계(쿼리 분해 등)에서의 오류가 최종 답변까지 전파될 가능성이 존재한다.
4. **실패 원인 분석:** 분석 결과 전체 에러의 63.5%가 기초 모델의 한계(검색 실패, 생성 실패)에서 기인하였다. 이는 프레임워크의 논리적 구조를 개선하는 것만큼이나, 기반이 되는 Retriever와 Generator의 성능 향상이 필수적임을 시사한다.

## 📌 TL;DR

FAIR-RAG는 복잡한 Multi-hop 쿼리를 해결하기 위해 **구조화된 증거 평가(SEA)**를 도입한 에이전트 기반 RAG 프레임워크이다. 쿼리를 체크리스트로 분해하고 정보 공백을 명시적으로 찾아내어 반복적으로 메우는 과정을 통해, HotpotQA 등 주요 벤치마크에서 SOTA 성능을 달성하였다. 이 연구는 단순한 반복 검색보다 **체계적인 정보 공백 분석**이 복잡한 지식 집약적 작업의 정확도와 충실도를 높이는 데 결정적임을 입증하였다. 향후 검색 모델의 성능 개선 및 RL을 통한 동적 제어 정책 학습이 결합된다면 더 효율적인 시스템이 될 가능성이 높다.
