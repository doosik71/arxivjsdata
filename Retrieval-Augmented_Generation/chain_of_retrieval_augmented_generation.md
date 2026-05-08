# Chain-of-Retrieval Augmented Generation

Liang Wang et al. (2025)

## 🧩 Problem to Solve

본 논문은 기존의 Retrieval-Augmented Generation (RAG) 시스템이 가진 구조적 한계, 즉 **단일 단계 검색(single-step retrieval)의 취약성**을 해결하고자 한다. 일반적인 RAG 방식은 최종 답변을 생성하기 전 단 한 번의 검색 과정을 거치는데, 이는 다음과 같은 문제를 야기한다.

첫째, 복잡한 쿼리(complex queries)의 경우 단 한 번의 검색만으로는 필요한 모든 정보를 수집하기 어려우며, 검색 결과가 불완전할 경우 모델이 잘못된 정보를 생성하거나 환각(hallucination)을 일으킬 가능성이 높다. 둘째, 특히 Multi-hop reasoning이 필요한 작업에서는 초기 검색 단계에서 무엇을 찾아야 할지 불분명하며, 검색된 정보에 따라 다음 검색 방향을 동적으로 결정해야 하는 특성이 있다.

따라서 본 연구의 목표는 모델이 최종 답변을 내놓기 전, 관련 정보를 단계적으로 검색하고 추론할 수 있도록 하여, 복잡한 질의에 대해 보다 정확하고 근거 있는(grounded) 답변을 생성하는 **CoRAG (Chain-of-Retrieval Augmented Generation)** 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 OpenAI의 o1 모델과 유사하게, **추론 단계에서 검색 과정을 체인(chain) 형태로 확장하여 test-time compute를 스케일링**하는 것이다. 주요 기여 사항은 다음과 같다.

1. **동적 쿼리 재구성(Dynamic Query Reformulation):** 고정된 검색 방식에서 벗어나, 모델이 현재 상태(evolving state)에 따라 쿼리를 동적으로 수정하고 다음 검색 단계를 계획하는 메커니즘을 도입하였다.
2. **Rejection Sampling을 통한 데이터 증강:** 중간 검색 체인(intermediate retrieval chains)이 없는 기존 RAG 데이터셋의 한계를 극복하기 위해, Rejection Sampling 기법을 사용하여 정답을 도출하는 최적의 검색 경로를 자동으로 생성하고 이를 학습에 활용하였다.
3. **Test-time Compute Scaling 전략:** Greedy decoding, Best-of-N sampling, Tree Search 등 다양한 디코딩 전략을 제안하여, 추론 시 투입되는 연산량(토큰 소비량)에 따라 성능을 조절할 수 있는 가능성을 입증하였다.

## 📎 Related Works

본 논문은 기존의 RAG 연구와 LLM의 추론 스케일링 연구를 계승하고 발전시킨다.

- **Iterative RAG:** FLARE, IRCoT, Self-RAG 등 기존 연구들이 반복적 검색이나 자기 성찰(self-reflection)을 통해 RAG의 성능을 높이려 했다. 그러나 이러한 방식들은 주로 Few-shot prompting이나 proprietary 모델로부터의 distillation에 의존하는 경향이 있었다. CoRAG는 모델이 단계적으로 검색하도록 **명시적으로 파인튜닝(explicit training)**한다는 점에서 차별화된다.
- **Test-time Compute Scaling:** Chain-of-Thought (CoT)나 Tree-of-Thought (ToT), 그리고 최근의 OpenAI o1 모델은 추론 단계에서 더 많은 연산을 수행함으로써 복잡한 문제 해결 능력을 높였다. CoRAG는 이러한 개념을 RAG 영역으로 가져와, 검색 단계의 수와 샘플링 횟수를 조절함으로써 성능을 높이는 Scaling Law를 탐색하였다.

## 🛠️ Methodology

### 1. Retrieval Chain Generation (데이터 생성)

대부분의 RAG 데이터셋은 $\{Q, A\}$ 쌍으로만 구성되어 있어 중간 단계의 검색 경로가 없다. 이를 해결하기 위해 Rejection Sampling을 통해 검색 체인을 생성한다.

- **체인 구성:** 각 체인은 서브 쿼리 $\{Q_1, \dots, Q_L\}$와 그에 대응하는 서브 답변 $\{A_1, \dots, A_L\}$의 시퀀스로 구성된다.
- **생성 절차:**
    1. LLM이 현재까지의 상태를 바탕으로 서브 쿼리 $Q_i$를 생성한다.
    2. 검색기(Retriever)가 $Q_i$에 대해 상위 $k$개의 문서 $D^{(i)}_{1:k}$를 검색한다.
    3. LLM이 $Q_i$와 $D^{(i)}_{1:k}$를 이용해 서브 답변 $A_i$를 생성한다.
    4. 정답 $A$에 도달하거나 최대 길이 $L$에 도달할 때까지 반복한다.
- **최적 체인 선택:** 생성된 여러 체인 중, 정답 $A$에 대한 조건부 로그 가능도(log-likelihood) $\log P(A | Q, Q_{1:L}, A_{1:L})$가 가장 높은 체인을 최종 선택하여 학습 데이터로 사용한다.

### 2. Model Training (학습)

증강된 데이터셋 $(Q, A, Q_{1:L}, A_{1:L})$을 사용하여 표준 Next-token prediction 목표로 LLM을 파인튜닝한다. 모델은 다음 세 가지 태스크를 동시에 학습하는 multi-task learning 프레임워크를 따른다.

- **서브 쿼리 예측:** $\mathcal{L}_{\text{sub\_query}} = -\log P(Q_i | Q, Q_{<i}, A_{<i})$
- **서브 답변 예측:** $\mathcal{L}_{\text{sub\_answer}} = -\log P(A_i | Q_i, D^{(i)}_{1:k})$
- **최종 답변 예측:** $\mathcal{L}_{\text{final\_answer}} = -\log P(A | Q, Q_{1:L}, A_{1:L}, D_{1:k})$

### 3. Test-time Scaling (추론 전략)

추론 시 연산량과 성능 사이의 트레이드-오프를 조절하기 위해 세 가지 디코딩 전략을 사용한다.

- **Greedy Decoding:** 가장 확률이 높은 서브 쿼리와 답변을 순차적으로 생성한다.
- **Best-of-N Sampling:** 온도(temperature) 0.7로 $N$개의 체인을 샘플링한다. 정답을 알 수 없는 테스트 상황에서는 "No relevant information found"라는 문구가 생성될 확률(penalty score)이 가장 낮은 체인을 선택한다.
- **Tree Search:** BFS(너비 우선 탐색) 변형 방식을 사용하여, 각 단계에서 여러 서브 쿼리를 샘플링하고 롤아웃(rollout)을 통해 평균 페널티 점수가 가장 낮은 경로를 확장한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Multi-hop QA (2WikiMultihopQA, HotpotQA, Bamboogle, MuSiQue) 및 KILT 벤치마크.
- **모델 및 검색기:** Llama-3.1-8B-Instruct 기반 파인튜닝, E5-large 검색기 사용.
- **평가 지표:** Exact Match (EM) 및 F1 Score.

### 2. 주요 결과

- **Multi-hop QA 성능:** CoRAG-8B는 강력한 베이스라인들을 압도하였다. 특히 Multi-hop 추론이 필수적인 작업에서 EM 점수가 10점 이상 향상되는 결과를 보였다.
- **KILT 벤치마크:** 대부분의 지식 집약적 태스크에서 새로운 State-of-the-Art (SOTA) 성능을 달성하였다.
- **Scaling Behavior:** 토큰 소비량과 EM 점수 사이의 관계가 약 $\log$-linear 관계를 따름을 확인하였다. 즉, 추론 시 더 많은 토큰을 사용할수록(체인 길이를 늘리거나 샘플링 횟수를 높일수록) 성능이 지속적으로 향상되는 경향을 보였다.

### 3. 분석 결과

- **검색 효율성:** CoRAG는 단순 RAG 대비 Retrieval Recall을 대폭 향상시켰다. 특히 MuSiQue와 같은 고난도 데이터셋에서 단일 단계 검색의 한계를 극복하고 정답 관련 문서를 훨씬 더 잘 찾아냈다.
- **태스크별 효용성:** Multi-hop QA에서는 성능 향상이 뚜렷했으나, 단일 단계 검색으로 충분한 NQ나 TriviaQA 같은 태스크에서는 성능 이득이 미미했다. 이는 쿼리의 복잡도에 따라 연산 자원을 동적으로 할당해야 함을 시사한다.

## 🧠 Insights & Discussion

본 논문은 RAG 시스템에서 '생각의 사슬(Chain-of-Thought)'과 유사한 '검색의 사슬(Chain-of-Retrieval)'을 도입함으로써 얻을 수 있는 이점을 명확히 제시하였다.

**강점:**

- **자기 수정 능력:** 사례 분석을 통해 모델이 초기 검색에서 잘못된 정보를 얻더라도, 후속 검색 단계를 통해 이를 스스로 수정하고 정답에 도달하는 과정을 보여주었다.
- **모델 일반화:** Llama뿐만 아니라 Qwen3 모델 가족에서도 유사한 성능 향상이 나타나, 제안된 프레임워크가 모델 아키텍처에 구애받지 않고 일반화될 수 있음을 입증하였다.
- **강건성:** E5-base나 BM25와 같은 상대적으로 약한 검색기를 사용하더라도, test-time compute를 늘림으로써 성능 저하를 상당 부분 보완할 수 있었다.

**한계 및 논의:**

- **연산 비용:** 성능 향상을 위해 더 많은 토큰과 검색 호출이 필요하며, 이는 응답 지연 시간(latency)의 증가로 이어진다.
- **정지 시점 결정:** 모델이 언제 검색을 멈추고 최종 답변을 생성해야 할지 결정하는 'Learning to Stop' 메커니즘을 실험했으나, 너무 일찍 멈출 경우 성능이 저하되는 트레이드-오프가 존재한다.
- **범위의 제한:** 본 연구는 주로 단답형 혹은 검증이 쉬운 QA 태스크에 집중되어 있으며, 긴 형태의 답변을 생성해야 하는 Long-form generation 태스크에서의 효용성은 아직 검증되지 않았다.

## 📌 TL;DR

CoRAG는 복잡한 질의 해결을 위해 **단계적 검색 및 추론(Iterative Retrieval & Reasoning)**을 수행하는 RAG 프레임워크이다. Rejection Sampling으로 생성한 최적의 검색 경로를 학습시켜 모델이 스스로 쿼리를 재구성하고 보완하도록 만들었으며, 추론 시 연산량을 늘림으로써 성능을 높이는 Scaling Law를 확인하였다. 이 연구는 특히 Multi-hop QA에서 탁월한 성능을 보이며, 향후 더 신뢰할 수 있고 사실에 기반한(grounded) 파운데이션 모델을 개발하는 데 중요한 기여를 할 것으로 보인다.
