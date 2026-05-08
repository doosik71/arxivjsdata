# Active Retrieval Augmented Generation

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig (2023)

## 🧩 Problem to Solve

대규모 언어 모델(Large Language Models, LMs)은 뛰어난 언어 이해 및 생성 능력을 갖추고 있으나, 사실과 다른 내용을 생성하는 Hallucination(환각) 현상이 빈번하게 발생한다. 이를 해결하기 위해 외부 지식 리소스에서 정보를 검색하여 모델의 입력을 보강하는 Retrieval Augmented Generation (RAG) 방식이 제안되었다.

그러나 기존의 대부분의 RAG 시스템은 사용자의 입력 단계에서 단 한 번만 검색을 수행하는 'retrieve-and-generate' 구조를 취한다. 이러한 방식은 단답형 질문 응답(Short-form QA)에는 효과적이지만, 생성 과정에서 지속적으로 새로운 정보가 필요한 긴 문장 생성(Long-form generation) 작업에서는 한계가 있다. 사용자가 입력한 초기 쿼리만으로는 생성될 전체 텍스트에 필요한 모든 세부 정보를 미리 파악하여 검색하기 어렵기 때문이다.

본 논문의 목표는 생성 과정 전반에 걸쳐 **언제(When)** 그리고 **무엇을(What)** 검색할지를 능동적으로 결정하는 Active Retrieval Augmented Generation 프레임워크를 제안하고, 이를 통해 긴 문장 생성 시의 사실성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델이 생성할 미래의 내용을 예측하여 검색 쿼리로 활용하는 **Forward-Looking** 전략과, 모델의 확신도(Confidence)에 따라 검색 여부를 결정하는 **Active** 전략을 결합한 **FLARE (Forward-Looking Active REtrieval augmented generation)** 방법론이다.

중심적인 직관은 다음과 같다.

1. **언제 검색하는가**: 모델이 생성하는 토큰의 확률값이 낮을 때, 즉 모델이 해당 지식에 대해 확신이 없을 때만 검색을 수행하여 불필요한 노이즈 유입을 방지한다.
2. **무엇을 검색하는가**: 과거의 문맥이 아닌, 모델이 생성하고자 하는 '다음 문장'의 예측값을 쿼리로 사용하여 미래의 생성 내용에 직접적으로 필요한 정보를 가져온다.

## 📎 Related Works

기존의 다중 검색(Multi-time retrieval) 접근 방식은 다음과 같은 한계가 있다.

- **Passive Retrieval**: 고정된 간격(예: 매 $l$ 토큰마다)으로 검색을 수행하는 방식(RETRO, IC-RALM 등)은 모델이 실제로 정보를 필요로 하는 시점과 일치하지 않을 수 있으며, 부적절한 시점의 검색으로 인해 효율성이 떨어진다.
- **Question Decomposition**: 복잡한 질문을 여러 개의 하위 질문으로 분해하여 검색하는 방식(Self-ask 등)은 효과적이지만, 각 작업마다 사람이 직접 하위 질문 생성 예시(Exemplars)를 작성해야 하는 Task-specific한 프롬프트 엔지니어링 비용이 크다.

FLARE는 추가적인 학습 없이 추론 시점에 적용 가능하며, 범용적인 능동 검색 메커니즘을 통해 위와 같은 한계점을 극복하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

FLARE는 다음과 같은 반복적인 루프로 작동한다.

1. 현재까지 생성된 텍스트를 바탕으로 다음 문장의 임시 예측값 $\hat{s}_t$를 생성한다.
2. $\hat{s}_t$ 내의 토큰 중 확률값이 임계치 $\theta$보다 낮은 토큰이 있는지 확인한다.
3. 저신뢰도 토큰이 있다면, $\hat{s}_t$를 기반으로 쿼리 $q_t$를 생성하여 관련 문서를 검색한다.
4. 검색된 문서 $D_{q_t}$를 컨텍스트에 추가하여 문장 $s_t$를 다시 생성한다.
5. 종료 조건에 도달할 때까지 이 과정을 반복한다.

### 상세 방법론

#### 1. Confidence-based Active Retrieval (언제 검색할 것인가)

모델이 생성한 임시 문장 $\hat{s}_t$의 각 토큰 확률을 확인하여, 하나라도 임계값 $\theta \in [0, 1]$보다 낮으면 검색을 트리거한다.

$$
y_t =
\begin{cases}
\hat{s}_t & \text{if all tokens of } \hat{s}_t \text{ have probs } \ge \theta \\
s_t = \text{LM}([D_{q_t}, x, y_{<t}]) & \text{otherwise}
\end{cases}
$$

#### 2. Confidence-based Query Formulation (무엇을 검색할 것인가)

단순히 예측 문장 $\hat{s}_t$를 그대로 쿼리로 사용할 경우, 잘못 생성된 정보가 검색 결과까지 오염시킬 위험이 있다. 이를 방지하기 위해 두 가지 쿼리 생성 방식을 제안한다.

- **Implicit Query (Masking)**: $\hat{s}_t$에서 확률값이 임계치 $\beta$보다 낮은 토큰들을 마스킹 처리하여 검색기에 전달함으로써, 잘못된 정보로 인한 간섭을 줄인다.
- **Explicit Query (Question Generation)**: 저신뢰도 구간(Span) $z$를 추출하고, 이를 답변으로 하는 질문 $q_{t,z}$를 별도의 LM(GPT-3.5-turbo)을 통해 생성하여 검색한다. (예: "조 바이든은 펜실베이니아 대학을 다녔다" $\rightarrow$ "조 바이든은 어느 대학을 다녔는가?")

#### 3. $\text{FLARE}_{\text{instruct}}$

별도의 신뢰도 계산 없이, 모델이 스스로 `[Search(query)]`라는 특수 토큰을 생성하도록 few-shot prompting을 통해 학습시키는 방식이다. 모델이 이 토큰을 생성하면 생성을 멈추고 검색을 수행한 뒤 다시 생성을 이어간다.

## 📊 Results

### 실험 설정

- **모델**: `text-davinci-003` (GPT-3.5)
- **데이터셋 및 작업**:
  - **2WikiMultihopQA**: 다단계 추론이 필요한 복합 QA.
  - **StrategyQA**: 상식 추론 기반의 Yes/No QA.
  - **ASQA / ASQA-hint**: 모호한 질문에 대한 상세한 답변 생성 (Long-form QA).
  - **WikiAsp**: 특정 속성 기반의 개방형 도메인 요약.
- **리트리버**: BM25 (Wikipedia), Bing Search Engine (Open Web).
- **지표**: Exact Match (EM), $F_1$, ROUGE, UniEval (사실 일관성 측정).

### 주요 결과

- **전반적 성능**: FLARE는 모든 테스트 데이터셋에서 단일 검색(Single-time) 및 기존 다중 검색(Multi-time) 베이스라인보다 우수하거나 경쟁력 있는 성능을 보였다.
- **Multihop QA**: 가장 뚜렷한 성능 향상을 보였으며, 이는 미래의 생성 내용을 예측하여 검색하는 전략이 복잡한 추론 단계의 정보 요구사항을 정확히 반영했기 때문이다.
- **정량적 지표**: 2WikiMultihopQA에서 $\text{FLARE}_{\text{direct}}$는 EM 기준 51.0를 기록하여, 기존의 Question Decomposition 방식(47.8)과 단일 검색 방식(39.4)을 크게 상회하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **Forward-Looking의 유효성**: 실험을 통해 이전 문맥(Previous context)을 쿼리로 사용하는 것보다 다음 문장을 예측하여 사용하는 것이 훨씬 효과적임을 입증하였다. 이는 생성하려는 의도(Intent)가 검색 쿼리에 직접적으로 반영되기 때문이다.
- **능동적 검색의 필요성**: 모든 문장에서 검색을 수행하는 것보다, 모델의 신뢰도가 낮을 때만 수행하는 것이 성능이 더 좋았다. 특히 StrategyQA에서는 과도한 검색이 오히려 노이즈를 유발하여 성능을 떨어뜨리는 현상이 관찰되었다.
- **범용성**: 특정 작업에 맞춘 수동 어노테이션 없이도, 신뢰도 기반의 능동 검색만으로 다양한 long-form 생성 작업에 적용 가능하다는 점이 큰 강점이다.

### 한계 및 비판적 해석

- **효율성 문제**: 추론 시점에 모델을 여러 번 호출하고 검색을 반복해야 하므로, 단일 생성 방식에 비해 시간적/비용적 오버헤드가 크다. 저자들은 이를 해결하기 위해 독립적인 인코딩 구조의 아키텍처 설계가 필요함을 언급한다.
- **데이터셋 제약**: Wizard of Wikipedia나 ELI5와 같은 데이터셋에서는 유의미한 이득이 없었다. 이는 답변의 길이가 너무 짧거나, 검색된 정보를 생성물에 결합(Grounding)하는 것 자체가 매우 어려운 작업일 경우 RAG의 효과가 제한적임을 시사한다.

## 📌 TL;DR

본 논문은 긴 문장 생성 시 발생하는 환각 현상을 줄이기 위해, 모델의 토큰 생성 확률(Confidence)을 기반으로 검색 시점을 결정하고 생성될 미래 문장을 쿼리로 활용하는 **FLARE** 프레임워크를 제안한다. 제안 방법론은 추가 학습 없이 추론 단계에서 적용 가능하며, 다단계 QA 및 요약 등 다양한 지식 집약적 작업에서 기존 RAG 방식보다 뛰어난 사실성과 성능을 보였다. 이 연구는 향후 LLM이 외부 지식을 보다 효율적이고 능동적으로 통합하는 방향으로 발전하는 데 중요한 기여를 할 것으로 보인다.
