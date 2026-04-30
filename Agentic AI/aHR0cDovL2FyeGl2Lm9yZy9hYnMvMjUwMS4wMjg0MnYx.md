# Foundations of GenIR

Qingyao Ai, Jingtao Zhan, Yiqun Liu (2025)

## 🧩 Problem to Solve

본 논문(챕터)은 현대의 생성형 AI(Generative AI) 모델이 정보 접근(Information Access, IA) 시스템에 미치는 기초적인 영향과 그로 인해 발생하는 새로운 패러다임을 분석한다. 

기존의 정보 검색(Information Retrieval, IR) 및 추천 시스템은 이미 존재하는 정보를 찾아 제시하는 데 특화되어 있어, 매우 희소한 요구사항(Long-tail information needs)이나 창의적인 영감이 필요한 작업에서는 한계를 보였다. 또한, 대규모 언어 모델(LLM)은 뛰어난 생성 능력을 갖추었으나, 사실에 근거하지 않은 내용을 생성하는 환각(Hallucination) 문제와 최신 또는 외부의 비공개 데이터를 즉각적으로 반영하지 못하는 지식의 제약이라는 문제를 안고 있다.

따라서 본 연구의 목표는 생성형 AI를 활용하여 정보를 직접 생성하는 '정보 생성(Information Generation)'과 기존 정보를 통합하여 신뢰성 있는 답변을 만드는 '정보 합성(Information Synthesis)'이라는 두 가지 관점에서 정보 접근 시스템의 새로운 발전 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 생성형 AI 기반의 정보 접근(GenIR)을 **정보 생성(Information Generation)**과 **정보 합성(Information Synthesis)**이라는 두 가지 핵심 축으로 정의하고, 각각의 이론적 기초와 방법론을 체계적으로 정리한 것에 있다.

1.  **정보 생성(Information Generation):** 사용자의 요구에 맞춘 맞춤형 콘텐츠를 직접 생성하여 사용자 경험을 극대화하는 방향을 제시하며, 이를 위한 Transformer 아키텍처의 발전, Scaling Law, 학습 단계 및 멀티모달 확장성을 분석한다.
2.  **정보 합성(Information Synthesis):** 외부 지식을 통합하여 모델의 환각 문제를 완화하고 정밀도를 높이는 방향을 제시하며, 특히 검색 증강 생성(Retrieval-Augmented Generation, RAG)의 고도화와 생성적 검색(Generative Retrieval)의 가능성을 탐구한다.

## 📎 Related Works

논문은 전통적인 AI 기술과 현대 생성형 AI의 차이점을 명확히 하며, 다음과 같은 관련 연구 흐름을 언급한다.

-   **전통적 IA 시스템:** 검색 엔진과 추천 플랫폼은 기존 데이터의 인덱싱과 랭킹에 의존하며, 이는 정해진 데이터셋 내에서 최적의 결과를 찾는 데 집중한다.
-   **Transformer 및 LLM:** Vaswani et al. [4]의 Transformer 이후, 모델의 크기와 데이터량을 늘리는 Scaling Law [24]에 의해 성능이 비약적으로 향상되었으며, 이는 단순 검색을 넘어 생성의 영역으로 확장되었다.
-   **RAG (Retrieval-Augmented Generation):** LLM의 내부 지식에만 의존하지 않고 외부 문서를 참조하는 방식이며, 이는 과거의 추출적/추상적 요약(Extractive/Abstractive Summarization) 연구의 연장선에 있다.
-   **Generative Retrieval (GR):** 기존의 역색인(Inverted Index)이나 벡터 기반 인덱스를 신경망의 파라미터 공간으로 대체하려는 시도(Differentiable Index)를 다룬다.

## 🛠️ Methodology

본 논문은 특정 알고리즘 하나를 제안하는 것이 아니라, GenIR을 구현하기 위한 전반적인 기술적 토대를 설명한다.

### 1. 정보 생성 (Information Generation) 기반

#### 모델 아키텍처 (Model Architecture)
Transformer 구조를 중심으로 하며, 특히 효율성과 안정성을 높이기 위한 다음의 구성 요소들을 분석한다.
-   **Word Embedding:** 학습 초기 단계의 불안정성을 해결하기 위해 Layer Normalization을 추가하거나 그라디언트 크기를 조정하는 기법이 사용된다.
-   **Position Embedding:** 단순 Sinusoidal 방식에서 학습 가능한(Trainable) 방식, 그리고 최근의 RoPE(Rotary Position Embedding)로 발전하여 긴 시퀀스 모델링 능력을 높였다.
-   **Attention:** $O(n^2)$의 복잡도를 해결하기 위한 Sparse Attention, Reformer(LSH 기반) 등이 있으며, 추론 속도 향상을 위해 KV Cache를 최적화하는 MQA(Multi-Query Attention), GQA(Grouped Query Attention), MLA(Multi-head Latent Attention) 등이 도입되었다.
-   **Layer Normalization:** Post-LN의 불안정성을 해결하기 위해 Pre-LN, Sandwich-LN, DeepNorm 등이 제안되어 모델의 층을 1,000층까지 쌓을 수 있게 하였다.

#### Scaling 및 학습 절차
모델의 손실 함수 $L$은 모델 크기나 데이터 크기 $x$에 대해 다음과 같은 로그-선형 관계로 감소한다.
$$L(x) = L_{\infty} + k \cdot x^{-\alpha}$$
학습은 일반적으로 **Pre-training $\rightarrow$ Supervised Fine-Tuning (SFT) $\rightarrow$ Reinforcement Learning from Human Feedback (RLHF)**의 3단계로 진행된다. 학습 목표는 기본적으로 Next Token Prediction이며, 수식으로는 다음과 같이 표현된다.
$$P(x_{t+1} | x_1, \dots, x_t)$$

### 2. 정보 합성 (Information Synthesis) 기반

#### 검색 증강 생성 (RAG)
RAG는 외부 지식을 LLM의 입력(Prompt)에 포함시켜 답변을 생성하는 구조이다.
-   **Naive RAG:** 'Retrieve-then-Read' 구조로, 단순히 검색 결과와 쿼리를 LLM에 입력한다.
-   **Modular RAG:** 다음과 같은 세 가지 핵심 질문에 집중한다.
    -   *When to retrieve:* LLM이 환각을 일으키거나 불확실성이 높을 때 호출하는 타이밍 결정.
    -   *What to retrieve:* LLM의 내부 어텐션 분포 등을 분석하여 실제 필요한 정보 요구사항을 쿼리로 변환.
    -   *Where to retrieve:* 다양한 데이터 소스(법률, 금융, 뉴스 등) 중 최적의 소스를 선택하는 내비게이션.

#### 생성적 검색 (Generative Retrieval, GR)
전통적인 인덱싱을 대체하여 모델 파라미터 자체를 인덱스로 사용하는 방식이다.
-   **Differentiable Index:** 문서의 내용을 파라미터 공간에 암시적으로 저장한다.
-   **Doc ID Generation:** 쿼리가 입력되면 모델이 해당 문서의 고유 ID(Explicit 또는 Implicit Token)를 직접 생성하여 문서를 식별한다.

## 📊 Results

본 논문은 특정 실험 데이터셋에 대한 결과 보고서라기보다 종합적인 분석 보고서(Survey/Foundation)의 성격을 띤다. 다만, 논문 내에서 인용된 주요 정량적/정성적 경향은 다음과 같다.

-   **RAG의 효과:** Su et al. [111]의 연구에 따르면, LLM의 내부 어텐션 분포를 기반으로 쿼리를 생성했을 때 RAG의 성능이 여러 벤치마크에서 약 20% 향상되었다.
-   **Scaling Law의 논쟁:** 손실(Loss)의 감소가 실제 성능 지표(Metric)의 초선형적(Super-linear) 향상(Emergent abilities)으로 이어지는지에 대해 학계의 의견이 갈리고 있으며, 일부 연구는 연속적 지표에서는 이러한 현상이 나타나지 않는다고 주장한다.
-   **추론 비용:** 모델 크기가 커질수록 성능은 향상되지만 추론 비용이 급증하므로, 최근에는 Llama나 MiniCPM과 같이 작은 모델을 매우 많은 데이터로 학습시키는 방향이 주목받고 있다.

## 🧠 Insights & Discussion

### 강점 및 통찰
본 논문은 생성형 AI를 단순히 기존 시스템의 보조 도구로 보는 것이 아니라, '생성'과 '합성'이라는 두 가지 패러다임으로 나누어 IR의 영역을 확장했다는 점에서 학술적 가치가 크다. 특히 RAG의 최적화 방향을 'When, What, Where'라는 구체적인 질문으로 세분화하여 향후 연구 방향을 명확히 제시하였다.

### 한계 및 미해결 과제
-   **GR의 통제 가능성:** 생성적 검색(GR)의 경우, 파라미터 공간에 저장된 특정 문서를 삭제하거나 업데이트하는 것이 이론적으로 매우 어려워 시스템의 제어 가능성(Controllability) 문제가 남아 있다.
-   **결합 최적화의 부재:** 현재 RAG 시스템은 검색기(Retriever)와 생성기(Generator)가 프롬프트를 통해 느슨하게 연결되어 있다. 이 두 구성 요소를 단일 손실 함수로 직접 연결하여 공동 최적화(Joint Optimization)하는 방법론은 여전히 미개척 분야로 남아 있다.

### 비판적 해석
논문은 IR 커뮤니티가 LLM의 등장으로 위기감을 느꼈으나, 오히려 복합적인 정보 요구사항(Composite information needs)을 해결하는 '정보 에이전트'로 진화할 기회를 맞이했다고 분석한다. 이는 단순한 결과 리스트 제공(SERPs)에서 복합적 과제 해결(Planning & Execution)로 IR의 정의가 확장되어야 함을 시사한다.

## 📌 TL;DR

본 논문은 생성형 AI가 정보 접근(IA) 시스템을 어떻게 혁신하는지 분석하며, 이를 **정보 생성(맞춤형 콘텐츠 직접 제작)**과 **정보 합성(외부 지식 통합 및 환각 제거)**의 두 가지 패러다임으로 체계화하였다. 특히 Transformer의 구조적 발전, RAG의 모듈화, 그리고 생성적 검색(GR)의 가능성을 다루었으며, 향후 IR 시스템이 단순 검색을 넘어 복합적 계획 수립과 실행이 가능한 **지능형 정보 에이전트**로 진화해야 한다고 강조한다.