# $R^2AG$: Incorporating Retrieval Information into Retrieval Augmented Generation

Fuda Ye, Shuangyin Li, Yongqi Zhang, Lei Chen (2024)

## 🧩 Problem to Solve

본 논문은 Retrieval Augmented Generation (RAG) 시스템에서 발생하는 Retriever(리트리버)와 Large Language Model (LLM, 생성기) 사이의 **Semantic Gap (의미적 격차)** 문제를 해결하고자 한다.

기존의 RAG 프레임워크는 리트리버가 검색한 문서들을 단순히 텍스트 형태로 결합하여 LLM의 입력으로 제공한다. 그러나 리트리버는 주로 Encoder 구조를 가지며 '가장 관련성 높은 문서를 찾는 것'을 목표로 학습되는 반면, LLM은 주로 Decoder 구조를 가지며 '주어진 맥락을 바탕으로 정답을 생성하는 것'을 목표로 한다. 이러한 학습 목적과 아키텍처의 차이로 인해 LLM은 리트리버가 제공한 문서들을 수동적으로 수용할 수밖에 없으며, 검색 결과에 포함된 노이즈(관련 없는 문서)를 구분하기 위해 자체적인 지식에 의존해야 하는 부담을 안게 된다. 특히 문서의 길이가 길어질 경우 'Lost-in-the-middle' 현상이 발생하여 생성 성능이 저하되는 문제가 나타난다.

따라서 본 연구의 목표는 리트리버가 가진 세밀한 시맨틱 표현(Semantic Representation)을 LLM이 이해할 수 있는 형태로 변환하여 전달함으로써, LLM이 검색된 문서들 사이의 관계를 더 잘 파악하고 정확한 답변을 생성하도록 돕는 $R^2AG$ 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 리트리버의 내부 정보를 단순히 텍스트로 변환하는 것이 아니라, **리트리버의 시맨틱 특징(Retrieval Information)을 추출하여 LLM의 입력 임베딩 공간에 직접 주입**하는 것이다. 이를 통해 LLM이 어떤 문서가 더 중요한지, 문서 간의 관계가 어떠한지를 안내하는 '앵커(Anchor)' 역할을 제공한다.

주요 기여 사항은 다음과 같다.

1. **$R^2AG$ 프레임워크 제안**: 리트리버와 LLM 사이의 Semantic Gap을 메우기 위해 리트리버 정보를 통합하는 강화된 RAG 구조를 제안하였다. 특히 리트리버와 LLM을 모두 동결(Frozen)한 상태에서도 작동 가능하여 자원 효율적이다.
2. **$R^2\text{-Former}$ 설계**: 리트리버에서 추출된 리스트 형태의 특징을 입력받아 핵심 정보를 캡처하는 가볍고 플러그인 가능한 Transformer 기반 모델을 설계하였다.
3. **Retrieval-aware Prompting 전략**: 추출된 리트리버 정보를 LLM의 토큰 임베딩 공간에 삽입하여, 문서의 내용이나 순서를 변경하지 않고도 LLM이 문서 간의 관계를 이해할 수 있도록 하는 프롬프팅 기법을 도입하였다.

## 📎 Related Works

논문에서는 기존의 RAG 강화 방법들을 크게 두 가지 방향으로 설명한다.

1. **전처리 및 압축 기반 접근 방식**: RECOMP, CRAG, LongLLMLingua 등은 검색된 문서에서 불필요한 부분을 제거하거나 요약하여 LLM에 전달한다. 하지만 이러한 방식은 리트리버와 생성기를 여전히 분리된 프로세스로 취급하며, 압축 과정에서 필수 정보가 손실될 위험이 있고 추론 시 추가 비용이 발생한다.
2. **Joint Modeling 및 Latent Representation 방식**: 일부 연구는 문서를 잠재 표현(Latent Representation)으로 통합하거나 리트리버와 LLM을 공동 최적화한다. 그러나 이는 주로 Encoder-Decoder 구조의 LLM에 한정되거나, LLM의 일반적인 성능을 해칠 수 있는 과도한 학습이 필요하다는 한계가 있다.

$R^2AG$는 이러한 기존 방식과 달리, 리트리버와 LLM의 기존 가중치를 유지하면서도 그 사이를 연결하는 가벼운 모듈($R^2\text{-Former}$)만을 학습시켜 비용 효율적으로 Semantic Gap을 해결한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. Retrieval Feature Extraction (리트리버 특징 추출)

리트리버 $f_R$을 통해 쿼리 $q$와 문서 $d$의 벡터 표현 $x_q, x_d$를 얻은 후, LLM이 문서 간의 관계를 파악할 수 있도록 세 가지 정량적 특징을 추출한다.

- **Relevance ($r_i$)**: 쿼리와 $i$번째 문서 간의 유사도(코사인 유사도 또는 내적)이다.
- **Precedent Similarity ($\gamma_i$)**: 해당 문서와 그 이전 문서들의 가중 합산 벡터 간의 유사도이다.
  $$\gamma_i = \text{sim} \left( x_{d_i}, \sum_{j=1}^{i-1} w_j \cdot x_{d_j} \right), \quad w_j = \frac{\exp(r_j)}{\sum_{\ell=1}^{k} \exp(r_\ell)}$$
- **Neighbor Similarity ($\zeta_i$)**: 해당 문서와 인접한 문서(이전, 이후) 간의 평균 유사도이다.
  $$\zeta_i = \frac{\text{sim}(x_{d_{i-1}}, x_{d_i}) + \text{sim}(x_{d_i}, x_{d_{i+1}})}{2}$$

최종적으로 각 문서 $i$에 대해 $\text{input}_i = \{r_i, \gamma_i, \zeta_i\}$라는 특징 벡터가 생성된다.

### 2. $R^2\text{-Former}$

추출된 $\text{input}_i$ 리스트를 입력으로 받는 가벼운 Transformer Encoder 모델이다.

- **구조**: Linear Projection $\to$ Positional Embedding $\to$ Transformer Encoder 순으로 구성된다.
- **역할**: 리스트 형태의 특징들 사이의 복잡한 의존성을 Self-attention 메커니즘을 통해 캡처하여, 최종적으로 깊은 수준의 리트리버 정보인 $H \in \mathbb{R}^{k \times h_1}$를 출력한다.

### 3. Retrieval-aware Prompting (리트리버 인지 프롬프팅)

$R^2\text{-Former}$의 출력 $H$를 LLM이 사용할 수 있도록 임베딩 공간에 주입하는 과정이다.

- **차원 정렬**: MLP 기반의 투영 층 $f_{\to h_2}$를 통해 $H$를 LLM의 토큰 임베딩 차원 $h_2$로 변환하여 $E^R = \{e^R_1, \dots, e^R_k\}$를 생성한다.
- **임베딩 구성**: LLM의 입력 시퀀스를 구성할 때, 각 문서의 텍스트 임베딩 $E_{d_i}$ 바로 앞에 해당 문서의 리트리버 정보 임베딩 $e^R_i$를 배치한다.
  $$E = [E_q, e^R_1, E_{d_1}, \dots, e^R_k, E_{d_k}]$$
이를 통해 LLM은 각 문서를 읽기 전, 해당 문서의 중요도와 관계에 대한 힌트를 먼저 얻게 된다.

### 4. Training Strategy (학습 전략)

$R^2\text{-Former}$와 LLM을 정렬하기 위해 두 가지 손실 함수를 결합하여 공동 학습을 수행한다.

- **Query-Document Matching (QDM) Loss**: $R^2\text{-Former}$가 문서의 관련성을 정확히 예측하도록 하는 이진 분류 작업이다.
  $$L_{QDM} = -\sum_{i=1}^{k} [s_i \log(\hat{s}_i) + (1-s_i) \log(1-\hat{s}_i)]$$
- **Language Modeling (LM) Loss**: 주어진 리트리버 정보와 맥락을 바탕으로 다음 토큰을 정확히 예측하도록 하는 일반적인 생성 학습 손실 함수이다.
- **최종 손실 함수**: $L = L_{QDM} + L_{LM}$

## 📊 Results

### 실험 설정

- **데이터셋**: Natural Questions (NQ-10, 20, 30), HotpotQA, MuSiQue, 2WikiMultiHopQA, DuReader(중국어) 등 5개 데이터셋을 사용하였다.
- **평가 지표**: Accuracy (Acc), F1 score, Rouge 등을 사용하였다.
- **비교 대상**: LLaMA2, LLaMA3, LongChat1.5, GPT-3.5, GPT-4 등 다양한 LLM 기반의 Standard RAG 및 CoT, RECOMP, CRAG, RAFT 등 강화된 RAG 기법들과 비교하였다.

### 주요 결과

1. **성능 향상**: $R^2AG$는 Standard RAG 대비 모든 데이터셋에서 유의미한 성능 향상을 보였다. 특히 Multi-hop 추론이 필요한 복잡한 데이터셋(HotpotQA, MuSiQue 등)에서 LLM의 추론 능력을 효과적으로 보완하였다.
2. **강건성 및 효율성**: 리트리버의 성능이 낮거나 문서의 수가 많아지는 상황에서도 $R^2AG$는 일관된 성능 향상을 보였으며, 추론 지연 시간(Latency) 증가는 단 0.8%에 불과하여 매우 효율적이다.
3. **타 방법론과의 비교**: CRAG와 같은 필터링 기반 방식은 Multi-hop 작업에서 필수 연결 고리를 제거하여 성능이 급락하는 반면, $R^2AG$는 정보를 보존하면서 가이드를 제공하므로 더 우수한 성능을 냈다. 또한, 압축 기반 방식(RECOMP 등)에서 발생하는 정보 손실 및 환각 현상을 극복하였다.
4. **시너지 효과**: 도메인 적응 학습 기법인 RAFT와 결합했을 때($R^2AG + RAFT$) 가장 높은 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **Anchor 역할의 입증**: Self-attention 맵 시각화 분석 결과, LLM이 생성 단계에서 $R^2AG$가 제공한 리트리버 정보 임베딩($e^R_i$)에 높은 주의도(Attention)를 할당하는 것이 확인되었다. 이는 리트리버 정보가 LLM이 유용한 문서에 집중하도록 돕는 '앵커' 역할을 수행함을 의미한다.
- **범용성**: 특정 LLM이나 리트리버에 종속되지 않고 플러그인 형태로 적용 가능하며, 리트리버-LLM 쌍이 고정된(Frozen) 상황에서도 학습 가능한 $R^2\text{-Former}$만으로 성능을 높일 수 있다는 점이 실용적이다.

### 한계 및 향후 과제

- **리트리버 의존성**: 본 연구는 Encoder 기반의 Dense Retriever를 가정하고 있다. Sparse Retriever나 Cross-encoder 방식의 리트리버에서도 동일한 효과가 있을지는 추가 연구가 필요하다.
- **Closed-source 모델 적용 제한**: 임베딩 공간에 직접 접근하여 정보를 주입해야 하므로, 내부 임베딩에 접근할 수 없는 폐쇄형 LLM(예: GPT-4)에는 직접 적용하기 어렵다.
- **정답 문서 부재 상황**: 모든 실험 데이터셋에 정답 문서가 포함되어 있다고 가정하였다. 실제 환경에서 관련 문서가 전혀 검색되지 않은 경우에 대한 처리(예: Self-RAG 기법 통합)가 필요하다.

## 📌 TL;DR

본 논문은 리트리버와 LLM 사이의 의미적 격차(Semantic Gap)를 해결하기 위해, 리트리버의 내부 시맨틱 특징을 추출하여 LLM의 임베딩 공간에 직접 주입하는 **$R^2AG$** 프레임워크를 제안한다. 가벼운 **$R^2\text{-Former}$** 모듈을 통해 리트리버 정보를 가공하고 이를 **Retrieval-aware Prompting**으로 전달함으로써, LLM이 수많은 문서 중 정답에 필요한 핵심 정보에 더 잘 집중하게 만든다. 실험 결과, 추론 속도 저하 없이 다양한 QA 데이터셋에서 성능을 크게 향상시켰으며, 특히 리트리버와 LLM을 수정하지 않고도 성능을 높일 수 있어 실제 시스템 적용 가능성이 매우 높은 연구이다.
