# Neural ranking models for document retrieval

Mohamed Trabelsi, Zhiyu Chen, Brian D. Davison, Jeff Heflin (2021)

## 🧩 Problem to Solve

본 논문은 정보 검색(Information Retrieval, IR) 시스템의 핵심 구성 요소인 랭킹 모델, 특히 문서 검색(Document Retrieval)을 위한 신경망 기반 랭킹 모델(Neural Ranking Models)의 발전 과정과 구조를 체계적으로 분석하는 것을 목표로 한다.

전통적인 랭킹 방식은 OKAPI/BM25와 같이 단어의 출현 빈도에 기반하거나, 전문가가 직접 설계한 수작업 특징(Hand-crafted features)을 사용하는 Learning to Rank(LTR) 방식에 의존하였다. 그러나 이러한 방식은 다음과 같은 한계가 존재한다.

1. **특징 설계의 비용:** 도메인별로 특화된 특징을 정의, 추출 및 검증하는 과정에 많은 시간이 소요된다.
2. **어휘 불일치(Vocabulary Mismatch):** 사용자의 쿼리와 문서가 의미적으로는 유사하더라도 서로 다른 단어를 사용할 경우, 단순한 정확 일치(Exact matching) 모델로는 관련 문서를 정확히 찾아낼 수 없다.
3. **문맥 이해의 부족:** 쿼리와 문서의 길이는 서로 매우 다르며, 의미적 유사성은 문맥에 따라 달라지므로 이를 포착할 수 있는 정교한 모델이 필요하다.

따라서 본 연구는 원시 텍스트 데이터를 입력으로 받아 특징 추출과 랭킹 함수를 동시에 학습하는 신경망 모델들을 비교 분석하여, 각 모델의 기여점과 한계를 명확히 하고 향후 연구 방향을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 파편화되어 있던 신경망 랭킹 모델들을 일관된 기준과 차원으로 분류하고 비교 분석한 체계적인 서베이(Survey)를 제공하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **모델 분류 체계 제안:** 신경망 구성 요소와 설계 방식에 따라 랭킹 모델을 다섯 가지 주요 그룹으로 분류하였다.
2. **9가지 세부 특징(Features) 정의:** 모델의 대칭성(Symmetric), 어텐션(Attention) 사용 여부, 토큰 순서 보존(Ordered tokens) 등 9가지 정밀한 분석 차원을 정의하여 기존 모델들을 심층 비교하였다.
3. **의미적 일치(Semantic Matching)와 관련성 일치(Relevance Matching)의 구분:** 단순한 의미 유사도를 넘어, 키워드 기반 검색에서 중요한 관련성 일치 신호의 중요성을 강조하고 두 메커니즘의 결합 필요성을 논의하였다.
4. **범용적 확장성 논의:** 문서 검색 모델의 구조가 구조화된 문서(표), 질의응답(QA), 이미지 및 비디오 검색 등 다른 검색 작업으로 어떻게 일반화될 수 있는지 분석하였다.

## 📎 Related Works

논문은 기존의 신경망 랭킹 모델 관련 서베이들이 주로 단어 임베딩(Word Embedding) 계층에만 집중했다는 점을 지적하며 차별성을 둔다.

- **기존 연구의 한계:** 이전의 서베이들은 주로 사전 학습된 임베딩의 통합 방식이나, 특정 IR 작업별 임베딩 활용법에 치중하였다. 일부 연구는 표현 학습(Representation learning)과 매칭 함수 학습(Matching function learning)이라는 두 가지 범주로만 간단히 구분하였다.
- **본 논문의 차별점:** 본 연구는 단순히 임베딩을 넘어 전체 시스템 아키텍처를 분석하며, 9가지의 세분화된 특징을 통해 모델들을 비교한다. 또한, BERT와 같은 최신 심층 문맥화 언어 모델(Deep Contextualized Language Models)이 랭킹 모델에 통합되는 방식과 그에 따른 계산 복잡도 문제를 심도 있게 다룬다.

## 🛠️ Methodology

### 1. Task Formulation 및 LTR 프레임워크

문서 검색 작업은 주어진 쿼리 $q_i$에 대해 문서 집합 $D$에서 관련성 점수가 높은 순서대로 문서를 정렬하는 작업이다. 본 논문은 이를 Learning to Rank(LTR) 프레임워크로 공식화한다.

- **목표 함수:** 쿼리 $q_i$와 문서 $d_{ij}$의 쌍에 대해 관련성 점수 $z_{ij}$를 예측하는 함수 $f_w$를 학습하는 것이다.
$$z_{ij} = f_w(q_i, d_{ij}) = M \circ F(q_i, d_{ij})$$
여기서 $F$는 특징 추출기(Feature extractor)이며, $M$은 추출된 특징을 실제 점수로 매핑하는 랭킹 모델이다.

- **학습 전략:**
  - **Pointwise:** 개별 쿼리-문서 쌍에 대해 정확한 관련성 점수를 예측하는 회귀/분류 문제로 접근한다.
  - **Pairwise:** 두 문서 중 어느 것이 더 관련성이 높은지 상대적 순서를 학습하는 이진 분류 문제로 접근한다.
  - **Listwise:** 쿼리 하나에 대응하는 전체 문서 리스트의 순서를 최적화하며, NDCG와 같은 랭킹 지표를 직접 또는 근사적으로 최적화한다.

### 2. 신경망 랭킹 모델의 주요 아키텍처 분류

#### (1) Representation-focused Models

쿼리와 문서를 각각 독립적인 신경망($NN_Q, NN_D$)에 통과시켜 고정된 길이의 특징 벡터로 변환한 후, 두 벡터 간의 유사도를 계산한다.

- **구조:** $F(q, d) = (NN_Q(q), NN_D(d))$
- **점수 계산:** 주로 코사인 유사도(Cosine similarity)나 MLP를 사용한다.
- **특징:** 동일한 네트워크를 사용하는 Siamese 구조가 많으며, 계산 효율성이 높지만 쿼리와 문서 간의 세밀한 상호작용을 놓칠 위험이 있다. (예: DSSM, C-DSSM)

#### (2) Interaction-focused Models

초기 단계부터 쿼리와 문서의 토큰 간 상호작용 행렬(Interaction Matrix)을 생성하고, 이를 신경망이 학습하여 패턴을 추출한다.

- **구조:** 쿼리 토큰과 문서 토큰 간의 코사인 유사도 등을 통해 상호작용 행렬을 먼저 만든다.
- **특징:** 세밀한 매칭 신호를 포착할 수 있어 성능이 우수하지만, 계산 비용이 높다. (예: DRMM, K-NRM)
- **주요 구성 요소:**
  - **CNN-based:** 상호작용 행렬 상의 국소적 패턴을 추출한다.
  - **RNN/GRU-based:** 상호작용 신호의 순차적 흐름이나 2차원 공간의 정보를 축적한다.

#### (3) Combined & Context-aware Models

- **Representation + Interaction:** 두 방식의 장점을 결합하여 국소적 일치와 전역적 의미를 모두 포착한다. (예: DUET)
- **Query-centric Assumption:** 관련 정보가 쿼리 단어 주변에 집중되어 있다는 가정하에, 쿼리 단어 매칭 지점 주변의 문맥을 집중적으로 분석한다. (예: DeepRank)

#### (4) Deep Contextualized Language Models (BERT 등)

Transformer 기반의 BERT 등을 사용하여 문맥이 반영된 임베딩을 추출하거나, $[CLS]$ 토큰의 hidden state를 사용하여 직접 관련성 점수를 예측한다.

- **한계:** BERT의 입력 길이 제한(512 토큰)으로 인해 긴 문서를 처리하기 위해 문서를 문장/구절 단위로 나누어 처리한 후 집계하는 방식을 사용한다.

## 📊 Results

본 논문은 특정 실험 데이터셋에 대한 성능 수치를 제시하는 제안 논문이 아니라, 기존의 수많은 모델을 분석한 서베이 논문이다. 따라서 결과는 **'분석 결과'**로 대체한다.

- **정량적 분석 (Table 1):** DSSM부터 ColBERT에 이르기까지 수십 개의 모델을 9가지 특징(Symmetry, Attention, Ordered tokens, Representation, Interaction, CI Injection, Exact matching, KB, Deep LM)으로 분석하여 표로 제시하였다.
- **주요 분석 결과:**
  - **Interaction vs Representation:** 많은 실증 연구 결과, 상호작용 기반(Interaction-focused) 모델이 표현 기반 모델보다 IR 작업에서 일반적으로 더 나은 성능을 보인다고 분석한다.
  - **BERT의 영향:** 최신 모델들은 대부분 BERT 기반이며, 이는 강력한 언어 이해 능력을 제공하지만 계산 복잡도라는 trade-off를 가진다.
  - **시너지 효과:** 의미적 매칭(Semantic matching)과 관련성 매칭(Relevance matching) 신호를 모두 사용하는 모델(예: Joint BERT)이 단일 신호만 사용하는 모델보다 우수하다.

## 🧠 Insights & Discussion

### 1. Semantic Matching vs Relevance Matching

저자는 IR 시스템에서 두 가지 매칭 신호의 구분이 매우 중요하다고 주장한다.

- **Semantic Matching:** 문장 전체의 문법, 구성, 의미적 유사성을 포착한다. QA 작업 등에 유리하다.
- **Relevance Matching:** 키워드 기반 검색에서 핵심적인 '정확 일치(Exact matching)' 신호를 포착한다. 일반적인 웹 검색과 같은 ad-hoc retrieval에서는 관련성 매칭이 필수적이다.
- **결론:** 최상의 성능을 위해서는 BERT와 같은 모델의 의미적 신호와 전통적인 관련성 매칭 신호를 결합하는 하이브리드 접근 방식이 필요하다.

### 2. 계산 복잡도와 효율성 문제

Transformer 기반 모델은 $O(m^2)$의 시간/메모리 복잡도를 가지므로 매우 긴 문서에 적용하기 어렵다. 이를 해결하기 위한 방안으로 다음과 같은 방향이 제시된다.

- **Late Interaction:** ColBERT와 같이 문서 표현을 오프라인에서 미리 계산하고, 쿼리 시점에 늦게 상호작용을 계산하여 속도를 높이는 방식이다.
- **Multi-stage Ranking:** BM25 $\rightarrow$ ElMo 기반 모델(가벼움) $\rightarrow$ BERT 기반 모델(무거움) 순으로 필터링하여 BERT가 처리해야 할 문서 수를 줄이는 전략이다.

### 3. 비판적 해석

본 논문은 광범위한 모델을 매우 세밀하게 분류하여 연구자들에게 훌륭한 지도(Map)를 제공한다. 다만, 각 모델의 성능을 동일한 데이터셋과 환경에서 직접 비교한 벤치마크 결과가 부족하여, 표에 제시된 특징들이 실제 성능 향상에 어느 정도 기여하는지에 대한 직접적인 상관관계 분석은 다소 부족한 측면이 있다.

## 📌 TL;DR

본 논문은 문서 검색을 위한 신경망 랭킹 모델들을 표현 기반(Representation)과 상호작용 기반(Interaction) 모델로 구분하고, 9가지 세부 특징을 통해 체계적으로 분석한 서베이 논문이다. 특히 **의미적 유사도(Semantic)**와 **키워드 관련성(Relevance)**이라는 두 가지 핵심 신호의 조화가 중요함을 강조하며, BERT와 같은 거대 모델의 계산 복잡도 문제를 해결하기 위한 효율적인 아키텍처 방향(Late Interaction, Multi-stage)을 제시하였다. 이 연구는 향후 고성능-저비용의 신경망 검색 엔진을 설계하려는 연구자들에게 중요한 이론적 기반을 제공한다.
