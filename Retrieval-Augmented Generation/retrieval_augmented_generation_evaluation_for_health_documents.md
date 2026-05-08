# Retrieval Augmented Generation Evaluation for Health documents

Ceresa, M; Bertolini, L., Comte, V.; Spadaro N.; Raffael, B.; Toussaint, B.; Consoli, S.; Muñoz Piñeiro A.; Patak, A.; Querci M.; Wiesenthal T. (2024)

## 🧩 Problem to Solve

본 연구는 보건 의료 분야의 과학 논문 및 정책 보고서의 폭발적인 증가로 인한 정보 과부하 문제를 해결하고자 한다. 연구자와 정책 입안자들은 최신 연구 성과를 빠르게 파악해야 하지만, 수동적인 문헌 검토 방식으로는 더 이상 불가능한 수준에 이르렀다.

Large Language Models (LLMs)는 이러한 정보를 처리하고 요약하는 데 강력한 도구가 될 수 있으나, 의료 도메인에 적용할 때 다음과 같은 치명적인 문제들이 발생한다. 첫째, 사실과 다른 정보를 그럴듯하게 생성하는 Hallucination(환각) 현상이 발생하며, 이는 의료 분야에서 심각한 결과를 초래할 수 있다. 둘째, 모델의 의사결정 과정을 이해하기 어려운 Black-box 특성으로 인해 설명 가능성(Explainability)이 부족하다. 셋째, 민감한 의료 데이터의 개인정보 유출 및 편향성(Bias) 문제가 존재한다.

따라서 본 논문의 목표는 의료 문서 처리에 있어 안전하고 신뢰할 수 있는 LLM 활용 방안을 조사하고, 이를 위해 Retrieval Augmented Generation (RAG) 기술을 적용한 참조 파이프라인인 RAGEv를 구축하여 그 효용성과 한계를 평가하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 도메인에서 LLM의 신뢰성을 높이기 위한 RAG 시스템의 설계 및 체계적인 평가 프레임워크를 제시한 점이다. 주요 기여 사항은 다음과 같다.

1. **RAGEv (Retrieval Augmented Generation Evaluation) 구축**: 최신 RAG 기법들을 적용하여 의료 및 과학 문서 분석을 수행하는 Proof-of-Concept (PoC) 파이프라인을 개발하였다. 특히 단순한 Top-K 검색의 한계를 극복하기 위해 모든 문서를 개별적으로 처리하는 SHy 파이프라인을 제안하였다.
2. **RAGEv-Bench 데이터셋 공개**: 시스템의 정확성과 진실성을 검증하기 위해 PubMedQA의 부분 집합과 실제 정책 지원 업무와 관련된 세 가지 사용자 맞춤형 데이터셋(Horizon Research, Virtual Human Twins, Bacteriophages)을 포함하는 벤치마크를 구축하였다.
3. **다양한 검색 전략의 정량적/정성적 분석**: Vector Search, Full-text Search, Hybrid Search, ColBERTv2 등 다양한 검색 파이프라인의 성능을 비교 분석하여, 의료 문서 요약 및 질의응답에 최적화된 구성 요소를 식별하였다.

## 📎 Related Works

논문에서는 LLM의 성능을 높이기 위한 세 가지 주요 접근 방식인 Fine-tuning, Prompt Engineering, 그리고 RAG를 비교 설명한다. Fine-tuning은 모델의 가중치를 직접 수정하여 도메인 특화 용어에 적응시키지만 계산 비용이 매우 높고 유연성이 떨어진다. Prompt Engineering은 모델 수정 없이 입력을 최적화하는 방식이며, 특히 파라미터 규모가 큰 모델에서 나타나는 In-Context Learning (ICL) 능력을 활용한다.

RAG는 외부 지식 베이스에서 관련 정보를 검색하여 모델의 입력으로 제공함으로써, 모델을 재학습시키지 않고도 최신 정보와 사실에 기반한 답변을 생성하게 한다. 기존의 RAG 시스템들은 검색 단계에서의 Precision/Recall 문제(관련 없는 청크 선택 또는 중요 정보 누락)와 생성 단계에서의 Hallucination 및 중복성 문제를 공통적으로 겪고 있다. 본 연구는 이러한 한계를 극복하기 위해 다양한 검색 파이프라인을 실험적으로 검증함으로써 기존 방식과의 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

RAGEv 시스템은 전형적인 RAG의 3단계 구조인 Indexing $\rightarrow$ Retrieval $\rightarrow$ Generation 과정을 따른다.

1. **Indexing**: PDF, HTML 등 다양한 형식의 raw 데이터를 텍스트로 변환한 후, 작은 단위의 Chunk로 분할한다. 이후 Embedding 모델을 통해 벡터로 변환하여 Vector Database(Milvus)에 저장한다.
2. **Retrieval**: 사용자 쿼리를 벡터로 변환하고, 저장된 벡터들과의 유사도를 계산하여 가장 관련성이 높은 $K$개의 청크를 추출한다.
3. **Generation**: 추출된 컨텍스트와 쿼리를 프롬프트로 구성하여 LLM(GPT-4, Llama 3 등)에 전달하고 최종 답변을 생성한다.

### 주요 검색 파이프라인 및 SHy 방식

본 연구에서는 다음과 같은 다양한 검색 전략을 실험하였다.

- **Vector Search**: 고차원 벡터 공간에서의 유사도 기반 검색.
- **Full-text Search**: 키워드 일치 기반의 전통적인 검색.
- **Hybrid Search**: 위 두 방식을 결합하고 Reciprocal Rank Fusion (RRF) 알고리즘을 통해 순위를 재조정하는 방식.
- **ColBERTv2**: Late Interaction 메커니즘을 사용하여 토큰 수준의 세밀한 임베딩을 통해 검색 정확도를 높인 방식.
- **SHy (Single Hybrid)**: 본 연구에서 제안한 방식으로, 컬렉션 내의 각 문서를 개별적인 컬렉션으로 취급하여 검색한다. 이는 Top-K 검색이 놓칠 수 있는 수평적 지식(Horizontal Knowledge)을 확보하여 광범위한 요약 질문에 더 적절한 답변을 제공하기 위함이다.

### 시스템 아키텍처

RAGEv는 Hybrid Cloud 아키텍처를 채택하였다.

- **On-premises (JRC Datacentre)**: 모델 학습 및 Fine-tuning을 위한 BDAP 서버와 NextJS 기반의 프론트엔드를 운영한다.
- **Azure Cloud**: 지식 베이스 관리 및 API 서비스, Vector DB(Milvus)를 포함하는 백엔드 시스템을 운영한다.
- **GPT@JRC**: 보안이 확보된 환경에서 다양한 상용 및 오픈소스 LLM을 서빙하는 인프라를 활용한다.

### 평가 지표 및 방정식

정량적 평가를 위해 분류 작업에는 Accuracy, Precision, Recall, F1 Score를 사용하였으며, 생성 작업에는 ROUGE와 BERTScore를 사용하였다.

특히 ROUGE-1은 예측값과 정답 간의 단일 단어(uni-gram) 중첩도를 측정하며, 다음과 같은 수식으로 정의된다.
$$\text{ROUGE-1} = \frac{\sum_{S \in \text{Ref}} \text{Count}_{\text{match}}(\text{uni-gram})}{\sum_{S \in \text{Ref}} \text{Count}(\text{uni-gram})}$$
여기서 $\text{Count}_{\text{match}}$는 정답(Gold Standard)과 시스템 생성 답변에서 공통으로 나타나는 n-gram의 최대 개수를 의미한다.

BERTScore는 단어 간의 단순 일치가 아닌, 사전 학습된 BERT 임베딩의 Cosine Similarity를 이용해 의미론적 유사도를 측정한다.

## 📊 Results

### 자동 성능 평가 (APE)

PubMedQA 데이터셋을 활용한 720회의 팩토리얼 실험 결과, RAG 구성 요소를 추가한 모든 파이프라인이 Baseline(RAG 없음) 대비 월등한 성능 향상을 보였다. 특히 **SHy 파이프라인**이 가장 높은 성능을 기록하였으며, 이진 질의(Yes/No)에 대해 평균 Precision 0.85, 긴 답변 생성에 대해 BERTScore F1 0.83을 달성하였다.

### 사용자 편의성 및 정성 평가 (Usability Checks)

HR, VHT, AMR 세 가지 실제 도메인 데이터셋을 통해 전문가 평가를 수행한 결과는 다음과 같다.

- **강점**: 개념적인 질문이나 요약 요청에 대해 매우 높은 품질의 답변을 제공하며, 환각 현상을 효과적으로 억제한다.
- **약점**: 표(Table), 그림(Figure), 소제목(Subtitle)에서 정보를 추출하는 능력이 현저히 떨어진다. 이는 텍스트 분할(Chunking) 과정에서 표의 구조가 파괴되어 발생하는 문제로 분석되었다.
- **수치 결과**: VHT 데이터셋의 경우, 15개 질문 중 9개가 4-5점(정답 또는 사소한 누락)을 기록하여 전반적으로 유용한 도구임을 입증하였다.

### 정량적 분석 및 상관관계

인간 평가자의 점수와 기계적 지표를 비교한 결과, ROUGE보다는 의미론적 유사도를 측정하는 **BERTScore F1**이 인간의 평가 점수와 더 강한 양의 상관관계를 보였다. 이는 의료 문서의 특성상 단어의 정확한 일치보다 의미의 정확성이 더 중요하다는 점을 시사한다.

## 🧠 Insights & Discussion

본 연구를 통해 RAG 기술이 의료 도메인 LLM의 고질적인 문제인 Hallucination을 최소화하고 신뢰성을 높일 수 있음을 확인하였다. 특히, 특정 문서의 일부만 추출하는 일반적인 RAG보다 모든 문서를 훑는 SHy 방식이 종합적인 분석 작업에 더 유리하다는 인사이트를 얻었다.

그러나 다음과 같은 한계점이 명확히 드러났다.
첫째, **비정형 데이터 처리의 한계**이다. 현재의 시스템은 텍스트 기반의 임베딩에 의존하므로 표나 그림 내의 수치 데이터를 정확히 읽어내지 못한다.
둘째, **프롬프트 민감도(Prompt Sensitivity)**이다. 사용자가 질문을 어떻게 구성하느냐(Wording)에 따라 답변의 품질이 크게 달라지며, 이는 일반 사용자에게 상당한 학습 곡선(Learning Curve)을 요구한다.
셋째, **참조의 정확성**이다. 시스템이 제공하는 참조 문헌(Reference) 중 일부가 실제 답변 생성에 사용되지 않았거나 불필요한 내용이 포함되는 경우가 발견되었다.

결론적으로, RAGEv는 초기 문헌 조사 시간을 획기적으로 줄여줄 수 있는 강력한 도구이지만, 최종 답변의 검증은 반드시 도메인 전문가(Human-in-the-loop)에 의해 이루어져야 한다.

## 📌 TL;DR

본 논문은 의료 및 과학 문서의 정보 과부하를 해결하기 위해 RAG 기반의 참조 파이프라인 **RAGEv**와 벤치마크 데이터셋 **RAGEv-Bench**를 제안하였다. 다양한 검색 전략 중 모든 문서를 개별적으로 처리하는 **SHy 파이프라인**이 가장 우수한 성능을 보였으며, 특히 의미론적 평가 지표인 BERTScore가 인간의 평가와 높은 상관관계를 가짐을 확인하였다. 표/그림 데이터 처리의 한계와 프롬프트 의존성이라는 과제가 남아있으나, 본 연구는 의료 도메인에서 LLM을 안전하고 신뢰할 수 있게 활용하기 위한 실무적인 가이드라인과 기술적 토대를 제공한다.
