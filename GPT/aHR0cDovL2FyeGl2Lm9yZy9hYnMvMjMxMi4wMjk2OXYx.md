# Rank-without-GPT: Building GPT-Independent Listwise Rerankers on Open-Source Large Language Models

Xinyu Zhang, Sebastian Hofstätter, Patrick Lewis, Raphael Tang, Jimmy Lin (2023)

## 🧩 Problem to Solve

텍스트 검색(Text Retrieval) 시스템은 일반적으로 효율적인 리트리버(Retriever)가 후보군을 뽑고, 이후 리랭커(Reranker)가 이들의 순위를 정교하게 재조정하는 다단계 파이프라인을 따른다. 최근 대규모 언어 모델(LLM)의 생성 능력과 긴 컨텍스트 처리 능력을 활용하여, 여러 문서의 리스트를 한 번에 입력받아 직접 정렬 순서를 출력하는 **Listwise Reranking** 방식이 등장하며 SOTA(State-of-the-art) 성능을 기록하고 있다.

그러나 기존의 Listwise 리랭커 연구들은 추론 시 GPT 모델을 직접 사용하거나, 학습 단계에서 GPT 기반 모델을 교사 모델(Teacher Model)로 사용하는 등 **GPT 모델에 대한 의존성**이 매우 높다. 이는 과학적 재현성(Reproducibility) 측면에서 단일 장애점(Single point of failure)이 될 뿐만 아니라, 현재의 연구 결과들이 일반적인 LLM이 아닌 오직 GPT 모델에서만 유효한 것일 수 있다는 우려를 낳는다. 따라서 본 논문의 목표는 GPT에 대한 의존성을 완전히 제거하고, 오픈 소스 LLM만을 활용하여 효과적인 Listwise 리랭커를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 다음과 같다.

1. **GPT 독립적 Listwise 리랭커 구현**: GPT 모델에 전혀 의존하지 않고도 GPT-3.5 기반 리랭커보다 성능이 뛰어나며, GPT-4 기반 모델과 대등한 수준의 성능을 내는 Listwise 리랭커를 최초로 구현하였다.
2. **학습 데이터 품질의 중요성 규명**: 기존의 Pointwise 랭킹용으로 구축된 데이터셋으로는 Listwise 리랭커를 학습시키기에 불충분하며, 고품질의 Listwise 랭킹 데이터(정교하게 정렬된 리스트)가 모델 성능의 핵심 병목 구간임을 밝혀냈다.
3. **데이터 효율성 증명**: 단 5,000개의 고품질 쿼리 세트(각 쿼리당 20개의 문서 리스트 포함)만으로도 효과적인 리랭커 구축이 가능함을 보여, 향후 인간이 직접 어노테이션한 Listwise 데이터셋 구축의 현실적 가능성을 제시하였다.

## 📎 Related Works

기존의 리랭킹 접근 방식은 크게 세 가지 패러다임으로 나뉜다.
- **Pointwise Reranking**: 각 문서와 쿼리의 관련성 점수를 독립적으로 계산하여 정렬한다. BERT나 T5 기반의 모델들이 대표적이다.
- **Pairwise Reranking**: 두 문서의 상대적 우위를 비교하여 순위를 결정한다.
- **Listwise Reranking**: 여러 문서의 리스트를 동시에 입력받아 최종 순위를 직접 예측한다. RankGPT, LRL 등이 있으며, 이들은 주로 GPT 모델의 제로샷 능력이나 생성 능력을 활용한다.

본 논문이 언급하는 Listwise 랭킹은 정보 검색(IR) 분야의 전통적인 Listwise Loss(각 문서의 점수를 독립적으로 내되 손실 함수만 리스트 단위로 계산하는 방식)와는 다르다. 본 논문의 방식은 모델이 **리스트 전체를 한 번에 처리하고 텍스트 생성 형태로 순위를 출력**하는 생성적(Generative) 접근 방식이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
본 연구는 오픈 소스 LLM인 **Code-LLaMA-Instruct** (7B, 13B, 34B)를 기반으로 하며, 쿼리와 문서 리스트를 입력받아 정렬된 식별자(Identifier)의 시퀀스를 생성하는 텍스트-투-텍스트(Text-to-Text) 구조를 가진다.

### 2. 학습 데이터 구축 (Silver Ranking)
기존의 Pointwise 데이터(단순히 관련 있음/없음의 이진 레이블)는 False-negative 문서가 많고 관련성의 세밀한 단계(Graded relevance)를 반영하지 못해 Listwise 학습에 부적합하다. 이를 해결하기 위해 본 연구는 다음과 같은 **Silver Ranking** 데이터를 생성하여 사용하였다.
- **P-GT (Pointwise Ground Truth)**: 레이블링된 관련 문서를 앞에 배치하고 나머지는 임의로 배치한 베이스라인 데이터이다.
- **Silver Ranking**: 성능이 검증된 기존 시스템의 랭킹 결과를 정답으로 사용한다.
    - **BM25**: 전통적인 어휘 매칭 기반 알고리즘.
    - **Contriever+ft**: MS MARCO로 파인튜닝된 Contriever 모델.
    - **co.rerank**: Cohere의 상용 리랭킹 API (가장 높은 품질의 데이터 생성).

### 3. 프롬프트 설계 및 추론 절차
모델은 다음과 같은 형식의 프롬프트를 입력받는다.
- **입력**: "I will provide you with {num} passages... Rank the passages based on their relevance to the search query: {query}." 이후 $[1] \text{제목} \text{본문} \dots$ 형태로 문서들을 나열한다.
- **출력**: 모델은 오직 `[4] > [2] > [5] > [1] > [3]`와 같은 식별자 정렬 순서만을 생성하도록 강제된다.

### 4. 슬라이딩 윈도우(Sliding Window) 전략
LLM의 입력 길이 제한으로 인해 수백 개의 문서를 한 번에 처리할 수 없다. 따라서 다음과 같은 전략을 사용한다.
- 크기가 $n$인 윈도우를 리스트의 끝에서 앞방향으로 이동시킨다.
- 각 단계에서 $m$개의 문서만큼 스트라이드(Stride)하며 윈도우 내 문서들을 리랭킹한다.
- 상위 $(n-m)$개의 문서를 보존하여 다음 윈도우의 입력으로 사용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: TREC-DL-19, TREC-DL-20.
- **평가 지표**: $nDCG@10$.
- **기준선**: monoBERT, monoT5, RankLLaMA(Pointwise LLM), RankGPT-3.5/4(Listwise GPT).

### 2. 주요 결과
- **GPT 모델과의 비교**: 34B 모델 기반의 Rank-wo-GPT는 GPT-3.5 기반 리랭커보다 성능이 13% 높았으며, GPT-4 기반 모델의 성능 대비 약 97% 수준에 도달하였다.
- **데이터 품질의 영향**: BM25 $\rightarrow$ Contriever $\rightarrow$ co.rerank 순으로 교사 모델의 품질이 높아질수록 학생 모델(LLM)의 성능이 선형적으로 증가하였다. 이는 데이터 품질이 성능의 핵심임을 시사한다.
- **데이터 양의 영향**: 학습 데이터가 5k 쿼리에서 10k 쿼리로 증가할 때 성능 향상이 뚜렷했으나, 20k로 늘렸을 때는 향상이 미미했다. 즉, 약 10k의 고품질 데이터만으로도 충분한 학습이 가능하다.
- **모델 크기의 영향**: 모델 크기가 7B $\rightarrow$ 13B $\rightarrow$ 34B로 커질수록 성능이 일관되게 향상되었으며, 13B 모델은 이미 일부 교사 모델의 성능을 추월하였다.

## 🧠 Insights & Discussion

### 1. 슬라이딩 윈도우의 한계 (Trapped Phenomenon)
본 논문은 히트맵 분석을 통해 Listwise 리랭커가 슬라이딩 윈도우 전략을 사용할 때, 많은 관련 문서들이 **국소적 블록(Local block)에 갇히는 현상**이 발생함을 발견하였다. 즉, 관련 문서가 리스트의 아주 먼 곳에서 상위권으로 이동하기보다는, 현재 윈도우나 바로 다음 윈도우 범위 내에서만 순위가 소폭 상승하는 경향이 있다.

### 2. Pointwise vs Listwise
Listwise 리랭커는 Pointwise 방식보다 정답 레이블이 없는(Unlabeled) 문서를 상위권으로 올리는 경향이 더 강했다. 이는 Listwise 방식이 문서 간의 상대적 관계를 파악하는 데 강점이 있지만, 동시에 노이즈에 더 취약할 수 있음을 의미한다.

### 3. 도메인 일반화 문제
BEIR 데이터셋을 이용한 Zero-shot 테스트 결과, 파인튜닝된 Listwise 리랭커의 도메인 외 일반화 성능은 낮게 나타났다. 이는 향후 다양한 도메인에 적응시키기 위한 추가 연구가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 GPT 모델에 의존하지 않고 오픈 소스 LLM(Code-LLaMA)을 활용하여 고성능의 **Listwise 리랭커**를 구축하는 방법을 제안하였다. 연구 결과, 모델의 구조보다 **학습 데이터의 정렬 품질(Ranking Quality)**이 성능을 결정짓는 핵심 요소임을 확인하였으며, 고품질의 Silver data를 통해 GPT-4에 근접하는 성능을 달성하였다. 이 연구는 폐쇄적인 GPT 생태계에서 벗어나 오픈 소스 기반의 효율적인 검색 시스템을 구축할 수 있는 가능성을 열었으며, 향후 인간 주도의 고품질 Listwise 데이터셋 구축의 필요성을 강조한다.