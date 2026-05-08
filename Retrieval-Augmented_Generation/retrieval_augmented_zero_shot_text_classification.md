# Retrieval Augmented Zero-Shot Text Classification

Tassallah Abdullahi, Ritambhara Singh, Carsten Eickhoff (2024)

## 🧩 Problem to Solve

본 논문은 Zero-shot Text Classification (ZSC)에서 발생하는 성능 저하 문제를 해결하고자 한다. 일반적으로 ZSC는 텍스트 쿼리와 잠재적 클래스 라벨 간의 Embedding을 생성하고, 그 거리(예: Cosine Similarity)를 측정하여 분류를 수행한다. 그러나 쿼리가 클래스의 문맥을 명시적으로 포함하지 않는 'Implicit Query'인 경우, Embedding 모델이 해당 쿼리와 클래스 간의 관계를 파악하기 어려워 분류 성능이 급격히 떨어진다는 문제가 있다.

기존에는 이를 해결하기 위해 모델을 고비용으로 재학습시키거나, 거대 언어 모델(LLM)을 사용했다. 하지만 LLM은 추론 비용이 매우 높고, 사용자가 지정한 클래스 외의 엉뚱한 답을 내놓는 제어 불가능성(Lack of user control)의 한계가 있다. 따라서 본 연구의 목표는 모델의 재학습 없이, 외부 지식을 활용해 쿼리를 보강함으로써 ZSC의 성능을 높이는 효율적인 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **QZero**라는 학습이 필요 없는(Training-free) 지식 증강 접근 방식이다. QZero의 중심 직관은 쿼리 자체에 정보가 부족하다면, 외부 지식 베이스(Wikipedia)에서 관련 카테고리를 검색하여 쿼리를 재구성(Reformulation)함으로써 Embedding 모델이 더 풍부한 문맥 정보를 가질 수 있도록 만드는 것이다.

특히, 사용하는 Embedding 모델의 특성(Static vs. Contextual)에 따라 쿼리 재구성 방식을 다르게 적용하여, 작은 크기의 모델로도 거대 모델에 상응하는 성능을 낼 수 있도록 하는 '지식 증폭기(Knowledge Amplifier)' 역할을 수행하게 한다.

## 📎 Related Works

1. **Generative LLMs 기반 ZSC**: GPT와 같은 모델은 뛰어난 Zero-shot 능력을 갖추고 있으나, 막대한 계산 자원이 필요하며 사용자가 정의한 클래스 범위 밖의 예측을 수행하는 경향이 있다.
2. **Semantic Comparison 기반 ZSC**: 텍스트와 클래스 간의 시맨틱 유사도를 비교하는 방식이다. 최근 연구들은 Wikipedia와 같은 외부 지식으로 모델을 Fine-tuning 하여 성능을 높였으나, 도메인이 빠르게 변하는 환경에서는 지속적인 재학습이 필요하다는 단점이 있다.
3. **Retrieval Augmented Learning**: 추론 시점에 외부 문서를 검색하여 쿼리를 확장하는 방식이다. 주로 생성 모델의 성능 향상에 사용되었으며, 본 논문은 이를 Embedding 모델의 ZSC에 적용하여 차별점을 둔다. 특히, 고품질 Embedding을 근사하는 ERATE와 달리, QZero는 쿼리 자체를 재구성하여 문맥을 풍부하게 하는 데 집중한다.
4. **Query Enrichment and Expansion**: 동의어, 지식 그래프 등을 이용해 쿼리를 보강하는 연구들이 있었으나, 본 논문은 이를 Zero-shot 분류 작업에 특화하여 적용하였다.

## 🛠️ Methodology

### 전체 파이프라인

QZero는 **[검색 $\rightarrow$ 쿼리 재구성 $\rightarrow$ 분류]**의 2단계 과정을 거친다.

### 1. 지식 코퍼스 및 검색 시스템 (Knowledge Corpus & Retrieval)

- **코퍼스**: English Wikipedia를 사용하며, 문서의 내용과 하단에 정의된 카테고리(Categories) 정보를 인덱싱한다. (약 585만 개의 문서)
- **검색**: 입력 쿼리와 가장 관련이 깊은 상위 50개의 Wikipedia 문서를 검색하고, 해당 문서들에 할당된 카테고리들을 추출한다. 검색기(Retriever)로는 Sparse 방식인 $\text{BM25}$와 Dense 방식인 $\text{Contriever}$를 모두 사용 가능하다.

### 2. 쿼리 재구성 (Query Reformulation)

사용하는 Embedding 모델의 종류에 따라 재구성 방식이 달라진다.

**A. Contextual Embedding 모델을 위한 재구성**
검색된 모든 카테고리들을 순서대로 연결(Concatenation)하여 새로운 쿼리를 생성한다.
$$\tilde{x} = (C_1, C_2, \dots, C_n)$$
여기서 $C_i$는 $i$번째로 랭크된 문서의 카테고리 집합이다. 모델의 입력 길이 제한을 고려하여 최대 512 토큰까지만 사용한다.

**B. Static Word Embedding 모델을 위한 재구성**
단어 수준의 입력을 받는 정적 모델을 위해, 검색된 카테고리에서 키워드를 추출하고 빈도수에 기반한 가중치를 부여한다.
$$\hat{x} = ((K_1, w_1), (K_2, w_2), \dots, (K_n, w_n))$$

- **키워드 추출($K$)**: $\text{SpaCy}$의 POS tagging(명사 추출), 대문자 기반 추출(고유명사), 또는 의료 도메인의 경우 $\text{MedCAT}$을 사용한다.
- **가중치($w$)**: 재구성된 쿼리 내에서 해당 키워드가 등장한 빈도를 가중치로 설정한다.

### 3. Zero-shot 분류 절차

- **Contextual 모델**: 재구성된 쿼리 $\tilde{x}$와 각 클래스 라벨의 Embedding 간의 Cosine Similarity를 계산하여 가장 유사도가 높은 클래스를 선택한다.
- **Static 모델**: 각 키워드 $K$와 클래스 라벨 $\sim$ 사이의 유사도에 가중치 $w$를 곱하여 합산한 최종 점수가 가장 높은 클래스를 선택한다.
$$\text{Score}(\sim) = \sum_{(K, w) \in \hat{x}} \text{cosine\_similarity}(\text{embed}(K), \text{embed}(\sim)) \times w$$

## 📊 Results

### 실험 설정

- **데이터셋**: AG News, DBPedia, Yahoo! Answers, Yummly, TagMyNews, Ohsumed 등 6개의 다양한 도메인 데이터셋을 사용하였다.
- **비교 모델**:
  - Static: Word2Vec, GloVe, FastText
  - Contextual: All-mpnet-base-v2, TE-3-small, TE-3-large (OpenAI)
- **지표**: Test set에 대한 평균 정확도(Accuracy)를 측정하였다.

### 주요 결과

1. **전반적인 성능 향상**: 거의 모든 모델에서 QZero 적용 시 성능이 향상되었다. 특히 TagMyNews 데이터셋에서 Word2Vec은 13.00%의 큰 폭으로 향상되었으며, TE-3-large 또한 6.61% 향상되었다.
2. **소형 모델의 효율성**: QZero를 적용한 소형 모델(Word2Vec)이 QZero를 적용하지 않은 거대 모델(TE-3-small, TE-3-large)보다 더 높은 성능을 보이는 경우가 확인되었다. 이는 계산 자원이 제한된 환경에서 매우 유용한 결과이다.
3. **도메인 적응력**: 의료(Ohsumed) 및 요리(Yummly) 도메인에서 뚜렷한 성능 향상이 나타났다. 특히 Yummly의 경우 정적 임베딩 모델들이 최대 38.00%의 성능 향상을 보였다.
4. **거대 모델의 성능 하락**: 일부 데이터셋(DBPedia, Yahoo Answers)에서 TE-3-large와 같은 거대 모델의 성능이 소폭 하락하는 현상이 관찰되었다. 이는 거대 모델이 이미 학습 데이터에서 해당 지식을 가지고 있어, 검색된 정보가 중복되거나 노이즈로 작용했을 가능성이 크다.
5. **검색 모델 비교**: 일반적인 뉴스/QA 작업에는 Dense retriever(Contriever)가 유리했으나, 문서가 길거나 특수 도메인(의료, 요리)에서는 Sparse retriever(BM25)가 더 효과적이었다.

## 🧠 Insights & Discussion

### 강점 및 기여

QZero는 모델의 가중치를 수정하지 않고 입력 쿼리를 보강하는 것만으로 ZSC 성능을 높였다. 이는 도메인이 빠르게 변하는 환경에서 모델을 매번 재학습시킬 필요 없이 지식 베이스(Wikipedia)만 업데이트하면 된다는 점에서 매우 실용적이다. 또한, 검색된 카테고리를 통해 모델이 왜 특정 클래스로 분류했는지에 대한 해석 가능성(Interpretability)을 제공한다는 점이 큰 강점이다.

### 한계 및 비판적 해석

1. **거대 모델의 역효과**: 거대 모델에서 성능이 하락하는 지점은 '지식의 중복' 문제로 해석된다. 이는 QZero가 모든 모델에 일률적으로 적용되기보다, 모델의 사전 지식 수준에 따라 검색 정보의 양이나 필터링 강도를 조절해야 함을 시사한다.
2. **검색 품질 의존성**: 본 논문의 성능은 Wikipedia의 카테고리 정보에 전적으로 의존한다. 만약 Wikipedia에 정의되지 않은 최신 용어나 매우 지엽적인 도메인 정보가 필요할 경우, 검색 결과가 부적절하여 오히려 오분류를 유발할 위험이 있다.
3. **토큰 제한**: Contextual 모델의 경우 512 토큰 제한으로 인해 많은 양의 카테고리를 활용하지 못하는 한계가 있으며, 이는 성능의 상한선을 결정짓는 요소가 된다.

## 📌 TL;DR

본 논문은 외부 지식(Wikipedia)을 이용해 입력 쿼리를 재구성함으로써, 추가 학습 없이 Zero-shot 텍스트 분류 성능을 높이는 **QZero** 프레임워크를 제안한다. 정적/문맥적 임베딩 모델의 특성에 맞춰 쿼리 보강 방식을 다르게 적용하였으며, 실험 결과 소형 모델이 거대 모델에 필적하는 성능을 낼 수 있음을 입증하였다. 이 연구는 특히 자원이 제한된 환경이나 전문 도메인의 ZSC 작업에서 비용 효율적인 대안이 될 가능성이 높다.
