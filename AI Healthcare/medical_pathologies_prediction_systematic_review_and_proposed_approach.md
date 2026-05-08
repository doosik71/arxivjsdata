# Medical Pathologies Prediction : Systematic Review and Proposed Approach

Chaimae Taoussi, Imad Hafidi, and Abdelmoutalib Metrane (2022)

## 🧩 Problem to Solve

현대 헬스케어 분야에서는 의료 기록, 환자 데이터, 검사 결과 등 방대한 양의 빅데이터가 생성되고 있다. 하지만 이러한 데이터는 구조화된 데이터뿐만 아니라 비구조화된 텍스트, 반구조화된 데이터 등 형식이 매우 다양하고 복잡하여, 기존의 전통적인 관리 및 처리 방식으로는 효율적으로 활용하기 어렵다.

특히, 의료 전문가가 단순히 증상만으로 질병을 정확하게 예측하는 것은 매우 도전적인 과제이며, 오진이나 의료 사고의 위험이 존재한다. 따라서 대규모의 Electronic Health Records (EHR) 데이터를 효율적으로 처리하고 분석하여, 빈번하게 발생하는 병리적 상태(pathologies)를 높은 정밀도와 빠른 시간 내에 예측할 수 있는 체계적인 방법론의 필요성이 대두되었다.

## ✨ Key Contributions

본 논문의 핵심 기여는 최신 AI 및 빅데이터 기술을 활용한 의료 병리 예측을 위해 다음과 같은 두 가지 측면의 접근 방식을 제시한 것이다.

첫째, 2018년부터 2022년까지의 문헌을 대상으로 PRISMA 가이드라인을 준수한 체계적 문헌 고찰(Systematic Review)을 수행하였다. 이를 통해 데이터 수집, 전처리, 매핑, 분류 및 클러스터링, 그리고 AI 기반 예측이라는 다섯 가지 핵심 연구 질문(RQ)에 대한 최신 기술 동향을 분석하였다.

둘째, 문헌 고찰 결과를 바탕으로 데이터 수집부터 최종 병리 예측까지 이어지는 5단계의 통합 파이프라인(Proposed Approach)을 설계하였다. 이 설계는 특히 비구조화된 의료 데이터의 정형화와 의료 전문 용어의 표준화(Mapping) 단계를 포함함으로써 예측 모델의 정확도를 높이는 데 집중한다.

## 📎 Related Works

본 논문은 체계적 문헌 고찰을 통해 다음과 같은 관련 연구들을 분석하였다.

- **데이터 수집 및 전처리:** Natural Language Processing (NLP) 및 NLTK 라이브러리를 활용하여 전자 의료 기록(EMR)에서 의료 이벤트를 탐지하고 텍스트를 정제하는 연구들이 진행되었다. 특히 GATE (General Architecture for Text Engineering)와 같은 프레임워크가 의료 텍스트 분석에 널리 사용되고 있음을 확인하였다.
- **데이터 매핑:** Unified Medical Language System (UMLS)을 활용하여 다양한 의료 용어를 표준화하고, Bio-YODIE와 같은 Named Entity Recognition (NER) 시스템을 통해 생물 의학적 엔티티를 식별하는 접근 방식이 제시되었다.
- **분류 및 클러스터링:** 환자 군집화를 위해 $k$-means clustering, Random Forest, Latent Dirichlet Allocation (LDA) 등이 사용되었으며, 최근에는 데이터 프라이버시 보호를 위해 Fully Homomorphic Encryption (FHE)을 적용한 Naive Bayes 분류 방식 등이 연구되었다.
- **병리 예측:** 질병 예측을 위해 Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Logistic Regression 등이 활용되고 있으며, 특히 암이나 알츠하이머와 같은 중증 질환 예측에서 딥러닝의 효용성이 입증되었다.

## 🛠️ Methodology

저자들은 문헌 분석 결과를 토대로 다음과 같은 5단계의 병리 예측 시스템 구조를 제안한다.

### 1. 데이터 수집 (Collecting Data)

다양한 형식(PDF, RTF, HTML, XML 등)의 의료 데이터를 수집한다.

- **Data Analyzer:** 수집된 파일을 비구조화 데이터(Text: RTF, PDF)와 반구조화 데이터(Relational DB: .csv, .db)로 분리한다.
- **Wrapper 및 Parser:** 각 데이터 소스에서 쿼리를 실행하여 결과를 JSON 문서로 변환하고, 최종적으로 메타데이터와 분석에 필요한 필드만 추출한 새로운 EMR 데이터셋을 생성한다.

### 2. 데이터 전처리 (Preprocessing Data)

수집된 DME(Data Medical Electronic) 데이터셋에 대해 데이터 정제, 통합, 축소 및 변환을 수행한다.

- **NLTK (Natural Language Toolkit):** Python 기반의 NLTK 라이브러리를 사용하여 Treebank word tokenizer와 POS tagger를 적용함으로써, 텍스트 데이터를 구조화된 SDME 데이터셋으로 변환한다.

### 3. 의료 데이터 매핑 (Mapping Medical Data)

전처리된 SDME 데이터셋의 생물 의학 용어를 표준화한다.

- **UMLS API 및 GATE (Bio-YODIE) API:** 이 도구들을 사용하여 텍스트 내의 용어를 UMLS 메타테사우러스(metathesaurus)에 매핑한다.
- **추출 항목:** 매핑 결과로 필수 용어(Mandatory terms), 개념(Concept), 의미론적 유형(Semantic Type), 엔티티 유형(Entity type)을 추출하여 'Mapped SDME' 데이터셋을 구축한다.

### 4. 분류 및 클러스터링 (Classification and Clustering)

환자들의 프로필을 유사한 특성별로 그룹화한다.

- **$k$-means Clustering:** 의료 전문가가 사전에 정의한 규칙을 기반으로 $k$-means 알고리즘을 적용하여 환자들을 $k$개의 클러스터로 분류한다. 이를 통해 환자별 특성이 반영된 'Clustered Mapped SDME' 데이터셋이 생성된다.

### 5. 병리 예측 (Pathology Prediction)

최종적으로 AI 모델을 통해 질병을 예측한다.

- **Recurrent Neural Networks (RNN):** 의료 데이터의 시계열적 특성과 순차적 정보를 처리하기 위해 RNN 알고리즘을 사용한다. RNN은 내부 루프와 메모리를 통해 이전 계산 값을 유지하므로 순차 데이터 분석에 적합하다.
- **출력 변수:** 모델은 최종적으로 예측된 병리(Pathologies), 최적 예측 결과(Best Prediction), 예측 정밀도(Best Precision)를 포함하는 데이터셋을 출력한다.

## 📊 Results

본 논문은 제안한 방법론에 대한 직접적인 실험 수치나 성능 평가 결과(예: Accuracy, F1-score 등)를 제시하지 않았다. 대신, 체계적 문헌 고찰(Systematic Review)에 대한 정량적 결과만을 명시하고 있다.

- **문헌 선정 과정:** 2018년부터 2022년까지 Scopus, PubMed, Google Scholar에서 총 734편의 논문을 식별하였다.
- **필터링 결과:** 제목 및 초록 분석을 통해 614편을 제외하였고, 이후 상세 제외 기준(중복, 연구 질문 불일치 등)을 적용하여 120편을 추가 제외하였다.
- **최종 분석 대상:** 최종적으로 49편의 논문이 선정되어 분석에 활용되었으며, 이 논문들이 제안하는 방법론의 근거가 되었다.

## 🧠 Insights & Discussion

본 논문은 단편적인 예측 모델 제시가 아니라, 데이터 수집부터 예측까지의 전체 파이프라인을 정의했다는 점에서 강점을 가진다. 특히 전처리 단계에서 NLTK를, 표준화 단계에서 UMLS를, 예측 단계에서 RNN을 배치함으로써 데이터의 품질 향상이 모델의 성능으로 이어지도록 설계하였다.

그러나 몇 가지 한계점이 존재한다. 첫째, 제안한 시스템의 실제 구현 결과나 성능 검증 데이터가 누락되어 있어, 이론적인 프레임워크 수준에 머물러 있다. 둘째, 문헌 고찰 대상 데이터베이스가 3개로 제한적이며, Medline이나 Web of Science와 같은 추가 데이터베이스를 활용하지 않았다. 셋째, RNN 외에 최신 트랜스포머(Transformer) 기반 모델(예: BERT, BioBERT)에 대한 고려가 부족하여 최신 SOTA(State-of-the-art) 성능을 반영하지 못했을 가능성이 크다.

## 📌 TL;DR

본 논문은 의료 빅데이터를 활용한 병리 예측을 위해 최신 AI/ML 기술을 분석하고, **[데이터 수집 $\rightarrow$ NLTK 전처리 $\rightarrow$ UMLS 매핑 $\rightarrow$ $k$-means 클러스터링 $\rightarrow$ RNN 예측]**으로 이어지는 5단계 통합 파이프라인을 제안한다. 이 연구는 비구조화된 의료 데이터를 표준화하여 예측 모델의 입력값으로 만드는 체계적인 프로세스를 정의함으로써, 향후 정밀 의료 및 자동 진단 시스템 구축을 위한 기초 프레임워크를 제공한다.
