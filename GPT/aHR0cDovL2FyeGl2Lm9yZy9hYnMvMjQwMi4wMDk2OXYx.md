# SPARQL Generation with Entity Pre-trained GPT for KG Question Answering

Diego Bustamante, Hideaki Takeda (2024)

## 🧩 Problem to Solve

본 논문은 지식 그래프(Knowledge Graph, KG)에 저장된 방대한 정보에 대해 비전문가(non-programmer)가 자연어 질문을 통해 효율적으로 접근할 수 있도록, 자연어 질문을 SPARQL 쿼리로 변환하는 문제를 해결하고자 한다. 

지식 그래프의 데이터 규모가 기하급수적으로 증가함에 따라 정보 추출의 중요성이 커지고 있으나, SPARQL과 같은 쿼리 언어의 높은 진입 장벽이 문제가 된다. 최근 거대 언어 모델(LLM)의 발전에도 불구하고, 일반적인 LLM들은 SPARQL 쿼리 생성이라는 특수 작업에서 낮은 성능을 보이는 경향이 있다. 따라서 본 연구의 목표는 SPARQL 생성 작업의 난이도를 높이는 핵심 요인을 식별하고, 이를 개선하기 위한 훈련 방법론과 모델 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Entity Linking(개체 연결)과 SPARQL 생성을 분리**하고, 모델이 개체(Entity) 정보를 정확하게 처리하도록 **개체 정체성 함수(Identity Function)에 대한 사전 학습(Pre-training)**을 수행하는 것이다.

저자들은 모델이 쿼리 템플릿의 문법을 배우는 과정에서, 질문 내의 소수 개체 이름을 쿼리 내의 IRI(Internationalized Resource Identifier)로 정확히 옮기는 작업을 무시하는 'Shortcut Learning' 현상이 발생한다는 점을 발견하였다. 이를 해결하기 위해 Closed World Assumption(CWA) 하에 모든 개체에 대해 $f(\text{entity}) = \text{entity}$ 형태의 정체성 학습을 먼저 수행함으로써, 모델이 개체 정보를 손실 없이 전달하는 능력을 갖추게 한 뒤 최종 작업을 학습시키는 전략을 제안한다.

## 📎 Related Works

본 연구는 지식 그래프 질의응답(KGQA) 분야의 기존 연구들을 바탕으로 한다. 특히 학술 지식 분야의 데이터셋인 DBLP-QuAD 및 SciQA와 ISWC 2023에서 개최된 Scholarly QALD 챌린지를 주요 배경으로 한다.

기존의 접근 방식들은 주로 NLP 도구를 사용하여 자연어를 텍스트 형태로 변환하거나 LLM을 활용하는 방식을 취했다. 특히 Rajpal & Usbeck의 연구가 언급되었으며, 본 논문은 이들의 접근 방식을 복제하고 개선하는 것에서 시작한다. 기존 방식의 한계점은 개체 연결 이후의 Multi-hop 쿼리 생성 능력이 부족하다는 점이며, 본 논문은 이를 해결하기 위해 모델의 학습 구조와 사전 학습 단계에 집중하여 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 파이프라인
시스템은 크게 두 단계로 구성된다: **Entity Linking $\rightarrow$ SPARQL Generation**.

1.  **Entity Linking**: 자연어 질문 내의 텍스트 개체를 해당 KG의 IRI로 치환한다. 예를 들어, "Wei Li"라는 텍스트를 `<https://dblp.org/pid/64/6025-131>`로 변경한다. 이는 모델이 배워야 할 어휘 사전(Vocabulary)의 크기를 46,000개에서 10,399개로 크게 줄여 학습 효율을 높인다.
2.  **SPARQL Generation**: IRI가 포함된 질문을 입력으로 받아 최종 SPARQL 쿼리를 생성한다.

### 2. 모델 아키텍처
본 연구에서는 Andrej Karpathy의 GPT 구현체를 기반으로 하되, 입력 질문을 처리하기 위한 Encoder를 추가하여 **Encoder-Decoder Transformer** 구조를 구축하였다. 모델의 규모는 약 3.47M 파라미터로 매우 가볍게 설계되었다.

### 3. 사전 학습 및 학습 절차
모델은 두 단계의 학습 과정을 거친다.

-   **Step 1: Entity Pre-training**: 모든 개체들에 대해 정체성 함수를 학습시킨다. 단순히 하나의 개체만 학습시키는 것이 아니라, 여러 개체가 나열된 시퀀스를 그대로 출력하도록 학습시켜 정체성 보존 능력을 극대화한다.
    $$f(\text{entity}_1, \text{entity}_2, \dots) = \text{entity}_1, \text{entity}_2, \dots$$
-   **Step 2: KGQA Fine-tuning**: 사전 학습된 모델을 사용하여 자연어 질문(IRI 포함)을 SPARQL 쿼리로 변환하는 작업을 학습한다.

### 4. 학습 설정 및 손실 함수
-   **손실 함수**: Cross Entropy Loss를 사용하여 토큰 예측 오류를 최소화한다.
-   **하이퍼파라미터**: Learning rate 0.0007, Dropout 1%, 내부 벡터 차원 128, Attention Head 8개, Encoder/Decoder 각각 4개 레이어를 사용한다.
-   **학습 주기**: 사전 학습에 14,400 epoch, 본 작업 학습에 4,800 epoch를 할당하였다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: DBLP-QuAD (총 10,000개 항목 중 Entity Linking에 성공한 9,289개 사용)
-   **데이터 분할**: Train 93.1%, Validation 5.9%, Test 2%
-   **평가 지표**: 
    -   $\text{Acc@1}, \text{Acc@3}$: 상위 1개 또는 3개 생성 결과 중 정답 쿼리와 완전히 일치하는 비율
    -   $\text{aHD (Average Hamming Distance)}$: 정답 쿼리와 생성 쿼리 간의 평균 토큰 거리
    -   $\text{Precision, Recall, F1}$: Scholarly QALD 챌린지 제출 결과

### 2. 정량적 결과
사전 학습을 적용한 모델이 모든 지표에서 성능 향상을 보였다.

| 모델 | Acc@1 (%) | Acc@3 (%) | aHD (tokens) | Precision (%) | Recall (%) | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Not pre-trained | 31.89 | 43.78 | 41.72 | 40.50 | 20.60 | 0.005 |
| **Pre-trained** | **49.18** | **62.70** | **2.01** | **72.51** | **29.10** | **0.009** |

### 3. 결과 분석
사전 학습 모델의 $\text{Acc@1}$이 약 17.3%p 상승한 것은 주목할 만한 성과이다. 흥미로운 점은 사전 학습 모델의 aHD가 급격히 낮아졌다는 것인데, 이는 모델이 단순히 토큰 오류를 줄이는 '안전한 예측'을 하는 대신, 정답 쿼리를 정확히 맞추려는 경향을 갖게 되었음을 시사한다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과
본 연구는 매우 작은 모델 사이즈(3.47M)와 제한된 데이터셋을 사용했음에도 불구하고, 사전 학습 전략을 통해 상용 LLM들과 경쟁 가능한 수준의 쿼리 템플릿 생성 능력을 보여주었다. 특히 Shortcut Learning 문제를 정체성 함수 사전 학습으로 해결한 점이 핵심적인 기여이다.

### 2. 한계 및 분석
-   **Zero-shot 문제**: 테스트 데이터 중 학습 과정에서 보지 못한 개체(Unseen Entity)가 등장할 경우, 모델은 쿼리 템플릿은 정확히 생성하지만 해당 개체 IRI를 정확히 출력하지 못하는 현상이 발견되었다.
-   **Identity Hallucination**: 저자들은 이를 'Comprehension-topic hallucinations'라고 정의하며, Zero-shot 상황에서의 정체성 함수 구현 능력이 이 작업의 가장 큰 난제임을 밝혔다.
-   **의존성**: 시스템 전체 성능이 전 단계인 Entity Linking의 정확도에 크게 의존한다.

### 3. 비판적 해석
F1 스코어가 0.009로 매우 낮게 나타난 점은, 쿼리 생성의 정확도가 올라갔음에도 불구하고 실제 최종 답변(Triples) 추출 단계까지 이어지는 과정에서 누수가 많음을 의미한다. 이는 단순히 쿼리 텍스트를 맞추는 것과 실제 KG에서 유효한 답을 얻는 것 사이의 간극이 크다는 것을 보여준다.

## 📌 TL;DR

본 논문은 자연어 질문을 SPARQL 쿼리로 변환할 때 발생하는 **개체 정보 손실(Shortcut Learning)** 문제를 해결하기 위해, **개체 정체성 함수($f(x)=x$)를 사전 학습**하는 전략을 제안한다. 이를 통해 매우 가벼운 GPT 기반 모델로도 $\text{Acc@3}$ 기준 62.7%의 정확도를 달성하였다. 이 연구는 데이터가 적은 특수 도메인의 지식 그래프를 구축하는 배포자가 적은 비용으로 전용 쿼리 생성 모델을 구축할 수 있는 실용적인 경로를 제시한다.