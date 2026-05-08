# Innovative Sentiment Analysis and Prediction of Stock Price Using FinBERT, GPT-4 and Logistic Regression: A Data-Driven Approach

Olamilekan Shobayo, Sidikat Adeyemi-Longe, Olusogo Popoola and Bayode Ogunleye (2024)

## 🧩 Problem to Solve

본 연구는 금융 시장의 복잡성과 변동성으로 인해 발생하는 주가 예측의 어려움을 해결하고자 한다. 특히 전통적인 통계적 기법들은 뉴스나 시장 정서(sentiment)와 같은 외부 변수가 주가에 미치는 복잡한 패턴을 포착하는 데 한계가 있다.

따라서 본 논문의 목표는 금융 뉴스 텍스트에서 추출한 정서 분석 결과가 나이지리아 증권거래소(NGX)의 All-Share Index 주가 흐름을 예측하는 데 얼마나 효과적인지를 분석하는 것이다. 이를 위해 도메인 특화 모델인 FinBERT, 범용 대규모 언어 모델인 GPT-4, 그리고 전통적인 머신러닝 모델인 Logistic Regression의 성능을 비교 평가하여, 어떤 접근 방식이 가장 효율적으로 시장 트렌드를 예측할 수 있는지 규명하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 최첨단 NLP(Natural Language Processing) 모델과 전통적인 머신러닝 모델 간의 비교 분석을 통해, 특정 금융 데이터셋에서는 단순한 모델이 더 높은 성능을 보일 수 있음을 실증적으로 입증한 것이다.

특히, 복잡한 Transformer 기반 모델들이 반드시 높은 예측력을 보장하는 것이 아니라, 적절한 Feature Engineering(예: TF-IDF)과 하이퍼파라미터 최적화가 이루어진 단순 모델이 더 강건한(robust) 결과를 낼 수 있다는 점을 제시하였다. 이는 '오캄의 면도날(Occam's Razor)' 원칙이 금융 예측 모델링에도 적용될 수 있음을 시사한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하며 기존 접근 방식의 한계를 지적한다.

- **FinBERT 관련 연구**: Liu 등은 FinBERT가 금융 용어 이해와 정서 분류에서 기존 모델보다 우수함을 보였으나, 다른 모델과의 비교 평가가 부족했다.
- **GPT 계열 연구**: Leippold는 GPT-3가 적대적 공격(adversarial attacks)에 취약하며 해석 가능성이 부족함을 지적하였다.
- **하이브리드 모델 연구**: Yang 등은 LASSO, LSTM, FinBERT를 결합하여 높은 정확도를 얻었으나, 비선형 관계를 완전히 포착하지 못하는 Feature Extraction의 한계가 있었다.
- **시계열 예측 연구**: Sidogi 등은 FinBERT와 LSTM을 사용했으나, 모델 자체의 성능 평가보다는 RMSE, MAE와 같은 예측 오차 지표에만 집중했다는 한계가 있다.

본 연구는 이러한 한계들을 극복하기 위해 다양한 모델을 동일한 데이터셋에서 비교하고, 단순 정확도뿐만 아니라 ROC AUC, F1 Score 등 다각적인 지표로 성능을 검증하며, Time Series Cross-Validation을 통해 데이터 누수(data leakage) 문제를 방지하였다.

## 🛠️ Methodology

### 전체 파이프라인

연구의 전체 흐름은 `데이터 수집 $\rightarrow$ 전처리 $\rightarrow$ 특성 추출 $\rightarrow$ 모델 학습 및 최적화 $\rightarrow$ 평가` 순으로 진행된다. Nairametric과 Proshare 웹사이트에서 2010년부터 2024년까지의 뉴스 헤드라인 24,923건을 수집하였으며, 이를 3,573개의 시간적 관측치로 집계하였다.

### 데이터 전처리 및 특성 추출

NLTK 라이브러리를 사용하여 불용어 제거, 소문자 변환, 토큰화, 정규화(Stemming 및 Lemmatization)를 수행하였다. 모델별 특성 추출 방식은 다음과 같다.

- **FinBERT**: BERT Embedding을 사용하여 금융 용어의 문맥적 의미를 캡처한다.
- **GPT-4**: 별도의 특성 추출 없이 사전 학습된 아키텍처를 활용하며, API를 통해 직접 텍스트를 입력한다.
- **Logistic Regression**: TF-IDF(Term Frequency-Inverse Document Frequency) 벡터화를 통해 텍스트를 수치형 특성으로 변환한다.

### 학습 및 추론 절차

- **데이터 분할**: 시계열 순서를 유지하며 훈련 세트 70%, 검증 세트 15%, 테스트 세트 15%로 분할하였다.
- **검증 방법**: 데이터 누수를 방지하기 위해 일반적인 k-fold 대신 Time Series Cross-Validation (TSCV, $n=5$)을 적용하였다.
- **하이퍼파라미터 최적화**: Optuna를 사용하여 각 모델의 최적 파라미터를 탐색하였다.

### 모델별 상세 설명

1. **Logistic Regression (LR)**: 이진 분류를 수행하며, 가중치 합 $z$를 시그모이드 함수에 통과시켜 0과 1 사이의 확률값을 생성한다.
   $$\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$
   L2 정규화(penalty='l2')와 'liblinear' 솔버를 사용하여 과적합을 방지하고 계산 효율성을 높였다.
2. **FinBERT**: FinBERT-base 모델을 사용하였으며, PyTorch 텐서 기반으로 학습되었다. GPU 메모리 효율을 위해 Automatic Mixed Precision (AMP) 학습을 적용하였고, Patience가 5인 Early Stopping을 통해 과적합을 제어하였다.
3. **GPT-4**: API를 이용한 'Predefined approach'를 사용하였다. 모델이 내부적으로 문맥과 의미를 분석하여 정서 점수를 생성하고 클래스를 분류하도록 지시하는 방식이다.

## 📊 Results

### 실험 설정

- **데이터셋**: NGX All-Share Index 뉴스 데이터
- **평가 지표**: Accuracy, Precision, Recall, F1 Score, ROC AUC
- **기준선**: FinBERT, GPT-4, Logistic Regression

### 정량적 결과

실험 결과, 전통적인 머신러닝 모델인 Logistic Regression이 모든 지표에서 가장 우수한 성능을 기록하였다.

| Metric | GPT-4 (Predefined) | FinBERT | Logistic Regression |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 54.19% | 63.33% | **81.83%** |
| **Precision** | 72.66% | 63.76% | **82.57%** |
| **Recall** | 32.69% | 63.33% | **81.15%** |
| **F1 Score** | 45.09% | 63.30% | **81.85%** |
| **ROC AUC** | 65.37% | 65.59% | **89.76%** |

### 결과 분석

- **Logistic Regression**: 훈련 정확도(80.93%)와 테스트 정확도(81.83%)가 매우 유사하여 일반화 성능이 뛰어나며, ROC AUC가 89.76%에 달해 클래스 구분 능력이 매우 탁월함을 보였다.
- **FinBERT**: 63.33%의 정확도로 보통 수준의 성능을 보였으나, 계산 자원 소모가 매우 컸다(학습에 약 90분 소요).
- **GPT-4**: 정밀도(Precision)는 높았으나 재현율(Recall)이 32.69%로 매우 낮아, 긍정적인 신호를 많이 놓치는 경향이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 최신 딥러닝 모델이 항상 정답은 아니라는 점을 보여준다. Logistic Regression의 압도적인 성능은 다음과 같은 이유로 해석된다.

1. **선형 분리 가능성**: 사용된 금융 뉴스 데이터셋의 정서 신호가 비교적 명확하여, 복잡한 비선형 모델보다 단순한 선형 모델이 더 효율적으로 작동했을 가능성이 크다.
2. **효과적인 특성 공학**: TF-IDF가 금융 뉴스의 핵심 단어들을 잘 포착하여 모델에 적절한 입력을 제공하였다.
3. **과적합 방지**: 모델의 단순함 자체가 일종의 정규화 역할을 하여, 데이터가 제한적인 상황에서 딥러닝 모델보다 더 나은 일반화 성능을 보였다.

### 한계 및 비판적 해석

- **데이터의 특성**: 본 실험 결과는 특정 데이터셋(NGX 뉴스)에 국한된 것일 수 있다. 훨씬 더 방대하고 복잡한 비정형 데이터에서는 FinBERT나 GPT-4가 더 유리할 수 있으나, 본 연구의 데이터 규모와 특성에서는 단순 모델이 우세했다.
- **GPT-4 활용 방식**: GPT-4를 단순한 '사전 정의된 접근 방식(Predefined approach)'으로 사용한 점은 아쉽다. 적절한 Few-shot prompting이나 Fine-tuning이 이루어졌다면 결과가 달랐을 가능성이 있다.
- **자원 효율성**: FinBERT는 성능 대비 계산 비용(A100 GPU 필요)이 너무 높아 실시간 예측 시스템으로의 적용에는 한계가 명확하다.

## 📌 TL;DR

본 논문은 나이지리아 주식 시장 예측을 위해 FinBERT, GPT-4, Logistic Regression의 성능을 비교 분석하였다. 실험 결과, **Logistic Regression이 정확도 81.83%, ROC AUC 89.76%로 가장 우수한 성능**을 보였으며, 이는 복잡한 LLM보다 잘 튜닝된 단순 모델이 특정 금융 태스크에서 더 효율적일 수 있음을 시사한다. 향후 연구에서는 이러한 단순 모델의 강건함과 LLM의 심층적 문맥 이해 능력을 결합한 **하이브리드 모델** 개발이 중요할 것으로 보인다.
