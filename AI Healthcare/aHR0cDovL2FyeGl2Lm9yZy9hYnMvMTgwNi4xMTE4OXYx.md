# A hybrid deep learning approach for medical relation extraction

Veera Raghavendra Chikka, Kamalakar Karlapalem (2018)

## 🧩 Problem to Solve

본 연구의 목적은 생의학 문헌 및 임상 기록 내에서 치료법(Treatment)과 의학적 문제(Medical Problem) 사이의 관계를 자동으로 추출하는 시스템을 구축하는 것이다. 치료와 문제 간의 관계를 파악하는 것은 의사 결정 지원 시스템(Decision Support System), 안전 감시(Safety Surveillance), 그리고 새로운 치료법 발견과 같은 응용 분야에서 매우 중요하다.

특히 본 논문은 i2b2 2010 relation extraction task에서 정의한 다음 다섯 가지 관계 유형을 분류하는 문제를 해결하고자 한다:
1. **TrAP (Treatment Administered for Problem)**: 치료법이 해당 문제의 해결을 위해 처방됨.
2. **TrIP (Treatment Improves Problem)**: 치료법이 해당 문제를 개선함.
3. **TrWP (Treatment Worsens Problem)**: 치료법이 해당 문제를 악화시킴.
4. **TrCP (Treatment Causes Problem)**: 치료법이 해당 문제를 유발함.
5. **TrNAP (Treatment Not Administered because of Problem)**: 해당 문제로 인해 치료법이 처방되지 않음.

관계 추출은 단순한 단어의 공출현(Co-occurrence)만으로는 해결할 수 없으며, 텍스트 내의 문맥과 구조적 정보를 정교하게 파악해야 하는 복잡한 과제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝 모델의 한계를 보완하기 위해 규칙 기반 시스템(Rule-based system)을 결합한 하이브리드 접근 방식을 제안한 것이다. 주요 설계 아이디어는 다음과 같다:

- **Bi-directional LSTM (Bi-LSTM) 도입**: 관계 인자(Relation Arguments) 간의 순차적 의존성을 효과적으로 캡처하기 위해 CNN 대신 Bi-LSTM 구조를 사용하였다.
- **다각적 표현 학습**: 단어 수준(Word-level)의 특징뿐만 아니라 문장 수준(Sentence-level)의 특징을 함께 사용하여 모델의 입력값으로 활용하였다.
- **하이브리드 전략**: 데이터 양이 충분한 클래스는 딥러닝 모델을 통해 학습하고, 샘플 수가 적어 학습이 어려운 클래스는 고정밀 규칙 기반 시스템을 통해 보완함으로써 전체적인 성능을 향상시켰다.

## 📎 Related Works

기존의 i2b2 2010 챌린지에 참여한 대부분의 시스템은 Support Vector Machines (SVM)와 같은 전통적인 머신러닝 기법을 사용하였다. 특히 SVM과 규칙 기반 방식을 결합한 하이브리드 접근법이 높은 성능을 보였다는 보고가 있었다.

기존 딥러닝 기반의 관계 추출 연구들은 주로 단어 수준의 특징이나 좁은 윈도우 크기(three word window proximity) 내의 근접성만을 고려하는 경향이 있었다. 반면, 본 연구는 문장 전체를 단어 벡터와 위치 인덱스의 행렬로 처리하여 더 넓은 문맥적 정보를 활용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조
시스템은 입력 문장에서 치료법과 문제라는 두 엔티티 쌍을 식별한 후, 이들의 관계를 분류하는 구조로 이루어져 있다. 전체 흐름은 **특징 추출 $\rightarrow$ Bi-LSTM 처리 $\rightarrow$ 문장 수준 특징 결합 $\rightarrow$ 최종 분류** 단계로 진행된다.

### 2. 주요 구성 요소 및 역할
- **Word-level Features**:
    - 단어 토큰, POS(Part-of-Speech) 태그, Chunk phrase 태그를 사용한다.
    - **Position Vector**: 각 단어가 치료법과 문제 엔티티로부터 얼마나 떨어져 있는지를 나타내는 상대적 거리 벡터를 추가하여 엔티티의 위치 정보를 제공한다.
- **Sentence-level Features**:
    - **POS tag sequence**: 엔티티 사이에 나타나는 POS 태그 시퀀스 중 빈도가 높은 상위 100개를 추출하여 벡터화한다.
    - **PMI (Point-wise Mutual Information)**: 두 엔티티 간의 통계적 공출현 빈도를 측정한다.
    - **Assertion Words**: Apache cTAKES의 사전(allergy, cause, fail 등)을 사용하여 문장의 확언(Assertion) 상태를 나타내는 키워드의 인덱스를 특징으로 사용한다.
- **Bi-LSTM 및 Merge Layer**:
    - Bi-LSTM은 문장을 양방향으로 처리하여 현재 단어의 앞뒤 문맥 의존성을 학습한다.
    - Bi-LSTM의 출력값과 앞서 추출한 문장 수준 특징들을 **Merge Layer**에서 결합(Concatenate)한다.
    - 최종적으로 Fully Connected Linear Layer를 통해 관계 클래스 수($ntags$)만큼의 출력값을 생성한다.

### 3. 규칙 기반 접근법 (Rule-based Approach)
딥러닝 모델이 학습하기 어려운 소수 클래스를 위해 다음 두 가지 규칙을 적용한다:
- **문장/구 패턴(Pattern of sentence/phrase)**: 전문가가 도메인 지식을 바탕으로 직접 설계한 패턴을 사용한다. (예: `<problem> is diagnosed with <treatment>` $\rightarrow$ TrAP)
- **최단 의존 경로(Shortest Dependency Path, SDP)**: 두 엔티티 사이의 의존 구문 분석 경로 상에 존재하는 동사(Verb)를 확인한다. 예를 들어, 경로 내에 'treated'라는 동사가 있다면 TrAP 관계로 판단하며, 'control', 'regulate' 등이 있다면 TrCP 관계로 판단한다.

### 4. 평가 지표
모델의 성능은 Precision($P$), Recall($R$), F-score($F$)를 통해 측정하며, 수식은 다음과 같다:
$$Precision(P) = \frac{TP}{TP + FP}$$
$$Recall(R) = \frac{TP}{TP + FN}$$
$$F\text{-score}(F) = \frac{2 \times Recall \times Precision}{Recall + Precision}$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: i2b2 2010 챌린지 데이터셋의 서브셋을 사용하였다. (훈련 데이터 384개 문서, 테스트 데이터 42개 문서)
- **구현**: Keras를 사용하여 Bi-LSTM을 구현하였으며, Word Embedding은 GloVe를 사용하였다.
- **하이퍼파라미터**: `LSTM_output_size = 64`, `epochs = 20`.

### 2. 주요 정량적 결과
- **Negative Samples의 영향**: 관계가 없는 샘플(null relationships)의 수를 5,000에서 30,000까지 변화시켰을 때, 20,000개일 때 가장 높은 F-score를 기록하였다.
- **Embedding Size의 영향**: 40, 100, 200 차원 중 40차원일 때 Total F-score 0.50으로 가장 우수한 성능을 보였다.
- **모델 간 비교**:
    - **Bi-LSTM**은 샘플 수가 많은 **TrAP** 관계 식별에서 가장 우수한 성능을 보였다.
    - **SVM**은 상대적으로 적은 샘플로도 TrAP, TrCP, TrIP에서 일관된 성능(F-score $\approx 0.45$)을 보였다.
    - **Rule-based** 방식은 Precision은 매우 높으나 Recall이 매우 낮아 단독으로는 한계가 있었다.

### 3. 하이브리드 시스템 성능
Bi-LSTM 결과에 고정밀 규칙 기반 결과를 결합한 하이브리드 시스템의 결과는 다음과 같다 (Table 4 기준):
- **Total F-score**: 0.52
- 특히 **TrIP**와 **TrNAP** 관계에서 성능 향상이 뚜렷하게 나타났다. TrCP와 TrWP는 이미 Bi-LSTM이 잘 식별하고 있어 변화가 거의 없었다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- 본 연구는 딥러닝의 '데이터 갈증(Data-hungry)' 특성을 정확히 짚어내어, 데이터가 부족한 클래스에 대해서는 규칙 기반 시스템으로 보완하는 현실적인 하이브리드 전략을 취하였다.
- 단어 수준의 임베딩뿐만 아니라 문장 수준의 통계적/언어적 특징(PMI, POS 시퀀스 등)을 결합함으로써 단순한 순차 모델보다 풍부한 문맥 정보를 활용할 수 있었다.

### 2. 한계 및 비판적 해석
- **데이터셋의 모호성**: 저자들은 골드 표준(Gold standard) 데이터셋 자체에 모호한 어노테이션이 존재함을 언급하였다. 예를 들어, 동일한 패턴(`treatment treated for problem`)이 TrAP와 TrIP로 서로 다르게 태깅된 경우가 발견되었다. 이는 모델의 성능 저하가 모델 자체의 문제보다 데이터의 노이즈에서 기인했을 가능성을 시사한다.
- **규칙 설계의 어려움**: 규칙 기반 시스템은 Precision은 높지만, 언어의 다양성으로 인해 Recall을 높이는 데 한계가 있으며, 규칙을 수동으로 프레이밍하는 과정에 많은 비용이 소모된다.
- **실험 규모**: 사용된 데이터셋이 i2b2 2010의 서브셋으로 규모가 작아, 제안된 하이브리드 방식의 일반화 성능을 완전히 검증하기에는 부족함이 있다.

## 📌 TL;DR

본 논문은 의료 텍스트에서 치료법과 질환 간의 관계를 추출하기 위해 **Bi-LSTM 딥러닝 모델과 규칙 기반 시스템을 결합한 하이브리드 접근법**을 제안하였다. 단어 수준의 특징과 문장 수준의 특징을 통합하여 학습하였으며, 데이터가 부족한 특정 관계 클래스에 대해 고정밀 규칙을 적용함으로써 전체적인 F-score를 향상시켰다. 이 연구는 데이터 희소성 문제가 심한 의료 도메인에서 딥러닝과 도메인 지식(규칙)을 어떻게 상호 보완적으로 사용할 수 있는지 보여주었으며, 향후 다양한 생의학 관계 추출 과제로 확장될 가능성이 크다.