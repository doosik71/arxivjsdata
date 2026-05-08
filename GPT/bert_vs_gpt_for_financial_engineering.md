# Comparison of BERT vs GPT: Although GPT receives all the publicity, is BERT the new XGBOOST?

Edward Sharkey, Philip Treleaven (연도 미표기)

## 🧩 Problem to Solve

본 논문은 원자재 시장, 특히 구리(Copper) 거래를 위한 뉴스 이벤트의 감성 분석(Sentiment Analysis) 작업에서 BERT와 GPT 모델의 성능을 벤치마킹하여 비교하는 것을 목표로 한다.

현대 금융 시장, 특히 원자재 시장은 매우 변동성이 크며, 기업들은 적절한 헤징(Hedging) 결정을 내리는 데 어려움을 겪고 있다. 특히 최근의 '블랙 스완(Black Swan)' 이벤트와 같은 예측 불가능한 상황에서 뉴스 헤드라인을 통해 시장의 감성을 정확히 파악하는 것은 매우 중요하다. 따라서 본 연구는 금융 도메인 특화 모델들이 원자재 가격에 영향을 미칠 수 있는 뉴스 감성을 얼마나 정확하게 판단하는지 분석하고, 모델의 예측력과 해석 가능성(Interpretability) 사이의 트레이드-오프를 조사하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **도메인 특화 모델 제안**: 구리 관련 뉴스 데이터에 특화된 BERT 기반 모델인 **CopBERT**와 GPT 기반 모델인 **CopGPT**를 구축하고 이를 벤치마킹하였다.
2. **예측력과 해석 가능성의 비교 분석**: 단순한 성능 지표(F1 Score) 비교를 넘어, GPT 모델의 한계인 환각(Hallucination) 현상과 불투명한 내부 구조를 지적하며, BERT 모델이 제공하는 기계론적 해석 가능성(Mechanistic Interpretability)의 가치를 강조하였다.
3. **BERT의 새로운 위상 정립**: BERT가 절대적인 예측력에서는 대규모 언어 모델(LLM)에 밀릴 수 있으나, 해석 가능성과 정확성의 조화가 필요한 금융 공학 작업에서는 과거의 XGBoost가 표 형식 데이터(Tabular Data)에서 가졌던 위상과 유사한 대안이 될 수 있음을 시사하였다.

## 📎 Related Works

논문은 Transformer 아키텍처를 기반으로 하는 다양한 언어 모델들을 소개하며, 특히 다음과 같은 기존 접근 방식들을 언급한다.

- **RNN 및 LSTM**: 순차적 데이터 처리 및 메모리 기능을 통해 상태를 유지하지만, Transformer의 등장 이후 어텐션(Attention) 메커니즘에 의해 대체되는 추세이다.
- **Transformer**: 재귀(Recurrency)를 제거하고 오직 어텐션 메커니즘만을 사용하여 문맥을 학습하는 구조이다.
- **BERT (Bidirectional Encoder Representations from Transformers)**: 양방향 인코더를 통해 텍스트의 좌우 문맥을 동시에 학습하며, 주로 언어 이해(Understanding) 작업에 강점을 가진다.
- **GPT (Generative Pre-trained Transformer)**: 디코더 전용 구조로, 왼쪽에서 오른쪽으로 이어지는 자기회귀(Autoregressive) 방식을 통해 텍스트 생성(Generation)에 최적화되어 있다.
- **FinBERT 및 BloombergGPT**: 일반 목적의 모델은 금융 분야의 특수한 언어를 처리하는 데 한계가 있으므로, 금융 도메인에 특화된 사전 학습 모델들이 제안되었다. 본 논문은 이러한 흐름을 이어받아 더 구체적인 원자재(구리) 특화 모델을 구축하였다.

## 🛠️ Methodology

### 전체 파이프라인

본 연구는 뉴스 헤드라인을 수집하고, 이를 인간이 직접 레이블링(Positive, Neutral, Negative)한 데이터를 사용하여 모델을 훈련 및 평가하는 파이프라인을 따른다.

### 주요 모델 구성

- **CopBERT**: 구리 관련 뉴스 아이템에 집중하여 튜닝된 BERT 모델이다. 양방향 표현을 학습하여 복잡한 문장의 감성을 파악하는 데 유리하다.
- **CopGPT**: 구리 뉴스 데이터로 파인튜닝(Fine-tuning)하거나 퓨샷 학습(Few-shot learning) 및 프롬프트 엔지니어링을 적용한 GPT-3.5 기반 모델이다.

### 핵심 메커니즘 및 방정식

모델의 기초가 되는 Transformer의 **Scaled Dot-Product Attention** 함수는 다음과 같이 정의된다.

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

여기서 $Q$는 Query, $K$는 Key, $V$는 Value 행렬을 의미하며, $d_k$는 Query와 Key의 차원 수이다. Softmax 함수를 통해 계산된 어텐션 스코어(Attention Scores)는 모델이 입력 문장의 어떤 부분에 집중하는지를 결정한다.

### 평가 지표

모델의 성능은 정밀도(Precision)와 재현율(Recall)의 조화 평균인 $F1\ Score$를 사용하여 측정한다.

$$F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

### 해석 가능성 분석 방법

논문은 **기계론적 해석 가능성(Mechanistic Interpretability)**을 위해 CopBERT의 어텐션 맵(Attention Map) 간의 코사인 유사도(Cosine Similarity)를 계산하여 그래프 네트워크 형태로 시각화하였다. 각 노드는 문장을 나타내며, 에지의 색상은 두 문장의 어텐션 패턴이 얼마나 유사한지를 나타낸다.

## 📊 Results

### 실험 설정

- **데이터 소스**: Reuters(0.32%), Argus(0.36%), Mining Journal(0.05%), FT(0.27%) 등 공개 웹사이트에서 수집하였다.
- **비교 대상**: Random Baseline, FinBERT, CopBERT, CopGPT (GPT-3.5 turbo), GPT-4, GPT-2.

### 정량적 결과 (F1 Score)

실험 결과, 모델별 F1 스코어는 다음과 같다.

| 모델 이름 | F1 Score (%) |
| :--- | :---: |
| Random Baseline | 0.23 |
| GPT-2 | 0.26 |
| FinBERT Sentiment | 0.37 |
| **CopBERT Sentiment** | **0.41** |
| **GPT-4 Sentiment** | **0.49** |
| **CopGPT (GPT-3.5 turbo)** | **0.56** |

- **분석**: 예측력(Predictive Power) 관점에서는 CopGPT와 GPT-4 같은 대규모 LLM이 BERT 기반 모델보다 우수한 성능을 보였다.
- **도메인 특화 효과**: 구리 전용으로 학습된 CopBERT(0.41)는 일반 금융 모델인 FinBERT(0.37)보다 높은 성능을 기록하였다.

### 정성적 결과 및 논의

- **환각 현상**: GPT 모델들은 높은 예측력을 보였으나, 감성 분석 과정에서 심각한 환각(Hallucinations) 증상을 보였다.
- **투명성**: GPT 모델의 내부 작동 방식은 불투명(Obfuscated)하여 해석이 어려운 반면, CopBERT는 로컬 환경에서 실행 가능하며 어텐션 맵 분석을 통해 모델이 결론에 도달한 과정을 기계론적으로 평가할 수 있다.

## 🧠 Insights & Discussion

본 논문은 단순히 성능 수치만으로 모델을 평가하는 것이 아니라, 금융 공학이라는 특수한 환경에서 **'신뢰성'**과 **'해석 가능성'**이 얼마나 중요한지를 강조한다.

1. **BERT의 강점**: BERT의 양방향 문맥 이해 능력은 문장 끝부분에서 감성이 급변하는 복잡한 문장을 처리하는 데 유리하다. 또한, 모델의 가중치에 직접 접근하여 어텐션 패턴을 분석할 수 있다는 점은 블랙박스 형태인 LLM에 비해 큰 강점이다.
2. **GPT의 한계**: GPT-4와 같은 최신 모델이 가장 높은 F1 스코어를 기록했음에도 불구하고, 금융 의사결정 시스템에 그대로 적용하기에는 환각 위험과 낮은 해석 가능성이 치명적인 약점으로 작용한다.
3. **비판적 해석**: 논문 본문 중 "CopBERT가 GPT-4보다 F1 스코어가 약 10% 높고, CopGPT보다 16% 높다"는 서술이 있으나, 정작 제시된 표(Table 2)에서는 CopGPT(0.56)와 GPT-4(0.49)가 CopBERT(0.41)보다 높은 수치를 보이고 있어 텍스트와 표 사이에 모순이 발견된다. 다만, 결론 부분에서 "더 큰 LLM들이 예측력 면에서 BERT 모델들을 능가한다"고 명시한 점으로 보아, 표의 수치가 정확하며 본문의 특정 서술은 오기일 가능성이 높다.

## 📌 TL;DR

본 연구는 원자재(구리) 시장 감성 분석을 위해 BERT와 GPT 기반 모델을 비교 분석하였다. **예측 성능은 GPT-3.5/4 기반 모델들이 월등히 높았으나, BERT 기반의 CopBERT는 해석 가능성이 높고 환각 위험이 적다는 명확한 장점을 보였다.** 결론적으로 BERT는 순수 예측력은 낮을지라도, 해석 가능성과 정확성의 균형이 필수적인 금융 공학 작업에서 매우 유용한 대안이 될 수 있으며, 이러한 점에서 "새로운 XGBoost"와 같은 역할을 할 잠재력이 있음을 시사한다.
