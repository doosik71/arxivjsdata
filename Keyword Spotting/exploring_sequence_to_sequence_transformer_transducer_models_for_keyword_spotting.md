# EXPLORING SEQUENCE-TO-SEQUENCE TRANSFORMER-TRANSDUCER MODELS FOR KEYWORD SPOTTING

Beltrán Labrador, Guanlong Zhao, Ignacio López Moreno, Angelo Scorza Scarpati, Liam Fowl, Quan Wang (2022)

## 🧩 Problem to Solve

본 논문은 특정 단어나 구절을 검출하는 Keyword Spotting (KWS) 작업에 sequence-to-sequence (seq2seq) 기반의 Transformer-Transducer (T-T) 모델을 효율적으로 적용하는 방법을 다룬다.

일반적으로 Automatic Speech Recognition (ASR) 시스템을 KWS에 그대로 활용할 경우, 몇 가지 심각한 문제가 발생한다. 첫째, "Okay Google"을 "Okay GOOGL"로 인식하는 것과 같은 미세한 ASR 오류가 발생하면 KWS 관점에서는 검출 실패로 처리되어 정확도가 낮아진다. 둘째, KWS 대상이 되는 키워드는 일반적인 대화 데이터에 비해 출현 빈도가 매우 낮아 데이터 희소성(data sparsity) 문제가 발생하며, 이는 결국 낮은 검출 성능으로 이어진다. 따라서 본 연구의 목표는 T-T ASR 시스템을 KWS 작업에 적합하게 변형하여, ASR 기반 KWS의 한계를 극복하고 높은 검출 정확도를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 키워드를 개별 문자열로 인식하는 대신, 하나의 응집된 음향 이벤트(coherent acoustic event)로 취급하는 것이다. 이를 위해 다음과 같은 세 가지 주요 기여를 제시한다.

1. **특수 토큰 `<kw>` 도입**: 학습 데이터의 텍스트 전사(transcription)에서 키워드 전체를 특수 토큰인 `<kw>`로 대체하여, 모델이 키워드 전체를 하나의 토큰으로 예측하도록 강제한다.
2. **MBR(Minimum Bayes-Risk) 기반 학습**: KWS 오류율을 직접적으로 최소화하기 위해, sequence-discriminative MBR 학습 기법을 KWS 작업에 맞게 변형한 새로운 손실 함수를 도입한다.
3. **Confidence Score 기반 결정**: ASR의 디코딩 결과물을 파싱하는 대신, Joint Network의 소프트맥스 출력값에서 `<kw>` 토큰의 확률값을 KWS 점수로 사용하여 다양한 동작 지점(operation points)에서 유연하게 임계값을 조절할 수 있도록 한다.

## 📎 Related Works

기존의 KWS 접근 방식은 크게 세 가지로 나눌 수 있다.

- **전통적 방식**: Hidden Markov Model (HMM)과 Deep Neural Networks (DNN)를 결합하여 각 프레임을 키워드 혹은 filler 오디오로 분류하는 방식이다. 이후 CNN이나 RNN을 도입하여 시간 및 주파수 관계, 그리고 긴 시간적 의존성을 캡처하려는 시도가 있었다.
- **End-to-End (E2E) 방식**: 직접적으로 KWS 점수를 생성하는 컴팩트한 신경망 구조를 사용하며, 최근에는 SVDF(Singular Value Decomposition Filters) 등을 사용하여 리소스 효율성을 높인 모델들이 제안되었다.
- **seq2seq 기반 방식**: LSTM-CTC나 Attention 모델을 사용하여 키워드를 검출하려는 시도가 있었으나, 많은 경우 N-best 결과물에 의존하거나 계산 비용이 높은 추론 과정이 필요했다는 한계가 있다.

본 논문의 제안 방식은 seq2seq 모델의 긴 의존성 캡처 능력을 활용하면서도, 강제 정렬(forced-alignment) 없이 학습이 가능하다는 점, 그리고 `<kw>` 토큰의 직접적인 최적화를 통해 추론 효율성과 정확도를 동시에 높였다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 Transformer-Transducer (T-T) 구조를 기반으로 한다.

- **Audio Encoder**: Transformer 블록을 사용하여 입력 오디오 특징을 음향 임베딩 벡터로 변환한다.
- **Label Encoder**: LSTM 레이어를 사용하여 텍스트 토큰을 언어 임베딩 벡터로 변환한다.
- **Joint Network**: 두 임베딩 벡터를 입력받아 사전 정의된 토큰 집합(graphemes)에 대한 확률 분포를 출력하는 완전 연결 계층(fully-connected layers)으로 구성된다.

### TT-KWS 학습 전략

학습 시, 모든 키워드 출현 부분을 특수 토큰 `<kw>`로 대체한다. 예를 들어, "Okay Google"이라는 문구는 단일 토큰 `<kw>`로 취급된다. 이를 통해 모델은 키워드의 세부 철자 오류에 영향을 받지 않고, 해당 구간이 키워드라는 사실 자체에 집중하게 된다.

### MBR (Minimum Bayes-Risk) Training

데이터 희소성 문제를 해결하고 KWS 정확도를 직접 최적화하기 위해 MBR 학습을 도입한다. 빔 서치(beam search)를 통해 생성된 $N$-best 가설들을 바탕으로 False Negative (FN)와 False Positive (FP) 비율을 계산하여 손실 함수에 반영한다.

학습 샘플 $i$의 $j$번째 가설을 $H_{ij}$, 정답 전사를 $R_i$라고 할 때, 각각에 포함된 `<kw>` 토큰의 개수를 $K_{H_{ij}}$와 $K_{R_i}$라고 정의한다. 이때 FP와 FN은 다음과 같이 계산된다.
$$FP_{ij} = \max(0, K_{H_{ij}} - K_{R_i})$$
$$FN_{ij} = \max(0, K_{R_i} - K_{H_{ij}})$$

샘플당 손실 함수 $L_{ij}$는 다음과 같다.
$$L_{ij} = P_{ij} \cdot \frac{\alpha FP_{ij} + \beta FN_{ij}}{K_{R_i} + \epsilon}$$
여기서 $P_{ij}$는 가설의 확률이며, $\alpha, \beta$는 각 성분의 가중치, $\epsilon$은 수치적 안정성을 위한 작은 상수이다.

최종 배치 학습 손실 $L$은 RNN-T 손실과 MBR 손실의 결합으로 정의된다.
$$L = \sum_{i} \sum_{j} L_{ij} - \lambda \log P(Y|X)$$
여기서 $\lambda$는 RNN-T 손실의 강도를 조절하는 정규화 항이다.

### 추론 및 Scoring 방법

추론 시에는 Joint Network의 소프트맥스 출력값 중 `<kw>` 토큰에 해당하는 확률값을 KWS score로 사용한다. 전체 발화 구간에서 출력된 점수 중 최댓값을 해당 발화의 최종 점수로 채택하여, 이를 임계값과 비교해 키워드 검출 여부를 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 다양한 영어 억양, 소음 및 잔향이 포함된 구글 내부 데이터셋을 사용하였다. (양성 데이터 4,300시간, 음성 데이터 4,000시간)
- **비교 대상 (Baselines)**:
    1. **E2E KWS**: SVDF 기반의 최신 컴팩트 모델.
    2. **ASR based KWS**: 일반적인 RNN-T 손실로 학습하고, 추론 시 Bigram edit distance scoring을 통해 점수를 산출하는 방식.
- **평가 지표**: Equal Error Rate (EER), 그리고 실제 서비스 환경에서 중요한 FP 1% 및 0.5% 지점에서의 FN rate를 측정하고 DET curve를 통해 분석하였다.

### 주요 결과

1. **TT-KWS의 유효성**: 일반 ASR baseline 대비 `<kw>` 토큰을 사용한 TT-KWS가 훨씬 우수한 성능을 보였다. 특히 Small 모델에서 EER이 $8.24\% \rightarrow 5.68\%$로 약 $31\%$ 상대적으로 개선되었다.
2. **MBR 손실의 효과**: MBR 학습을 추가했을 때, 특히 낮은 FP 영역에서 FN rate가 크게 감소하였다. Small 모델 기준 1% FP 지점에서 FN rate가 $25.5\%$ 상대적으로 개선되었다.
3. **모델 크기별 성능**:
    - **Small 모델**: 리소스 제약이 심한 경우, SVDF 기반의 E2E KWS baseline이 여전히 가장 좋은 성능을 보였다.
    - **Large 모델**: TT-KWS + MBR 모델이 모든 지표에서 가장 우수한 성능을 기록하며, 전통적인 E2E KWS 모델과 유사하거나 더 나은 성능을 달성하였다.
4. **시스템 퓨전 (Fusion)**: SVDF 기반 Baseline KWS large 모델과 TT-KWS + MBR large 모델의 점수를 합산하여 퓨전했을 때, EER $3.09\%$라는 최적의 성능을 얻었으며, 이는 두 모델이 서로 보완적인 관계임을 시사한다.

## 🧠 Insights & Discussion

본 연구는 T-T 모델을 KWS에 적용하기 위해 `<kw>` 토큰화와 MBR 학습이라는 효과적인 전략을 제시하였다. 특히, ASR의 세부 인식 오류를 무시하고 키워드를 하나의 이벤트로 처리함으로써 ASR 기반 시스템의 고질적인 문제를 해결하였다.

**강점 및 시사점:**

- **유연성**: 새로운 키워드를 추가할 때 강제 정렬 과정 없이 전사 데이터 수정만으로 빠르게 적응할 수 있으며, 이는 연합 학습(Federated Learning)이나 일시적 학습(Ephemeral training)에 유리하다.
- **상보적 특성**: T-T 기반 모델과 전통적인 E2E KWS 모델이 서로 다른 오류 패턴을 가지므로, 두 모델을 결합했을 때 성능 향상이 뚜렷하게 나타났다.

**한계 및 비판적 해석:**

- **리소스 효율성**: 논문에서도 언급되었듯이, T-T 기반 모델은 모델 크기가 크고 디코딩 비용이 높아 배터리 및 CPU 제약이 심한 모바일 기기에 직접 탑재하기에는 한계가 있다.
- **적용 범위**: 본 실험은 영어 데이터셋에 국한되어 있으며, 다른 언어나 극단적으로 짧은 키워드에서도 동일한 성능 향상이 있을지는 명시되지 않았다.

## 📌 TL;DR

본 논문은 Transformer-Transducer (T-T) ASR 모델을 Keyword Spotting 작업에 최적화하기 위해 키워드를 특수 토큰 `<kw>`로 대체하고, KWS 오류율을 직접 최소화하는 MBR 손실 함수를 도입하였다. 실험 결과, 제안 방식은 기존 ASR 기반 KWS보다 훨씬 뛰어난 성능을 보였으며, 특히 모델 크기가 클 때 전통적인 E2E KWS 모델을 능가하거나 대등한 성능을 냈다. 또한, 두 방식을 결합한 퓨전 시스템이 가장 높은 정확도를 보였다. 이 연구는 계산 리소스가 충분한 환경(스마트 스피커, 차량 등)에서 고정밀 KWS 시스템을 구축하는 데 중요한 방향성을 제시한다.
