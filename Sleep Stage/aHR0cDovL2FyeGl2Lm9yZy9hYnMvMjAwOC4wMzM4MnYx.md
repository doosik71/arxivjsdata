# CLASSIFYING SLEEP-WAKE STAGES THROUGH RECURRENT NEURAL NETWORKS USING PULSE OXIMETRY SIGNALS

Ramiro Casal, Leandro E. Di Persia, Gastón Schlotthauer (2020)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 수면 단계 분류(Sleep Staging)를 위한 저비용 및 고효율의 스크리닝 방법을 개발하는 것이다. 현재 수면 장애 진단의 표준(Gold Standard)은 수면다원검사(Polysomnography, PSG)이다. 하지만 PSG는 전문 기술자의 감독이 필요하고, 분석 과정에서 수동 점수 매기기(Manual Scoring)라는 매우 번거로운 작업이 수반되며, 비용이 많이 들고 접근성이 낮다는 단점이 있다. 또한, 환자들이 검사 센터라는 낯선 환경에서 잠드는 데 어려움을 겪어 검사를 반복해야 하는 경우도 빈번하다.

특히 폐쇄성 수면 무호흡-저호흡 증후군(Obstructive Sleep Apnea/Hypopnea Syndrome, OSAHS)의 심각도를 평가하는 핵심 지표인 무호흡-저호흡 지수(Apnea/Hypopnea Index, AHI)를 산출할 때, 많은 기존 스크리닝 방식들이 총 수면 시간(Total Sleep Time, TST)을 정확히 측정하지 않고 단순히 전체 기록 시간(Total Recording Time, TRT)으로 대체하는 경향이 있다. 이는 TST의 과대평가로 이어져 결과적으로 AHI를 과소평가하게 만들며, 결국 질환의 과소 진단(Underdiagnosis)이라는 심각한 문제를 야기한다. 따라서 본 논문의 목표는 접근성이 좋은 Pulse Oximeter 신호만을 이용하여 환자가 깨어 있는 상태(Awake)인지 수면 상태(Asleep)인지를 정확하게 분류함으로써 TST 추정의 정확도를 높이는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 설계 아이디어는 수면 단계에 따라 자율신경계의 조절 능력이 변화하며, 이것이 심박수(Heart Rate, HR)와 말초 산소 포화도(Peripheral Oxygen Saturation, $\text{SpO}_2$)의 변화로 나타난다는 점에 착안한 것이다.

핵심 기여 사항은 다음과 같다. 첫째, 수동으로 특징을 추출하는 Hand-engineered features 방식이나 보조 센서(예: 가속도계) 없이, 원시 신호(Raw signals)만을 입력으로 사용하는 Recurrent Neural Network(RNN) 기반의 아키텍처를 제안하였다. 둘째, 수면 단계의 시간적 의존성(Temporal dependencies)을 학습하기 위해 Bidirectional Gated Recurrent Units(bi-GRU)를 사용하여 과거와 미래의 맥락을 모두 활용하였다. 셋째, 대규모 데이터셋(SHHS 1)을 통해 모델의 일반화 성능을 검증하였으며, 간단한 센서 데이터만으로도 최신 연구(State-of-the-art) 수준의 성능을 달성할 수 있음을 입증하였다.

## 📎 Related Works

기존의 자동 수면 단계 분류 연구들은 주로 뇌파(EEG)를 기반으로 하여 매우 높은 성능을 보였으나, 이는 PSG와 마찬가지로 장비의 복잡성과 비용 문제가 있다. 이를 해결하기 위해 심전도(ECG), 광전용적맥파(Photoplethysmography, PPG), 가속도계(Accelerometer) 등을 이용한 연구들이 진행되었다.

기존 접근 방식의 한계는 다음과 같다. 먼저, 많은 연구가 전통적인 머신러닝 방식을 사용하여 사람이 직접 특징을 정의하고 추출해야 했으며, 이는 데이터의 잠재적인 패턴을 놓칠 가능성이 크다. 또한, 일부 딥러닝 기반 연구(예: Malik et al.)는 Convolutional Neural Networks(CNN)를 사용하였으나, 수면과 같은 시계열 데이터의 긴 시간적 맥락을 파악하는 데에는 한계가 있다. Beattie et al.의 연구는 높은 성능을 보였으나 PPG 외에 가속도계 신호를 추가로 사용해야 한다는 제약이 있었다. 본 논문은 추가 센서 없이 오직 Pulse Oximeter에서 제공하는 $\text{HR}$과 $\text{SpO}_2$만으로 유사하거나 더 나은 성능을 내는 것을 목표로 하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 Pulse Oximeter에서 추출한 $\text{HR}$과 $\text{SpO}_2$ 신호를 입력받아, 매 초 단위로 수면/각성 상태를 예측하고, 이를 30초 단위의 다수결 투표(Majority vote)를 통해 최종 분류하는 구조를 가진다. 전체 파이프라인은 전처리 $\rightarrow$ Bidirectional GRU 층 $\rightarrow$ Softmax 층으로 구성된다.

### 주요 구성 요소 및 학습 절차

1. **전처리(Preprocessing)**: 센서 연결 상태를 나타내는 품질 신호를 이용해 유효하지 않은 데이터를 마스킹하고, 선형 보간법(Linear interpolation)으로 결측치를 채운다. 이후 훈련 데이터셋의 평균과 표준편차를 이용하여 모든 데이터를 표준화(Standardization)한다.
2. **Bidirectional GRU**: GRU는 LSTM의 단순화된 버전으로, 메모리 사용량이 적으면서도 vanishing gradient 문제를 해결한다. 본 모델은 두 개의 bi-GRU 층을 쌓아 올린 구조를 사용하며, 정방향과 역방향으로 동시에 신호를 처리하여 시간적 맥락을 효과적으로 학습한다.
3. **Softmax Layer**: GRU의 출력값은 ReLU 활성화 함수를 거친 후 Softmax 함수를 통해 'Awake' 또는 'Asleep' 클래스에 대한 확률값으로 변환된다.

### 주요 방정식

**GRU의 업데이트 과정**은 다음과 같은 수식으로 설명된다:

- 업데이트 게이트: $u^{(t)}_i = \sigma(W_{u,i} \cdot [h^{(t-1)}_i, x^{(t)}] + b_{u,i})$
- 리셋 게이트: $r^{(t)}_i = \sigma(W_{r,i} \cdot [h^{(t-1)}_i, x^{(t)}] + b_{r,i})$
- 후보 상태 벡터: $\tilde{h}^{(t)}_i = \tanh(W \cdot [r^{(t)}_i h^{(t-1)}_i, x^{(t)}] + b_{s,i})$
- 최종 상태 벡터: $h^{(t)}_i = (1 - u^{(t)}_i) h^{(t-1)}_i + u^{(t)}_i \tilde{h}^{(t)}_i$

여기서 $\sigma$는 시그모이드 함수이며, $u$ 게이트는 이전 상태를 얼마나 유지할지를, $r$ 게이트는 이전 상태의 어떤 부분을 사용할지를 결정한다.

최종 분류 단계에서는 다음의 선형 변환과 ReLU를 적용한다:
$$y = \text{relu}(Wx + b)$$

학습 시 손실 함수로는 Cross-entropy를 사용하였으며, 최적화 알고리즘으로는 Adam Optimizer를 채택하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Sleep Heart Health Study (SHHS 1) 데이터셋에서 무작위로 선정된 5,000명의 환자 데이터를 사용하였다. (훈련 2,500명, 검증 1,250명, 테스트 1,250명)
- **비교 지표**: 정확도(Accuracy), 민감도(Sensitivity), 특이도(Specificity), 정밀도(Precision), 음성 예측도(NPV), Cohen's Kappa ($\kappa$) 계수를 사용하였다.
- **TST 오차 측정**: 실제 수면 시간($\text{TST}$)과 추정 수면 시간($\hat{\text{ST}}$)의 차이를 측정하기 위해 평균 절대 오차($E_1$)와 평균 절대 오차 백분율($E_2$)을 산출하였다:
$$E_2 = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{TST}_i - \hat{\text{ST}}_i|}{\text{TST}_i} \cdot 100$$

### 주요 결과

가장 우수한 성능을 보인 모델은 **$\text{HR}$과 $\text{SpO}_2$를 모두 입력으로 사용하고 256개의 은닉 유닛(Hidden units)을 가진 2층 bi-GRU 모델**이었다. 테스트 데이터셋에서의 결과는 다음과 같다:

- **정확도**: $90.13\%$
- **민감도**: $94.13\%$
- **특이도**: $80.26\%$
- **정밀도**: $92.05\%$
- **Cohen's Kappa**: $0.74$
- **TST 평균 절대 오차 백분율 ($E_2$)**: $8.9\%$

실험 결과, $\text{SpO}_2$ 신호를 추가했을 때 성능이 소폭 향상되었으며, 은닉 유닛의 크기가 커질수록 성능이 향상되는 경향을 보였으나 과적합(Overfitting) 위험과 계산 비용이 증가하는 Trade-off가 존재함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 매우 단순한 신호인 $\text{HR}$과 $\text{SpO}_2$만을 이용하여 높은 정확도로 수면 단계를 분류할 수 있음을 보여주었다. 특히, $\text{SpO}_2$ 신호가 무호흡 이벤트 이후의 빠른 회복 양상을 통해 각성 상태를 구분하는 데 중요한 보조 정보를 제공한다는 점이 확인되었다.

**강점 및 의의**:
가장 큰 강점은 대규모 데이터셋을 사용하여 모델의 일반화 능력을 확보했다는 점이다. 또한, EEG나 가속도계 같은 추가 장비 없이 Pulse Oximeter라는 저비용 장비만으로 구현 가능하므로, 실제 의료 현장에서의 스크리닝 비용을 획기적으로 낮출 수 있다.

**한계 및 논의**:
$\text{HR}$ 신호는 Pulse Oximetry 장치마다 변동성이 크고 움직임 잡음(Motion artifacts)에 취약하다는 한계가 있다. 또한, 본 연구에서는 'Awake'와 'Asleep'이라는 이진 분류에 집중하였으나, 실제 임상에서는 N1, N2, N3, REM과 같은 세부 단계 분류가 필요할 수 있다. 다만, bi-GRU를 통해 시계열의 문맥을 파악함으로써 이러한 저해상도 신호의 한계를 상당 부분 극복하였음을 보여주었다.

## 📌 TL;DR

본 논문은 Pulse Oximeter의 $\text{HR}$과 $\text{SpO}_2$ 신호를 입력으로 하는 **Bidirectional GRU 기반의 딥러닝 모델**을 통해 수면/각성 상태를 자동으로 분류하는 방법을 제안한다. 별도의 특징 추출 없이 원시 데이터를 학습시킨 결과, 테스트셋에서 **$90.13\%$의 정확도**를 달성하였으며, 이는 추가 센서를 사용한 기존 연구들과 대등한 수준이다. 이 연구는 저비용 수면 스크리닝을 가능하게 하여 OSAHS의 정확한 진단을 돕고, 향후 졸음운전 방지 시스템이나 웨어러블 헬스케어 기기에 적용될 가능성이 매우 높다.
