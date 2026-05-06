# Impact of Recurrent Neural Networks and Deep Learning Frameworks on Real-time Lightweight Time Series Anomaly Detection

Ming-Chang Lee, Jia-Chun Lin, and Sokratis Katsikas (2024)

## 🧩 Problem to Solve

본 논문은 실시간 경량 시계열 이상 탐지(Real-time Lightweight Time Series Anomaly Detection) 시스템을 구축할 때, 사용되는 Recurrent Neural Network(RNN)의 변형 모델(Variant)과 이를 구현하는 Deep Learning(DL) 프레임워크가 실제 성능에 어떠한 영향을 미치는지를 분석한다.

시계열 이상 탐지는 사이버 보안, IoT 센서 데이터 분석, 클라우드 인프라 관리 등 다양한 분야에서 매우 중요하며, 특히 오프라인 학습 없이 실시간으로 적응(Adaptability)하고 가볍게(Lightweight) 동작하는 모델이 요구된다. 그러나 기존의 최신 연구들은 주로 단일 종류의 RNN(예: LSTM)만을 사용하거나 특정 프레임워크 내에서만 구현되어, 다른 RNN 변형 모델이나 다른 프레임워크를 사용했을 때의 성능 변화에 대한 종합적인 평가가 부족한 실정이다. 따라서 임의로 선택한 RNN 모델과 프레임워크가 모델의 진정한 성능을 반영하지 못하거나 사용자에게 잘못된 선택을 유도할 수 있다는 점이 본 연구의 핵심 문제 의식이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 실시간 경량 시계열 이상 탐지를 위한 대표적인 접근 방식인 RePAD2를 기반으로, 다양한 RNN 변형 모델(RNN, LSTM, GRU)과 널리 쓰이는 3가지 DL 프레임워크(TensorFlow-Keras, PyTorch, Deeplearning4j)의 조합이 탐지 정확도와 시간 효율성에 미치는 영향을 체계적으로 분석하고 정량적으로 평가한 것이다.

## 📎 Related Works

논문에서는 DL 프레임워크 간의 성능을 비교한 기존 연구들을 소개한다. 일부 연구는 완전 연결 신경망(Fully Connected Neural Network)의 학습 및 예측 시간을 비교하거나, 모바일 기기에서의 이미지 분류, 객체 탐지 등의 벤치마크를 수행하였다. 또한 NLP 작업에 특화된 프레임워크 비교 연구도 존재한다.

하지만 이러한 기존 연구들은 다음과 같은 한계가 있다.

1. **작업의 복잡도 차이**: CNN이나 NLP 기반의 모델들은 본 연구에서 다루는 경량 시계열 이상 탐지 모델보다 훨씬 복잡하여, 그 결과가 경량 모델에 그대로 적용되기 어렵다.
2. **분석 요소의 부재**: DL 프레임워크의 영향만을 분석했을 뿐, RNN의 변형 모델(RNN, LSTM, GRU)이 프레임워크와 결합했을 때 발생하는 상호작용에 대해서는 다루지 않았다.

## 🛠️ Methodology

### 전체 시스템 구조 및 대상 모델: RePAD2

본 연구는 실시간, 경량, 적응형 특성을 모두 갖춘 대표적인 모델인 RePAD2를 평가 대상으로 선정하였다. RePAD2는 오프라인 학습 없이 실시간으로 모델을 학습시키고 예측하는 'Look-Back and Predict-Forward' 전략을 사용한다.

### 상세 동작 절차 및 방정식

RePAD2는 현재 시점 $T$에서 과거 3개의 데이터 포인트를 사용하여 LSTM 모델을 학습시키고, 다음 데이터 포인트 $\hat{D}_T$를 예측한다.

1. **오차 계산 (AARE)**:
   예측 정확도를 측정하기 위해 다음과 같이 평균 절대 상대 오차(Average Absolute Relative Error, AARE)를 계산한다.
   $$\text{AARE}_T = \frac{1}{3} \sum_{y=T-2}^{T} \frac{|D_y - \hat{D}_y|}{D_y}$$
   여기서 $D_y$는 실제 값, $\hat{D}_y$는 예측 값이다.

2. **탐지 임계값 (Threshold, $\text{thd}$) 설정**:
   이상치 판별을 위한 임계값 $\text{thd}$는 과거 AARE 값들의 평균($\mu_{aare}$)과 표준편차($\sigma_{aare}$)를 이용하여 다음과 같이 계산한다.
   $$\text{thd} = \mu_{aare} + 3\sigma_{aare}$$
   이때, 리소스 고갈 문제를 방지하기 위해 최근 $W$개의 AARE 값만을 사용하도록 제한한다.

3. **이상 탐지 및 적응 로직**:
   - $\text{AARE}_T \le \text{thd}$인 경우: 정상으로 판단하고 현재 모델을 유지한다.
   - $\text{AARE}_T > \text{thd}$인 경우: 패턴 변화 가능성이 있다고 판단하여 최신 데이터로 모델을 재학습(Retraining)하고 다시 예측한다.
   - 재학습 후에도 $\text{AARE}_T > \text{thd}$라면, 해당 데이터 포인트 $D_T$를 최종적으로 이상치(Anomaly)로 보고한다.

### 실험 설계 및 구현 조합

본 연구는 다음과 같이 총 7가지의 구현 조합을 생성하여 비교 분석하였다.

| RNN Variant | TensorFlow-Keras | PyTorch | Deeplearning4j |
| :--- | :---: | :---: | :---: |
| **RNN** | TFK-RNN | PT-RNN | N/A |
| **LSTM** | TFK-LSTM | PT-LSTM | DL4J-LSTM |
| **GRU** | TFK-GRU | PT-GRU | N/A |

## 📊 Results

### 실험 환경 및 데이터셋

- **데이터셋**: UCI Machine Learning Repository의 공기 질(Air Quality) 데이터셋 3종(PT08.S1, C6H6, PT08.S2)을 사용하였다. 각 데이터셋은 9,357개의 포인트로 구성되며, 누락된 데이터(Missing points)를 포인트 이상치 및 집단 이상치로 간주하였다.
- **평가 지표**:
  - **정확도**: Precision, Recall, F1-score.
  - **효율성**: 온라인 모델 재학습 비율(Retraining ratio), 재학습 시 소요 시간($\text{DT-Train}$), 재학습 없을 시 소요 시간($\text{DT-noTrain}$).
- **환경**: GPU 없는 일반 MacBook (Intel Core i7, 16GB RAM)에서 수행하여 범용적인 환경에서의 성능을 측정하였다.

### 주요 결과

1. **탐지 정확도**:
   - **DL4J-LSTM**이 모든 데이터셋에서 가장 높은 F1-score를 기록하며 최적의 성능을 보였다. 이는 높은 Recall뿐만 아니라 오탐(False Positive)을 가장 적게 생성했기 때문이다.
   - **TFK-RNN**이 두 번째로 높은 성능을 보였으나, DL4J-LSTM보다는 오탐이 많았다.
   - PyTorch 기반 구현체들은 데이터셋에 따라 정확도가 불안정하게 나타났다.

2. **시간 효율성**:
   - **재학습 시 ($\text{DT-Train}$)**: PyTorch 기반 모델들이 압도적으로 빠른 속도를 보였다.
   - **단순 예측 시 ($\text{DT-noTrain}$)**: **DL4J-LSTM**이 가장 빨랐으며(약 0.013~0.014초), PyTorch가 그 뒤를 이었다.
   - **TensorFlow-Keras**는 모든 지표에서 가장 느린 처리 속도를 기록하였다.

3. **재학습 비율**:
   - DL4J-LSTM과 TFK-RNN의 재학습 비율이 가장 낮았으며, 이는 해당 모델들의 예측 성능이 상대적으로 안정적임을 의미한다. 반면 PT-LSTM과 PT-GRU는 재학습 빈도가 더 높았다.

## 🧠 Insights & Discussion

본 연구 결과는 실시간 경량 이상 탐지 시스템에서 단순한 알고리즘 선택보다 **"어떤 프레임워크의 어떤 RNN 변형을 사용하는가"**가 실제 성능에 지대한 영향을 미친다는 것을 입증한다.

- **DL4J-LSTM의 우수성**: Deeplearning4j의 LSTM 구현체는 정확도와 추론 속도(no-train 시) 면에서 모두 최적의 선택지임이 확인되었다.
- **프레임워크 간 특성 차이**: PyTorch는 모델의 학습/재학습 속도가 매우 빠르지만, 본 연구의 경량 설정에서는 정확도가 일관되지 않은 경향을 보였다. TensorFlow-Keras는 편의성은 높으나 실시간 경량 시스템으로서는 시간 효율성이 너무 낮아 부적합한 것으로 판단된다.
- **한계 및 시사점**: 본 실험은 GPU 없는 CPU 환경에서 진행되었으므로, GPU 가속 시의 결과는 달라질 수 있다. 또한, 사용된 데이터셋이 공기 질 데이터에 한정되어 있어, 더 다양한 도메인의 시계열 데이터에 대한 검증이 필요하다.

## 📌 TL;DR

이 논문은 실시간 경량 시계열 이상 탐지 모델인 RePAD2를 구현함에 있어 RNN 종류와 DL 프레임워크의 조합이 성능에 미치는 영향을 분석하였다. 실험 결과, **Deeplearning4j(DL4J)의 LSTM** 조합이 정확도와 실시간 처리 속도 면에서 가장 뛰어난 성능을 보였으며, TensorFlow-Keras는 효율성이 낮아 추천되지 않는다. 이 연구는 실시간 제약 조건이 강한 환경(IoT, 사이버 보안 등)에서 이상 탐지 시스템을 설계할 때 구체적인 구현 스택 선택의 중요성을 제시한다.
