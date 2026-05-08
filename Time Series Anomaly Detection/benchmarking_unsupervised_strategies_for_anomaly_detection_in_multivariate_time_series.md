# Benchmarking Unsupervised Strategies for Anomaly Detection in Multivariate Time Series

Laura Boggia, Rafael Teixeira de Lima, Bogdan Malaescu (2025)

## 🧩 Problem to Solve

본 논문은 다변량 시계열 데이터(Multivariate Time Series)에서 이상치 탐지(Anomaly Detection, AD)를 수행할 때 발생하는 기술적 난제들을 해결하고자 한다. 다변량 시계열 이상치 탐지는 헬스케어, 금융 서비스, 제조 공정 및 물리 탐지기 모니터링 등 다양한 산업 분야에서 매우 중요한 문제이다.

해결하고자 하는 핵심 문제는 크게 세 가지이다. 첫째, 이상치의 특성이 사전에 정의되지 않았거나 매우 드물게 발생하기 때문에 지도 학습(Supervised Learning)을 적용하기 어렵다는 점이다. 둘째, 여러 변수 간의 복잡한 상호 의존성(Interdependencies)으로 인해 단순한 단변량 분석으로는 이상치를 정확히 식별하기 어렵다. 셋째, 다변량 이상치 점수(Anomaly Score)를 어떻게 하나의 이진 레이블(Binary Label)로 변환하고 평가할 것인지에 대한 표준화된 방법론이 부족하다는 점이다. 따라서 본 연구의 목표는 최근 제안된 iTransformer 아키텍처를 이상치 탐지에 적용하고, 레이블 추출 전략 및 손실 함수의 영향을 분석하여 종합적인 벤치마크 결과를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 시계열 예측을 위해 제안된 iTransformer의 **Inverted Embedding** 구조가 이상치 탐지에서도 효과적일 것이라는 직관에서 출발한다. 주요 기여 사항은 다음과 같다.

1. **iTransformer의 AD 적용 및 분석**: iTransformer 아키텍처를 reconstruction-based 및 forecasting-based 이상치 탐지에 적용하고, 윈도우 크기(Window size), 스텝 크기(Step size), 모델 차원 등 핵심 하이퍼파라미터가 성능에 미치는 영향을 분석하였다.
2. **이상치 레이블 추출 전략 연구**: 다차원 이상치 점수를 단일 레이블로 변환하기 위한 Global 및 Local 결합 전략을 제안하고, 데이터셋의 특성(이상치 길이 등)에 따른 최적의 추출 방법을 고찰하였다.
3. **훈련 데이터 내 이상치의 영향 분석**: 훈련 데이터에 이상치가 포함되어 있을 때의 성능 저하 문제를 분석하고, 이를 완화하기 위해 MSE 외에 Huber loss 및 Soft-DTW loss와 같은 대안적 손실 함수의 효과를 검증하였다.
4. **포괄적인 벤치마킹**: 다양한 도메인의 데이터셋을 사용하여 iTransformer를 포함한 여러 Transformer 기반 모델(TranAD, USAD, Vanilla Transformer)의 성능을 비교 분석하였다.

## 📎 Related Works

### 기존 연구 및 한계

시계열 이상치 탐지(TSAD)는 전통적으로 Isolation Forest, One-Class SVM, k-NN 등의 비지도 학습 방식이 사용되었다. 최근에는 Deep Learning 기반의 접근 방식이 우세하며, 크게 두 가지 가정에 기반한다.

- **Reconstruction-based**: 정상 데이터는 잘 재구성되지만, 이상치는 재구성 오차(Reconstruction Error)가 클 것이라는 가정이다. Autoencoder(AE), VAE, GAN 등이 이에 해당한다.
- **Forecasting-based**: 정상 데이터는 예측이 쉽지만, 이상치는 예측 오차(Prediction Error)가 클 것이라는 가정이다. RNN, LSTM 등이 주로 사용되었다.

최근에는 Long-term dependency를 효과적으로 캡처하는 Transformer 모델이 주목받고 있으며, TranAD, Anomaly Transformer 등이 제안되었다. 그러나 기존의 벤치마크 데이터셋(Yahoo, NASA 등)은 이상치가 너무 명백하여 단순한 규칙만으로도 탐지가 가능하다는 비판이 제기되었으며, 이는 모델의 실질적인 성능 평가를 어렵게 만든다.

### 본 연구의 차별점

본 연구는 단순히 모델의 성능을 높이는 것에 그치지 않고, **Inverted Embedding**이라는 구조적 변화가 다변량 상관관계 학습에 어떤 영향을 주는지 분석한다. 또한, 모델 출력물인 다차원 점수를 최종 레이블로 변환하는 후처리 과정(Label Extraction)과 훈련 데이터의 오염(Contamination) 문제를 구체적으로 다루어 end-to-end 파이프라인의 완성도를 높였다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 시스템 구조 및 모델

본 논문은 크게 세 가지 모델 구조를 비교한다.

- **Vanilla Transformer**: 표준 Transformer 인코더를 사용하여 데이터를 재구성한다. 시간축에 대한 순서 정보를 유지하기 위해 Positional Encoding을 사용한다.
- **iTransformer**: 기존 Transformer와 달리 변수(Variates)와 시간(Time)의 역할을 반전시킨다. 즉, 각 변수의 전체 시계열을 하나의 토큰으로 취급하여 Attention 메커니즘이 시간축이 아닌 **변수 간의 상관관계**를 학습하게 한다.
- **USAD & TranAD**: 대조 학습(Adversarial Training)과 자기 조건화(Self-conditioning)를 통해 재구성 오차를 증폭시켜 미세한 이상치를 더 잘 찾도록 설계된 모델들이다.

### 2. 이상치 점수 및 레이블 추출 (Label Extraction)

모델은 각 변수별로 이상치 점수를 생성하므로, 이를 하나의 결정으로 통합하는 과정이 필요하다.

- **Global Strategy**: 모든 변수의 점수를 평균 내어 단일 시계열 점수를 만든 후, 임계값(Threshold)을 적용한다.
- **Local Strategy**: 각 변수별로 먼저 이상 여부를 판단한 뒤, 이를 통합한다.
  - **Inclusive OR**: 하나라도 이상치이면 $\text{Anomaly}$로 판단한다. (Point anomaly 탐지에 유리)
  - **Majority Voting**: 과반수 이상의 변수가 이상치이면 $\text{Anomaly}$로 판단한다. (Collective anomaly 탐지에 유리)

임계값 결정에는 Peak-over-Threshold (POT) 방법이나 백분위수(Percentile) 방식이 사용된다.

### 3. 손실 함수 (Loss Functions)

훈련 데이터에 이상치가 섞여 있을 경우, MSE는 이상치에 과하게 반응하여 모델이 이상치까지 학습해버리는 문제가 발생한다. 이를 해결하기 위해 다음을 도입한다.

- **Huber Loss**: 오차가 작을 때는 MSE를, 클 때는 MAE(Mean Absolute Error)를 사용하여 이상치에 대한 민감도를 낮춘다.
$$\ell_{\delta}(\hat{y},y) = \begin{cases} \frac{1}{2}(y-\hat{y})^2, & \text{if } |y-\hat{y}| < \delta \\ \delta(|y-\hat{y}| - \frac{1}{2}\delta), & \text{otherwise} \end{cases}$$
- **Soft-DTW Loss**: 시계열의 위상 변화에 강건한 Dynamic Time Warping을 미분 가능하게 구현한 손실 함수이다.

### 4. 학습 및 추론 절차

- **Sliding Window**: 데이터를 크기 $W$, 스텝 $S$의 윈도우로 분할한다.
- **Reconstruction-based**: 입력 윈도우를 압축 후 재구성하여 입력과 출력의 MSE를 점수로 사용한다.
- **Forecasting-based**: 이전 윈도우를 통해 마지막 타임스탬프를 예측하고, 실제 값과의 오차를 점수로 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Credit Card, GECCO, IEEECIS, MSL, SMAP, SMD, SWaT, WADI 등 다양한 도메인의 10개 데이터셋을 사용하였다.
- **평가 지표**: 불균형 데이터셋의 특성을 고려하여 Accuracy 대신 **Matthews Correlation Coefficient (MCC)**를 주 지표로 사용하였다.
$$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### 주요 결과

1. **iTransformer의 우수성**: 10개 데이터셋 중 8개에서 iTransformer가 가장 높은 MCC를 기록하였다. 특히 **iTransformer-reco (재구성 기반)**가 유연성과 효율성 측면에서 가장 뛰어난 성능을 보였다.
2. **Inverted Embedding의 효과**: Vanilla Transformer보다 iTransformer의 성능이 월등히 높았는데, 이는 변수 간의 상관관계를 직접적으로 학습하는 구조가 다변량 이상치 탐지에 훨씬 효과적임을 입증한다.
3. **레이블 추출 전략의 영향**:
    - **Point Anomaly**(짧은 이상치)가 많은 데이터셋 $\to$ **Local (Inclusive OR)** 방식이 가장 효과적이다.
    - **Collective Anomaly**(긴 이상치)가 많은 데이터셋 $\to$ **Global** 또는 **Local (Majority Voting)** 방식이 더 적합하다.
4. **훈련 데이터 오염**: 훈련 데이터에 매우 낮은 비율의 이상치만 섞여 있어도 성능이 눈에 띄게 저하되었다. 이때 Huber loss나 Soft-DTW loss를 사용하면 MSE보다 성능 저하를 완화할 수 있었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 iTransformer라는 예측 모델을 이상치 탐지에 성공적으로 이식하였으며, 단순한 모델 적용을 넘어 데이터의 특성(이상치 길이)에 맞는 후처리 전략(Label Extraction)과 훈련 데이터 상태에 따른 손실 함수 선택의 중요성을 체계적으로 분석하였다. 특히 Inverted Embedding이 다변량 데이터의 복잡한 의존성을 캡처하는 데 있어 복잡한 아키텍처 수정보다 더 효율적인 해결책이 될 수 있음을 시사한다.

### 한계 및 비판적 논의

- **데이터셋의 편향**: 일부 데이터셋(SMD, SWaT_1D)에서는 단순 Baseline 알고리즘(절대값 사용)이 이미 매우 높은 성능을 보였다. 이는 해당 데이터셋의 이상치가 너무 명백하여 복잡한 딥러닝 모델의 필요성이 낮음을 의미하며, 벤치마크 데이터셋의 질적 개선이 여전히 필요함을 보여준다.
- **훈련 데이터 레이블 부족**: 훈련 데이터 내 이상치의 영향을 분석하고자 했으나, 실제 훈련 데이터에 레이블이 제공되는 데이터셋이 적어(Credit Card, GECCO 등) 분석 범위가 제한적이었다.
- **계산 비용**: Transformer 모델의 특성상 변수 수가 매우 많은 데이터셋에서는 여전히 계산 비용 문제가 존재하며, 일부 모델(TranAD 등)은 메모리 제한으로 인해 입력 변수를 일부만 사용해야 했다.

## 📌 TL;DR

본 논문은 **iTransformer**의 Inverted Embedding 구조가 다변량 시계열 이상치 탐지(TSAD)에서 기존 Transformer 및 타 모델(USAD, TranAD)보다 뛰어난 성능을 보임을 입증하였다. 특히 **재구성 기반(Reconstruction-based)** 접근 방식이 가장 효율적이며, 이상치의 길이에 따라 **Inclusive OR**(짧은 이상치) 또는 **Global/Majority Voting**(긴 이상치) 레이블 추출 전략을 선택하는 것이 중요함을 밝혔다. 또한, 훈련 데이터의 오염 문제를 완화하기 위해 **Huber loss** 등의 대안적 손실 함수 사용을 권장한다. 이 연구는 향후 다변량 시계열 분석에서 복잡한 모델 설계보다 데이터 인코딩 방식(Inverted Embedding)의 변경이 더 큰 성능 향상을 가져올 수 있음을 시사한다.
