# Impact of Physical Activity on Sleep: A Deep Learning Based Exploration

Aarti Sathyanarayana, Shafiq Joty, Luis Fernandez-Luque, Ferda Ofli, Jaideep Srivastava, Ahmed Elmagarmid, Shahrad Taheri, Teresa Arora (2016)

## 🧩 Problem to Solve

수면은 신체적, 정서적, 정신적 웰빙을 유지하는 데 매우 중요하며, 부족한 수면은 인슐린 저항성, 고혈압, 심혈관 질환, 인지 기능 저하 등 다양한 건강 문제를 야기한다. 수면의 질을 진단하는 표준 방법인 수면다원검사(Polysomnography, PSG)는 정확하지만, 병원 내에서 수행되어야 하는 번거로움이 있고 비용이 많이 들며 환자의 자연스러운 환경을 반영하지 못한다는 한계가 있다.

이에 대한 대안으로 웨어러블 기기를 이용한 액티그래피(Actigraphy)가 널리 사용되고 있으나, 액티그래피 데이터 분석에는 여전히 병목 현상이 존재한다. 기존의 분석 방식은 수면 전문가가 수동으로 파라미터를 설정해야 하거나, RAHAR와 같은 비지도 학습 기반의 전처리 알고리즘을 통해 특징 공간(feature space)을 생성해야 한다. 이러한 전처리 과정은 데이터의 풍부함을 손실시키고 워크플로우를 복잡하게 만든다. 따라서 본 논문의 목표는 딥러닝을 활용하여 전처리 과정 없이 원시(raw) 가속도계 데이터로부터 수면의 질을 직접 예측하는 모델을 구축하고, 그 효과를 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 딥러닝의 고차원 데이터 처리 능력을 활용하여, 복잡한 특징 추출(feature extraction)이나 인간 활동 인식(Human Activity Recognition, HAR) 단계 없이 원시 센서 데이터에서 수면의 질을 직접 예측하는 것이다. 특히, 시계열 데이터의 지역적 패턴을 학습할 수 있는 Convolutional Neural Network(CNN)를 도입함으로써, 기존의 수동 전처리 방식이나 전통적인 머신러닝 모델보다 더 높은 예측 성능을 달성하고 분석 워크플로우를 단순화하는 것을 목표로 한다.

## 📎 Related Works

수면 연구에서는 전통적으로 PSG가 사용되었으나 휴대성이 낮아, 최근에는 가속도계를 이용한 액티그래피가 대안으로 제시되었다. 액티그래피는 주관적인 수면 일기보다 신뢰도가 높고 장기간 모니터링이 가능하다는 장점이 있다.

기존의 자동화된 접근 방식으로는 저자들의 이전 연구인 RAHAR 알고리즘이 있으며, 이는 가속도 데이터를 4가지 활동 강도(sedentary, light, moderate, vigorous)로 변환하여 특징을 생성한다. 그러나 RAHAR는 비지도 학습 방식이기에 특정 작업(task)에 최적화된 특징을 학습하지 못하며, 데이터 집계(aggregation) 과정에서 정보 손실이 발생한다는 한계가 있다. 최근 센서 데이터 기반의 HAR 분야에서는 CNN, RNN, LSTM 등의 딥러닝 모델이 가속도계의 원시 신호에서 직접 특징을 학습하여 SOTA(State-of-the-art) 성능을 보이고 있으며, 본 연구는 이러한 경향을 수면의 질 예측 문제에 적용한다.

## 🛠️ Methodology

### 1. 데이터 표현 및 라벨링

데이터는 92명의 청소년을 대상으로 7일간 ActiGraph GT3X+ 기기를 통해 수집되었다. 수면의 질을 측정하기 위해 수면 효율(Sleep Efficiency) 지표를 사용하며, 정의는 다음과 같다.

$$\text{Sleep Efficiency} = \frac{\text{Total Sleep Time}}{\text{Total Minutes in Bed}} = \frac{\text{length(Sleep Period)} - \text{WASO}}{\text{length(Sleep Period)} + \text{Latency}}$$

여기서 $\text{WASO}$(Wake After Sleep Onset)는 수면 기간 중 깨어 있는 시간의 합이며, $\text{Latency}$는 잠들기까지 걸리는 시간이다. 수면 효율이 $85\%$ 이상이면 "Good", 그렇지 않으면 "Poor"로 라벨링한다.

### 2. 모델 아키텍처

본 연구에서는 네 가지 딥러닝 모델을 실험하였다. 모든 모델은 입력 $X = (x_1, \dots, x_T)$를 받아 수면의 질 $y \in \{\text{good, poor}\}$를 예측하며, 출력층은 다음과 같은 베르누이 분포를 정의하는 시그모이드 함수를 사용한다.

$$p(y|X, \theta) = \text{Ber}(y|\text{sig}(w^T \phi(X) + b))$$

- **Multi-Layer Perceptron (MLP):** 입력 데이터를 하나로 연결하여 완전 연결층(fully-connected layer)을 통과시킨다. $\phi(X) = f(V x_{1:T})$ 형태로 구현되며, 복잡한 의존성을 모델링한다.
- **Convolutional Neural Network (CNN):** 지역적인 시간 윈도우에 필터를 적용하여 특징 맵(feature map)을 생성한다.
  $$h_t = f(u \cdot x_{t:t+L-1})$$
  이후 Max-pooling을 통해 가장 중요한 특징만을 추출하여 위치 불변성(location invariance)을 확보한다.
- **Recurrent Neural Network (RNN):** 데이터를 순차적으로 처리하며 이전 상태 $h_{t-1}$을 현재 입력 $x_t$와 결합하여 동적 시간 행동을 모델링한다.
  $$h_t = f(U h_{t-1} + V x_t)$$
- **Long Short-Term Memory (LSTM):** RNN의 기울기 소실(vanishing gradient) 문제를 해결하기 위해 입력, 출력, 망각 게이트(input, output, forget gates)를 갖춘 메모리 블록을 사용하여 장기 의존성을 학습한다.

### 3. 학습 절차

모든 모델은 예측 분포 $\hat{y}$와 실제 라벨 $y$ 사이의 교차 엔트로피(Cross-Entropy) 손실 함수를 최소화하는 방향으로 학습된다.

$$J(\theta) = -\sum [y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$$

최적화 알고리즘으로는 RMSprop를 사용하였으며, 과적합 방지를 위해 ReLU 활성화 함수, Dropout, 그리고 검증 세트를 이용한 조기 종료(early stopping) 기법을 적용하였다.

## 📊 Results

실험은 RAHAR로 전처리된 데이터셋과 원시 가속도계 데이터셋 두 가지 환경에서 수행되었으며, 데이터는 70:15:15(훈련:테스트:검증) 비율로 분할되었다.

### 1. RAHAR 전처리 데이터 결과

특징 수가 4개로 매우 제한적인 이 데이터셋에서는 MLP가 가장 좋은 성능을 보였으며, 기존의 Random Forest(RF)나 Logistic Regression(LR)보다 약간 높은 AU-ROC 성능을 기록하였다.

### 2. 원시 가속도계 데이터 결과

전처리 없이 원시 데이터를 입력으로 사용했을 때, 딥러닝 모델의 성능이 비약적으로 향상되었다.

- **CNN:** 가장 우수한 성능을 기록하였다. $\text{AU-ROC} = 0.9456$, $\text{F1-Score} = 0.9444$, $\text{Accuracy} = 0.9286$을 달성하였다.
- **MLP:** $\text{AU-ROC} = 0.9449$로 CNN과 유사하게 매우 높은 성능을 보였으며, 이는 원시 데이터의 풍부한 특징 공간이 모델 성능을 끌어올렸음을 시사한다.
- **LSTM 및 RNN:** LR보다는 성능이 좋았으나 CNN/MLP에 비해 낮았다. 특히 RNN은 $\text{AU-ROC} = 0.7143$에 그쳤다.
- **비교 분석:** 원시 데이터를 사용한 CNN은 기존의 SOTA 비-딥러닝 접근 방식보다 예측 가치를 약 $8\%$ 추가로 향상시켰으며, 이는 다시 현재의 일반적인 관행보다 $15\%$ 더 높은 수치이다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 딥러닝이 가속도계의 원시 데이터에서 직접 유용한 특징을 추출할 수 있음을 입증하였다. 특히 CNN이 가장 높은 성능을 보인 이유는, 수면의 질이 전체 깨어 있는 시간의 특정 지역적 활동 패턴(예: 특정 시간대의 고강도 운동)에 의해 결정될 가능성이 높기 때문이며, CNN의 필터와 풀링 구조가 이러한 시간 불변적(time-invariant) 패턴을 포착하는 데 최적이었기 때문으로 해석된다.

### 한계 및 비판적 해석

RNN과 LSTM의 성능이 기대보다 낮게 나타난 점은 주목할 만하다. 저자들은 데이터의 세밀함(granularity)과 순차적 의존성으로 인해 기울기 소실 문제가 발생했을 가능성을 언급하며, 더 긴 시간 단위의 에포크(epoch)로 데이터를 집계하는 추가 연구가 필요함을 명시하였다. 또한, 92명이라는 상대적으로 작은 표본 크기로 인해 모델의 일반화 성능을 완전히 확신하기 어렵다는 점이 한계로 작용한다.

### 임상적 중요성

본 연구는 수면 연구의 워크플로우에서 수동 전처리 단계를 제거함으로써 분석 효율성을 극대화할 수 있음을 보여주었다. 이는 향후 웨어러블 기기를 통한 대규모 인구 통계학적 수면 연구나 실시간 수면 모니터링 시스템 구축에 있어 중요한 기술적 토대가 될 수 있다.

## 📌 TL;DR

본 논문은 웨어러블 가속도계의 원시 데이터를 활용하여 수면의 질을 예측하는 딥러닝 모델을 제안한다. 전처리 과정 없이 원시 데이터를 그대로 사용한 **CNN 모델이 가장 높은 성능($\text{AU-ROC} = 0.9456$)**을 보였으며, 이는 기존의 수동 특징 추출 방식보다 예측력을 크게 향상시킨다. 이 연구는 딥러닝이 수면 연구의 복잡한 데이터 전처리 과정을 단순화하고 진단 정확도를 높일 수 있는 강력한 도구임을 시사한다.
