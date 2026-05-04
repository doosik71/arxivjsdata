# Using Photoplethysmography to Detect Real-time Blood Pressure Changes with a Calibration-free Deep Learning Model

Jingyuan Hong, Manasi Nandi, Weiwei Jin, and Jordi Alastruey (2024)

## 🧩 Problem to Solve

본 연구는 실시간 혈압(Blood Pressure, BP) 변화를 비침습적으로 감지하는 것을 목표로 한다. 혈압의 급격한 변화는 임상 환경에서 즉각적인 의료 개입이 필요한 응급 상황을 나타내며, 비임상 환경에서도 심혈관 질환 예방 및 건강 상태 모니터링에 중요한 지표가 된다.

현재 혈압 측정 방식은 침습적 방식(Arterial Catheter)과 비침습적 방식(Cuff-based)으로 나뉜다. 침습적 방식은 정확도가 매우 높으나 환자에게 통증과 감염 위험을 초래하며, 비침습적 커프 방식은 불편함을 유발하고 연속적인 비트 단위(beat-to-beat)의 혈압 변화를 캡처하는 데 한계가 있다. 최근 광혈류측정(Photoplethysmography, PPG) 신호를 이용한 혈압 추정 연구가 진행되고 있으나, 대부분의 모델이 개인별 보정(Calibration) 과정과 연령 등의 인구통계학적 정보가 필요하며, 절대적인 혈압 수치를 정확히 예측하는 데 어려움이 있다.

따라서 본 논문은 절대적인 수치 예측 대신, PPG 파형을 이용하여 혈압의 변화 상태를 범주형(Categorical)으로 분류하는 보정 없는(Calibration-free) 딥러닝 모델을 개발하여 실시간 혈압 모니터링의 가능성을 제시하고자 한다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 PPG 파형만을 이용하여 별도의 개인별 보정 과정 없이 수축기 혈압(SBP), 이완기 혈압(DBP), 평균 혈압(MBP)의 변화를 실시간으로 감지할 수 있는 분류 모델을 제안한 것이다.

특히, PPG 신호뿐만 아니라 PPG의 2차 미분 파형인 $\text{second-derivative PPG (sdPPG)}$를 입력 데이터로 함께 활용함으로써 모델의 예측 성능을 유의미하게 향상시켰다. 또한, Transformer 아키텍처의 Encoder 구조와 Softmax 가중치 기반의 Attention 메커니즘을 도입하여 시계열 데이터의 중요한 특징에 집중함으로써 분류 정확도를 높였다.

## 📎 Related Works

기존의 혈압 측정 방식은 앞서 언급한 바와 같이 침습적 방식의 위험성과 비침습적 방식의 낮은 시간 해상도라는 한계가 있었다. PPG를 이용한 절대 혈압 추정 연구들은 많은 시도가 있었으나, 개인마다 다른 혈관 특성으로 인해 높은 정확도를 얻기 위해서는 개인별 기준 신호를 이용한 보정(Calibration) 단계가 필수적이었다.

본 연구는 이러한 기존 접근 방식과 달리, 절대 수치 추정이 아닌 '변화량'에 집중하는 범주형 분류 문제로 접근함으로써 보정 과정의 필요성을 제거하고, PPG 파형의 형태적 변화(Morphology)와 혈압 변화 사이의 상관관계를 직접적으로 학습하는 방식을 취하여 차별점을 두었다.

## 🛠️ Methodology

### 1. 데이터셋 및 전처리

본 연구는 1,005명의 ICU 환자 데이터가 포함된 Vital Signs Database (VitalDB)를 사용하였다. PPG와 BP 신호는 모두 $125\text{ Hz}$의 샘플링 속도로 측정되었으며, 모든 PPG 신호는 완전한 심장 주기(cardiac cycle)로 시작하도록 7초 길이로 절단하여 표준화하였다.

### 2. 혈압 변화의 정의 및 라벨링

혈압 변화 $\Delta \text{BP}$는 초기 시점 $i$와 이후 시점 $i+j$ 사이의 차이로 계산된다.
$$\Delta \text{BP} = \text{BP}_{i+j} - \text{BP}_i$$
여기서 $i$는 초기 혈압 측정 인덱스, $j$는 이후 측정까지의 간격, $N$은 전체 측정 횟수를 의미한다. 혈압 변화는 설정된 임계값(Threshold)에 따라 다음과 같이 세 가지 라벨로 분류된다.

- **Spike**: $\Delta \text{BP} > \text{Threshold}$ (유의미한 증가)
- **Stable**: $-\text{Threshold} \le \Delta \text{BP} \le \text{Threshold}$ (정상 범위 내 유지)
- **Dip**: $\Delta \text{BP} < -\text{Threshold}$ (유의미한 감소)

초기 임계값은 $\text{SBP}$는 $30\text{ mmHg}$, $\text{DBP}$는 $15\text{ mmHg}$, $\text{MBP}$는 $20\text{ mmHg}$로 설정하였다.

### 3. 분류 모델 아키텍처

본 연구에서는 네 가지 시계열 분류 모델을 비교 분석하였다.

- **Multi-layer Perceptron (MLP)**: 4개의 Fully Connected 레이어로 구성된 기본 구조이다.
- **Convolutional Neural Network (CNN)**: 3개의 Convolution 블록과 Global Average Pooling을 통해 특징을 추출한다.
- **Residual Network (ResNet)**: CNN에 Shortcut 연결을 추가하여 기울기 소실 문제를 해결하고 더 깊은 층을 학습한다.
- **Encoder**: CNN 블록 뒤에 Attention 메커니즘을 결합한 구조이다. Softmax 함수를 통해 각 특징에 대한 가중치를 생성하고, 이를 통해 정보량이 많은 부분에 집중하여 출력값을 산출한다.

### 4. 입력 데이터 구성 및 학습 절차

모델의 성능을 높이기 위해 세 가지 입력 조합을 실험하였다.

1. **PPG-waveform**: 기본 PPG 파형만 사용.
2. **Waveform-feature**: PPG 파형과 $\text{sdPPG}$에서 추출한 5가지 수치 특징(Feature)을 결합.
3. **PPG-sdPPG-waveform**: PPG 파형과 $\text{sdPPG}$ 파형 전체를 함께 입력.

또한, 초기 혈압 값 $\text{BP}_i$를 추가 입력 정보로 제공하여 모델이 기준점을 인지하도록 설계하였다. 손실 함수로는 분류 문제에 적합한 $\text{Cross-Entropy Loss}$를 사용하였다.
$$L = H(p, q) = \sum_{x=1}^{C} p(x) \log q(x)$$
여기서 $p(x)$는 실제 라벨의 확률, $q(x)$는 모델이 예측한 확률, $C$는 클래스의 수(3개)이다.

## 📊 Results

### 1. 모델 성능 비교

실험 결과, **Encoder 모델**이 모든 혈압 유형(SBP, DBP, MBP)에서 가장 높은 정확도와 F1-score를 기록하였다. 특히 MBP의 변화를 감지하는 성능이 SBP나 DBP보다 더 높게 나타났다.

### 2. 입력 데이터의 영향

$\text{sdPPG}$ 파형을 함께 입력한 **PPG-sdPPG-waveform** 조합이 가장 우수한 성능을 보였다. 이는 단순한 특징 추출(Waveform-feature)보다 파형 전체를 딥러닝 모델에 입력하는 것이 더 효과적임을 시사한다.

### 3. 정량적 결과 (Encoder 모델 기준)

최적 임계값($\text{SBP}: 30, \text{DBP}: 15, \text{MBP}: 20\text{ mmHg}$) 적용 시 결과는 다음과 같다.

- **Test-I (균등 분포 샘플링)**: 정확도 $71.3\% \sim 76.7\%$, F1-score $71.8\% \sim 76.9\%$
- **Test-II (실제 환자 5명 전체 데이터)**: 정확도 $85.4\% \sim 90.8\%$, F1-score $88.5\% \sim 90.8\%$

### 4. 임계값 및 입력 길이 분석

- **임계값 영향**: Test-I에서는 임계값이 낮을 때 정확도가 높았으나, Test-II에서는 임계값이 높을수록(Stable 범위가 넓어질수록) 정확도가 $100\%$에 근접하는 경향을 보였다. 이는 Test-II 데이터의 분포가 정규 분포와 유사하여 대부분의 데이터가 Stable 범위에 포함되기 때문이다.
- **신호 길이**: PPG 입력 길이를 $3\text{s}, 5\text{s}, 7\text{s}$로 변경하며 테스트한 결과, $7\text{s}$일 때 전반적으로 가장 좋은 성능을 보였으나 길이 감소에 따른 성능 하락 폭은 크지 않았다.

## 🧠 Insights & Discussion

본 연구는 PPG 신호와 초기 혈압 값만을 이용하여 보정 과정 없이 혈압 변화를 분류할 수 있음을 입증하였다. 특히 $\text{sdPPG}$ 파형이 혈압 변화와 밀접한 관련이 있다는 기존 연구 결과를 딥러닝 모델에 성공적으로 적용하여 성능을 향상시켰다.

**강점 및 의의**:
가장 큰 의의는 'Calibration-free' 모델이라는 점이다. 기존 연구들이 연령, 성별 등 인구통계학적 정보나 개인별 맞춤형 튜닝이 필요했던 것과 달리, 본 모델은 일반화된 PPG 특징을 통해 변화를 감지한다. 이는 실제 웨어러블 기기 등에 적용했을 때 사용자 편의성을 획기적으로 높일 수 있는 요소이다.

**한계 및 비판적 해석**:

1. **데이터 편향**: VitalDB는 ICU 환자 데이터이므로, 의료적 처치나 기저 질환에 의한 혈압 변화가 주를 이룬다. 따라서 건강한 일반인 집단에 이 모델을 그대로 적용했을 때 동일한 성능이 나올지는 추가 검증이 필요하다.
2. **시간적 지연(Lag)**: 실제 혈압이 변한 후 모델이 이를 감지하기까지 약간의 시간적 지연이나 임계값 오판단(Misjudgement)이 발생하였다. 이는 PPG와 BP 사이의 히스테리시스(Hysteresis) 효과나 PPG 측정 자체의 오차에서 기인한 것으로 분석된다.
3. **절대치 추정의 한계**: 모델이 변화의 방향(증가/감소)은 잘 파악하지만, 정확한 절대 수치를 계산하는 것이 아니므로 임계값 경계 근처에서 오분류가 발생하는 한계가 있다.

## 📌 TL;DR

본 논문은 PPG 및 $\text{sdPPG}$ 파형을 입력으로 하여 혈압의 변화 상태를 **Spike, Stable, Dip**의 세 가지 범주로 분류하는 **보정 없는(Calibration-free)** 딥러닝 모델을 제안하였다. Encoder 기반 아키텍처를 통해 실시간 혈압 변화 감지에서 높은 정확도를 달성하였으며, 이는 향후 환자 맞춤형 보정 과정 없이도 사용 가능한 실시간 비침습적 혈압 모니터링 시스템 구축에 중요한 기초 연구가 될 것으로 평가된다.
