# Forecasting Hierarchical Time Series

Seema Sangari, Xinyan Zhang (2022)

## 🧩 Problem to Solve

본 논문은 계층적 시계열(Hierarchical Time Series, HTS) 데이터의 예측 과정에서 발생하는 일관성 유지와 성능 저하 문제를 해결하고자 한다. 계층적 시계열은 하위 레벨의 시계열 값들의 합이 상위 레벨의 값과 일치해야 하는 구조적 제약을 가진다.

이러한 구조로 인해 두 가지 주요 문제가 발생한다. 첫째, 계층의 각 레벨에 존재하는 개별 시계열 모델들을 각각 독립적으로 추정해야 하는 번거로움이 있다. 둘째, 하위 레벨의 모델들이 가진 노이즈가 상위 레벨로 합산되면서 상위 레벨 모델의 성능이 저하되거나, 반대로 상위 레벨에서 하위 레벨로 분배할 때 개별 시계열의 동적인 특성이 사라지는 문제가 발생한다. 따라서 본 논문의 목표는 상위 레벨의 총합을 유지하면서도 하위 레벨 간의 수평적 관계를 보존할 수 있는 새로운 Top-down 예측 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 비율(Proportions) 기반 접근 방식 대신 **Odds**를 활용하고, 선형 회귀(Linear Regression) 대신 **선형 연립방정식(Systems of Linear Equations)**을 통해 하위 레벨의 예측값을 도출하는 것이다.

- **Odds의 도입**: 특정 변수와 나머지 변수들의 합 사이의 관계를 나타내는 Odds를 예측함으로써, 동일 레벨 내 변수들 간의 수평적 관계(Horizontal relationship)를 보존한다.
- **선형 연립방정식 기반 복원**: 예측된 상위 레벨의 총합($\hat{S}$)과 각 변수의 예측된 Odds를 이용하여 선형 연립방정식을 구성하고, 이를 통해 하위 레벨의 구체적인 예측값을 유일하게 결정한다.

## 📎 Related Works

기존의 계층적 시계열 예측 방식은 크게 두 가지로 나뉜다.

1. **Top-down approach**: 상위 레벨을 먼저 예측하고 이를 하위로 분배하는 방식이다. 하지만 이 방식은 하위 레벨 개별 시계열이 가진 고유한 동적 특성과 추세(Trend) 정보를 무시한다는 한계가 있다.
2. **Bottom-up approach**: 최하위 레벨을 예측하고 이를 합산하여 상위 레벨을 구하는 방식이다. 하위 레벨의 특성을 잘 반영하지만, 하위 레벨의 노이즈(Noise)가 상위로 갈수록 누적되어 상위 레벨의 예측 정확도가 떨어지는 문제가 있다.

Athanasopoulos et al. (2009)은 예측된 비율(Proportions)에 선형 회귀를 적용하여 이 문제를 해결하려 했으나, 본 논문은 비율 대신 Odds를 사용하고 회귀 분석 대신 정교한 연립방정식 풀이 방식을 채택함으로써 차별성을 둔다.

## 🛠️ Methodology

### 1. 계층 구조 정의

본 논문에서는 3단계 계층 구조를 가정한다.

- **Bottom-level**: $X_{ij}$ (최하위 단위의 카운트)
- **Mid-level**: $Y_i = \sum_{j=1}^{n} X_{ij}$ (중간 단위의 합)
- **Top-level**: $S = \sum_{i=1}^{m} Y_i$ (전체 총합)

### 2. 예측 절차

- **상위 레벨 예측**: $S$는 단변량 시계열(Univariate time series)이므로, 일반적인 시계열 예측 모델을 통해 $\hat{S}$를 예측한다.
- **중간 및 하위 레벨 예측**: 개별 값을 직접 예측하는 대신, 변수 간의 관계를 나타내는 **Odds**를 예측한다. 특정 변수 $Y_k$의 Odds 정의는 다음과 같다.
$$\text{Odds}(Y_k) = \frac{Y_k}{\sum_{i \neq k} Y_i}$$

### 3. 선형 연립방정식 구성

예측된 상위 레벨 값 $\hat{S}$와 예측된 Odds $\widehat{\text{Odds}}(Y_i)$를 이용하여 다음과 같이 식을 전개한다.
$$1 + \widehat{\text{Odds}}(Y_1) = 1 + \frac{\hat{Y}_1}{\hat{Y}_2 + \hat{Y}_3} = \frac{\hat{Y}_1 + \hat{Y}_2 + \hat{Y}_3}{\hat{Y}_2 + \hat{Y}_3}$$
여기서 $\hat{Y}_1 + \hat{Y}_2 + \hat{Y}_3 = \hat{S}$ 이므로, 다음과 같은 관계식이 성립한다.
$$\hat{Y}_2 + \hat{Y}_3 = \frac{\hat{S}}{1 + \widehat{\text{Odds}}(Y_1)}$$

이러한 방식으로 모든 변수에 대해 식을 세우면, 다음과 같은 행렬 형태로 표현할 수 있다.
$$\begin{bmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{bmatrix} \begin{bmatrix} \hat{Y}_1 \\ \hat{Y}_2 \\ \hat{Y}_3 \end{bmatrix} = \begin{bmatrix} \frac{\hat{S}}{1 + \widehat{\text{Odds}}(Y_1)} \\ \frac{\hat{S}}{1 + \widehat{\text{Odds}}(Y_2)} \\ \frac{\hat{S}}{1 + \widehat{\text{Odds}}(Y_3)} \end{bmatrix}$$
대각 성분이 0이고 나머지 성분이 1인 이 이진 행렬(Binary matrix)은 항상 가역적(Invertible)이므로, 유일한 해 $\hat{Y}_i$를 구할 수 있다.

### 4. 음수 값 처리

선형 방정식의 해로 음수 값이 도출될 수 있다. 카테고리 데이터 특성상 음수는 불가능하므로, 음수 값을 0으로 처리하고 해당 차이만큼을 나머지 변수들에 비례적으로 배분하여 합계를 유지한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: INARMA(Integer-valued ARMA) 방식을 사용하여 생성된 시뮬레이션 데이터. (1,000개 변수, 1,000개 타임스텝, 포아송 분포 기반 정수값)
- **비교 모델**: ARIMA, Vanilla LSTM, Stacked LSTM, BiDirectional LSTM, CNN LSTM, Convolutional LSTM.
- **평가 지표**: RMSPE (Root Mean Square Percentage Error).
- **데이터 분할**: 970 스텝 학습, 30 스텝 예측.

### 2. 주요 결과

- **전반적 성능**: ARIMA와 Stacked LSTM이 가장 우수한 성능을 보였으며, 오차의 75%가 5% 미만으로 나타났다.
- **레벨별 성능**:
  - **Top-level**: ARIMA, Stacked LSTM, BiDirectional LSTM의 RMSPE가 2.5% 미만으로 매우 높게 예측되었다.
  - **Mid/Bottom-level**: 일부 이상치(Outlier)가 발생하였으나, ARIMA와 Stacked LSTM은 대부분 10% 미만의 오차율을 유지하였다.
- **모델 간 비교**: CNN LSTM과 Convolutional LSTM은 다른 모델들에 비해 상대적으로 저조한 성능을 보였다.

## 🧠 Insights & Discussion

본 연구는 Top-down 방식의 고질적 문제인 '개별 특성 상실'을 Odds 예측과 선형 방정식 풀이라는 수학적 접근으로 해결하려 했다. 특히, 단순한 비율 예측보다 Odds를 활용함으로써 변수 간의 상대적 관계를 더 잘 보존할 수 있음을 시사한다.

실험 결과 ARIMA가 매우 높은 성능을 보인 점은 시뮬레이션 데이터가 ARMA 기반으로 생성되었기 때문이라는 해석이 가능하나, 딥러닝 모델인 Stacked LSTM 역시 유사한 성능을 낸 점은 본 제안 방법론이 다양한 예측 모델과 결합 가능함을 보여준다. 다만, LSTM 계열 모델의 하이퍼파라미터 최적화가 충분히 이루어지지 않았음을 언급하고 있어, 최적화 시 성능 향상의 여지가 있다.

한계점으로는 실제 데이터가 아닌 시뮬레이션 데이터만을 사용했다는 점이며, 음수 값을 0으로 처리하고 재배분하는 휴리스틱한 방식이 실제 데이터에서 어떤 영향을 줄지에 대한 분석이 추가될 필요가 있다.

## 📌 TL;DR

이 논문은 계층적 시계열 예측에서 상위 레벨의 합계 일관성을 유지하면서 하위 레벨의 관계를 보존하기 위해 **Odds 예측 및 선형 연립방정식 기반의 복원 방법론**을 제안한다. 시뮬레이션 실험을 통해 ARIMA 및 Stacked LSTM과 결합했을 때 매우 낮은 RMSPE를 기록하며 유효함을 입증하였으며, 이는 향후 복잡한 계층 구조를 가진 시계열 데이터의 정밀한 예측에 기여할 수 있다.
