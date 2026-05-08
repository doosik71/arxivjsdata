# Conditional normalization in time series analysis

Puwasala Gamakumara, Edgar Santos-Fernandez, Priyanga Dilini Talagala, Rob J Hyndman, Kerrie Mengersen, Catherine Leigh (2023)

## 🧩 Problem to Solve

본 논문은 시계열 데이터 분석에서 외부 변수(Covariates)로 인해 발생하는 변동성을 효과적으로 제어하기 위한 정규화 방법론을 제안한다.

일반적으로 시계열 데이터는 다른 관련 변수들의 영향을 받아 변동하는 특성이 있으며, 이러한 외부 변수의 효과를 제어하는 것은 시계열 모델링 및 분석의 정확도를 높이는 데 중요하다. 기존의 Min-Max Scaling이나 Z-score Normalization과 같은 표준 정규화 방식은 데이터가 Stationary(정상성)를 가진다고 가정하며, 정규화 상수가 시간이 지남에 따라 변할 수 있는 비정상성(Non-stationary) 시계열 데이터에는 적용하기 어렵다. 또한, 최근 제안된 Sliding-window normalization 역시 외부 변수가 시계열의 변동에 미치는 영향을 고려하지 못한다는 한계가 있다.

따라서 본 연구의 목표는 외부 공변량 세트에 조건부로 시계열 데이터를 정규화하는 **Conditional Normalization** 방법론을 구축하여, 공변량에 의해 유도된 조건부 변동성을 제거하고 데이터의 본질적인 특성을 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Z-score normalization의 확장판으로, 정규화에 사용되는 평균과 분산을 상수가 아닌 **공변량의 함수**로 정의하는 것이다.

이를 위해 Generalized Additive Models (GAMs)를 도입하여 공변량과 시계열 변수 간의 비선형 관계를 유연하게 모델링한다. 단순하게 조건부 평균만을 빼는 기존의 방식과 달리, 본 논문은 **조건부 평균(Conditional Mean)과 조건부 분산(Conditional Variance)을 모두 모델링**하여 정규화를 수행함으로써 훨씬 일반적이고 강력한 정규화 프레임워크를 제공한다.

## 📎 Related Works

논문에서는 기존의 정규화 방식들을 다음과 같이 언급하며 차별점을 제시한다.

- **표준 정규화 (Min-Max, Z-score):** 정상성 과정을 가정하며, 미래에 정규화 상수가 변할 수 있어 비정상성 시계열에 부적합하다.
- **Sliding-window Normalization:** 비정상성 데이터에 대응하기 위해 윈도우 내에서 정규화를 수행하지만, 외부 변수의 영향을 반영하지 못한다.
- **조건부 평균 차감 (Conditional Mean Subtraction):** 일부 연구에서 조건부 평균을 빼는 방식을 사용하지만, 이는 분산의 변화를 고려하지 않으며 비선형 관계 모델링에 제한적이다.

본 연구는 GAM을 통해 평균과 분산을 모두 비선형 함수로 모델링함으로써, 공변량에 따른 데이터의 scale 변화까지 정규화 과정에 포함시켰다는 점에서 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 1. Conditional Normalization 과정

시계열 변수를 $y_t$, 공변량 벡터를 $z_t = (z_{1,t}, \dots, z_{p,t})$라고 할 때, 정규화된 변수 $y^*_t$는 다음과 같이 정의된다.

$$y^*_t = \frac{y_t - \hat{m}(z_t)}{\sqrt{\hat{v}(z_t)}}$$

여기서 $\hat{m}(z_t)$는 추정된 조건부 평균이고, $\hat{v}(z_t)$는 추정된 조건부 분산이다. 이들의 추정 과정은 두 단계의 GAM 모델링을 통해 이루어진다.

**단계 1: 조건부 평균 $\hat{m}(z_t)$ 추정**
다음의 GAM 모델을 적합시킨다.
$$y_t = \alpha_0 + \sum_{i=1}^{p} f_i(z_{i,t}) + \epsilon_t$$
여기서 $f_i(\cdot)$는 Smooth function이며, $\hat{m}(z_t) = \hat{\alpha}_0 + \sum_{i=1}^{p} \hat{f}_i(z_{i,t})$가 된다.

**단계 2: 조건부 분산 $\hat{v}(z_t)$ 추정**
평균 모델에서 얻은 오차의 제곱 $[y_t - \hat{m}(z_t)]^2$을 종속변수로 하여 Gamma 분포를 따르는 GAM 모델을 적합시킨다.
$$\log(v(z_t)) = \beta_0 + \sum_{i=1}^{p} g_i(z_{i,t})$$
최종적인 분산 추정치는 $\hat{v}(z_t) = \exp(\hat{\beta}_0 + \sum_{i=1}^{p} \hat{g}_i(z_{i,t}))$가 된다.

### 2. 활용 방안

**결측치 대체 (Imputation of missing values):**
정규화된 시리즈 $y^*_t$가 ARIMA 프로세스를 따른다고 가정하면, Kalman smoother를 통해 $\hat{y}^*_t$를 구한 후, 이를 다시 역정규화하여 원래 스케일의 결측치를 대체한다.
$$y_t = \hat{y}^*_t \sqrt{\hat{v}(z_t)} + \hat{m}(z_t)$$

**조건부 자기상관함수 (Conditional ACF):**
시차 $k$에서의 조건부 ACF를 다음과 같이 정의하고 GAM으로 추정한다.
$$r_k(z_t) = E[y^*_t y^*_{t-k} | z_t]$$
이때 $\eta(r_k(z_t))$ 형태의 link function을 사용하여 $[-1, 1]$ 범위를 실수 전체로 매핑하여 모델링한다.

**조건부 상호상관함수 (Conditional CCF):**
두 변수 $x_t$와 $y_t$를 각각 조건부 정규화한 후, 시차 $k$에서의 상호상관을 다음과 같이 추정한다.
$$c_k(z_t) = E[y^*_{t+k} x^*_t | z_t]$$

## 📊 Results

### 실험 1: 하천 수온 결측치 대체 (Stream temperature imputation)

- **데이터셋:** 미국 북서부 Boise River의 42개 지점에서 측정된 일일 평균 수온 데이터.
- **설정:** 공변량으로 기온(Air temperature), 하천 경사, 고도, 배수 면적 및 계절성을 반영한 Fourier terms를 사용하였다. 모델 추정에는 Stan의 Hamiltonian Monte Carlo(HMC)를 이용하였다.
- **결과:**
  - AR(8) 모델이 RMSPE 기준으로 최적의 성능을 보였다.
  - 예측값의 95% Highest Posterior Density (HPD) 구간 내에 실제 값이 포함될 확률이 $0.946$으로 나타나 매우 높은 정확도로 결측치가 대체되었음을 확인하였다.
  - 여름철의 표준편차가 겨울철보다 약 3배 높다는 조건부 분산의 특성을 성공적으로 포착하였다.

### 실험 2: 하천 유량의 지체 시간 예측 (Predicting lag time on river flow)

- **데이터셋:** 미국 텍사스 Pringle Creek의 상류 및 하류 센서에서 측정된 탁도(Turbidity) 데이터.
- **설정:** 상류의 수위(Water level)와 수온(Temperature)을 공변량 $z_t$로 설정하여 조건부 CCF를 계산하였다. 지체 시간 $d_t$는 상호상관이 최대가 되는 시차 $k$로 정의하였다.
$$\hat{d}_t(z_t) = \text{argmax}_k \hat{c}_k(z_t)$$
- **결과:**
  - **수위의 영향:** 상류 수위가 높아지면 지체 시간이 감소하는 음(-)의 관계가 명확히 나타났다. 이는 수위 상승이 유량 증가로 이어져 물이 더 빠르게 하류로 이동한다는 물리적 특성과 일치한다.
  - **수온의 영향:** 수온이 $10^\circ\text{C}$ 미만일 때는 수온 상승 시 지체 시간이 증가하는 경향을 보였으나, 그 이상에서는 낮게 유지되었다.
  - **평가:** 추정된 $d_t$를 적용했을 때의 상호상관값이 다른 모든 시차 $k$에서의 값보다 약 96%의 확률로 더 높게 나타나, 제안 방법론이 유효함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 데이터 전처리를 넘어, 공변량에 의존적인 데이터의 통계적 특성(평균, 분산)을 모델링함으로써 시계열 분석의 정밀도를 높였다는 점에서 큰 강점을 가진다. 특히, 물리적 의미를 가진 하천 데이터에 적용하여 수위와 유속의 관계를 데이터 기반(Data-driven)으로 증명해낸 점이 인상적이다.

**한계 및 논의사항:**

- **데이터 의존성:** 실험 2에서 수위가 매우 높은 특정 이벤트 구간에서 지체 시간이 갑자기 증가하는 패턴이 발견되었으나, 이는 단일 이벤트에 의한 결과로 정확한 원인 분석에는 한계가 있었다.
- **모델 확장성:** 본 연구는 단변량 시계열을 대상으로 하였으나, 실제 환경에서는 여러 변수가 복합적으로 작용하므로 Vector Autoregressions (VAR)와 같은 다변량 모델로의 확장이 필요하다.
- **공간적 의존성:** 여러 지점에서 측정된 데이터의 경우 지점 간의 공간적 상관관계(Spatial dependence)를 고려한 모델링이 추가될 필요가 있다.

## 📌 TL;DR

본 논문은 공변량(Covariates)에 따라 변화하는 시계열의 평균과 분산을 GAM(Generalized Additive Models)으로 모델링하여 정규화하는 **Conditional Normalization** 기법을 제안한다. 이 방법론을 통해 외부 변수로 인한 변동성을 제거함으로써, 비정상성 시계열 데이터의 결측치 대체 정확도를 높이고, 환경 조건에 따라 변하는 하천의 유량 지체 시간(Lag time)을 정밀하게 예측할 수 있음을 보였다. 이는 값비싼 현장 실험(예: 염분 추적자 실험)을 대체할 수 있는 데이터 기반의 분석 가능성을 제시한다.
