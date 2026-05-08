# Grouped Convolutional Neural Networks for Multivariate Time Series

Subin Yi, Janghoon Ju, Man-Ki Yoon, Jaesik Choi (2018)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터 분석에서 발생하는 고차원 입력 데이터의 처리 문제를 해결하고자 한다. 자동 제어, 결함 진단, 이상 탐지 등 다양한 응용 분야에서 다변량 시계열 분석은 필수적이지만, 입력 변수의 차원이 매우 높을 경우 기존의 Convolutional Neural Networks (CNN)를 그대로 적용하기에는 어려움이 있다.

가장 핵심적인 문제는 CNN의 커널(kernel)이 입력 볼륨의 전체 차원에 걸쳐 확장되어야 한다는 점이다. 이는 모델 설계의 난이도를 높일 뿐만 아니라, 학습해야 할 파라미터 수를 급격히 증가시켜 과적합(overfitting)의 위험을 높이고 연산 효율성을 떨어뜨린다. 따라서 본 논문의 목표는 입력 변수 간의 공분산 구조(covariance structure)를 활용하여 입력 볼륨을 그룹으로 분할함으로써, 파라미터 수를 줄이면서도 효율적으로 잠재 특징(latent features)을 학습할 수 있는 Group CNN (G-CNN) 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다변량 시계열의 입력 변수들이 서로 상관관계(correlation)를 가지고 있다는 점에 착안하여, 관련 있는 변수들끼리 그룹화하여 처리하는 것이다. 이를 통해 다음과 같은 이점을 얻을 수 있다.

1. **파라미터 효율성:** 커널이 전체 차원이 아닌 분할된 그룹 내에서만 작동하므로, 모델의 파라미터 수를 획기적으로 줄여 모델을 더 콤팩트하게 만들 수 있다.
2. **강건성(Robustness) 향상:** 상관관계가 높은 신호들의 그룹에 대해 컨볼루션을 수행함으로써 신호 노이즈에 더 강건한 특징 추출이 가능하다.
3. **구조 학습 알고리즘 제공:** 그룹을 나누는 방식을 명시적으로 학습하는 방법(Spectral Clustering 기반)과 암시적으로 학습하는 방법(Clustering Coefficient 기반) 두 가지를 제안하였다.

## 📎 Related Works

기존의 다변량 시계열 처리 방식은 다음과 같은 한계점을 가진다.

- **Fully-connected Networks:** 전체 시퀀스를 한꺼번에 처리해야 하므로 메모리 사용량이 많고 학습 효율이 매우 낮다.
- **RNN 및 LSTM:** 시계열의 전이 함수를 학습하여 시간적 변화를 표현하지만, 일반적으로 마르코프 가정(Markov assumption) 하에 모든 시계열 간에 완전 연결된 네트워크를 가정한다. 이는 고차원 다변량 회귀 문제에서 정밀도가 떨어지는 원인이 된다.
- **Standard CNN:** 공간 도메인에서의 파라미터 공유를 통해 일반화된 특징 추출기 역할을 수행하지만, 앞서 언급했듯이 고차원 입력 시 커널 크기가 너무 커지는 문제가 있다.
- **RCNN (Recurrent Convolutional Neural Network):** 컨볼루션 층을 재귀적으로 구성하여 표현력을 높인 모델로, 최근 시계열 분류에서 좋은 성과를 보였으나 대규모 다변량 시퀀스의 공분산 구조를 명시적으로 모델링하는 방법은 명확히 제시되지 않았다.

## 🛠️ Methodology

본 논문은 입력 변수들을 그룹으로 나누어 처리하는 G-CNN의 두 가지 구조 학습 알고리즘을 제안한다.

### 1. Spectral Clustering을 이용한 명시적 구조 학습 (Explicit Grouping)

입력 변수 $X = [x_1, \dots, x_N]$와 각 변수의 클러스터 멤버십 정보 $C = [c_1, \dots, c_N]$를 함께 입력으로 받는다. 일반적인 CNN과 달리, 그룹화된 컨볼루션 층은 $c_i$를 기반으로 입력 볼륨을 나누어 동일한 클러스터에 속한 변수들에 대해서만 컨볼루션을 수행한다. $k$번째 그룹의 출력 $H^k$는 다음과 같이 정의된다.

$$H^k = \sigma \left( \sum_{i \in \{j|c_j=k\}} W^k \cdot x_i + b^k \right)$$

여기서 $W^k$는 $k$번째 그룹의 가중치 행렬, $b^k$는 편향(bias) 벡터이며, $(\cdot)$은 컨볼루션 연산을 의미한다. 이 방식은 불필요한 파라미터를 제거하여 모델을 가볍게 만든다.

### 2. Clustering Coefficient를 이용한 암시적 구조 학습 (Implicit Grouping)

변수 간의 상관관계를 모델 내에서 직접 학습하기 위해 클러스터링 계수 행렬 $U = [u_{i,j}]$를 도입한다. $u_{i,j}$는 변수 $x_i$가 $j$번째 클러스터에 속하는 비율을 나타내며, $\sum_{j=1}^K u_{i,j} = 1$을 만족한다. $j$번째 클러스터를 대표하는 노드 $h_j$는 다음과 같이 계산된다.

$$h_j = \sigma \left( \sum_{i=1}^N u_{i,j} x_i^T \cdot W_{i,j}^1 + b_j^1 \right)$$

이 모델은 오차 역전파(error backpropagation)를 통해 $W, U, B$를 동시에 최적화한다. 특히 $u_{i,j}$에 대한 기울기(gradient)를 계산하여, 손실을 최소화하는 방향으로 변수가 어떤 그룹에 더 많이 기여할지를 학습한다. 선형 활성화 함수를 가정했을 때, 오차 $\text{Err} = \frac{1}{2}(y-t)^2$에 대한 $u_{i,j}$의 기울기는 다음과 같다.

$$\frac{\partial \text{Err}}{\partial u_{i,j}} = (y-t) \left( K x_i^T \cdot W_{i,j}^1 - \sum_{j=1}^K x_i^T \cdot W_{i,j}^1 \right) (x_i^T \cdot W_{i,j}^1)$$

이 식을 통해 $u_{i,j}$ 값은 해당 클러스터의 손실이 전체 클러스터의 평균 손실보다 작아지는 방향으로 업데이트된다.

### 3. Recurrent Convolutional Layer (RCL)

본 논문은 G-CNN 구조에 RCNN의 개념을 결합하여 G-RCNN을 구성할 수 있다. RCL은 동일한 파라미터를 공유하는 여러 컨볼루션 층의 합성으로 이루어지며, 다음과 같은 재귀적 구조를 가진다.

- 첫 번째 단계: $\sigma(W * x)$
- 이후 단계: $\sigma(W * (x + \text{previous\_output}))$

이는 ResNet의 Skip-connection과 유사한 효과를 주어, Gradient Vanishing 문제를 완화하고 더 깊은 네트워크 학습을 가능하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Groundwater level data (88개 변수), Drone flight data (148개 변수)
- **작업:** 다변량 회귀(Regression). 특정 변수 $x_p$를 타겟으로 설정하고, 나머지 상관관계 변수들의 과거 값들을 이용하여 $x_p(t)$를 예측한다.
- **지표:** Standardized Root Mean Square Error (SRMSE)를 사용한다.
  $$\text{SRMSE} = \frac{\sqrt{\sum_{t=1}^N (t_t - y_t)^2 / N}}{\text{SE}}, \quad \text{SE} = \sqrt{\frac{1}{N} \sum_i (t_i - \bar{t})^2}$$
- **비교 대상:** Linear Regression, Ridge Regression, vanilla CNN, vanilla RCNN

### 주요 결과

- **Groundwater 데이터:** G-CNN 모델들이 vanilla 모델들보다 우수한 성능을 보였다. 특히 Clustering Coefficient를 적용한 RCNN 모델이 $0.754$ SRMSE로 가장 낮은 오차를 기록하여, vanilla RCNN($0.985$) 대비 성능 향상이 뚜렷했다.
- **Drone 데이터:** G-CNN 계열이 전반적으로 vanilla CNN보다 나은 경향을 보였으며, Spectral Clustering을 적용한 RCNN 모델이 $0.438$ SRMSE로 가장 우수한 성능을 나타냈다.

결과적으로, 입력 변수를 그룹화하여 처리하는 방식이 단순한 전체 연결 구조보다 고차원 다변량 시계열 회귀 작업에서 더 정밀한 예측이 가능함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 다변량 시계열 데이터의 특성인 '변수 간 상관관계'를 네트워크 구조(Grouping)에 직접 반영함으로써 모델의 효율성과 성능을 동시에 잡았다. 특히, 모든 변수를 동일한 비중으로 처리하는 대신, 공분산 구조를 이용해 그룹화함으로써 파라미터 수를 줄인 점은 실무적으로 매우 중요하다.

**강점:**

- 고차원 데이터에서 CNN 커널 크기 문제를 해결하여 모델의 복잡도를 낮추었다.
- 명시적(Spectral Clustering) 방법과 암시적(Learning Coefficient) 방법 모두를 제안하여 데이터 특성에 맞는 선택지를 제공하였다.

**한계 및 논의사항:**

- 본 논문에서는 회귀(Regression) 작업에 집중하였으나, 분류(Classification)나 이상 탐지(Anomaly Detection) 작업에서도 동일한 그룹화 전략이 유효할지에 대한 추가 검증이 필요하다.
- 클러스터의 개수 $K$를 설정하는 기준이 명확히 제시되지 않았으며, 이는 하이퍼파라미터 튜닝에 의존해야 하는 한계가 있다.
- Clustering Coefficient 방법에서 $u_{i,j}$의 학습 안정성이 실제 구현 시 어떻게 보장되는지에 대한 상세한 분석이 부족하다.

## 📌 TL;DR

본 논문은 고차원 다변량 시계열 데이터를 처리할 때 CNN 커널의 크기가 지나치게 커지는 문제를 해결하기 위해, 입력 변수를 상관관계에 따라 그룹화하여 처리하는 **Group CNN (G-CNN)**을 제안한다. Spectral Clustering을 통한 명시적 그룹화와 학습 가능한 계수를 통한 암시적 그룹화 두 가지 방식을 제시하였으며, 실험 결과 파라미터 수를 크게 줄이면서도 vanilla CNN/RCNN보다 더 낮은 예측 오차(SRMSE)를 달성하였다. 이 연구는 대규모 센서 데이터와 같이 변수가 많은 시계열 시스템의 효율적인 특징 추출 및 예측 모델 설계에 기여할 가능성이 높다.
