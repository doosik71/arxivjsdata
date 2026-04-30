# Deep Mixed Effect Model using Gaussian Processes: A Personalized and Reliable Prediction for Healthcare

Ingyo Chung, Saehoon Kim, Juho Lee, Kwang Joon Kim, Sung Ju Hwang, Eunho Yang (2020)

## 🧩 Problem to Solve

본 논문은 전자 건강 기록(Electronic Health Records, EHR)과 같은 시계열 의료 데이터에서 **개인 맞춤형(Personalized)이면서도 신뢰할 수 있는(Reliable) 예측 모델**을 구축하는 것을 목표로 한다. 

의료 데이터는 환자 개개인의 생물학적, 인구통계학적 특성 및 환경적 요인으로 인해 매우 큰 가변성과 이질성(Heterogeneity)을 가진다. 기존의 접근 방식은 크게 두 가지로 나뉘지만 각각 명확한 한계가 존재한다. 
첫째, RNN과 같은 **인구 기반(Population-based) 모델**은 모든 환자에 대해 하나의 모델을 학습하므로 환자 간의 이질성을 반영하지 못해 개인화된 예측에 실패하고 평균적인 예측값으로 편향되는 경향이 있다. 
둘째, **개별 모델(Separate models/Personalized GP)**은 환자별 특성을 잘 포착할 수 있으나, 환자 한 명당 가용한 데이터 포인트(방문 횟수)가 매우 적어 글로벌 트렌드를 학습하지 못하며 일반화 성능이 떨어진다. 

따라서 본 연구는 대규모 환자 집단에서 나타나는 공통적인 글로벌 트렌드와 개별 환자의 고유한 가변성을 동시에 모델링하여, 데이터가 적은 상황에서도 신뢰할 수 있는 개인 맞춤형 예측을 수행하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep Neural Network(DNN)의 표현력**과 **Gaussian Process(GP)의 확률적 유연성**을 결합한 **Deep Mixed Effect Model (DME-GP)** 프레임워크를 제안하는 것이다.

핵심 직관은 함수 $f^{(i)}$를 글로벌 성분 $g(\cdot)$와 환자별 로컬 성분 $l^{(i)}(\cdot)$의 합으로 분해하는 Mixed Effect Model 구조를 채택하는 것이다. 여기서 $g(\cdot)$는 많은 수의 환자로부터 복잡한 글로벌 트렌드를 학습하는 DNN(예: RNN)으로 구성하고, $l^{(i)}(\cdot)$는 소수의 데이터로도 개별 환자의 특성을 확률적으로 모델링할 수 있는 GP로 구성한다. 이를 통해 모델은 글로벌 지식을 공유하면서도 개별 환자에게 최적화된 예측을 제공하며, GP의 특성을 이용하여 예측의 불확실성(Uncertainty)을 함께 산출함으로써 의료 분야에서 필수적인 신뢰성을 확보한다.

## 📎 Related Works

논문에서는 EHR 모델링과 관련된 기존 연구들을 다음과 같이 분류하고 한계를 지적한다.

1.  **Multiple Gaussian Processes**: 환자별로 독립적인 GP를 사용하는 방식이다. 데이터 결손 처리 등에 유용하지만, 환자들 사이의 공통된 글로벌 트렌드를 활용하지 못한다는 단점이 있다.
2.  **Multi-task Gaussian Process (MTGP)**: 여러 작업(환자) 간의 상관관계를 공유 공분산 행렬(Shared covariance matrix)을 통해 학습한다. 그러나 전체 데이터 포인트에 대한 공분산 행렬의 역행렬을 계산해야 하므로, 계산 복잡도가 $O(P^3 T^3)$에 달해 대규모 EHR 데이터셋에 적용하기에는 연산 비용이 너무 크다($P$: 환자 수, $T$: 방문 횟수).
3.  **Deep Learning models (RNN/LSTM/GRU)**: 시계열 EHR 데이터에서 뛰어난 성능을 보이지만, 결정론적(Deterministic)인 특성 때문에 예측의 확신도나 불확실성을 제공하지 못하며, 환자 간의 극심한 이질성을 처리하는 데 한계가 있다.

DME-GP는 글로벌 함수로 DNN을, 로컬 함수로 개별 GP를 사용하여 MTGP의 계산 복잡도 문제를 해결(선형 확장 가능)함과 동시에 RNN의 불확실성 부재 문제를 극복한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
DME-GP는 i-번째 환자의 함수 $f^{(i)}$를 다음과 같이 정의한다.
$$f^{(i)}(x_t) = g(x_t) + l^{(i)}(x_t)$$
여기서 $g(\cdot)$는 모든 환자가 공유하는 **Fixed Effect(글로벌 성분)**이며, $l^{(i)}(\cdot)$는 각 환자 고유의 **Random Effect(로컬 성분)**이다.

### 주요 구성 요소
1.  **Global Component ($g(\cdot)$)**: 
    - MLP나 RNN과 같은 심층 신경망을 사용하여 대규모 데이터로부터 복잡한 글로벌 트렌드를 포착한다. 
    - 입력 $x_t$를 임베딩 함수 $\phi(\cdot|v)$를 통해 $h_t$로 변환하고, 이를 다시 평균 함수 $\mu(h_t|w)$에 통과시켜 값을 산출한다.
2.  **Individual Component ($l^{(i)}(\cdot)$)**: 
    - 환자별로 독립적인 GP를 할당한다. 
    - 비매개변수(Non-parametric) 모델인 GP를 사용하여 매우 적은 방문 데이터만으로도 개별 환자의 신호를 안정적으로 추정한다.

### 최종 복합 모델 (Composite Model)
결과적으로 i-번째 환자의 예측 모델은 다음과 같은 GP로 표현된다.
$$f^{(i)}(x_t) \sim GP(\mu(h_t|w), k^{(i)}(h_t, h'_t|\theta_i))$$
여기서 $\mu(\cdot|w)$는 공유된 심층 신경망 기반의 평균 함수이며, $k^{(i)}(\cdot, \cdot|\theta_i)$는 환자 $i$의 개별 커널 함수이다.

### 학습 절차 및 손실 함수
학습 목표는 모든 환자의 주변 로그 가능도(Marginal Log-likelihood)를 최대화하는 글로벌 파라미터 $\{w, v\}$와 개별 파라미터 $\{\theta_i\}$를 찾는 것이다.
$$\theta^*, w^*, v^* = \arg\max_{\theta, w, v} \sum_{i=1}^{P} \log p(y_i | X_i, \theta, w, v)$$

가우시안 가능도를 가정할 때, 개별 환자의 로그 가능도 $L_i$는 다음과 같다.
$$L_i = -\frac{1}{2} (y_i - \mu_i)^T K^{(i)-1} (y_i - \mu_i) - \frac{1}{2} \log |K^{(i)}| - \frac{T_i}{2} \log 2\pi$$
여기서 $\mu_i$는 DNN의 출력 벡터이고, $K^{(i)}$는 GP의 공분산 행렬이다. 학습은 Stochastic Gradient Ascent를 사용하여 글로벌 파라미터와 로컬 파라미터를 교대로 업데이트하는 방식으로 진행된다.

### 추론 절차 (Inference)
새로운 환자 $j$가 등장했을 때:
1.  글로벌 파라미터 $\{w, v\}$는 고정한다.
2.  해당 환자의 데이터를 사용하여 개별 파라미터 $\theta_j$를 최적화한다.
3.  표준 GP 추론 식을 통해 예측값 $\bar{y}_t$와 불확실성 $\sigma_t^2$를 산출한다.
$$\bar{y}_t = \mu(h_t) + k_t^T K^{-1} (y - \mu(H))$$
$$\sigma_t^2 = k(h_t, h_t) - k_t^T K^{-1} k_t$$

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - Vital-Sign (Physionet 2012): 865명 환자의 심박수(HR) 시계열 데이터 (회귀 작업).
    - Medical Checkup (NHIS): 32,927명 환자의 건강검진 기록 기반 12가지 질병 위험 예측 (분류 작업).
- **비교 대상 (Baselines)**: Linear Model, MLP, RNN(LSTM), RETAIN, MTGP-RNN, MAML.
- **측정 지표**: 회귀 작업은 RMSE, 분류 작업은 AUC를 사용한다.

### 주요 결과
1.  **심박수 예측 (Vital-Sign Analysis)**: 
    - DME-GP의 RMSE는 $0.150$으로, MLP($0.194$)와 p-GPs($0.243$)보다 우수한 성능을 보였다. 
    - 특히 p-GPs는 글로벌 트렌드 부재로 저평가(underestimated)하는 경향이 있고, MLP는 단순 추종(follower) 경향을 보였으나, DME-GP는 두 성분을 모두 활용해 정확한 예측을 수행했다.
2.  **질병 위험 예측 (Disease Risk Prediction)**: 
    - 12가지 질병에 대해 DME-GP (특히 RNN 기반)가 대부분의 baseline보다 높은 AUC를 기록했다.
    - 평균 AUC 결과에서 DME-GP가 가장 우수했으며, 이는 환자 간의 이질성을 모델링하는 것이 의료 데이터 예측에 핵심적임을 시사한다.
3.  **신뢰성 연구 (Reliability Study)**: 
    - 과거 기록에는 음성이었으나 예측 시점에 양성으로 변하는 'Hard Patient'와 일반 'Easy Patient'를 구분하여 분석했다.
    - DME-GP는 두 그룹 간의 확신도 차이를 명확하게 구분했으며, 확신이 높은 경우 거의 완벽한 예측 성능을 보였다. 이는 의료진이 모델의 예측 결과를 신뢰하고 개입 여부를 결정하는 데 중요한 근거가 된다.

## 🧠 Insights & Discussion

### 강점
DME-GP는 DNN의 확장성과 GP의 개인화 및 확률적 특성을 성공적으로 결합하였다. 특히 기존 MTGP가 가졌던 $O(P^3 T^3)$의 계산 복잡도를 $O(PT^3)$로 줄여, 환자 수 $P$에 대해 선형적으로 확장 가능한 구조를 만들면서도 성능을 유지했다는 점이 기술적으로 매우 뛰어나다. 또한, 단순한 예측값뿐만 아니라 불확실성을 제공함으로써 Safety-critical한 의료 환경에 적합한 설계를 갖추었다.

### 한계 및 논의사항
본 논문은 수식의 명확성을 위해 가우시안 가능도를 중심으로 설명하였으나, 실제 분류 문제에서는 시그모이드 함수 등을 사용한 변분 근사(Variational Approximation)가 필요함을 언급하고 있다. 
또한, 부록에서 제시된 Mixture of Experts (MoE) 실험을 통해 환자들을 여러 코호트(Cohort)로 나누어 글로벌 평균 함수를 구성했을 때 일부 질병(부정맥 등)에서 성능 향상이 있었음을 보여준다. 이는 단순한 하나의 글로벌 트렌드보다, 환자 그룹별로 세분화된 트렌드를 학습하는 것이 더 효과적일 수 있음을 시사하며 향후 연구 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 의료 데이터의 극심한 환자 간 이질성 문제를 해결하기 위해 **심층 신경망(DNN) 기반의 글로벌 트렌드 학습**과 **가우시안 프로세스(GP) 기반의 개인별 가변성 모델링**을 결합한 **DME-GP**를 제안한다. 이 모델은 계산 효율성을 확보하면서도 개인 맞춤형 예측과 예측 불확실성 산출이 가능하며, 실제 EHR 데이터셋의 회귀 및 분류 작업에서 기존 SOTA 모델들을 상회하는 성능과 높은 신뢰성을 입증하였다. 이는 정밀 의료(Precision Medicine) 구현을 위한 실무적인 프레임워크로 활용될 가능성이 높다.