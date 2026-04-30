# Deep Learning

Nicholas G. Polson, Vadim O. Sokolov (2018)

## 🧩 Problem to Solve

본 논문은 고차원 데이터에서 입력-출력 모델(input-output models)을 구축하기 위한 고차원 예측자(high-dimensional predictors)를 생성하는 문제, 즉 고차원 함수 추정(high-dimensional function estimation) 문제를 다룬다. 

고차원 데이터는 일반적으로 '차원의 저주(curse of dimensionality)'로 인해 전통적인 통계적 방법으로는 효율적인 예측이 어렵다. 특히 입력 변수 간의 복잡한 비선형 상호작용을 모델링하고, 과적합(overfitting)을 방지하면서도 데이터의 핵심 특징을 추출하는 것이 주요한 난제이다. 따라서 본 논문의 목표는 딥러닝(Deep Learning, DL)을 고차원 데이터 축소 기술 및 예측 도구로서 정의하고, 이를 모델링 및 알고리즘 관점에서 학술적으로 분석하여 리뷰하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥러닝을 단순한 블랙박스 모델이 아닌, 통계적 관점에서의 '고차원 데이터 축소 기술' 및 '계층적 비선형 요인 모델(hierarchical nonlinear factor model)'로 재정의했다는 점이다. 

중심적인 아이디어는 딥러닝의 계층 구조가 데이터를 단계적으로 추상화하여 고차원 입력을 저차원의 잠재 특징(latent features)으로 변환하며, 이를 통해 복잡한 비선형 함수를 효율적으로 근사할 수 있다는 것이다. 또한, 딥러닝 모델을 '적층형 일반화 선형 모델(Stacked Generalized Linear Models, GLM)'로 해석함으로써 전통적인 통계학의 GLM과 현대적인 딥러닝 아키텍처 사이의 이론적 연결 고리를 제시한다.

## 📎 Related Works

논문은 통계적 모델링과 머신러닝 모델링의 두 문화(Two Cultures) 사이의 차이를 언급하며 논의를 시작한다. 

1. **전통적 통계 모델링**: 데이터가 특정 확률적 데이터 모델(stochastic data model)에 의해 생성되었다고 가정하며, 추론(inference)에 집중한다.
2. **알고리즘 모델링(머신러닝)**: 데이터 생성 기제를 알 수 없다고 가정하며, 예측(prediction)의 정확도에 집중한다.

기존의 얕은 학습기(shallow learners)인 주성분 분석(PCA), 선형 판별 분석(LDA), 투영 추구 회귀(PPR) 등은 데이터를 저차원으로 축소하지만, 이 과정이 출력 변수 $Y$와 독립적으로 이루어지는 경우가 많아 예측에 중요한 정보를 손실할 수 있다는 한계가 있다. 반면, 딥러닝은 $Y$와 $X$의 관계를 함께 고려하여 저차원 특징을 학습함으로써 이러한 한계를 극복한다.

## 🛠️ Methodology

### 1. 네트워크 아키텍처 및 수식 모델
딥러닝 예측자는 $L$개의 계층을 가진 합성 함수(composite map)로 정의된다. 각 계층 $l$에서의 활성화 규칙(activation rule)은 다음과 같은 준-아핀(semi-affine) 형태로 표현된다.

$$f_{W,b}^l := f^l \left( \sum_{j=1}^{N_l} W_{lj} X_j + b_l \right) = f^l (W^l X^l + b^l)$$

여기서 $f^l$은 단변량 활성화 함수이며, $W^l$은 가중치 행렬, $b^l$은 편향(bias)이다. 전체 예측 모델 $\hat{Y}(X)$는 다음과 같은 합성 함수로 나타낼 수 있다.

$$\hat{Y}(X) := F(X) = (f_{W,b}^1 \circ \dots \circ f_{W,b}^L)(X)$$

구체적인 계층 구조는 다음과 같이 전개된다.
- $Z^{(1)} = f^{(1)}(W^{(0)}X + b^{(0)})$
- $Z^{(2)} = f^{(2)}(W^{(1)}Z^{(1)} + b^{(1)})$
- $\dots$
- $\hat{Y}(X) = W^{(L)}Z^{(L)} + b^{(L)}$

### 2. 특수 구조 및 모델
- **Autoencoder**: 입력 $X$를 동일한 출력 $Y$로 복원하도록 학습하는 구조로, 병목(bottleneck) 구조를 통해 $X$의 효율적인 표현을 학습한다.
- **GAN (Generative Adversarial Networks)**: 생성자(Generator, $G$)와 판별자(Discriminator, $D$)가 서로 대립하며 학습하는 구조이다. 목적 함수는 다음과 같은 Min-Max 게임으로 정의된다.
$$\min_{\theta_G} \max_{\theta_D} V(D,G) = \mathbb{E}_{x \sim p(x)}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

### 3. 학습 및 최적화 절차
학습의 목표는 손실 함수 $l(Y, \hat{Y})$와 정규화 항 $\phi(W,b)$를 결합하여 가중치와 편향을 최적화하는 것이다.

$$\arg \min_{W,b} \sum_{i=1}^T l(Y_i, \hat{Y}(X_i)) + \lambda \phi(W,b)$$

- **Backpropagation**: 확률적 경사 하강법(SGD)의 일종으로, 출력층의 오차를 입력층 방향으로 전파하며 가중치를 업데이트한다.
- **Approximate Inference**: 변분 추론(Variational Inference)을 통해 사후 분포 $p(\theta|D)$를 근사 분포 $q(\theta|D, \phi)$로 대체하며, 증거 하한(Evidence Lower Bound, ELBO)을 최대화하는 방식으로 학습한다. 이때 $\theta$를 결정론적 함수로 표현하는 Reparameterization trick을 사용하여 몬테카를로 기울기를 계산한다.
- **Dropout**: 과적합 방지를 위해 학습 과정에서 무작위로 입력 차원을 제거하는 기법이다. 이는 수학적으로 Bayesian ridge regression과 유사한 효과를 가진다.
- **Batch Normalization**: 각 층의 활성화 값을 정규화하여 내부 공변량 변화(internal covariate shift)를 줄이고 학습 속도를 높인다.
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

## 📊 Results

본 논문은 특정 실험 데이터셋에 대한 결과보다는, 딥러닝이 적용된 다양한 실제 사례의 성과를 통해 그 유효성을 입증한다.

- **언어 및 음성**: Google Neural Machine Translation의 번역 정확도 향상, Google WaveNet의 텍스트-음성 변환 성능 개선.
- **이미지 처리**: CNN을 이용한 흉부 X-ray 기반 폐렴 검출(전문의 수준의 정확도), 피부암 진단.
- **기타 과학 분야**: NASA의 케플러 망원경 데이터를 이용한 새로운 행성 발견, 암 치료를 위한 새로운 분자 구조 생성(Adversarial Auto-encoder 사용).
- **금융 및 시공간 분석**: 전통적인 통계 학습 기법 대비 우수한 예측 성능을 보였다.

## 🧠 Insights & Discussion

### 이론적 강점 및 차별점
딥러닝은 Kolmogorov-Arnold Representation Theorem에 근거하여 모든 다변수 함수를 단변수 함수들의 합성으로 표현할 수 있음을 시사한다. 특히, 타겟 함수가 'G-함수(G-function, 국소적 구조를 가진 함수)' 형태일 때, 딥러닝은 얕은 네트워크보다 훨씬 적은 파라미터로도 함수를 근사할 수 있어 차원의 저주를 피할 수 있다.

### 한계 및 미해결 과제
논문은 여전히 해결되지 않은 두 가지 핵심 질문을 던진다.
1. **아키텍처 선택 문제**: 주어진 문제에 최적인 네트워크 구조(깊이, 너비)를 결정하는 체계적인 방법이 부족하며, 여전히 시행착오(trial-and-error) 방식에 의존하고 있다.
2. **일반화(Generalization)의 원리**: 딥러닝 모델이 매우 많은 파라미터를 가졌음에도 불구하고 왜 과적합되지 않고 새로운 데이터에 대해 잘 작동하는지에 대한 이론적 설명이 부족하다. 일부 연구는 딥러닝이 화이트 노이즈까지 학습할 수 있음을 보여, 전통적인 정규화 이론만으로는 설명이 불가능함을 시사한다.

## 📌 TL;DR

본 논문은 딥러닝을 고차원 데이터의 효율적인 축소 및 예측을 위한 '계층적 비선형 모델'로 분석한 리뷰 논문이다. 딥러닝의 수학적 구조를 적층형 GLM으로 해석하고, GAN, Autoencoder와 같은 최신 구조와 Dropout, Batch Normalization 등의 최적화 기법을 체계적으로 정리하였다. 특히 딥러닝이 특정 구조의 함수에서 차원의 저주를 극복할 수 있음을 이론적으로 설명하며, 향후 아키텍처 자동 선택 및 일반화 원리 규명이 중요한 연구 과제임을 제시한다.