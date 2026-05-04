# Trainable Activation Function in Image Classification

Zhaohe Liao (2020)

## 🧩 Problem to Solve

본 논문은 심층 신경망(Deep Neural Networks)에서 활성화 함수(Activation Function)가 인간에 의해 수동으로 지정되며, 학습 과정 동안 변경되지 않고 고정된다는 점에 주목한다. 저자는 신경망의 조합 함수(Combination Function, 예: Convolution)에 대해서는 복잡한 파라미터 학습이 이루어짐에도 불구하고, 정보를 전달하는 핵심 단계인 활성화 함수는 매우 단순하고 불변한다는 점을 지적한다.

데이터의 특성은 매우 복잡하기 때문에, 인간의 사전 지식만으로 최적의 활성화 함수를 결정하는 것은 어렵다. 따라서 본 연구의 목표는 활성화 함수 자체를 학습 가능한 형태로 만들어, 네트워크가 주어진 데이터로부터 최적의 활성화 함수를 직접 추론하도록 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 활성화 함수를 고정된 형태가 아닌, 파라미터화된 **급수(Series)** 또는 **선형 결합(Linear Combination)** 형태로 설계하여 역전파(Backpropagation)를 통해 최적화하는 것이다. 이를 통해 활성화 함수가 학습 데이터에 맞춰 연속적으로 변할 수 있게 하며, 결과적으로 모델의 표현력(Expressiveness)과 일반화 능력(Generalization ability)을 향상시키고자 한다.

## 📎 Related Works

논문은 M-P 뉴런 모델(McCulloch and Pitts, 1943)을 언급하며, 현대의 신경망 구조가 기본적으로 정보를 모으는 조합 함수와 이를 전달하는 활성화 함수의 결합으로 이루어져 있음을 설명한다. ResNet이나 R-CNN과 같은 유명한 모델들 역시 활성화 함수를 고정된 상태로 사용하며, 학습은 주로 조합 함수의 파라미터를 변경하는 방식으로 진행된다.

기존 방식의 한계는 활성화 함수를 변경(예: ReLU에서 Sigmoid로 교체)하기 위해서는 사람이 직접 임계값을 결정하거나 수동으로 실험해야 한다는 점이다. 하지만 신경망은 기본적으로 설명 가능성(Explainability)이 낮기 때문에, 어떤 시점에 어떤 활성화 함수가 더 효율적인지 판단하는 것은 매우 어려운 작업이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 연구에서는 CIFAR-10 데이터셋을 사용하여 이미지 분류 작업을 수행하며, 필터 수에 따라 세 가지 크기(Small, Middle, Large)의 단순 CNN 구조를 사용한다.

- **Small**: 필터 수 $[16, 32, 32]$
- **Middle**: 필터 수 $[32, 64, 64]$
- **Large**: 필터 수 $[48, 96, 96]$

각 모델 크기별로 ReLU(기준), Fourier series, Linear combination 세 가지 활성화 함수를 적용하여 총 9개의 네트워크를 비교 분석한다.

### 2. Fourier Series Simulated Activation (Fourier-CNN)

저자는 푸리에 급수가 적절한 조건(Dirichlet 조건)을 만족하는 모든 함수를 표현할 수 있다는 점에 착안하여, 활성화 함수를 다음과 같이 정의한다.

$$act(x) = A + \sum_{n=1}^{\infty} (a_n \cos(n\omega x) + b_n \sin(n\omega x))$$

여기서 $A, \omega, a_n, b_n$은 모두 학습 가능한 파라미터이다. 실제 실험에서는 계산 복잡도를 줄이기 위해 급수의 차수 $n$을 5로 고정하였다. 이 함수는 미분 가능하므로 다음과 같은 기울기(Gradient)를 통해 경사 하강법으로 학습할 수 있다.

- $\frac{\partial act(x)}{\partial A} = 1$
- $\frac{\partial act(x)}{\partial a_n} = -n\omega \sin(n\omega x)$
- $\frac{\partial act(x)}{\partial b_n} = n\omega \cos(n\omega x)$
- $\frac{\partial act(x)}{\partial \omega} = \sum_{n=1}^{\infty} nx(a_n \cos(n\omega x) - b_n \sin(n\omega x))$

### 3. Linear Combination of Activation Functions (LC-CNN)

또 다른 방법은 여러 후보 활성화 함수들의 가중 평균을 사용하는 것이다.

$$act(x) = \frac{\sum_{i=1}^{n} w_i act_i(x)}{\sum_{i=1}^{n} w_i}$$

본 실험에서는 $act_i(x)$로 $\text{ReLU, Sigmoid, tanh, linear}$ 네 가지 함수를 사용하였다. 가중치 $w_i$가 one-hot 벡터가 되면 특정 하나의 활성화 함수로 수렴하므로, 이론적으로는 기존의 단일 활성화 함수보다 성능이 낮아질 가능성이 없음을 보장한다.

### 4. 학습 절차

- **손실 함수**: 교차 엔트로피(Cross Entropy)를 사용한다.
- **최적화 알고리즘**: RMSProp을 사용하며, 에폭(Epoch) 구간에 따라 학습률(Learning Rate)을 $0.001 \rightarrow 0.0001 \rightarrow 0.00001 \rightarrow 0.000001$로 단계적으로 감소시킨다.
- **데이터**: 데이터 증강(Data Augmentation) 없이 CIFAR-10 원본 데이터를 사용한다.

## 📊 Results

### 1. 실험 설정 및 지표

- **데이터셋**: CIFAR-10 (60,000장, 10개 클래스)
- **평가 지표**: 학습 및 검증 데이터셋에 대한 정확도(Accuracy)와 손실(Loss)

### 2. 주요 결과

실험 결과, 학습 가능한 활성화 함수를 사용한 모델이 표준 ReLU 모델보다 전반적으로 더 높은 정확도와 낮은 손실을 기록하였다.

- **네트워크 크기별 경향**:
  - **Small 모델**: Fourier-CNN이 가장 높은 성능 향상을 보였으며, 일반화 성능이 크게 개선되었다.
  - **Large 모델**: LC-CNN이 가장 좋은 성능을 보였으며, 특히 검증 데이터셋에서의 정확도가 표준 CNN보다 높았다.
- **정량적 향상**: Table 2에 따르면, Small 모델에서 Fourier-CNN은 검증 정확도를 약 $5.09\%$ 포인트 향상시켰다.
- **학습 특성**: 모델의 크기가 커질수록 표준 CNN은 학습 데이터에 과적합(Overfitting)되는 경향이 강해져 학습 정확도는 높으나 검증 정확도가 낮아지는 반면, 학습 가능한 활성화 함수를 사용한 모델은 검증 정확도가 더 높게 유지되는 일반화 능력을 보였다.

## 🧠 Insights & Discussion

### 1. 표현력과 일반화 능력의 상관관계

저자는 모델 크기에 따라 성능 향상의 원인이 다르다고 분석한다.

- **Small 모델**: 파라미터 수가 적어 Underfitting이 발생하는 단계이므로, 학습 가능한 활성화 함수가 모델의 **표현력(Expressiveness)**을 직접적으로 높여 성능을 향상시킨다.
- **Middle/Large 모델**: 이미 충분한 파라미터를 가지고 있어 표현력보다는 **일반화 능력(Generalization ability)**이 병목 구간이 된다. 학습 가능한 활성화 함수가 데이터에 최적화된 비선형성을 학습함으로써 과적합을 방지하고 일반화 성능을 높인 것으로 해석된다.

### 2. 한계 및 비판적 해석

- **구조적 한계**: 본 실험은 네트워크의 깊이(Depth)를 변경하지 않고 레이어의 필터 수(Width)만 변경하여 실험하였다. 따라서 최신 딥러닝 모델의 핵심인 '깊은 구조'에서의 효과는 검증되지 않았다.
- **SOTA 모델 적용 부재**: 단순한 CNN 구조에서만 실험이 진행되었으므로, ResNet이나 EfficientNet과 같은 최신 SOTA 모델에 적용했을 때도 동일한 효과가 있을지는 불분명하다.
- **최적화 문제**: 푸리에 급수 기반의 활성화 함수가 지역 최적점(Local Minima)에 빠져 단순 함수보다 성능이 낮아질 가능성에 대한 방안이 구체적으로 제시되지 않았다.

### 3. 추가 실험 (Autoencoder & PSO)

- **Autoencoder**: Middle Fourier-CNN을 오토인코더로 사전 학습(Pre-training)했을 때, 수렴 속도가 빨라지고 더 높은 정확도에 도달함을 확인하였다.
- **PSO (Particle Swarm Optimization)**: 경사 하강법(BP) 대신 PSO 알고리즘을 사용하여 파라미터를 최적화했으나, 결과는 BP보다 훨씬 낮았다. 이는 스웜(Swarm)의 크기가 충분하지 않았거나 업데이트 주기가 너무 이산적이었기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 고정된 활성화 함수 대신, **푸리에 급수(Fourier Series)**와 **선형 결합(Linear Combination)**을 이용하여 학습 가능한 활성화 함수를 제안하였다. CIFAR-10 실험 결과, 이러한 방식은 단순 ReLU보다 모델의 표현력과 일반화 능력을 향상시키며, 특히 소규모 모델에서는 푸리에 기반 방식이, 대규모 모델에서는 선형 결합 방식이 효과적임을 입증하였다. 이 연구는 활성화 함수를 하이퍼파라미터가 아닌 학습 파라미터로 취급함으로써 신경망 설계의 유연성을 높였으며, 향후 최신 아키텍처에 적용될 경우 추가적인 성능 향상 가능성을 제시한다.
