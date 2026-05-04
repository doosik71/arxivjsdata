# Parametric Rectified Power Sigmoid Units: Learning Nonlinear Neural Transfer Analytical Forms

Abdourrahmane M. ATTO, Sylvie GALICHET, Dominique PASTOR, Nicolas MÉGER (2021)

## 🧩 Problem to Solve

본 논문은 기존의 신경망 활성화 함수(Activation Function)들이 고정된 비매개변수적(Non-parametric) 형태를 가지고 있어, 데이터의 특성이나 네트워크의 깊이에 최적화된 비선형성을 학습하지 못한다는 문제를 해결하고자 한다.

구체적으로, 가장 널리 쓰이는 ReLU는 음수 입력에 대해 출력을 0으로 강제하여 희소성(Sparsity)을 제공하지만, 0에서의 불연속성과 음수 영역에서의 기울기 소실(Dying ReLU) 문제가 존재한다. 반면 Sigmoid 함수는 모든 영역에서 매끄러운 미분값을 가지지만, 입력값이 크거나 작을 때 기울기가 0으로 수렴하는 기울기 소실(Vanishing Gradient) 문제가 발생한다.

따라서 본 연구의 목표는 ReLU의 희소성과 Sigmoid의 매끄러운 특성을 동시에 확보하면서, 네트워크가 학습 과정에서 데이터에 가장 적합한 활성화 함수의 형태를 스스로 결정할 수 있도록 매개변수화된 새로운 활성화 함수 클래스인 Rectified Power Sigmoid Unit(RePSU)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비선형 활성화 함수를 고정된 함수가 아닌, 학습 가능한 매개변수($\lambda, \sigma, \mu, \beta, \alpha$)를 가진 함수군으로 정의하고, 이를 Convolutional weights와 함께 동시에 학습(Joint Learning)하는 것이다.

RePSU는 입력 신호를 감쇄시키는 Shrinkage unit(RePSKU)과 신호를 증폭시키는 Stretching unit(RePSHU)을 결합한 형태이다. 이를 통해 네트워크는 특정 레이어에서 신호를 억제해야 할지 혹은 강조해야 할지를 매개변수 $\alpha$를 통해 스스로 학습할 수 있다. 또한, 매개변수 조정을 통해 표준 ReLU를 특수 사례(Limit case)로 포함함으로써 기존 ReLU의 장점을 유지하면서도 더 넓은 범위의 비선형 형태를 표현할 수 있도록 설계되었다.

## 📎 Related Works

논문에서는 다음과 같은 기존 활성화 함수들의 한계를 지적한다.

- **ReLU 및 Sigmoid**: 각각 기울기 소실과 불연속성 및 희소성 부족이라는 치명적인 단점을 가진다.
- **Parametric ReLU (PReLU) 및 Parametric Sigmoid**: 매개변수를 도입하여 일부 문제를 해결하려 했으나, 특정 데이터셋이나 특수 네트워크에 국한되어 사용되며 일반화 능력이 충분히 검증되지 않았다.
- **MISH 및 SWISH (SiLU)**: 최근 제안된 비단조(Non-monotonic) 함수들로 성능 향상이 보고되었으나, 본 논문은 이들이 매우 큰 음수 입력에 대해서도 출력이 0이 되지 않아 ReLU와 같은 완전한 희소성을 제공하지 못하며, 학습 과정에서 기울기 소실 문제가 여전히 발생할 수 있음을 지적한다.

## 🛠️ Methodology

### 1. RePSU 아키텍처 및 수식

RePSU는 세 단계의 정의 과정을 거쳐 구성된다. 우선, $\mathbb{1}_\lambda(x)$를 $x > \lambda$일 때 1, 그렇지 않으면 0인 지시 함수(Indicator function)로 정의한다.

**Rectified Power Sigmoid shrinKage Units (RePSKU)**는 다음과 같이 정의된다:
$$f_{\lambda, \sigma, \mu, \beta}(x) = \frac{(x-\lambda)\mathbb{1}_\lambda(x)}{1 + e^{-\text{sgn}(x-\mu) \left(\frac{|x-\mu|}{\sigma}\right)^\beta}}$$
여기서 $\lambda$는 임계값(Threshold), $\mu$는 이동(Shift), $\sigma$는 스케일(Scale), $\beta$는 모양(Shape) 매개변수이다.

**Rectified Power Sigmoid stretcHage Units (RePSHU)**는 RePSKU를 이용하여 다음과 같이 정의된다:
$$g_{\lambda, \sigma, \mu, \beta}(x) = 2x\mathbb{1}_\lambda(x) - f_{\lambda, \sigma, \mu, \beta}(x)$$

최종적으로 **Rectified Power Sigmoid Unit (RePSU)**는 위 두 유닛의 가중 합으로 표현된다:
$$A_{\lambda, \sigma, \mu, \beta, \alpha}(x) = \alpha g_{\lambda, \sigma, \mu, \beta}(x) + (1-\alpha)f_{\lambda, \sigma, \mu, \beta}(x)$$
여기서 $\alpha$는 Stretching과 Shrinkage 사이의 비중을 조절하는 매개변수이다.

### 2. 주요 특성

- **범용성**: $\alpha=0, \beta=1$인 특수 사례인 ReSKU는 SWISH와 SiLU를 포함하며, 매개변수 극한값에 따라 표준 ReLU로 수렴한다.
- **불변성(Invariance)**: ReSKU 서브클래스는 매개변수 시프트를 통해 입력의 평행 이동(Translation)과 스케일링(Scaling)에 대한 불변성을 내부적으로 처리할 수 있는 능력을 갖춘다.
- **미분 특성**: RePSU의 미분값은 "No-jump" 특성을 가지며 매우 매끄럽게 변화한다. 또한 입력값이 무한히 커질 때 미분값이 1로 수렴하여 매우 안정적인 학습이 가능하다.

### 3. 학습 절차

본 논문에서는 Convolutional weights와 RePSU의 매개변수($\lambda, \sigma, \mu, \alpha$)를 동시에 학습시킨다. 단, $\beta$ 매개변수는 학습 시 매우 민감하여 최적화가 어렵기 때문에 실험에서는 $1$로 고정하였다. 학습은 표준적인 역전파(Back-propagation) 알고리즘과 경사 하강법(Gradient Descent)을 사용하여 Cross-entropy 손실 함수를 최소화하는 방향으로 진행된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 손글씨 숫자 인식(MNIST 계열) 및 합성 텍스처 데이터셋(GFBF, Generalized Fractional Brownian Fields)을 사용하였다.
- **비교 대상**: ReLU, MISH, PMISH(Parametric MISH), SWISH, PSWISH(Parametric SWISH) 기반 CNN.
- **평가 지표**: 테스트 정확도(Accuracy) 및 검증 손실(Validation Loss), 수렴 속도.
- **네트워크 구조**: 연산 복잡도를 제한하기 위해 CNN의 여러 레이어 중 단 하나의 레이어에만 RePSU(또는 비교 대상 함수)를 적용하고, 나머지 레이어는 ReLU를 사용하였다.

### 2. 주요 결과

- **손글씨 숫자 인식 (Shallow CNN)**: 100회의 몬테카를로 시뮬레이션 결과, RePSU 기반 CNN이 모든 비교 대상보다 빠르게 높은 정확도에 도달하였다. 특히 학습 에포크(Epoch)가 1~2회로 매우 짧은 상황에서 ReLU 및 다른 함수들보다 월등한 성능을 보였다.
- **텍스처 분류 (Deep CNN)**: 복잡한 GFBF 데이터셋 실험에서 RePSU는 가장 빠른 수렴 속도와 가장 높은 검증 정확도를 기록하였다.
- **정량적 성능**: Table 4에 따르면, RePSU 기반 모델은 100 에포크 기준 약 74.27%의 정확도를 보이며, 이는 ReLU(58.96%)나 MISH(66.25%)보다 유의미하게 높은 수치이다.

## 🧠 Insights & Discussion

### 강점 및 분석

RePSU는 비선형 활성화 함수 자체를 학습 가능하게 함으로써, 데이터의 통계적 특성에 맞는 최적의 비선형 변환을 찾을 수 있음을 입증하였다. 특히 ReLU의 희소성과 Sigmoid의 매끄러운 특성을 결합하여, 기존의 PMISH나 PSWISH가 해결하지 못한 음수 영역의 기울기 소실 및 희소성 부족 문제를 효과적으로 해결하였다.

### 한계 및 비판적 해석

- **계산 복잡도**: 모든 레이어에 RePSU를 적용할 경우 매개변수 수가 급격히 증가하여 계산 비용이 매우 높아진다. 본 논문에서는 이를 피하기 위해 단일 레이어에만 적용하는 절충안을 사용하였는데, 이는 RePSU의 완전한 잠재력을 확인하기에는 제한적인 설정일 수 있다.
- **매개변수 학습의 어려움**: $\beta$ 매개변수를 학습시키지 못하고 $1$로 고정한 점은 이 함수군의 완전한 자동 학습이 아직 어렵다는 것을 시사한다. $\beta$를 학습시키기 위한 정교한 업데이트 전략이 추가적으로 필요하다.

### 향후 전망

저자들은 RePSU 기반의 얕은(Shallow) 네트워크가 기존의 ReLU 기반 깊은(Deep) 네트워크를 대체할 가능성을 제기한다. 비선형성 자체가 강력해지면 레이어 수를 줄여도 동일하거나 더 높은 성능을 낼 수 있으며, 이는 모델의 해석 가능성(Interpretability)과 효율성을 크게 높일 수 있는 방향이다.

## 📌 TL;DR

본 논문은 학습 가능한 매개변수를 가진 새로운 활성화 함수 클래스인 **RePSU(Rectified Power Sigmoid Unit)**를 제안한다. RePSU는 ReLU의 희소성과 Sigmoid의 매끄러운 특성을 통합하며, 네트워크가 학습 과정에서 최적의 비선형 형태를 스스로 결정하게 한다. 실험 결과, RePSU는 기존의 ReLU, MISH, SWISH 및 그 매개변수 버전들보다 더 빠른 수렴 속도와 높은 분류 정확도를 보였으며, 이는 향후 딥러닝 아키텍처를 더 효율적이고 얕은 구조로 설계할 수 있는 가능성을 제시한다.
