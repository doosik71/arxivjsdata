# SwishReLU: A Unified Approach to Activation Functions for Enhanced Deep Neural Networks Performance

Jamshaid Ul Rahman, Rubiqa Zulfiqar, Asad Khan, Nimra (2024)

## 🧩 Problem to Solve

본 논문은 심층 신경망(Deep Neural Networks, DNNs)에서 널리 사용되는 활성화 함수인 ReLU(Rectified Linear Unit)의 치명적인 결함인 "Dying ReLU" 문제를 해결하고자 한다. ReLU는 입력값이 0보다 작을 때 출력이 0이 되어 기울기(Gradient)가 소멸하며, 이로 인해 일부 뉴런이 영구적으로 비활성화되어 학습이 중단되는 현상이 발생한다.

이를 해결하기 위해 ELU, SeLU, Swish와 같은 다양한 변형 함수들이 제안되었으나, 여전히 한계가 존재한다. 특히 Swish는 ReLU와 유사하게 부드러운 전이를 제공하여 성능을 높이지만, 지수 함수를 포함하고 있어 ReLU에 비해 연산 비용(Computational Burden)이 크고, 매우 깊은 네트워크에서는 학습 불안정성이나 발산 가능성이 있다는 점이 문제로 지적된다. 따라서 본 연구의 목표는 ReLU의 단순성과 효율성, 그리고 Swish의 효과적인 성능을 결합하여 연산 비용은 낮추면서도 성능은 향상시킨 새로운 활성화 함수인 **SwishReLU**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 입력값의 부호에 따라 ReLU와 Swish의 특성을 선택적으로 결합하는 것이다.

- **하이브리드 설계**: 양수 영역에서는 ReLU의 선형성을 유지하여 연산 효율성을 확보하고, 음수 영역에서는 Swish의 비선형적 특성을 도입하여 Dying ReLU 문제를 방지한다.
- **최적화된 효율성**: 전체 영역에 Swish를 적용하는 것보다 연산량을 줄이면서도, ReLU보다는 우수한 학습 능력을 갖춘 통합적 접근 방식을 제시한다.
- **미분 가능성 확보**: ReLU와 달리 SwishReLU는 전 영역에서 연속적인 미분 가능성(Continuous Differentiability)을 가져, Gradient Descent 기반의 최적화 알고리즘에 더 적합한 구조를 가진다.

## 📎 Related Works

논문에서는 다음과 같은 기존 활성화 함수들의 특성과 한계를 언급한다.

- **Sigmoid & Tanh**: 초기 얕은 네트워크에서 사용되었으나, 네트워크가 깊어질수록 기울기 소멸(Vanishing Gradient) 문제가 심화되어 학습이 어려워진다.
- **ReLU**: 기울기 소멸 문제를 완화하여 표준으로 자리 잡았으나, 음수 입력에 대해 출력이 0이 되는 "Dying ReLU" 현상이 발생한다.
- **ELU & SeLU**: ReLU의 변형으로 음수 영역에서의 출력을 허용하여 Dying ReLU를 방지하려 했으나, 적용 환경에 따라 이점이 일관되지 않을 수 있다.
- **Swish**: $f(x) = \frac{x}{1+e^{-x}}$ 형태로 정의되며 ReLU보다 우수한 성능을 보이지만, 연산 복잡도가 높고 매우 깊은 모델에서 불안정할 수 있다는 한계가 있다.

SwishReLU는 이러한 기존 함수들의 장점(ReLU의 효율성, Swish의 성능)을 취하고 단점(Dying ReLU, 높은 연산 비용)을 상쇄하도록 설계되었다.

## 🛠️ Methodology

### 1. SwishReLU 정의

제안된 SwishReLU 함수는 다음과 같이 정의된다.

$$
f(x) =
\begin{cases}
\frac{x}{1+e^{-x}} & \text{if } x < 0 \\
x & \text{if } x \geq 0
\end{cases}
$$

이 함수는 $x \geq 0$일 때는 ReLU와 동일하게 동작하며, $x < 0$일 때는 Swish 함수를 적용하여 음수 가중치에 대해 0이 아닌 값을 할당함으로써 뉴런의 비활성화를 막는다. 또한, Zero-centered 특성을 가져 기울기 소멸 문제를 완화하며, 전 구간에서 미분이 가능하다.

### 2. 실험 아키텍처

본 연구에서는 SwishReLU의 성능을 검증하기 위해 세 가지 모델 구조에 적용하였다.

- **FCNN (Fully Connected Neural Network)**: MNIST 데이터셋을 위해 입력층(300 뉴런), 은닉층(100 뉴런), 출력층(10 뉴런)으로 구성된 MLP 구조를 사용한다.
- **Custom CNN**: CIFAR-10/100 데이터셋을 위해 5개의 Convolutional Layer와 Max Pooling, 그리고 Dense Layer로 구성된 구조를 사용한다. 모든 활성화 함수는 SwishReLU로 대체하여 비교한다.
- **VGG-16**: 13개의 Conv Layer와 3개의 Dense Layer로 구성된 깊은 네트워크에 SwishReLU를 적용하여 대규모 모델에서의 효용성을 검증한다.

### 3. 학습 절차 및 설정

- **손실 함수**: 모든 모델에서 $\text{Sparse Categorical Cross-entropy}$ 손실 함수를 사용한다.
- **최적화 알고리즘**: FCNN과 CNN에서는 $\text{Adam}$ 옵티마이저를 사용하며, VGG-16 모델에서는 $\text{SGD}$ (Learning rate=0.01, Momentum=0.9)를 사용한다.
- **학습 횟수**: FCNN은 30 epoch, CNN은 50 epoch 동안 학습하며, VGG-16은 `val_loss`를 모니터링하는 Early Stopping(patience=5)을 적용한다.

## 📊 Results

### 1. 데이터셋 및 지표

- **데이터셋**: MNIST (손글씨 숫자), CIFAR-10 (10개 클래스 이미지), CIFAR-100 (100개 클래스 이미지)
- **측정 지표**: Training/Testing Accuracy, Training/Testing Loss

### 2. 정량적 결과 분석

- **MNIST (FCNN)**: SwishReLU는 ReLU와 유사한 훈련 정확도($99.80\%$)를 보였으나, 테스트 정확도에서는 $98.23\%$를 기록하여 ReLU($98.17\%$) 및 타 함수(ELU, SeLU, Tanh)보다 소폭 우세한 성능을 보였다.
- **CIFAR-10 (CNN)**: SwishReLU는 훈련 정확도 $96.11\%$, 테스트 정확도 $70.45\%$를 달성하여 ReLU($69.71\%$) 등 다른 함수들을 능가하였다.
- **CIFAR-100 (CNN)**: 훈련 정확도에서 $88.55\%$를 기록하며 ReLU($80.00\%$) 대비 뚜렷한 성능 향상을 보였다.
- **VGG-16 (CIFAR-10)**: SwishReLU를 적용했을 때 테스트 정확도 $81\%$를 달성하였으며, 이는 기존 ReLU 기반 모델 대비 약 $6\%$의 성능 향상이 있음을 의미한다.

### 3. Dense Layer에서의 전략적 활용

본 논문에서는 Conv Layer에는 다른 함수(ReLU, ELU 등)를 유지하고 **Dense Layer에만 SwishReLU를 적용**하는 실험을 진행하였다. 그 결과, CIFAR-10과 CIFAR-100 모두에서 정확도가 약 $1\%$ 이상 상승하는 경향을 보였으며, 이는 SwishReLU가 모델의 최종 결정 단계(Dense Layer)에서 특히 효과적임을 시사한다.

## 🧠 Insights & Discussion

### 강점

- **Dying ReLU의 실질적 해결**: 음수 영역에 Swish의 특성을 부여함으로써 뉴런의 비활성화를 방지하고 정보의 흐름을 원활하게 하여 학습 성능을 높였다.
- **범용적 적용 가능성**: FCNN, 단순 CNN, 그리고 깊은 VGG-16까지 다양한 구조에서 일관된 성능 향상을 보였다. 특히 Dense Layer에서의 부분적 교체만으로도 성능 이득을 얻을 수 있다는 점은 실용적인 통찰을 제공한다.
- **연산 효율성**: 전 구간 Swish를 사용하는 것보다 계산 복잡도를 낮추면서도 성능 손실이 없음을 확인하였다.

### 한계 및 논의사항

- **비교 대상의 한계**: 비교 대상이 ReLU, ELU, SeLU, Tanh 등 비교적 고전적인 함수들에 집중되어 있다. 최근 제안된 최신 활성화 함수들과의 비교 데이터가 부족하다.
- **가정**: 논문은 SwishReLU가 모든 도메인에서 효과적일 것이라고 주장하지만, 실험은 이미지 분류 작업(Image Classification)에 국한되어 있다. NLP나 시계열 데이터와 같은 다른 도메인에서의 성능 검증이 필요하다.
- **추가 분석 부족**: 연산 비용이 Swish보다 낮다고 언급하였으나, 실제 FLOPs나 추론 시간(Inference Time)에 대한 구체적인 정량적 수치는 제시되지 않았다.

## 📌 TL;DR

본 논문은 ReLU의 효율성과 Swish의 성능을 결합한 **SwishReLU** 활성화 함수를 제안한다. 이 함수는 $x \geq 0$에서는 $x$를, $x < 0$에서는 Swish 함수를 사용하여 "Dying ReLU" 문제를 해결하고 연산 비용을 최적화한다. MNIST, CIFAR-10/100 데이터셋 실험 결과, 다양한 아키텍처(FCNN, CNN, VGG-16)에서 기존 ReLU 및 변형 함수들보다 우수한 정확도를 보였으며, 특히 VGG-16 모델의 CIFAR-10 분류 성능을 $6\%$ 향상시켰다. 이 연구는 기존 모델의 활성화 함수를 단순 교체하는 것만으로도 성능을 높일 수 있음을 보여주며, 향후 다양한 컴퓨터 비전 응용 분야에 적용될 가능성이 크다.
