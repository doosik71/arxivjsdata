# ReCA, a parametric ReLU composite activation function

John Chidiac, Danielle Azar (2025)

## 🧩 Problem to Solve

심층 신경망(Deep Neural Networks)에서 활성화 함수(Activation Function)는 모델의 성능에 결정적인 영향을 미친다. 현재 Rectified Linear Unit (ReLU)가 가장 널리 사용되고 있으나, ReLU는 음수 입력에 대해 그래디언트가 0이 되어 뉴런이 영구적으로 비활성화되는 'Dying ReLU' 문제와 같은 한계를 가지고 있다. 또한, 고정된 수학적 형태를 가진 전통적인 활성화 함수들은 모든 뉴런에 동일한 동작을 강제하므로, 데이터의 복잡한 비선형 관계를 학습하는 데 한계가 있을 수 있다.

본 논문의 목표는 ReLU의 효율성과 희소성(Sparsity)을 유지하면서도, 학습 가능한 파라미터를 통해 곡률(Curvature)과 부드러움(Smoothness)을 조절할 수 있는 새로운 파라미터 기반 활성화 함수인 ReCA를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 ReLU를 기반으로 하되, $\tanh$와 $\sigma(\text{sigmoid})$ 함수의 특성을 합성하여 입력 값 $x > 0$ 영역에서의 선형성 정도를 동적으로 제어하는 것이다. 

ReCA는 $\alpha, \beta, \delta$라는 세 가지 학습 가능한 파라미터를 도입하여, 네트워크가 주어진 태스크에 최적화된 활성화 함수 형태를 스스로 학습하게 한다. 특히 $\beta$는 $\tanh$를 통해 작은 $x$ 값 영역에서의 미세한 형태 조정을 담당하고, $\delta$는 $\sigma$를 통해 더 넓은 범위의 곡률 조정을 담당하도록 설계되었다. 이를 통해 ReLU의 급격한 전이(Sharp transition)를 완화하고 더 부드럽고 적응적인 특징 공간(Feature space)을 형성한다.

## 📎 Related Works

논문에서는 다음과 같은 기존 활성화 함수들의 특성과 한계를 설명한다.

- **Sigmoid ($\sigma$) 및 Tanh**: 입력을 특정 범위로 매핑하지만, 입력값이 매우 크거나 작을 때 그래디언트가 소실되는 Vanishing Gradient 문제가 발생하여 심층 신경망에 부적합하다.
- **ReLU**: 계산 효율성이 높고 vanishing gradient 문제를 완화하지만, 음수 입력에 대해 출력이 0이 되어 뉴런이 죽는 Dying ReLU 문제가 발생한다.
- **Leaky ReLU 및 PReLU**: 음수 영역에 작은 기울기를 도입하여 Dying ReLU 문제를 해결하려 했으며, PReLU는 이 기울기를 학습 가능하게 만들었다.
- **Swish**: $x \cdot \sigma(x)$ 형태로 정의되며, 비단조적(Non-monotonic)이고 부드러운 특성을 가져 ReLU보다 우수한 성능을 보이기도 한다.
- **ELU 및 SELU**: 음수 영역에서 지수 함수를 사용하여 평균 출력을 0에 가깝게 유지하고 학습 안정성을 높였다.

ReCA는 이러한 기존 함수들의 장점을 결합하되, 특히 $x > 0$ 영역에서의 비선형성을 파라미터로 정밀하게 제어함으로써 기존의 고정된 함수나 단순한 파라미터 함수보다 더 높은 표현력을 갖추고자 한다.

## 🛠️ Methodology

### 전체 구조 및 정의

ReCA 함수 $f(x)$는 다음과 같이 정의된다.

$$f(x) = \alpha \text{ReLU}(x) \left( \left( \frac{1 + \tanh(x)}{2} \right)^\beta + \sigma(x)^\delta \right)$$

여기서 파라미터의 범위는 $\alpha \in (0, +\infty)$, $\beta, \delta \in [0, +\infty)$이다. 이를 구간별 함수로 나타내면 다음과 같다.

$$f(x) = \begin{cases} \alpha x \left( \left( \frac{1 + \tanh(x)}{2} \right)^\beta + \sigma(x)^\delta \right) & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 주요 구성 요소 및 역할

- **$\alpha$ (Scaling parameter)**: 전체적인 출력 크기를 조절하는 스케일링 계수이다.
- **$\beta$ ($\tanh$ term)**: $\tanh$의 그래디언트는 $e^{-2x}$의 속도로 빠르게 감소하므로, 주로 $x$가 작은 영역에서 함수의 형태를 미세하게 조정(Fine-tuning)하는 역할을 한다.
- **$\delta$ ($\sigma$ term)**: $\sigma$의 그래디언트는 $e^{-x}$의 속도로 감소하며, $\tanh$보다 완만하게 변화하므로 더 넓은 범위의 곡률 조정을 담당한다.
- **학습 절차**: 초기값은 $\alpha = 0.5, \beta = \delta \approx 0$으로 설정하여 초기 상태에서는 ReLU와 유사하게 동작하도록 하며, 이후 역전파(Backpropagation)를 통해 파라미터를 최적화한다.

### 학습 및 구현 상세

- **미분 가능성**: $x > 0$ 영역에서 미분 가능하며, 구체적인 도함수 $f'(x)$는 논문 내 식 (2)에 명시되어 있다.
- **정규화**: 학습 가능한 파라미터 $\alpha, \beta, \delta$가 과도하게 커지는 것을 방지하기 위해 $10^{-7}$ 강도의 $L_2$ Regularization을 적용하였다.
- **적용 방식**: 기존 활성화 함수를 대체하는 **Channel-wise** 방식으로 구현되었다. 이는 뉴런별(Per-neuron) 적용보다 계산 비용이 저렴하면서도 충분한 성능 향상을 제공한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-10, CIFAR-100, Tiny ImageNet을 사용하여 이미지 분류 작업을 수행하였다.
- **모델 아키텍처**: ResNet-20, ResNet-32, ResNet-56, Wide ResNet (WRN-16-8), DenseNet-BC-121, MobileNetV3-Small 등 다양한 구조에서 테스트하였다.
- **비교 대상**: ReLU, PReLU, Swish.
- **평가 지표**: Top-1 및 Top-5 Accuracy.

### 정량적 결과

1. **CIFAR-10**: ResNet-20 모델에서 ReCA는 평균 $85.90\%$의 정확도를 기록하여 ReLU($83.78\%$), PReLU($82.97\%$), Swish($82.10\%$)보다 우수한 성능을 보였다. WRN-16-8 모델에서도 ReLU 대비 $1.24\%$의 성능 향상을 보였다.
2. **CIFAR-100**: 난이도가 높은 CIFAR-100에서 성능 향상이 더 두드러졌다. ResNet-32에서는 PReLU 대비 $4.59\%$, ResNet-56에서는 Swish($51.11\%$) 대비 $5.19\%$ 높은 $56.30\%$의 정확도를 달성하였다.
3. **Tiny ImageNet**: DenseNet-BC-121 모델에서 ReCA의 Top-1 정확도는 $41.29\% \sim 41.80\%$로 ReLU($39.95\% \sim 40.74\%$)를 일관되게 앞섰다. MobileNetV3-Small에서도 ReLU 대비 우수한 성능을 확인하였다.

### 자원 소모 및 트레이드오프

- **파라미터 수**: ReLU 대비 평균 $0.84\%$ 증가하여 무시할 수 있는 수준이다.
- **학습 시간**: 평균 학습 시간이 $52.20\%$ 증가하였다. 특히 DenseNet-BC-121의 경우 학습 시간이 2.5배 이상 증가하였으며, 다른 모델들에서는 $17\% \sim 40\%$ 정도 증가하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

ReCA의 성능 향상은 $x > 0$ 영역에서의 학습 가능한 비선형성이 더 풍부한 특징 표현(Feature representation)을 가능하게 했기 때문으로 분석된다. 출력 랜드스케이프(Output Landscape) 분석 결과, ReLU의 급격한 조각별 선형(Piecewise-linear) 전이와 달리 ReCA는 더 부드럽고 연속적인 그래디언트 흐름을 제공하며, 이는 심층 네트워크의 수렴과 성능 향상에 기여한다. 또한, 음수 영역에서는 0을 출력함으로써 ReLU의 장점인 희소성을 그대로 유지한다.

### 한계 및 논의사항

가장 명확한 한계는 **학습 시간의 증가**이다. 복잡한 수학적 연산($\tanh, \sigma$ 및 지수 연산)이 매 채널마다 수행되므로 계산 오버헤드가 발생한다. 저자들은 이를 성능 향상을 위한 직접적인 트레이드오프라고 설명하지만, 실시간 학습이 중요하거나 자원이 극도로 제한된 환경에서는 부담이 될 수 있다.

또한, 본 논문은 이미지 분류 작업에 집중되어 있으며, 생성 모델(GAN, Diffusion)이나 자연어 처리(Transformer)와 같은 다른 도메인에서의 범용적인 성능 향상 여부는 명시적으로 확인되지 않았다.

## 📌 TL;DR

본 논문은 ReLU의 효율성과 $\tanh, \sigma$의 부드러움을 결합한 파라미터 기반 활성화 함수 **ReCA**를 제안한다. ReCA는 $x > 0$ 영역의 곡률을 학습 가능한 파라미터 $\alpha, \beta, \delta$로 제어하여, CIFAR-10/100 및 Tiny ImageNet 데이터셋의 다양한 모델(ResNet, DenseNet, MobileNet 등)에서 기존 ReLU, PReLU, Swish보다 높은 정확도를 달성하였다. 학습 시간은 증가하지만, 더 적응적이고 부드러운 특징 공간을 제공함으로써 심층 신경망의 성능을 유의미하게 끌어올릴 수 있음을 입증하였다.