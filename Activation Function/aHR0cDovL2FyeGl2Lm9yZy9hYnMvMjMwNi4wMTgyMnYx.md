# ErfReLU: Adaptive Activation Function for Deep Neural Network

Ashish Rajanand, Pradeep Singh (2022)

## 🧩 Problem to Solve

딥러닝 네트워크에서 비선형성을 추가하기 위해 선택되는 활성화 함수(Activation Function, AF)는 네트워크의 성능에 매우 큰 영향을 미친다. 그러나 기존의 고정된 활성화 함수들은 학습 과정에서 데이터의 특성에 맞게 스스로 조정되지 못한다는 한계가 있다.

특히, 본 논문은 다음과 같은 구체적인 문제들을 해결하고자 한다.
1. **Dying ReLU 문제**: ReLU 함수는 입력값이 0보다 작을 때 기울기가 0이 되어 뉴런이 더 이상 업데이트되지 않고 '죽어버리는' 현상이 발생한다.
2. **포화(Saturation) 문제**: Sigmoid나 Tanh와 같은 함수들은 입력값이 매우 크거나 작을 때 기울기가 0에 수렴하여 학습이 정체되는 포화 현상이 나타난다.
3. **적응성 부족**: 학습 데이터의 복잡도에 따라 활성화 함수의 형태가 동적으로 변하는 적응형 활성화 함수(Adaptive Activation Function, AAF)에 대한 연구가 아직 초기 단계에 머물러 있다.

본 논문의 목표는 ReLU의 장점과 가우스 오차 함수(Error Function, $\text{erf}$)를 결합하여, 단 하나의 학습 가능한 파라미터만으로 Dying ReLU 문제와 포화 문제를 동시에 완화하는 새로운 적응형 활성화 함수인 **ErfReLU**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 ReLU의 양수 영역 특성과 $\text{erf}$ 함수의 음수 영역 특성을 조각별(piecewise)로 결합하는 것이다.

- **단순성 및 효율성**: 많은 적응형 활성화 함수들이 여러 개의 학습 파라미터를 사용하는 것과 달리, ErfReLU는 $\alpha$라는 단 하나의 학습 가능한 파라미터만을 사용하여 계산 복잡도를 낮추면서도 성능을 최적화한다.
- **음수 영역의 보존**: 음수 입력에 대해 $\text{erf}$ 함수를 적용함으로써 ReLU에서 발생하는 '죽은 뉴런' 문제를 방지하고, 소량의 음수 정보를 보존하여 네트워크의 표현력을 높인다.
- **포화 방지**: 양수 영역에서는 ReLU와 동일하게 동작하여 상한선 없는(unbounded) 특성을 가지므로, 양수 영역에서의 기울기 소실 및 포화 문제를 원천적으로 차단한다.

## 📎 Related Works

논문에서는 기존의 활성화 함수를 두 가지 범주로 나누어 설명한다.

**1. 전통적인 활성화 함수(Traditional AF)**
- **Sigmoid & Tanh**: S-자 형태의 함수로, 양 끝단에서 기울기가 0이 되는 vanishing gradient 및 saturation 문제가 심각하다.
- **ReLU**: 계산이 빠르고 양수 영역에서 효율적이지만, 음수 영역에서 기울기가 0이 되는 Dying ReLU 문제가 있다.
- **Leaky ReLU & ELU**: ReLU의 단점을 보완하기 위해 음수 영역에 작은 기울기나 지수 함수를 도입하였으나, 여전히 고정된 파라미터를 사용한다.
- **Swish & Mish**: 비단조성(non-monotonicity)과 부드러운 곡선을 특징으로 하며, ReLU보다 성능이 우수하다고 알려져 있다.

**2. 적응형 활성화 함수(Adaptive Activation Function, AAF)**
- **TanhSoft, TanhLU, SAAF, Serf, ErfAct, Pserf, Smish, IpLU** 등이 언급된다.
- 이러한 함수들은 학습 과정에서 데이터에 따라 함수의 형태를 결정하는 학습 가능한 파라미터를 포함한다.
- 기존 AAF들은 복잡한 수식을 사용하거나 많은 수의 파라미터를 필요로 하여 계산 비용이 증가하는 경향이 있다.

## 🛠️ Methodology

### 전체 구조 및 정의

제안된 **ErfReLU**는 입력값 $x$의 부호에 따라 서로 다른 함수를 적용하는 조각별 함수로 정의된다.

$$
f(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\
\alpha \text{erf}(x) & \text{if } x < 0 
\end{cases}
$$

여기서 $\text{erf}(x)$는 가우스 오차 함수(Gauss error function)로, 다음과 같이 정의된다.

$$
\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^2} dt
$$

### 학습 파라미터 및 미분

- **학습 가능 파라미터 $\alpha$**: 음수 영역에서의 함수의 기울기(slope)를 조절하는 파라미터이다. 이 값은 역전파(back-propagation) 과정에서 경사 하강법을 통해 데이터에 최적화된 값으로 업데이트된다.
- **미분 함수**: 학습을 위한 기울기 계산식은 다음과 같다.

$$
f'(x) = 
\begin{cases} 
1 & \text{if } x \geq 0 \\
\alpha \frac{2}{\sqrt{\pi}} e^{-x^2} & \text{if } x < 0 
\end{cases}
$$

### 주요 특성
- **부드러움(Smoothness)**: $\text{erf}$와 ReLU 모두 미분 가능하므로 ErfReLU 역시 부드러운 함수 특성을 가진다.
- **비선형성(Non-linearity)**: $\text{erf}$ 함수를 통해 복잡한 관계를 학습할 수 있는 비선형성을 제공한다.
- **유계성(Boundedness)**: 음수 영역에서는 $\text{erf}$ 함수의 특성상 하한선이 존재(bounded below)하지만, 양수 영역에서는 ReLU의 특성을 따라 상한선이 없다(unbounded above).

## 📊 Results

### 실험 설정
- **데이터셋**: 이미지 분류를 위해 CIFAR-10, MNIST, FMNIST를 사용하였으며, 기계 번역 성능 측정을 위해 Multi30k 데이터셋을 사용하였다.
- **모델**: 이미지 분류에는 MobileNetV1, ResNet18, VGG16을 사용하였고, 번역에는 LSTM 기반의 Seq2Seq 모델을 사용하였다.
- **학습 환경**: PyTorch 프레임워크, Nvidia Quardo Rtx 5000 GPU, Adam Optimizer ($\text{lr}=0.001$), Batch size 128 (이미지) / 256 (번역).

### 주요 결과

**1. 이미지 분류 (Classification)**
- **CIFAR-10**: ErfReLU가 MobileNet(92.78%)과 ResNet(94.04%)에서 다른 AAF 및 전통적 AF(ReLU, Swish, Mish 등)보다 우수한 정확도를 보였다. 다만 VGG16에서는 ErfAct가 약간 더 높은 성능을 기록했다.
- **MNIST & FMNIST**: 전반적으로 ErfReLU가 경쟁력 있는 성능을 보였으며, 특히 MobileNet과 ResNet 조합에서 다른 함수들보다 우수하거나 대등한 결과를 나타냈다.

**2. 기계 번역 (Machine Translation)**
- **평가 지표**: BLEU score를 통해 성능을 측정하였다.
- **결과**: 실험 결과, Smish(21.43)와 Tanhsoft2(21.27) 등이 높은 BLEU score를 기록하였다. 제안된 ErfReLU는 19.54의 점수를 기록하였다.

## 🧠 Insights & Discussion

### 강점
- **효율적인 파라미터 설계**: 단 하나의 파라미터 $\alpha$만으로도 여러 복잡한 AAF와 대등하거나 더 나은 성능을 낸다는 점이 고무적이다. 이는 추가적인 계산 오버헤드를 최소화하면서 적응성을 확보했음을 의미한다.
- **모델 범용성**: 특히 MobileNet과 같은 경량 모델에서 성능 향상이 뚜렷하게 나타나, 효율적인 아키텍처와의 시너지가 좋음을 알 수 있다.

### 한계 및 비판적 해석
- **데이터셋별 성능 편차**: 이미지 분류에서는 매우 강력한 성능을 보였으나, 기계 번역(Seq2Seq) 작업에서는 Smish 등 다른 AAF에 비해 낮은 성능을 보였다. 이는 ErfReLU가 CNN 기반의 시각 지능 작업에 더 최적화되어 있을 가능성을 시사한다.
- **텍스트와 표의 불일치**: 논문 본문에서는 기계 번역 결과에 대해 "SAAF 활성화 함수가 다른 함수들보다 우수한 성능을 보였다"고 서술하고 있으나, 실제 Table 7의 수치를 보면 SAAF의 BLEU score는 13.05로 가장 낮다. 이는 저자의 서술 오류로 판단되며, 실제로는 Smish가 가장 우수한 성능을 보인 것으로 분석된다.
- **VGG16에서의 성능**: VGG16 모델에서는 ReLU나 ErfAct가 ErfReLU보다 더 나은 성능을 보이는 경우가 있어, 모델의 깊이나 구조에 따라 최적의 AF가 다를 수 있음을 보여준다.

## 📌 TL;DR

본 논문은 ReLU의 양수 영역과 가우스 오차 함수($\text{erf}$)의 음수 영역을 결합한 새로운 적응형 활성화 함수 **ErfReLU**를 제안한다. 이 함수는 단 하나의 학습 가능한 파라미터 $\alpha$를 통해 Dying ReLU 문제를 해결하고 학습 효율을 높인다. 실험 결과, CIFAR-10, MNIST 등 이미지 분류 작업에서 특히 MobileNet과 ResNet 모델을 사용할 때 기존 SOTA 함수 및 다른 적응형 함수들보다 우수한 정확도를 달성하였다. 이는 계산 비용을 최소화하면서도 딥러닝 모델의 수렴 속도와 정확도를 개선할 수 있는 실용적인 방안이 될 수 있다.