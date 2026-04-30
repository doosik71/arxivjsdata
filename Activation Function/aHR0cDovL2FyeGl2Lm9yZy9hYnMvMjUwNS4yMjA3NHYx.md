# The Resurrection of the ReLU

Coşku Can Horuz et al. (2025)

## 🧩 Problem to Solve

본 논문은 딥러닝 아키텍처에서 널리 사용되는 ReLU(Rectified Linear Unit) 활성화 함수의 치명적인 약점인 **Dying ReLU 문제**를 해결하고자 한다. ReLU는 단순함, 내재적 희소성(Sparsity), 그리고 효율적인 계산 성능 덕분에 오랜 기간 선호되어 왔으나, 입력값이 음수인 영역에서 기울기가 0이 되어 뉴런이 영구적으로 비활성화되는 현상이 발생한다.

이러한 문제는 모델의 전체적인 용량(Capacity)을 감소시키며, 학습의 효율성을 저해한다. 이를 해결하기 위해 GELU, SiLU, ELU와 같은 매끄러운 기울기를 가진 다양한 변형 활성화 함수들이 제안되었고, 최신 모델들은 대부분 이러한 함수들을 채택하고 있다. 그러나 본 연구의 목표는 ReLU가 가진 고유의 이점(희소성 및 위상적 특성)을 그대로 유지하면서, 오직 역전파(Backward pass) 과정에서의 기울기 흐름만을 개선하여 ReLU의 성능을 현대적으로 부활시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **SUGAR (Surrogate Gradient learning for ReLU)**라는 새로운 정규화 기법을 제안하는 것이다. SUGAR의 핵심 직관은 **순전파(Forward pass)에서는 표준 ReLU를 사용하여 희소성을 유지하고, 역전파(Backward pass)에서는 매끄러운 대리 기울기(Surrogate Gradient)를 사용하여 기울기 소실 및 뉴런 사멸을 방지**하는 것이다.

주요 기여 사항은 다음과 같다.
- **SUGAR 프레임워크 제안**: 순전파와 역전파의 활성화 함수를 분리하여 ReLU의 장점은 취하고 단점은 보완하는 plug-and-play 방식의 정규화 기법을 제시하였다.
- **새로운 대리 함수 제안**: SUGAR에 최적화된 두 가지 새로운 함수인 **B-SiLU (Bounded Sigmoid Linear Unit)**와 **NeLU (Negative slope Linear Unit)**를 설계하였다.
- **범용적 성능 검증**: VGG-16, ResNet-18과 같은 고전적 구조뿐만 아니라 Conv2NeXt, Swin Transformer와 같은 최신 아키텍처에서도 GELU를 대체하여 경쟁력 있거나 더 우수한 성능을 보임을 입증하였다.
- **심층 분석**: 활성화 분포 분석과 손실 함수 지형(Loss Landscape) 시각화를 통해 SUGAR가 실제로 사멸한 뉴런을 되살리고 최적화 경로를 안정화함을 증명하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들의 한계를 극복하고 차별점을 둔다.

- **SNN (Spiking Neural Networks)의 대리 기울기**: SNN은 불연속적이고 미분 불가능한 특성 때문에 역전파가 불가능하여, 이를 근사하는 Surrogate Gradient 학습 방식이 사용되어 왔다. 본 논문은 이 개념을 일반적인 인공신경망(ANN)의 ReLU 문제 해결로 확장하였다.
- **ProxyGrad**: 순전파에서는 ReLU를 쓰고 역전파에서는 LeakyReLU를 사용하여 활성화 최대화(Activation Maximization) 성능을 높인 연구가 있었으나, 본 논문은 이를 일반적인 모델 학습 전체의 일반화 성능 향상으로 확대 적용하였다.
- **기존 ReLU 변형 함수들 (LeakyReLU, GELU 등)**: 이들은 순전파 단계에서 음수 영역에 값을 할당하여 문제를 해결하려 하지만, 이 과정에서 ReLU 고유의 희소성(Sparsity)이 훼손되는 경향이 있다. 반면 SUGAR는 순전파에서 $\text{ReLU}(x)$를 그대로 유지하므로 희소성을 완전히 보존한다.

## 🛠️ Methodology

### 전체 파이프라인 및 FGI (Forward Gradient Injection)
SUGAR는 **Forward Gradient Injection (FGI)** 알고리즘을 기반으로 구현된다. FGI는 `stop gradient` 연산자(PyTorch의 `detach()`)를 사용하여 순전파 결과에는 영향을 주지 않으면서 역전파 시의 기울기만 조작하는 기법이다.

간접적인 FGI 방식의 수식은 다음과 같다.
$$y = f(x) - sg(f(x)) + sg(\text{ReLU}(x))$$
여기서 $sg(\cdot)$는 stop gradient 연산자이며, $f(x)$는 대리 함수이다. 순전파 시에는 $f(x) - f(x) + \text{ReLU}(x)$가 되어 결과적으로 $\text{ReLU}(x)$만 남게 되지만, 역전파 시에는 $sg$ 내부의 값은 미분되지 않으므로 $f(x)$의 기울기만이 전달된다.

더 효율적인 직접 주입 방식(Multiplication trick)은 다음과 같이 정의된다.
$$m = x \cdot sg(\tilde{f}(x))$$
$$y = m - sg(m) + sg(\text{ReLU}(x))$$
여기서 $\tilde{f}(x)$는 대리 기울기의 동작을 명시적으로 정의하며, 이를 통해 역전파 시 $\text{ReLU}$ 대신 $\tilde{f}$의 도함수가 사용되도록 한다.

### 제안하는 대리 함수 (Surrogate Functions)

#### 1. B-SiLU (Bounded Sigmoid Linear Unit)
Self-gating 특성과 조절 가능한 하한선을 결합한 함수로, 특히 일반화 성능 향상에 효과적이다.
$$\text{B-SiLU}(x) = (x + \alpha) \cdot \sigma(x) - \frac{\alpha}{2}, \quad (\alpha = 1.67)$$
$\sigma(x)$는 시그모이드 함수이며, 이 함수의 도함수는 다음과 같다.
$$\frac{d}{dx}\text{B-SiLU}(x) = \sigma(x) + (x + \alpha)\sigma(x)(1 - \sigma(x))$$

#### 2. NeLU (Negative slope Linear Unit)
ReLU의 양수 영역 기울기(1)와 GELU의 음수 영역 매끄러운 특성을 결합한 형태이다.
$$\frac{d}{dx}\text{NeLU}(x) = 
\begin{cases} 
1, & \text{if } x > 0 \\
\frac{\alpha 2x}{(1 + x^2)^2}, & \text{else}
\end{cases}$$
여기서 $\alpha$는 음수 영역 기울기의 크기를 조절하여 학습 안정성을 확보한다.

## 📊 Results

### 1. CIFAR-10/100 성능 평가
VGG-16 및 ResNet-18 아키텍처를 사용하여 다양한 대리 함수를 비교 실험하였다. 데이터 증강을 제거하여 활성화 함수의 순수 효과를 측정하였다.

- **VGG-16 (CIFAR-100)**: B-SiLU를 사용한 SUGAR 적용 시, 테스트 정확도가 $48.73\% \rightarrow 64.47\%$로 비약적으로 상승하였다.
- **ResNet-18 (CIFAR-100)**: B-SiLU 적용 시 $48.99\% \rightarrow 56.51\%$로 성능이 향상되었다.
- **결과 분석**: B-SiLU가 가장 큰 성능 향상을 보였으며, ELU와 SELU 또한 유의미한 개선을 보였다. 반면 LeakyReLU나 NeLU는 이 설정에서 큰 이득이 없었다.

### 2. 최신 아키텍처 적용 (Conv2NeXt, Swin Transformer)
GELU를 기본으로 사용하는 최신 모델들에 SUGAR를 적용하여 비교하였다.

- **Conv2NeXt**: NeLU ($\alpha=0.1$)를 사용했을 때 $83.95\%$의 정확도를 기록하여, GELU 기반 베이스라인($83.74\%$)보다 소폭 우수하거나 대등한 성능을 보였다.
- **Swin Transformer**: NeLU ($\alpha=0.01$)가 베이스라인보다 더 높은 정확도를 기록하였다.
- **시사점**: 이미 고도로 정규화된 최신 모델에서는 B-SiLU보다 ReLU의 기울기와 더 유사한 NeLU가 더 효과적임을 발견하였다.

### 3. 분석적 결과
- **뉴런 활성화 프로필**: VGG-16 분석 결과, vanilla ReLU 모델은 깊은 층에서 많은 뉴런이 완전히 사멸(Activation count = 0)하는 반면, SUGAR 모델은 활성화 분포가 정규분포 형태로 나타나 사멸한 뉴런이 성공적으로 되살아났음을 확인하였다.
- **손실 지형 (Loss Landscape)**: ResNet-18의 손실 지형 시각화 결과, vanilla ReLU는 매우 가파른 절벽(Sharp cliffs)이 존재하는 불안정한 구조인 반면, SUGAR는 훨씬 더 볼록(Convex)하고 매끄러운 표면을 형성하여 최적화가 용이함을 보여주었다.

## 🧠 Insights & Discussion

### SUGAR의 정규화 관점 해석
저자들은 SUGAR를 일종의 **적응형 정규화(Adaptive Regularization)** 기법으로 해석한다. 일반적인 Weight Decay가 가중치 크기를 직접 조절한다면, SUGAR는 활성화 패턴에 따라 기울기를 조절함으로써 정규화를 수행한다.

- **정규화 강도에 따른 선택**: 
    - VGG-16, ResNet-18처럼 정규화가 약한 모델 $\rightarrow$ ReLU의 기울기와 차이가 큰 **B-SiLU**가 강력한 정규화 효과를 주어 일반화 성능을 크게 높인다.
    - Conv2NeXt, Swin Transformer처럼 이미 정규화가 강한 모델 $\rightarrow$ 과도한 정규화는 과소적합(Underfitting)을 유발하므로, ReLU 기울기와 유사한 **NeLU**를 사용하여 조심스럽게 기울기 흐름만 확보하는 것이 유리하다.

### 비판적 해석 및 한계
- **경험적 설계**: 제안된 B-SiLU와 NeLU 함수가 수학적 원리보다는 실험적 직관과 튜닝을 통해 설계되었다는 점이 한계로 지적된다.
- **도메인 확장성**: 이미지 분류 작업 위주로 검증되었으므로, NLP나 강화학습 등 다른 도메인에서의 효과는 추가 검증이 필요하다.
- **이론적 보장**: 학습 역학의 개선은 확인되었으나, 수렴성이나 안정성에 대한 엄밀한 수학적 증명은 제시되지 않았다.

## 📌 TL;DR

본 논문은 순전파에서는 **ReLU**를 유지하여 희소성을 챙기고, 역전파에서는 **매끄러운 대리 기울기(Surrogate Gradient)**를 주입하는 **SUGAR** 프레임워크를 제안하여 **Dying ReLU 문제**를 해결하였다. 특히 새롭게 제안된 **B-SiLU**와 **NeLU** 함수를 통해 고전적 모델(VGG, ResNet)부터 최신 모델(Conv2NeXt, Swin)까지 폭넓게 성능을 향상시켰다. 이는 복잡한 최신 활성화 함수 없이도, 적절한 기울기 처리만 있다면 ReLU가 여전히 매우 강력하고 범용적인 도구가 될 수 있음을 시사한다.