# Feedback-Gated Rectified Linear Units

Marco Kemmerling (2023)

## 🧩 Problem to Solve

본 논문은 인간의 뇌에서 핵심적인 역할을 하는 피드백 연결(Feedback connections)이 인공 신경망(ANN) 연구에서는 상대적으로 간과되었다는 점에 주목한다. 대부분의 인공 신경망은 Feedforward 방식의 패러다임을 따르며, Recurrent Neural Networks(RNN)와 같은 구조가 존재하지만, 이들 역시 주로 동일 레이어 내의 순환 연결(self-recurrent)에 집중할 뿐, 상위 레이어에서 하위 레이어로 전달되는 Top-down 방식의 피드백 연결을 충분히 활용하지 않는다.

따라서 본 연구의 목표는 생물학적 뇌의 메커니즘에서 영감을 얻은 피드백 기법을 제안하고, 이를 통해 ReLU(Rectified Linear Unit) 활성화 함수를 게이팅(Gating)함으로써 인공 신경망의 학습 수렴 속도, 성능, 그리고 노이즈에 대한 강건성(Robustness)을 향상시킬 수 있는지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 뇌의 신피질(Neocortex)에 존재하는 피라미드 뉴런(Pyramidal neuron)의 구조를 모사하여, **상위 레이어의 신호가 하위 레이어 뉴런의 이득(Gain)을 조절하는 피드백 게이팅 메커니즘**을 설계한 것이다.

단순히 값을 더하거나 곱하는 방식이 아니라, 상위 레이어에서 오는 피드백 신호가 하위 레이어의 활성화 함수 기울기를 동적으로 변화시킴으로써, 네트워크가 첫 번째 패스(First pass)에서 얻은 대략적인 정보를 바탕으로 두 번째 패스(Second pass)에서 더 정밀한 처리를 수행하도록 유도한다.

## 📎 Related Works

논문은 기존의 인공 신경망이 주로 Feedforward 구조에 의존하고 있음을 지적한다. RNN이나 LSTM 같은 모델이 존재하지만, 이는 주로 시퀀스 데이터의 메모리를 유지하기 위한 동일 레이어 내의 재귀적 연결에 집중되어 있어, 본 논문이 추구하는 Top-down 방식의 피드백과는 차이가 있다.

또한, 생물학적 배경으로 신피질의 피라미드 뉴런 구조를 언급한다. 피라미드 뉴런은 기저 수지상 돌기(Basal dendrites)를 통해 Feedforward 입력을 받고, 정단 수지상 돌기(Apical dendrites)를 통해 Feedback 입력을 받는다. 특히 정단 수지상 돌기의 입력이 뉴런의 이득(Gain)을 조절한다는 신경과학적 근거(Larkum, 2004, 2013)를 바탕으로 방법론을 전개하며, 기존 ANN의 단순한 활성화 함수와 차별화를 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 동작 원리

제안된 모델은 데이터를 네트워크에 두 번 통과시키는 **Two-pass** 방식을 사용한다.

- **First Pass**: 입력 데이터가 Feedforward 방식으로 통과하여 상위 레이어의 활성화 값을 생성한다.
- **Second Pass**: 첫 번째 패스에서 생성된 상위 레이어의 값이 피드백 경로를 통해 하위 레이어로 전달되며, 이때 하위 레이어의 ReLU 활성화 함수의 이득을 조절하여 최종 출력을 계산한다.

학습 시에는 이 과정을 펼쳐서(Unroll) 하나의 거대한 Feedforward 네트워크로 간주하며, 표준 역전파(Backpropagation) 알고리즘을 사용하여 학습한다.

### 2. Feedback-Gated ReLU의 수학적 정의

본 논문은 생물학적 뉴런의 발화율(Firing rate) 모델인 $f=g(\mu_S + \alpha\mu_D + \sigma + f\beta(\mu_D) - \theta)$에서 시작하여, ANN에 적용 가능한 형태로 간소화한다.

여기서 $\mu_S$는 Feedforward 입력(Somatic current), $\mu_D$는 Feedback 입력(Distal current)을 의미한다. 이득 조절의 핵심인 $\beta(\mu_D)$는 입력값에 따라 증가하다가 특정 임계값에서 포화(Saturate)되는 조각별 선형 함수(Piecewise linear function)로 정의한다.

$$\beta(\mu_D) = \min\left(\frac{\beta_{\max}}{\eta}\mu_D, \beta_{\max}\right)$$

이를 바탕으로 최종적인 **Feedback-Gated ReLU** 공식은 다음과 같다.

$$f = \frac{\max(0, \mu_S)}{1 - \min\left(\frac{\beta_{\max}}{\eta}\mu_D, \beta_{\max}\right)}$$

- $\beta_{\max}$: 이득이 증가할 수 있는 최대치 (분모가 0이 되거나 음수가 되지 않도록 $0 < \beta_{\max} < 1$ 범위를 가져야 함).
- $\eta$: 최대 이득에 도달하는 임계값.
- 이 두 변수는 하이퍼파라미터로 취급되어 Grid Search를 통해 최적화한다.

### 3. 구현 세부 사항

- **가중치 추가**: 피드백을 받는 각 레이어 $h_i$ (크기 $n$)가 상위 레이어 $h_j$ (크기 $m$)로부터 신호를 받을 때, $n \times m$ 크기의 추가 가중치 행렬이 필요하다.
- **CNN 적용**: Convolutional Neural Network에서는 개별 뉴런이 아닌 필터 단위(Filter-wise)로 피드백 신호를 공유하여 적용한다.
- **Dropout**: 일관성을 위해 모든 패스에서 동일한 유닛을 드롭아웃해야 한다.

## 📊 Results

### 1. MNIST 데이터셋 (Autoencoder)

- **설정**: 2개의 Encoding 레이어와 2개의 Decoding 레이어로 구성된 Autoencoder를 사용하였다.
- **수렴 속도 및 성능**: 피드백을 적용한 모델이 표준 모델보다 눈에 띄게 빠르게 수렴하였다. 특히 Bottleneck(두 번째 Encoding 레이어의 차원)을 10으로 줄여 난이도를 높였을 때 피드백의 효과가 더욱 극명하게 나타났다.
- **Comprehensive Feedback**: 일부 레이어만 연결한 Partial Feedback보다 모든 상위 레이어에서 하위 레이어로 연결을 구축한 Comprehensive Feedback이 더 빠른 수렴과 더 낮은 최종 손실(Loss) 값을 보였다.
- **노이즈 강건성**: 입력 활성화 값에 가우시안 노이즈를 추가했을 때, 피드백 모델이 표준 모델보다 훨씬 더 강건한 복원 성능을 보였다.

### 2. CIFAR-10 데이터셋

- **Autoencoder**: Convolutional/Transposed Convolutional 레이어를 사용하였다. 피드백 적용 시 성능 향상이 있었으나 MNIST만큼 드라마틱하지는 않았다.
- **Batch Normalization과의 충돌**: 활성화 함수 뒤에 Batch Normalization을 적용하면 피드백의 성능 향상 효과가 사라졌다. 이는 BN이 피드백을 통해 조절된 이득(Gain)을 다시 정규화하여 무효화하기 때문으로 추측된다.
- **노이즈 강건성**: MNIST와 달리, CIFAR-10에서는 피드백 모델이 오히려 노이즈에 더 민감하게 반응하는 경향을 보였다.
- **분류 작업(Classification)**: CNN 기반 분류 모델에서 피드백을 적용했을 때, 테스트 셋 정확도가 약 0.7% 소폭 향상되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과

본 연구는 생물학적인 Top-down 피드백 메커니즘을 단순한 수식으로 구현하여 ANN에 성공적으로 통합하였다. 특히 MNIST와 같은 정형화된 데이터셋에서 수렴 속도 향상과 노이즈 강건성 증명이라는 가시적인 성과를 거두었다.

### 2. 한계 및 비판적 해석

- **데이터 의존성**: MNIST에서는 효과적이었으나 CIFAR-10에서는 효과가 감소하거나 노이즈에 취약해지는 모습이 관찰되었다. 이는 제안된 메커니즘이 MNIST처럼 구조가 단순하고 규칙적인 데이터에는 유리하지만, 복잡한 자연 이미지에는 충분하지 않을 수 있음을 시사한다.
- **BN과의 상충 관계**: Batch Normalization이 피드백의 효과를 상쇄한다는 점은 매우 흥미로운 지점이다. 이는 피드백의 본질이 '값의 스케일 조절'에 있는데, BN이 이를 강제로 표준화하기 때문에 발생하는 현상으로 보인다.
- **계산 비용**: 두 번의 패스를 거쳐야 하며 추가적인 피드백 가중치가 필요하므로, 추론 시간과 메모리 사용량이 증가하는 트레이드-오프가 존재한다.

## 📌 TL;DR

본 논문은 뇌의 피라미드 뉴런이 수행하는 이득 조절(Gain modulation) 기능을 모사하여, 상위 레이어가 하위 레이어의 ReLU 활성화 함수를 게이팅하는 **Feedback-Gated ReLU**를 제안하였다. 실험 결과, MNIST 데이터셋에서 학습 수렴 속도와 노이즈 강건성이 크게 향상되었으며, CIFAR-10에서도 소폭의 성능 향상을 확인하였다. 다만, Batch Normalization과의 부정적인 상호작용 및 데이터 복잡도에 따른 성능 편차가 존재하며, 이는 향후 피드백 메커니즘의 일반화 가능성을 연구하는 데 중요한 실마리가 될 것이다.
