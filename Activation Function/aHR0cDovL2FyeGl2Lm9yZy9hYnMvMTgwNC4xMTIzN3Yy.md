# Deep learning improved by biological activation functions

Gardave S. Bhumbra (2018)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델에서 사용되는 활성화 함수(activation function)가 실제 생물학적 뉴런의 입력-출력 관계를 충분히 반영하지 못하고 있다는 점을 지적한다. 과거에는 Logistic Sigmoid 함수가 생물학적 영감을 받은 함수로 사용되었으나, 이는 실제 생리적 조건에서의 뉴런 동작과 일치하지 않을 뿐만 아니라, 딥러닝의 오차 역전파(error back-propagation) 과정에서 기울기 소실(vanishing gradient) 문제를 야기한다.

이를 해결하기 위해 최근에는 ReLU(Rectified Linear Unit)나 ELU(Exponential Linear Unit)와 같이 양의 영역에서 선형적인 특성을 가진 함수들이 사용되어 기울기 소실 문제를 완화하고 성능을 높였다. 하지만 이러한 함수들 역시 생물학적 근거보다는 수치적 최적화에 치중되어 있다. 따라서 본 연구의 목표는 실제 뉴런의 생물물리학적 특성을 기반으로 한 새로운 활성화 함수인 Bionodal Root Unit(BRU)을 제안하고, 이것이 기존의 ReLU, ELU 대비 학습 속도와 일반화 성능을 향상시킬 수 있는지 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 실제 뉴런의 전류-주파수 관계(current-frequency relationships) 및 생물물리학적 특성을 수학적으로 모델링하여 **Bionodal Root Unit(BRU)**이라는 새로운 활성화 함수군을 설계한 것이다. 

단순히 기존 함수를 변형한 것이 아니라, 전압 민감성 이온 채널(voltage-sensitive ion channels)의 특성과 뉴런의 발화 패턴을 반영하여 설계되었으며, 이를 통해 딥러닝 네트워크의 학습 속도를 높이고 명시적인 규제(regularisation) 없이도 더 나은 일반화 성능을 달성할 수 있음을 보였다.

## 📎 Related Works

기존의 인공 신경망은 퍼셉트론(perceptron)의 임계값(threshold) 특성이나 Sigmoid 함수의 뉴런 발화율(firing rates) 묘사 등 생물학적 영감에서 시작되었다. 그러나 딥러닝 시대에 접어들면서 다음과 같은 한계들이 나타났다.

1. **Sigmoid 계열**: 기울기 소실 문제로 인해 깊은 네트워크 학습이 어렵다.
2. **ReLU 및 변형 함수(ELU 등)**: 양의 영역에서의 비수축적(non-contractive) 활성화를 통해 기울기 소실을 해결하고 성능을 높였으나, 이는 생물학적 실제와는 거리가 멀며, 때로는 과적합(overfitting)에 취약할 수 있다.

본 논문은 이러한 기존 접근 방식이 생물학적 통찰력을 수치적 최적화와 맞바꾼 것이라고 보며, 실제 뉴런의 입력-출력 관계가 Sigmoid 형태가 아니라는 점을 근거로 BRU를 제안하여 차별점을 둔다.

## 🛠️ Methodology

### 1. Bionodal Root Units (BRU) 정의

BRU는 크게 두 가지 하위 가족인 **Exponential Root Unit(ERU)**과 **Odd Root Unit(ORU)**으로 구성된다. 여기서 $r$은 비선형성의 정도를 결정하는 양의 형상 매개변수인 radix(기수)이다.

#### (1) Exponential Root Unit (ERU)
ERU는 뉴런의 전류-주파수 관계를 적합시키기 위해 루트 함수를 사용한다.
$$f(z) = \begin{cases} (r^2 z + 1)^{1/r} - \frac{1}{r}, & \text{if } z \ge 0 \\ e^{rz} - \frac{1}{r}, & \text{if } z < 0 \end{cases}$$
그 미분값은 다음과 같다.
$$\frac{df(z)}{dz} = \begin{cases} r(r^2 z + 1)^{\frac{1-r}{r}}, & \text{if } z \ge 0 \\ re^{rz}, & \text{if } z < 0 \end{cases}$$
$r=1$일 때 ERU는 ELU와 동일하며, $r$이 커질수록 양의 영역에서 기울기가 감소하지만 0으로 수렴하지는 않는다.

#### (2) Odd Root Unit (ORU)
ORU는 대칭적인 형태를 가지며, 활성화 함수를 기함수(odd function)로 만든다.
$$f(z) = \text{sgn}(z)((r^2|z|+1)^{1/r}-1)$$
그 미분값은 다음과 같다.
$$\frac{df(z)}{dz} = r(r^2|z|+1)^{\frac{1-r}{r}}$$
$r=1$일 때 ORU는 항등 함수(identity function)가 되며, $\tanh$와 유사한 형태를 띠지만 점근적 경계(asymptotic limits)가 없어 기울기 소실 문제에서 자유롭다.

### 2. 가중치 초기화 (Weight Initialisation)

BRU의 곡률(curvature)이 $r$에 따라 달라지므로, 기존의 He 초기화 방식을 수정하여 다음과 같은 분산 $\sigma^2$ 스케일링을 적용한다.
- **ERU 레이어**: $\sigma^2(\text{ERU}) = \frac{6}{n_i(2r+1)}$
- **ORU 레이어**: $\sigma^2(\text{ORU}) = \frac{2}{n_i r}$
여기서 $n_i$는 입력의 개수이다. $r=1$일 때 표준적인 fan-in 방식과 동일해진다.

### 3. 네트워크 구성 및 학습 절차
- **구조**: 다층 퍼셉트론(MLP), Stacked Auto-encoder, Convolutional Network(LeNet-5 및 ConvPool 기반)를 사용한다.
- **BRU 배치**: 레이어마다 서로 다른 $r$ 값을 가진 BRU를 배치한다. 예를 들어, 초기 레이어에는 높은 $r$ 값을, 후기 레이어에는 낮은 $r$ 값을 배치하여 생물학적 감각 뉴런(높은 비선형성)에서 운동 뉴런(선형성에 가까움)으로 이어지는 흐름을 모사한다.
- **학습**: TensorFlow 프레임워크를 사용하며, 200 epoch 동안 학습을 진행한다.

## 📊 Results

### 1. MLP 기반 지도 학습 (MNIST)
- **설정**: 4~8개의 은닉층, 각 층 128개 유닛, SGD 최적화.
- **결과**: 모든 깊이에서 BRU 네트워크가 ReLU와 ELU보다 학습 속도가 가장 빨랐다. 특히 8개 층의 깊은 네트워크에서 BRU는 가장 낮은 테스트 손실(test loss)을 기록하며, 명시적인 규제 없이도 과적합에 가장 강한 모습을 보였다.

### 2. Stacked Auto-encoder 비지도 학습 (MNIST)
- **설정**: $[1000, 500, 250, 30, 250, 500, 1000]$ 유닛의 대칭 구조, Adam 최적화.
- **결과**: ELU가 ReLU보다 성능이 좋다는 기존 연구 결과와 일치하게 나타났으나, BRU는 ELU와 ReLU 모두를 능가하는 손실 및 재구성 오차(reconstruction error) 결과를 보였다.

### 3. Convolutional Networks (MNIST 및 CIFAR-10/100)
- **MNIST (LeNet-5)**: BRU가 가장 빠른 학습 속도를 보였으며, 최소 테스트 손실 및 에러 몫(error quotient) 측면에서 가장 우수한 일반화 능력을 보였다.
- **CIFAR-10/100 (ConvPool)**: 10개의 은닉층을 가진 복잡한 구조에서도 BRU는 ReLU와 ELU보다 빠르게 수렴하였고, 최종적인 테스트 성능 또한 가장 높았다. 이는 데이터의 차원과 모델의 복잡도가 증가해도 BRU의 성능 이점이 유지됨을 의미한다.

## 🧠 Insights & Discussion

### 학습 속도의 원인
BRU의 학습 속도가 빠른 이유는 활성화 함수의 최대 기울기가 1로 제한된 ReLU/ELU와 달리, BRU는 최대 기울기가 $r$ 값에 의해 결정되기 때문이다. 특히 입력층에 가까운 초기 레이어에 높은 $r$ 값을 배치함으로써 초기 학습 단계를 가속화하는 효과(early layer pre-training과 유사)를 얻을 수 있다.

### 일반화 성능의 원인
ReLU의 선형적인 양의 영역은 기울기 소실을 막아주지만, 제어되지 않은 유닛 기울기가 과적합을 유발할 수 있다. 반면, $r > 1$인 BRU는 더 희소한 인코딩(sparse encoding)을 유도한다. 또한 ORU의 대칭적 구조는 활성화 값을 0 근처로 유지시켜 효율적인 경사 하강법을 가능하게 하며, 자연스러운 자기 정규화(self-normalisation) 효과를 제공하여 일반화 성능을 높이는 것으로 분석된다.

### 생물학적 해석
본 연구는 감각 세포(Photoreceptors)의 강한 비선형성(높은 $r$)과 운동 뉴런(Motoneurone)의 선형적 특성(낮은 $r$)을 네트워크 구조에 반영하였다. 이는 인공지능 설계에 있어 단순한 영감을 넘어 실제 생물학적 해결책을 적용하는 것이 유효할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 실제 뉴런의 생물물리학적 특성을 반영한 **Bionodal Root Unit(BRU)** 활성화 함수(ERU, ORU)를 제안한다. 실험 결과, BRU는 MNIST와 CIFAR 데이터셋 모두에서 ReLU 및 ELU보다 **학습 속도가 빠르고 일반화 성능이 뛰어남**을 입증하였다. 특히 별도의 규제 기법 없이도 깊은 네트워크에서 과적합을 억제하는 능력이 탁월하며, 이는 향후 생물학적 근거에 기반한 새로운 딥러닝 아키텍처 설계에 중요한 방향성을 제시한다.