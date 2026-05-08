# Improving Deep Neural Network with Multiple Parametric Exponential Linear Units

Yang Li, Chunxiao Fan, Yong Li, Qiong Wu, Yue Ming

## 🧩 Problem to Solve

기존 딥러닝 활성화 함수(Activation Function)에는 다음과 같은 문제가 있습니다.

- **ReLU(Rectified Linear Unit) 및 PReLU(Parametric Rectified Linear Unit)**: 음수 입력에 대해 선형 함수를 표현할 수 있지만, 비선형 함수는 표현할 수 없습니다. 이는 표현 능력의 간극(representation gap)을 야기합니다.
- **ELU(Exponential Linear Unit)**: 음수 입력에 대해 비선형 함수를 표현할 수 있지만, 선형 함수는 표현할 수 없습니다. 또한, Batch Normalization(BN)과 함께 사용할 때 분류 정확도를 저하시킬 수 있는 잠재적인 단점이 있습니다.
- **가중치 초기화(Weight Initialization)**: ELU와 같은 지수 선형 단위(exponential linear units)를 사용하는 매우 깊은 네트워크를 훈련하기 위한 이론적으로 뒷받침되는 적절한 초기화 방법이 부족합니다. 기존 방법들은 ReLU 계열 활성화 함수에 중점을 둡니다.

이러한 문제들은 딥 뉴럴 네트워크의 표현력과 학습 효율성을 저해합니다.

## ✨ Key Contributions

- **새로운 활성화 함수 MPELU(Multiple Parametric Exponential Linear Unit) 제안**: ReLU 계열(Rectified Linear Unit)과 ELU 계열(Exponential Linear Unit) 활성화 함수의 솔루션 공간을 모두 포괄하는 일반화된 형태를 제시합니다. 이는 기존 활성화 함수의 장점을 결합하여 더 나은 분류 성능과 수렴 특성을 제공합니다.
- **ELU 및 MPELU 네트워크를 위한 가중치 초기화 기법 제안**: 지수 선형 단위를 사용하는 매우 깊은 네트워크의 훈련을 가능하게 하며, 기존 이론을 확장합니다. 이는 ReLU 계열 활성화 함수를 위한 MSRA filler를 ELU 및 MPELU에 맞게 일반화한 분석적 솔루션입니다.
- **MPELU ResNet 아키텍처 제시**: 제안된 활성화 함수와 초기화 기법을 활용한 MPELU ResNet 잔차 아키텍처를 통해 CIFAR-10/100 데이터셋에서 최신 성능을 달성합니다.

## 📎 Related Works

- **활성화 함수**:
  - **ReLU [4,5]**: 기울기 소실(vanishing gradient) 문제 완화.
  - **LReLU [6]**: ReLU의 0 기울기 문제를 해결하기 위해 음수 입력에 작은 기울기 부여.
  - **PReLU [7]**: LReLU의 기울기 인자를 학습 가능하게 하여 성능 향상.
  - **ELU [8]**: 학습 속도 및 일반화 성능 향상. BN과 함께 사용할 때 성능 저하 문제.
  - **SReLU [19]**: 볼록(convex) 및 비볼록(non-convex) 함수를 모두 학습할 수 있는 S자형 활성화 함수.
  - **RReLU [20]**: 기울기를 무작위화하여 과적합 위험 감소.
- **가중치 초기화**:
  - **사전 훈련(Pre-training) [21,22,23]**: 깊은 네트워크 훈련을 위한 초기화 방법.
  - **Xavier 초기화 [13]**: 선형 활성화 함수를 가정하고 제안.
  - **MSRA filler (He 초기화) [7]**: ReLU 및 PReLU와 같은 Rectified Linear Unit을 고려하여 Xavier를 확장.
  - **LSUV 초기화 [24]**: 데이터 기반 초기화 방법으로, 분석적 솔루션 없이 $E(x_l^2)$와 $V ar(y_{l-1})$의 관계를 피함.

## 🛠️ Methodology

### MPELU (Multiple Parametric Exponential Linear Unit) 활성화 함수

MPELU는 ReLU, LReLU, PReLU, ELU를 통합하는 일반화된 활성화 함수입니다.

- **정의**:
  $$f(y_i) = \begin{cases} y_i & \text{if } y_i > 0 \\ \alpha_c (e^{\beta_c y_i} - 1) & \text{if } y_i \le 0 \end{cases}$$
  여기서 $y_i$는 입력, $\alpha_c$와 $\beta_c$는 학습 가능한 파라미터입니다. $\beta_c > 0$로 제한됩니다.
- **일반화 능력**:
  - $\beta_c$가 매우 작은 값(예: 0.01)일 때, 음수 부분은 선형 함수에 근사하여 PReLU처럼 동작합니다.
  - $\beta_c$가 큰 값(예: 1.0)일 때, 음수 부분은 비선형 함수가 되어 ELU처럼 동작합니다.
  - $\alpha_c=1, \beta_c=1$일 때, ELU와 동일합니다.
  - $\alpha_c=0$일 때, ReLU와 동일합니다.
- **학습 가능한 파라미터**: $\alpha_c$와 $\beta_c$는 채널별(channel-wise) 또는 채널 공유(channel-shared) 방식으로 학습되며, 네트워크의 파라미터 증가를 최소화합니다.
- **역전파(Backward Pass)**: $\alpha_c$, $\beta_c$, $y_i$에 대한 미분 공식이 제시되어 엔드-투-엔드(end-to-end) 훈련이 가능합니다.
  - $\frac{\partial f(y_i)}{\partial \alpha_c} = \begin{cases} 0 & \text{if } y_i > 0 \\ e^{\beta_c y_i}-1 & \text{if } y_i \le 0 \end{cases}$
  - $\frac{\partial f(y_i)}{\partial \beta_c} = \begin{cases} 0 & \text{if } y_i > 0 \\ y_i \cdot top'_i & \text{if } y_i \le 0 \end{cases}$ (여기서 $top'_i = f(y_i) + \alpha_c$)
  - $\frac{\partial f(y_i)}{\partial y_i} = \begin{cases} 1 & \text{if } y_i > 0 \\ \beta_c \cdot top'_i & \text{if } y_i \le 0 \end{cases}$
- **초기화**: $\alpha=1$ 또는 $0.25$, $\beta=1$로 초기화하고, 두 파라미터에 대해 가중치 감소(weight decay)를 적용합니다.

### MPELU를 위한 가중치 초기화

ELU 및 MPELU와 같은 지수 선형 단위를 사용하는 깊은 네트워크를 위한 초기화 방법을 제안합니다.

- **접근 방식**: He et al. [7]의 MSRA filler 유도를 따르며, MPELU의 음수 부분을 0에서 1차 테일러 확장으로 근사합니다.
  $$\alpha(e^{\beta y}-1) \approx \alpha \beta y$$
- **분산 보존 조건**: 활성화 함수의 입출력 분산을 일정하게 유지하기 위해 다음과 같은 가중치 분산 조건을 도출합니다.
  $$\frac{1}{2} k_i^2 c_i (1 + \alpha_i^2 \beta_i^2) V ar(w_i) = 1, \forall i$$
  여기서 $k_i$는 커널 크기, $c_i$는 입력 채널 수입니다.
- **초기화 공식**: 각 레이어의 가중치는 평균이 0인 가우시안 분포로 초기화됩니다.
  $$\mathcal{N}\left(0, \sqrt{\frac{2}{k_i^2 c_i (1 + \alpha_i^2 \beta_i^2)}}\right)$$
  이 공식은 $\alpha=0$일 때 ReLU 초기화, $\alpha=1, \beta=1$일 때 ELU 초기화가 됩니다.

### MPELU 잔차 아키텍처

기존 ResNet에 MPELU를 적용하고 최적화된 블록 구조를 제안합니다.

- **MPELU Non-bottleneck Residual Block**: Fig. 4(b)와 같이 합산 후의 MPELU를 제거하여 항등 사상(identity mapping)을 더 쉽게 학습하도록 합니다.
- **MPELU Bottleneck Residual Block (Nopre-activation)**: Fig. 5(d)와 같이 사전 활성화(pre-activation) 부분을 제거하여 파라미터와 복잡도를 줄이면서 성능을 유지합니다. 특히 첫 번째 합성곱 레이어 직후와 네트워크의 마지막 요소별 합산 직후에 BN과 MPELU를 사용하는 것이 중요함을 발견했습니다.

## 📊 Results

- **NIN (Network in Network) on CIFAR-10**:
  - MPELU는 ELU보다 일관되게 우수한 성능을 보이며(예: 데이터 증강 없음 9.06% vs 9.39%), 학습 가능한 파라미터 $\alpha, \beta$의 이점을 입증합니다.
  - ReLU, PReLU보다 더 빨리 수렴하며, 15% 테스트 에러에 도달하는 데 MPELU는 9k iterations, PReLU는 15k, ReLU는 25k iterations이 소요됩니다.
- **15-Layer Network on ImageNet**:
  - MPELU는 PReLU 및 ELU보다 전반적으로 우수한 성능을 달성합니다 (최고 Top-1 에러율 37.33%).
  - $\alpha, \beta$에 가중치 감소를 사용하면 성능이 크게 향상됩니다.
  - 제안된 초기화 방법으로 초기화된 MPELU 네트워크는 Gaussian 초기화보다 일관되게 성능이 좋습니다.
  - 실행 시간은 PReLU와 유사하게 최적화될 수 있습니다.
- **30-Layer Network (Initialization Test) on ImageNet**:
  - Gaussian 초기화는 30-layer ELU/MPELU 네트워크 훈련에 실패하지만, 제안된 초기화 방법은 수렴을 가능하게 합니다 (MPELU 36.49% Top-1 에러).
  - BN과 함께 사용될 때도 제안된 초기화 방법은 성능을 향상시킵니다.
  - 30-layer 네트워크가 15-layer 네트워크보다 성능이 낮은 '퇴화 문제(degradation problem)'가 BN 사용 시 나타날 수 있음을 시사합니다. (BN 없는 30-layer 네트워크는 15-layer보다 우수)
  - LSUV 초기화와 비교했을 때, 제안된 초기화는 분석적이며 약간 더 좋은 성능을 보입니다.
- **Deep MPELU Residual Networks on CIFAR-10/100**:
  - **MPELU Non-bottleneck ResNet (Fig. 4(b))**: 기존 ResNet보다 일관되게 낮은 테스트 에러율을 달성 (예: ResNet-110 6.43% vs MPELU ResNet-110 5.47%).
  - **MPELU Nopre-activation Bottleneck (Fig. 5(d))**: 164-layer 네트워크에서 기존 Pre-ResNet (5.46%)보다 낮은 4.87%의 테스트 에러를 달성하며, 파라미터와 복잡도는 유사합니다.
  - BN1과 BNend (Fig. 5(d)의 BN 위치)가 nopre-activation 아키텍처에 중요함을 보여줍니다.
  - CIFAR-100에서 기존 Pre-ResNet-1001 (22.71%)보다 훨씬 낮은 18.81%의 테스트 에러율을 달성하며 최신 성능을 달성합니다.

## 🧠 Insights & Discussion

- **MPELU의 유연성**: MPELU는 학습 가능한 파라미터 $\alpha, \beta$를 통해 음수 부분에서 선형 및 비선형 함수 공간을 모두 포괄하여, 기존 활성화 함수들의 표현력 간극을 메웁니다. 이는 ResNet 아키텍처에서 항등 사상(identity mapping)을 더 잘 근사하는 데 도움이 됩니다.
- **Batch Normalization과의 호환성**: ELU와 달리 MPELU는 BN과 함께 사용할 때 성능 저하가 발생하지 않고 오히려 향상됩니다. 이는 MPELU가 PReLU의 특성을 공유하여 BN의 출력이 PReLU 서브모듈로 직접 흘러들어가는 구조적 이점 때문일 수 있습니다.
- **가중치 감소의 중요성**: MPELU의 $\alpha, \beta$에 가중치 감소를 적용하는 것이 유리합니다. 이는 $\alpha, \beta$를 0에 가깝게 밀어내어 더 작은 스케일의 활성화(sparse representation)를 유도하고, 결과적으로 선형적으로 분리 가능한 특성 공간을 만들 수 있기 때문으로 분석됩니다.
- **초기화의 중요성 및 효과**: 제안된 초기화 방법은 기존 방법으로는 훈련 불가능했던 매우 깊은 ELU/MPELU 네트워크의 수렴을 가능하게 합니다. 이는 1차 테일러 근사가 실제 환경에서 효과적임을 보여줍니다.
- **퇴화 문제와 BN의 역할**: BN이 깊은 네트워크 훈련에 중요하지만, 특정 조건(예: 매우 깊은 잔차 네트워크)에서는 퇴화 문제를 유발하는 요인이 될 수 있음을 실험적으로 시사합니다. BN 없이 제안된 초기화만으로 깊은 네트워크가 깊이의 이점을 누릴 수 있었습니다.
- **분산 보존**: MPELU는 ELU와 마찬가지로 활성화의 분산을 점진적으로 감소시켜, ReLU 네트워크에서 발생할 수 있는 활성화 발산(overflow) 문제를 방지합니다.

## 📌 TL;DR

본 논문은 딥 뉴럴 네트워크의 표현력과 학습 효율성을 개선하기 위해 **MPELU(Multiple Parametric Exponential Linear Unit)**라는 새로운 활성화 함수와 **지수 선형 단위를 위한 새로운 가중치 초기화 방법**을 제안합니다. MPELU는 학습 가능한 파라미터 $\alpha, \beta$를 통해 ReLU/PReLU 계열의 선형 음수부와 ELU 계열의 비선형 음수부를 모두 아우르며, Batch Normalization과 잘 작동합니다. 제안된 초기화는 ELU 및 MPELU를 사용하는 매우 깊은 네트워크의 훈련을 가능하게 합니다. 이들을 결합한 **MPELU ResNet 아키텍처**는 CIFAR-10/100 데이터셋에서 최신 성능을 달성하며, 깊은 네트워크에서 BN이 퇴화 문제의 원인이 될 수 있음을 시사합니다.
