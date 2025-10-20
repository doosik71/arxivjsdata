# FReLU: Flexible Rectified Linear Units for Improving Convolutional Neural Networks

Suo Qiu, Xiangmin Xu and Bolun Cai

## 🧩 Problem to Solve

심층 합성곱 신경망(CNN)에서 널리 사용되는 활성화 함수인 ReLU(Rectified Linear Unit)는 음수 값을 0으로 고정하여 "음수 정보 손실(negative missing)"을 발생시키고, 표현력을 제한합니다. PReLU나 LReLU와 같은 ReLU 변형들은 음수 부분을 허용하여 이 문제를 완화하지만, 희소성(sparsity)을 잃을 수 있습니다. ELU는 활성화 평균을 0에 가깝게 유지하는 "제로-유사(zero-like)" 속성을 제공하지만, Batch Normalization(BN)과의 호환성 문제가 있고 지수 함수로 인해 계산 비용이 높습니다. 이 연구는 음수 값의 효과를 탐색하여 ReLU의 한계를 극복하고, 계산 효율적이며 BN과 호환되는 새로운 활성화 함수를 제안하는 것을 목표로 합니다.

## ✨ Key Contributions

- **FReLU(Flexible Rectified Linear Unit) 제안:** ReLU의 정류 지점(rectified point)을 학습 가능한 파라미터로 재설계하여 새로운 활성화 함수 FReLU를 제안합니다.
- **표현력 향상:** 학습 가능한 편향 $b$를 통해 활성화 출력의 상태를 확장하여 신경망의 표현력을 향상시킵니다. 학습이 성공적으로 이루어지면 $b$는 음수 값으로 수렴하는 경향을 보입니다.
- **빠른 수렴 및 높은 성능:** 다양한 CNN 아키텍처에서 기존 ReLU 및 그 변형들보다 더 빠르고 안정적인 수렴과 더 높은 분류 성능을 달성합니다.
- **낮은 계산 비용:** 지수 함수를 사용하지 않아 계산 비용이 낮으며, ReLU와 유사한 비선형성 및 희소성 특성을 유지합니다.
- **Batch Normalization 호환성:** Batch Normalization(BN)과 높은 호환성을 보여, 큰 학습률에서도 안정적인 학습과 성능 향상을 가능하게 합니다.
- **자기 적응성:** 특정 가정에 의존하지 않고 자체 적응적으로 최적의 활성화 함수 형태를 학습합니다.

## 📎 Related Works

- **ReLU [2, 5]:** 양수 입력은 그대로 전달하고 음수 입력은 0으로 만드는 활성화 함수입니다. 기울기 소실 문제를 완화하고 계산 효율적이지만, 음수 값 정보를 손실하고 비-제로 중심(non-zero-like) 특성을 가집니다.
- **LReLU [7], PReLU [8], RReLU [9]:** 음수 부분에 작은 고정 또는 학습 가능한 기울기를 부여하여 음수 값 정보를 일부 유지하는 ReLU의 변형들입니다. 하지만 희소성을 파괴할 수 있습니다.
- **ELU [10]:** 음수 값을 유지하고 음수 부분을 포화시켜 활성화 평균을 0에 가깝게 만드는 "제로-유사" 속성을 추구합니다. 그러나 BN과의 호환성이 낮고 지수 함수로 인해 계산 비용이 높습니다.
- **SReLU:** $max(x, Δ)$로 정의되며, $Δ$는 학습 가능한 파라미터입니다. FReLU와 유사하게 수평 및 수직 이동의 유연성을 가집니다.

## 🛠️ Methodology

1. **FReLU 정의:**
   - 기존 ReLU는 $relu(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$ 로 정의됩니다.
   - FReLU는 ReLU의 정류 지점을 학습 가능한 파라미터로 재설계하여 다음과 같이 정의됩니다:
     $$frelu(x) = relu(x) + b_l$$
     여기서 $b_l$은 $l$-번째 레이어의 학습 가능한 파라미터입니다.
   - **순전파(Forward Pass):**
     $$frelu(x) = \begin{cases} x + b_l & \text{if } x > 0 \\ b_l & \text{if } x \le 0 \end{cases}$$
   - **역전파(Backward Pass):**
     $$\frac{\partial frelu(x)}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \le 0 \end{cases}$$
     $$\frac{\partial frelu(x)}{\partial b_l} = 1$$
2. **파라미터 초기화:**
   - 안정적인 역전파를 위해 MSRA [8]와 동일한 초기화 조건 $(1/2) \hat{n_l} Var[w_l] = 1, \forall l$를 따릅니다.
   - Batch Normalization(BN)을 사용하면 초기화 방식에 덜 민감해지므로, 모든 실험에서 MSRA 방법을 사용합니다.
3. **FReLU 분석 및 논의:**
   - **상태 확장(State Extension):** $b < 0$일 때, FReLU는 세 가지 출력 상태(양수, 음수, 비활성화)를 가질 수 있어, ReLU의 두 가지 상태보다 $n$개의 유닛에 대해 $2^n$에서 $3^n$으로 더 많은 출력 상태를 생성하여 표현력을 높입니다.
   - **Batch Normalization과의 호환성:** FReLU의 $max(x,0)$ 구조는 BN의 스케일($\gamma$) 및 편향($\beta$) 파라미터와 FReLU의 $b$ 파라미터 사이의 학습 충돌을 분리하여 호환성을 높입니다.
   - **다른 활성화 함수와의 비교:**
     - ReLU와 비교하여 학습 가능한 편향 $b$를 추가하여 출력 범위를 확장하고 표현력을 높입니다.
     - PReLU/LReLU와 달리 음수 입력에 대한 기울기를 0으로 유지하여 희소성을 보존합니다.
     - ELU와 달리 지수 연산 없이 편향 $b$를 사용하므로 계산 복잡성이 낮고 BN 호환성이 뛰어납니다.

## 📊 Results

- **SmallNet (CIFAR-100):** FReLU는 다른 활성화 함수(ReLU, PReLU, ELU, SReLU) 대비 가장 빠른 수렴 속도와 가장 낮은 테스트 오류율(36.87%)을 달성했습니다. Batch Normalization을 사용할 때도 FReLU의 성능 향상이 두드러졌으며, 특히 높은 학습률(0.1)에서 BN과의 호환성이 뛰어남을 입증했습니다.
- **FReLU 초기화 값의 영향:** $b$의 초기화 값(양수 또는 음수)과 관계없이, FReLU의 학습 가능한 $b$ 파라미터는 유사한 음수 값(약 -0.3 ~ -0.4)으로 수렴하는 경향을 보였습니다. 이는 FReLU가 자기 적응적으로 "제로-유사" 속성을 학습함을 시사합니다.
- **FReLU의 표현력 시각화 (MNIST, LeNets++):** FReLU 레이어의 특징 임베딩이 ReLU보다 더 판별적이며, 더 넓은 특징 표현 공간을 제공함을 보여주었습니다. FReLU 네트워크의 정확도는 97.8%로 ReLU 네트워크의 97.05%보다 높았습니다.
- **NIN (CIFAR-10/100, ImageNet):** NIN 모델에서 FReLU는 CIFAR-10 (7.30%), CIFAR-100 (28.47%), ImageNet (Top-1 34.82%, Top-5 14.00%) 등 모든 데이터셋에서 가장 낮은 오류율을 기록했습니다.
- **Residual Networks (CIFAR-10/100):** ResNet-20부터 ResNet-110까지의 다양한 깊이에서, FReLU는 ReLU를 대체했을 때 원본 ResNet 및 ELU보다 더 낮은 오류율을 달성했습니다. 이는 FReLU가 ResNet 아키텍처 및 BN과 높은 호환성을 가지고 있음을 보여줍니다.

## 🧠 Insights & Discussion

- **음수 값의 중요성:** FReLU의 학습 가능한 편향 $b$가 음수 값으로 수렴하는 경향은 음수 값이 신경망의 표현력과 학습 성능에 긍정적인 영향을 미친다는 것을 시사합니다.
- **적응형 정류 지점의 이점:** 고정된 정류 지점을 사용하는 ReLU와 달리, FReLU는 네트워크가 데이터 분포에 따라 최적의 정류 지점을 유연하게 학습할 수 있도록 하여 성능 향상에 기여합니다.
- **효율성과 호환성의 균형:** FReLU는 ELU의 "제로-유사" 특성과 ReLU의 계산 효율성 및 희소성을 모두 유지하면서, BN과의 호환성 문제까지 해결하여 실용적인 활성화 함수로의 가치를 높입니다.
- **제한 사항 및 향후 연구:** "죽은 뉴런(dead neuron)" 문제에 대한 더 나은 해결책과 음수 값을 더 효과적으로 활용하고 학습 특성을 개선할 수 있는 활성화 함수 설계에 대한 추가 연구가 필요합니다.

## 📌 TL;DR

FReLU는 ReLU의 정류 지점을 학습 가능한 편향 $b$로 대체하여 신경망의 표현력을 높이고 음수 정보를 활용하는 새로운 활성화 함수입니다. FReLU는 계산 비용이 낮고, Batch Normalization과 높은 호환성을 가지며, $b$가 음수 값으로 수렴하면서 "제로-유사" 속성을 학습합니다. CIFAR-10/100 및 ImageNet 데이터셋을 사용한 실험에서 FReLU는 ReLU 및 그 변형들보다 빠르고 안정적인 수렴과 우수한 성능을 보여주었습니다.
