# Parametric Exponential Linear Unit for Deep Convolutional Neural Networks

Ludovic Trottier, Philippe Gigu`ere, Brahim Chaib-draa

## 🧩 Problem to Solve

최근 제안된 ELU(Exponential Linear Unit) 활성화 함수는 CNN(Convolutional Neural Networks)에서 바이어스 시프트(bias shift)를 효과적으로 관리하여 학습을 안정화하는 데 기여합니다. 그러나 ELU의 핵심 파라미터인 $a$는 수동으로 설정해야 하며, 특정 네트워크에서는 특정 파라미터가 더 적합할 수 있어 최적의 값을 찾기 어렵다는 한계가 있습니다. 이 논문은 이러한 수동 설정의 제약을 해결하고 각 레이어에서 적절한 활성화 함수 형태를 학습하는 방법을 제시하고자 합니다.

## ✨ Key Contributions

- **학습 가능한 ELU 파라미터화 정의**: 함수 $f(h)$의 양수 부분과 음수 부분에 모두 작용하여 미분 가능성을 유지하고 역전파 중에 학습될 수 있는 파라미터를 정의했습니다. 이 파라미터화는 단 $2L$ (여기서 $L$은 레이어 수)개의 추가 파라미터만 필요로 하며, ELU와 동일한 계산 복잡성을 가집니다.
- **광범위한 실험적 평가**: MNIST, CIFAR-10/100, ImageNet 데이터셋과 ResNet, NiN, All-CNN, Vgg, Overfeat 네트워크를 사용하여 PELU(Parametric ELU)가 비매개변수 ELU보다 우수한 성능을 보임을 입증했습니다.
- **배치 정규화(BN) 효과 분석**: PELU 활성화 전에 배치 정규화를 사용할 경우 ResNet의 오류율이 증가한다는 것을 보여주었습니다.
- **다양한 PELU 파라미터화 실험**: 가능한 여러 파라미터화 중 제안된 구성이 가장 좋은 성능을 얻음을 확인했습니다.
- **학습된 비선형 동작 시각화**: Vgg 네트워크가 학습 과정에서 PELU를 통해 채택한 다양한 비선형 활성화 형태를 시각적으로 분석하여, 네트워크가 추가된 유연성을 활용함을 보여주었습니다.

## 📎 Related Works

- **PReLU (Parametric ReLU)**: Leaky ReLU의 leak 파라미터를 학습하여 음수 입력에 대한 적절한 양수 기울기를 찾습니다. PELU는 PReLU가 ReLU의 성능을 개선했듯이 ELU의 성능을 향상시키는 것을 목표로 합니다.
- **APL (Adaptive Piecewise Linear)**: 여러 매개변수화된 힌지(Hinge) 함수의 가중합을 학습합니다. 하지만, 비미분점의 수가 증가하고, 가장 오른쪽 선형 함수에 특정 제약이 있어 CNN의 특징 학습 능력을 저해할 수 있습니다.
- **Maxout**: 각 입력 뉴런에 대해 $K$개의 아핀(affine) 함수 중 최댓값을 출력합니다. 주요 단점은 각 레이어에서 학습해야 할 가중치의 양을 $K$배로 증가시켜 깊은 네트워크에 계산 부담을 줍니다. PELU는 Maxout과 달리 $2L$개의 적은 파라미터만 추가합니다.
- **SReLU (S-Shaped ReLU)**: 세 개의 선형 함수 조합을 학습합니다. 두 개의 비미분점을 가집니다. PELU는 SReLU와 달리 함수 양수 및 음수 부분 모두에 파라미터화가 작용하여 완전히 미분 가능합니다.

## 🛠️ Methodology

기존 ELU 함수 $f(h) = \begin{cases} h & \text{if } h \geq 0 \\ \alpha(\exp(h) - 1) & \text{if } h < 0 \end{cases}$ (여기서 $\alpha=1$)의 수동 설정 파라미터 문제를 해결하기 위해, 논문은 세 가지 파라미터 $a, b, c$를 도입합니다.

1. **초기 파라미터화**:
   $$f(h) = \begin{cases} ch & \text{if } h \geq 0 \\ a\left(\exp\left(\frac{h}{b}\right) - 1\right) & \text{if } h < 0 \end{cases}, \quad a, b, c > 0$$
   - 파라미터 $c$는 양수 영역의 선형 기울기를, $b$는 지수 감쇠의 스케일을, $a$는 음수 영역의 포화점(saturation point)을 제어합니다.
2. **미분 가능성 유지**: $h=0$에서 함수가 미분 가능하도록 좌미분계수와 우미분계수를 같게 설정합니다.
   - $f'(0^+) = c$
   - $f'(0^-) = \frac{a}{b}\exp\left(\frac{h}{b}\right)|_{h=0} = \frac{a}{b}$
   - 따라서 $c = \frac{a}{b}$가 됩니다.
3. **최종 PELU 정의**: $c$를 $\frac{a}{b}$로 대체하여 미분 가능한 PELU 함수를 제안합니다.
   $$f(h) = \begin{cases} \frac{a}{b}h & \text{if } h \geq 0 \\ a\left(\exp\left(\frac{h}{b}\right) - 1\right) & \text{if } h < 0 \end{cases}, \quad a, b > 0$$
4. **역전파를 위한 그래디언트**: PELU는 네트워크의 모든 파라미터와 동시에 역전파를 통해 학습됩니다. $a$와 $b$에 대한 그래디언트는 다음과 같습니다.
   $$\frac{\partial f(h)}{\partial a} = \begin{cases} \frac{h}{b} & \text{if } h \geq 0 \\ \exp(h/b) - 1 & \text{if } h < 0 \end{cases}$$
   $$\frac{\partial f(h)}{\partial b} = \begin{cases} -\frac{ah}{b^2} & \text{if } h \geq 0 \\ -\frac{a}{b^2}\exp(h/b) & \text{if } h < 0 \end{cases}$$
   - 학습 중 파라미터 $a, b$의 양수 값을 유지하기 위해 항상 $0.1$보다 크도록 제한합니다.

## 📊 Results

- **MNIST Auto-Encoder**: PELU는 ELU 및 BN-ReLU보다 더 빠른 수렴 속도와 낮은 재구성 오류를 달성했습니다. (PELU MSE $1.04 \times 10^{-4}$ vs ELU MSE $1.12 \times 10^{-4}$)
- **CIFAR-10/100 Object Recognition**:
  - ResNet-110에서 PELU는 ELU보다 지속적으로 낮은 오류율을 보였습니다. CIFAR-10에서 PELU는 5.36%의 최소 중앙값 오류율을 기록하며 ELU(5.99%) 대비 10.52%의 상대적 개선을 이루었습니다. CIFAR-100에서는 24.55%로 ELU(25.08%) 대비 2.11% 개선되었습니다.
  - PELU는 ELU에 비해 수렴 행동이 더 안정적이었으며, 훈련 후반부의 오류율 증가가 적었습니다.
  - 추가된 파라미터는 112개로 전체 파라미터 수의 0.006%에 불과하여 매우 미미했습니다.
- **Batch Normalization (BN) 효과**: PELU 또는 ELU 활성화 전에 BN을 사용하는 것은 CIFAR-10/100 ResNet 실험에서 성능 저하를 야기했습니다. PELU의 경우, BN을 사용하면 오류율이 증가했지만, ELU보다는 증가폭이 작아 PELU가 BN의 해로운 영향을 다소 줄일 수 있음을 시사했습니다.
- **ImageNet Object Recognition**: ResNet18, NiN, All-CNN, Overfeat 등 4가지 네트워크 아키텍처에서 PELU는 ELU보다 일관되게 우수한 성능을 보였습니다. NiN 네트워크에서 ImageNet TOP-1 오류율 36.06%를 달성하여 ELU(40.40%) 대비 7.29%의 상대적 개선을 이루었으며, 추가된 파라미터는 24개에 불과했습니다.
- **파라미터 구성 실험**: 제안된 $(a, 1/b)$ 파라미터화가 $(a, b)$, $(1/a, b)$, $(1/a, 1/b)$ 중 가장 우수한 성능을 보였습니다. $1/b$를 사용하는 구성이 활성화 함수의 비선형적 특성을 유지하는 데 도움이 되어 더 나은 성능을 이끌었습니다.
- **파라미터 변화 시각화 (Vgg 네트워크)**: Vgg 네트워크는 훈련 중 각 레이어에서 다양한 비선형 활성화 형태를 학습했습니다. 일부 레이어는 ReLU와 유사한 형태(기울기 $a/b \approx 1$, 음수 포화 $a \approx 0$)를 학습하여 스파스성(sparsity)을 촉진하는 것으로 보였습니다. 대부분의 레이어는 0이 아닌 음수 포화 값을 가진 활성화 함수를 학습했으며, 이는 바이어스 시프트 관리에서 음수 값의 중요성을 강조합니다.

## 🧠 Insights & Discussion

- PELU의 유연성은 네트워크가 각 레이어에 맞춰 활성화 함수의 형태를 동적으로 학습하게 함으로써 성능 향상을 가져옵니다. 추가 파라미터의 수가 미미하다는 점은 성능 개선이 네트워크 용량 증가보다는 활성화 함수의 적응성에서 비롯됨을 시사합니다.
- ReLU와 달리 ELU와 PELU 앞에서 배치 정규화(BN)를 사용하는 것이 성능을 저하시키는 현상이 관찰되었습니다. 이는 ReLU가 양의 스케일 불변성(positively scale invariant)을 가지는 반면, ELU는 그렇지 않다는 점에서 기인할 수 있습니다. BN이 평균 및 표준편차 스케일링을 수행한 후 아핀 변환을 적용하기 때문에, 양의 스케일 불변 활성화 함수가 BN이 내부 공변량 시프트(internal covariate shift)를 효과적으로 줄이는 데 필수적일 수 있습니다. 이 가설의 검증은 향후 연구 과제로 남겨두었습니다.
- 네트워크가 훈련 초기에 기울기를 크게 증가시킨 후 ReLU와 유사한 형태로 수렴하는 경향은 중복된 뉴런을 분리하고 입력 값을 더 넓게 분산시켜 학습을 돕는 메커니즘일 수 있습니다. 또한, 대부분의 레이어에서 0이 아닌 음수 포화 값을 학습한 것은 바이어스 시프트를 관리하는 데 있어 음수 값의 중요성을 재확인합니다.

## 📌 TL;DR

**문제**: ELU 활성화 함수의 파라미터는 수동으로 설정해야 하므로 최적화에 한계가 있습니다.
**제안 방법**: ELU의 파라미터 $a, b$를 네트워크와 함께 학습하도록 하는 PELU(Parametric ELU)를 제안했습니다. PELU는 $h=0$에서 미분 가능성을 유지하도록 설계되었으며, 단 $2L$개의 추가 파라미터만 필요로 합니다.
**주요 결과**: PELU는 MNIST, CIFAR-10/100, ImageNet 등 다양한 데이터셋과 ResNet, NiN, All-CNN 등 여러 CNN 아키텍처에서 ELU보다 뛰어난 성능을 일관되게 보여주었습니다. 특히 ImageNet NiN에서는 7.29%의 상대적 오류 개선을 달성했습니다. 네트워크는 PELU를 통해 각 레이어에 맞는 최적의 활성화 함수 형태를 학습하는 것으로 나타났습니다. 또한, PELU/ELU 활성화 전에 배치 정규화를 사용하는 것은 성능을 저하시켰습니다.
