# Empirical Evaluation of Rectified Activations in Convolution Network

Bing Xuantinucleon, Naiyan Wang, Tianqi Chen, Mu Li

## 🧩 Problem to Solve

이 논문은 합성곱 신경망(CNN)에서 다양한 정류 활성화 함수들의 성능을 경험적으로 탐구합니다. 구체적으로, ReLU(Rectified Linear Unit)의 우수한 성능이 활성화의 '희소성(sparsity)'에 기인한다는 일반적인 통념에 의문을 제기하며, 이 희소성이 정말 가장 중요한 요소인지, 그리고 ReLU보다 더 나은 비포화(non-saturated) 활성화 함수를 설계할 수 있는지 두 가지 핵심 질문에 답하고자 합니다.

## ✨ Key Contributions

- ReLU의 음수 부분에 0이 아닌 기울기(non-zero slope)를 도입한 Leaky ReLU의 변형 활성화 함수들이 표준 ReLU보다 일관되게 더 나은 성능을 보인다는 것을 실험적으로 입증했습니다. 이는 ReLU 성능의 핵심이 희소성에 있다는 일반적인 믿음에 반하는 결과입니다.
- 새로운 RReLU(Randomized Leaky Rectified Linear Unit)를 평가하고, 훈련 중 음수 부분의 기울기를 무작위화하는 방식이 작은 규모 데이터셋에서 과적합(overfitting)을 효과적으로 줄여준다는 것을 보여주었습니다.
- CIFAR-100 데이터셋에서 RReLU를 사용하여 앙상블(ensemble)이나 다중 뷰 테스트(multiple view test) 없이 75.68%의 테스트 정확도를 달성했습니다.
- PReLU가 훈련 에러는 가장 낮지만, 작은 규모 데이터셋에서는 과적합에 취약할 수 있음을 발견했습니다. 반면 RReLU는 이러한 과적합 문제에 더 강건함을 입증했습니다.

## 📎 Related Works

- **ReLU (Rectified Linear Unit)**: Nair & Hinton (2010)에 의해 Restricted Boltzmann Machines에서 처음 사용되었으며, 현대 딥러닝에서 비포화 활성화 함수의 대표적인 예로 꼽힙니다.
- **Leaky ReLU (Leaky Rectified Linear Unit)**: Maas et al. (2013)에 의해 음향 모델에서 처음 도입되었으며, 음수 입력에 대해 작은 고정된 양의 기울기를 부여합니다.
- **PReLU (Parametric Rectified Linear Unit)**: He et al. (2015)에 의해 제안되었으며, 음수 입력에 대한 기울기를 역전파(back-propagation)를 통해 데이터로부터 학습합니다. ImageNet 분류 태스크에서 인간 수준의 성능을 뛰어넘는 데 중요한 요소로 언급되었습니다.
- **RReLU (Randomized Leaky Rectified Linear Unit)**: Kaggle NDSB(National Data Science Bowl) 대회에서 처음 제안 및 사용되었으며, 무작위성을 통해 과적합을 줄이는 효과가 보고되었습니다.

## 🛠️ Methodology

논문은 다음 네 가지 정류 활성화 함수를 비교 평가합니다:

1. **ReLU**: 음수 입력을 0으로 설정합니다.
   $$ y_i = \begin{cases} x_i & \text{if } x_i \ge 0 \\ 0 & \text{if } x_i < 0 \end{cases} $$
2. **Leaky ReLU**: 음수 입력에 고정된 상수 $a_i$를 곱하여 0이 아닌 기울기를 부여합니다. 논문에서는 $a_i = 100$과 $a_i = 5.5$ 두 가지 경우를 실험했습니다.
   $$ y_i = \begin{cases} x_i & \text{if } x_i \ge 0 \\ a_i x_i & \text{if } x_i < 0 \end{cases} $$
3. **PReLU**: Leaky ReLU와 동일한 형태지만, 음수 입력에 대한 기울기 $a_i$를 훈련 중 역전파를 통해 데이터로부터 학습합니다.
4. **RReLU**: 훈련 과정에서 음수 입력에 대한 기울기 $a_{ji}$를 균일 분포 $U(l, u)$에서 무작위로 샘플링합니다. (논문 정의상 $l, u \in [0, 1)$이지만, NDSB 우승자 제안에 따라 실험에서는 $U(3, 8)$ 범위가 사용되었습니다.) 테스트 단계에서는 $a_{ji}$를 평균값인 $(l+u)/2$로 고정하여 결정론적인 결과를 얻습니다.
   $$ y*{ji} = \begin{cases} x*{ji} & \text{if } x*{ji} \ge 0 \\ a*{ji} x*{ji} & \text{if } x*{ji} < 0 \end{cases} $$
    테스트 시 ($x*{ji} < 0$일 때): $y*{ji} = x\_{ji} \frac{l+u}{2}$

**실험 설정:**

- **네트워크 구조**: CIFAR-10/CIFAR-100 데이터셋에는 Network in Network (NIN) 구조를 사용했습니다. CIFAR-100의 경우 Batch Norm Inception Network에서도 RReLU를 테스트했습니다. NDSB 데이터셋에는 Kaggle NDSB 우승팀의 네트워크 구조를 사용했습니다.
- **데이터셋**: CIFAR-10, CIFAR-100 (32x32 RGB 이미지), National Data Science Bowl (NDSB, 플랑크톤 분류용 회색조 이미지).
- **평가 지표**: CIFAR 데이터셋에서는 에러율, NDSB 데이터셋에서는 다중 클래스 로그-손실(multi-class log-loss)을 사용했습니다.
- 모든 모델은 CXXNET을 사용하여 훈련되었고, 어떠한 전처리나 증강 없이 원본 이미지를 사용했으며, 단일 뷰 테스트(single view test) 결과를 기반으로 합니다.

## 📊 Results

- **Leaky ReLU 변형의 일관된 우수성**: CIFAR-10, CIFAR-100, NDSB 세 가지 데이터셋 모두에서 Leaky ReLU 및 그 변형(PReLU, RReLU)이 baseline ReLU보다 테스트 성능에서 일관되게 더 나은 결과를 보였습니다.
  - 특히 $a=5.5$인 Leaky ReLU가 $a=100$인 Leaky ReLU보다 훨씬 우수했습니다.
- **PReLU의 과적합 경향**: PReLU는 훈련 세트에서 가장 낮은 에러율을 기록했지만, 테스트 세트에서는 Leaky ReLU ($a=5.5$)나 RReLU보다 성능이 떨어졌습니다. 이는 작은 규모 데이터셋에서 PReLU가 심각한 과적합 문제를 겪을 수 있음을 시사합니다.
- **RReLU의 과적합 방지 효과**: RReLU는 NDSB 데이터셋(훈련 세트 규모는 작지만 네트워크가 더 큰 경우)에서 특히 뛰어난 성능 향상을 보여주었습니다. 이는 RReLU의 무작위성이 과적합을 방지하는 데 효과적임을 검증합니다.
- **최고 성능**: CIFAR-100에서 RReLU는 앙상블이나 다중 뷰 테스트 없이 75.68%의 테스트 정확도를 달성했습니다.

## 🧠 Insights & Discussion

- **희소성에 대한 재평가**: ReLU 성능의 핵심이 '희소성'에 있다는 일반적인 믿음과는 달리, 음수 부분에 0이 아닌 작은 기울기를 허용하는 것이 성능 향상에 더 효과적임을 시사합니다. 이는 활성화 함수의 동작 방식에 대한 새로운 관점을 제공합니다.
- **과적합 제어**: PReLU와 같이 기울기를 학습하는 방식은 작은 데이터셋에서 과적합에 취약할 수 있습니다. 반면 RReLU의 무작위화 기법은 훈련 시 강건성(robustness)을 높여 과적합 위험을 줄이는 데 기여합니다.
- **추가 연구 필요성**: 비록 Leaky ReLU 변형들이 ReLU보다 우수한 성능을 보였지만, 이러한 우수성의 이론적 근거는 여전히 부족합니다. 또한 대규모 데이터셋에서의 성능에 대해서도 추가적인 조사가 필요합니다. RReLU의 무작위성이 네트워크 훈련 및 테스트 과정에 어떻게 영향을 미치는지에 대한 심층적인 연구도 필요합니다.

## 📌 TL;DR

이 논문은 합성곱 신경망에서 ReLU, Leaky ReLU, PReLU, 그리고 새로운 RReLU를 포함한 정류 활성화 함수들의 성능을 경험적으로 평가합니다. 연구 결과, ReLU의 성능이 희소성 때문이라는 통념에 의문을 제기하며, 음수 입력에 대해 0이 아닌 기울기를 허용하는 Leaky ReLU 변형들이 ReLU보다 일관되게 우수함을 발견했습니다. 특히 RReLU는 훈련 중 음수 기울기를 무작위화함으로써 작은 데이터셋에서의 과적합을 효과적으로 줄여주며, CIFAR-100에서 높은 정확도를 달성했습니다. 이는 활성화 함수 설계 시 음수 부분의 처리가 성능에 중요한 영향을 미치며, 무작위성이 과적합 방지에 유용함을 보여줍니다.
