# Deep Learning with S-shaped Rectified Linear Activation Units

Xiaojie Jin, Chunyan Xu, Jiashi Feng, Yunchao Wei, Junjun Xiong, Shuicheng Yan

## 🧩 Problem to Solve

기존의 활성화 함수들(ReLU, LReLU, PReLU, Maxout 등)은 주로 볼록(convex) 함수만을 근사할 수 있어 비볼록(non-convex) 함수를 학습하는 데 제한적인 능력을 가집니다. APL(Adaptive Piecewise Linear)은 비볼록 함수를 근사할 수 있지만, 가장 오른쪽 선형 함수에 부적절한 제약(기울기 1, 절편 0)을 부여하여 표현 능력을 저해합니다. 이로 인해 심층 신경망의 특징 학습 능력이 제한되는 문제가 있습니다.

## ✨ Key Contributions

- **S-shaped Rectified Linear Activation Unit (SReLU) 제안:** 심리 물리학 및 신경 과학의 근본적인 Webner-Fechner 법칙(로그 함수)과 Stevens 법칙(거듭제곱 함수)에서 영감을 받아 볼록 및 비볼록 함수를 모두 학습할 수 있는 새로운 S자형 정류 선형 활성화 단위를 제안했습니다.
- **유연한 파라미터 학습:** SReLU는 네 개의 학습 가능한 파라미터($t^r$, $a^r$, $t^l$, $a^l$)로 구성된 세 개의 구간별 선형 함수로 이루어져, 어떠한 제약 없이 심층 신경망과 함께 백프로파게이션을 통해 학습됩니다.
- **적응형 초기화 방법:** SReLU 파라미터 초기화를 위해 "고정(freezing)" 방법을 제안합니다. 초기 몇 에포크 동안 SReLU를 사전 정의된 LReLU로 퇴화시킨 후, 입력 분포에 맞춰 적응적으로 파라미터의 좋은 초기값을 학습합니다.
- **경량화 및 범용성:** SReLU는 기존 심층 신경망에 미미한 추가 파라미터(예: GoogLeNet에 21.7K)와 계산 비용으로 보편적으로 사용될 수 있습니다.
- **최고 성능 달성:** CIFAR10, CIFAR100, MNIST, ImageNet 등 다양한 스케일의 벤치마크에서 Network in Network 및 GoogLeNet 아키텍처와 함께 사용될 때 다른 활성화 함수들에 비해 현저한 성능 향상을 달성했습니다.

## 📎 Related Works

- **정류 단위(Rectified Units):**
  - **ReLU (Rectified Linear Unit):** $h(x_i) = \max(0, x_i)$
  - **LReLU (Leaky ReLU):** $h(x_i) = \min(0, a_i x_i) + \max(0, x_i)$, 여기서 $a_i \in (0,1)$는 사전 정의된 기울기.
  - **PReLU (Parametric ReLU):** LReLU와 유사하지만, $a_i$ 파라미터가 백프로파게이션을 통해 학습됩니다.
  - **APL (Adaptive Piecewise Linear Units):** 여러 힌지(hinge) 형태의 선형 함수의 합으로 정의되지만, 가장 오른쪽 선의 기울기를 1, 절편을 0으로 강제하는 제약이 있습니다.
  - **Maxout Unit:** 여러 선형 함수의 최대값을 취하며 볼록 함수를 근사할 수 있으나, 많은 추가 파라미터를 요구합니다.
- **심리 물리학 및 신경 과학의 기본 법칙:**
  - **Webner-Fechner 법칙:** 지각된 자극의 크기가 자극 강도의 로그 함수에 비례합니다 ($s = k \log p$).
  - **Stevens 법칙:** 지각된 자극의 크기가 자극 강도의 거듭제곱 함수에 비례합니다 ($s = k p^e$).
    이 두 법칙은 SReLU의 비선형 함수 근사 능력에 대한 영감을 제공합니다.

## 🛠️ Methodology

SReLU는 세 개의 구간별 선형 함수로 정의됩니다.

$$
h(x*i) = \begin{cases} t_i^r + a_i^r(x_i - t_i^r), & x_i \geq t_i^r \\ x_i, & t_i^r > x_i > t_i^l \\ t_i^l + a_i^l(x_i - t_i^l), & x_i \leq t_i^l \end{cases}
$$

여기서 $\{t_i^r, a_i^r, t_i^l, a_i^l\}$는 각 SReLU 활성화 단위에 대한 네 개의 학습 가능한 파라미터입니다.

1. **SReLU 정의:**

   - $x_i \geq t_i^r$일 때, 기울기 $a_i^r$를 가진 선형 함수를 따릅니다.
   - $x_i \leq t_i^l$일 때, 기울기 $a_i^l$를 가진 선형 함수를 따릅니다.
   - $t_i^l < x_i < t_i^r$일 때, 출력은 입력 $x_i$와 동일(기울기 1, 절편 0)합니다. 이 중간 부분은 Webner-Fechner 법칙과 Stevens 법칙의 함수 형태를 더 잘 근사하기 위해 설계되었습니다.
   - 파라미터에 대한 어떠한 제약이나 정규화도 적용하지 않아 높은 유연성을 확보합니다.
   - 각 커널 채널마다 독립적인 SReLU가 학습되며, 전체 네트워크에서 추가되는 파라미터 수는 $4N$ (N은 커널 채널 수)으로 무시할 수 있는 수준입니다.

2. **훈련 과정:**

   - SReLU 파라미터들은 백프로파게이션을 통해 심층 신경망과 함께 공동으로 학습됩니다.
   - 파라미터 $o_i \in \{t_i^r, a_i^r, t_i^l, a_i^l\}$에 대한 기울기는 연쇄 법칙(chain rule)을 사용하여 계산됩니다.
   - 구체적인 파라미터별 기울기는 다음과 같습니다:
     $$
     \begin{aligned}
     \frac{\partial h(x_i)}{\partial t_i^r} &= I\{x_i \geq t_i^r\}(1 - a_i^r)
     \\
     \frac{\partial h(x_i)}{\partial a_i^r} &= I\{x_i \geq t_i^r\}(x_i - t_i^r)
     \\
     \frac{\partial h(x_i)}{\partial t_i^l} &= I\{x_i \leq t_i^l\}(1 - a_i^l)
     \\
     \frac{\partial h(x_i)}{\partial a_i^l} &= I\{x_i \leq t_i^l\}(x_i - t_i^l)
     \end{aligned}
     $$
   - 모멘텀(momentum) 방법을 사용하여 파라미터를 업데이트하며, 파라미터에는 가중치 감쇠(weight decay)를 적용하지 않습니다.

3. **적응형 초기화 ("고정" 방법):**
   - **초기화 문제:** 수동 초기화는 어렵고, 각 레이어의 입력 분포에 따라 부적절할 수 있습니다.
   - **해결책:**
     1. 초기에는 각 SReLU의 파라미터를 LReLU처럼 동작하도록 초기값($\{\tilde{t}_i, 1, 0, \tilde{a}_i\}$)을 설정합니다.
     2. 훈련 초기 몇 에포크 동안 SReLU 파라미터 업데이트를 "고정"하여 SReLU를 LReLU와 동일하게 만듭니다.
     3. "고정" 단계가 끝난 후, $t_i^r$를 해당 SReLU 입력값 전체 중 $k$번째로 큰 값($t_i^r = \text{supp}(X_i, k)$)으로 설정하여 적응적으로 초기화합니다.
   - **장점:** 실제 훈련 데이터의 입력 분포에 더 잘 맞는 초기값을 학습하며, LReLU로 사전 훈련된 모델을 재사용하여 훈련 시간을 단축할 수 있습니다.

## 📊 Results

- **CIFAR-10 및 CIFAR-100:**
  - 데이터 증강(Data Augmentation) 없이: SReLU는 ReLU 대비 CIFAR-10에서 1.26%, CIFAR-100에서 4.86%의 오류율 감소를 달성하며, 모든 비교 방법(LReLU, PReLU, APL, Maxout 등) 중 최고 성능을 기록했습니다.
  - SReLU가 NIN에 추가하는 파라미터는 5.68K로, 전체 NIN 파라미터(0.97M)에 비해 무시할 수 있는 수준입니다.
  - 데이터 증강을 적용한 경우에도 SReLU는 일관되게 우수한 성능을 보였습니다.
- **MNIST:**
  - 데이터 증강 없이: SReLU는 0.35%의 오류율로 다른 방법들보다 뛰어난 성능을 보였습니다.
- **ImageNet (GoogLeNet 사용):**
  - SReLU를 적용한 GoogLeNet은 기존 GoogLeNet 대비 1.24%의 Top-5 오류율 감소(11.1% $\rightarrow$ 9.86%)를 달성했습니다.
  - 이러한 성능 향상은 5M개의 전체 파라미터 중 단 21.6K개의 추가 파라미터만으로 이루어졌습니다.
- **학습된 SReLU 파라미터 분석:**
  - 다른 레이어의 SReLU는 동기 부여와 일치하는 의미 있는 파라미터들을 학습했습니다. 예를 들어, CIFAR-10의 conv1 레이어에서는 $a_r$이 1보다 작게(0.81), CIFAR-100의 conv3 레이어에서는 1보다 크게(1.42) 학습되어 다양한 볼록/비볼록 함수 형태를 학습하는 능력을 입증했습니다.
  - 상위 레이어에서는 입력의 평균값이 높기 때문에 $t_r$ 값이 더 크게 학습되어, SReLU가 입력 분포에 강력하게 적응하는 능력을 보여주었습니다.

## 🧠 Insights & Discussion

SReLU는 심리 물리학적 법칙에서 영감을 받아 볼록 및 비볼록 함수를 모두 학습할 수 있는 유연한 활성화 함수를 제공합니다. 이는 기존 ReLU 계열 활성화 함수들이 가지는 표현 능력의 한계를 극복하며, 신경망이 더욱 복잡한 비선형 변환을 학습할 수 있게 합니다. 특히, 학습 가능한 네 개의 파라미터와 적응형 초기화 전략은 SReLU가 각 레이어의 고유한 입력 분포에 맞춰 최적의 활성화 형태를 찾아내도록 돕습니다.

실험 결과는 SReLU가 다양한 스케일의 데이터셋과 인기 있는 CNN 아키텍처에서 일관되게 최고 성능을 달성하며, 이는 SReLU가 딥러닝 모델의 성능을 향상시키는 데 효과적임을 시사합니다. 추가되는 파라미터와 계산 비용이 미미하다는 점은 SReLU의 실용성을 더욱 높입니다.

이 연구는 활성화 함수의 설계가 단순한 비선형성 도입을 넘어, 생물학적/심리 물리학적 원리를 모방하여 모델의 학습 능력을 심화할 수 있음을 보여줍니다. 향후 연구에서는 SReLU를 컴퓨터 비전 외의 자연어 처리(NLP)와 같은 다른 도메인에 적용하여 그 범용성을 탐구할 계획입니다.

## 📌 TL;DR

**문제:** 기존 ReLU 기반 활성화 함수들은 주로 볼록 함수에 국한되어 비볼록 함수를 학습하는 데 제한적이며, 각 레이어의 입력 분포에 대한 적응력이 부족했습니다.

**방법:** 본 논문은 심리 물리학의 Webner-Fechner 및 Stevens 법칙에서 영감을 받은 새로운 S자형 정류 선형 활성화 단위(SReLU)를 제안합니다. SReLU는 네 개의 학습 가능한 파라미터($t_r, a_r, t_l, a_l$)를 가진 세 개의 구간별 선형 함수로 구성되어, 볼록 및 비볼록 함수를 모두 근사할 수 있습니다. 또한, SReLU는 훈련 초기 "고정" 단계를 통해 입력 분포에 맞춰 파라미터의 초기값을 적응적으로 학습하는 전략을 사용합니다.

**결과:** SReLU는 CIFAR-10, CIFAR-100, MNIST, ImageNet과 같은 다양한 이미지 분류 벤치마크에서 NIN 및 GoogLeNet 아키텍처와 함께 사용될 때, 다른 활성화 함수들보다 현저히 뛰어난 성능을 달성했습니다. 이러한 성능 향상은 전체 네트워크 파라미터 대비 무시할 수 있는 수준의 추가 파라미터와 계산 비용으로 이루어졌으며, SReLU의 다양한 비선형 매핑 학습 능력을 입증했습니다.
