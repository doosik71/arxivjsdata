# Mathematical theory of deep learning

Philipp Petersen and Jakob Zech (2025)

## 🧩 Problem to Solve

본 논문(또는 저서)은 딥러닝의 폭발적인 실무적 성공에도 불구하고, 왜 이러한 모델들이 실제로 잘 작동하는지에 대한 엄밀한 수학적 이해가 부족하다는 문제의식에서 출발한다. 특히, 딥러닝의 성공을 뒷받침하는 세 가지 핵심 축인 **근사 이론(Approximation Theory)**, **최적화 이론(Optimization Theory)**, 그리고 **통계적 학습 이론(Statistical Learning Theory)** 사이의 이론적 간극을 메우고자 한다.

연구의 주된 목표는 딥러닝 모델의 표현력(Expressivity), 학습 역학(Training Dynamics), 그리고 일반화 성능(Generalization Performance)을 수학적으로 규명하는 것이다. 특히 고차원 데이터에서 발생하는 '차원의 저주(Curse of Dimensionality)'를 어떻게 극복하는지, 그리고 모델의 파라미터 수가 학습 데이터보다 훨씬 많은 '과잉 매개변수화(Overparameterization)' 상태에서도 왜 과적합(Overfitting)이 발생하지 않고 성능이 향상되는지를 이론적으로 설명하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 딥러닝의 이론적 토대를 세 가지 관점에서 체계적으로 통합하고 분석한 점에 있다.

1.  **표현력의 수학적 증명**: ReLU 신경망이 연속 조각별 선형 함수(Continuous Piecewise Linear, CPWL) 공간과 동일함을 증명하고, 깊이(Depth)가 증가함에 따라 표현 가능한 선형 영역(Linear Regions)의 수가 지수적으로 증가함을 보였다. 또한, $C^{k,s}$ 클래스의 함수에 대해 딥러닝 모델이 달성할 수 있는 최적의 근사 속도를 정량화하였다.
2.  **학습 역학의 선형화 분석**: 신경망의 너비(Width)가 무한대로 갈 때, 학습 과정이 **Neural Tangent Kernel (NTK)** 기반의 선형 모델로 수렴함을 분석하였다. 이를 통해 비볼록(Non-convex) 최적화 문제임에도 불구하고 경사 하강법(Gradient Descent)이 어떻게 전역 최적해(Global Minimum)에 도달할 수 있는지를 수학적으로 설명하였다.
 uma 3.  **일반화의 패러다임 전환**: 전통적인 통계적 학습 이론의 복잡도-일반화 트레이드오프를 넘어, 최근 관찰되는 **이중 하강(Double Descent)** 현상을 이론적으로 분석하였다. 파라미터 수가 임계점을 넘어서는 과잉 매개변수화 영역에서 일반화 오차가 다시 감소하는 메커니즘을 규명하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들을 기반으로 하며, 그 한계를 보완한다.

-   **Universal Approximation Theorems**: Cybenko와 Hornik 등의 초기 연구는 단층 신경망이 충분한 너비를 가질 때 임의의 연속 함수를 근사할 수 있음을 보였으나, 이는 '효율성'이나 '깊이의 이점'에 대해서는 설명하지 못했다. 본 연구는 이를 확장하여 깊은 구조가 파라미터 효율성을 어떻게 높이는지 분석한다.
-   **VC Dimension 및 Covering Numbers**: Vapnik-Chervonenkis 이론은 모델의 복잡도가 높을수록 일반화 오차가 커진다고 예측한다. 그러나 현대의 거대 모델들은 이 이론적 예측과 달리 매우 높은 일반화 성능을 보인다. 본 연구는 이를 해결하기 위해 NTK와 과잉 매개변수화 이론을 도입한다.
-   **Adversarial Examples**: Szegedy 등의 연구에서 발견된 적대적 예제 문제를 다루며, 이를 방어하기 위한 Lipschitz 연속성 및 Local Regularity의 중요성을 수학적으로 연결한다.

## 🛠️ Methodology

본 연구는 딥러닝 시스템을 세 가지 수학적 파이프라인으로 나누어 분석한다.

### 1. 근사 이론 (Approximation Theory)
신경망 $\Phi$를 함수 공간의 부분집합으로 정의하고, 목표 함수 $f$와의 거리 $\inf_{\Phi \in \mathcal{H}} \|f - \Phi\|$를 분석한다.
-   **ReLU 네트워크의 특성**: ReLU 네트워크는 CPWL 함수임을 보였으며, 깊이 $L$과 너비 $n$에 대해 생성 가능한 선형 영역의 수는 최대 $(p \cdot n)^L$에 비례함을 증명하였다.
-   **근사 속도**: $C^{k,s}$ 함수에 대해 ReLU 네트워크의 근사 오차는 모델 크기 $N$에 대해 $O(N^{-(k+s)/d})$의 속도로 감소함을 보였다.

### 2. 최적화 이론 (Optimization Theory)
가중치 $w$에 대한 손실 함수 $f(w)$의 최소화 과정을 분석한다.
-   **경사 하강법 (GD)**: 업데이트 식 $w_{k+1} = w_k - h_k \nabla f(w_k)$를 기반으로, $\mu$-강볼록(Strongly Convex) 및 $L$-매끄러움(Smooth) 조건 하에서의 선형 수렴성을 증명하였다.
-   **Neural Tangent Kernel (NTK)**: 무한 너비 한계에서 모델의 출력을 다음과 같이 선형화한다.
$$\Phi(x, w) \approx \Phi(x, w_0) + \langle \nabla_w \Phi(x, w_0), w - w_0 \rangle$$
이때 커널 $\hat{K}_n(x, z) = \langle \nabla_w \Phi(x, w_0), \nabla_w \Phi(z, w_0) \rangle$가 학습 역학을 결정하며, 이는 커널 리지 회귀(Kernel Ridge Regression)와 동일한 거동을 보임을 보였다.

### 3. 통계적 학습 이론 (Statistical Learning Theory)
경험적 위험(Empirical Risk) $\hat{R}_S(\Phi)$와 실제 위험(Population Risk) $R(\Phi)$ 사이의 간극인 일반화 오차 $\varepsilon_{gen}$을 분석한다.
-   **Covering Number**: 함수 공간 $\mathcal{H}$를 $\varepsilon$-볼로 덮는 최소 개수 $G(\mathcal{H}, \varepsilon, \|\cdot\|_\infty)$를 이용하여 일반화 경계(Generalization Bound)를 유도하였다.
-   **Double Descent**: 모델의 복잡도가 증가함에 따라 일반화 오차가 증가하다가, 보간 임계값(Interpolation Threshold)을 지나 과잉 매개변수화 영역으로 진입하면 다시 감소하는 현상을 최소 노름 솔루션(Minimum Norm Solution) 관점에서 설명하였다.

## 📊 Results

본 연구는 이론적 증명을 통해 다음과 같은 정량적/정성적 결과를 도출하였다.

-   **깊이의 효율성**: 동일한 근사 오차 $\varepsilon$를 달성하기 위해 얕은 네트워크는 파라미터 수가 다항적으로 증가해야 하지만, 깊은 네트워크는 $\log(1/\varepsilon)$ 수준의 깊이 증가만으로도 효율적인 근사가 가능함을 보였다.
-   **차원의 저주 극복**: 일반적인 함수 공간에서는 $O(\varepsilon^{-d/k})$의 파라미터가 필요하지만, **Barron Class**에 속하는 함수나 **Compositional Structure**를 가진 함수, 또는 **Low-dimensional Manifold** 상의 데이터의 경우, 차원 $d$에 독립적이거나 훨씬 낮은 의존성을 가진 근사 속도를 가짐을 증명하였다.
-   **NTK 수렴성**: 충분히 너비가 넓은 신경망은 경사 하강법을 통해 높은 확률로 전역 최적해에 수렴하며, 그 결과는 NTK 커널을 이용한 최소 제곱 추정치와 일치함을 보였다.
-   **과잉 매개변수화의 이점**: 파라미터 수가 데이터 수 $m$보다 훨씬 많을 때, 모델은 데이터를 완벽하게 보간(Interpolation)하면서도, 가중치의 노름이 제어된다면(Implicit Bias) 오히려 더 좋은 일반화 성능을 보임을 이론적으로 확인하였다.

## 🧠 Insights & Discussion

**강점 및 통찰**:
본 연구는 딥러닝의 '블랙박스'적 성격을 수학적인 언어로 체계화하였다. 특히, 단순히 "깊은 모델이 좋다"는 직관을 넘어, 깊이가 선형 영역의 수를 지수적으로 늘려 복잡한 함수를 효율적으로 표현하게 한다는 점과, 무한 너비 한계에서의 선형화(NTK)가 최적화 가능성을 보장한다는 점을 연결한 것이 매우 강력한 통찰이다.

**한계 및 논의사항**:
1.  **가정의 현실성**: NTK 분석이나 일반화 경계 유도를 위해 사용된 '무한 너비'나 '가중치 유계(Bounded Weights)' 등의 가정은 실제 모델의 유한한 크기와 다양한 가중치 분포를 완전히 반영하지 못할 수 있다.
2.  **아키텍처의 다양성**: 본 논문은 주로 Feedforward NN을 다루고 있으나, 실제로는 CNN, Transformer 등 특수한 구조가 일반화 성능에 기여하는 바가 크다. 이러한 구조적 유도 편향(Inductive Bias)이 수학적으로 어떻게 반영되는지에 대한 추가 연구가 필요하다.
3.  **최적화 알고리즘의 영향**: Adam과 같은 적응형 학습률 알고리즘이 단순 GD와 비교하여 실제 손실 곡면(Loss Landscape)을 어떻게 탐색하는지에 대한 엄밀한 분석은 여전히 도전적인 과제이다.

## 📌 TL;DR

본 연구는 딥러닝의 **근사-최적화-일반화**라는 세 가지 핵심 축을 수학적으로 통합 분석한 보고서이다. 딥러닝이 고차원 데이터의 '차원의 저주'를 극복하는 방법(Barron space, Compositionality), 깊은 구조가 표현력을 지수적으로 높이는 원리, 그리고 과잉 매개변수화 상태에서 일반화 성능이 향상되는 '이중 하강' 현상을 이론적으로 규명하였다. 이 연구는 향후 더 효율적이고 안정적인 신경망 아키텍처 설계와 학습 알고리즘 개발을 위한 수학적 가이드라인을 제공한다.