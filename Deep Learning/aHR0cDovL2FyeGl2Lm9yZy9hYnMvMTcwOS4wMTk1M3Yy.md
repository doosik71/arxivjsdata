# Implicit Regularization in Deep Learning

Behnam Neyshabur (2017)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델이 가진 **과잉 매개변수화(Over-parameterization)** 특성에도 불구하고, 왜 실제 환경에서 뛰어난 **일반화(Generalization)** 성능을 보이는지를 이론적, 실험적으로 분석하는 것을 목표로 한다.

일반적인 통계적 학습 이론에 따르면, 모델의 파라미터 수가 학습 데이터 수보다 훨씬 많을 경우 모델은 훈련 데이터를 단순히 암기(memorization)하게 되어 과적합(overfitting)이 발생해야 한다. 특히 VC-dimension과 같은 전통적인 복잡도 측정 방식은 파라미터 수에 비례하므로, 현대의 거대 신경망이 가진 높은 용량을 설명하지 못하며 이는 이론과 실제 사이의 큰 간극을 만든다.

따라서 본 연구는 네트워크의 물리적인 크기(파라미터 수)가 아닌, 최적화 알고리즘에 의해 유도되는 **암묵적 정규화(Implicit Regularization)**가 일반화의 핵심 역할을 한다는 가설을 세우고, 이를 설명할 수 있는 적절한 복잡도 측정 지표와 최적화 방법론을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 딥러닝의 일반화가 네트워크의 크기가 아니라, 최적화 과정에서 선택되는 **가중치의 노름(Norm)과 기하학적 구조**에 의해 결정된다는 것이다. 주요 기여 사항은 다음과 같다.

1.  **암묵적 정규화의 역할 규명**: 네트워크 크기를 키워도 테스트 오차가 계속 감소하는 현상을 통해, 파라미터 수가 아닌 다른 암묵적인 복잡도 제어 기제가 작동하고 있음을 실험적으로 증명한다.
2.  **노름 기반 복잡도 제어**: 네트워크 크기에 의존하지 않고 가중치의 노름(Norm)만으로 일반화 경계를 정의할 수 있는 Group-norm 정규화 프레임워크를 제안한다.
3.  **Sharpness와 PAC-Bayes의 연결**: 파라미터 공간에서의 평탄함(Sharpness)이 일반화 성능과 밀접한 관련이 있음을 PAC-Bayesian 프레임워크를 통해 이론적으로 분석한다.
4.  **불변성(Invariance) 기반 최적화**: ReLU 네트워크의 **노드별 재스케일링(Node-wise rescaling)** 불변성을 분석하고, 이에 최적화된 **Path-SGD** 알고리즘을 제안하여 학습 안정성과 일반화 성능을 높인다.
5.  **통합 프레임워크 제안**: Path-SGD, Batch Normalization, Natural Gradient를 하나의 데이터 의존적 경로 정규화(DDP) 관점에서 통합하여 설명한다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들의 한계와 차별점을 제시한다.

-   **VC-Dimension 기반 접근**: 네트워크의 에지(edge) 수에 비례하여 복잡도를 측정한다. 하지만 과잉 매개변수화된 모델에서는 이 수치가 너무 커서 일반화 성능을 설명하는 데 무의미하다.
-   **노름 기반 경계(Norm-based bounds)**: 가중치의 $\ell_1, \ell_2$ 노름을 사용하여 복잡도를 제한하려는 시도가 있었다. 본 논문은 이를 확장하여 다양한 Group-norm $\mu_{p,q}$를 제안하고, 깊이에 따른 지수적 의존성 문제를 분석한다.
-   **Flat Minima (Sharpness)**: 최적화 결과가 평탄한 최소값(Flat minima)에 도달할수록 일반화가 잘 된다는 직관이 있었으나, 본 논문은 이를 PAC-Bayes 이론과 결합하여 보다 엄밀한 일반성 보장(Generalization guarantee)을 시도한다.
-   **Natural Gradient**: 파라미터 공간의 기하학적 구조를 고려하여 불변성을 확보하려 하지만, 계산 비용이 매우 높다는 한계가 있다. 본 논문은 이를 효율적으로 근사한 DDP-SGD를 제안한다.

## 🛠️ Methodology

### 1. Group-Norm 및 복잡도 측정
네트워크의 복잡도를 측정하기 위해 다음과 같은 $\mu_{p,q}$ 정규화 식을 정의한다.

$$\mu_{p,q}(w) = \left( \sum_{v \in V} \left( \sum_{(u \to v) \in E} |w_{u \to v}|^p \right)^{q/p} \right)^{1/q}$$

-   $p$는 개별 유닛으로 들어오는 가중치의 노름을, $q$는 유닛 간의 노름을 합산하는 방식을 결정한다.
-   $q = \infty$인 경우 **Per-unit regularization**이 되며, $q = p$인 경우 **Overall regularization** (예: Weight Decay)이 된다.
-   ReLU의 동차성(Homogeneity) 덕분에, 가중치를 적절히 재분배하면 $\mu_{p,q}$는 경로 기반 측정치인 $\psi_{p,q}$와 등가임을 증명한다.

### 2. PAC-Bayesian Framework와 Sharpness
파라미터 $\mathbf{w}$에 섭동(perturbation) $\mathbf{u}$를 가했을 때의 기대 손실을 통해 일반화를 분석한다.

$$E_{u} [L(f_{\mathbf{w}+\mathbf{u}})] \leq \hat{L}(f_{\mathbf{w}}) + \text{expected sharpness} + \text{KL divergence term}$$

여기서 **Expected Sharpness**는 $\mathbf{w}$ 주변의 손실 함수가 얼마나 급격히 변하는지를 측정하며, 이는 $\mathbf{w}$의 노름과 트레이드-오프 관계에 있다. 즉, 노름을 줄이면 Sharpness가 커질 수 있으므로, 두 지표의 균형을 맞추는 것이 일반화의 핵심이다.

### 3. Path-SGD 및 불변성 (Invariances)
ReLU 네트워크는 가중치를 $w_{u \to v} \cdot \alpha$와 $w_{v \to k} / \alpha$로 변경해도 동일한 함수를 출력하는 **Node-wise rescaling** 불변성을 가진다. 하지만 일반적인 SGD는 이 불변성을 보장하지 않아, 가중치가 불균형한(unbalanced) 네트워크에서 학습 성능이 급격히 떨어진다.

이를 해결하기 위해 **Path-regularizer** $\gamma_{\text{net}}^2(w)$를 도입한다. 이는 입력에서 출력까지의 모든 경로에 대해 가중치 곱의 제곱 합으로 정의된다.

$$\gamma_{\text{net}}^2(w) = \sum_{\zeta \in P} \prod_{e \in \zeta} w_e^2$$

**Path-SGD**는 이 Path-norm에 대한 최급강하법(Steepest Descent)으로, 업데이트 식은 다음과 같다.

$$w_{t+1} = w_t - \eta \kappa^{-1} \nabla L(w_t)$$

여기서 $\kappa$는 Path-regularizer의 2차 미분 값으로, 이를 통해 가중치 스케일에 상관없이 일관된 업데이트 방향을 유지하며 불변성을 확보한다.

### 4. Unified Framework (DDP)
데이터 의존적 경로 정규화(Data-Dependent Path, DDP) 프레임워크를 통해 다음과 같이 정의한다.

$$\gamma_v(w) = \sqrt{\mathbf{w}_{\to v}^\top R_v \mathbf{w}_{\to v}}$$

-   $R_v$가 대각 행렬이면 **Path-SGD**가 된다.
-   $R_v$가 공분산 행렬(Covariance)이면 **Batch Normalization**의 기하학적 해석이 된다.
-   $R_v$가 2차 모멘(Second Moment) 행렬이면 **Diagonal Natural Gradient**와 동일해진다.

## 📊 Results

### 1. 네트워크 크기와 일반화 (Fig 4.1, 4.2)
-   **실험**: MNIST와 CIFAR-10 데이터셋에서 은닉층 유닛 수 $H$를 증가시키며 학습.
-   **결과**: 훈련 오차가 0이 된 이후에도 $H$를 더 키우면 테스트 오차가 오히려 감소하는 현상이 관찰되었다. 이는 파라미터 수 자체가 복잡도를 결정하는 것이 아니라, 최적화 과정의 암묵적 정규화가 작동하고 있음을 시사한다.

### 2. True Labels vs. Random Labels (Fig 7.1)
-   **실험**: 실제 라벨과 무작위 라벨로 학습된 모델의 노름 기반 복잡도 비교.
-   **결과**: 실제 라벨로 학습된 모델이 무작위 라벨 모델보다 훨씬 낮은 복잡도(Norm)를 가졌으며, 데이터 수가 늘어날수록 이 격차가 커졌다. 이는 $\ell_2$ 노름과 Path-norm이 일반화 성능을 잘 설명하는 지표임을 보여준다.

### 3. Path-SGD의 성능 (Fig 10.1, 10.2)
-   **실험**: MNIST, CIFAR-10/100, SVHN 데이터셋에서 SGD, AdaGrad와 비교.
-   **결과**: Path-SGD는 가중치가 불균형하게 초기화된 환경에서도 SGD/AdaGrad와 달리 안정적으로 학습되었으며, 최종 테스트 오차 또한 더 낮게 나타났다.

### 4. RNN 및 장기 의존성 문제 (Fig 10.4, Table 10.2)
-   **실험**: Addition problem (시퀀스 길이 $T=100, 400, 750$) 및 Sequential MNIST 수행.
-   **결과**: 기존의 IRNN, uRNN, LSTM 등이 $T=750$과 같은 매우 긴 시퀀스에서 실패한 반면, **RNN-Path**는 0%의 오차를 기록하며 탁월한 장기 의존성 학습 능력을 보였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 가치
본 논문은 딥러닝의 일반화 미스터리를 '파라미터 수'라는 정적인 관점이 아닌, '최적화 기하학'이라는 동적인 관점에서 풀이했다. 특히 단순한 경험적 관찰을 넘어, Path-norm이라는 구체적인 지표와 이를 활용한 Path-SGD라는 알고리즘으로 연결하여 이론과 실무의 간극을 좁혔다.

### 한계 및 비판적 해석
-   **깊이에 대한 의존성**: 노름 기반 복잡도 제어를 시도했음에도 불구하고, 일반화 경계에서 깊이(depth)에 대한 지수적 의존성을 완전히 제거하지 못했다. 이는 깊은 신경망의 복잡도를 측정하는 것이 여전히 매우 어려운 문제임을 보여준다.
-   **계산 복잡도**: Path-SGD의 $\kappa^{(2)}$ 항(RNN의 경우)은 계산 비용이 높다. 비록 실험적으로 $\kappa^{(1)}$만으로도 충분함을 보였으나, 이는 특정 태스크에 국한된 결과일 수 있다.
-   **ReLU 중심의 분석**: 본 연구의 불변성 및 정규화 분석은 주로 ReLU의 동차성에 기반하고 있다. Sigmoid나 Tanh와 같은 다른 활성화 함수에서도 동일한 메커니즘이 작동하는지에 대한 논의가 부족하다.

## 📌 TL;DR

이 논문은 딥러닝 모델의 일반화가 네트워크의 크기가 아니라, 최적화 알고리즘이 유도하는 **암묵적 정규화(Implicit Regularization)**에 의해 결정된다는 것을 증명한다. 저자는 가중치의 노름(Norm)과 경로(Path) 기반 복잡도 지표를 제안하고, 노드별 재스케일링 불변성을 갖는 **Path-SGD** 최적화 알고리즘을 통해 과잉 매개변수화된 모델, 특히 RNN에서의 학습 안정성과 일반화 성능을 획기적으로 개선하였다. 이 연구는 딥러닝의 최적화 기하학(Geometry of Optimization)과 일반화 사이의 관계를 이해하는 데 중요한 이론적 토대를 제공한다.