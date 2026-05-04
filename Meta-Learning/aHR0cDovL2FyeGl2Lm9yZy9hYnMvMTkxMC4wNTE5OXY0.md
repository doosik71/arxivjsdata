# Bayesian Meta-Learning for the Few-Shot Setting via Deep Kernels

Massimiliano Patacchiola, Jack Turner, Elliot J. Crowley, Michael O’Boyle, Amos Storkey (2020)

## 🧩 Problem to Solve

본 논문은 매우 적은 양의 레이블링된 데이터만으로 새로운 태스크를 학습해야 하는 Few-Shot Learning(FSL) 문제를 해결하고자 한다. 기존의 딥러닝 모델은 패턴을 찾기 위해 방대한 양의 데이터를 필요로 하며, 데이터가 부족한 상황에서 예측의 불확실성(uncertainty)을 정량화하는 데 어려움을 겪는다.

전형적인 Meta-learning 접근 방식은 모든 태스크에 공통적인 파라미터 $\theta$와 각 태스크별 파라미터 $\rho_t$를 구분하여 학습하는 계층적 모델 구조를 가진다. 그러나 MAML과 같은 미분 가능한 메타 러닝 방법론은 inner loop와 outer loop의 공동 최적화 과정에서 학습이 불안정해지며, 특히 가중치 업데이트를 위해 고차 미분(gradient of the gradient)을 계산해야 하는 계산적 부담과 수치적 불안정성 문제가 존재한다. 따라서 본 논문의 목표는 복잡한 메타 러닝 최적화 루틴을 제거하고, Bayesian 관점에서 inner loop를 분석적으로 처리하여 단순하면서도 강력한 Few-Shot 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Deep Kernels를 사용하여 메타 러닝의 inner loop를 Bayesian 적분으로 대체하는 것이다. 이를 **Deep Kernel Transfer (DKT)**라고 정의하며, 주요 기여 사항은 다음과 같다.

1.  **분석적 Marginal Likelihood 활용**: 각 태스크별 파라미터를 개별적으로 최적화하는 대신, Gaussian Process(GP) 접근 방식을 통해 태스크별 파라미터를 주변화(marginalize)함으로써 closed-form의 marginal likelihood를 유도하였다. 이를 통해 inner loop의 최적화 과정 없이 단일 optimizer만으로 학습이 가능하다.
2.  **단순성과 효율성**: 복잡한 메타 최적화 루틴을 제거하여 구현이 간단하며, 데이터가 적은 regime에서 효율적으로 작동한다.
3.  **불확실성 정량화**: Bayesian 모델의 특성을 그대로 유지하여, 새로운 인스턴스에 대한 예측 시 불확실성을 제공함으로써 의사결정의 신뢰도를 높였다.
4.  **범용적 적용 가능성**: 회귀(Regression)와 분류(Classification) 작업 모두에 적용 가능하며, 특히 도메인이 다른 데이터셋으로 전이되는 Cross-domain adaptation에서 탁월한 성능을 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들과의 차별점을 제시한다.

-   **Feature Transfer**: 사전 학습 후 미세 조정(fine-tuning)하는 방식은 새로운 태스크마다 일부 모델을 처음부터 학습시켜야 하며, 과적합(overfitting) 위험이 크다. Baseline++와 같은 개선안이 있으나, 여전히 고정된 fine-tuning 프로토콜에 의존한다.
-   **Metric Learning**: Matching Networks, Prototypical Networks, Relation Networks 등은 학습된 메트릭 공간에서 거리 기반으로 분류를 수행한다. DKT는 단순한 거리 비교를 넘어 Bayesian 추론을 통해 예측 분포를 생성한다.
-   **Gradient-based Meta-Learning**: MAML은 초기 파라미터를 빠르게 적응시키는 방향으로 학습하지만, 앞서 언급한 수치적 불안정성과 고차 미분 계산 문제가 있다. DKT는 이를 분석적 적분으로 대체하여 해결한다.
-   **Bayesian Meta-Learning**: VERSA나 LLAMA 같은 계층적 Bayesian 모델들이 존재하지만, 이들은 종종 복잡한 amortization network나 샘플링 기반의 추론을 요구한다. 반면 DKT는 Deep Kernel을 통해 더 단순한 구조로 유사한 효과를 낸다.
-   **Adaptive Deep Kernel Learning (ADKL)**: ADKL은 태스크 인코더 네트워크를 통해 태스크별 커널을 찾지만, DKT는 공유된 일반 목적 하이퍼파라미터 세트를 사용하여 추가적인 인코더 모듈 없이도 전이가 가능하다.

## 🛠️ Methodology

### 전체 시스템 구조
DKT는 신경망(Neural Network)과 커널(Kernel)을 결합하여 확장 가능하고 표현력이 풍부한 공분산 함수를 생성한다. 전체 파이프라인은 다음과 같다.
1.  입력 $x$가 파라미터 $\phi$를 가진 비선형 함수 $F_\phi$ (예: CNN)를 통해 저차원의 잠재 벡터(latent vector) $h$로 매핑된다.
2.  매핑된 $h$는 하이퍼파라미터 $\theta$를 가진 커널 $k'$에 입력된다.
3.  최종적인 딥 커널은 다음과 같이 정의된다:
    $$k(x, x' | \theta, \phi) = k'(F_\phi(x), F_\phi(x') | \theta)$$

### 훈련 목표 및 학습 절차
DKT는 모든 태스크에 걸쳐 Marginal Likelihood를 최대화하는 방향으로 $\theta$와 $\phi$를 동시에 학습한다.
-   **Marginal Likelihood**: 태스크 $t$에 대한 데이터 $(T_x^t, T_y^t)$가 주어졌을 때, 태스크별 파라미터 $\rho_t$를 적분하여 제거한 확률 분포는 다음과 같다.
    $$P(T_y^t | T_x^t, \theta, \phi) = \int \prod_k P(y_k | x_k, \theta, \phi, \rho_t) d\rho_t$$
-   **전체 목표 함수**: 모든 태스크의 로그 주변 가능도(log marginal likelihood) 합을 최대화한다.
    $$\log P(D_y | D_x, \hat{\theta}, \hat{\phi}) = \sum_t \log P(T_y^t | T_x^t, \hat{\theta}, \hat{\phi})$$
-   **학습 과정**: Stochastic Gradient Descent를 사용하여 매 반복마다 하나의 태스크를 샘플링하고, 해당 태스크의 marginal likelihood를 계산하여 $\theta$와 $\phi$를 업데이트한다.

### 태스크별 상세 방법론

#### 1. 회귀 (Regression)
회귀 문제에서는 출력 $y$가 Gaussian noise $\epsilon$을 가진 신호 $f^*(x)$라고 가정한다.
-   **예측 분포**: 새로운 포인트 $x^*$에 대한 예측 평균과 공분산은 다음과 같다.
    $$E[f^*] = k^* > (K + \sigma^2 I)^{-1} y$$
    $$\text{cov}(f^*) = k^{**} - k^* > (K + \sigma^2 I)^{-1} k^*$$
    여기서 $K$는 훈련 데이터 간의 공분산 행렬, $k^*$는 테스트 포인트와 훈련 데이터 간의 공분산 벡터이다.
-   **손실 함수**: 다음과 같은 log marginal likelihood를 최대화한다.
    $$\log P(D_y | D_x, \hat{\theta}, \hat{\phi}) = \sum_t \left( -\frac{1}{2} y^T [K_t(\hat{\theta}, \hat{\phi})]^{-1} y - \frac{1}{2} \log |K_t(\hat{\theta}, \hat{\phi})| + c \right)$$
    앞의 항은 데이터 적합도(data-fit)를, 뒤의 항은 복잡도에 대한 페널티(penalty)를 의미한다.

#### 2. 분류 (Classification)
분류 문제는 non-Gaussian likelihood로 인해 분석적 적분이 불가능하다. 이를 해결하기 위해 **Label Regression (LR)** 방식을 도입한다.
-   **방식**: 이진 분류 문제에서 클래스 0과 1을 각각 $-1$과 $1$이라는 실수 값으로 매핑하여 회귀 문제로 취급한다.
-   **다중 클래스 확장**: One-versus-rest 전략을 사용하여 $C$개의 이진 분류기를 구축한다.
-   **최종 결정**: 각 분류기의 예측 평균 $m_c(x^*)$를 sigmoid 함수 $\sigma(\cdot)$에 통과시켜 확률을 계산하고, 가장 높은 확률을 가진 클래스를 선택한다.
    $$c^* = \arg\max_c (\sigma(m_c(x^*)))$$

## 📊 Results

### 실험 설정
-   **회귀**: 주기 함수(Periodic functions) 예측, 머리 자세 궤적(Head pose trajectory) 추정. MSE(Mean-Squared Error)를 지표로 사용.
-   **분류**: CUB-200, mini-ImageNet 데이터셋. 1-shot, 5-way 설정에서 Accuracy 측정.
-   **Cross-domain**: Omniglot $\to$ EMNIST, mini-ImageNet $\to$ CUB 설정으로 일반화 성능 측정.
-   **지표**: Accuracy, MSE, ECE(Expected Calibration Error).

### 주요 결과
1.  **회귀 성능**: DKT는 Feature Transfer 및 MAML보다 낮은 MSE를 기록하였다. 특히 주기 함수 예측에서 Spectral 커널을 사용했을 때 매우 높은 정밀도를 보였으며, 학습 데이터 범위를 벗어난 out-of-range 영역에서도 강건한 예측 능력을 보였다.
2.  **분류 성능**: mini-ImageNet (1-shot)에서 DKT + BNCosSim 조합이 $49.73\%$의 정확도를 기록하며, LLAMA, VERSA 등 최신 Bayesian 메타 러닝 방법론보다 우수한 성능을 보였다.
3.  **Cross-domain 성능**: mini-ImageNet $\to$ CUB 전이 학습에서 DKT(CosSim)가 $40.22\%$의 정확도를 기록하며 타 방법론들을 압도하였다. 이는 DKT가 도메인 변화에 더 유연하게 대응함을 시사한다.
4.  **불확실성 정량화**: 데이터가 오염된(Cutout noise) 입력에 대해 DKT는 예측값은 유지하면서 불확실성(variance)을 높게 측정하여, 모델이 스스로의 예측을 신뢰하지 않음을 정확히 표현하였다.

## 🧠 Insights & Discussion

본 논문은 복잡한 최적화 루틴을 가진 메타 러닝 모델들이 실제로는 단순한 계층적 Bayesian 모델로 대체될 수 있음을 입증하였다.

**강점 및 통찰:**
-   **커널 선택의 중요성**: 데이터의 특성에 맞는 커널(주기 함수 $\to$ Spectral, 이미지 $\to$ CosSim/BN)을 선택하는 것이 복잡한 적응형(adaptive) 알고리즘보다 더 효과적일 수 있음을 보여주었다.
-   **계산 효율성**: inner loop를 분석적으로 제거함으로써 학습 및 추론 속도를 획기적으로 개선하였으며, 동시에 Bayesian의 장점인 불확실성 추정 능력을 확보하였다.

**한계 및 비판적 해석:**
-   **커널 의존성**: 성능이 커널 선택에 크게 의존하므로, 최적의 커널을 찾기 위한 사전 지식이나 탐색 과정이 필요하다.
-   **데이터 편향**: 저데이터 환경(low-data regime) 특성상 훈련 데이터에 편향이 있을 경우, Bayesian 모델 역시 잘못된 추정을 할 위험이 있으며 이는 실제 적용 시 주의가 필요하다.
-   **구현상의 가정**: 분류 문제를 회귀 문제로 치환하여 푼 Label Regression 방식이 항상 최적인지에 대한 추가적인 논의가 필요하지만, 실험적으로는 효율성과 성능의 균형이 좋음을 확인하였다.

## 📌 TL;DR

본 논문은 Few-Shot Learning을 위해 메타 러닝의 복잡한 inner-loop 최적화를 Deep Kernel 기반의 Bayesian Marginal Likelihood 계산으로 대체한 **Deep Kernel Transfer (DKT)**를 제안한다. DKT는 구현이 간단하고 학습이 안정적이며, 특히 회귀, 분류, 그리고 Cross-domain 전이 학습에서 기존 SOTA 모델들을 능가하는 성능을 보인다. 또한, Bayesian 프레임워크를 통해 예측의 불확실성을 정량적으로 제공함으로써 신뢰할 수 있는 Few-Shot 추론을 가능하게 한다. 이 연구는 복잡한 메타 러닝 구조 없이도 적절한 커널과 Bayesian 처리를 통해 높은 효율성과 정확도를 달성할 수 있음을 시사한다.