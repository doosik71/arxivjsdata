# Few-Shot Learning by Dimensionality Reduction in Gradient Space

Martin Gauch, Maximilian Beck, Thomas Adler, Dmytro Kotsur, Stefan Fiel, Hamid Eghbal-zadeh, Johannes Brandstetter, Johannes Kofler, Markus Holzleitner, Werner Zellinger, Daniel Klotz, Sepp Hochreiter, and Sebastian Lehner (2022)

## 🧩 Problem to Solve

본 논문은 데이터가 극도로 제한된 상황에서 모델을 빠르게 적응시켜야 하는 Few-Shot Learning(FSL) 문제를 해결하고자 한다. 특히, 저자들은 일반적인 이미지 분류 기반의 FSL 벤치마크와 달리, 데이터 생성 프로세스의 매개변수가 변화함에 따라 새로운 작업이 생성되는 **Parametric Transfer** 상황에 주목한다. 

이러한 문제는 과학 및 공학 분야의 동적 시스템(Dynamical Systems)에서 빈번하게 발생하며, 예를 들어 전기 회로의 소자 값이 변하거나 기후 변화로 인해 환경 모델을 수정해야 하는 경우가 이에 해당한다. 데이터 부족으로 인한 그래디언트 추정의 불확실성은 모델의 과적합을 유발하며, 이는 적은 양의 샘플만으로 새로운 작업에 일반화해야 하는 FSL의 핵심 난제이다. 따라서 본 연구의 목표는 학습된 정보를 활용하여 최적화 경로를 제한함으로써 샘플 효율성을 높이고 일반화 성능을 향상시키는 새로운 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **SubGD (Subspace Gradient Descent)**라는 새로운 Few-Shot Learning 방법론을 제안한 것이다. SubGD의 중심 아이디어는 "신경망의 확률적 경사 하강법(SGD) 업데이트는 파라미터 공간의 매우 낮은 차원의 부분 공간(Low-dimensional subspace)에 존재한다"는 최근의 발견을 FSL에 접목한 것이다.

SubGD는 이전의 훈련 작업(Training tasks)들에서 얻은 그래디언트 업데이트 방향들의 **자기 상관 행렬(Auto-correlation matrix)**을 분석하여, 여러 작업에 걸쳐 공유되는 지배적인 부분 공간을 식별한다. 이후 새로운 작업에 적응할 때, 업데이트 방향을 이 부분 공간으로 투영하고 고윳값(Eigenvalue)에 따라 학습률을 조정함으로써, 정보에 기반한 정규화(Informed Regularization)를 수행한다.

## 📎 Related Works

본 논문은 FSL 접근 방식을 크게 세 가지 범주로 나누어 설명하며 SubGD의 위치를 정의한다.

1.  **Meta-Learning**: MAML, Reptile과 같이 최적의 초기값(Initialization)을 찾는 방법들이나, RNN 기반의 최적화 도구들이 존재한다. SubGD는 초기값 학습 방법과 직교(Orthogonal)하며, 오히려 MAML 등이 찾은 초기값 위에 결합되어 성능을 높일 수 있는 상호 보완적인 관계이다.
2.  **Learning in a Parameter Subspace**: 일부 파라미터만 업데이트하는 방식이나 무작위 부분 공간에서 학습하는 연구들이 있었다. SubGD는 이를 확장하여, 훈련 작업의 궤적(Trajectory)을 통해 데이터에 기반한 최적의 부분 공간을 자동으로 식별한다는 점에서 차별화된다.
3.  **Learning to Precondition**: MetaSGD나 Meta-Curvature처럼 프리컨디셔닝 행렬을 통해 업데이트를 조정하는 방식이 있다. 하지만 기존 방식들은 주로 대각 행렬(Diagonal matrix) 형태의 제한된 구조를 사용하는 반면, SubGD는 아키텍처에 독립적이며 더 복잡한 부분 공간 구조를 활용한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조
SubGD의 프로세스는 크게 세 단계로 나뉜다.
1.  **훈련 단계**: 기존 훈련 작업들에 대해 모델을 개별적으로 미세 조정(Fine-tuning)하여 업데이트 궤적을 생성한다.
2.  **부분 공간 식별**: 생성된 업데이트 방향들의 자기 상관 행렬을 구성하고, 고윳값 분해(Eigendecomposition)를 통해 지배적인 부분 공간을 찾는다.
3.  **테스트 단계**: 새로운 작업이 주어지면, 식별된 부분 공간으로 그래디언트 업데이트를 제한하여 모델을 빠르게 적응시킨다.

### 2. 핵심 수식 및 업데이트 규칙
SGD의 확률적 그래디언트를 $g = \nabla_\theta L_S(\theta) \in \mathbb{R}^n$라 할 때, 그 자기 상관 행렬 $C$는 다음과 같이 정의된다.
$$C = \mathbb{E}[gg^T]$$

SubGD는 이 행렬 $C$를 프리컨디셔너로 사용하여 다음과 같은 업데이트 규칙을 따른다.
$$\theta \leftarrow \theta - \eta d, \quad \text{where } d = Cg$$
여기서 $\eta$는 학습률이다. 실제 구현에서는 $C$의 크기가 너무 커서 직접 계산이 불가능하므로, 절단된 고윳값 분해(Truncated Eigendecomposition)를 통한 저차원 근사 행렬 $\hat{C}$를 사용한다.
$$\hat{C} = V \Sigma V^T$$
- $V \in \mathbb{R}^{n \times r}$: 가장 큰 $r$개의 고윳값에 대응하는 고유벡터들을 열로 가지는 행렬이다.
- $\Sigma \in \mathbb{R}^{r \times r}$: 상위 $r$개의 고윳값이 포함된 대각 행렬이다.

결과적으로 SubGD는 업데이트 방향을 $V$가 스팬(span)하는 $r$차원 부분 공간으로 투영하고, 각 방향의 학습률을 $\Sigma$의 고윳값으로 스케일링하는 효과를 가진다.

### 3. 부분 공간 차원 결정 및 일반화 이론
부분 공간의 차원 $r$을 결정하기 위해 본 논문은 섀넌 엔트로피(Shannon entropy) 개념을 도입한 **Effective Rank**를 사용한다.
$$erank(C) = \exp \left( - \sum_{i=1}^n p_i \log p_i \right), \quad p_i = \frac{\sigma_i}{\sum_{j=1}^n \sigma_j}$$

또한, 저자들은 SubGD의 일반화 능력을 이론적으로 뒷받침하기 위해 다음과 같은 일반화 경계(Generalization bound)를 제시한다.
$$L(\theta) \le L_S(\theta) + \zeta \frac{\sqrt{rank(\hat{C})} \left( \|\theta - \theta_0\|_1 + \log(l) \right) + \log(1/\delta)}{\sqrt{m}}$$
이 식은 프리컨디셔닝 행렬의 랭크($rank(\hat{C})$)가 작을수록(즉, 부분 공간의 차원이 낮을수록) 일반화 오차가 줄어들 수 있음을 시사한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: (1) Sinusoid Regression, (2) Non-linear RLC Circuit (전기 회로), (3) Climate Change Adaptation (강수-유출 시뮬레이션).
- **비교 대상**: MAML, Reptile, MetaSGD, MT-Net, Meta-Curvature, JFR 등.
- **지표**: 평균 제곱 오차(MSE), Nash-Sutcliffe Efficiency (NSE).

### 2. 주요 결과
- **Sinusoid Regression**: Support size가 작을 때(예: 5-shot) SubGD가 다른 모든 방법론을 압도하는 성능을 보였다. 특히 Support size가 증가함에 따라 MAML 등의 성능이 올라오지만, 적은 데이터 환경에서의 샘플 효율성은 SubGD가 가장 뛰어났다.
- **Non-linear RLC Circuit**: 모든 Support size에서 SubGD가 가장 낮은 Median MSE를 기록하며 최적의 성능을 보였다. 특히 단순 지도 학습(Supervised pre-training) 후 SubGD를 적용한 것만으로도 고도화된 메타 학습 초기값 기반의 SGD보다 좋은 성능을 냈다.
- **Climate Change Adaptation**: 하이드롤로지(Hydrology) 데이터셋에서 SubGD는 모든 초기화 전략에 대해 SGD보다 높은 NSE를 달성했다.

### 3. 한계점 실험 (miniImageNet)
저자들은 SubGD가 모든 문제에 적용 가능한지 확인하기 위해 이미지 분류 벤치마크인 miniImageNet에 적용했다. 결과적으로 SubGD는 일반 SGD보다 성능이 낮게 나타났다. 분석 결과, 이미지 분류 작업들 간에는 공유되는 저차원 부분 공간이 존재하지 않아 Effective Rank가 수렴하지 않고 계속 증가하는 특성을 보였으며, 이는 SubGD의 전제가 성립하지 않음을 의미한다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
SubGD의 성공 요인은 **"정보에 기반한 정규화"**에 있다. 무작위 부분 공간을 사용하는 것이 아니라, 유사한 작업들의 학습 궤적에서 추출한 지배적인 방향만을 활용함으로써, 적은 데이터로 인해 발생할 수 있는 그래디언트의 노이즈를 효과적으로 제거하고 일반화 성능을 높였다.

### 2. 적용 가능성 및 제약
본 연구는 SubGD가 **Parametric Transfer**가 일어나는 동적 시스템 문제에서 매우 강력함을 입증했다. 즉, 서로 다른 작업들이 근본적으로는 동일한 물리 법칙이나 메커니즘을 공유하고 매개변수만 다른 경우, 파라미터 공간에서 공유되는 저차원 부분 공간이 존재하며 이를 통해 극도의 샘플 효율성을 달성할 수 있다.

### 3. 비판적 논의
논문은 이론적 경계와 실험적 결과를 통해 방법론을 정당화하고 있으나, 부분 공간의 차원 $r$을 결정하는 하이퍼파라미터 설정에 여전히 의존적이다. 비록 고윳값 가중치(Weighting)를 통해 어느 정도 완화할 수 있다고 주장하지만, 실무적으로 최적의 $r$을 찾는 기준에 대한 더 구체적인 가이드라인이 필요해 보인다. 또한, 이미지 분류와 같은 고차원 데이터에서 부분 공간을 식별하기 위한 비선형 매니폴드(Non-linear manifold) 접근법에 대한 가능성을 언급한 점은 향후 연구 방향으로서 매우 유의미하다.

## 📌 TL;DR

**SubGD**는 SGD 업데이트가 저차원 부분 공간에 존재한다는 통찰을 이용해, 이전 작업들의 그래디언트 자기 상관 행렬로부터 최적의 업데이트 부분 공간을 찾아내고 이를 새로운 작업의 적응에 활용하는 Few-Shot Learning 방법론이다. 이 연구는 특히 물리적/동적 시스템과 같이 매개변수 전이가 발생하는 문제에서 기존 메타 학습 방법론보다 뛰어난 샘플 효율성과 일반화 성능을 보였으며, 이는 딥러닝 모델을 실제 공학 및 과학 현장에 빠르게 적응시키는 데 중요한 역할을 할 가능성이 크다.