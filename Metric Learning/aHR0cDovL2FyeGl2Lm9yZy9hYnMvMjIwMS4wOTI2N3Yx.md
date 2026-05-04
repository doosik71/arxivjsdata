# Spectral, Probabilistic, and Deep Metric Learning: Tutorial and Survey

Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley (2022)

## 🧩 Problem to Solve

본 논문은 데이터 분석 및 머신러닝의 핵심 과제 중 하나인 거리 측정법 학습(Metric Learning)에 대한 포괄적인 튜토리얼 및 서베이(Survey)를 제공한다. 거리 측정법 학습의 근본적인 목적은 유사한 데이터 포인트 사이의 거리는 좁히고, 서로 다른 데이터 포인트 사이의 거리는 멀게 만드는 최적의 거리 척도(Distance Metric) 또는 임베딩 공간(Embedding Space)을 학습하는 것이다.

거리 측정법 학습은 단순한 거리 계산을 넘어, 고차원 데이터에서 유의미한 특징을 추출하고 클래스 간의 분별력을 높이는 차원 축소(Dimensionality Reduction) 및 매니폴드 학습(Manifold Learning)의 일환으로서 매우 중요하다. 특히 지도 학습 기반의 거리 측정법 학습은 적절한 메트릭을 통해 클래스 간의 변별력을 극대화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 방대한 거리 측정법 학습 알고리즘을 크게 세 가지 관점인 **Spectral(스펙트럼)**, **Probabilistic(확률적)**, 그리고 **Deep(딥러닝)** 방법론으로 체계적으로 분류하고 각각의 수학적 기초와 메커니즘을 상세히 분석한 점이다.

중심적인 직관은 다음과 같다.
1. **Spectral 방법론**: 기하학적 접근 방식을 취하며, 주로 일반화된 고유값 문제(Generalized Eigenvalue Problem)로 귀결시켜 최적의 가중치 행렬을 찾는다.
2. **Probabilistic 방법론**: 확률 분포를 기반으로 하며, 이웃을 선택할 확률을 최대화하거나 확률 분포 간의 거리(예: KL-divergence)를 최소화하는 방향으로 학습한다.
3. **Deep 방법론**: 신경망을 통해 비선형 임베딩 공간을 학습하며, 다양한 손실 함수(Loss Function)와 샘플링 전략을 통해 데이터의 표현력을 극대화한다.

## 📎 Related Works

논문은 기존의 거리 측정법 학습 관련 서베이(Yang & Jin, 2006; Kulis, 2013; Bellet et al., 2013 등)와 딥러닝 기반의 특정 서베이(Kaya & Bilge, 2019)를 언급한다. 기존 연구들이 특정 분야에 치우치거나 최신 딥러닝 트렌드를 완전히 포괄하지 못한 반면, 본 논문은 고전적인 Mahalanobis 거리부터 최신 Riemannian Manifold 기반의 딥러닝 방법론까지 통합적으로 다룬다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 기초: Generalized Mahalanobis Distance
모든 방법론의 근간이 되는 거리 척도로 **Generalized Mahalanobis Distance**를 정의한다. 두 점 $x_i, x_j$ 사이의 거리는 다음과 같이 정의된다.
$$\|x_i - x_j\|_W = \sqrt{(x_i - x_j)^T W (x_i - x_j)}$$
여기서 $W \succeq 0$ (Positive Semi-definite)인 가중치 행렬이다. $W$가 단위 행렬($I$)이면 유클리드 거리와 동일하며, $W$를 어떻게 학습하느냐가 거리 측정법 학습의 핵심이다.

### 2. Spectral Metric Learning
스펙트럼 방법론은 데이터를 투영하는 행렬 $U$를 학습하여 $W = UU^T$ 형태로 구성한다.
- **Scatter-based methods**: 유사한 쌍의 산포 행렬 $\Sigma_S$와 서로 다른 쌍의 산포 행렬 $\Sigma_D$를 정의하고, $\text{tr}(W\Sigma_S)$는 최소화하고 $\text{tr}(W\Sigma_D)$는 최대화하는 고유값 문제를 해결한다.
- **Hinge Loss-based**: Large-margin metric learning과 같이 마진(Margin) 개념을 도입하여 $\text{dist}(\text{similar}) + \text{margin} \le \text{dist}(\text{dissimilar})$ 조건을 만족하도록 SDP(Semidefinite Programming) 문제를 푼다.
- **Geometric methods**: SPD(Symmetric Positive Definite) 매니폴드 위에서 Riemannian distance를 사용하여 $W$를 학습하는 GMML(Geometric Mean Metric Learning) 등이 포함된다.

### 3. Probabilistic Metric Learning
확률적 방법론은 한 점이 다른 점을 이웃으로 선택할 확률 분포를 정의한다.
- **Collapsing Classes**: 유사한 클래스의 점들이 임베딩 공간에서 하나의 점으로 수렴(collapse)하도록 KL-divergence를 최소화한다.
- **NCA (Neighborhood Component Analysis)**: 유사한 점들이 이웃으로 선택될 확률 $\sum p_{ij}$를 최대화하는 투영 행렬 $U$를 학습한다.
- **Bayesian methods**: 가중치 행렬이나 고유값에 대한 사전 분포(Prior)를 설정하고, Variational Inference를 통해 사후 분포를 추정한다.

### 4. Deep Metric Learning
딥러닝 방법론은 가중치 행렬 $W$를 직접 학습하는 대신, 신경망 $f(x; \theta)$를 통해 저차원 임베딩 공간으로 매핑한다.
- **Siamese Networks**: 동일한 가중치를 공유하는 두 개 이상의 네트워크를 사용하여 쌍(Pair)이나 트리플렛(Triplet)의 거리를 비교한다.
- **핵심 손실 함수**:
    - **Contrastive Loss**: 유사한 쌍은 거리를 $0$으로, 서로 다른 쌍은 특정 마진 $m$ 이상으로 밀어낸다.
    - **Triplet Loss**: 앵커($a$), 긍정($p$), 부정($n$) 샘플을 사용하여 $\|f(x_a) - f(x_p)\|^2 + m \le \|f(x_a) - f(x_n)\|^2$를 만족하도록 학습한다.
- **Triplet Mining**: 모든 조합을 학습하는 것은 비효율적이므로, 가장 학습하기 어려운 샘플을 찾는 Batch-hard, Batch-semi-hard 등의 전략을 사용한다.

## 📊 Results

본 논문은 새로운 알고리즘을 제안하고 실험하는 연구 논문이 아니라, 기존의 수많은 알고리즘을 분석하고 체계화한 **튜토리얼 및 서베이 논문**이다. 따라서 특정 데이터셋에 대한 성능 수치보다는 다음과 같은 방법론적 결과와 이론적 관계를 제시한다.

- **이론적 통합**: Spectral 방법론의 고유값 문제와 Deep 방법론의 Triplet Loss가 본질적으로 유사한 목적(내부 분산 최소화, 외부 분산 최대화)을 가지고 있음을 수학적으로 보여준다.
- **분류 체계**: Mahalanobis 거리 $\rightarrow$ Kernel-based $\rightarrow$ Manifold-based $\rightarrow$ Deep Embedding으로 이어지는 거리 측정법의 진화 과정을 정립하였다.
- **구현 가이드**: NCA, Siamese Network, Autoencoder 등의 구조와 학습 절차를 명확히 서술하여 실제 구현이 가능하도록 상세히 설명하였다.

## 🧠 Insights & Discussion

### 강점
본 보고서는 거리 측정법 학습이라는 광범위한 주제를 매우 치밀한 수학적 구조로 연결했다. 특히 단순한 나열이 아니라, Generalized Mahalanobis Distance라는 공통 분모를 통해 스펙트럼, 확률, 딥러닝 방법론이 어떻게 상호 연결되는지 논리적으로 설명한 점이 탁월하다.

### 한계 및 논의사항
- **계산 복잡도**: Spectral 방법론의 고유값 분해나 SDP 문제는 데이터 크기가 커질수록 계산 비용이 기하급수적으로 증가한다. 딥러닝 방법론이 이를 어떻게 해결하고 있는지에 대한 더 깊은 분석이 필요하다.
- **마진(Margin) 설정**: 많은 알고리즘이 하이퍼파라미터인 마진 $m$에 크게 의존한다. 데이터셋마다 최적의 마진을 자동으로 결정하는 메커니즘에 대한 논의가 부족하다.
- **가정의 한계**: 많은 확률적 모델이 Gaussian 분포를 가정하고 있으나, 실제 복잡한 데이터의 분포를 충분히 반영하고 있는지에 대한 비판적 검토가 필요하다.

## 📌 TL;DR

본 논문은 거리 측정법 학습(Metric Learning)의 전 분야를 아우르는 종합 안내서이다. **Spectral(고유값 문제)**, **Probabilistic(확률 분포)**, **Deep(신경망 임베딩)**이라는 세 가지 큰 축을 중심으로, Mahalanobis 거리라는 기초 이론부터 최신 Riemannian Manifold 및 Few-shot 학습까지 방대한 알고리즘을 체계적으로 정리하였다. 이 연구는 향후 새로운 거리 학습 모델을 설계하거나 기존 모델의 수학적 배경을 이해하려는 연구자들에게 필수적인 이론적 지도 역할을 할 것으로 기대된다.