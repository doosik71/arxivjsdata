# Spectral, Probabilistic, and Deep Metric Learning: Tutorial and Survey

Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley (2022)

## 🧩 Problem to Solve

본 논문은 데이터의 유사성과 비유사성을 효과적으로 측정하기 위한 거리 함수(distance metric)를 학습하는 **Metric Learning** 분야의 전반적인 이론과 알고리즘을 체계적으로 정리하고 분석하는 것을 목표로 한다.

전통적인 Euclidean distance는 모든 차원을 동일하게 가중치 처리하므로 데이터의 분산이나 클래스 간의 특성을 반영하지 못하는 한계가 있다. Metric Learning의 핵심 문제는 데이터의 특성에 맞게 공간을 변형하여 유사한 샘플 간의 거리는 좁히고, 서로 다른 샘플 간의 거리는 넓히는 최적의 거리 척도를 찾는 것이다. 이는 k-Nearest Neighbors (kNN) 분류, 클러스터링, 이미지 검색 및 얼굴 인식과 같은 수많은 컴퓨터 비전 및 기계 학습 작업의 성능을 결정짓는 매우 중요한 문제이다.

## ✨ Key Contributions

본 논문은 Metric Learning의 방대한 알고리즘들을 수학적 접근 방식에 따라 크게 세 가지 범주인 **Spectral, Probabilistic, Deep Metric Learning**으로 분류하고, 이를 튜토리얼 형식으로 상세히 설명한다.

가장 중심적인 설계 아이디어는 거리 측정의 기본이 되는 **Generalized Mahalanobis Distance**의 개념을 정의하고, 이를 어떻게 최적화하느냐에 따라 세 가지 경로로 분기된다는 점을 보여주는 것이다.

- **Spectral methods**: 기하학적 접근 방식을 통해 최적화 문제를 일반화된 고유값 문제(generalized eigenvalue problem)로 귀결시킨다.
- **Probabilistic methods**: 확률 분포와 최대 우도 추정(Maximum Likelihood Estimation) 또는 Bayesian 추론을 통해 거리 행렬을 학습한다.
- **Deep methods**: 신경망을 통해 데이터를 저차원의 임베딩 공간(embedding space)으로 투영하고, 이 공간에서의 Euclidean distance를 최적화하는 비선형 매핑 함수를 학습한다.

## 📎 Related Works

논문은 기존의 Metric Learning 관련 서베이 논문들(Yang & Jin, 2006; Kulis, 2013; Bellet et al., 2013 등)과 Deep Metric Learning에 특화된 서베이(Kaya & Bilge, 2019)를 언급한다.

기존 연구들과의 차별점은 단순히 알고리즘을 나열하는 것이 아니라, 거리 척도의 수학적 정의부터 시작하여 Spectral, Probabilistic, Deep이라는 세 가지 큰 줄기의 연결 고리를 제공함으로써 독자가 전체적인 프레임워크를 이해할 수 있도록 돕는 튜토리얼 성격을 강하게 띤다는 점이다.

## 🛠️ Methodology

### 1. 거리 척도의 기초: Generalized Mahalanobis Distance

모든 방법론의 기초가 되는 Generalized Mahalanobis Distance는 가중치 행렬 $W$를 사용하여 다음과 같이 정의된다.
$$\|x_i - x_j\|_W^2 = (x_i - x_j)^T W (x_i - x_j)$$
여기서 $W$는 Positive Semi-definite ($W \succeq 0$) 행렬이어야 하며, 이 행렬이 공간의 변형을 결정한다. $W=I$일 경우 이는 Euclidean distance와 동일해진다.

### 2. Spectral Metric Learning

Spectral 방법론은 주로 데이터의 Scatter(분산)를 이용하거나 Hinge loss를 최적화하는 기하학적 접근을 취한다.

- **Scatter-based**: 유사 쌍(similar pairs)의 거리 합은 최소화하고, 비유사 쌍(dissimilar pairs)의 거리 합은 최대화하는 방향으로 $W$를 학습한다. 이는 주로 다음과 같은 형태의 일반화된 고유값 문제로 풀린다:
  $$\Sigma_D u = \lambda \Sigma_S u$$
- **Kernel Spectral**: 데이터를 고차원 RKHS(Reproducing Kernel Hilbert Space)로 매핑하여 비선형성을 확보한 뒤 Spectral 최적화를 수행한다.
- **Geometric**: SPD(Symmetric Positive Definite) 매니폴드 상에서 Riemannian geometry를 이용하여 거리 척도를 학습하며, Riccati 방정식 등을 통해 최적해를 구한다.

### 3. Probabilistic Metric Learning

확률적 방법론은 특정 점이 다른 점의 이웃이 될 확률을 정의하고, 이를 실제 유사도 분포와 일치시키려 한다.

- **Collapsing Classes**: 유사한 클래스들은 같은 점으로 붕괴(collapse)시키고 다른 클래스는 멀어지게 하며, 이를 위해 KL-divergence를 최소화한다.
- **Neighborhood Component Analysis (NCA)**: 유사한 샘플이 이웃으로 선택될 확률을 최대화하는 투영 행렬 $U$를 학습한다.
- **Bayesian Approach**: 거리 행렬의 파라미터에 확률 분포를 가정하고 Variational Inference를 통해 사후 분포(posterior)를 추정한다.

### 4. Deep Metric Learning

딥러닝 기반 방법론은 $W$라는 행렬을 직접 학습하는 대신, 신경망 $f(\cdot)$를 통해 데이터를 임베딩 공간으로 보내는 함수를 학습한다.

- **Siamese & Triplet Networks**: 동일한 가중치를 공유하는 신경망을 사용하여 쌍(pair)이나 트리플렛(triplet) 데이터를 처리한다.
- **Triplet Loss**: 앵커($x_a$), 긍정($x_p$), 부정($x_n$) 샘플을 이용하여 다음의 조건을 만족하도록 학습한다.
  $$\|f(x_a) - f(x_p)\|_2^2 + m \le \|f(x_a) - f(x_n)\|_2^2$$
  여기서 $m$은 Margin으로, 긍정과 부정 샘플 간의 최소 거리 차이를 강제한다.
- **Triplet Mining**: 모든 조합을 학습하는 것은 비효율적이므로, 학습에 가장 도움이 되는 Hard positive/negative 샘플을 선택하는 전략(Batch-hard, Batch-semi-hard 등)이 필수적으로 사용된다.

## 📊 Results

본 논문은 새로운 알고리즘을 제안하거나 실험을 수행한 연구 논문이 아니라, 기존의 방대한 연구들을 정리한 **Tutorial and Survey** 논문이다. 따라서 특정 데이터셋에 대한 정량적 성능 지표나 실험 결과 섹션은 존재하지 않는다. 대신, 각 방법론의 수학적 유도 과정과 알고리즘의 흐름을 상세히 기술하여 이론적 배경을 제공하는 것에 집중하고 있다.

## 🧠 Insights & Discussion

본 논문을 통해 Metric Learning의 발전 흐름을 다음과 같이 해석할 수 있다.
첫째, **선형성에서 비선형성으로의 확장**이다. 초기 Spectral/Probabilistic 방법론은 Mahalanobis distance라는 선형 변환에 의존했으나, 이후 Kernel trick을 통해 RKHS로 확장되었고, 최종적으로 신경망의 다층 구조를 통해 복잡한 비선형 임베딩을 학습하는 Deep Metric Learning으로 진화하였다.

둘째, **최적화 목표의 정교화**이다. 단순히 거리를 좁히고 넓히는 것에서 시작하여, Margin의 개념을 도입한 Large-margin learning, 확률적 이웃 개념의 NCA, 그리고 최근의 Adversarial learning 및 Few-shot learning으로 목표가 구체화되고 있다.

셋째, **계산 복잡도와 성능의 트레이드오프**이다. Spectral 방법론은 수학적 해석력이 높고 전역 최적해를 찾기 쉬운 경우가 많지만, 고차원 데이터에서 계산 비용이 급증한다. 반면 Deep 방법론은 대규모 데이터에서 압도적인 성능을 보이지만, 하이퍼파라미터에 민감하고 모델의 해석력이 떨어진다는 한계가 있다.

## 📌 TL;DR

이 논문은 Metric Learning의 핵심인 거리 함수 학습 알고리즘을 **Spectral(기하학/고유값), Probabilistic(확률/분포), Deep(임베딩/신경망)** 세 가지 관점에서 집대성한 종합 가이드북이다. 기초적인 Mahalanobis distance부터 최신 Deep Triplet Mining 및 Few-shot Metric Learning까지의 수학적 연결 고리를 제공하며, 향후 거리 기반 표현 학습(representation learning) 연구를 수행하려는 연구자들에게 필수적인 이론적 지도 역할을 할 것으로 보인다.
