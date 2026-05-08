# A Framework and Benchmark for Deep Batch Active Learning for Regression

David Holzmüller, Viktor Zaverkin, Johannes Kästner, and Ingo Steinwart (2023)

## 🧩 Problem to Solve

본 논문은 지도 학습(Supervised Learning)의 회귀(Regression) 문제에서 데이터 라벨링 비용을 줄이기 위한 Batch Mode Deep Active Learning (BMDAL)의 프레임워크 구축과 벤치마크 설정을 목표로 한다.

일반적으로 딥러닝 기반의 능동 학습(Active Learning)은 새로운 샘플을 라벨링할 때마다 모델을 재학습시켜야 하므로 계산 비용이 매우 높다. 이를 해결하기 위해 한 번에 여러 개의 샘플을 선택하는 Batch mode 방식이 사용되지만, 기존의 많은 BMDAL 연구는 분류(Classification) 문제에 집중되어 있으며 회귀 문제에 대한 표준 벤치마크나 효율적인 방법론은 상대적으로 부족한 실정이다.

특히, 저자들은 다음과 같은 실무적 요구사항을 해결하고자 한다:

1. **확장성(Scalability):** 대규모 풀(Pool) 데이터셋과 큰 배치 사이즈에서도 효율적으로 동작해야 한다.
2. **범용성(Generality):** 신경망 구조나 학습 코드를 수정하지 않고도 적용 가능해야 한다.
3. **효율성(Efficiency):** 배치 선택을 위해 여러 개의 신경망을 학습시키는 등의 과도한 계산 비용이 발생해서는 안 된다.

## ✨ Key Contributions

본 논문의 핵심 기여는 BMDAL 알고리즘을 구성 요소별로 분해하여 조합할 수 있는 모듈형 프레임워크를 제안하고, 이를 통해 최적의 조합을 찾아낸 것이다.

1. **BMDAL 구성 프레임워크 제안:** BMDAL 과정을 $\text{Base Kernel} \rightarrow \text{Kernel Transformation} \rightarrow \text{Selection Method}$의 세 단계로 구조화하였다. 이를 통해 베이지안 방법론과 기하학적 방법론을 유연하게 통합하여 구현할 수 있다.
2. **LCMD (Largest Cluster Maximum Distance) 방법론 제안:** 데이터의 대표성(Representativity)과 다양성(Diversity)을 동시에 확보하기 위해, 가장 크기가 큰 클러스터 내에서 가장 거리가 먼 샘플을 선택하는 새로운 결정론적 선택 방법을 제안하였다.
3. **효율적인 NTK 구현:** 고차원 특징 공간을 갖는 Neural Tangent Kernel (NTK)의 계산 비용을 줄이기 위해 Sketching 기법을 도입하여 정확도는 유지하면서 계산 속도를 획기적으로 개선하였다.
4. **회귀용 BMDAL 벤치마크 구축:** 15개의 대규모 tabular 데이터셋으로 구성된 오픈소스 벤치마크를 제공하여, 다양한 커널과 선택 방법의 성능을 정량적으로 비교 분석하였다.

## 📎 Related Works

기존의 BMDAL 연구들은 주로 다음과 같은 접근 방식을 취해왔다:

- **Bayesian Methods:** Laplace approximation이나 Ensemble 등을 통해 모델의 불확실성(Uncertainty)을 추정하여 정보량이 많은 샘플을 선택한다. 하지만 회귀 문제에서는 분류 문제와 달리 소프트맥스 층의 확률 벡터를 사용할 수 없어 적용 방식이 다르다.
- **Geometric Methods:** Core-set approach나 BADGE와 같이 특징 공간(Feature Space)에서 데이터의 분포를 대표하는 샘플들을 기하학적으로 선택하여 다양성을 확보한다.
- **NTK 기반 방법론:** 신경망의 무한 너비 극한에서 정의되는 Neural Tangent Kernel을 사용하여 모델의 거동을 근사하지만, 실제 유한한 너비의 네트워크에 적용하고 대규모 데이터로 확장하는 데에는 어려움이 있었다.

본 논문은 이러한 기존 방식들을 하나의 프레임워크로 통합하고, 특히 회귀 문제에 특화된 벤치마크가 없었다는 점을 지적하며 이를 보완한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (Framework)

BMDAL 알고리즘은 다음과 같은 파이프라인을 따른다:
$$\text{Base Kernel} \xrightarrow{\text{Transformation}} \text{Transformed Kernel} \xrightarrow{\text{Selection}} \text{Batch } X_{batch}$$

### 2. 주요 구성 요소

#### (1) Base Kernels

신경망 $f_\theta$의 거동을 대리(proxy)하는 유사도 측정 도구이다.

- **Linear Kernel:** 가장 단순한 형태의 $\langle x, \tilde{x} \rangle$.
- **Full Gradient Kernel (NTK):** 학습된 파라미터 $\theta^T$에서의 그래디언트 $\nabla_\theta f_{\theta^T}(x)$를 특징 맵으로 사용한다. 이는 네트워크의 선형화를 통해 모델의 변화량을 측정한다.
- **Last-layer Kernel:** 마지막 층의 파라미터에 대한 그래디언트만을 고려하며, 사실상 마지막 은닉층의 출력값 $\tilde{x}^{(L-1)}$을 특징 맵으로 사용하는 것과 같다.
- **NNGP Kernel:** 무한 너비 신경망의 초기 분포가 가우시안 프로세스로 수렴한다는 점을 이용한 커널이다.

#### (2) Kernel Transformations

베이스 커널을 특정 목적에 맞게 변형한다.

- **Scaling:** 커널의 분산을 정규화하여 이후의 GP 계산에서 수치적 안정성을 확보한다.
- **GP Posterior Transformation:** 학습 데이터 $D_{train}$을 관찰한 후의 사후 공분산(Posterior Covariance)을 계산한다.
  $$k \rightarrow post(X_{train}, \sigma^2)(x, \tilde{x}) = k(x, \tilde{x}) - k(x, X_{train})(k(X_{train}, X_{train}) + \sigma^2 I)^{-1} k(X_{train}, \tilde{x})$$
- **Sketching:** 고차원 특징 맵 $\phi(x)$에 랜덤 행렬 $U$를 곱해 저차원으로 투영하는 Gaussian sketch를 통해 계산 복잡도를 $\Theta(d_{feat})$에서 $\Theta(p)$로 줄인다.
- **Ensembling:** 여러 모델의 커널을 합산하여 불확실성 추정치를 개선한다.

#### (3) Selection Methods

변형된 커널을 사용하여 실제로 어떤 샘플을 뽑을지 결정한다.

- **MaxDiag:** 단순하게 커널의 대각 성분(불확실성)이 가장 큰 샘플을 선택한다.
- **MaxDet:** 행렬식(Determinant)을 최대화하여 정보량과 다양성을 동시에 고려한다.
- **Bait:** 풀 세트 전체의 총 불확실성을 최소화하는 방향으로 샘플을 선택한다.
- **LCMD (Proposed):**
  1. 모든 풀 샘플 $x \in X_{rem}$을 가장 가까운 선택된 중심점 $c(x) \in X_{sel}$에 할당하여 클러스터를 형성한다.
  2. 각 클러스터의 크기를 거리의 제곱 합으로 정의한다: $s(\tilde{x}) = \sum_{x: c(x)=\tilde{x}} d_k(x, \tilde{x})^2$.
  3. 가장 크기가 큰(가장 불확실성이 높은 영역인) 클러스터를 찾고, 그 내부에서 중심점과 가장 멀리 떨어진 샘플을 선택한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** 15개의 대규모 Tabular 회귀 데이터셋 (UCI, OpenML 등에서 수집).
- **모델:** 2개의 은닉층(각 512 뉴런)을 가진 Fully-connected NN.
- **평가 지표:** RMSE (Root Mean Squared Error), MAE, MAXE (최대 오차) 등.
- **절차:** 초기 학습 데이터 256개로 시작하여, 배치 사이즈 256으로 총 16번의 BMDAL 스텝을 수행.

### 2. 주요 결과

- **최적의 조합:** **LCMD-TP 선택 방법 + Sketched NTK ($k_{grad}$)** 조합이 RMSE와 MAE 측면에서 기존의 모든 SOTA 방법론(Bait, BADGE, Core-set 등)을 압도하였다.
- **커널의 영향:** NTK ($k_{grad}$)가 Last-layer 커널 ($k_{ll}$)보다 일관되게 더 나은 성능을 보였으며, 이는 NTK가 네트워크의 전체적인 거동을 더 잘 반영하기 때문으로 해석된다.
- **Sketching의 효율성:** NTK에 Sketching을 적용했을 때, 계산 시간은 획기적으로 단축되었음에도 불구하고 예측 정확도는 거의 손실되지 않았다.
- **샘플 효율성:** 제안된 방법은 무작위 선택(Random) 대비 약 절반의 샘플만으로도 유사한 RMSE 성능에 도달하였다.

## 🧠 Insights & Discussion

### 1. LCMD의 강점

LCMD는 능동 학습의 세 가지 핵심 기준인 **정보성(Informativeness), 다양성(Diversity), 대표성(Representativity)**을 모두 충족한다. 클러스터의 크기를 통해 데이터 밀도가 높은 지역(대표성)을 찾고, 그 안에서 가장 먼 점을 뽑음으로써 모델이 모르는 영역(정보성)과 중복되지 않는 점(다양성)을 동시에 확보한다.

### 2. 적용 시점의 판단 기준

저자들은 초기 학습 데이터셋에서 $\frac{RMSE}{MAE}$ 비율이 높을수록(즉, 오차의 분포가 불균일하고 큰 오차가 존재할수록) BMDAL을 적용했을 때의 이득이 더 크다는 상관관계를 발견하였다. 이는 실무자가 BMDAL을 적용할지 말지 결정하는 사전 지표로 활용될 수 있다.

### 3. 한계점

- 본 연구는 Tabular 데이터에 집중되어 있어, 이미지나 시계열 데이터와 같은 특수 구조 데이터에서의 성능은 검증되지 않았다.
- 풀 데이터와 테스트 데이터 간의 분포 변화(Distribution Shift)가 있는 상황에 대한 분석이 부족하다.

## 📌 TL;DR

본 논문은 딥러닝 회귀 문제에서 효율적인 배치 샘플 선택을 위한 **모듈형 프레임워크(Base Kernel $\rightarrow$ Transformation $\rightarrow$ Selection)**를 제안하였다. 특히, 새로운 결정론적 클러스터링 기반 선택 방법인 **LCMD**와 **Sketched NTK**를 결합하여 계산 효율성과 샘플 효율성을 동시에 달성하였으며, 15개의 대규모 데이터셋 벤치마크를 통해 그 우수성을 입증하였다. 이 연구는 대규모 회귀 문제에서 라벨링 비용을 최소화하면서 모델 성능을 빠르게 올릴 수 있는 실용적인 가이드를 제공한다.
