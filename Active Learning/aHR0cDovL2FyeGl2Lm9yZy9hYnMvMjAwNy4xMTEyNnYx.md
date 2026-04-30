# Efficient Graph-Based Active Learning with Probit Likelihood via Gaussian Approximations

Kevin Millermiller, Hao Li, Andrea L. Bertozzi (2020)

## 🧩 Problem to Solve

본 논문은 그래프 기반 준지도 학습(Graph-based Semi-Supervised Learning, SSL) 환경에서 비가우시안 베이지안 모델(non-Gaussian Bayesian models)을 사용할 때 발생하는 계산 효율성 문제를 해결하고자 한다. 

일반적으로 그래프 기반 SSL에서 액티브 러닝(Active Learning)을 수행하려면, 새로운 데이터를 추가했을 때 모델이 어떻게 변할지를 예측하는 'look-ahead' 계산이 필요하다. 기존의 가우시안 모델(예: Gaussian Random Field)에서는 닫힌 형태(closed-form)의 해가 존재하여 효율적인 계산이 가능했으나, 실제 분류 문제에 더 적합한 비가우시안 모델(예: Probit 모델)을 사용할 경우 매 쿼리마다 모델을 전체적으로 다시 학습시켜야 하는 막대한 계산 비용이 발생한다. 따라서 본 연구의 목표는 비가우시안 분포를 가우시안으로 근사하여, 가우시안 모델에서 사용하던 효율적인 획득 함수(acquisition function)들을 비가우시안 모델에서도 사용할 수 있도록 하는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **비가우시안 모델을 위한 가우시안 근사 프레임워크**: Laplace Approximation을 통해 Probit 모델과 같은 비가우시안 사후 분포를 가우시안 분포로 근사하여, 기존의 가우시안 기반 획득 함수들을 적용할 수 있게 하였다.
2. **Newton Approximation (NA) 기반의 효율적인 업데이트**: 모델의 전체 재학습 없이도 'look-ahead' 상태의 MAP(Maximum A Posteriori) 추정치와 공분산 행렬을 빠르게 계산할 수 있는 랭크-1(rank-one) 업데이트 방식을 제안하였다.
3. **새로운 Model Change (MC) 획득 함수 제안**: 모델의 파라미터 변화량을 기반으로 가장 정보량이 많은 지점을 선택하는 MC 획득 함수를 도입하여, 계산 효율성과 성능의 균형을 맞추었다.

## 📎 Related Works

논문에서는 그래프 기반 SSL의 기초가 되는 Harmonic Functions (HF) 모델과 Gaussian Regression (GR) 모델을 언급한다. 기존의 액티브 러닝 연구들은 주로 다음과 같은 획득 함수들을 사용해 왔다.

- **Uncertainty/Margin**: 모델이 예측하기 가장 불확실한 지점을 선택하는 방식이다.
- **VOpt / $\Sigma$-Opt**: 분산 최소화나 공분산 구조를 이용하여 대표성 있는 지점을 선택한다.
- **MBR (Minimum Expected Risk)**: 새로운 점을 추가했을 때 기대 리스크를 최소화하는 지점을 찾는 look-ahead 방식이다.

기존의 이러한 방법들은 대부분 가우시안 가정 하에서 유도되었으며, 비가우시안 모델의 경우 계산 복잡도 문제로 인해 이러한 정교한 획득 함수들을 적용하는 데 한계가 있었다. 본 논문은 이러한 간극을 가우시안 근사와 Newton 업데이트를 통해 극복한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 최적화 문제
입력 데이터 $X$에 대해 유사도 커널 $\kappa$를 이용해 가중치 행렬 $W$를 생성하고, 그래프 라플라시안(Graph Laplacian) $L$을 정의한다. 정규화된 라플라시안 $L_n$ 또는 비정규화된 라플라시안 $L_u$를 사용하며, $\tau > 0$에 대해 $L_\tau = \tau^{-2}(L + \tau^2 I)$를 정의하여 가우시안 사전 분포 $\mathcal{N}(0, L_\tau^{-1})$를 설정한다.

모델은 다음의 목적 함수를 최소화하는 함수 $u$를 찾는 문제로 정의된다.
$$u^* = \arg \min_{u \in \mathbb{R}^N} \frac{1}{2} \langle u, L_\tau u \rangle + \sum_{j \in L} \ell(u_j, y_j)$$
여기서 $\ell$은 손실 함수이며, $\ell(x, y) = (x-y)^2/2\gamma^2$인 경우 Gaussian Regression (GR) 모델이 되고, $\ell(x, y) = -\log \Psi_\gamma(xy)$ ($\Psi_\gamma$는 Probit CDF)인 경우 Probit 모델이 된다.

### 2. Laplace Approximation
비가우시안 사후 분포를 가우시안 분포 $\mathcal{N}(\hat{u}, \hat{C})$로 근사한다.
- **평균 $\hat{u}$**: 목적 함수 $J_\ell(u; y)$를 최소화하는 MAP 추정치이다.
- **공분산 $\hat{C}$**: $\hat{u}$ 지점에서의 헤시안(Hessian) 행렬의 역행렬로 정의된다.
$$\hat{C} = (\nabla \nabla J_\ell(u; y) | _{u=\hat{u}})^{-1}$$

### 3. Newton Approximation (NA) Update
새로운 점 $k$와 라벨 $y_k$가 추가되었을 때, 전체 재학습 대신 Newton's Method의 단일 단계를 사용하여 $\tilde{u}_{k, y_k}$를 근사한다.
$$\tilde{u}_{k, y_k} = \hat{u} - \frac{F(\hat{u}_k, y_k)}{1 + \hat{C}_{k,k} F'(\hat{u}_k, y_k)} \hat{C}_{:,k}$$
여기서 $F$와 $F'$는 손실 함수의 1차 및 2차 도함수이다. 공분산 행렬 $\hat{C}$ 또한 다음과 같은 랭크-1 업데이트로 근사된다.
$$\tilde{C}_{k, y_k} = \hat{C} - \frac{F'(\tilde{u}_{k, y_k, k}, y_k)}{1 + \hat{C}_{k,k} F'(\tilde{u}_{k, y_k, k}, y_k)} \hat{C}_{:,k} \hat{C}_{:,k}^T$$

### 4. Model Change (MC) 획득 함수
모델의 변화량을 최대화하는 지점을 찾는 max-min 프레임워크를 제안한다.
$$k_{MC} = \arg \max_{k \in U} \min_{y_k \in \{\pm 1\}} \| \hat{u} - \tilde{u}_{k, y_k} \|^2$$
이는 새로운 데이터가 추가되었을 때 모델의 예측값(분류기)이 가장 크게 변하는 지점을 선택함으로써 정보 획득량을 최대화하려는 전략이다.

## 📊 Results

### 1. 실험 설정
- **Checkerboard Dataset**: $2,000$개의 점을 $4 \times 4$ 체커보드 패턴으로 분류. 군집 탐색과 결정 경계 학습 능력을 평가한다.
- **MNIST Dataset**: $4,000$개의 이미지(각 숫자 400개)를 사용하여 짝수/홀수 이진 분류 수행. $15$-nearest neighbor 그래프를 생성하여 사용한다.
- **비교 대상**: HF, GR, Probit 모델 각각에 대해 MC, VOpt, MBR, Uncertainty, Random 획득 함수를 적용하여 정확도를 비교하였다.

### 2. 주요 결과
- **Checkerboard 결과**: GR과 Probit 모델에서의 MC 방법, 그리고 Probit-MBR이 가장 우수한 성능을 보였다. 특히 MC는 모든 군집을 식별함과 동시에 군집 간의 결정 경계를 효율적으로 탐색하였다. 반면 VOpt는 군집의 대표점만 찾고 경계를 탐색하지 못해 정확도가 낮았다.
- **MNIST 결과**: MBR이 가장 높은 정확도를 보였으나 계산 비용이 매우 높았다. 제안된 MC 방법은 MBR보다 훨씬 효율적이면서도 매우 경쟁력 있는 정확도를 달성하였다.
- **NA 업데이트의 정확성**: Newton Approximation을 통해 계산된 $\tilde{u}$와 $\tilde{C}$가 실제 모델을 전체 재학습시켜 얻은 $\hat{u}$와 $\hat{C}$와 매우 유사함을 확인하였다 (Figure 3).

## 🧠 Insights & Discussion

본 논문은 비가우시안 모델의 현실적인 분류 성능과 가우시안 모델의 계산 효율성이라는 두 마리 토끼를 잡기 위해 근사 기법을 도입하였다. 

- **강점**: Laplace Approximation과 Newton Approximation의 조합을 통해, 기존에 가우시안 모델 전유물이었던 look-ahead 기반 획득 함수들을 비가우시안 모델로 확장시킨 점이 매우 뛰어나다. 특히 MC 함수는 계산 효율성이 높으면서도 실질적인 성능 향상을 이끌어냈다.
- **한계**: 본 연구는 이진 분류(Binary Classification) 문제에 국한되어 있으며, 한 번에 하나의 점만 선택하는 순차적(sequential) 방식만을 다루었다. 
- **해석**: 비가우시안 모델의 사후 분포를 근사함으로써, 단순한 가우시안 회귀보다 더 풍부한 라벨링 정보를 모델에 반영할 수 있게 되었으며, 이것이 결과적으로 더 나은 쿼리 선택으로 이어졌다고 판단된다.

## 📌 TL;DR

이 논문은 그래프 기반 준지도 학습에서 계산 비용이 큰 비가우시안 모델(Probit)을 효율적으로 사용하기 위해 **Laplace 근사와 Newton Approximation(NA) 기반의 랭크-1 업데이트**를 제안하였다. 이를 통해 복잡한 재학습 없이도 정교한 액티브 러닝 획득 함수를 사용할 수 있게 되었으며, 특히 제안된 **Model Change (MC)** 함수는 계산 효율성과 분류 정확도 면에서 매우 뛰어난 성능을 보였다. 이 연구는 향후 다중 클래스 분류나 배치 모드 액티브 러닝으로 확장될 가능성이 크다.