# Domain Adaptation with Optimal Transport on the Manifold of SPD matrices

Or Yair, Felix Dietrich, Ioannis G. Kevrekidis, and Ronen Talmon (2020)

## 🧩 Problem to Solve

본 논문은 서로 다른 도메인에서 수집된 데이터 간의 차이를 줄이기 위한 Domain Adaptation (DA) 문제를 다룬다. 특히, 데이터가 Euclidean 공간이 아닌 Symmetric and Positive-Definite (SPD) 행렬의 Cone Manifold 상에 존재하는 경우에 집중한다.

데이터 수집 시스템의 차이, 환경 설정의 변화, 또는 측정 대상(피험자)의 차이로 인해 동일한 작업에 대해서도 데이터의 표현 방식이 달라지는 문제가 발생한다. 이러한 도메인 간의 불일치는 분류기나 분석 모델의 성능을 저하시키는 주요 원인이 된다. 따라서 본 연구의 목표는 소스 도메인(Source Domain)의 데이터를 타겟 도메인(Target Domain)으로 매핑하여, 두 도메인의 표현을 일치시키면서도 데이터가 가진 본질적인 정보는 유지하는 효율적인 DA 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Riemannian Manifold 상에서 Optimal Transport (OT)를 적용하여 소스 도메인을 타겟 도메인으로 전송하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **이론적 분석**: McCann의 Polar Factorization Theorem을 이용하여, Riemannian Manifold 상에서의 OT가 부피 보존 사상(Volume preserving map)을 제외하고는 DA를 위한 최적의 해결책임을 이론적으로 제시하였다.
2.  **알고리즘 제안**: SPD 행렬 manifold의 기하학적 특성과 Sinkhorn 알고리즘을 결합하여, 구현이 쉽고 효율적인 DA 알고리즘을 제안하였다.
3.  **성능 검증**: Brain-Computer Interface (BCI)의 두 가지 데이터셋(Motor Imagery, P300 ERP)에 적용하여 기존 최신 기법(SOTA)보다 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 Domain Adaptation 연구들은 주로 Euclidean 공간에서의 데이터 분포를 맞추는 방식에 집중해 왔다. 특히 Courty 등이 제안한 OT 기반의 DA 방식은 유클리드 공간에서 효과적이었으나, SPD 행렬과 같이 특수한 기하학적 구조를 가진 데이터에는 직접 적용하기 어렵다.

또한, BCI 분야에서는 SPD 공분산 행렬(Covariance Matrix)을 이용한 Riemannian 기하학적 접근법이 사용되어 왔으며, Affine Transform (AT)이나 Parallel Transport (PT)와 같은 기법들이 도메인 간 차이를 줄이기 위해 제안되었다. 본 논문은 이러한 기존의 기하학적 접근법과 달리, 두 확률 분포 간의 최적 전송 계획(Transport Plan)을 찾는 OT 관점에서 접근함으로써 더 일반적이고 강력한 매핑 방법을 제공한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. SPD 행렬의 Cone Manifold
본 논문은 SPD 행렬 집합을 Riemannian Manifold $\mathcal{P}_d$로 모델링한다. 이 공간에서의 내적은 다음과 같이 정의된다.
$$\langle A, B \rangle_{T_P \mathcal{P}_d} = \langle P^{-1/2} A P^{-1/2}, P^{-1/2} B P^{-1/2} \rangle$$
여기서 $\langle \cdot, \cdot \rangle$은 Euclidean 내적이다. 두 SPD 행렬 $P, Q$ 사이의 Riemannian 거리 $d_R(P, Q)$는 다음과 같다.
$$d_R^2(P, Q) = \| \log(Q^{-1/2} P Q^{-1/2}) \|_F^2$$

### 2. Optimal Transport (OT) 기반 DA 파이프라인
제안하는 알고리즘은 소스 집합 $\mathcal{P} = \{P_i\}_{i=1}^{N_1}$를 타겟 집합 $\mathcal{Q} = \{Q_j\}_{j=1}^{N_2}$로 매핑하는 과정을 거친다.

**단계 1: 비용 행렬(Cost Matrix) 계산**
소스와 타겟의 모든 쌍에 대해 Riemannian 거리의 제곱을 비용으로 설정한다.
$$C[i, j] = d_R^2(P_i, Q_j)$$

**단계 2: 전송 계획(Transport Plan) 도출**
계산 효율성을 위해 엔트로피 정규화 항 $h(\Gamma)$가 추가된 Sinkhorn OT 문제를 해결하여 최적의 전송 계획 $\Gamma$를 구한다.
$$\min_{\Gamma \in \mathcal{F}} \langle \Gamma, C \rangle - \frac{1}{\lambda} h(\Gamma)$$
여기서 $h(\Gamma) = -\sum_{i,j} \Gamma_{ij} \log \Gamma_{ij}$이며, $\lambda$는 정규화 파라미터이다.

**단계 3: Barycentric Mapping을 통한 데이터 변환**
구해진 $\Gamma$를 이용하여 소스 데이터 $P_i$를 타겟 도메인의 가중 평균 지점으로 이동시킨다.
$$\tilde{P}_i = \arg \min_{P \in \mathcal{P}_d} \sum_{j=1}^{N_2} \Gamma_{ij} d_R^2(P, Q_j)$$
SPD Manifold에서는 Riemannian 가중 평균(Weighted Mean)이 유일하게 존재하므로, 이 최적화 문제는 엄격하게 볼록(strictly convex)하며 유일한 해 $\tilde{P}_i$를 갖는다.

### 3. 레이블 정보의 활용
소스 데이터에 레이블 $y$가 존재하는 경우, 동일한 레이블을 가진 샘플들이 서로 가깝게 매핑되도록 유도하는 정규화 항을 OT 목적 함수에 추가한다.
$$\min_{\Gamma \in \mathcal{F}} \langle \Gamma, C \rangle - \frac{1}{\lambda} h(\Gamma) + \eta \sum_{j=1}^{N_2} \sum_{y=1}^{|Y|} \| \Gamma(I_y, j) \|_q^p$$
이 식은 레이블 정보가 없는 unsupervised setting보다 더 정교한 도메인 적응을 가능하게 한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: BCI Competition IV (Motor Imagery), Brain Invaders (P300 ERP)
- **비교 대상**: Affine Transform (AT), Parallel Transport (PT), Euclidean 기반 OT
- **평가 지표**: 분류 정확도(Accuracy), 정밀도(Precision)

### 2. 주요 결과
- **Motor Imagery Task**:
    - cross-session 및 cross-subject 분류 실험에서 제안 방법이 AT와 PT보다 높은 정확도를 기록하였다.
    - 특히 Riemannian 거리를 사용한 것이 Euclidean 거리를 사용한 것보다 성능이 월등히 높았다.
- **P300 ERP Task**:
    - Augmented Covariance Matrix를 사용한 실험에서 제안 방법이 가장 높은 정밀도를 보였으며, 소스 레이블을 활용했을 때 성능이 추가로 향상됨을 확인하였다.
- **정성적 결과**: t-SNE 시각화 결과, 적응 전에는 데이터가 세션이나 피험자별로 뭉쳐 있었으나, 알고리즘 적용 후에는 클래스(움직임 종류 등)별로 명확하게 클러스터링되는 것을 확인하였다.

## 🧠 Insights & Discussion

### 1. 이론적 한계와 강점
본 논문은 Polar Factorization Theorem을 통해 OT 기반 DA의 근본적인 한계를 명시하였다. 소스와 타겟의 분포만으로 DA를 수행할 경우, 최적의 전송 맵 $t$는 구할 수 있지만, 부피 보존 사상(Volume preserving map) $u$로 인한 왜곡까지는 완전히 회복할 수 없다. 즉, 데이터 밀도만으로는 완벽한 점대점(point-wise) 매핑을 보장할 수 없다는 점을 이론적으로 밝혔으며, 이는 모든 unsupervised DA 방법론에 적용되는 한계이다.

### 2. Riemannian Metric의 중요성
실험 결과, SPD 행렬을 단순히 Euclidean 공간의 벡터로 취급하여 OT를 적용하는 것보다 Riemannian Manifold의 기하학적 구조를 반영한 거리를 사용하는 것이 훨씬 효과적이었다. 이는 SPD 행렬이 갖는 비유클리드적 특성이 데이터의 본질적인 정보를 담고 있기 때문으로 해석된다.

### 3. 비판적 해석
제안된 방법론은 이론적으로 탄탄하고 실용적인 성능을 보였으나, 부피 보존 사상 $u$에 의한 왜곡 문제를 해결하기 위해 소량의 대응 쌍(source-target pairs)을 사용하는 방향의 후속 연구가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 **SPD 행렬의 Riemannian Manifold 상에서 Optimal Transport를 이용한 Domain Adaptation 기법**을 제안하였다. Polar Factorization Theorem을 통해 OT 기반 DA의 이론적 최적성과 한계를 분석하였으며, Sinkhorn 알고리즘과 Barycentric Mapping을 통해 구현 가능한 효율적인 알고리즘을 제시하였다. BCI 데이터셋 실험을 통해 **기존의 AT, PT 및 Euclidean OT보다 뛰어난 성능**을 입증함으로써, 고차원 특징 공간의 기하학적 구조를 반영한 도메인 적응의 중요성을 보여주었다. 이 연구는 뇌-컴퓨터 인터페이스뿐만 아니라 SPD 행렬을 사용하는 다양한 컴퓨터 비전 및 신호 처리 분야의 도메인 불일치 문제를 해결하는 데 기여할 가능성이 크다.