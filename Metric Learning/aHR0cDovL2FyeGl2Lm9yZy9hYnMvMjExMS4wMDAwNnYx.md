# Adaptive Hierarchical Similarity Metric Learning with Noisy Labels

Jiexi Yan, Lei Luo, Cheng Deng, Heng Huang (2021)

## 🧩 Problem to Solve

본 논문은 Deep Metric Learning (DML) 모델이 학습 데이터 내의 Noisy Labels(노이즈 섞인 레이블)에 매우 민감하게 반응하여 성능이 급격히 저하되는 문제를 해결하고자 한다. 

일반적인 DML 방식은 두 샘플이 같은 클래스인지 아닌지를 구분하는 이진 유사도(Binary Similarity) 정보에 의존한다. 하지만 실제 세계의 데이터에는 잘못된 레이블이 포함될 가능성이 높으며, 이러한 환경에서 단순히 이진 유사도만을 활용해 모델을 학습시키면 잘못된 Positive pair는 서로 가깝게 당기고, 잘못된 Negative pair는 서로 멀리 밀어내게 되어 임베딩 공간의 변별력이 상실된다. 따라서 본 연구의 목표는 노이즈에 둔감한(Noise-insensitive) 정보를 발굴하여, 레이블 노이즈가 존재하는 상황에서도 강건하고 일반화 능력이 뛰어난 DML 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이진 유사도의 한계를 넘어, 데이터 내에 잠재된 다층적 구조를 반영하는 **Adaptive Hierarchical Similarity (적응형 계층적 유사도)** 전략을 제안하는 것이다.

중심적인 설계 직관은 레이블 노이즈에 민감한 개별 샘플 간의 이진 관계 대신, 다음의 두 가지 노이즈-둔감 정보를 통합하여 사용하는 것이다:
1. **Class-wise Divergence (클래스 수준의 발산)**: 클래스 간의 평균적인 유사도를 측정하여, 단순한 '같음/다름'이 아닌 클래스 간의 상대적 거리를 반영한 동적 마진(Dynamic Margin)을 부여한다.
2. **Sample-wise Consistency (샘플 수준의 일관성)**: Self-supervised learning의 데이터 증강(Augmentation) 기법을 활용하여, 레이블과 무관하게 동일 샘플에서 파생된 뷰(view) 간의 일관성을 유지함으로써 모델의 강건성을 높인다.

이러한 정보를 통합하여 각 샘플 쌍에 적응형 마진을 부여함으로써, 데이터의 계층적 구조를 효율적으로 학습할 수 있도록 설계하였다.

## 📎 Related Works

### 1. Deep Metric Learning (DML)
기존 DML은 주로 Pair-based 방법(Triplet loss, Lifted Structure loss, MS loss 등)과 Proxy-based 방법으로 나뉜다. Pair-based 방법은 샘플 간의 거리를 직접 최적화하며, 특히 MS loss와 같은 최신 기법들은 변별력 있는 임베딩 공간을 구축하는 데 효과적이다. 그러나 이러한 방법들은 대부분 고정된 마진(Fixed Margin)과 이진 유사도에 의존하므로 노이즈 레이블에 매우 취약하다는 한계가 있다.

### 2. Learning with Noisy Labels
분류(Classification) 작업에서는 노이즈 전이 행렬(Noise transition matrix) 추정이나 소실 함수 수정, 또는 Small-loss 인스턴스 추출 등의 방법이 연구되었다. 하지만 이러한 방식들은 레이블 정보를 직접 사용하는 분류 작업에 특화되어 있어, 샘플 간의 유사도를 학습하는 DML의 원리에 직접 적용하기에는 어려움이 있다.

### 3. Hyperbolic Geometry 및 Self-Supervised Learning
하이퍼볼릭 공간(Hyperbolic Space)은 유클리드 공간과 달리 음의 곡률을 가져 트리(Tree) 구조와 같은 계층적 데이터를 표현하는 데 매우 적합하다. 또한, Contrastive Learning과 같은 자기지도 학습은 데이터 증강을 통해 레이블 없이도 유용한 특징을 학습할 수 있게 하며, 이는 레이블 노이즈 문제로부터 자유로운 대안이 될 수 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인
제안된 방법은 크게 세 단계로 구성된다: **Class-wise Hierarchy Construction $\rightarrow$ Contrastive Augmentation $\rightarrow$ Adaptive Hierarchical Similarity Integration**. 
먼저 하이퍼볼릭 공간에서 클래스 간 계층 구조를 파악하고, 동시에 강약 증강(Weak/Strong augmentation)을 통해 샘플 일관성을 확보한다. 마지막으로 이를 통합하여 동적 마진을 산출하고, 이를 기반으로 DML 손실 함수를 최적화한다.

### 2. Adaptive Hierarchical Similarity 상세 설명

#### A. Class-wise Divergence
클래스 내부의 평균 유사도(Intra-similarity, $S_{aa}$)와 클래스 간의 평균 유사도(Inter-similarity, $S_{ab}$)를 계산한다.
- **Intra-similarity**: $S_{aa} = \frac{2}{n_a^2 - n_a} \sum_{z_{ai}, z_{aj} \in C_a} \text{sim}(z_{ai}, z_{aj})$
- **Inter-similarity**: $S_{ab} = \frac{1}{n_a n_b} \sum_{z_{ai} \in C_a, z_{bj} \in C_b} \text{sim}(z_{ai}, z_{bj})$

이 값들을 $[0, 0.2]$ 범위로 매핑한 $\hat{S}_{aa}, \hat{S}_{ab}$를 사용하여, Positive pair와 Negative pair에 대한 동적 마진 $M_p, M_n$을 다음과 같이 정의한다:
$$M_p = \gamma + \hat{S}_{aa}$$
$$M_n = \gamma - \hat{S}_{ab}$$
여기서 $\gamma$는 고정된 하이퍼파라미터이다. 이는 클래스 간의 실제 유사도에 따라 밀어내는 정도를 다르게 설정함으로써 계층적 구조를 반영한다.

#### B. Sample-wise Consistency
레이블 노이즈의 영향을 피하기 위해, 동일 샘플에 대해 약한 증강($\alpha^-$)과 강한 증강($\alpha^+$)을 적용한다. 증강된 샘플들은 레이블과 상관없이 항상 Positive sample이므로, 이들 간의 거리를 가장 가깝게 유지하도록 유도한다.
이를 위한 증강 쌍의 마진 $M_a$는 다음과 같이 정의된다:
$$M_a = \min_{z_{ai}, z_{aj} \in C_a} \text{sim}(z_{ai}, z_{aj})$$

#### C. 통합 적응형 마진 및 손실 함수
위에서 정의한 $\{M_a, M_p, M_n\}$을 통합하여, 예를 들어 MS loss를 다음과 같이 수정하여 사용한다:
$$L_{MS}^* = \frac{1}{n} \sum_{i=1}^{n} \left\{ \frac{1}{\rho} \log \left[ 1 + \sum_{j \in A_i} e^{-\rho(d_{ij}-M_a)} \right] + \frac{1}{\%}\log \left[ 1 + \sum_{j \in P_i} e^{-\%(d_{ij}-M_p)} \right] + \frac{1}{\sigma} \log \left[ 1 + \sum_{j \in N_i} e^{\sigma(d_{ij}-M_n)} \right] \right\}$$
여기서 $A_i, P_i, N_i$는 각각 anchor $z_i$에 대한 증강 집합, positive 집합, negative 집합이다.

### 3. Hyperbolic DML Paradigm
데이터의 복잡한 계층 구조를 더 잘 포착하기 위해 유클리드 공간이 아닌 Poincaré ball 모델 기반의 하이퍼볼릭 공간을 사용한다. 두 점 $z_i, z_j$ 사이의 geodesic distance $d^D$는 다음과 같다:
$$d^D(z_i, z_j) = \cosh^{-1} \left( 1 + 2 \frac{\|z_i - z_j\|^2}{(1-\|z_i\|^2)(1-\|z_j\|^2)} \right)$$
유클리드 특징 벡터 $x$를 하이퍼볼릭 공간으로 매핑하기 위해 exponential map $\exp_\tau(x)$를 사용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Cars196, CUB-200-2011, Standard Online Products (SOP)
- **지표**: Recall@K (K=1, 2, 4, 8 등)
- **환경**: Inception backbone (ImageNet pre-trained), Adam optimizer, $\gamma = 0.5$
- **노이즈 설정**: 원본 데이터에 무작위로 레이블을 뒤바꾸어 30%, 50%, 70%의 노이즈 비율을 시뮬레이션함.

### 2. 주요 결과
- **Clean Dataset**: 제안 방법($MS^*$)이 기존 $MS$ loss보다 성능이 향상되었으며, 특히 CUB-200-2011의 Recall@1에서 66.8%로 SOTA 수준의 성능을 보였다.
- **Noisy Dataset (30% Noise)**: 기존 Baseline 모델들은 노이즈 발생 시 성능이 급격히 하락하였으나, 제안 방법은 이를 상당 부분 완화했다. (예: CUB-200-2011의 Recall@1이 62.0% $\rightarrow$ 65.3%로 3.3%p 향상)
- **Ablation Study**: 
    - Class-wise Divergence와 Sample-wise Consistency 모두 단독으로 사용했을 때보다 함께 사용했을 때 성능 향상이 뚜렷했다.
    - 하이퍼볼릭 기하학(HG)을 적용했을 때 유클리드 기반보다 더 높은 성능을 기록하여 계층 구조 표현의 유효성을 입증했다.
    - **극심한 노이즈 상황 (70% Noise)**: 노이즈 비율이 70%에 달하는 매우 어려운 상황에서도 Baseline 대비 Recall@1을 7.2%p 높이며 강력한 강건성을 보여주었다.

## 🧠 Insights & Discussion

본 논문의 강점은 단순히 노이즈를 제거하는 필터링 기법을 적용한 것이 아니라, DML의 본질인 '유사도 학습' 관점에서 노이즈에 둔감한 새로운 정보(계층적 유사도)를 정의하고 이를 동적 마진으로 연결했다는 점이다. 특히, 클래스 전체의 통계적 특성을 이용하는 Class-wise Divergence와 레이블이 필요 없는 Contrastive Augmentation을 결합함으로써 개별 레이블의 오류가 모델 전체의 학습 방향을 흐리는 것을 효과적으로 방지하였다.

또한, Poincaré ball 모델을 통한 하이퍼볼릭 임베딩의 도입은 실제 데이터(예: 새의 종 $\rightarrow$ Albatross $\rightarrow$ Bird)가 가진 자연스러운 계층 구조를 기하학적으로 잘 표현할 수 있음을 시사한다. 

다만, 본 논문에서는 모델 초기화를 위해 pre-trained 모델을 사용하였는데, 이는 초기 임베딩 공간이 어느 정도 정돈되어 있어야 Class-wise Divergence 계산이 유의미하다는 가정을 내포하고 있다. 만약 완전히 random하게 초기화된 상태에서 극심한 노이즈가 존재한다면, 초기 $S_{aa}, S_{ab}$ 값이 불안정하여 학습 초기 단계에서 불안정성이 나타날 가능성이 있다.

## 📌 TL;DR

본 연구는 레이블 노이즈에 취약한 기존 Deep Metric Learning의 문제를 해결하기 위해, 클래스 수준의 발산(Class-wise Divergence)과 샘플 수준의 일관성(Sample-wise Consistency)을 결합한 **Adaptive Hierarchical Similarity** 전략을 제안하였다. 이를 통해 샘플 쌍마다 최적화된 동적 마진을 부여하고 하이퍼볼릭 공간에서 학습함으로써, 최대 70%의 레이블 노이즈가 존재하는 환경에서도 기존 방식보다 뛰어난 강건성과 검색 성능을 달성하였다. 이 연구는 실제 세계의 불완전한 레이블 데이터를 활용한 정교한 특징 임베딩 학습에 중요한 기여를 한다.