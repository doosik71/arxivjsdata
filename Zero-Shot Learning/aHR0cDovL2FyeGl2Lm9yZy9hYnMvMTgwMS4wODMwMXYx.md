# Class label autoencoder for zero-shot learning

Guangfeng Lin, Caixia Fan, Wanjun Chen, Yajun Chen, Fan Zhao (2018)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL)에서 발생하는 두 가지 주요 문제점을 해결하고자 한다. 첫째는 기존의 ZSL 방법론들이 일반적으로 특징 공간(feature space)과 단일한 시맨틱 임베딩 공간(semantic embedding space, 예: 텍스트 또는 속성 공간) 사이의 투영 함수(projection function)를 학습한다는 점이다. 그러나 실제로는 동일한 클래스를 설명하는 다양한 시맨틱 정보가 존재하며, 기존의 단일 투영 방식으로는 이러한 multi-semantic embedding spaces의 다양성을 충분히 수용하기 어렵다.

둘째는 학습 데이터에 포함된 seen classes와 테스트 데이터의 unseen classes가 서로 분리되어 있어 발생하는 project domain shift 문제이다. 이로 인해 seen classes에서 학습된 투영 함수가 unseen classes에 그대로 적용될 때 성능이 저하되는 문제가 발생한다. 따라서 본 논문의 목표는 multi-semantic 정보를 통합적으로 활용할 수 있는 균일한 프레임워크를 구축하고, 특징 공간과 클래스 레이블 공간 사이의 양방향 투영 제약을 통해 unseen classes의 인식 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 클래스 레이블 오토인코더(Class Label Autoencoder, CLA)를 도입하여 특징 공간과 클래스 레이블 공간 사이의 양방향 제약 조건을 형성하는 것이다. 주요 기여 사항은 다음과 같다.

1. **양방향 투영 구조의 설계**: 특징 공간에서 클래스 레이블 공간으로의 인코딩과 다시 특징 공간으로의 디코딩 과정을 포함하는 오토인코더 메커니즘을 구축함으로써, 단순한 단방향 투영보다 더 강력한 제약 조건을 제공한다.
2. **Multi-semantic Structure Evolution Fusion**: 속성(attribute), 단어 벡터(word vector), GloVe, 계층적 임베딩(hierarchical embedding) 등 다양한 시맨틱 소스에서 얻은 구조 정보와 시각적 특징의 구조 정보를 선형 결합하여 최적의 유사도 행렬을 구성한다.
3. **반복적 진화 프로세스(Evolution Process)**: 초기 추정된 unseen classes의 레이블을 바탕으로 시각적 구조 정보를 업데이트하고, 이를 통해 다시 레이블을 정교화하는 반복적인 최적화 과정을 통해 인식 성능을 높인다.

## 📎 Related Works

ZSL 연구는 주로 시맨틱 정보를 활용하여 unseen classes를 인식하는 방향으로 발전해 왔다. 기존 접근 방식은 크게 중간 속성 분류기 학습, seen class의 조합, 그리고 시맨틱 임베딩 공간에서의 호환성 학습(compatibility learning) 등으로 나뉜다.

특히 본 연구와 밀접한 관련이 있는 두 가지 접근법은 Semantic Autoencoder (SAE)와 Structure Propagation (SP)이다. SAE는 특징-시맨틱 공간 간의 양방향 제약을 통해 latent manifold 구조를 탐색하지만, 단일 시맨틱 공간만을 다룬다는 한계가 있다. 반면 SP는 시맨틱 및 이미지 공간의 매니폴드 구조를 최적화하고 반복적인 전파를 통해 성능을 높이지만, 특징-레이블 공간 사이의 양방향 제약을 고려하지 않는다. CLA는 이 두 가지 아이디어를 결합하여 multi-semantic 정보를 수용하면서도 양방향 투영 제약을 동시에 달성함으로써 기존 방법론들과 차별화된다.

## 🛠️ Methodology

### 1. Linear Autoencoder 기반의 기초 모델

CLA는 기본적으로 특징 공간 $X \in \mathbb{R}^{d \times N}$를 클래스 레이블 공간 $Y \in \mathbb{R}^{k \times N}$로 투영하는 변환 행렬 $Q \in \mathbb{R}^{k \times d}$를 학습한다. 오토인코더 구조를 적용하여 $X \to Y \to \hat{X}$의 과정을 거치며, 이때 디코딩 행렬은 $Q^T$를 사용하여 가중치를 공유(tied weights)한다. 목적 함수는 다음과 같이 원래 데이터와 재구성된 데이터 사이의 오류를 최소화하는 것이다.

$$Q = \arg \min_{Q} \|X - Q^T Q X\|_F^2 \quad \text{s.t. } QX = Y$$

이를 제약 조건이 없는 최적화 문제로 변환하면 다음과 같다.
$$\min_{Q} \|X - Q^T Y\|_F^2 + \lambda \|QX - Y\|_F^2$$
여기서 $\lambda$는 인코더와 디코더 사이의 균형을 조절하는 trade-off 파라미터이다.

### 2. ZSL 모델로의 확장 및 분해

ZSL 상황에서는 seen classes의 투영 행렬 $Q_s$에서 unseen classes의 투영 행렬 $Q_u$로 지식을 전이해야 한다. 본 논문에서는 $Q$를 유사도 행렬 $W$ (클래스 간의 구조적 관계)와 투영 행렬 $A$ (공통 정보 추출)의 곱으로 분해한다.

- **Seen classes**: $Q_s = W_s^T A_s$
- **Unseen classes**: $Q_u = W_u^T A_u$
- **Knowledge Transfer**: $A_u = W_{su}^T A_s$

여기서 $W_s, W_u, W_{su}$는 각각 seen, unseen, 그리고 seen-unseen 간의 유사도 행렬이다. 유사도 $w_{ij}$는 다음과 같은 가우시안 커널 형태의 거리 함수를 통해 계산된다.
$$w_{ij} = \frac{\exp(-d(z_i, z_j))}{\sum_{n_i, n_j} \exp(-d(z_i, z_j))}, \quad d(z_i, z_j) = (z_i - z_j)^T \Sigma_z^{-1} (z_i - z_j)$$

결과적으로 seen classes에 대해 다음과 같은 최적화 문제를 푼다.
$$A_s = \arg \min_{A_s} \|X_s - A_s^T W_s Y_s\|_F^2 + \lambda \|W_s^T A_s X_s - Y_s\|_F^2$$
이 식은 Sylvester 방정식으로 변환되어 Bartels-Stewart 알고리즘을 통해 효율적으로 해결된다.

### 3. Multi-semantic Structure Evolution Fusion

다양한 시맨틱 소스가 있을 때, 최종 유사도 행렬 $W$를 각 소스에서 얻은 유사도 행렬 $W^{(i)}$의 선형 결합으로 정의한다. 또한, 시각적 특징의 구조 정보인 $W^{(I)}$를 함께 포함한다.

$$W = \sum_{i=1}^{M} \beta_i W^{(i)} + \beta_{M+1} W^{(I)} \quad (\sum \beta_i = 1)$$

학습 과정은 다음과 같은 반복적 진화 프로세스를 따른다.

1. **초기화**: $W^{(I)}$를 0으로 설정하고 가중치 $\beta, \gamma$를 균등하게 설정한다.
2. **Seen 학습**: $\beta$를 고정하여 $A_s$를 최적화하고, 다시 $A_s$를 고정하여 $\beta$를 선형 프로그래밍으로 최적화한다.
3. **Unseen 추정**: 학습된 $A_s$와 유사도 행렬을 이용하여 unseen classes의 레이블 $\hat{y}$를 추정한다.
4. **진화(Evolution)**: 추정된 $\hat{y}$를 사용하여 시각적 구조 행렬 $W^{(u, I)}$와 $W^{(su, I)}$를 업데이트하고, $\gamma$ 가중치를 최적화하여 유사도 행렬을 갱신한다. 이 과정을 $P$번 반복한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AwA, CUB, Dogs, ImNet-2의 4개 벤치마크 데이터셋을 사용하였다.
- **특징 추출**: 이미지 특징은 pre-trained GoogleNet의 1024차원 벡터를 사용하였으며, 시맨틱 특징은 attributes, word2vec, GloVe, hierarchical embedding 등 4가지 방식을 사용하였다.
- **평가 지표**: 각 클래스별 평균 Top-1 정확도를 측정하였다.

### 주요 결과

- **Baseline 비교**: SAE 및 SP 방법론과 비교했을 때, CLA는 모든 데이터셋에서 우수한 성능을 보였다. 특히 AwA, CUB, Dogs에서 각각 최소 2%, 3.7%, 3.2%의 성능 향상을 기록하였다.
- **Multi-semantic Fusion**: SJE, LatEm, SynC 등 다른 융합 방법론보다 높은 성능을 보였으며, 특히 다양한 시맨틱 소스를 모두 활용했을 때($w$ 설정) 성능이 극대화되었다.
- **State-of-the-art 비교**: 대부분의 최신 방법론보다 우수한 성능을 보였으나, CUB 데이터셋의 attribute 공간에서는 DMaP가 더 높은 성능을 보였다. 이는 DMaP가 세밀한 분류(fine-grained)에 유리한 매니폴드 구조 일관성에 집중했기 때문으로 분석된다.
- **Top-n 정확도**: ImNet-2와 같은 대규모 데이터셋에서는 Top-1 성능 향상이 미미했으나, Top-2부터 Top-5까지의 정확도는 CLA가 가장 높게 나타났다.

## 🧠 Insights & Discussion

본 논문은 특징 공간과 클래스 레이블 공간 사이의 **양방향 제약(bidirectional constraints)**과 **구조적 진화(structure evolution)**를 결합하는 것이 ZSL 성능 향상의 핵심임을 입증하였다.

특히, 단순히 시맨틱 정보를 고정해서 사용하는 것이 아니라, 추정된 레이블을 바탕으로 시각적 구조 정보를 동적으로 업데이트하는 진화 프로세스가 unseen classes의 결정 경계를 정교화하는 데 큰 역할을 한다. 대규모 데이터셋(ImNet-2)에서 Top-1보다 Top-n 성능이 더 두드러지게 나타난 점은, 클래스 수가 많아질수록 클래스 내 분산(intra-class divergence)이 커져 단일 정답을 맞히는 것이 어려워지지만, 후보군을 좁히는 능력은 CLA가 탁월함을 시사한다.

계산 복잡도 측면에서는 최적화 과정에서 Sylvester 방정식과 선형 프로그래밍을 해결해야 하므로 $O(d^3)$의 복잡도를 가진다. 여기서 $d$는 특징 차원이므로, 시맨틱 공간의 차원이나 반복 횟수 $P$보다는 특징 차원의 영향이 가장 크다.

## 📌 TL;DR

본 논문은 multi-semantic 임베딩 공간을 효율적으로 활용하기 위해 특징 공간과 클래스 레이블 공간 사이의 양방향 투영을 학습하는 **Class Label Autoencoder (CLA)**를 제안한다. 다양한 시맨틱 소스와 시각적 구조 정보를 통합하고 이를 반복적으로 정교화하는 진화 프로세스를 통해, 기존의 단방향 투영 기반 ZSL 방법론들의 한계를 극복하고 unseen classes 인식 성능을 유의미하게 향상시켰다. 이 연구는 향후 복합적인 시맨틱 정보가 제공되는 실제 환경의 Zero-Shot 인식 시스템 구축에 중요한 기반이 될 가능성이 높다.
