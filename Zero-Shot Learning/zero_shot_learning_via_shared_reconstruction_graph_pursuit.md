# Zero-shot Learning via Shared-Reconstruction-Graph Pursuit

Bo Zhao, Xinwei Sun, Yuan Yao, Yizhou Wang (2017)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 **Zero-shot Learning (ZSL)** 환경에서 발생하는 **Space Shift Problem**이다. ZSL은 학습 데이터가 전혀 없는 새로운 클래스(unseen classes)를 인식하는 것을 목표로 하며, 이를 위해 일반적으로 이미지 특징 공간(Image Feature Space)과 시맨틱 임베딩 공간(Semantic Embedding Space) 사이의 지식을 전이하는 방식을 사용한다.

기존의 구조 전이(structure-transfer) 기반 방법론들은 두 공간의 기하학적 구조가 유사할 것이라고 가정하고 시맨틱 공간의 구조를 이미지 공간으로 직접 전이한다. 하지만 저자들은 이미지 공간(시각적 인식 기반), 속성 공간(인간의 지식 기반), 단어 벡터 공간(언어 모델 기반)이 각각 서로 다른 데이터와 방법으로 구축되었기 때문에, 두 공간 사이의 기하학적 구조가 일치하지 않는 **Space Shift Problem**이 발생한다고 지적한다. 이로 인해 단순한 구조 전이는 성능 저하를 야기하며, 이를 해결하기 위해 두 공간 모두에 적응 가능한 공통의 구조적 지식을 학습하는 것이 논문의 주요 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 두 임베딩 공간 사이의 공통 구조를 캡처하는 **Shared Reconstruction Graph (SRG)**를 제안한 것이다.

중심 아이디어는 각 클래스의 프로토타입(prototype)을 다른 클래스들의 선형 결합으로 재구성(reconstruct)할 때, 이미지 공간과 시맨틱 공간에서 동일한 재구성 계수(reconstruction coefficients)를 공유하도록 학습시키는 것이다. 즉, 특정 클래스가 다른 클래스들과 갖는 관계(가중치)가 두 공간에서 공통적으로 적용된다고 가정함으로써 Space Shift Problem을 완화한다. 학습된 SRG를 통해 보이지 않는 클래스(unseen class)의 이미지 프로토타입을 합성하고, 이를 기반으로 테스트 이미지를 분류한다.

## 📎 Related Works

ZSL의 기존 문제점과 접근 방식은 다음과 같이 구분된다.

1. **Domain Shift Problem**: 매핑 전이(mapping-transfer) 프레임워크에서 주로 발생하며, 학습된 클래스(seen)와 테스트 클래스(unseen) 간의 시각적 특징 분포가 달라 발생하는 문제이다. 이를 해결하기 위해 transductive setting(레이블 없는 테스트 데이터를 학습에 이용) 등이 제안되었으나, 이는 실제 상황에서 비현실적일 수 있다는 한계가 있다.
2. **Structure-transfer Methods**: 매니폴드(manifold) 구조 등을 전이하여 Domain Shift를 완화하려 했으나, 앞서 언급한 Space Shift Problem으로 인해 한계가 존재한다.
3. **Graph-based Methods**: 주로 단일 임베딩 공간의 그래프 구조만을 활용하거나 하이퍼그래프를 이용한 랜덤 워크 방식을 사용했다. 반면, 본 논문은 두 공간의 클래스 간 관계를 모두 고려하는 공유 그래프를 제안하여 차별점을 둔다.
4. **Sparse Subspace Clustering (SSC)**: 데이터가 선형 부분 공간(linear subspace)에서 생성되었다는 가정을 바탕으로 희소성 제약을 사용하는 방식이다. 본 논문은 SSC의 아이디어를 차용하여 클래스 간의 희소한 관계를 학습하고, 이를 통해 해석 가능한 클러스터를 발견한다.

## 🛠️ Methodology

### 1. 클래스 프로토타입 (Class Prototype)

각 클래스의 데이터를 대표하는 중심점을 프로토타입으로 정의한다.

- **이미지 프로토타입 ($f_k$)**: 해당 클래스에 속하는 모든 이미지 특징 벡터의 평균값이다.
- **시맨틱 프로토타입 ($e_k$)**: 제공된 속성(attribute) 벡터나 단어 벡터(word vector)이다.

### 2. Shared Reconstruction Graph (SRG) 구축

각 클래스 프로토타입을 다른 클래스들의 선형 결합으로 표현한다. 시맨틱 공간에서의 재구성은 다음과 같다.
$$e_k = \sum_{i=1, i \neq k}^{K} \alpha_{ik} e_i = E\alpha_k$$
여기서 $\alpha_k$는 재구성 계수 벡터이며, $\alpha_{kk}=0$ 제약을 통해 자기 자신을 제외한 다른 클래스들로만 재구성하게 한다.

본 논문은 이미지 공간에서도 동일한 계수 $\alpha_k$를 사용하여 $f_k \approx F\alpha_k$가 성립하도록 하는 **Shared Reconstruction Graph (SRG)**를 학습한다. 전체 손실 함수 $\mathcal{L}$은 다음과 같이 정의된다.
$$\mathcal{L} = \sum_{k=1}^{K} (\|e_k - E\alpha_k\|_F^2 + \gamma\|f_k - F\alpha_k\|_F^2), \quad \text{s.t. } \alpha_{kk}=0, \gamma < 1$$
여기서 $\gamma$는 이미지 공간의 재구성 손실에 적용되는 가중치이며, unseen 클래스의 이미지 프로토타입 $F_u$가 없으므로 $\gamma < 1$로 설정하여 불확실성을 조절한다.

### 3. 희소성 및 지역성 규제 (Sparsity and Locality Regularization)

의미 없는 약한 연결을 제거하고 관련성이 높은 클래스만 선택하기 위해 $\ell_1$-norm 규제와 지역성 페널티를 추가한다.
$$\mathcal{L} = \sum_{k=1}^{K} (\|e_k - E\alpha_k\|_F^2 + \gamma\|f_k - F\alpha_k\|_F^2 + \lambda\|D_k \alpha_k\|_1)$$

- **Sparsity**: $\ell_1$-norm을 통해 $\alpha_k$를 희소하게 만들어 소수의 관련 클래스만 선택하게 한다.
- **Locality**: 대규모 데이터셋(ImageNet 등)에서 멀리 떨어진 클래스의 영향을 배제하기 위해 대각 행렬 $D_k$를 도입한다. $D_{ii,k} = g(e_i, e_k)$로 설정하여, 시맨틱 거리 $g(\cdot)$가 멀수록 더 큰 페널티를 부여한다.

### 4. 최적화 (Optimization)

손실 함수가 $F_u$와 $A$에 대해 동시에는 비볼록(non-convex)하지만 각각에 대해서는 볼록(convex)하므로, **교대 최적화(Alternating Optimization)** 알고리즘을 사용한다.

1. **$A$ 업데이트**: $F_u$를 고정한 상태에서 $\alpha_k$를 최적화한다. 이는 전형적인 LASSO 문제로 변환되어 LeastR 등의 솔버로 해결 가능하다.
2. **$F_u$ 업데이트**: $A$를 고정한 상태에서 unseen 이미지 프로토타입 $F_u$를 최적화한다. 최적해는 다음과 같이 도출된다.
    $$F_u = -F_s \theta_s (\theta_u)^{-1}, \quad \text{where } \theta = I - A$$
이 과정을 수렴할 때까지 반복한다.

### 5. Zero-shot 분류 및 확장

학습된 $F_u$를 통해 unseen 클래스의 가상 프로토타입을 생성하고, 테스트 이미지 $x^u_i$와 가장 가까운 프로토타입을 찾는 **Nearest Neighbor (NN)** 분류기를 사용한다. 또한, 테스트 데이터가 seen과 unseen 클래스 모두에서 나올 수 있는 **Generalized Zero-shot Learning (GZSL)** 설정으로 쉽게 확장 가능하다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: AwA (coarse-grained), CUB (fine-grained), ImageNet (large-scale).
- **특징 추출**: VGG-19 및 GoogLeNet+ResNet 특징 사용.
- **지표**: Top-1 및 Top-5 정확도.

### 2. 정량적 결과

- **소규모 데이터셋 (AwA, CUB)**: AwA에서 83.62%, CUB에서 58.10%의 정확도를 달성하여 SOTA 성능을 보였다. 특히 RKT 방법론보다 높은 성능을 기록하며 Space Shift Problem 완화의 효과를 입증했다.
- **대규모 데이터셋 (ImageNet)**: Top-1 8.14%, Top-5 18.26%를 기록하여 기존의 ConSE, DeViSE 등의 방법론을 상회했다.
- **학습 샘플 수에 따른 성능**: 매우 적은 수의 학습 샘플(클래스당 1~3개 이미지)만으로도 다른 SOTA 방법론보다 빠르게 성능이 상승하는 효율성을 보였다.

### 3. 규제 항 분석

실험 결과, $\ell_1$ 희소성 규제는 모든 데이터셋에서 성능을 유의미하게 향상시켰으며, ImageNet과 같은 대규모 데이터셋에서는 지역성(Locality) 규제가 추가적인 성능 향상을 가져옴을 확인했다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석 가능성

본 논문의 가장 큰 강점은 SRG가 단순히 분류 성능만 높이는 것이 아니라 **해석 가능성(Interpretability)**을 제공한다는 점이다. 학습된 SRG에 대해 Spectral Clustering을 적용한 결과, 생물학적/시맨틱적으로 의미 있는 클러스터(예: 수생 동물, 고양잇과 동물 등)가 자동으로 형성됨을 확인했다. 이는 모델이 두 공간의 공통적인 구조적 지식을 성공적으로 학습했음을 시사한다.

### 2. 한계 및 가정

본 모델은 각 클래스가 특징 공간에서 tight한 클러스터를 형성한다는 가정을 전제로 한다. 만약 클래스 내 분산이 매우 크거나 멀티모달 분포를 가진다면, 단일 프로토타입으로 대표하는 방식이 한계가 있을 수 있다. 또한, $\ell_1$ 규제와 관련된 하이퍼파라미터 $\lambda$와 $\gamma$에 대한 교차 검증 과정이 필수적이다.

### 3. 비판적 해석

Space Shift Problem이라는 개념을 명확히 정의하고 이를 해결하기 위해 '공유 재구성 계수'라는 단순하지만 강력한 제약을 도입한 점이 훌륭하다. 다만, 선형 재구성에 의존하고 있으므로 클래스 간의 관계가 복잡한 비선형 구조를 가질 경우 이를 완전히 캡처하기 어려울 수 있다.

## 📌 TL;DR

본 논문은 ZSL에서 이미지 공간과 시맨틱 공간의 기하학적 구조가 서로 다른 **Space Shift Problem**을 정의하고, 이를 해결하기 위해 두 공간에서 공통으로 사용되는 **Shared Reconstruction Graph (SRG)**를 제안하였다. 클래스 프로토타입 간의 재구성 계수를 공유하도록 학습함으로써 unseen 클래스의 이미지 특징을 성공적으로 합성하였으며, 이를 통해 AwA, CUB, ImageNet 등 주요 벤치마크에서 SOTA 성능을 달성하였다. 특히 적은 양의 학습 데이터로도 높은 효율을 보이며, 학습된 그래프를 통해 클래스 간의 의미 있는 군집을 발견할 수 있다는 점에서 학술적, 실무적 가치가 높다.
