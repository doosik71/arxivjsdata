# Unsupervised Domain Adaptation with Progressive Domain Augmentation

Kevin Hua and Yuhong Guo (2020)

## 🧩 Problem to Solve

본 논문은 레이블이 풍부한 Source domain의 지식을 활용하여 레이블이 부족하거나 없는 Target domain의 분류기를 학습시키는 Unsupervised Domain Adaptation (UDA) 문제를 다룬다. 특히, 두 도메인 간의 분포 차이인 Domain divergence(또는 Covariate shift)가 매우 클 때, 기존의 직접적인 정렬(Direct alignment) 방식은 정보 손실을 초래하거나 성능이 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 Source domain과 Target domain 사이의 거대한 간극을 한 번에 메우려 하지 않고, 점진적인 보간(Interpolation)을 통해 가상 도메인을 생성함으로써 이 간극을 여러 개의 작은 단계로 나누어 해결하는 Progressive Domain Augmentation (PrDA) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **가상 중간 도메인(Virtual Intermediate Domains)**을 생성하여 Source domain을 Target domain 방향으로 점진적으로 확장(Augmentation)시키는 것이다.

구체적으로는 Mixup 기법을 활용해 두 도메인 사이의 보간된 샘플들을 생성하고, 이를 통해 생성된 가상 도메인을 징검다리 삼아 Source domain의 분포를 Target domain으로 서서히 이동시킨다. 이 과정에서 Grassmann manifold 상의 Multiple Subspace Alignment를 수행하여 기하학적으로 두 도메인을 정렬하며, 신뢰도가 높은 가상 샘플들을 Source domain에 추가함으로써 Target domain에 최적화된 분류기를 학습시킨다.

## 📎 Related Works

논문에서는 기존의 UDA 방법론을 세 가지 범주로 분류하여 설명한다.

1.  **Divergence-based methods**: MMD, CORAL, Wasserstein distance 등을 사용하여 두 도메인 간의 분포 차이를 최소화하고 도메인 불변 특징(Domain-invariant features)을 학습한다.
2.  **Adversarial-based methods**: GAN의 원리를 이용하여 도메인 판별기(Domain discriminator)를 속이는 방식으로 특징 공간을 정렬한다. (예: DANN, CyCADA)
3.  **Subspace-based methods**: 저차원 부분 공간(Subspace)에서 도메인을 정렬한다. (예: GFK, SA)

**기존 방식의 한계 및 차별점**: 위 방법들은 대부분 Source와 Target 도메인을 직접적으로 정렬하려고 시도한다. 하지만 도메인 간의 괴리가 매우 클 경우, 직접적인 정렬은 심각한 정보 손실을 유발할 수 있다. 반면, 제안된 PrDA는 가상 도메인을 통한 **점진적 정렬**을 수행함으로써 큰 도메인 간극을 보다 세밀하고 안정적으로 극복한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. Virtual Domain Generation (가상 도메인 생성)
Mixup의 선형 보간 원리를 이용하여 Source 샘플 $x_s^i$와 Target 샘플 $x_t^j$로부터 가상 샘플 $\hat{x}$를 생성한다.

$$\hat{x} = \lambda x_s^i + (1-\lambda) x_t^j$$

여기서 $\lambda$는 가상 샘플이 어느 도메인에 더 가깝게 위치할지를 결정하는 가중치이다. $\lambda$ 값이 높으면 Source domain에 가깝고, 낮으면 Target domain에 가깝다. PrDA는 $\lambda$ 값을 초기값(예: 0.8)에서 점차 낮추어 가며 가상 도메인이 Source에서 Target 방향으로 서서히 이동하도록 제어한다.

### 2. Multiple Subspace Generation and Alignment (다중 부분 공간 생성 및 정렬)
단일 부분 공간만으로는 도메인의 복잡한 기하학적 구조를 모두 담기 어려우므로, PCA를 이용해 여러 개의 부분 공간을 생성하는 Multiple Subspace Alignment 기술을 사용한다.

-   **부분 공간 생성**: PCA를 통해 상위 $k$개의 주성분을 추출하여 첫 번째 부분 공간을 만든다. 이후, 해당 부분 공간으로 설명되지 않는 오차가 큰 샘플들($\tau$ 임계값 기준)만을 모아 다시 PCA를 수행하는 과정을 반복하여 부분 공간 집합 $M$을 구성한다.
-   **부분 공간 매칭**: Grassmann manifold 상의 두 부분 공간 $B_s$와 $B_u$ 사이의 거리를 측정하기 위해 Chordal metric $d^\Delta$를 사용한다.
    $$d^\Delta(B_s, B_u) = \left( k - \sum_{i,j=1}^{k} (b_{si}^T \cdot b_{uj})^2 \right)^{1/2}$$
    이 거리가 가장 가까운 부분 공간 쌍을 순차적으로 매칭한다.
-   **선형 변환 및 투영**: 매칭된 각 부분 공간 쌍에 대해 Frobenius norm을 최소화하는 변환 행렬 $A^*$를 구한다.
    $$A^* = \arg \min_A ||B_{si}A - B_{uj}||_F = (B_{si})^T B_{uj}$$
    이 행렬을 통해 Source 데이터를 Target 정렬 부분 공간으로 투영하여 $Z_s$를 얻고, 가상 도메인 데이터를 $Z_u$로 투영한다.

### 3. Overall Training Pipeline (전체 학습 절차)
전체 프로세스는 다음과 같은 루프로 진행된다 (Algorithm 2 참조).

1.  **사전 학습**: Source 데이터로 특징 추출기 $G$와 분류기 $F$를 학습시킨다.
2.  **가상 도메인 생성**: 현재 $\lambda$ 값으로 가상 샘플 집합 $X_u$를 생성한다.
3.  **부분 공간 정렬**: Source와 가상 도메인의 부분 공간을 생성하고, 위에서 설명한 방식으로 정렬하여 데이터를 투영시킨다.
4.  **의사 레이블링(Pseudo-labeling)**: 정렬된 공간에서 학습된 분류기 $H$를 이용해 가상 샘플 $X_u$의 레이블 $\hat{Y}_u$를 예측한다.
5.  **도메인 확장**: 예측된 레이블의 신뢰도가 임계값 $\rho$보다 높은 샘플들만 선택하여 Source 데이터셋에 추가한다.
6.  **업데이트**: 확장된 Source 데이터셋으로 다시 모델을 학습하고, $\lambda$를 감소시켜 다음 단계로 넘어간다.

## 📊 Results

### 실험 설정
-   **데이터셋**: Office-31, Office-Caltech-10, ImageCLEF-DA
-   **백본**: ResNet-50 (Source domain에서 파인튜닝)
-   **하이퍼파라미터**: 부분 공간 차원 $k=44$, $\lambda$ 값은 $[0.8, 0.6, 0.4, 0.2]$ 순으로 감소.

### 주요 결과
-   **정량적 성능**: Office-31 데이터셋에서 SA, CORAL, RWA, MEDA 등 기존 방법론보다 전반적으로 우수한 성능을 보였으며, 특히 $A \to D$ 태스크에서 가장 높은 성능을 기록했다. Office-Caltech-10의 12개 태스크 중 7개에서 최고 성능(SOTA)을 달성하였다.
-   **도메인 간극 분석**: 본 논문은 '도메인 분류기'를 이용해 두 도메인이 얼마나 쉽게 구분되는지(즉, Divergence가 얼마나 큰지) 측정하는 실험을 추가로 진행했다. 실험 결과, **도메인 간의 괴리가 컸던 태스크일수록 PrDA의 성능 향상 폭이 더 컸음**이 확인되었다. 이는 큰 간극을 작은 단계로 나누어 해결하려는 PrDA의 접근 방식이 유효함을 입증한다.

## 🧠 Insights & Discussion

**강점**: 
본 연구는 기존의 "한 번에 정렬하는" 방식의 위험성을 지적하고, "점진적 보간"이라는 직관적인 해결책을 제시했다. 특히 수학적으로 엄밀한 Grassmann manifold 상의 다중 부분 공간 정렬을 결합하여, 데이터 증강의 불안정성을 기하학적 정렬로 보완한 점이 돋보인다.

**한계 및 논의사항**:
-   **하이퍼파라미터 의존성**: $\lambda$의 감소 순서, 신뢰도 임계값 $\rho$, 부분 공간 오차 임계값 $\tau$ 등 튜닝해야 할 하이퍼파라미터가 많다. 이러한 값들이 데이터셋마다 어떻게 최적화되어야 하는지에 대한 일반적인 가이드라인이 부족하다.
-   **선형 보간의 가정**: Mixup을 통한 선형 보간이 실제 데이터 매니폴드(Manifold) 상의 분포 이동을 정확히 반영하는지에 대한 이론적 분석보다는 실험적 결과에 의존하고 있다.

## 📌 TL;DR

이 논문은 Source와 Target 도메인 간의 큰 차이로 인해 발생하는 UDA의 성능 저하 문제를 해결하기 위해, **Mixup 기반의 가상 중간 도메인을 생성하고 이를 점진적으로 정렬하는 PrDA 방법론**을 제안한다. 다중 부분 공간 정렬(Multiple Subspace Alignment)을 통해 기하학적으로 도메인을 매칭하고 신뢰도 높은 샘플을 Source에 추가하는 방식으로, 특히 **도메인 간격이 큰 환경에서 기존 SOTA 방법론들을 능가하는 성능**을 보여주었다. 향후 매우 이질적인 도메인 간의 지식 전이 연구에 중요한 기초가 될 수 있는 연구이다.