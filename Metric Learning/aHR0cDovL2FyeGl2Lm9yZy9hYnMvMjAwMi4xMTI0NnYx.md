# Supervised Categorical Metric Learning with Schatten p-Norms

Xuhui Fan, Eric Gaussier (2020)

## 🧩 Problem to Solve

본 논문은 범주형 데이터(categorical data)를 위한 지도 학습 기반의 거리 학습(Metric Learning) 문제를 해결하고자 한다. 일반적으로 수치형 데이터에 대한 거리 학습은 많은 성과를 거두었으나, 범주형 데이터의 경우 여전히 탐색이 필요한 분야이다. 범주형 데이터를 처리하는 표준적인 방법은 이를 이진 벡터(binary vectors)로 변환하여 수치형 데이터처럼 취급하는 것이지만, 이 방식은 특성(feature)의 수가 다항적으로 증가하여 차원의 저주 문제를 야기한다.

기존의 범주형 거리 측정 방식 중 비지도 학습 방법은 단순한 중첩 유사도에 의존하며, 지도 학습 방법들은 데이터 샘플 간의 상관관계를 무시하거나 막대한 계산 비용이 발생한다는 한계가 있다. 또한, 학습된 거리 척도에 대한 일반화 성능의 이론적 보장(generalization bound)이 부족한 상태이다. 따라서 본 논문의 목표는 계산 효율성과 예측 정확도를 동시에 높이면서, 이론적 보장을 제공하는 범주형 투영 거리 학습(Categorical Projected Metric Learning, CPML) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 범주형 특성을 클래스 기반의 벡터로 투영한 뒤, 투영된 공간에서 특성 간의 상관관계를 반영하는 거리 척도를 학습하는 것이다. 이를 위해 다음과 같은 기여를 한다.

1. **CPML 프레임워크 제안**: Value Distance Metric(VDM)을 사용하여 범주형 데이터를 투영하고, 이를 바탕으로 새로운 거리 함수인 CPm(Multi-metric)과 CPs(Single-metric)를 정의하여 효율적인 학습을 가능케 한다.
2. **Schatten $p$-norm 정규화**: 다양한 정규화 도구(Trace norm, Frobenius norm 등)를 일반화할 수 있는 Schatten $p$-norm을 도입하여 메트릭의 고윳값(eigenvalues)을 정규화하고 저차원 해(low-rank solutions)를 유도한다.
3. **이론적 일반화 경계 제시**: Schatten $p$-norm에 대한 Rademacher complexity를 분석하여, 범주형 거리 학습의 일반화 경계(generalization bound)를 수학적으로 증명한다.
4. **효율성 및 성능 검증**: 제안 방법이 기존 벤치마크 모델보다 계산 속도가 빠르면서도 경쟁력 있는 분류 정확도를 보임을 입증한다.

## 📎 Related Works

기존의 거리 학습 연구는 주로 수치형 데이터와 Mahalanobis 거리에 집중되었으며, LMNN이나 ITML과 같은 방법들이 대표적이다. 범주형 데이터의 경우 Hamming 거리와 같이 단순한 일치 여부를 따지는 방식이나, 조건부 확률 밀도 함수(cpdf)를 이용한 연관 기반 척도(association-based metric), 그리고 상호 정보량을 이용한 문맥 기반 척도(context-based metric) 등이 제안되었다.

그러나 이러한 기존 방식들은 다음과 같은 한계가 있다.

- **Hamming 거리**: 특성 간의 의존성이나 클래스 정보를 모델링할 수 없다.
- **연관/문맥 기반 척도**: 특성들이 서로 독립적일 경우 거리 값이 0이 되는 등 제대로 작동하지 않는 경우가 많다.
- **최근 지도 학습 방법**: KDML과 같은 방법은 계산 복잡도가 클래스 수 $C$와 특성 수 $D$에 대해 $O(C^3 D^3)$에 비례하여, 클래스 수가 많아질 경우 사실상 적용이 불가능하다.

## 🛠️ Methodology

### 1. Value Distance Metric (VDM) 기반 투영

범주형 특성 $x_{id}$가 값 $f$를 가질 때, 이를 클래스 분포를 나타내는 $C$차원 벡터 $\phi(x_{id}=f)$로 변환한다. 각 요소는 해당 특성 값이 특정 클래스 $c$에 속할 확률의 추정치로 계산된다.
$$\hat{P}(y=c|x_{id}=f) = \frac{N_{cd}(f)}{\sum_{c=1}^C N_{cd}(f)}$$
여기서 $N_{cd}(f)$는 클래스 $c$에서 특성 $d$의 값이 $f$로 나타난 횟수이다.

### 2. 거리 정의 (CPm 및 CPs)

투영된 데이터 $\phi(x_i) \in \mathbb{R}^{D \times C}$를 사용하여 두 가지 거리 함수를 정의한다.

- **Categorical Projected Multi (CPm) distance**: 클래스마다 서로 다른 메트릭 $M_c$를 학습한다.
$$d_M(x_i, x_j) = \sum_{c=1}^C \text{Tr}(A_{ij}^c M_c^\top)$$
여기서 $A_{ij}^c = (\phi^c(x_i) - \phi^c(x_j))(\phi^c(x_i) - \phi^c(x_j))^\top$이며, $M_c \in \mathbb{R}^{D \times D}_+$이다.

- **Categorical Projected Single (CPs) distance**: 모든 클래스가 하나의 메트릭 $M$을 공유한다고 가정한다.
$$d_M(x_i, x_j) = \text{Tr}(A_{ij} M^\top)$$
여기서 $A_{ij} = (\phi(x_i) - \phi(x_j))(\phi(x_i) - \phi(x_j))^\top$이다.

### 3. 목적 함수 및 최적화

학습 목표는 다음의 목적 함수 $f(M)$을 최소화하는 것이다.
$$\arg \min_M f(M) = \arg \min_M \{ \epsilon(M) + \lambda r(M) \}$$

- **손실 함수 $\epsilon(M)$**: 주로 Triplet constraint set $T = \{(i, j, k) | d_M(x_i, x_j) < d_M(x_i, x_k)\}$를 기반으로 한 Hinge loss를 사용한다.
$$\epsilon_T(M) = \frac{1}{|T|} \sum_{(i,j,k) \in T} [d_M(x_i, x_j) + b - d_M(x_i, x_k)]_+$$
- **정규화 함수 $r(M)$**: Schatten $p$-norm $\|M\|_p = [\text{Tr}(M^p)]^{1/p}$을 사용한다. 이는 $p=1$일 때 Trace norm, $p=2$일 때 Frobenius norm이 된다.

**학습 절차**:
Projected Subgradient Descent 방법을 사용한다. 매 반복마다 subgradient를 계산하여 $M$을 업데이트한 후, $M$의 음수 고윳값을 0으로 설정하여 Positive Semi-Definite(PSD) 원뿔 영역으로 다시 투영(Project)한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 합성 데이터(Synthetic data) 및 23개의 UCI 실세계 데이터셋.
- **비교 모델**: Euclidean, LMNN, KDML, DM3, HELIC, POLA.
- **평가 지표**: Triplet comparison accuracy 및 k-Nearest Neighbor 기반의 Classification accuracy.

### 2. 주요 결과

- **정확도**: 23개의 실세계 데이터셋 대부분에서 CPML 방법이 다른 베이스라인보다 우수하거나 대등한 성능을 보였다. 특히 CPML-multi가 CPML-single보다 약간 더 높은 성능을 내는 경향이 있다.
- **강건성**: 합성 데이터 실험을 통해 노이즈 특성(noisy features)이 다수 포함되어 있어도 Schatten $p$-norm 정규화를 통해 노이즈의 영향을 $1.7\%$ 미만으로 억제하며 성공적으로 "디노이징"할 수 있음을 확인하였다.
- **계산 효율성**:
  - CPML-s의 시간 복잡도는 $O(nDCs_{\max} + LTD^3)$이며, CPML-m은 $O(nDCs_{\max} + LTCD^3)$이다.
  - 이는 기존 KDML의 $O(C^3 D^3)$에 비해 훨씬 효율적이다. 실험 결과, 특히 클래스 수 $C$가 증가할 때 CPML-s의 실행 시간이 클래스 수의 영향을 거의 받지 않고 가장 빠르게 작동함을 보였다.

## 🧠 Insights & Discussion

본 논문은 범주형 데이터의 거리 학습에서 계산 복잡도와 이론적 보장이라는 두 가지 난제를 동시에 해결하였다.

**강점**:

- **효율적 투영**: VDM을 통해 범주형 데이터를 직접 다루지 않고 투영된 공간에서 연산함으로써 차원의 증가 문제를 피하고 계산량을 획기적으로 줄였다.
- **일반화된 정규화**: Schatten $p$-norm을 도입하여 기존의 다양한 정규화 기법을 하나의 틀로 통합하였으며, 이에 대한 일반화 경계를 수학적으로 증명함으로써 모델의 신뢰성을 높였다.

**한계 및 논의**:

- **지도 학습의 제약**: 본 모델은 모든 학습 데이터의 레이블이 필요하다는 전제가 있다. 저자들은 향후 연구로 레이블 정보가 일부만 있는 준지도 학습(semi-supervised learning) 시나리오를 위한 새로운 투영 함수 설계의 필요성을 언급하였다.
- **특성 값 분포의 영향**: 각 특성이 가지는 값의 개수나 분포가 거리 계산에 영향을 줄 수 있으나, 본 논문에서는 이 부분을 심도 있게 다루지 않았다.

## 📌 TL;DR

본 논문은 범주형 데이터 학습의 효율성을 높이기 위해 **Value Distance Metric(VDM) 투영**과 **Schatten $p$-norm 정규화**를 결합한 **CPML(Categorical Projected Metric Learning)** 프레임워크를 제안한다. 제안된 방법은 기존 모델 대비 계산 시간을 크게 단축시키면서도(특히 클래스 수가 많을 때) 높은 분류 정확도를 유지하며, 이론적인 일반화 성능 보장까지 제공한다. 이 연구는 대규모 범주형 데이터셋을 활용한 거리 기반 머신러닝 알고리즘의 실용성을 높이는 데 기여할 것으로 보인다.
