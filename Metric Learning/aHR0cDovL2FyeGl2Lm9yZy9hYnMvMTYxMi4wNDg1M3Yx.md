# Constraint Selection in Metric Learning

Hoel Le Capitaine (2016)

## 🧩 Problem to Solve

본 논문은 데이터 간의 유사도를 측정하기 위해 거리 함수(metric)를 학습하는 Metric Learning 분야에서, 제약 조건(constraint)을 선택하는 효율적인 방법에 대해 다룬다. 일반적으로 Euclidean distance가 널리 사용되지만, 데이터의 분포에 따라 Mahalanobis metric과 같은 파라미터화된 거리를 학습하는 것이 더 효율적이다.

기존의 많은 Metric Learning 알고리즘들은 학습 과정에서 사용할 제약 조건(쌍 또는 삼조합)을 무작위로 선택(random selection)한다. 그러나 이러한 무작위 선택 방식은 다음과 같은 두 가지 주요 문제점을 가진다. 첫째, 클래스 간의 경계(boundary)와 같이 판별이 어려운 중요한 특징 공간 영역에 집중하지 못한다. 둘째, 학습이 진행됨에 따라 업데이트되는 현재의 metric 상태를 반영하지 못하고 정적인 선택 방식을 유지한다. 특히 클래스가 서로 겹쳐 있거나(overlapping) 데이터의 규모가 매우 큰 경우, 이러한 한계는 정확도 저하와 확장성(scalability) 문제로 이어진다. 따라서 본 논문의 목표는 반복적인 metric 학습 알고리즘에서 학습 효율과 정확도를 높이기 위해, 현재의 loss에 기반하여 제약 조건을 동적으로 선택하는 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Loss-dependent Weighted Instance Selection (LWIS)**라는 제약 조건 선택 메커니즘을 제안한 것이다. 

중심 아이디어는 Boosting의 원리와 유사하게, 현재의 metric으로 측정했을 때 제약 조건을 만족하지 못하는 '어려운(hard)' 샘플에 더 높은 가중치를 부여하고, 이미 잘 만족되고 있는 '쉬운' 샘플의 가중치는 낮추는 것이다. 이를 통해 학습 알고리즘이 특징 공간의 경계 영역에 위치한 까다로운 관측치들에 집중하게 함으로써, 판별력을 극대화하고 학습 속도를 개선한다. 이 방법은 특정 알고리즘에 종속되지 않고, 제약 조건을 기반으로 하는 모든 반복적 Metric Learning 알고리즘에 적용 가능하다는 범용성을 가진다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들을 소개하고 차별점을 제시한다.

- **LMNN (Large Margin Nearest Neighbors):** PSD(Positive Semi-Definite) 행렬 공간에서 convex optimization을 통해 Mahalanobis distance를 학습한다. 하지만 정규화 항이 없어 overfitting에 취약하며, 초기 이웃 선택 시 Euclidean distance를 사용하므로 초기 metric이 부적절할 수 있다는 한계가 있다.
- **ITML (Information-Theoretic Metric Learning):** LogDet divergence를 regularizer로 사용하여 PSD 제약 조건을 보장한다. 효율적이지만 업데이트 복잡도가 차원의 제곱($O(cp^2)$)에 비례하여 고차원 데이터셋에 부적합하다.
- **OASIS (Online Algorithm for Scalable Image Similarity):** Passive-Aggressive 알고리즘을 기반으로 한 온라인 학습 방식이다. PSD 제약 조건을 제거하여 속도를 높였으나, 본 논문의 제안처럼 동적인 샘플 가중치 조절은 수행하지 않는다.
- **Liu & Vemuri 및 Mei et al.:** 경계 지역의 샘플이 중요하다는 점에 착안한 연구들이다. Liu & Vemuri 방식은 학습 시작 전 단 한 번 제약 조건을 선택하므로 metric의 변화를 반영하지 못하며, Mei et al. 방식은 매 반복마다 제약 조건을 업데이트하지만 모든 샘플 쌍의 거리 행렬(distance matrix)을 계산해야 하므로 대규모 데이터셋에서 계산 비용이 매우 높다.

본 논문의 제안 방식인 LWIS는 Mei et al.처럼 동적으로 업데이트를 수행하면서도, 거리 행렬 전체를 계산하지 않고 현재 선택된 제약 조건에 대해서만 loss를 계산하여 가중치를 업데이트하므로 계산 효율성과 정확도를 동시에 잡았다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. Mahalanobis Distance
본 논문은 다음과 같이 정의된 squared Mahalanobis distance를 기본 metric으로 사용한다.
$$d^2_A(x_i, x_j) = (x_i - x_j)^T A (x_i - x_j)$$
여기서 $A$는 $p \times p$ 크기의 positive semi-definite (PSD) 행렬이다. 학습의 목표는 주어진 제약 조건을 가장 잘 만족하는 행렬 $A$를 찾는 것이다.

### 2. 제약 조건별 Loss 함수 정의
LWIS는 제약 조건의 유형에 따라 서로 다른 Hinge loss 형태의 함수를 정의하여 각 샘플의 '어려움'을 측정한다. (단, $\gamma$는 마진 파라미터이다.)

- **Pairwise (Similar):** 두 샘플 $x_i, x_j$가 유사할 때, 거리가 $\gamma$보다 작아야 한다.
  $$\ell_i = \max(0, d_A(x_i, x_j) - \gamma)$$
- **Pairwise (Dissimilar):** 두 샘플 $x_i, x_k$가 서로 다를 때, 거리가 $\gamma$보다 커야 한다.
  $$\ell_i = \max(0, \gamma - d_A(x_i, x_k))$$
- **Triplet:** $x_i$가 $x_j$보다 $x_k$에 더 멀리 있어야 한다.
  $$\ell_i = \max(0, \gamma - d_A(x_i, x_k) + d_A(x_i, x_j))$$
- **Relative:** $x_i, x_j$의 거리가 $x_k, x_l$의 거리보다 작아야 한다.
  $$\ell_i = \max(0, \gamma - d_A(x_k, x_l) + d_A(x_i, x_j))$$

### 3. 동적 가중치 업데이트 (LWIS)
각 샘플 $x_i$에 대한 가중치 $w_i$는 매 반복($t$)마다 다음과 같이 업데이트된다.
$$w^{t+1}_i = \frac{1}{Z_{t+1}} w^t_i e^{(\delta \alpha_i)}$$
여기서:
- $Z_{t+1}$은 $\sum w^{t+1}_i = 1$이 되도록 하는 정규화 계수이다.
- $\delta$는 가중치 변화율을 조절하는 파라미터이다.
- $\alpha_i$는 정규화된 오차이며 다음과 같이 계산된다: $\alpha_i = \frac{\ell_i}{d_A(x_i, \bullet)}$ (여기서 $\bullet$은 제약 조건에 따른 상대 샘플이다).

### 4. 학습 절차 (Algorithm 1)
1. 초기 metric 행렬 $A_0$를 단위 행렬 $I$로 설정하고, 가중치 $w$를 균등 분포로 초기화한다.
2. **반복 수행:**
   - 현재 가중치 $w$에 따라 샘플을 무작위로 추출하여 제약 조건(Pair, Triplet 등)을 생성한다.
   - 선택된 제약 조건을 이용하여 기존 metric 학습 알고리즘(예: ITML)에 따라 $A$를 업데이트한다.
   - 업데이트된 $A$를 사용하여 해당 제약 조건의 loss $\ell$을 계산한다.
   - 계산된 loss를 바탕으로 모든 샘플의 가중치 $w$를 업데이트한다.
3. $A$가 수렴할 때까지 반복한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** UCI 저장소의 8개 데이터셋(Balance, Wine, Iris, Ionosphere, Seeds, PenDigits, Sonar, Breast Cancer) 및 대규모 데이터셋인 Forest Cover Type을 사용하였다.
- **비교 대상:** ITML(Random selection), Liu & Vemuri, Mei et al.
- **측정 지표:** k-NN($k=3$) 분류 정확도 및 학습 소요 시간.

### 2. 주요 결과
- **정확도:** 모든 UCI 데이터셋에서 LWIS가 다른 세 가지 방식보다 일관되게 높은 예측 정확도를 보였다. 특히 제약 조건의 수가 적을 때 LWIS의 성능 향상이 두드러졌다.
- **시간 복잡도:** 
    - **Random**과 **Liu & Vemuri** 방식은 제약 조건을 사전에 결정하므로 가장 빨랐다.
    - **Mei et al.** 방식은 매 반복마다 거리 행렬을 계산하므로 데이터 규모가 커질수록 시간이 기하급수적으로 증가하여 대규모 데이터셋에서는 실행 불가능(untractable)했다.
    - **LWIS**는 매 반복 가중치를 업데이트하지만, 현재 제약 조건에 대해서만 거리를 계산하므로 시간 복잡도가 Random 방식과 유사하게 낮으면서도 Mei et al.보다 훨씬 빠른 속도를 보였다.
- **대규모 실험 (Forest Cover Type):** 관측치 수가 최대 581,012개인 환경에서 LWIS는 정확도와 확장성 면에서 압도적인 성능을 보였다. 데이터 수가 증가함에 따라 정확도가 상승하며, 타 방식(Mei et al., Liu & Vemuri)이 $O(n^2)$의 복잡도로 인해 무너지는 반면 LWIS는 안정적인 실행 시간을 유지했다.
- **파라미터 분석:** 
    - 마진 $\gamma$의 변화는 통계적으로 유의미한 성능 차이를 만들지 않았다 ($\gamma=2$가 일반적으로 좋은 결과를 보임).
    - 학습률 $\delta$의 경우 데이터셋마다 최적값이 달랐으나, 최대/최소 정확도 차이는 2% 이내로 영향이 제한적이었다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
LWIS의 성공 요인은 **학습의 초점을 동적으로 변경**했다는 점에 있다. 데이터가 잘 분리된 영역에서는 Euclidean distance만으로도 충분하므로, 굳이 복잡한 metric을 학습할 필요가 없다. 반면 클래스가 겹치는 경계 영역은 metric 학습의 결과가 성능에 직접적인 영향을 미친다. LWIS는 loss가 높은 샘플의 선택 확률을 높임으로써, 한정된 학습 자원을 가장 정보량이 많은 '어려운' 샘플에 집중 배치하는 효율적인 전략을 취했다.

### 2. 한계 및 논의사항
- **초기값 의존성:** 초기 metric $A_0$를 단위 행렬 $I$로 설정하는데, 만약 초기 상태에서 데이터 분포가 매우 특이하다면 초기 가중치 설정 단계에서 편향이 발생할 가능성이 있다.
- **수렴 속도:** 실험 결과, 제약 조건이 고정된 방식(Random, Liu & Vemuri)보다 제약 조건이 계속 변하는 방식(LWIS, Mei et al.)이 수렴까지 더 많은 반복 횟수를 요구하는 경향이 있다. 이는 동일한 정보만 반복 학습하는 것보다 새로운 정보를 계속 학습하는 것이 수렴은 느리지만 최종 성능은 더 높다는 것을 시사한다.
- **정규화 효과:** 단순히 경계 샘플만 학습하는 것이 아니라, 가중치가 낮더라도 모든 샘플이 선택될 가능성을 열어둠으로써 모델이 지나치게 경계에만 overfitting되는 것을 방지하고 일반화 성능을 유지하였다.

## 📌 TL;DR

본 논문은 Mahalanobis metric 학습 시 제약 조건을 무작위로 선택하던 기존 방식의 한계를 극복하기 위해, **현재의 loss에 따라 샘플 가중치를 동적으로 조절하는 LWIS(Loss-dependent Weighted Instance Selection) 방법론**을 제안한다. 분류가 어려운 샘플(높은 loss)에 가중치를 부여하여 학습 효율을 높였으며, 실험을 통해 **기존의 정적 선택 방식보다 높은 정확도**를, **동적 선택 방식(Mei et al.)보다 월등한 시간 복잡도**를 달성함을 입증하였다. 특히 대규모 데이터셋에서도 확장성이 뛰어나 향후 대용량 데이터의 Metric Learning 및 유사도 학습 시스템에 적용될 가능성이 높다.