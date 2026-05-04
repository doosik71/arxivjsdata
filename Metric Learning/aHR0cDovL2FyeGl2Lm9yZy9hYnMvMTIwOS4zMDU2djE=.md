# Parametric Local Metric Learning for Nearest Neighbor Classification

Jun Wang, Adam Woznica, Alexandros Kalousis (2012)

## 🧩 Problem to Solve

본 논문은 Nearest Neighbor (NN) 분류기에서 사용할 수 있는 Local Metric을 효율적으로 학습하는 방법을 다룬다. NN 분류기의 성능은 거리 측정 방식에 크게 의존하며, 기존의 Mahalanobis metric learning과 같은 Global Metric 방식은 데이터 공간의 모든 영역에 동일한 가중치를 적용하므로, 영역마다 변별력 있는 특징이 다른 데이터 매니폴드(Data Manifold)의 특성을 충분히 반영하지 못한다는 한계가 있다.

이를 해결하기 위해 각 데이터 포인트나 지역별로 서로 다른 Metric을 학습하는 Local Metric Learning 방식이 제안되어 왔으나, 대부분의 기존 연구들은 지역별 Metric들을 서로 독립적으로 학습시킨다. 이러한 독립적 접근 방식은 유연성을 높여주지만, 학습해야 할 파라미터 수가 급격히 증가하여 심각한 Overfitting(과적합) 위험을 초래한다는 치명적인 단점이 있다. 따라서 본 논문의 목표는 데이터 매니폴드 상에서 부드럽게 변화하는(Smooth) Metric Matrix 함수를 학습함으로써, 유연성과 일반화 성능을 동시에 확보하는 Parametric Local Metric Learning (PLML) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 지역의 Metric을 개별적으로 학습하는 대신, 소수의 앵커 포인트(Anchor Points)에서 정의된 Basis Metric들의 선형 결합으로 각 지점의 Local Metric을 표현하는 것이다.

구체적으로, 데이터 공간의 특정 지점 $x$에서의 Metric Matrix $M(x)$를 다음과 같은 파라미터화된 형태로 정의한다. 앵커 포인트들에 대응하는 Basis Metric들을 미리 정해두고, 각 인스턴스는 이들의 가중 합으로 자신의 Local Metric을 구성한다. 여기에 Manifold Regularization을 도입하여, 매니폴드 상에서 서로 가까운 데이터 포인트들은 유사한 가중치(Weight)를 갖도록 강제함으로써 Local Metric이 공간적으로 부드럽게 변화하도록 설계하였다. 이러한 구조는 학습 파라미터 수를 획기적으로 줄여 Overfitting을 방지하는 동시에, 지역적 특성을 반영할 수 있는 유연성을 제공한다.

## 📎 Related Works

논문에서는 기존의 Local Metric Learning 접근 방식들의 한계를 다음과 같이 지적한다.

1.  **DANN (Discriminant Adaptive Nearest Neighbor):** 지역 결정 경계에 수직인 방향으로 이웃을 축소하고 평행한 방향으로 확장하여 Metric을 학습한다. 그러나 지역 Metric 간의 정규화(Regularization)가 없어 Overfitting에 매우 취약하다.
2.  **LMNN-MM (LMNN-Multiple Metric):** 학습되는 Metric의 수를 제한하거나 특정 지역 내의 모든 인스턴스가 동일한 Metric을 공유하게 하여 Overfitting을 억제하려 한다. 하지만 여전히 각 지역의 Metric을 독립적으로 학습하므로 특정 지역에 너무 특화되는 경향이 있다.
3.  **GLML (Generative Local Metric Learning):** 생성 모델 기반으로 기대 분류 오류를 최소화하여 Metric을 학습하며, 각 클래스를 가우시안 분포로 모델링한다. 그러나 이러한 강한 모델 가정(Model Assumption)은 실제 데이터의 복잡한 분포를 반영하지 못해 유연성이 떨어진다.

PLML은 이러한 기존 방식들과 달리, Basis Metric의 선형 결합이라는 파라미터화와 Manifold Regularization을 통해 지역적 유연성과 전역적 부드러움을 동시에 달성하여 차별점을 가진다.

## 🛠️ Methodology

### 1. Local Metric의 파라미터화
본 연구에서는 각 인스턴스 $x_i$의 Local Metric $M_i$를 $m$개의 앵커 포인트 $U = \{u_1, \dots, u_m\}$에 대응하는 Basis Metric $\{M_{b_1}, \dots, M_{b_m}\}$의 가중 합으로 정의한다.

$$M_i = \sum_{b_k} W_{ib_k} M_{b_k}, \quad W_{ib_k} \ge 0, \quad \sum_{b_k} W_{ib_k} = 1$$

여기서 $W$는 각 인스턴스가 어떤 Basis Metric을 얼마나 사용할지를 결정하는 가중치 행렬이다. 이를 통해 두 인스턴스 $x_i, x_j$ 사이의 거리는 다음과 같이 계산된다.

$$d^2_{M_i}(x_i, x_j) = \sum_{b_k} W_{ib_k} d^2_{M_{b_k}}(x_i, x_j)$$

### 2. 1단계: Smooth Local Linear Weight Learning
먼저 Basis Metric들은 고정되어 있다고 가정하고, 각 인스턴스의 가중치 $W$를 학습한다. 목적 함수는 세 가지 항의 합으로 구성된다.

$$\min_{W} g(W) = \|X - WU\|_F^2 + \lambda_1 \text{tr}(WG) + \lambda_2 \text{tr}(W^T LW)$$

-   $\|X - WU\|_F^2$: 인스턴스가 앵커 포인트들의 선형 결합으로 잘 표현되도록 하는 재구성 오차항이다.
-   $\text{tr}(WG)$: 가중치가 인스턴스의 지역적 특성을 반영하도록 강제하는 항으로, $G$는 앵커 포인트와 인스턴스 간의 유클리드 거리 행렬이다.
-   $\text{tr}(W^T LW)$: Manifold Regularization 항이다. $L$은 Graph Laplacian 행렬이며, 이는 매니폴드 상에서 유사한 인스턴스들이 유사한 가중치 $W$를 갖도록 하여 Metric의 변화를 부드럽게 만든다.

이 문제는 Convex Quadratic 문제이며, Simplex 제약 조건($\sum W = 1, W \ge 0$) 하에서 FISTA(Fast Iterative Shrinkage-Thresholding Algorithm)를 사용하여 효율적으로 해결한다.

### 3. 2단계: Large Margin Basis Metric Learning
가중치 $W$가 결정되면, 이를 바탕으로 Basis Metric $\{M_{b_k}\}$들을 학습한다. 본 논문은 Large Margin Triplet Constraint를 사용하여 다음 목적 함수를 최소화한다.

$$\min_{M_{b_l}, \xi} \alpha_1 \sum_{b_l} \|M_{b_l}\|_F^2 + \sum_{ijk} \xi_{ijk} + \alpha_2 \sum_{ij} \sum_{b_l} W_{ib_l} d^2_{M_{b_l}}(x_i, x_j)$$
$$\text{s.t. } \sum_{b_l} W_{ib_l}(d^2_{M_{b_l}}(x_i, x_k) - d^2_{M_{b_l}}(x_i, x_j)) \ge 1 - \xi_{ijk}, \quad M_{b_l} \succeq 0$$

-   $\alpha_1 \|M_{b_l}\|_F^2$: Frobenius norm을 통해 정규화를 수행하며, 이는 SVM의 마진과 연관된다.
-   $\sum \xi_{ijk}$: Triplet 제약 조건을 만족하지 못한 오차의 합이다.
-   $\alpha_2 \sum \dots$: 각 인스턴스와 그 이웃들 간의 거리를 최소화하여 응집력을 높인다.

이 문제는 PSD(Positive Semi-Definite) 제약 조건 때문에 최적화가 어렵지만, 본 논문은 **Lagrangian Dual** 형태로 변환하여 해결한다. Dual formulation을 통해 $Z_{b_l}$에 대한 폐쇄형 솔루션(Closed-form solution)을 얻을 수 있으며, 최종적으로는 $\gamma$에 대한 최적화 문제로 단순화되어 FISTA 알고리즘을 통해 빠르게 학습할 수 있다.

## 📊 Results

### 실험 설정
-   **데이터셋:** Letter, USPS, Pendigits, Optdigits, Isolet, MNIST (5K~70K 샘플의 대규모 데이터셋).
-   **비교 대상:** 
    -   Single Metric: SML, LMNN, BoostMetric.
    -   Local Metric: CBLML(클러스터 기반), LMNN-MM, GLML.
    -   기타: 자동 커널 선택을 적용한 Multi-class SVM.
-   **평가 지표:** 1-NN 분류 정확도 및 McNemar 통계 테스트를 통한 유의성 검정.

### 주요 결과
1.  **예측 성능:** Table 1에 따르면, PLML은 대부분의 데이터셋에서 다른 모든 Metric Learning 방법들보다 우수하거나 대등한 성능을 보였다. 특히 Global Metric 방식인 LMNN이나 BoostMetric보다 유의미하게 높은 정확도를 기록했다.
2.  **Overfitting 방지:** Local Metric 방법인 CBLML과 LMNN-MM은 SML(Global)보다 성능이 낮게 나오는 경우가 많았는데, 이는 과적합이 발생했음을 시사한다. 반면 PLML은 Local Metric의 유연성을 가지면서도 Manifold Regularization 덕분에 Overfitting 없이 높은 성능을 유지했다.
3.  **시각화 분석:** MNIST 데이터의 일부를 사용하여 학습된 Metric을 시각화한 결과(Figure 1), PLML이 학습한 Local Metric들이 데이터의 분포를 가장 잘 반영하며 공간적으로 부드럽게 변화함을 확인하였다.
4.  **Basis Metric 수의 영향:** Basis Metric의 수 $m$을 증가시킬수록 성능이 점진적으로 향상되다가 포화(Saturate)되는 양상을 보였으며, $m$이 커져도 Overfitting이 발생하지 않았다. 이는 CBLML이 $m$이 커질 때 성능이 하락하는 것과 대조적이다.

## 🧠 Insights & Discussion

본 논문은 Local Metric Learning의 고질적인 문제인 Overfitting을 **'파라미터화'**와 **'매니폴드 정규화'**라는 두 가지 전략으로 해결하였다. 

특히 주목할 점은 Local Metric을 독립적인 변수가 아닌, 소수의 Basis Metric들의 선형 결합으로 정의함으로써 모델의 복잡도를 획기적으로 낮췄다는 점이다. 여기에 Graph Laplacian을 이용한 정규화를 추가함으로써, 데이터가 존재하는 저차원 매니폴드의 기하학적 구조를 학습 과정에 반영하였다. 이는 단순히 파라미터 수를 줄이는 것을 넘어, 인접한 데이터 포인트들이 유사한 거리 측정 방식을 공유해야 한다는 물리적/기하학적 타당성을 부여한 설계이다.

다만, 앵커 포인트 $U$를 설정할 때 $k$-means 클러스터링의 중심점을 사용하였는데, 만약 데이터의 분포가 매우 복잡하거나 클러스터링 자체가 잘 되지 않는 데이터셋의 경우 앵커 포인트의 배치가 성능에 영향을 미칠 가능성이 있다. 또한, 테스트 데이터의 가중치를 결정할 때 단순히 훈련 데이터의 Nearest Neighbor 가중치를 사용하는 방식을 취했는데, 이 부분에 대한 최적화 여지가 남아 있을 수 있다.

## 📌 TL;DR

-   **주요 기여:** Local Metric을 Basis Metric들의 선형 결합으로 파라미터화하고, Manifold Regularization을 통해 지역적 유연성과 전역적 부드러움을 동시에 확보한 PLML 방법을 제안하였다.
-   **핵심 성과:** 대규모 데이터셋 실험을 통해 기존의 Local Metric 방식들이 겪던 Overfitting 문제를 해결하였으며, Global Metric 및 최적화된 SVM보다 우수한 분류 성능을 입증하였다.
-   **의의:** 본 연구는 데이터 매니폴드의 구조를 활용한 Metric 학습이 고차원 데이터의 거리 측정 최적화에 매우 효과적임을 보여주었으며, 향후 복잡한 구조의 데이터 분류 및 검색 시스템의 성능 향상에 기여할 가능성이 크다.