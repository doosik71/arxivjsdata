# Distance Metric Learning for Kernel Machines

Zhixiang (Eddie) Xuxu, Kilian Q. Weinberger, Olivier Chapelle (2013)

## 🧩 Problem to Solve

본 논문은 Support Vector Machine (SVM), 특히 Radial Basis Function (RBF) 커널을 사용하는 SVM의 성능을 향상시키기 위한 거리 측정법(Distance Metric) 학습 문제를 다룬다.

대부분의 머신러닝 알고리즘은 데이터 간의 유사성을 측정하기 위해 거리 함수를 사용하며, 전통적으로는 Euclidean distance가 널리 쓰인다. 그러나 Euclidean distance는 데이터의 의미적 구조나 클래스 레이블 정보를 전혀 고려하지 않는 'uninformed norm'이기에 최적이 아닐 가능성이 높다.

기존의 Mahalanobis metric learning 알고리즘들은 주로 k-Nearest Neighbor (kNN) 분류기의 성능을 높이는 데 집중되어 왔다. 하지만 이러한 알고리즘들을 SVM-RBF의 전처리 단계로 적용했을 때, 실제로 유의미한 성능 향상이 나타나는지에 대해서는 의문이 제기되었다. 따라서 본 논문의 목표는 기존 kNN 기반 metric learning의 한계를 분석하고, SVM의 손실 함수를 직접 최적화 과정에 통합하여 최적의 거리 측정법을 학습하는 새로운 알고리즘인 Support Vector Metric Learning (SVML)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **kNN 기반 Metric Learning의 한계 증명**: NCA, LMNN, ITML과 같은 대표적인 Mahalanobis metric learning 알고리즘들이 SVM-RBF의 전처리로서 통계적으로 유의미한 개선을 이루지 못함을 실험적으로 입증하였다.
2. **Support Vector Metric Learning (SVML) 제안**: 거리 측정 행렬 $L$의 학습과 SVM 파라미터 학습을 단일 최적화 프레임워크로 통합하였다. 특히, 검증 데이터셋(validation set)의 분류 오류를 최소화하는 방향으로 metric을 학습함으로써 모델 선택(model selection) 과정을 자동화하였다.
3. **유연한 구조적 제약 제공**: 학습하려는 행렬 $L$의 형태를 구형(Sphere), 대각(Diagonal), 또는 직사각형(Rectangular)으로 제한함으로써, 단순한 하이퍼파라미터 튜닝부터 특징 재가중치 부여(feature re-weighting), 그리고 저차원 투영을 통한 시각화까지 가능하게 하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존의 Mahalanobis metric learning 기법들을 소개한다.

- **Neighborhood Component Analysis (NCA)**: 확률적 이웃 할당 방식을 통해 leave-one-out 분류 오류의 기댓값을 최소화하며, gradient ascent를 사용하여 학습한다. 다만, 계산 복잡도가 높고 목적 함수가 non-convex하다는 단점이 있다.
- **Large Margin Nearest Neighbor (LMNN)**: target neighbors를 설정하여 유사한 레이블의 데이터는 가깝게, 다른 레이블의 데이터는 큰 마진을 두고 멀리 밀어내는 방식이다. Semi-definite programming (SDP)을 통해 convex loss를 최소화한다.
- **Information-Theoretic Metric Learning (ITML)**: 가우시안 교차 엔트로피(Gaussian cross entropy)를 사용하여 초기 metric과의 거리를 정규화하며, 유사도와 비유사도에 대한 제약 조건을 부과하여 학습한다.

**기존 방식과의 차별점**: 위 알고리즘들은 kNN의 leave-one-out 오류를 줄이는 데 특화되어 있다. 반면 SVML은 SVM의 결정 경계와 손실 함수를 직접 고려한다. SVM-RBF는 kNN과 유사하게 지역적 특성을 활용하지만,- weight $\alpha_j$가 부여된 'soft' nearest neighbor rule로 동작하므로, kNN을 위해 설계된 metric이 SVM에 그대로 적용되지 않는다는 점을 간과해서는 안 된다.

## 🛠️ Methodology

### 전체 시스템 구조

SVML은 Mahalanobis 거리를 RBF 커널에 통합하여, 거리 측정 행렬 $L$을 직접 학습한다. Mahalanobis 거리는 다음과 같이 정의된다.
$$d_M(x_i, x_j) = \sqrt{(x_i-x_j)^T M (x_i-x_j)}$$
여기서 $M = L^T L$이며, $L \in \mathbb{R}^{r \times d}$이다. 이를 RBF 커널에 적용하면 다음과 같은 커널 행렬 $K$를 얻는다.
$$k_L(x_i, x_j) = e^{-(x_i-x_j)^T L^T L (x_i-x_j)}$$

### 손실 함수 (Loss Function)

단순한 $0/1$ loss는 미분이 불가능하므로, SVML은 검증 데이터셋 $V$에 대해 다음과 같은 부드러운 근사 손실 함수 $L_V$를 정의한다.
$$L_V(L) = \frac{1}{|V|} \sum_{(x,y) \in V} s_a(y h(x))$$
여기서 $h(x)$는 SVM의 예측 함수이며, $s_a(z) = \frac{1}{1 + e^{az}}$는 mirror-sigmoid 함수로 $0/1$ loss를 미분 가능하게 근사한 것이다. 파라미터 $a$는 곡선의 가파른 정도를 조절한다.

### 학습 절차 및 최적화

SVML은 Gradient Descent 또는 2차 최적화 방법을 사용하여 $L$을 업데이트한다. 핵심은 체인 룰(chain rule)을 통해 $\frac{\partial L_V}{\partial L}$을 계산하는 것이다.

1. **그래디언트 계산**: $h(x)$는 SVM 파라미터 $\alpha$와 $b$에 의존하며, 이들은 다시 커널 행렬 $K$(즉, $L$)에 의존한다.
   $$\frac{\partial h}{\partial L} = \frac{\partial h}{\partial \alpha} \frac{\partial \alpha}{\partial L} + \frac{\partial h}{\partial K} \frac{\partial K}{\partial L} + \frac{\partial h}{\partial b} \frac{\partial b}{\partial L}$$
2. **폐형식(Closed-form) 유도**: Support vector들에 대해 $y_i (\sum K_{ij} \alpha_j y_j + b) = 1$이라는 조건을 이용하여 $(\alpha, b)$를 $L$에 대한 함수로 표현하고, 행렬 역함수 미분 법칙을 통해 $\frac{\partial (\alpha, b)}{\partial L}$을 계산한다.
3. **정규화**: 과적합을 방지하기 위해 초기 행렬 $L_0$와의 Frobenius norm 차이를 정규화 항으로 추가한다.
   $$L_V(L) = \frac{1}{|V|} \sum_{(x,y) \in V} s_a(y h(x)) + \lambda \|L - L_0\|_F^2$$
4. **조기 종료(Early Stopping)**: 별도의 hold-out set을 사용하여 검증 성능이 더 이상 개선되지 않을 때 학습을 멈춘다.

## 📊 Results

### 실험 설정

- **데이터셋**: UCI Machine Learning Repository의 9개 데이터셋 (Haber, Credit, ACredit, Trans, Diabts, Mammo, CMC, Page, Gamma) 사용.
- **비교 대상**: Euclidean distance (1, 3, 5-fold CV), ITML+SVM, NCA+SVM, LMNN+SVM.
- **지표**: 분류 오류율(Error Rate) 및 학습 시간.

### 주요 결과

1. **정량적 성능**: SVML은 9개 데이터셋 중 6개에서 최상의 성능(또는 통계적으로 동등한 성능)을 보였다. 특히, kNN 기반 metric learning 알고리즘들이 Euclidean distance보다 눈에 띄게 좋지 않은 결과를 보인 것과 달리, SVML은 일관되게 Euclidean distance보다 우수한 성능을 나타냈다.
2. **계산 효율성**: SVML의 학습 시간은 Euclidean distance를 사용한 3~5-fold cross validation 시간과 유사한 수준으로, 실용적인 계산 비용을 가진다.
3. **시각화 및 차원 축소**: $L$을 직사각형 행렬($r=2$ 또는 $3$)로 제한했을 때, PCA보다 훨씬 더 해석 가능하고 명확한 결정 경계를 가진 2D 시각화 결과를 얻었다.

## 🧠 Insights & Discussion

본 논문은 단순히 새로운 알고리즘을 제안한 것을 넘어, 왜 기존의 metric learning이 SVM에 작동하지 않았는지를 분석했다는 점에서 가치가 있다. kNN은 'Hard'한 이웃 선택 방식을 취하는 반면, SVM-RBF는 가중치 $\alpha$를 통한 'Soft'한 이웃 접근 방식을 취하기 때문에, 최적화 목표(Objective)가 일치해야 함을 시사한다.

SVML의 강점은 하이퍼파라미터 $\sigma$나 $C$를 그리드 서치(grid search)로 찾는 대신, 거리 행렬 $L$ 자체를 학습 가능한 파라미터로 취급하여 검증 오차를 직접 최소화한다는 점이다. 이는 모델 선택 과정을 최적화 문제로 변환하여 자동화한 효율적인 접근법이다.

다만, 본 연구는 이진 분류(binary classification)에 집중하고 있으며, 대규모 데이터셋에 대한 확장성 문제는 SVM 솔버의 속도에 의존하고 있다는 한계가 있다. 하지만 저자들은 그래디언트 계산 과정의 상당 부분이 병렬화 가능함을 언급하며 향후 개선 가능성을 제시하였다.

## 📌 TL;DR

본 논문은 kNN용 metric learning 기법들이 SVM-RBF 성능 향상에 기여하지 못한다는 점을 밝히고, SVM의 손실 함수를 직접 최적화하여 Mahalanobis metric을 학습하는 **Support Vector Metric Learning (SVML)**을 제안한다. SVML은 기존 기법 및 단순 Euclidean distance보다 높은 정확도를 보이며, 계산 시간은 CV 기반 튜닝과 비슷하거나 더 빠르다. 또한, 학습된 metric을 통해 고차원 데이터를 효과적으로 저차원 시각화할 수 있음을 보여주었다. 이 연구는 SVM의 커널 파라미터 최적화를 자동화하고 성능을 극대화하는 실용적인 방법론을 제시한다.
