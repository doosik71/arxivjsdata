# Provably Robust Metric Learning

Lu Wang, Xuanqing Liu, Jinfeng Yi, Yuan Jiang, Cho-Jui Hsieh (2020)

## 🧩 Problem to Solve

본 논문은 분류 및 유사도 검색에서 널리 사용되는 Metric Learning 알고리즘이 적대적 섭동(adversarial perturbations)에 취약하다는 문제를 해결하고자 한다. 기존의 Metric Learning 방법론들은 주로 깨끗한 데이터에 대한 정확도(clean accuracy)를 높이는 데 집중해 왔으며, 이로 인해 학습된 메트릭이 오히려 표준 Euclidean distance보다 적대적 공격에 더 취약해지는 결과가 발생할 수 있다.

특히 $K$-Nearest Neighbor ($K$-NN) 분류기는 입력 공간에서의 작은 변화만으로도 예측 결과가 바뀔 수 있다는 점이 이미 알려져 있다. 하지만 $K$-NN은 불연속적인 계단 함수(discrete step function) 형태를 띠고 있어 그래디언트(gradient)가 존재하지 않으며, 이 때문에 기존 신경망(Neural Networks)에서 사용하던 매끄러운 함수 기반의 적대적 방어 기법을 적용하기 어렵다. 따라서 본 연구의 목표는 적대적 섭동에 대해 강건하며, 그 강건성이 수학적으로 증명 가능한(certifiable) Mahalanobis distance를 학습하는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 각 샘플에 대해 예측을 바꿀 수 있는 '최소 적대적 섭동(minimal adversarial perturbation)'의 크기를 최대화하는 Mahalanobis distance를 학습하는 것이다.

가장 큰 기술적 기여는 계산적으로 다루기 어려운(intractable) $K$-NN의 최소 적대적 섭동에 대해, 미분 가능한 형태의 효율적인 하한선(lower bound)을 도출해낸 것이다. 이를 통해 적대적 강건성을 직접적인 학습 목표(objective function)에 포함시킬 수 있게 되었으며, 결과적으로 깨끗한 데이터에 대한 성능 저하 없이 인증된 강건성(certified robustness)을 확보한 Adversarial Robust Metric Learning (ARML) 알고리즘을 제안하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Metric Learning**: Mahalanobis distance를 학습하는 NCA, LMNN, ITML, LFDA 등 다양한 선형 메트릭 학습 방법이 존재한다. 그러나 이들은 모두 정밀도 향상에만 치중했을 뿐, 적대적 강건성은 고려하지 않았다.
2. **신경망의 적대적 강건성**: 경험적 방어(Empirical defense)와 인증된 방어(Certified defense)로 나뉜다. 인증된 방어는 특정 입력 영역 내에 적대적 예제가 없음을 보장하지만, 이는 모델의 매끄러움(smoothness) 가정에 의존하므로 $K$-NN과 같은 이산 모델에는 적용할 수 없다.
3. **$K$-NN의 강건성**: 기존 연구들은 주로 $K$-NN에 대한 적대적 공격(attack)이나 Euclidean distance 기반의 강건성 검증(verification)에 집중하였다. 일반적인 Mahalanobis distance에 대한 강건성 검증 및 학습 방법은 본 논문 이전까지 연구되지 않았다.

### 기존 방식과의 차별점

ARML은 단순히 공격을 막아내는 경험적 방어를 넘어, 수학적 하한선을 통해 강건성을 보장하는 '인증된 방어'를 Mahalanobis $K$-NN에 최초로 도입하였다. 또한, 신경망의 강건한 학습에서 흔히 나타나는 '깨끗한 정확도'와 '강건한 정확도' 사이의 트레이드오프(trade-off) 문제를 해결하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 방법론은 양의 준정부호(positive semi-definite) 행렬 $M$으로 매개변수화되는 Mahalanobis distance $d_M(x, x') = (x-x')^T M (x-x')$를 학습하여, $K$-NN 분류기의 인증된 강건성 오류(certified robust error)를 최소화하는 것을 목표로 한다. $M$의 양의 준정부호 성질을 보장하기 위해 $M = G^T G$로 정의하여 $G$를 학습한다.

### 핵심 구성 요소 및 수식

**1. 최소 적대적 섭동의 정의**
특정 샘플 $(x, y)$에 대해 예측을 바꾸는 가장 작은 섭동 $\delta$의 크기는 다음과 같이 정의된다.
$$\epsilon^*(x, y) = \min_{\delta} \|\delta\| \quad \text{s.t. } f(x+\delta) \neq y$$

**2. Triplet 문제와 닫힌 형태의 해 (Closed-form solution)**
두 샘플 $x^+, x^-$와 테스트 샘플 $x$가 있을 때, $x$에 가해지는 최소 섭동을 통해 $x$가 $x^+$보다 $x^-$에 더 가깝게 만드는 문제는 다음과 같은 닫힌 형태로 표현된다.
$$\tilde{\epsilon}(x^+, x^-, x; M) = \frac{[d_M(x, x^-) - d_M(x, x^+)]_+}{2\sqrt{(x^+ - x^-)^T M^T M (x^+ - x^-)}}$$
여기서 $[ \cdot ]_+$는 $\max(\cdot, 0)$을 의미한다.

**3. 강건성 검증 하한선 (Theorem 1)**
Mahalanobis $K$-NN의 최소 적대적 섭동 $\epsilon^*$의 하한선은 다음과 같이 계산될 수 있다.
$$\epsilon^*(x_{test}, y_{test}; M) \geq \text{kth min}_{j: y_j \neq y_{test}} \text{kth max}_{i: y_i = y_{test}} \tilde{\epsilon}(x_i, x_j, x_{test}; M)$$
단, $k = (K+1)/2$이다. 이는 각 클래스 내외의 샘플들에 대해 위에서 정의한 $\tilde{\epsilon}$ 값을 계산하여 $k$번째 최소/최대값을 취함으로써 얻어진다.

### 학습 절차

전체 학습 목표는 모든 훈련 데이터에 대해 위에서 도출한 강건성 하한선의 손실(loss)을 최소화하는 것이다.
$$\min_{G} \frac{1}{N} \sum_{i=1}^N \ell(\epsilon^* \text{ lower bound})$$
실제 구현에서는 계산 효율성을 위해 모든 쌍을 계산하는 대신, 각 샘플의 주변(neighborhood)에서 양성 및 음성 샘플을 무작위로 샘플링하는 $\text{randnear}^+_M$ 및 $\text{randnear}^-_M$ 절차를 사용하여 근사적으로 최적화한다. $G$는 Adam optimizer를 통해 업데이트된다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, Fashion-MNIST, Splice, Pendigits, Satimage, USPS (총 6개).
- **비교 대상**: Euclidean distance, NCA, LMNN, ITML, LFDA.
- **평가 지표**: Clean error, Certified robust error ($\text{cre}$), Empirical robust error ($\text{ere}$).
- **설정**: $K=11$ (K-NN의 경우), $M$은 정방 행렬.

### 주요 결과

1. **강건성 향상**: 모든 데이터셋에서 ARML은 다른 모든 메트릭 학습 방법 및 Euclidean distance보다 훨씬 낮은 인증된 강건성 오류와 경험적 강건성 오류를 기록하였다.
2. **트레이드오프 부재**: NCA나 LMNN은 깨끗한 정확도를 높였지만 강건성이 Euclidean보다 떨어지는 경향을 보였다. 반면, ARML은 깨끗한 정확도 면에서 NCA/LMNN과 경쟁 가능한 수준을 유지하면서도 강건성을 획기적으로 높였다.
3. **신경망과의 비교**: Randomized Smoothing을 적용한 신경망의 경우 강건성을 높이면 깨끗한 정확도가 떨어지는 트레이드오프가 뚜렷하게 나타났으나, ARML은 이러한 현상 없이 두 성능을 모두 확보하였다.
4. **계산 효율성**: GPU 구현 시 평균 런타임이 약 10.6초(USPS 기준)로 매우 효율적이며, 기존의 LMNN이나 ITML과 비교해도 오버헤드가 크지 않다.

## 🧠 Insights & Discussion

본 논문은 Metric Learning 분야에서 '강건성'이라는 관점을 정교하게 도입하였다. 특히, 기존의 메트릭 학습 알고리즘들이 단순히 데이터 간의 거리를 최적화하는 과정에서 결정 경계(decision boundary)를 불안정하게 만들어, 오히려 단순한 Euclidean distance보다 적대적 공격에 취약해질 수 있음을 정량적으로 보여준 점이 인상적이다.

강점으로는 $K$-NN이라는 비미분 가능 모델에 대해 미분 가능한 강건성 하한선을 유도하여, 이를 통해 직접적으로 모델을 최적화할 수 있는 프레임워크를 구축했다는 점이다. 또한, 이론적 보장(Certification)과 실제 공격에 대한 방어(Empirical) 모두에서 우수한 성적을 거두었다.

다만, 본 연구는 선형적인 Mahalanobis distance에 국한되어 있다. 딥 메트릭 학습(Deep Metric Learning)이나 커널 기반의 비선형 메트릭 학습으로 확장했을 때도 동일한 방식의 인증 가능한 강건성을 확보할 수 있을지는 여전히 미해결 과제로 남아 있다.

## 📌 TL;DR

본 논문은 $K$-NN 분류기를 위한 최초의 인증 가능한 강건한 메트릭 학습 방법인 **ARML**을 제안한다. 최소 적대적 섭동의 미분 가능한 하한선을 도출하여 이를 학습 목표로 설정함으로써, 깨끗한 데이터에 대한 성능 저하 없이 적대적 공격에 대해 수학적으로 증명 가능한 강건성을 확보하였다. 이 연구는 고도의 보안이 필요한 유사도 검색 및 분류 시스템에서 신뢰할 수 있는 거리 측정 방식을 설계하는 데 중요한 기초가 될 것으로 보인다.
