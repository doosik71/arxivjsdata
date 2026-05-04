# Metric Learning from Imbalanced Data

Léo Gautheron, Emilie Morvant, Amaury Habrard and Marc Sebban (2019)

## 🧩 Problem to Solve

본 논문은 데이터셋 내 클래스 불균형(class imbalance)이 존재할 때, 거리 측정 함수를 학습하는 Metric Learning 알고리즘이 다수 클래스(majority class)에 편향되는 문제를 해결하고자 한다. 

일반적인 Metric Learning 알고리즘은 같은 라벨을 가진 샘플 간의 거리는 좁히고, 서로 다른 라벨을 가진 샘플 간의 거리는 멀게 하는 손실 함수를 최적화한다. 그러나 다수 클래스의 샘플 수가 압도적으로 많은 불균형 시나리오에서는, 손실 함수가 다수 클래스 간의 제약 조건(constraints)에 의해 지배된다. 결과적으로 모델은 모든 샘플을 다수 클래스로 분류하려는 경향을 보이며, 이는 단순 정확도(Accuracy)는 높게 나타날 수 있으나 실제로 중요한 소수 클래스(minority class)를 제대로 식별하지 못하는 결과를 초래한다.

따라서 본 연구의 목표는 클래스 불균형 상황에서도 소수 클래스의 특성을 효과적으로 반영하여 강건한 거리 측정 함수를 학습할 수 있는 새로운 Mahalanobis Metric Learning 알고리즘인 IML(Imbalanced Metric Learning)을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **쌍별 제약 조건(pairwise constraints)의 세분화된 분해 및 재가중치 부여(re-weighting)**이다.

기존 방식은 단순히 '유사한 쌍(similar pairs)'과 '유사하지 않은 쌍(dissimilar pairs)'으로만 나누어 처리했지만, IML은 이를 더 세분화하여 어떤 클래스의 샘플들이 쌍을 이루고 있는지를 고려한다. 구체적으로는 유사/비유사 쌍을 각각 소수 클래스 포함 여부에 따라 4개의 그룹으로 나누고, 각 그룹의 크기에 반비례하는 가중치를 부여함으로써 데이터의 양과 관계없이 각 제약 조건이 최적화 과정에서 동일한 영향력을 가지도록 설계하였다.

## 📎 Related Works

논문에서는 Mahalanobis 거리 기반의 대표적인 알고리즘들로 LMNN(Large Margin Nearest Neighbor), ITML(Information-Theoretic Metric Learning), GMML(Geometric Mean Metric Learning)을 언급한다. 이들은 주로 잠재 공간(latent space)에서 $k\text{NN}$ 분류기의 정확도를 높이는 것을 목표로 한다. 하지만 이러한 기존 방식들은 제약 조건에 가중치를 두더라도 클래스 라벨 자체를 고려하지 않으므로 불균형 데이터에 취약하다는 한계가 있다.

또한, 불균형 데이터를 처리하기 위한 일반적인 접근법으로 오버샘플링/언더샘플링(SMOTE 등), 비용 민감 학습(cost-sensitive methods), 앙상블 방법 등이 존재한다. 그러나 이러한 방법들은 과적합(overfitting)이나 과소적합(underfitting)의 위험이 있으며, 특히 매우 심한 불균형 상황에서는 충분한 다양성을 생성하지 못하는 한계가 있다.

## 🛠️ Methodology

### 1. Mahalanobis Distance
본 연구는 양의 준정부호(positive semidefinite, PSD) 행렬 $M$으로 파라미터화되는 Mahalanobis 거리를 학습한다.
$$d_M(x, x') = \sqrt{(x-x')^T M (x-x')}$$
이는 $M=L^T L$로 분해될 수 있으며, 데이터 $x$를 $L$을 통해 투영한 후 유클리드 거리를 계산하는 것과 동일하다.

### 2. Loss Functions
IML은 두 가지 형태의 Hinge loss를 사용한다.
- **유사도 제약 ($\ell_1$):** 같은 클래스인 경우 거리를 1보다 작게 유지한다.
  $$\ell_1(M, z, z') = [d_M^2(x, x') - 1]^+$$
- **비유사도 제약 ($\ell_2$):** 다른 클래스인 경우 거리를 $1+m$ (여기서 $m$은 마진)보다 크게 유지한다.
  $$\ell_2(M, z, z') = [1 + m - d_M^2(x, x')]^+$$
여기서 $[a]^+ = \max(0, a)$이다.

### 3. IML Objective Function
IML은 전체 손실 함수를 다음과 같이 정의하여 최적화한다.
$$\min_{M \succeq 0} F(M) = \frac{a}{4|\text{Sim}^+|} \sum_{(z,z') \in \text{Sim}^+} \ell_1 + \frac{a}{4|\text{Sim}^-|} \sum_{(z,z') \in \text{Sim}^-} \ell_1 + \frac{(1-a)}{4|\text{Dis}^+|} \sum_{(z,z') \in \text{Dis}^+} \ell_2 + \frac{(1-a)}{4|\text{Dis}^-|} \sum_{(z,z') \in \text{Dis}^-} \ell_2 + \lambda \|M-I\|_F^2$$

여기서 각 쌍의 집합은 다음과 같이 정의된다.
- $\text{Sim}^+ \subseteq S^+ \times S^+$: 소수 클래스 간의 유사 쌍
- $\text{Sim}^- \subseteq S^- \times S^-$: 다수 클래스 간의 유사 쌍
- $\text{Dis}^+ \subseteq S^+ \times S^-$: 소수-다수 클래스 간의 비유사 쌍
- $\text{Dis}^- \subseteq S^- \times S^+$: 다수-소수 클래스 간의 비유사 쌍

**핵심 메커니즘:** 각 항에 $\frac{1}{4|\text{set}|}$ 가중치를 부여함으로써, 샘플 수의 차이로 인해 발생하는 손실 값의 불균형을 상쇄하고 네 가지 제약 조건이 동일한 중요도로 학습되게 한다. 또한 $\|M-I\|_F^2$ (Frobenius norm)를 통해 학습된 메트릭이 유클리드 거리에서 너무 멀어지지 않도록 규제한다.

### 4. 학습 및 추론 절차
- **쌍 선택(Pair Selection):** 무작위 선택보다 $k\text{NN}$ 기반의 이웃 선택 방식이 불균형 시나리오에서 더 효과적임을 확인하고 이를 채택하였다.
- **최적화:** $M$을 직접 학습하는 대신 $M=L^T L$ 관계를 이용하여 투영 행렬 $L$을 학습함으로써 PSD 제약 조건을 효율적으로 만족시켰으며, L-BFGS-B 알고리즘을 통해 최적화하였다.
- **분류:** 학습된 메트릭을 적용한 후 $3\text{NN}$ 분류기를 사용하여 최종 분류를 수행한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** UCI 및 Keel 저장소의 22개 이진 분류 데이터셋을 사용하였다.
- **비교 대상:** 유클리드 거리, GMML, ITML, LMNN.
- **평가 지표:** 불균형 데이터에 적합한 F1-measure를 주 지표로 사용하였다.

### 2. 주요 결과
- **정량적 성능:** 전반적인 평균 F1-measure에서 IML(72.1%)이 LMNN(70.8%), ITML(70.1%), GMML(69.3%)보다 우수한 성능을 보였으며, 평균 순위(Average Rank) 또한 1.48로 가장 낮아(가장 좋음) 우수함을 입증하였다.
- **데이터 전처리 결합:** SMOTE 오버샘플링을 IML과 함께 사용했을 때 성능이 더욱 향상되었으며, 이는 SMOTE의 데이터 재균형 기능과 IML의 표현 학습 기능이 상호 보완적임을 시사한다. 반면, Random Under Sampling(RUS)은 큰 이득이 없었다.
- **불균형도 변화 실험:** 소수 클래스의 비율을 50%에서 1%까지 인위적으로 낮추었을 때, 모든 알고리즘의 F1-measure가 하락하였으나 IML의 하락 폭이 가장 작아 불균형에 대해 가장 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
IML의 성공 요인은 단순히 손실 함수를 정의한 것에 그치지 않고, 데이터의 라벨 분포에 따라 손실 기여도를 정규화(normalization)했다는 점에 있다. 특히, $k\text{NN}$ 기반의 쌍 선택 전략이 소수 클래스의 지역적 구조를 보존하는 데 중요한 역할을 했음을 분석을 통해 확인하였다. 이는 불균형 데이터 학습에서 단순한 샘플링 기법뿐만 아니라, 학습 목적 함수 자체를 라벨 인식적으로(label-aware) 설계하는 것이 중요함을 보여준다.

### 한계 및 미해결 질문
본 연구에서 제안한 IML은 Mahalanobis 거리라는 선형 메트릭에 국한되어 있다. 따라서 데이터의 분포가 매우 복잡하거나 비선형적인 구조를 가질 경우 성능에 한계가 있을 수 있다. 또한, 현재의 최적화 방식은 계산 비용이 발생하므로, GMML과 같이 닫힌 형태의 해(closed-form solution)를 도출하여 연산 속도를 높이는 연구가 필요하다.

## 📌 TL;DR

본 논문은 클래스 불균형 상황에서 다수 클래스에 편향되는 Metric Learning의 문제를 해결하기 위해, 유사/비유사 쌍을 클래스 라벨별로 4개 그룹으로 분해하고 각 그룹의 크기에 반비례하는 가중치를 부여하는 **IML(Imbalanced Metric Learning)** 알고리즘을 제안한다. 실험 결과, IML은 기존의 LMNN, ITML, GMML보다 F1-measure 측면에서 우수한 성능을 보였으며, 특히 클래스 불균형이 심해질수록 타 알고리즘 대비 강건한 성능 유지 능력을 보여주었다. 이는 향후 불균형 데이터셋을 활용한 표현 학습 및 이상치 탐지 연구에 중요한 기초가 될 수 있다.