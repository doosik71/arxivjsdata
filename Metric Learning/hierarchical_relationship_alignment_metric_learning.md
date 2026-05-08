# Hierarchical Relationship Alignment Metric Learning

Lifeng Gu (2021)

## 🧩 Problem to Solve

기존의 Metric Learning 방법론들은 주로 샘플 쌍(sample pair)이 '유사한가(similar)' 또는 '유사하지 않은가(dissimilar)'라는 이분법적인 관계에 의존하여 유사도나 거리 측정 방식을 학습한다. 그러나 실제 많은 응용 분야, 특히 Multi-label Learning이나 Label Distribution Learning과 같은 작업에서는 샘플 간의 관계를 단순히 유사함과 유사하지 않음으로 정의하기 어렵다.

이러한 문제를 해결하기 위해 Relation Alignment Metric Learning (RAML) 프레임워크가 제안되었으나, RAML은 선형 메트릭(linear metric)을 학습하기 때문에 복잡한 데이터셋의 비선형적인 특성을 모델링하는 데 한계가 있다. 따라서 본 논문의 목표는 딥러닝의 강력한 표현 학습 능력과 RAML의 관계 정렬(relationship alignment) 개념을 결합하여, 다양한 학습 작업에서 특성 공간(feature space)과 레이블 공간(label space) 간의 일관성을 극대화하는 계층적 관계 정렬 메트릭 학습 모델인 HRAML을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **관계 정렬(Relationship Alignment)** 개념을 딥러닝 아키텍처에 통합하는 것이다. 특성 공간에서 계산된 샘플 쌍의 관계(거리)가 레이블 공간에서 정의된 샘플 쌍의 관계와 일치하도록 강제함으로써, 단순한 분류를 넘어 레이블 간의 상대적 관계까지 학습할 수 있도록 설계하였다. 특히, 선형 변환에 그쳤던 기존 RAML과 달리 다층 퍼셉트론(MLP) 구조를 도입하여 복잡한 데이터의 고차원적 특징을 추출하고 이를 정렬함으로써 모델의 표현력을 크게 향상시켰다.

## 📎 Related Works

논문에서는 Contrastive Loss, Center Loss, Hierarchical Triplet Loss, Angle Loss, N-pair Loss, Circle Loss 등 다양한 Deep Metric Learning 기법들을 언급한다. 이러한 기존 방식들은 주로 다음과 같은 한계를 가진다:

- **이분법적 제약**: 대부분의 손실 함수가 샘플 쌍을 단순히 긍정(positive) 또는 부정(negative)으로 나누어 처리하며, 이는 Multi-label이나 Label Distribution과 같은 연속적 혹은 다중적 관계를 표현하기에 부족하다.
- **샘플링 효율성**: 일부 방식은 학습 과정에서 효율적인 네거티브 샘플 선택이 매우 중요하며, 이를 잘못 처리할 경우 수렴 속도가 느려지거나 성능이 저하된다.

HRAML은 이러한 기존의 '쌍 기반 제약(pairwise constraints)' 방식에서 벗어나, 레이블 공간의 관계 자체를 타겟으로 삼아 이를 특성 공간에 투영하는 '관계 정렬' 방식을 채택함으로써 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

HRAML은 입력 데이터를 고차원 특성 공간으로 매핑하는 딥러닝 인코더와, 매핑된 특성 간의 거리를 레이블 공간의 거리와 일치시키는 정렬 메커니즘으로 구성된다. 인코더로는 일반적인 MLP(Multi-Layer Perceptron)를 사용하며, 활성화 함수로는 $\tanh$를 채택하였다. 네트워크의 최종 출력 $f(x)$는 다음과 같이 정규화되어 성능을 최적화한다.
$$f(x) = \frac{f(x)}{\|f(x)\|}$$

### 2. 관계 정렬 및 손실 함수

특성 공간에서의 샘플 쌍 $(x_i, x_j)$의 관계를 $R(f(x_i), f(x_j))$로, 레이블 공간에서의 관계를 $g(y_i, y_j)$로 정의한다. 본 논문에서는 특성 공간의 관계를 유클리드 거리의 제곱으로 정의하고, 레이블 공간의 관계는 $L_1$ norm으로 정의한다.

- **특성 공간 거리**: $D_{ij} = \|f(x_i) - f(x_j)\|^2$
- **레이블 공간 거리**: $g(y_i, y_j) = \|y_i - y_j\|_1$

이 두 거리의 차이를 최소화하기 위해 Mean Square Error (MSE) 기반의 손실 함수를 사용하며, 모델의 과적합을 방지하기 위한 정규화 항 $r(\theta)$를 추가한다. 최종 목적 함수는 다음과 같다.
$$\min J = \frac{1}{4} \sum_{(i,j)} (D_{ij}^2 - g(y_i, y_j))^2 + r(\theta)$$
여기서 $r(\theta) = \lambda \sum_{m=1}^{M} (\|W^{(m)}\|_F^2 + \|b^{(m)}\|_2^2)$ 이며, $W$와 $b$는 각각 가중치와 편향을 의미한다.

### 3. 학습 절차 및 최적화

학습은 확률적 경사 하강법(SGD)을 통해 수행된다. 각 레이어 $m$에 대한 가중치 $W^{(m)}$와 편향 $b^{(m)}$의 그래디언트는 체인 룰(chain rule)을 통해 계산되며, 논문에서 제시된 방정식 (6)~(9)에 따라 역전파(back-propagation)가 이루어진다. 특히, 학습 효율을 높이기 위해 Hard Example Mining 기법을 도입하여 학습이 어려운 샘플 쌍을 우선적으로 선택해 학습에 활용한다.

## 📊 Results

### 1. 실험 설정

HRAML의 성능을 검증하기 위해 세 가지 서로 다른 태스크에서 실험을 진행하였다:

- **단일 레이블 분류 (Single-label Classification)**: binalpha, caltech101, Mnist, Mpeg7, news20, TDT2, uspst 데이터셋 사용.
- **다중 레이블 학습 (Multi-label Learning)**: emotion, flags, corel800 데이터셋 사용.
- **레이블 분포 학습 (Label Distribution Learning)**: Nature Scene 데이터셋 사용.

비교 대상으로는 ITML, LMNN, DML, DSVM, GMML과 같은 전통적인 메트릭 학습 방법론과, 선형 관계 정렬 방식인 RAML(SVR, KRR 버전)이 포함되었다.

### 2. 주요 결과

- **분류 작업**: Table 1에 따르면, HRAML은 대부분의 데이터셋에서 가장 높은 정확도를 기록하였다. 특히 선형 모델인 RAML이나 커널 기반 모델보다 우수한 성능을 보이며, 복잡한 데이터셋에서 딥러닝 기반 인코더의 효과를 입증하였다.
- **다중 레이블 학습**: Hamming Loss, Ranking Loss, Average Precision 등의 지표를 통해 평가한 결과, HRAML이 RAML보다 우수한 성능을 보였다. 이는 인코더가 다양한 레이블 간의 복잡한 관계를 더 잘 추출했음을 의미한다.
- **레이블 분포 학습**: Nature Scene 데이터셋에서 Chebyshev, Cosine 등 다양한 거리 지표를 통해 평가한 결과, HRAML이 RAML 및 AAKNN보다 뛰어난 성능을 보였다.

## 🧠 Insights & Discussion

HRAML은 기존의 단순한 유사도 학습 방식을 넘어, 레이블 공간의 기하학적 구조를 특성 공간으로 직접 전이(transfer)시키는 접근 방식을 취함으로써 범용적인 메트릭 학습 프레임워크를 제시하였다.

**강점:**

- **범용성**: 관계 함수 $g(y_i, y_j)$를 어떻게 정의하느냐에 따라 단일 분류, 다중 레이블, 분포 학습 등 다양한 작업에 유연하게 적용 가능하다.
- **표현력**: MLP 구조를 도입함으로써 기존 RAML의 선형적 한계를 극복하고 비선형적인 데이터 관계를 성공적으로 모델링하였다.

**한계 및 논의사항:**

- **관계 함수의 고정**: 본 논문에서는 레이블 공간의 거리 함수로 $L_1$ norm을 고정하여 사용하였다. 저자가 언급했듯이 KL-divergence와 같은 더 정교한 분포 측정 함수를 사용한다면 성능이 더욱 향상될 가능성이 있다.
- **계산 복잡도**: 샘플 쌍의 조합이 $O(n^2)$으로 증가하므로, 대규모 데이터셋에서 Hard Example Mining 외에 더 효율적인 샘플링 전략이 필요할 것으로 보인다.
- **아키텍처의 단순함**: 일반적인 MLP를 사용하였으나, 데이터의 특성에 따라 CNN이나 Transformer 기반의 인코더를 적용했을 때의 결과에 대해서는 명시되지 않았다.

## 📌 TL;DR

본 논문은 샘플 간의 관계를 단순한 '유사/비유사'로 나누지 않고, 레이블 공간의 거리를 특성 공간의 거리와 정렬시키는 **HRAML(Hierarchical Relationship Alignment Metric Learning)** 모델을 제안한다. 딥러닝 기반 인코더를 통해 비선형 표현력을 확보함으로써 단일 분류, 다중 레이블 학습, 레이블 분포 학습 등 다양한 작업에서 기존의 선형/커널 기반 메트릭 학습 방법 및 RAML보다 우수한 성능을 입증하였다. 이 연구는 복잡한 레이블 관계를 가진 데이터셋에서 효과적인 임베딩 공간을 학습하는 데 중요한 방법론을 제공한다.
