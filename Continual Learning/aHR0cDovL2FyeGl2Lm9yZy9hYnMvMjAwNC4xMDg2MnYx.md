# Continual Learning of Object Instances

Kishan Parshotam and Mert Kilickaya (2020)

## 🧩 Problem to Solve

본 논문은 동일한 객체 카테고리 내에서 서로 다른 개체(instance)를 구분해야 하는 **Continual Instance Learning (CIL)** 문제를 해결하고자 한다. 일반적인 Continual Learning (CL) 연구들이 새로운 '클래스'(예: 의자, 자동차, 자전거)를 학습하는 것에 집중했다면, 본 연구는 '자동차'라는 동일 클래스 내의 서로 다른 개별 차량들을 지속적으로 구분하는 능력을 학습하는 것에 초점을 맞춘다.

이러한 연구가 중요한 이유는 현실 세계의 데이터 흐름이 동적이기 때문이다. 예를 들어, 온라인 자동차 판매 회사는 매일 새로운 차량 광고를 접수하는데, 개인정보 보호 문제로 인해 과거의 데이터를 삭제해야 하므로 모델을 처음부터 다시 학습시키는 것이 불가능할 수 있다. 또한, 도시 감시 시스템에서는 차량 재식별(Vehicle Re-Identification, ReID)을 위해 계속해서 증가하는 차량 데이터를 학습해야 하며, 매번 전체 데이터를 사용해 학습하는 것은 매우 비효율적이다. 따라서 본 논문의 목표는 이전 데이터에 대한 성능을 유지하면서 새로운 개체를 효율적으로 학습하는 CIL 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **CIL 문제 정의 및 평가**: 기존의 Continual Learning 방법론들을 Metric Learning 설정에 적용하여 평가함으로써, 개체 수준의 지속 학습에서도 **Catastrophic Forgetting**(치명적 망각)이 뚜렷하게 나타남을 입증하였다.
2. **Normalised Cross-Entropy (NCE) 도입**: Regression 기반의 손실 함수(예: Triplet Loss)가 Outlier에 민감하여 망각이 가속화되는 문제를 해결하기 위해, NCE를 통해 Metric Learning을 분류 문제 형태로 변환하여 학습의 안정성을 높였다.
3. **Synthetic Visual Data Transfer 제안**: 실제 데이터 학습 전, 합성 데이터(Synthetic Data)를 통해 객체의 세부적인 시각적 특징을 먼저 학습하게 함으로써, 데이터 분포가 불균형한 지속 학습 환경에서도 망각을 줄이는 전이 학습 방법을 제안하였다.

## 📎 Related Works

### 1. Continual Learning (CL)

기존의 CL 연구들은 새로운 클래스가 추가될 때 이전 클래스의 성능이 급격히 떨어지는 Catastrophic Forgetting을 막기 위해 다양한 정규화(Regularisation) 기법을 사용해 왔다. 대표적으로 **LFL (Less Forgetting Learning)**과 **LwF (Learning without Forgetting)**는 이전 모델의 출력(decision boundary)을 유지하려 하며, **EWC (Elastic Weight Consolidation)**는 파라미터별 중요도를 계산하여 중요한 가중치의 변화를 억제한다. 하지만 이들은 주로 분류(Classification) 작업에 최적화되어 있어, 임베딩 공간을 직접 학습하는 Metric Learning에 그대로 적용하기 어렵다.

### 2. Instance Learning & Metric Learning

개체 학습은 주로 **Metric Learning**을 통해 수행되며, Siamese Network와 **Triplet Loss**를 사용하여 동일 개체는 가깝게, 다른 개체는 멀게 배치하는 매니폴드(Manifold)를 학습한다. 그러나 Triplet Loss는 그래디언트의 범위가 제한되지 않아 Outlier에 매우 취약하며, 이는 지속 학습 환경에서 모델의 불안정성을 초래한다.

### 3. Synthetic Data Transfer

합성 데이터를 이용한 전이 학습은 레이블링 비용을 줄이고 모델의 일반화 성능을 높이는 데 효과적이라고 알려져 있다. 본 논문은 이를 CIL에 도입하여, 실제 데이터의 스트림을 학습하기 전에 견고한 초기 특징 추출기를 구축하는 도구로 활용한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 학습 절차

본 논문은 데이터를 **Continuous Incremental Batches** 형태로 제공받는 시나리오를 가정한다. 즉, 한 번에 소수의 새로운 개체들만 학습할 수 있으며, 이전 데이터에는 접근할 수 없는 제약 조건 하에서 학습이 진행된다.

### 2. 기존 CL 방법론의 CIL 적용 (Benchmarking)

기존의 분류 기반 CL 알고리즘들을 Metric Learning 설정으로 변경하여 성능을 측정하였다.

- **Nave**: 이전 가중치 $\theta_o$에서 시작하여 새로운 데이터 $(X_n, Y_n)$으로 단순히 재학습한다.
- **Fine-Tuning (FT)**: 판별 레이어(discriminant layers)만을 미세 조정한다.
- **LFL**: $\theta_n = \theta_o$로 초기화하고 Softmax 레이어를 동결하며, 다음과 같은 손실 함수를 사용한다.
  $$L_{LFL}(x_n; \theta_o; \theta_n) = \lambda_c L_c(x_n; \theta_n) + \lambda_e L_e(x_n; \theta_o, \theta_n)$$
  여기서 $L_c$는 분류 손실(본 논문에서는 Triplet Loss로 대체)이며, $L_e$는 이전 모델과 현재 모델의 예측값 사이의 유클리드 거리 손실이다.
- **EWC**: 파라미터의 중요도를 나타내는 Fisher Information Matrix $F_i$를 사용하여 가중치 변화를 정규화한다.
  $$L_{EWC} = L_n + \sum_i \lambda^2 F_i (\theta_{n,i} - \theta_{o,i})$$
  여기서 $L_n$은 새로운 작업에 대한 손실 함수(Triplet Loss)이다.

### 3. Normalised Cross-Entropy (NCE)

Triplet Loss의 Outlier 취약성을 해결하기 위해, 앵커-포지티브-네거티브 쌍의 관계를 분류 문제로 정의한다. 앵커($x_a$)와 포지티브($x_p$)의 내적 값은 1(Positive)로, 앵커와 네거티브($x_{n_j}$)들의 내적 값은 0(Negative)으로 타겟팅하여 다음의 NCE 손실 함수를 적용한다.

$$\ell(z_i) = -\log \frac{\exp(z_i/\tau)}{\sum_{j=1}^{K+1} \exp(z_j/\tau)}$$

여기서 $z$는 다음과 같이 정의된 벡터이다.
$$z = (x_a \cdot x_p, x_a \cdot x_{n_1}, \dots, x_a \cdot x_{n_K})$$
$\tau$는 온도 파라미터(Temperature)이며 본 논문에서는 1로 설정하였다. 이 방식은 그래디언트를 제한하여 Outlier에 의한 급격한 가중치 변화를 방지한다.

### 4. Synthetic Visual Data Transfer

실제 데이터를 학습하기 전, **Cars3D**와 같은 대규모 합성 데이터셋으로 모델을 먼저 학습시킨다. 이후 실제 데이터에 대해 CIL을 수행할 때, 합성 데이터로 학습된 특징 추출 능력을 기반으로 학습을 시작함으로써 분포가 편향된 적은 양의 실제 데이터 스트림에서도 빠르게 적응하고 망각을 줄이도록 한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Cars3D(합성), MVCD(실제), CompCars(실제)의 3가지 데이터셋을 사용하였다.
- **아키텍처**: LeNet-5와 ResNet-18 두 가지 백본을 사용하였다.
- **평가 지표**: mean-Average-Precision (mAP) 및 **Forget ratio**(전체 데이터를 한 번에 학습했을 때의 mAP 대비 CIL 방식의 mAP 하락 비율)를 측정하였다.

### 2. 주요 결과

- **기존 방법론 평가 (Q1)**: 놀랍게도 단순한 **Nave** 방식이 많은 경우 다른 CL 방법론보다 우수하거나 유사한 성능을 보였다. 특히 LFL은 임베딩 자체를 학습해야 하는 CIL 특성상, 임베딩을 고정하려는 성질 때문에 성능이 가장 낮았다.
- **NCE의 효과 (Q2)**: NCE를 적용했을 때 대부분의 방법론에서 Forget ratio가 감소하였다. 특히 가중치 변화를 제한하는 **EWC**와 오버피팅에 취약한 **LeNet** 구조에서 그 효과가 두드러졌다.
- **합성 데이터 전이의 효과 (Q3)**: 합성 데이터로 프리트레이닝을 수행한 후 CIL을 적용했을 때 성능이 크게 향상되었다. 특히 MVCD 데이터셋에서 큰 이득을 보았으나, CompCars에서는 효과가 적었다. 이는 CompCars가 정면 뷰(frontal view)만 포함하고 있어, 다양한 각도를 제공하는 합성 데이터와의 분포 차이가 컸기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

본 논문은 단순한 클래스 분류를 넘어 개체 수준의 지속 학습이라는 새로운 문제 영역을 제시하였다. 특히, Metric Learning의 고질적인 문제인 Outlier 민감성을 NCE라는 분류 기반 손실 함수로 해결하려 한 접근이 매우 효율적이었다. 또한, 합성 데이터를 통한 전이 학습이 실제 환경의 데이터 부족 및 편향 문제를 완화할 수 있음을 정량적으로 보여주었다.

### 2. 한계 및 논의사항

- **데이터 분포의 영향**: 합성 데이터 전이 학습이 모든 데이터셋에서 동일하게 작동하지 않았다는 점은, 합성 데이터가 타겟 데이터의 도메인 특성(예: 시점, 각도)을 얼마나 잘 반영하느냐가 결정적임을 시사한다.
- **방법론의 범위**: 본 연구는 정규화(Regularisation) 기반의 CL 방법론에만 집중하였다. 저자들 또한 언급했듯이, 과거 샘플의 일부를 저장하여 함께 학습하는 **Replay-based** 기법을 적용한다면 망각 문제를 더 효과적으로 해결할 수 있을 것으로 보인다.
- **객체 범위**: 본 연구는 자동차라는 경직된(rigid) 객체에 한정되었으므로, 인간의 얼굴이나 의류와 같은 비경직(non-rigid) 객체에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 동일 카테고리 내의 개별 객체를 지속적으로 학습하는 **Continual Instance Learning (CIL)** 문제를 정의하고, 기존 CL 방법론들이 개체 학습 환경에서 발생하는 치명적 망각(Catastrophic Forgetting)에 취약함을 밝혔다. 이를 해결하기 위해 **Normalised Cross-Entropy (NCE)**를 도입하여 손실 함수의 안정성을 높이고, **합성 데이터 전이 학습**을 통해 초기 특징 표현력을 강화함으로써 망각을 유의미하게 감소시켰다. 이 연구는 향후 차량 재식별(Vehicle Re-ID)이나 실시간 감시 시스템과 같이 데이터가 끊임없이 유입되는 실무 환경에서 매우 중요한 기초 연구가 될 가능성이 높다.
