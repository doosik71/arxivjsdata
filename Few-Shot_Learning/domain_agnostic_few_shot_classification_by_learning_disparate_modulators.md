# Domain-Agnostic Few-Shot Classification by Learning Disparate Modulators

Yongseok Choi, Junyoung Park, Subin Yi, Dong-Yeon Cho (2020)

## 🧩 Problem to Solve

기존의 Few-shot learning 연구들은 주로 메타-학습(Meta-learning)의 도움을 받아 빠르게 발전해 왔으나, 대부분의 연구가 메타-훈련(Meta-training)과 메타-테스트(Meta-testing) 데이터가 단일 도메인(Single domain)에서 온다는 가정을 전제로 한다. 그러나 실제 환경에서는 새로운 태스크가 훈련 과정에서 보지 못한 도메인을 포함하여 매우 다양한 도메인에서 임의로 발생한다.

따라서 본 논문은 메타-훈련 단계에서 접하지 못한 도메인을 포함하여, 여러 도메인에 걸쳐 있는 태스크 분포에서도 일반화 가능한 **Domain-agnostic few-shot classification** 문제를 해결하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다양한 태스크 분포를 커버할 수 있는 **모델 풀(Pool of models)**을 구축하고, 주어진 특정 태스크에 대해 가장 적합한 모델을 선택하는 방법을 학습하는 것이다.

구체적으로는 모든 모델이 하나의 **Base network**를 공유하여 도메인 불변 특징(Domain-invariant features)을 유지하게 하고, 각 모델마다 개별적인 **Modulator**를 두어 도메인 특성에 맞게 네트워크를 미세 조정함으로써 표현의 다양성(Representational diversity)을 확보하는 구조를 제안한다.

## 📎 Related Works

기존의 Few-shot learning은 주로 태스크 불변 메트릭 공간을 학습하거나(Metric-based), 최적화 과정을 학습(Optimization-based)하는 방식에 집중했다. 최근에는 도메인 시프트(Domain shift) 문제를 다루거나, 여러 모델을 결합하여 다양성을 활용하려는 시도가 있었다.

하지만 기존의 앙상블 기반 접근 방식들은 독립적인 모델들을 결합하는 형태였으며, 본 논문과 같이 공유 네트워크와 도메인 특화 모듈을 결합한 구조는 아니라는 점에서 차별점을 가진다. 본 연구는 멀티-태스크 학습 및 멀티-도메인 학습의 파라미터 공유 전략에서 영감을 받아, 효율적인 파라미터 구성과 긍정적인 지식 전이(Knowledge transfer)를 유도한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

전체 시스템은 크게 세 단계의 학습 절차를 거친다.

**Step 1: Base network 구축**
모든 소스 데이터셋을 통합한 거대 분류 데이터셋 $(x_{agg}, y_{agg})$을 구축하고, 이를 통해 모든 모델이 공유할 기본 네트워크 $\theta$를 지도 학습 방식으로 먼저 학습시킨다.

**Step 2: Modulator 학습**
동결된 Base network 위에 각 소스 도메인별로 개별 Modulator $\alpha_i$를 추가한다. 각 Modulator는 Prototypical Network(ProtoNet)와 동일한 방식의 메트릭 기반 메타-학습을 통해 해당 도메인의 특성에 맞게 학습된다. Modulator는 ResNet-18 기반 아키텍처의 각 잔차 블록(Residual block) 내에 삽입되며, 두 가지 형태(Convolution $1 \times 1$ 또는 Channel-wise transform)가 제안된다.

**Step 3: Selection network 학습**
구축된 모델 풀에서 주어진 태스크에 최적의 모델을 선택하는 메타-모델 $\phi$ (Selection network)를 학습시킨다.

### 2. 주요 구성 요소 및 알고리즘

**태스크 표현 (Task Representation)**
Selection network의 입력이 되는 태스크 표현 $z_{task}$는 서포트 세트(Support set)의 모든 예시를 Base network에 통과시켜 얻은 임베딩 벡터들의 평균값으로 정의된다.
$$z_{task} = \frac{1}{NK} \sum_{i=1}^{NK} f_e(x^s_i; \theta, \alpha_0)$$
여기서 $\alpha_0$는 변조가 없는 상태를 의미한다.

**모델 선택 및 학습 목표**
Selection network $f_s(\cdot; \phi)$는 $z_{task}$를 입력받아 모델 풀 내에서 가장 높은 정확도를 보이는 모델의 인덱스 $y_{sel}$을 예측하도록 학습된다. 이때 정답 레이블 $y_{sel}$은 쿼리 세트(Query set)에 대해 실제로 가장 높은 정확도를 기록한 모델의 인덱스로 결정된다.

**손실 함수**
Selection network의 학습에는 다음과 같은 교차 엔트로피(Cross-entropy) 손실 함수가 사용된다.
$$\hat{y}_{sel} = \text{softmax}(f_s(z_{task}; \phi))$$
$$\text{loss}_{sel} = \text{cross\_entropy}(\hat{y}_{sel}, y_{sel})$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Visual Decathlon의 8개 데이터셋(Aircraft, CIFAR100, DTD, GTSRB, ImageNet12, Omniglot, SVHN, UCF101, Flowers)을 사용한다.
- **평가 설정**: 5-way 5-shot 설정을 기본으로 하며, 훈련에 사용된 도메인에서 테스트하는 **Seen domains**와 전혀 새로운 도메인에서 테스트하는 **Unseen domains** 두 가지 시나리오로 평가한다.
- **비교 대상**: Fine-tune, Simple-Avg, ProtoNet, FEAT, ProtoMAML 등과 비교한다.
- **제안 방법**:
  - **DoS (Domain-generalized by Selection)**: Selection network가 선택한 최적의 단일 모델을 사용.
  - **DoA (Domain-generalized by Averaging)**: 풀 내 모든 모델의 예측 확률을 평균하여 사용.

### 2. 주요 결과

- **Seen Domains**: 제안 방법(DoS, DoA)이 대부분의 기존 방법론을 압도한다. 특히 선택 기반 방법인 **DoS**가 평균 방법인 DoA보다 성능이 좋은데, 이는 Selection network가 도메인 식별자 없이도 적절한 모델을 정확히 선택하고 있음을 시사한다.
- **Unseen Domains**: 여전히 제안 방법들이 우수한 성능을 보이지만, 여기서는 **DoA(평균 기반 방법)**가 DoS보다 더 높은 성능을 기록한다. 이는 보지 못한 도메인에서는 특정 모델 하나를 선택하는 것보다, 유용한 모델들의 결과물을 시너지 있게 결합하는 것이 더 효과적임을 의미한다.
- **파라미터 효율성**: Channel-wise transform modulator는 $1 \times 1$ Conv modulator보다 파라미터 수가 훨씬 적음에도 불구하고 매우 경쟁력 있는 성능을 보여준다.

## 🧠 Insights & Discussion

본 연구는 단일 메트릭 공간에 복잡한 도메인 분포를 매핑하는 대신, 모델 풀을 구축하고 선택/결합하는 전략을 취함으로써 도메인 일반화 문제를 효과적으로 해결하였다.

**강점 및 해석**:

- **표현의 다양성과 불변성의 조화**: Base network를 공유함으로써 도메인 공통의 특징을 유지하고, Modulator를 통해 도메인 특화 기능을 추가함으로써 다양성을 동시에 확보하였다.
- **학습의 용이성**: 고차원의 파라미터를 직접 조작하는 대신 '어떤 모델이 최적인가'라는 단순한 분류 문제로 태스크 적응을 재정의하여 학습 효율을 높였다.

**한계 및 논의**:

- **부정적 전이(Negative Transfer)**: 실험 결과, 소스 도메인의 수가 증가함에 따라 일부 데이터셋(예: CIFAR100)의 성능이 하락하는 경향이 관찰되었다. 이는 서로 다른 도메인 간의 부정적 전이를 방지하는 메커니즘이 추가로 필요함을 시사한다.
- **선택 방식의 단순함**: 현재는 단일 모델 선택 또는 단순 평균을 사용하고 있으나, 가중 평균(Weighted averaging)이나 소프트 선택(Soft selection) 방식이 도입된다면 성능을 더 높일 수 있을 것이다.

## 📌 TL;DR

본 논문은 다양한 도메인에 걸쳐 있는 Few-shot classification 문제를 해결하기 위해, 공유된 Base network와 도메인별 Modulator로 구성된 **모델 풀**과 최적의 모델을 고르는 **Selection network**를 제안한다. 실험을 통해 제안 방법이 기존의 메타-학습 알고리즘보다 보지 못한 도메인에 대해서도 뛰어난 일반화 성능을 보임을 입증하였으며, 이는 향후 확장 가능한 멀티-도메인 Few-shot 학습 연구에 중요한 기여를 할 것으로 보인다.
