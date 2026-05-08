# Few-shot Learning with Meta Metric Learners

Yu Cheng, Mo Yu, Xiaoxiao Guo, Bowen Zhou (2017/2019)

## 🧩 Problem to Solve

본 논문은 매우 적은 수의 학습 데이터만으로 새로운 클래스의 분류기를 학습해야 하는 Few-shot Learning(FSL)에서 기존 방법론들이 가진 두 가지 핵심적인 한계를 해결하고자 한다.

첫째, 기존의 Meta-learning 기반 접근 방식은 Task-specific 네트워크의 가중치를 예측하는 Meta-learner를 학습시키는데, 이때 각 Task가 동일한 구조(Homogeneous-structured)를 가져야 한다는 전제가 필요하다. 즉, 모든 Task에서 클래스의 개수가 동일해야 하는 'N-way' 제약 조건이 존재하며, 이는 실제 환경에서 클래스 개수가 가변적인 상황을 반영하지 못한다.

둘째, Metric-learning 기반 접근 방식은 모든 Task에 대해 하나의 공통된 Metric(거리 측정 방식)을 학습한다. 그러나 Task들이 서로 다른 도메인에서 오거나 성격이 크게 다를 경우(Diverge), 단일한 Metric으로는 각 Task의 특성을 충분히 반영할 수 없어 성능이 저하되는 문제가 발생한다.

따라서 본 연구의 목표는 가변적인 클래스 개수를 처리할 수 있으면서도, 각 Task의 특성에 맞는 Task-specific Metric을 생성할 수 있는 'Meta Metric Learning' 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Matching Networks의 유연한 구조(Base Learner)**와 **LSTM 기반 Meta-learner의 최적화 능력(Meta Learner)**을 결합하는 것이다.

구체적으로, Matching Networks는 비모수적(Non-parametric) 특성을 가져 클래스 개수에 상관없이 동작할 수 있으며, LSTM Meta-learner는 일반적인 경사 하강법(Gradient Descent)의 업데이트 규칙을 학습하여 Base learner의 파라미터를 Task에 맞게 최적화한다. 이를 통해 모델은 불균형한 클래스 분포를 처리함과 동시에, 각 Task에 최적화된 Metric을 동적으로 생성할 수 있게 된다. 또한, 텍스트 도메인에서의 Few-shot deep learning 연구를 시도했다는 점이 주요 기여 사항이다.

## 📎 Related Works

본 논문은 크게 두 가지 기존 연구 흐름을 분석하고 그 한계를 지적한다.

1. **Metric-learning (e.g., Matching Networks):** 입력 데이터를 임베딩 공간으로 매핑하고 유사도를 측정하는 방식이다. 특히 Matching Networks는 지원 집합(Support set)과 쿼리 샘플 간의 유사도를 통해 라벨을 예측한다. 하지만 모든 Task에 대해 동일한 Metric을 사용하므로 Task 간의 다양성이 클 때 표현력의 한계가 있다.
2. **Meta-learning (e.g., LSTM-based Meta-learner):** '학습하는 법을 학습(Learning to learn)'하는 방식으로, 최적화 알고리즘 자체를 신경망으로 구현하여 파라미터를 빠르게 업데이트한다. 그러나 이 방식은 Task-specific 네트워크의 구조가 동일해야 하므로, 클래스 개수가 다른 가변적인 상황에서는 적용하기 어렵다.

본 제안 방법론은 Matching Network를 Base learner로 사용하여 클래스 개수의 제약을 없애고, LSTM Meta-learner를 통해 Task-invariant Metric의 한계를 극복함으로써 두 접근 방식의 장점을 통합하였다.

## 🛠️ Methodology

### 1. Base Learner: Matching Networks

Base learner는 Matching Networks를 사용한다. 두 개의 임베딩 함수 $f(\cdot)$와 $g(\cdot)$가 텍스트 $\mathbf{x}$를 벡터 공간으로 매핑한다. 지원 집합 $S = \{(\mathbf{x}_i, y_i)\}_{i=1}^k$가 주어졌을 때, 새로운 샘플 $\hat{\mathbf{x}}$에 대한 라벨 예측 확률은 다음과 같이 정의된다.

$$P(y|\hat{\mathbf{x}}, S) = \sum_{i=1}^k \alpha(\hat{\mathbf{x}}, \mathbf{x}_i; \theta) y_i$$

여기서 유사도 함수 $\alpha$는 다음과 같이 Softmax 분포로 계산된다.

$$\alpha(\hat{\mathbf{x}}, \mathbf{x}_i; \theta) = \frac{\exp(f(\hat{\mathbf{x}}) \cdot g(\mathbf{x}_i))}{\sum_{j=1}^k \exp(f(\hat{\mathbf{x}}) \cdot g(\mathbf{x}_j))}$$

$\theta$는 임베딩 함수 $f$와 $g$의 파라미터이다.

### 2. Meta Learner: LSTM-based Optimizer

Meta learner는 파라미터 $\theta$를 업데이트하는 최적화 규칙을 학습하는 LSTM 네트워크이다. 표준적인 경사 하강법은 다음과 같다.

$$\theta_{t+1} = \theta_t - \alpha_{t+1} \nabla L(\theta_t)$$

본 모델은 LSTM의 Cell state $c_t$를 learner의 파라미터 $\theta_t$로 설정하고, LSTM의 업데이트 메커니즘을 통해 위 수식의 $\alpha_{t+1}$(학습률) 및 업데이트 방향을 학습한다. 즉, Meta-learner $R$은 손실 함수 값 $L_t$와 그 기울기 $\nabla_{\theta_{t-1}} L_t$를 입력으로 받아 새로운 파라미터 $\theta_t$를 출력한다.

### 3. 전체 파이프라인 및 학습 절차

1. **Meta-training:** 여러 Task에서 샘플링된 $\mathcal{D}_{meta-train}$을 사용하여 LSTM Meta-learner $\Theta$를 학습시킨다. 각 Task 내에서 Base learner의 파라미터를 업데이트하고, 최종 Test loss를 통해 $\Theta$를 최적화한다.
2. **Updating Learner Parameters:** 새로운 Task가 주어지면, 학습된 Meta-learner를 사용하여 Base learner의 파라미터 $\theta$를 빠르게 업데이트한다.
3. **Auxiliary Task Retrieval:** 데이터가 매우 부족한 경우, 타겟 Task와 관련성이 높은 보조 Task $\mathcal{D}_{aux}$를 검색하여 학습에 활용한다. 관련성은 각 Task의 모델 $M_i$를 타겟 Task에 적용했을 때의 정확도 $acc_{i \to k}$를 기준으로 상위 $s$개를 선정하여 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Sentence Classification Service (SCS, 텍스트), Omniglot (이미지), Amazon Reviews (텍스트)
- **비교 대상:** Matching Network (Basic, FCE), Meta-learner LSTM
- **지표:** Average Accuracy (1-shot 및 5-shot 설정)

### 주요 결과

1. **Multi-tasks/domains Setting:**
   - **SCS:** Meta Metric-learner가 모든 설정에서 가장 높은 정확도를 기록하였다. 특히 1-shot에서 Matching Network보다 우수한 성능을 보였다.
   - **Omniglot:** 제안 방법론이 Baseline들을 상회하였으며, 보조 데이터를 활용했을 때 Matching Network보다 더 높은 성능 향상을 보였다.
   - **Amazon Reviews:** 5-shot 설정에서 Matching Network 대비 5% 이상의 성능 향상을 달성하였다. 반면, 기존 Meta-learner LSTM은 이 데이터셋에서 제대로 동작하지 않음을 확인하였다.

2. **Single Task Setting:**
   - 보조 데이터 없이 단일 리소스만 사용하는 환경에서도 실험을 진행하였다.
   - 클래스 개수가 적은 훈련 집합과 많은 테스트 집합을 가진 '3 vs 5 split'과 같은 도전적인 설정에서도 Meta Metric-learner가 가장 우수한 성능을 보였다. 특히 Meta-learner LSTM은 구조적 제약으로 인해 일부 설정(3 vs 5 split)에서 사용이 불가능했으나, 제안 방법은 성공적으로 동작하였다.

## 🧠 Insights & Discussion

본 연구의 강점은 Metric-learning의 **'구조적 유연성'**과 Meta-learning의 **'최적화 효율성'**을 성공적으로 결합했다는 점이다. 기존의 LSTM Meta-learner가 고정된 네트워크 구조에 묶여 있었다면, 본 모델은 Matching Network를 Base learner로 채택함으로써 클래스 개수가 가변적인 실제 시나리오에서도 적용 가능한 범용성을 확보하였다.

또한, 단순한 데이터 증강이 아니라 Task-level의 유사도를 측정하여 보조 Task를 선정하는 Retrieval 메커니즘을 도입함으로써, 관련 없는 데이터가 학습을 방해하는 Negative transfer 문제를 완화하였다.

한계점으로는 Meta-learner의 학습과 Base learner의 파라미터 업데이트가 분리된 2단계 절차로 이루어진다는 점이 있다. 저자들은 향후 연구에서 이를 End-to-end 프레임워크로 발전시키고, 다양한 도메인의 데이터를 더 효율적으로 통합하는 방법을 탐구할 필요가 있다고 언급한다.

## 📌 TL;DR

본 논문은 클래스 개수가 가변적인 환경에서도 작동하며 Task별 최적의 Metric을 생성할 수 있는 **Meta Metric Learner**를 제안한다. Matching Network를 Base learner로, LSTM을 Meta-learner로 구성하여, 기존의 'N-way' 제약을 극복하고 Task-specific한 파라미터 업데이트를 가능하게 하였다. 텍스트 및 이미지 데이터셋 실험 결과, 제안 방법은 특히 도메인이 다양하고 클래스가 불균형한 실제 Few-shot 상황에서 기존 SOTA 모델들보다 뛰어난 성능을 보였다.
