# Few-shot Learning with Meta Metric Learners

Yu Cheng, Mo Yu, Xiaoxiao Guo, Bowen Zhou (2017/2019)

## 🧩 Problem to Solve

본 논문은 적은 수의 훈련 데이터만으로 새로운 클래스의 분류기를 학습시켜야 하는 Few-shot Learning(FSL)에서 기존 방식들이 가진 두 가지 핵심적인 한계점을 해결하고자 한다.

첫째, 기존의 Meta-learning 기반 접근 방식들은 메타 학습자가 태스크별 네트워크의 가중치를 예측하는 구조를 가지는데, 이는 모든 태스크가 동일한 수의 클래스를 가진다는 'N-way' 가정에 의존한다. 따라서 실제 환경처럼 태스크마다 클래스의 수가 다른 이질적인(heterogeneous) 구조를 가진 경우에는 가중치 예측이 매우 어려워진다.

둘째, Metric-learning 기반 접근 방식들은 모든 태스크에 대해 하나의 태스크 불변 메트릭(task-invariant metric)을 학습한다. 그러나 실제 데이터는 다양한 도메인에서 발생하며 태스크 간의 이질성이 클 경우, 단일 메트릭으로는 모든 태스크를 효과적으로 표현할 수 없으며 성능이 급격히 저하되는 문제가 발생한다.

결과적으로 본 논문의 목표는 클래스 수의 유연성을 확보하면서도, 각 태스크의 특성에 맞는 태스크별 메트릭(task-specific metric)을 생성할 수 있는 Meta Metric Learning 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **LSTM 기반의 Meta-learner**와 **Matching Networks 기반의 Base-learner**를 결합하는 것이다.

1. **유연한 클래스 대응**: 비모수적(non-parametric) 알고리즘인 Matching Networks를 Base-learner로 채택함으로써, 훈련과 테스트 단계에서 클래스 수가 서로 다르더라도 유연하게 대응할 수 있도록 설계하였다.
2. **태스크별 메트릭 생성**: 단순한 메트릭 학습에 그치지 않고, LSTM 기반의 메타 학습자가 Base-learner의 파라미터를 최적화하는 '학습 방법(update rule)'을 배우게 함으로써, 입력된 태스크의 특성에 최적화된 태스크별 메트릭을 생성할 수 있게 하였다.
3. **텍스트 도메인 확장**: 이미지 중심의 기존 FSL 연구에서 나아가, 텍스트 분류 도메인에서의 Few-shot deep learning 적용 가능성을 탐구하고 실증하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적하며 차별점을 둔다.

- **Siamese Networks & Matching Networks**: 이들은 데이터 간의 유사도를 측정하는 메트릭을 학습한다. 특히 Matching Networks는 서포트 세트(support set)를 활용해 새로운 클래스에 빠르게 적응할 수 있지만, 모든 태스크에 대해 단일한 메트릭을 사용하므로 태스크가 서로 이질적일 때 성능이 떨어진다.
- **LSTM-based Meta-learners**: 최적화 알고리즘 자체를 학습하여 모델 파라미터를 빠르게 업데이트하는 방식이다. 하지만 이 방식은 주로 고정된 구조의 네트워크를 가정하므로, 클래스 수가 변하는 환경에서는 적용이 어렵다는 한계가 있다.

본 연구는 Matching Networks의 유연성과 LSTM 메타 학습자의 적응력을 결합하여, 서로 다른 도메인과 가변적인 클래스 수라는 실제적인 제약 조건을 극복하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 Meta Metric Learner는 두 개의 모듈로 구성된다.

1. **Base Learner ($\mathcal{M}$)**: Matching Networks를 사용하여 데이터를 임베딩하고 유사도를 측정하여 분류를 수행한다.
2. **Meta Learner ($\mathcal{R}$)**: LSTM 구조를 통해 Base Learner의 파라미터를 어떻게 업데이트할지 결정하는 최적화 규칙을 학습한다.

### Base Learner: Matching Networks

입력 데이터 $x$를 벡터 공간으로 매핑하는 두 개의 임베딩 함수 $f(\cdot)$와 $g(\cdot)$를 사용한다. 서포트 세트 $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^k$가 주어졌을 때, 새로운 데이터 $\hat{x}$에 대한 예측 확률은 다음과 같이 계산된다.

$$y = P(\cdot | \hat{x}, \mathcal{S}) = \sum_{i=1}^k \alpha(\hat{x}, x_i; \theta) y_i$$

여기서 유사도 함수 $\alpha$는 다음과 같은 Softmax 분포로 정의된다.

$$\alpha(\hat{x}, x_i; \theta) = \frac{\exp(f(\hat{x}) \cdot g(x_i))}{\sum_{j=1}^k \exp(f(\hat{x}) \cdot g(x_j))}$$

$\theta$는 임베딩 함수 $f, g$의 파라미터이며, 본 논문에서는 $f$로 CNN 구조를 사용하였다.

### Meta Learner: LSTM-based Optimizer

전통적인 경사 하강법(Gradient Descent)의 업데이트 식 $\theta_{t+1} = \theta_t - \alpha_{t+1} \nabla L(\theta_t)$를 LSTM의 셀 상태 업데이트 과정으로 치환한다. LSTM의 셀 상태 $c_t$를 학습자의 파라미터 $\theta_t$로 간주하여, 메타 학습자가 손실 함수 $L$과 그 기울기 $\nabla L$을 입력받아 다음 파라미터 $\theta_{t+1}$을 직접 출력하도록 학습한다.

### Auxiliary Task Retrieval (보조 태스크 검색)

데이터가 극도로 부족한 상황(특히 one-shot)에서 모델을 안정적으로 업데이트하기 위해, 타겟 태스크와 유사한 보조 데이터셋 $\mathcal{D}_{aux}$를 검색하는 방법을 제안한다.

1. 각 가용 리소스 $T_i$에 대해 Matching Network $M_i$를 개별적으로 학습시킨다.
2. 타겟 태스크 $T_{target}$의 데이터에 대해 각 $M_i$의 정확도 $\text{acc}_{i \to k}$를 측정한다.
3. 정확도가 가장 높은 상위 $s$개의 태스크를 선정하여 $\mathcal{D}_{aux}$로 활용한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - **SCS (Sentence Classification Service)**: 12개 클라이언트의 텍스트 데이터 (클래스 수 10~28개).
  - **Omniglot**: 50개 알파벳 기반 이미지 데이터.
  - **Amazon Reviews**: 25개 제품 카테고리의 감성 분석 데이터.
- **비교 대상**: Matching Network (Basic/FCE), Meta-learner LSTM.
- **평가 지표**: 1-shot 및 5-shot 설정에서의 평균 정확도(Average Accuracy).

### 주요 결과

1. **멀티 도메인 설정**: SCS, Omniglot, Amazon Reviews 모든 데이터셋에서 Meta Metric-learner가 베이스라인 모델들보다 우수한 성능을 보였다. 특히 Amazon Reviews의 5-shot 설정에서는 Matching Network 대비 5% 이상의 성능 향상을 기록하였다.
2. **단일 태스크 설정 (SCS, Omniglot)**: 보조 데이터셋이 없는 환경에서도 우수한 성능을 보였다. 특히 메타-훈련 세트의 클래스 수가 메타-테스트 세트보다 적은 '3 vs 5' 분할(split)의 가혹한 조건에서도 Meta Metric-learner는 작동하였으나, Meta-learner LSTM은 적용이 불가능하였다.
3. **FCE(Fully Conditional Embedding)의 영향**: SCS 데이터셋에서는 FCE를 적용했을 때 성능이 향상되는 경향을 보였으나, Omniglot에서는 그 효과가 미미하였다.

## 🧠 Insights & Discussion

본 논문은 Meta-learning의 '최적화 능력'과 Metric-learning의 '유연한 구조'를 결합함으로써 FSL의 고질적인 문제인 가변적 클래스 수와 도메인 이질성 문제를 동시에 해결하였다.

**강점**:

- 단순한 메트릭 학습을 넘어, 메타 학습자가 태스크의 특성에 맞게 메트릭을 동적으로 조정하는 파라미터를 생성한다는 점이 매우 효율적이다.
- 텍스트와 이미지라는 서로 다른 도메인에서 일관된 성능 향상을 입증하여 제안 방법론의 범용성을 보여주었다.

**한계 및 비판적 해석**:

- **2단계 프로세스**: 보조 태스크를 먼저 검색한 후 학습하는 방식은 효율적이지만, 이를 하나의 end-to-end 프레임워크로 통합하지 못한 점은 아쉬움으로 남는다.
- **계산 복잡도**: LSTM이 매 스텝 파라미터 업데이트를 수행하고, 보조 태스크 검색을 위해 여러 개의 Matching Network를 미리 학습시켜야 하므로 연산 비용이 증가할 가능성이 크다.

## 📌 TL;DR

본 논문은 가변적인 클래스 수와 도메인 간의 차이를 극복하기 위해 **LSTM 기반 메타 학습자가 Matching Networks의 파라미터를 최적화하는 Meta Metric Learning** 기법을 제안하였다. 이를 통해 각 태스크에 최적화된 메트릭을 생성할 수 있게 되었으며, 텍스트 및 이미지 데이터셋 실험을 통해 기존의 단일 메트릭 기반 방식이나 고정 구조의 메타 학습 방식보다 뛰어난 성능을 입증하였다. 이 연구는 특히 실제 산업 현장처럼 데이터의 분포와 클래스 구조가 불규칙한 환경에서의 Few-shot Learning 적용에 중요한 기여를 한다.
