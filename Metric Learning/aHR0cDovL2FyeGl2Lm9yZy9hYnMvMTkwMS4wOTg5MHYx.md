# Few-shot Learning with Meta Metric Learners

Yu Cheng, Mo Yu, Xiaoxiao Guo, Bowen Zhou (2017/2019)

## 🧩 Problem to Solve

본 논문은 매우 적은 수의 학습 데이터만으로 새로운 클래스의 분류기를 학습시켜야 하는 Few-shot Learning(FSL)에서 기존 접근 방식들이 가진 한계를 해결하고자 한다.

기존의 FSL 방법론은 크게 두 가지 방향으로 나뉜다. 첫째, Meta-learning 기반 접근법은 메타 학습기를 통해 각 태스크에 특화된 네트워크의 가중치를 예측하지만, 이는 모든 태스크가 동일한 수의 클래스를 가진다는 'N-way' 가정을 전제로 하므로 클래스 수가 가변적인 실제 환경에 적용하기 어렵다. 둘째, Metric-learning 기반 접근법은 모든 태스크에 대해 동일한 Task-invariant metric을 학습하여 사용한다. 그러나 실제 세계의 태스크들은 도메인이 매우 다양하며, 각 태스크에 최적인 metric이 서로 다르기 때문에 단일한 metric만으로는 표현력의 한계가 발생하며 태스크 간의 괴리가 클 경우 성능이 급격히 저하된다.

따라서 본 논문의 목표는 다양한 도메인에서 발생하는 가변적인 클래스 개수를 유연하게 처리하면서도, 각 태스크의 특성에 맞는 Task-specific metric을 생성할 수 있는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Matching Networks의 유연성**과 **LSTM 기반 Meta-learner의 적응력**을 결합한 'Meta Metric Learner'를 제안하는 것이다.

핵심적인 설계 직관은 다음과 같다. Matching Networks는 비매개변수(non-parametric) 알고리즘의 특성을 가져 클래스 수의 변화에 유연하게 대응할 수 있지만, 단일 metric을 사용한다는 단점이 있다. 반면, LSTM 기반의 Meta-learner는 학습 알고리즘 자체를 학습하여 모델 파라미터를 최적화하는 능력이 뛰어나다. 이 두 가지를 결합하여, Meta-learner가 Matching Network의 파라미터를 최적화하도록 함으로써 각 태스크에 최적화된 metric을 동적으로 생성하도록 설계하였다.

또한, 서로 관련 없는 태스크의 데이터를 무분별하게 학습에 사용할 경우 성능이 저하되는 문제를 해결하기 위해, 타겟 태스크와 유사도가 높은 태스크만을 선별하여 학습에 활용하는 **Auxiliary Task Retrieval** 메커니즘을 제안하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들의 한계를 지적하며 차별점을 제시한다.

1.  **Matching Networks**: Support set과 Query instance 간의 유사도를 측정하여 분류하는 방식으로, 새로운 클래스에 대해 미세 조정(fine-tuning) 없이 대응 가능하다. 그러나 모든 태스크에 대해 하나의 고정된 metric을 학습하므로, 태스크 간의 도메인 차이가 클 때 최적의 성능을 내지 못한다.
2.  **LSTM-based Meta-learner**: 최적화 알고리즘을 모델로 구현하여 학습하는 방식으로, 빠른 지식 습득이 가능하다. 하지만 태스크별 네트워크 구조가 동일해야 한다는 제약이 있어, 클래스 수가 다른 이질적인(heterogeneous) 네트워크 구조를 처리하는 데 어려움이 있다.

제안 방법론은 Matching Network를 Base learner로 채택하여 클래스 수의 유연성을 확보하고, 이를 LSTM Meta-learner로 제어함으로써 태스크별 맞춤형 metric을 생성한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
Meta Metric Learner는 크게 두 가지 모듈로 구성된다.
- **Base Learner (Matching Network)**: 실제 데이터를 임베딩하고 유사도를 측정하여 분류를 수행하는 모듈이다.
- **Meta Learner (LSTM)**: Base learner의 파라미터를 어떻게 업데이트할지를 결정하는 상위 학습 모듈이다.

### 상세 구성 요소 및 절차

#### 1. Base Learner: Matching Network
임베딩 함수 $f(\cdot)$와 $g(\cdot)$를 통해 입력 데이터를 $N$차원 벡터로 매핑한다. Support set $S = \{(x_i, y_i)\}_{i=1}^k$가 주어졌을 때, 새로운 데이터 $\hat{x}$의 레이블 예측 확률은 다음과 같은 유사도 함수 $\alpha$를 통해 계산된다.

$$P(y|\hat{x}, S) = \sum_{i=1}^k \alpha(\hat{x}, x_i; \theta) y_i$$

여기서 유사도 함수 $\alpha$는 다음과 같이 Softmax 분포로 정의된다.

$$\alpha(\hat{x}, x_i; \theta) = \frac{\exp(f(\hat{x}) \cdot g(x_i))}{\sum_{j=1}^k \exp(f(\hat{x}) \cdot g(x_j))}$$

$\theta$는 임베딩 함수 $f, g$의 파라미터이며, 본 논문에서는 텍스트 데이터를 위해 CNN 구조를 사용하였다.

#### 2. Meta Learner: LSTM
표준적인 경사 하강법(Gradient Descent) 업데이트 식 $\theta_{t+1} = \theta_t - \alpha_{t+1} \nabla L(\theta_t)$을 LSTM의 셀 상태 업데이트 과정과 대응시킨다. LSTM의 cell state $c_t$를 learner의 파라미터 $\theta_t$로 설정하고, LSTM이 손실 함수 $L$의 기울기 $\nabla L(\theta_t)$와 손실 값 $L_t$를 입력받아 다음 파라미터 $\theta_{t+1}$을 예측하도록 학습한다.

#### 3. Auxiliary Task Retrieval
타겟 태스크와 관련성이 높은 보조 데이터셋 $D_{aux}$를 선택하는 과정이다.
1. 각 데이터 리소스 $T_i$에 대해 Matching Network $M_i$를 개별적으로 학습시킨다.
2. 타겟 태스크 $T_{target}$에 대해 각 $M_i$ 모델들을 적용하여 정확도 $acc_{i \to k}$를 측정한다.
3. 정확도가 가장 높은 상위 $s$개의 태스크를 선택하여 $D_{aux}$로 구성한다.

### 학습 및 추론 절차
- **Meta-training**: 여러 샘플링된 태스크들에 대해 LSTM Meta-learner가 파라미터 업데이트 규칙을 학습한다.
- **Updating Learner**: 학습된 Meta-learner를 사용하여, 선택된 $D_{aux}$ 또는 테스트 서브셋을 통해 Base learner의 파라미터를 빠르게 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Sentence Classification Service (SCS, 텍스트), Omniglot (이미지), Amazon Reviews (텍스트/감성 분석)
- **비교 모델**: Matching Network (Basic/FCE), Meta-learner LSTM
- **평가 지표**: 1-shot 및 5-shot 설정에서의 평균 정확도(Average Accuracy)

### 주요 결과
1.  **Multi-task Setting**: 
    - SCS 데이터셋에서 Meta Metric Learner는 1-shot(58.13%)과 5-shot(74.54%) 모두에서 가장 높은 성능을 보였다.
    - Amazon Reviews에서는 5-shot 기준 Matching Network 대비 5% 이상의 성능 향상을 보이며, 특히 LSTM Meta-learner가 단독으로 사용되었을 때보다 월등한 성능을 기록했다.
2.  **Single Task Setting**: 
    - 보조 데이터셋이 없는 환경에서도 Meta Metric Learner는 다른 모델들보다 우수한 성능을 보였으며, 특히 클래스 수가 적은 학습셋으로 많은 클래스의 테스트셋을 분류해야 하는 도전적인 상황(3 vs 5 split)에서도 강건함을 입증했다.
3.  **정성적 분석**: FCE(Fully-Conditional Embedding) 기능을 결합했을 때 Basic 버전보다 성능이 향상되는 경향을 확인하여, 임베딩의 정교함이 성능에 기여함을 알 수 있다.

## 🧠 Insights & Discussion

### 강점
본 연구는 Meta-learning의 최적화 능력과 Metric-learning의 유연성을 성공적으로 결합하였다. 특히 실제 서비스 환경(SCS)과 같이 클래스 수가 가변적이고 도메인이 다양한 상황에서 Task-specific metric을 생성하는 접근 방식이 매우 효과적임을 입증하였다. 또한, 단순히 데이터를 많이 사용하는 것이 아니라 관련성 높은 태스크를 선별(Retrieval)하여 학습에 활용함으로써 노이즈를 줄인 점이 돋보인다.

### 한계 및 비판적 해석
1.  **Two-stage 파이프라인**: 보조 태스크를 먼저 선택하고 이후에 학습하는 2단계 구조를 가지고 있다. 이는 전체 프로세스를 end-to-end로 학습하는 방식에 비해 효율성이 떨어질 수 있으며, 초기 $M_i$ 학습 단계에서의 비용이 발생한다.
2.  **데이터 의존성**: Auxiliary Task Retrieval 단계에서 타겟 태스크의 일부 데이터를 사용하여 유사도를 측정해야 하므로, 완전한 zero-shot 상황에서의 태스크 선택 문제는 해결하지 못했다.
3.  **텍스트 도메인 기여도**: 논문에서는 텍스트 도메인에 대한 Few-shot deep learning 연구가 처음이라고 주장하지만, 당시 시점의 다른 연구들과 비교했을 때 범위 설정이 다소 제한적일 수 있다.

## 📌 TL;DR

본 논문은 클래스 수가 가변적인 실제 환경의 Few-shot Learning 문제를 해결하기 위해, **LSTM 기반의 Meta-learner가 Matching Network의 파라미터를 최적화하도록 설계한 Meta Metric Learner**를 제안한다. 이를 통해 각 태스크에 최적화된 Metric을 생성할 수 있으며, 관련 태스크 선택 기법을 통해 도메인 간의 괴리 문제도 완화하였다. 이 연구는 특히 텍스트 분류와 같이 도메인 다양성이 큰 분야에서 기존의 고정된 Metric 방식이나 경직된 Meta-learning 구조보다 훨씬 뛰어난 적응력을 보여주며, 향후 다양한 도메인의 전이 학습 연구에 중요한 기초를 제공한다.