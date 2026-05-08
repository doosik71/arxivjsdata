# Lifelong Metric Learning

Gan Sun, Yang Cong, Ji Liu, Xiaowei Xu (2017)

## 🧩 Problem to Solve

기존의 최신 온라인 Metric Learning 방식들은 사전에 정의된(predefined) 태스크들에 대해서만 메트릭을 학습할 수 있다는 한계가 있다. 새로운 태스크가 추가될 경우, 대부분의 기존 모델들은 이전의 모든 학습 데이터를 저장하고 있어야 하며, 새로운 데이터를 포함해 모델을 처음부터 다시 학습시켜야 하는 매우 비효율적인 과정을 거친다. 이는 데이터가 순차적으로 도착하는 실제 환경에서 계산 비용과 저장 공간 측면에서 심각한 문제를 야기한다.

본 논문의 목표는 인간의 학습 방식과 유사하게, 새로운 온라인 샘플로부터 새로운 태스크에 대한 능력을 부여하는 동시에 이전의 경험과 지식을 유지하고 통합하는 Lifelong Learning 관점의 Metric Learning 프레임워크를 구축하는 것이다. 즉, 이전 태스크의 학습 데이터에 다시 접근하지 않고도 새로운 태스크를 효율적으로 학습하며, 동시에 기존 태스크의 성능을 유지하거나 향상시키는 시스템을 설계하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 메트릭 태스크가 저차원의 공통 부분 공간(common subspace)에 존재한다는 가정에서 출발한다. 이를 위해 저자는 **Lifelong Dictionary**라는 공유 지식 저장소를 제안한다.

1. **Lifelong Dictionary ($L_0$)**: 모든 태스크가 공유하는 공통의 기저(basis) 집합으로, 새로운 태스크가 등장했을 때 이 딕셔너리의 희소 결합(sparse combination)을 통해 해당 태스크의 메트릭을 모델링한다.
2. **지식 전이 및 업데이트**: 새로운 태스크 $t+1$이 들어오면, 기존의 $L_0$로부터 지식을 전이받아 빠르게 학습하고, 동시에 새로운 태스크에서 얻은 정보를 바탕으로 $L_0$를 다시 정의함으로써 전체 태스크의 성능을 최적화한다.
3. **효율적인 최적화**: 모든 데이터를 다시 학습하는 대신, 이전 데이터의 1차 정보(first-order information)만을 유지하여 저장 공간을 절약하고, Online Passive Aggressive(PA) 최적화 알고리즘을 통해 계산 효율성을 극대화한다.

## 📎 Related Works

### 관련 연구 및 한계

- **Single Metric Learning**: batch 방식(LMNN 등)과 online 방식(OASIS, OMLLR 등)으로 나뉜다. Batch 방식은 모든 데이터를 미리 가지고 있어야 하며, Online 방식은 데이터가 순차적으로 들어올 때 효율적이지만, 여전히 고정된 태스크 내에서만 작동하며 새로운 태스크 추가에 취약하다.
- **Multi-task Metric Learning (MTML)**: 여러 관련 태스크를 동시에 학습하여 일반화 성능을 높인다. 예를 들어 mtLMNN이나 mtSCML은 공통의 메트릭이나 부분 공간을 학습하지만, 태스크의 수가 많아지면 계산 및 저장 비용이 기하급수적으로 증가하며, 태스크가 순차적으로 추가되는 lifelong 시나리오를 고려하지 않는다.

### 차별점

LML은 MTML처럼 공통 부분 공간을 활용하지만, 이를 'Lifelong Dictionary' 형태로 구현하여 태스크가 동적으로 추가되는 환경에서도 대응 가능하게 한다. 특히 과거의 데이터를 저장하지 않고도 지식을 누적할 수 있다는 점에서 기존의 Batch-based MTML이나 고정 태스크 기반의 Online Learning과 차별화된다.

## 🛠️ Methodology

### 1. 시스템 구조 및 수식 정의

각 태스크 $t$에 대한 메트릭 행렬 $M_t$는 다음과 같이 Lifelong Dictionary $L_0$와 태스크별 가중치 행렬 $W_t$의 조합으로 표현된다.

$$M_t = L_t^T L_t = L_0^T W_t L_0 = \sum_{i=1}^{d} \sum_{j=1}^{d} w_{ij} l_i l_j^T$$

여기서 $L_0 \in \mathbb{R}^{d \times \hat{d}}$는 공통 부분 공간을 나타내는 Lifelong Dictionary이며, $W_t \in \mathbb{R}^{d \times d}$는 각 태스크의 특성을 반영하는 가중치 행렬이다. $W_t$의 비대각 성분(off-diagonal elements)을 희소하게 만듦으로써 각 태스크가 딕셔너리에서 가장 재사용 가능한 핵심 지식만을 선택적으로 사용하도록 유도한다.

### 2. 학습 목표 및 손실 함수

LML의 전체 최적화 문제는 다음과 같은 목적 함수를 최소화하는 것이다.

$$\min_{L_0, \{W_t\}} \frac{1}{m} \left\{ \sum_{t=1}^{m} \left( \ell_t(L_0^T W_t L_0) + \lambda_t \|W_t\|_{1,off} \right) \right\} + \gamma \|L_0\|_F^2$$

- $\ell_t$: 태스크 $t$에 대한 손실 함수(유사도 또는 거리 기반).
- $\|W_t\|_{1,off}$: $W_t$의 비대각 성분에 대한 $\ell_1$-norm으로, 태스크 간 공유 지식의 희소성을 강제한다.
- $\|L_0\|_F^2$: Frobenius norm으로, 모델의 과적합(overfitting)을 방지한다.

### 3. 최적화 절차 (Online Passive Aggressive)

위 문제는 $L_0$와 $W_t$에 대해 joint하게 non-convex하므로, 저자는 Online Passive Aggressive(PA) 전략을 사용하여 이를 근사한다. 전체 프로세스는 $L_0$와 $W_t$를 교대로 업데이트하는 두 개의 서브 문제로 나뉜다.

1. **$W_t$ 업데이트 (Given $L_0$)**:
    새로운 태스크 $M^*_t$가 주어졌을 때, $L_0$를 고정하고 $W_t$를 최적화한다. 이때 non-smooth한 $\ell_{1,off}$-norm을 처리하기 위해 **FISTA(Fast Iterative Shrinkage-Thresholding Algorithm)**라 불리는 Proximal Gradient Method를 사용하며, Soft-thresholding 연산자를 통해 희소성을 확보한다.
2. **$L_0$ 업데이트 (Given $W_t$)**:
    학습된 $W_t$들을 고정하고, 모든 태스크의 손실 함수에 대한 gradient를 계산하여 $L_0$를 Gradient Descent 방식으로 업데이트한다. 이때 과거 데이터 대신 gradient 정보만을 사용하므로 데이터 저장 부담이 없다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - Label-consistent: Sentiment (Amazon reviews), Isolet (Speech recognition).
  - Label-inconsistent: USPS (Digit images).
- **비교 대상**:
  - Single Metric: Euclidean, OASIS, LMNN, stSCML.
  - Multi-task: mtLMNN, mtSCML.
  - Lifelong: ELLA.
- **지표**: 분류 오류율(Classification Error, %) 및 학습 시간(Runtime).

### 주요 결과

1. **정확도**: Sentiment와 Isolet 데이터셋에서 LML은 기존 SOTA 모델들보다 낮은 평균 오류율(각각 20.3%, 21.7%)을 기록하였다. 특히 ELLA와 같은 기존 Lifelong 모델보다 우수한 성능을 보였는데, 이는 지속적으로 업데이트되는 Lifelong Dictionary 덕분이다.
2. **효율성**: 모든 태스크를 다시 학습시켜야 하는 traditional MTML이나 union-based 방식에 비해 학습 시간이 매우 짧았다. 이는 새로운 태스크에 대해 $W_t$만을 빠르게 최적화하고 $L_0$를 소폭 수정하는 구조 덕분이다.
3. **일반화 능력**: USPS 데이터셋(Label-inconsistent)에서도 mtSCML에 근접하는 성능을 보였으며, 이는 단일 태스크 데이터만으로 학습했음에도 불구하고 공유 딕셔너리를 통해 타 태스크의 지식을 활용했음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 해석

- **지식의 누적**: 실험 결과, 학습하는 태스크의 수가 증가할수록 평균 오류율이 점진적으로 감소하는 경향이 확인되었다. 이는 LML이 새로운 태스크를 배울수록 $L_0$가 풍부해지며, 이 혜택이 초기 학습 태스크들에게도 환원되는 "Lifelong" 특성을 가졌음을 증명한다.
- **희소성의 효과**: $\|W\|_{1,off}$ 정규화가 없는 모델(stSCML)보다 성능이 좋게 나타났는데, 이는 변환된 특징들 간의 상관관계를 효율적으로 모델링하여 재사용 가능한 지식 덩어리를 잘 포착했기 때문으로 해석된다.

### 한계 및 논의

- **하이퍼파라미터 민감도**: transformed features의 차원 $d$에 따라 성능 변화가 크게 나타났다. (Sentiment 데이터셋의 경우 $d=120$에서 최적 성능). 이는 적절한 저차원 부분 공간의 크기를 결정하는 것이 성능에 결정적인 영향을 미침을 의미한다.
- **가정의 제약**: 모든 태스크가 하나의 공통 부분 공간 $L_0$에 존재한다는 가정을 전제로 한다. 만약 태스크 간의 연관성이 매우 낮은 경우(extremely heterogeneous tasks), 하나의 공유 딕셔너리가 오히려 간섭(interference)을 일으켜 성능을 저하시킬 가능성이 있다.

## 📌 TL;DR

본 논문은 새로운 메트릭 태스크가 지속적으로 추가되는 환경에서 과거 데이터를 저장하지 않고도 지식을 전이하고 누적할 수 있는 **Lifelong Metric Learning (LML)** 프레임워크를 제안한다. 모든 태스크가 공유하는 **Lifelong Dictionary**를 구축하고, 각 태스크를 이 딕셔너리의 희소 결합으로 표현함으로써 계산 효율성과 분류 정확도를 동시에 잡았다. 이 연구는 향후 실시간으로 새로운 클래스나 도메인이 추가되는 적응형 시스템(Adaptive System) 및 지속적 학습(Continual Learning) 기반의 컴퓨터 비전 시스템에 중요한 기초를 제공할 것으로 보인다.
