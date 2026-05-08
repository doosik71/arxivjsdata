# Continuous Meta-Learning without Tasks

James Harrison, Apoorva Sharma, Chelsea Finn, Marco Pavone (2020)

## 🧩 Problem to Solve

본 논문은 기존 Meta-learning 알고리즘들이 가진 **Task segmentation(작업 분할)**에 대한 과도한 의존성 문제를 해결하고자 한다. 기존의 Meta-learning 방식은 학습 단계에서 오프라인 데이터가 이미 작업별로 분할되어 있다고 가정하며, 테스트 단계에서도 단일 작업 내에서 학습하도록 최적화되어 있다.

그러나 실제 환경, 예를 들어 로봇의 배포 과정에서는 환경적 요인이 관찰되지 않은 채 서서히 또는 급격하게 변할 수 있으며, 이러한 시계열 데이터에서 작업의 경계를 명확히 구분하는 Task segmentation을 수행하는 것은 매우 어렵거나 비용이 많이 든다. 따라서 본 연구의 목표는 작업 분할 정보가 없는 **Unsegmented time series data** 환경에서도 일반적인 Meta-learning 알고리즘을 적용할 수 있게 하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **MOCA (Meta-Learning via Online Changepoint Analysis)**라는 알고리즘 프레임워크를 제안한 것이다. MOCA의 중심 아이디어는 임의의 Meta-learning 알고리즘을 **미분 가능한 베이지안 변경점 탐지(Differentiable Bayesian changepoint detection)** 체계로 감싸는 것이다.

이를 통해 모델은 시계열 데이터를 처리하면서 현재 작업이 얼마나 지속되었는지를 나타내는 **Run length**에 대한 신념(Belief)을 유지하며, 이를 바탕으로 어떤 과거 데이터가 현재 작업에 유효한지를 동적으로 추론한다. 특히 이 전체 과정이 미분 가능하므로, 역전파(Backpropagation)를 통해 빠르게 적응하는 예측 모델(Meta-learning model)과 효과적인 변경점 탐지 알고리즘을 동시에 최적화할 수 있다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Online & Continual Learning**: 데이터 스트림에서 이전 작업의 지식을 재사용하면서 **Negative transfer**(이전 지식이 현재 학습을 방해하는 현상)를 방지하는 연구들이 진행되었다. 주로 정규화(Regularization) 기법을 통해 망각을 방지하지만, 이는 고정된 모델을 최적화하는 방식이며 빠르게 적응하는 학습 알고리즘 자체를 학습하는 Meta-learning과는 차이가 있다.
2. **Meta-Learning for Continual Learning**: 일부 연구에서는 고정된 크기의 **Sliding window** 방식을 사용하여 최근 데이터만으로 모델을 조건화한다. 하지만 윈도우 크기가 고정되어 있어 작업 변화에 유연하게 대응하지 못하며, Negative transfer의 위험이 여전히 존재한다.
3. **Empirical Bayes for Changepoint Models**: BOCPD(Bayesian Online Changepoint Detection)와 같은 연구들이 변경점 탐지를 다루어 왔으나, 주로 단순한 분포를 다루었으며 신경망 기반의 강력한 Meta-learning 모델과 결합하여 전체를 최적화하는 시도는 부족했다.

### 차별점

MOCA는 고정된 윈도우를 사용하는 대신, 현재 작업의 지속 시간(Run length)을 확률적으로 추론하여 **적응형 윈도우(Adaptive windowing)**를 구현한다. 또한, 미분 가능한 구조를 통해 변경점 탐지와 예측 모델을 통합적으로 학습시킨다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

MOCA는 베이지안 필터링(Bayesian Filtering)을 사용하여 Run length $r_t$(마지막 변경점 이후 경과 시간)에 대한 신념 분포 $b_t(r_t) = p(r_t | y_{1:t-1})$를 유지한다. 시스템은 다음과 같은 순환 과정을 거친다.

1. **입력 관찰 및 신념 업데이트**: 새로운 입력 $x_t$를 관찰하고, 모델의 생성적 특성을 이용해 Run length 신념을 업데이트한다.
2. **확률적 예측**: 모든 가능한 Run length에 대해 가중 평균을 내어 최종 예측값 $\hat{y}_t$를 산출한다.
3. **레이블 관찰 및 신념 업데이트**: 실제 레이블 $y_t$를 확인하고, 이를 통해 Run length 신념을 다시 업데이트한다.
4. **시간 전파**: 다음 시점으로 넘어가며, 일정 확률 $\lambda$(Hazard rate)로 작업이 변경되었다고 가정하여 신념을 전이시킨다.

### 주요 방정식 설명

**1. 입력 $x_t$에 의한 신념 업데이트**
입력 변수의 분포 변화를 감지하여 신념을 업데이트한다.
$$b_t(r_t | x_t) \propto p_\theta(x_t | \eta_{t-1}[r_t]) b_t(r_t)$$
여기서 $\eta_{t-1}[r_t]$는 과거 $r_t$개의 데이터에 적응한 모델의 파라미터이다.

**2. 레이블 $y_t$에 의한 신념 업데이트**
예측 오차를 바탕으로 현재의 Run length가 타당한지 평가한다.
$$b_t(r_t | x_t, y_t) \propto p_\theta(y_t | x_t, \eta_{t-1}[r_t]) b_t(r_t | x_t)$$

**3. 시간 전파 (Time Propagation)**
작업이 바뀔 확률 $\lambda$를 적용하여 다음 시점의 신념을 계산한다.
$$b_{t+1}(r_{t+1} = k) =
\begin{cases}
\lambda & \text{if } k=0 \\
(1-\lambda)b_t(r_t = k-1 | x_t, y_t) & \text{if } k > 0
\end{cases}$$

**4. 최종 예측 (Marginalization)**
단일 Run length를 선택하는 것이 아니라, 모든 가능성을 고려한 가중 합으로 예측한다.
$$p_\theta(\hat{y}_t | x_{1:t}, y_{1:t-1}) = \sum_{r_t=0}^{t-1} b_t(r_t | x_t) p_\theta(\hat{y}_t | x_t, \eta_{t-1}[r_t])$$

### 구현 모델 (Base Meta-learners)
MOCA는 확률적 해석이 가능한 모든 Meta-learning 모델을 사용할 수 있으며, 본 논문에서는 다음 세 가지를 구현하였다.
- **LSTM Meta-learner**: hidden state $h_t$와 cell state $c_t$를 $\eta$로 사용하여 시퀀스 정보를 인코딩한다.
- **ALPaCA**: 특징 공간(Feature space)에서 베이지안 선형 회귀를 수행하며, Matrix-normal 분포를 통해 파라미터를 재귀적으로 업데이트한다.
- **PCOC**: 분류 문제를 위해 특징 공간에서 베이지안 가우시안 판별 분석(GDA)을 수행하며, Dirichlet prior를 사용하여 클래스 확률을 추정한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**:
    - Regression: Sinusoid (정현파), Wheel Bandit (맥락적 밴딧), NBA Player Movement (선수 움직임 예측)
    - Classification: Rainbow MNIST, miniImageNet (5-way super-class 분류)
- **비교 대상 (Baselines)**:
    - **Train on Everything**: 작업 변화를 무시하고 전체를 하나의 데이터로 학습.
    - **Condition on Everything (COE)**: 모든 과거 데이터를 사용하여 파라미터를 업데이트.
    - **Sliding Window (SW)**: 최근 $n \in \{5, 10, 50\}$개의 데이터만 사용.
    - **Oracle**: 정확한 작업 변경 시점을 알고 있는 모델.
- **지표**: Negative Log-Likelihood (NLL), Regret (Bandit), Accuracy (Classification).

### 주요 결과
- **정량적 성과**: 모든 벤치마크에서 MOCA는 고정된 Sliding Window 방식보다 우수한 성능을 보였으며, Oracle 모델에 근접한 성능을 기록하였다. 특히 Hazard rate($\lambda$)가 낮을 때(작업이 천천히 바뀔 때) 효과가 극대화되었다.
- **정성적 성과**:
    - **NBA Player Movement**: MOCA가 예측한 변경점이 실제 선수의 의도 변화(예: 3점 슛 라인 이동 $\to$ 리바운드 시도)와 일치함을 확인하였다.
    - **Sinusoid**: 새로운 작업이 시작될 때 즉각적으로 Run length 신념을 0으로 리셋하며 빠르게 적응하는 모습을 보였다.
- **분류 작업**: Rainbow MNIST와 miniImageNet에서 MOCA는 Oracle에 근접한 성능을 내었으며, 이는 MOCA가 예측 전 단계에서 입력 데이터($x_t$)의 변화(예: 색상 변화)를 통해 변경점을 효과적으로 감지했기 때문이다.

## 🧠 Insights & Discussion

### 강점
MOCA는 **베이지안 추론과 미분 가능한 학습**을 결합함으로써, "어떻게 적응할 것인가(Meta-learning)"와 "언제 적응을 시작할 것인가(Changepoint detection)"를 동시에 최적화하였다. 또한, 특정 Meta-learner에 종속되지 않고 LSTM, ALPaCA, PCOC 등 다양한 모델에 적용 가능한 범용적인 프레임워크라는 점이 강력한 장점이다.

### 한계 및 논의사항
1. **i.i.d. Task 가정**: 본 논문은 새로운 작업이 항상 독립적으로 샘플링된다고 가정한다. 그러나 실제로는 과거에 수행했던 작업이 다시 나타나는 경우가 많으며, 이 경우 과거의 포스테리어(Posterior)를 재사용한다면 데이터 효율성을 더 높일 수 있을 것이다.
2. **계산 복잡도**: Run length에 대한 신념을 모든 $t$에 대해 유지하므로 시간 복잡도가 $O(t)$로 증가한다. 비록 테스트 시에는 매우 빠르지만(7ms/iter), 매우 긴 시퀀스를 다룰 때는 Pruning(가지치기) 기법이 필요할 수 있다.
3. **학습 시 윈도우 크기**: 학습 시 너무 짧은 시퀀스를 사용하면 Hazard rate가 인위적으로 높아지는 효과가 발생하므로, $\approx 1/\lambda$ 정도의 적절한 배치 길이를 설정하는 것이 중요하다.

## 📌 TL;DR

본 논문은 작업 분할 정보(Task segmentation)가 없는 시계열 데이터에서도 Meta-learning을 가능하게 하는 **MOCA** 프레임워크를 제안하였다. MOCA는 미분 가능한 베이지안 변경점 탐지 기법을 통해 현재 작업의 지속 시간을 확률적으로 추론하고, 이를 바탕으로 최적의 과거 데이터를 선택하여 예측을 수행한다. 실험을 통해 고정 윈도우 방식보다 월등한 성능을 입증하였으며, 이는 향후 로보틱스나 온라인 학습과 같이 데이터의 성격이 실시간으로 변하는 도메인에서 매우 유용한 도구가 될 가능성이 높다.
