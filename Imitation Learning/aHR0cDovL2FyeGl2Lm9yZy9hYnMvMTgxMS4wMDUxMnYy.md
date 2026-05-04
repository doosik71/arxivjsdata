# Learning Beam Search Policies via Imitation

Renato Negrinho, Matthew R. Gormley, Geoffrey J. Gordon (2019)

## 🧩 Problem to Solve

본 논문은 구조적 예측(Structured Prediction) 문제에서 널리 사용되는 Beam Search의 학습 단계와 테스트 단계 간의 불일치 문제를 해결하고자 한다. 일반적인 모델들은 테스트 시에는 Beam Search를 사용하여 근사 디코딩(Approximate Decoding)을 수행하지만, 학습 시에는 Beam의 존재를 무시하고 최대 우도 추정(Maximum Likelihood Estimation)과 같은 방식을 사용한다.

이로 인해 두 가지 주요 문제점이 발생한다. 첫째, 학습 과정이 Beam Search의 특성을 반영하지 못하며, 둘째, 학습 시 오라클 궤적(Oracle Trajectories)만을 사용하여 학습함으로써 실제 테스트 시 모델이 실수했을 때 이를 복구하는 방법을 배우지 못하는 '오차 누적(Error Compounding)' 현상이 발생한다. 따라서 본 연구의 목표는 모방 학습(Imitation Learning)을 통해 Beam Search 정책을 명시적으로 학습함으로써 이러한 불일치를 해결하고, 이론적인 No-regret 보장을 제공하는 통합 메타 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Beam Search를 단순한 디코딩 도구가 아니라 모델의 일부로 간주하고, 이를 학습시키기 위한 통합적인 프레임워크를 제안한 점이다.

1.  **Beam-aware Meta-algorithm 제안**: surrogate loss 함수와 데이터 수집 전략(Data Collection Strategy)의 조합으로 구성된 메타 알고리즘을 통해 기존의 다양한 Beam-aware 알고리즘들을 하나의 체계로 통합하고 새로운 알고리즘을 제안하였다.
2.  **'Continue' 데이터 수집 전략**: 기존 알고리즘들이 비용 증가가 발생했을 때 학습을 중단(Stop)하거나 초기화(Reset)했던 것과 달리, DAgger에서 영감을 얻어 실수를 한 상태에서도 계속해서 궤적을 수집하는 'Continue' 전략을 도입하여 분포 불일치 문제를 완화하였다.
3.  **다양한 Surrogate Loss 설계**: Perceptron 기반 손실 함수부터 convex upper bound loss, log loss에 이르기까지 Beam의 특성을 반영한 다양한 손실 함수들을 정의하였다.
4.  **이론적 보장**: Beam Search 정책 학습에 대해 최초로 No-regret 보장(No-regret guarantees)을 수학적으로 증명하였으며, 유한 샘플 분석(Finite Sample Analysis)을 통해 실제 적용 가능성을 뒷받침하였다.

## 📎 Related Works

기존의 Beam-aware 알고리즘으로는 Early Update [6], LaSO [7], BSO [11] 등이 존재한다. 이러한 방식들은 학습 과정에서 Beam Search를 실행하여 손실을 계산하고 모델을 업데이트한다는 공통점이 있다.

그러나 기존 연구들은 다음과 같은 한계가 있다.
- **오차 노출 부족**: 모델이 연속적인 실수를 저지르는 상황에 노출되지 않는다. 즉, 비용 증가가 발생하면 즉시 학습을 멈추거나(Stop), 오라클이 지정한 최적 상태로 강제 이동(Reset)시키기 때문에, 모델이 자신의 실수로부터 학습할 기회가 부족하다.
- **이론적 근거 부족**: 대부분의 알고리즘이 이론적 보장이 없거나, 매우 제한적인 Perceptron 스타일의 보장만을 제공한다.

본 논문은 모방 학습 프레임워크를 도입하여, 모델이 생성한 궤적 위에서 오라클의 가이드를 받는 방식을 통해 이러한 한계를 극복하고 보다 강력한 No-regret 이론을 제시한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 Beam Search 정의
본 논문은 학습하고자 하는 대상을 scoring function $s(v, \theta)$로 정의한다. 이 함수는 각 노드 $v$에 점수를 부여하며, 이 점수를 기반으로 Beam Search 정책 $\pi$가 유도된다. 

- **Beam Search 과정**: 현재 빔 $b$에 속한 모든 노드의 이웃 집합 $A^b$를 구하고, $s(\cdot, \theta)$에 의해 점수가 가장 높은 상위 $k$개의 노드를 선택하여 다음 빔 $b'$를 형성한다.
- **목표**: 기대 비용 $c(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \mathbb{E}_{\hat{y} \sim \pi(\cdot, x, \theta)} c_{x,y}(\hat{y})$를 최소화하는 파라미터 $\theta$를 찾는 것이다.

### 2. 메타 알고리즘 (Meta-Algorithm)
알고리즘의 핵심은 **Surrogate Loss**와 **Data Collection Strategy**의 선택에 있다.

#### A. Surrogate Loss Functions
Beam 내에 최적의 요소가 유지되도록 강제하는 다양한 손실 함수를 제안한다. $s_{\hat{\sigma}(1)}$은 현재 모델이 예측한 최고 점수 노드, $s_{\sigma^*(1)}$은 실제 최적 비용을 가진 노드의 점수이다.

- **Perceptron (First)**: 최적 노드가 빔의 최상단에 오지 않았을 때 패널티를 부여한다.
  $$\ell(s, c) = \max(0, s_{\hat{\sigma}(1)} - s_{\sigma^*(1)})$$
- **Perceptron (Last)**: 최적 노드가 빔(크기 $k$) 밖으로 밀려났을 때 패널티를 부여한다.
  $$\ell(s, c) = \max(0, s_{\hat{\sigma}(k)} - s_{\sigma^*(1)})$$
- **Upper Bound Loss**: 기대 빔 전이 비용의 Convex upper bound를 제공하는 손실 함수이다.
  $$\ell(s, c) = \max(0, \delta_{k+1}, \dots, \delta_n)$$
  여기서 $\delta_j = (c_{\sigma^*(j)} - c_{\sigma^*(1)})(s_{\sigma^*(j)} - s_{\sigma^*(1)} + 1)$이다.
- **Log Loss (Neighbors)**: 모든 이웃에 대해 Softmax 형태의 정규화를 수행한다.
  $$\ell(s, c) = -s_{\sigma^*(1)} + \log \left( \sum_{i=1}^n \exp(s_i) \right)$$

#### B. Data Collection Strategy
학습 시 어떤 궤적을 따라 데이터를 수집할 것인가에 대한 전략이다.

- **Oracle**: 항상 최적의 경로를 따라 데이터를 수집한다. (분포 불일치 문제 발생)
- **Stop**: 모델의 예측으로 진행하다가 비용 증가(Cost increase)가 발생하면 즉시 중단한다.
- **Reset**: 비용 증가 발생 시, 빔을 최적의 단일 상태로 강제 초기화한 후 다시 학습을 진행한다.
- **Continue**: 비용 증가가 발생하더라도 이를 무시하고 모델의 정책을 따라 계속 진행하며 오라클의 가이드를 받는다. 이는 DAgger와 유사한 방식으로, 테스트 시의 분포와 유사한 환경에서 학습하게 한다.

### 3. 이론적 보장 (Theoretical Guarantees)
본 논문은 deterministic no-regret online learning 알고리즘을 사용할 때, 모델의 평균 손실이 최적 모델의 손실에 수렴함을 증명한다. 특히 **Theorem 2**에서는 surrogate loss가 기대 비용 증가분의 상한(upper bound)일 때, 이 손실의 no-regret 보장이 실제 예측 비용의 보장으로 이어진다는 것을 보여준다. 또한 Azuma-Hoeffding 부등식을 사용하여 유한한 샘플 환경에서도 높은 확률로 보장이 성립함을 증명하였다.

## 📊 Results

본 논문의 텍스트에서는 구체적인 수치적 실험 결과(정량적 지표)는 제시되지 않았다. 대신, 제안한 메타 알고리즘이 기존의 알고리즘들을 어떻게 포함하는지를 보여주는 **Table 1**을 통해 프레임워크의 범용성을 입증하였다.

- **기존 알고리즘의 재해석**:
    - **Log-likelihood**: Oracle 수집 + Log loss (neighbors) + $k=1$
    - **DAGGER**: Continue 수집 + Log loss (neighbors) + $k=1$
    - **Early Update**: Stop 수집 + Perceptron (first) + $k>1$
    - **LaSO**: Reset 수집 + Perceptron/Margin (last) + $k>1$
    - **BSO**: Reset 수집 + Cost-sensitive margin (last) + $k>1$

이 표는 본 논문의 메타 알고리즘이 단순한 새로운 방법론 제시를 넘어, 기존의 Beam-aware 학습 방법론들을 이론적으로 통합하고 분석할 수 있는 틀을 제공함을 보여준다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 논문은 Beam Search라는 구체적인 디코딩 기법을 'Learning to Search' 관점에서 공식화하여, 단순한 휴리스틱이 아닌 이론적 보장이 있는 학습 문제로 전환시켰다. 특히 'Continue' 전략을 통해 모방 학습의 핵심인 분포 불일치 문제를 Beam Search 환경에서 해결하려 시도한 점이 돋보인다. 또한, 다양한 손실 함수들의 Convexity를 분석하여 최적화 가능성을 논의한 점이 학술적으로 가치가 높다.

### 한계 및 논의사항
- **실험 데이터의 부재**: 제공된 텍스트 내에는 실제 벤치마크 데이터셋(예: 기계 번역, 구문 분석 등)에서의 성능 향상 수치가 명시되지 않았다. 이론적 증명은 완벽하나, 실제 복잡한 신경망 모델에서도 이러한 No-regret 보장이 실질적인 성능 향상으로 직결되는지에 대한 실증적 확인이 필요하다.
- **오라클 비용 계산의 가정**: 본 알고리즘은 학습 시 임의의 상태 $v$에서 최적 완료 비용 $c^*(v)$를 계산할 수 있다는 가정을 전제로 한다. 하지만 매우 큰 탐색 공간을 가진 실제 문제에서는 이러한 오라클 비용을 계산하는 것 자체가 매우 무거운 작업이 될 수 있다.

## 📌 TL;DR

본 논문은 Beam Search를 사용하는 구조적 예측 모델의 학습-테스트 불일치 문제를 해결하기 위해, **모방 학습 기반의 Beam-aware 정책 학습 메타 알고리즘**을 제안한다. 핵심은 다양한 **Surrogate Loss**와 **'Continue' 데이터 수집 전략**을 결합하여 모델이 자신의 실수로부터 학습하게 하는 것이며, 이를 통해 최초로 Beam Search 정책 학습에 대한 **No-regret 이론적 보장**을 제공하였다. 이 연구는 향후 빔 서치를 사용하는 모든 딥러닝 디코더의 학습 효율성을 높이는 이론적 토대가 될 가능성이 크다.