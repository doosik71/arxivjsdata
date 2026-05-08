# Accelerating Imitation Learning with Predictive Models

Ching-An Cheng, Xinyan Yan, Evangelos A. Theodorou, Byron Boots (2018)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)의 샘플 효율성(Sample Efficiency) 문제를 해결하기 위해 모방 학습(Imitation Learning, IL)의 수렴 속도를 가속화하는 것을 목표로 한다.

일반적으로 RL은 환경과의 상호작용 비용이 매우 크기 때문에, 전문가의 데이터를 활용하는 IL이 효율적인 대안으로 사용된다. 특히, 정책 평가와 최적화를 교차로 수행하는 Online IL(예: DAgger)은 이론적인 성능 보장이 가능하여 널리 사용되어 왔다. 그러나 기존의 Online IL 방식은 기본적으로 'Follow-the-Leader (FTL)' 구조를 따르며, 이는 최악의 경우(adversarial setting)를 가정하는 Online Learning 프레임워크에 기반하고 있어 수렴 속도가 상대적으로 느리다는 한계가 있다.

따라서 본 연구의 목표는 IL 문제의 특수성, 즉 환경의 다이내믹스와 전문가 정책이 시간에 따라 변하지 않는다는 점을 이용하여, 미래의 그래디언트를 예측하는 모델을 도입함으로써 Online IL의 수렴 속도를 획기적으로 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **미래의 비용 함수에 대한 그래디언트를 예측하여 'Be-the-Leader (BTL)'에 가깝게 행동**하는 것이다.

기존의 FTL 방식은 과거의 데이터를 기반으로 최적의 정책을 선택하지만, BTL은 미래의 비용 함수까지 미리 알고 그에 맞춰 현재의 결정을 내리는 방식이다. 물론 미래의 비용 함수를 완벽히 알 수는 없으나, 본 논문에서는 예측 모델(Predictive Model)을 통해 이를 근사함으로써 수렴 속도를 가속화한다.

주요 기여 사항은 다음과 같다:

- **MoBIL-VI 및 MoBIL-Prox 알고리즘 제안**: 변분 부등식(Variational Inequality)을 해결하는 개념적 알고리즘(MoBIL-VI)과 이를 실용적으로 구현한 1차 최적화 알고리즘(MoBIL-Prox)을 제안하였다.
- **이론적 수렴 속도 향상**: 특정 조건(가중치 설정 및 모델 구현 가능성) 하에서 기존 DAgger의 $O(\ln N / N)$ 수렴 속도를 $O(1/N^2)$까지 가속화할 수 있음을 증명하였다.
- **Mirror-Prox의 일반화**: MoBIL-Prox가 기존의 stochastic Mirror-Prox 알고리즘을 일반화한 형태임을 보였으며, 가중치 체계와 온라인 학습된 모델을 통해 성능을 높였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별점을 제시한다:

- **DAgger 및 Online IL**: Ross et al. [2]이 제안한 DAgger는 IL을 Online Learning 문제로 환원하여 수렴성을 보장했다. 하지만 이는 환경이 적대적(adversarial)이라고 가정하는 일반적인 Online Learning의 보수적인 수렴 속도를 따른다.
- **Follow-the-Regularized-Leader (FTRL)**: DAgger의 1차 최적화 변형들이 제안되었으나, 여전히 과거 데이터의 합에 의존하며 미래 예측을 수행하지 않는다.
- **Stochastic Mirror-Prox**: 변분 부등식(VI) 문제를 해결하기 위해 제안된 알고리즘으로, 두 번의 그래디언트 단계를 거쳐 수렴 속도를 높인다. 본 논문은 이 구조를 IL에 접목하되, 두 번째 그래디언트를 실제 환경이 아닌 '예측 모델'에서 가져옴으로써 환경 상호작용 횟수를 줄였다.

## 🛠️ Methodology

### 1. 문제 설정 및 환원 (Reduction)

RL의 목적 함수 $J(\pi)$를 직접 최적화하는 대신, 전문가 정책 $\pi^*$와의 차이를 최소화하는 대리 문제(surrogate problem)를 푼다:
$$\min_{\pi \in \Pi} E_{(s,t) \sim d^\pi} [D(\pi^*_s || \pi_s)]$$
여기서 $D$는 두 분포 간의 거리(예: KL divergence)를 측정하는 함수이다. 이를 Online Learning 문제로 환원하면, 각 라운드 $n$에서 학습자는 정책 $\pi_n$을 실행하고, 환경은 다음의 비용 함수 $f_n(\pi)$를 제공한다:
$$f_n(\pi) = F(\pi_n, \pi) = E_{d^{\pi_n}} [D(\pi^* || \pi)]$$
여기서 $F(\pi', \pi)$는 이변수 함수(bivariate function)로, $\pi'$는 상태 분포를 결정하고 $\pi$는 실제 정책 비용을 결정한다.

### 2. MoBIL-Prox 알고리즘

MoBIL-Prox는 계산 효율성을 위해 1차 최적화 방식을 사용하며, 다음과 같이 두 단계의 업데이트를 수행한다.

**Step 1: 임시 정책 $\hat{\pi}_{n+1}$ 계산 (FTL 단계)**
실제 환경에서 얻은 그래디언트 $g_n \approx \nabla f_n(\pi_n)$를 사용하여 FTL 방식으로 업데이트한다.
$$\hat{\pi}_{n+1} = \arg \min_{\pi \in \Pi} \sum_{m=1}^n w_m (\langle g_m, \pi \rangle + r_m(\pi))$$
여기서 $w_m$은 가중치, $r_m$은 강볼록(strongly convex) 정규화 함수이다.

**Step 2: 최종 정책 $\pi_{n+1}$ 계산 (예측 단계)**
$\hat{\pi}_{n+1}$이 다음 라운드의 정책이 될 것이라고 가정하고, 예측 모델 $\nabla^2 \hat{F}_{n+1}$을 통해 미래의 그래디언트 $\hat{g}_{n+1}$을 쿼리하여 최종 업데이트를 수행한다.
$$\pi_{n+1} = \arg \min_{\pi \in \Pi} w_{n+1} \langle \hat{g}_{n+1}, \pi \rangle + \sum_{m=1}^n w_m (\langle g_m, \pi \rangle + r_m(\pi))$$

### 3. 예측 모델 (Predictive Models)의 구현

$\hat{g}_{n+1}$을 얻기 위한 예측 모델은 다음과 같은 방식으로 구현될 수 있다:

- **다이내믹스 모델 기반 시뮬레이터**: 학습된 환경 모델을 통해 미래 궤적을 시뮬레이션하고 전문가 정책을 쿼리하여 그래디언트를 계산한다.
- **최근 비용 함수 활용**: $\hat{g}_{n+1} = \nabla \tilde{f}_n(\hat{\pi}_{n+1})$으로 설정하여 최근의 상태 분포가 유지된다고 가정한다.
- **신경망 기반 직접 예측**: 과거 그래디언트 데이터 $\{g_n\}$를 이용해 $\nabla^2 F$를 직접 예측하는 모델을 학습시킨다.

## 📊 Results

### 1. 실험 설정

- **환경**: DART 물리 엔진 기반의 CartPole(저차원) 및 Reacher3D(고차원) 로봇 제어 작업.
- **정책**: Linear Policy 및 소규모 Neural Network.
- **비교 대상**:
  - True Dynamics (이상적인 모델)
  - Learned Dynamics (온라인 학습된 모델)
  - Last Cost Function (최근 그래디언트 활용)
  - No Model (기존 DAgger 방식)
- **가중치 설정**: $w_n = n^p$에서 $p=0$(기본)과 $p=2$(가속화)를 비교.

### 2. 주요 결과

- **가속화 효과**: $p=0$일 때는 모델의 유무에 따른 차이가 적었으나, $p=2$일 때 MoBIL-Prox가 모델이 없는 경우보다 훨씬 빠르게 수렴하였다. 이는 이론적 예측과 일치한다.
- **모델 정확도의 영향**:
  - **CartPole**: 저차원 문제에서는 학습된 다이내믹스 모델이 매우 정확하여 True Dynamics와 유사한 성능을 보였다.
  - **Reacher3D**: 고차원 문제에서는 모델의 일반화 성능이 떨어져 True Dynamics와의 성능 격차가 발생했다. 그러나 모델에 오류가 있더라도 결국에는 수렴한다는 점을 확인하였다.
- **효율성**: Last Cost Function 기반의 예측 모델이 True Dynamics를 사용한 경우와 유사하게 매우 효율적인 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 의의

본 논문은 IL을 단순한 데이터 모방이 아니라, 구조화된 Online Learning 문제로 해석하여 수렴 속도를 수학적으로 증명했다는 점에 큰 의의가 있다. 특히 가중치 체계($w_n = n^p, p>1$)와 예측 모델을 결합했을 때 수렴 속도가 $O(1/N^2)$까지 빨라질 수 있음을 보임으로써, 샘플 효율성을 극대화할 수 있는 방향을 제시하였다.

### 2. 모델 오류에 대한 강건성

분석 결과, 예측 모델이 완벽하지 않더라도($\epsilon_w^{\hat{F}} > 0$) 알고리즘은 여전히 수렴하며, 최악의 경우 수렴 속도가 다소 느려질 뿐 전문가 정책의 성능에 도달하는 데는 문제가 없음을 확인하였다. 이는 실전 적용 시 모델의 부정확함이 치명적인 실패로 이어지지 않음을 의미한다.

### 3. 한계 및 논의사항

- **볼록성 가정**: 이론적 분석은 정책 공간 $\Pi$와 비용 함수 $f_n$의 볼록성(convexity)을 가정하고 있다. 실제 신경망 정책은 비볼록(non-convex)하지만, 실험적으로는 여전히 성능 향상이 나타났다. 이 간극에 대한 이론적 분석은 여전히 미해결 과제로 남아있다.
- **모델 학습 비용**: 예측 모델을 온라인으로 학습시키는 추가적인 연산 비용이 발생한다. 하지만 환경과의 상호작용 비용이 훨씬 크다는 전제 하에서는 충분히 감수할 만한 트레이드-오프이다.

## 📌 TL;DR

본 논문은 Online 모방 학습(IL)의 수렴 속도를 높이기 위해 **미래 그래디언트를 예측하는 모델 기반 알고리즘 MoBIL-Prox**를 제안한다. 미래 정보를 예측하여 'Be-the-Leader' 방식으로 정책을 업데이트함으로써, 기존 DAgger의 수렴 속도를 이론적으로 $O(\ln N/N)$에서 $O(1/N^2)$까지 가속화하였다. 시뮬레이션 결과, 다이내믹스 모델이나 최근 비용 함수를 이용한 예측이 실제 학습 속도를 유의미하게 향상시킴을 확인하였다. 이 연구는 데이터 수집 비용이 매우 큰 실제 로봇 제어 시스템의 학습 효율을 높이는 데 기여할 가능성이 크다.
