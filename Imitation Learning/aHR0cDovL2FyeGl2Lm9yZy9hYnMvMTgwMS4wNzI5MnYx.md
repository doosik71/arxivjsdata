# Convergence of Value Aggregation for Imitation Learning

Ching-An Cheng, Byron Boots (2018)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)의 일반적인 프레임워크인 Value Aggregation의 수렴성 문제를 해결하고자 한다. 특히, 기존의 Value Aggregation 기반 알고리즘들은 생성된 정책 시퀀스 $\{\pi_n\}_{n=1}^N$ 내에 성능이 좋은 정책이 '존재'한다는 비점근적(non-asymptotic) 보장을 제공하지만, 시퀀스의 마지막 정책인 $\pi_N$이 반드시 수렴하거나 최적의 성능을 보장하는지에 대해서는 명확한 분석이 부족했다.

실제 확률적(stochastic) 환경에서는 시퀀스 내에서 가장 좋은 정책을 찾기 위해 각 반복 단계마다 막대한 양의 데이터를 수집하거나 모든 정책을 저장하여 최종 평가를 수행해야 하는 불편함이 있다. 이로 인해 많은 실무자들이 단순히 마지막 정책 $\pi_N$을 사용하지만, 이는 이론적 근거가 부족한 휴리스틱에 의존하는 방식이다. 따라서 본 연구의 목표는 마지막 정책 $\pi_N$의 수렴 조건을 이론적으로 규명하고, 수렴하지 않는 경우 이를 안정화할 수 있는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Value Aggregation 알고리즘의 수렴 여부를 결정짓는 임계 안정성 상수(critical stability constant) $\theta$를 정의하고 분석한 점이다. 

저자들은 상태 분포의 변화에 대한 민감도를 나타내는 $\beta$와 목적 함수의 강볼록성(strong convexity)을 나타내는 $\alpha$의 비율인 $\theta = \beta / \alpha$가 정책 시퀀스의 수렴성을 결정한다는 것을 증명하였다. 구체적으로 $\theta < 1$인 경우 마지막 정책 $\pi_N$이 수렴하며, $\theta > 1$인 경우 정책 시퀀스가 발산할 수 있음을 보였다. 또한, 정규화(regularization)를 통해 $\theta$ 값을 강제로 낮춤으로써 불안정한 문제를 안정화하고 마지막 정책의 성능을 보장할 수 있는 이론적 토대를 마련하였다.

## 📎 Related Works

본 논문은 Ross와 Bagnell(2014)이 제안한 AggreVaTe(Aggregate Value to Imitate) 프레임워크를 기반으로 한다. AggreVaTe는 온라인 볼록 최적화(Online Convex Optimization) 아이디어를 이용하여 정책 최적화와 평가를 반복적으로 수행함으로써, 전통적인 배치(batch) 모방 학습에서 발생하는 공변량 변화(covariate shift) 문제를 해결하고자 했다.

기존 연구에서는 $\pi_N$이 충분한 데이터를 통해 학습되었으므로 최선의 성능을 낼 것이라는 막연한 믿음이 있었으나, 본 논문은 이러한 믿음이 항상 옳지 않음을 지적한다. 기존 분석이 단순히 시퀀스 내 '최적 정책의 존재성'에만 집중했다면, 본 연구는 '마지막 정책의 수렴성'이라는 더 실질적인 문제로 관점을 전환하여 기존 접근 방식의 한계를 보완하였다.

## 🛠️ Methodology

### 전체 시스템 구조 및 원리

Value Aggregation의 기본 아이디어는 전문가 정책 $\pi^*$와 학습자 정책 $\pi$ 사이의 성능 차이를 최소화하는 것이다. 이를 위해 Kakade와 Langford(2002)의 성능 차이 보조정리(Performance Difference Lemma)를 활용한다.

$$J(\pi) = J(\pi') + T \mathbb{E}_{s,t \sim d^\pi} \mathbb{E}_{a \sim \pi} [A^{\pi'}_{|t}(s,a)]$$

여기서 $A^{\pi'}_{|t}(s,a)$는 정책 $\pi'$에 대한 어드밴티지 함수(advantage function)이다. AggreVaTe는 이 어드밴티지 함수를 순간 비용(instantaneous cost)으로 설정하여 다음과 같은 최적화 문제를 푼다.

$$\min_{\pi \in \Pi} \mathbb{E}_{d^\pi} \mathbb{E}_\pi [A^{\pi^*}_{|t}(s,a)]$$

### 학습 절차 및 업데이트 규칙

알고리즘은 초기 정책 $\pi_1$에서 시작하여 $n$번째 반복에서 다음과 같이 정책을 업데이트한다.

$$\pi_{n+1} = \arg \min_{\pi \in \Pi} \sum_{k=1}^n f_k(\pi)$$

여기서 $f_n(\pi) = \mathbb{E}_{d^{\pi_n}} \mathbb{E}_\pi [A^{\pi^*}_{|t}]$는 $n$번째 단계에서 수집된 상태 분포 $d^{\pi_n}$ 하에서의 기대 비용이다. 이는 Follow-the-Leader (FTL) 방식의 업데이트 규칙이다.

### 수렴성 분석을 위한 주요 방정식 및 상수

본 논문은 정책 $\pi$를 파라미터 $x \in X$로 표현하고, $F(y, x) = \mathbb{E}_{d^y} \mathbb{E}_x [A^{\pi^*}]$라고 정의한다. 여기서 두 가지 핵심 상수를 도입한다.

1. **$\alpha$ (Strong Convexity):** $F(z, \cdot)$가 $x$에 대해 $\alpha$-강볼록성을 가짐을 의미한다.
2. **$\beta$ (Lipschitz Continuity):** $\nabla_2 F(x, z)$가 첫 번째 인자인 $x$에 대해 $\beta$-립시츠 연속성을 가짐을 의미한다. 이는 정책 변화에 따라 상태 분포 $d^\pi$가 얼마나 민감하게 변하는지를 측정한다.

이 두 상수의 비율인 $\theta = \beta / \alpha$가 수렴의 핵심이다. $\theta < 1$일 때, 마지막 정책 $x_N$의 성능 $F(x_N, x_N)$에 대한 비점근적 상한(tight non-asymptotic bound)이 존재하며 시퀀스가 수렴한다.

### 확률적 문제로의 확장

데이터가 유한한 확률적 환경에서는 $f_n(\cdot)$을 유한 샘플의 합인 $g_n(\cdot)$으로 근사한다. 샘플 수 $m_n = m_0 n^r$ (여기서 $r \ge 0$)로 증가시킬 때, 수렴 속도는 $r$과 $\theta$의 관계에 의해 결정된다. 특히 $r > 0$인 경우, 샘플 수를 점진적으로 늘림으로써 결정론적 환경에서의 수렴 속도에 근접할 수 있음을 보였다.

## 📊 Results

### 결정론적 사례 분석 (Motivating Example)

저자들은 $\theta$의 영향력을 보여주기 위해 간단한 2단계 제어 문제를 설계하였다.
- 상태 전이: $s_1=0, s_2=\theta(s_1+a_1)$
- 비용 함수: $c_1=0, c_2=(s_2-a_2)^2$

실험 결과, $\theta > 1$인 경우 정책 시퀀스가 발산하며 성능이 지속적으로 악화되는 현상이 관찰되었다. 반면 $\theta < 1$인 경우에는 정책이 최적값 $x^*=0$으로 수렴하였다. 이는 $\theta$가 단순한 이론적 상수가 아니라 실제 시스템의 안정성을 결정하는 결정적 요소임을 입증한다.

### 이론적 성능 상한

Theorem 2에 따르면, $\theta < 1$일 때 마지막 정책의 성능 상한은 다음과 같다.

$$F(x_N, x_N) \le \tilde{\epsilon}_{\Pi, \pi^*} + \left( \frac{\theta e^{1-\theta} G_2}{2\alpha} \right)^2 N^{2(\theta-1)}$$

여기서 $G_2$는 립시츠 상수를 의미한다. $N \to \infty$일 때 $N^{2(\theta-1)}$ 항이 0으로 수렴하므로, 마지막 정책의 성능이 $\tilde{\epsilon}_{\Pi, \pi^*}$ (정책 클래스의 표현력 한계로 인한 오차) 수준으로 수렴함을 알 수 있다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 그동안 경험적으로만 사용되었던 "마지막 정책 선택" 전략에 대해 엄밀한 이론적 근거를 제시하였다. 특히 상태 분포의 민감도($\beta$)와 목적 함수의 곡률($\alpha$)의 관계를 통해 수렴성을 정의한 점은 매우 통찰력 있다. 이는 모방 학습에서 왜 어떤 문제는 쉽게 수렴하고 어떤 문제는 불안정한지를 수학적으로 설명해 준다.

### 한계 및 제언

논문에서 제시한 $\theta < 1$ 조건은 매우 강력한 제약 조건일 수 있다. 실제 복잡한 고차원 환경에서 $\beta$와 $\alpha$를 정확히 측정하는 것은 거의 불가능에 가깝다. 다만, 저자들은 이를 해결하기 위해 정규화 기법을 제안하였다.

### 정규화를 통한 안정화 (Regularization)

불안정한 문제($\theta > 1$)를 해결하기 위해 두 가지 정규화 방법을 제시한다.
1. **Mixing Policies:** 전문가 분포와 학습자 분포를 섞어서 샘플링함으로써 유효 $\beta$를 낮추는 방법이다.
2. **Weighted Regularization:** 목적 함수에 강볼록성 정규화 항 $\lambda R(x)$를 추가하는 방법이다. $\lambda > \theta - 1$이 되도록 설정하면 새로운 안정성 상수 $\tilde{\theta} = \beta / ((1+\lambda)\alpha) < 1$이 되어 수렴성이 보장된다.

## 📌 TL;DR

본 논문은 모방 학습의 Value Aggregation 프레임워크에서 마지막 정책 $\pi_N$의 수렴성을 결정하는 안정성 상수 $\theta = \beta / \alpha$를 정의하였다. $\theta < 1$일 때만 정책 시퀀스가 수렴하며, $\theta > 1$인 경우 발산할 수 있음을 이론적으로 증명하고 예시를 통해 보였다. 또한, 정규화(Regularization)를 통해 $\theta$를 낮춤으로써 불안정한 학습 과정을 안정화하고 마지막 정책의 성능을 보장할 수 있음을 제시하였다. 이 연구는 향후 복잡한 환경에서의 모방 학습 알고리즘 설계 시 정책의 안정성을 확보하기 위한 이론적 가이드라인을 제공한다.