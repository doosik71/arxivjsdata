# A Statistical Guarantee for Representation Transfer in Multitask Imitation Learning

Bryan Chan, Karime Pereida, & James Bergstra (2023)

## 🧩 Problem to Solve

본 논문은 Imitation Learning(IL, 모방 학습)에서 발생하는 극심한 데이터 효율성 문제를 해결하고자 한다. 일반적인 IL, 특히 Behavioural Cloning(BC)은 전문가의 시연 데이터(expert demonstrations)를 대량으로 필요로 하며, 이는 로봇 공학이나 의료 서비스와 같이 데이터 수집 비용이 매우 높거나 불가능한 도메인에서 큰 제약이 된다.

이를 해결하기 위해 여러 소스 작업(source tasks)에서 학습된 표현(representation)을 타겟 작업(target task)으로 전이하는 Representation Transfer 방식이 제안되어 왔으나, 기존 연구들은 전이된 표현이 특정 타겟 작업에서 실제로 얼마나 효율적인지에 대한 이론적 보장(statistical guarantee)이 부족했다. 따라서 본 연구의 목표는 소스 작업들의 다양성이 충분할 때, 타겟 작업에서 샘플 효율성이 실제로 개선됨을 수학적으로 증명하고 이를 실험적으로 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

첫째, 소스 작업과 타겟 작업 간의 관계를 **Task Diversity($\sigma$-diversity)**라는 개념을 통해 정의하고, 이를 바탕으로 전이 학습의 샘플 효율성에 대한 이론적 분석을 제공하였다. 이는 단순히 표현을 전이하는 것을 넘어, 어떤 조건에서 전이가 유효한지를 정량적으로 제시한 것이다.

둘째, 기존의 분석들이 Gaussian complexity를 사용했던 것과 달리 **Rademacher complexity**를 도입하였다. 이를 통해 로그 인자(log factor)만큼 더 타이트한 바운드(tighter bound)를 도출하였으며, 이는 실제 딥러닝 아키텍처의 복잡도를 측정하는 방식과 직접적으로 연결되어 이론적 실용성을 높였다.

셋째, 네 가지 시뮬레이션 환경을 통해 소스 작업의 수($T$)와 작업당 데이터 양($N$)이 증가함에 따라 타겟 작업의 학습 효율이 실제로 향상됨을 입증하여 이론적 결과를 뒷받침하였다.

## 📎 Related Works

기존의 Representation Transfer 연구(예: Arora et al., 2020)는 표현 전이의 이점을 조사했으나, 소스 작업과 타겟 작업 사이의 구체적인 관계를 설명하지 못했다. 즉, 특정 타겟 작업에 대해 전이가 항상 유익하다는 보장이 없었다.

또한, Tripuraneni et al. (2020)과 같은 연구들이 Task Diversity 개념을 도입했으나, 이들은 Gaussian complexity에 의존하여 분석을 진행했다. 본 논문은 Rademacher complexity를 사용하여 기존 방식보다 더 정밀한 오차 범위를 계산함으로써 딥러닝 이론과의 간극을 좁혔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 논문은 **Multitask Imitation Learning (MTIL)** 프레임워크를 제안하며, 이는 크게 두 단계의 파이프라인으로 구성된다.

1.  **훈련 단계 (Training Phase):** $T$개의 소스 작업에서 공통된 표현 $\hat{\phi}$를 학습한다.
2.  **전이 단계 (Transfer Phase):** 학습된 $\hat{\phi}$를 고정(freeze)한 상태에서, 타겟 작업 $\tau$에 특화된 매핑 함수 $\hat{f}_\tau$만을 학습한다.

정책 $\pi$는 다음과 같은 softmax 매개변수 구조를 가진다.
$$\pi_{f, \phi}(s) = \text{softmax}((f \circ \phi)(s))$$
여기서 $\phi \in \Phi$는 공유 표현(representation)이며, $f \in F$는 작업별 매핑(task-specific mapping)이다.

### 학습 및 손실 함수
훈련 단계에서는 모든 소스 작업에 대해 다음과 같은 **Empirical Training Risk**를 최소화하는 $\hat{\phi}$를 찾는다.
$$\hat{R}_{\text{train}}(f, \phi) := \frac{1}{NT} \sum_{t=1}^{T} \sum_{n=1}^{N} \ell(\pi_{f_t, \phi}(s_{t,n}), a_{t,n}) \quad (1)$$
여기서 $\ell$은 log loss로 정의되며, 이는 전문가와 학습자 사이의 KL-divergence를 최소화하는 것과 같다.

전이 단계에서는 고정된 $\hat{\phi}$를 사용하여 타겟 작업의 데이터 $M$개를 통해 다음의 **Empirical Test Risk**를 최소화하는 $\hat{f}_\tau$를 학습한다.
$$\hat{R}_{\text{test}}(f_\tau, \phi) := \frac{1}{M} \sum_{m=1}^{M} \ell(\pi_{f_\tau, \phi}(s_m), a_m) \quad (2)$$

### 이론적 보장 및 주요 방정식
본 논문의 핵심인 **Theorem 1**은 타겟 작업의 정책 오차(Policy Error)의 상한선을 다음과 같이 제시한다.
$$\text{Policy Error} = \|v^{\pi^*_\tau} - v^{\text{softmax}(\hat{f}_\tau \circ \hat{\phi})}\|_\infty \le \frac{2}{\sqrt{2}(1-\gamma)^2} \sqrt{\epsilon_{\text{gen}}} + 2\zeta \quad (3)$$
여기서 $\epsilon_{\text{gen}}$은 일반화 오차(generalization error)이며, 다음과 같은 복잡도를 가진다.
$$\epsilon_{\text{gen}} = O\left(\frac{1}{\sqrt{\sigma^2 NT}}, \frac{1}{\sqrt{M}}, \frac{R_{NT}(\Phi)}{\sigma}\right)$$
- $R_{NT}(\Phi)$: 표현 클래스의 Rademacher complexity.
- $\sigma$: 소스 작업들의 다양성(diversity). $\sigma$가 클수록 전이 효율이 높아진다.
- $N, T, M$: 각각 소스 작업당 데이터 수, 소스 작업 수, 타겟 데이터 수이다.

이 식은 타겟 데이터 $M$이 부족하더라도, 소스 작업의 수 $T$와 데이터 $N$이 충분하고 작업들이 다양($\sigma$가 큼)하다면 낮은 정책 오차를 달성할 수 있음을 수학적으로 보여준다.

## 📊 Results

### 실험 설정
- **환경:** Frozen Lake, Pendulum, Cheetah, Walker 4종의 시뮬레이션 환경을 사용하였다.
- **비교 대상:** 처음부터 학습하는 Behavioural Cloning(BC)과 제안된 Multitask Behavioural Cloning(MTBC)을 비교하였다.
- **지표:** Normalized Returns를 통해 성능을 측정하였다.
- **변수:** 소스 작업의 수 $T$, 소스 데이터 $N$, 타겟 데이터 $M$을 각각 변화시키며 분석하였다.

### 주요 결과
1.  **소스 작업의 영향 (Figure 1):** 타겟 데이터 $M$을 고정한 상태에서 $N$과 $T$를 증가시킨 결과, MTBC의 성능이 전반적으로 향상되었으며 특히 Cheetah와 Walker 같은 복잡한 환경에서 BC보다 월등한 성능을 보였다.
2.  **타겟 데이터의 영향 (Figure 2):** 소스 데이터 $N$을 충분히 확보한 상태에서 타겟 데이터 $M$을 늘렸을 때, 성능 향상 폭이 매우 적었다. 이는 높은 성능이 주로 소스 작업의 데이터와 다양성에서 기인한 표현 $\hat{\phi}$ 덕분이며, 타겟 작업의 매핑 $f$를 학습하는 데는 상대적으로 적은 데이터로도 충분함을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 딥러닝의 관행이었던 '사전 학습 후 미세 조정(pre-train then fine-tune)' 전략이 모방 학습에서 어떻게 작동하는지를 수학적으로 정립하였다. 특히 타겟 데이터 $M$과 소스 데이터 $(N, T)$ 사이의 trade-off 관계를 명확히 함으로써, 데이터 수집이 어려운 환경에서 소스 작업의 다양성을 확보하는 것이 얼마나 중요한지를 시사한다.

### 한계 및 비판적 해석
- **$\sigma$-diversity의 정량화 문제:** 이론적으로 $\sigma$가 중요함을 증명했으나, 실제 학습 전에 작업 간의 다양성 $\sigma$를 어떻게 수치적으로 측정할 수 있는지에 대한 실무적인 방법론은 제시되지 않았다.
- **공유 표현의 가정:** 모든 작업이 하나의 공유 표현 $\phi^*$를 가진다는 가정을 전제로 한다. 하지만 실제 환경에서는 작업마다 최적의 표현이 다를 수 있으며, 저자 또한 이를 해결하기 위해 향후 Meta-learning 관점에서의 접근(작업별 표현을 기준 표현의 근방에 두는 방식)이 필요함을 언급하였다.

## 📌 TL;DR

본 논문은 여러 소스 작업에서 학습된 표현을 타겟 작업으로 전이하는 **Multitask Imitation Learning**의 샘플 효율성에 대한 통계적 보장을 제공한다. **Rademacher complexity**와 **Task Diversity($\sigma$)** 개념을 도입하여, 소스 작업이 다양하고 충분할수록 타겟 작업에서 필요한 데이터 양을 획기적으로 줄일 수 있음을 수학적으로 증명하고 실험적으로 입증하였다. 이 연구는 데이터 효율적인 로봇 학습을 위한 사전 학습 전략의 이론적 토대를 마련했다는 점에서 큰 의미가 있다.