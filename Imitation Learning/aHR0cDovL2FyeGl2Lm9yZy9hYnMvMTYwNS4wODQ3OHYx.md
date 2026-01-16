# Model-Free Imitation Learning with Policy Optimization

Jonathan Ho, Jayesh K. Gupta, Stefano Ermon

## 🧩 Problem to Solve

모방 학습(Imitation Learning)은 알 수 없는 비용 함수를 가진 환경에서 전문가 시연(demonstrations)을 모방하여 에이전트가 행동을 학습하게 합니다. 기존 모방 학습 방법(특히 역강화 학습, IRL)은 일반적으로 계획(planning) 또는 강화 학습 문제를 반복적으로 해결해야 하므로 계산 비용이 매우 높습니다. 특히 대규모, 고차원 환경에서는 적용하기 어렵고, 내부 계획 문제가 최적으로 해결되지 않으면 성능이 크게 저하될 수 있습니다. 본 논문은 전문가 궤적(trajectory) 샘플을 사용하여 알 수 없는 비용 함수에 대해 전문가 정책만큼 잘 수행하는 매개변수화된 확률적 정책(parameterized stochastic policy)을 찾는 모델-프리(model-free) 알고리즘을 개발하는 것을 목표로 합니다.

## ✨ Key Contributions

- 정책 경사(policy gradient)를 기반으로 한 새로운 모델-프리 도제 학습(apprenticeship learning) 알고리즘을 제안합니다.
- `IM-REINFORCE`: 표준 정책 경사를 사용하여 정책 매개변수를 직접 최적화합니다.
- `IM-TRPO`: 신뢰 영역 정책 최적화(Trust Region Policy Optimization, TRPO)를 도제 학습 프레임워크에 통합하여 안정적이고 효율적인 정책 최적화를 가능하게 합니다.
- 다양한 비용 함수 클래스에 적용 가능하며, 특히 선형 비용 함수 클래스에서 효과적임을 보여줍니다.
- 최대 600개 이상의 연속 특징을 가진 대규모 고차원 환경에서 신경망 정책을 효과적으로 학습할 수 있음을 입증합니다.
- 내부 루프에서 값비싼 계획 또는 강화 학습 문제 해결 없이도 지역 최적해(local minima)로의 수렴을 보장합니다.

## 📎 Related Works

- **행동 복제(Behavioral Cloning, BC):** 가장 간단한 모방 학습 방식으로, 상태-액션 쌍을 지도 학습처럼 학습합니다. 하지만 학습된 모델의 작은 부정확성이 시간에 따라 누적되어 `cascading errors`를 유발할 수 있습니다.
- **역강화 학습(Inverse Reinforcement Learning, IRL):** 전문가가 알 수 없는 비용 함수에 대해 최적의 행동을 한다고 가정하고, 전문가 시연으로부터 비용 함수를 학습합니다. 하지만 각 반복마다 강화 학습을 실행해야 하므로 대규모 도메인에서 매우 비쌉니다.
- **도제 학습(Apprenticeship Learning, AL):** (Abbeel & Ng, 2004; Syed et al., 2008) 전문가 궤적을 통해 알 수 없는 참 비용 함수에 대해 전문가 정책만큼 잘 수행하는 정책을 찾는 프레임워크입니다.
  - `Feature Expectation Matching`: (Abbeel & Ng, 2004) 특징 기댓값(feature expectation)을 일치시키는 방식으로, `IRL`과 유사하게 내부 루프에서 강화 학습을 요구합니다.
  - `Game-theoretic approaches`: (Syed et al., 2008) `MWAL` 및 `LPAL`과 같이 게임 이론적 접근을 통해 도제 학습 문제를 해결하지만, 역시 계산 비용이 높거나 상태-액션 방문 분포에 직접 최적화됩니다.
- **신뢰 영역 정책 최적화(Trust Region Policy Optimization, TRPO):** (Schulman et al., 2015) 강화 학습을 위한 모델-프리 정책 검색 알고리즘으로, 안정적이고 단조로운(monotonic) 정책 개선을 보장합니다. 본 논문은 이를 모방 학습에 적용합니다.

## 🛠️ Methodology

- **문제 공식화:** 도제 학습 문제를 매개변수화된 확률적 정책 $\pi_\theta$를 최적화하는 것으로 재정의합니다. 이는 $\delta_\mathcal{C}(\pi_\theta, \pi_E) = \sup_{c \in \mathcal{C}} (\eta_c(\pi_\theta) - \eta_c(\pi_E))$를 최소화하는 문제입니다. 여기서 $\mathcal{C}$는 비용 함수 클래스입니다.
- **도제 학습을 위한 정책 경사:**
  - 경사 $\nabla_\theta \delta_\mathcal{C}(\pi_\theta, \pi_E) = \nabla_\theta \eta_{c^*}(\pi_\theta)$는 현재 정책 대비 전문가 정책의 "이점(advantage)"을 최대화하는 비용 함수 $c^*$에 대한 경사입니다.
  - 이는 (1) 고정된 $\theta$에 대해 $c^*$를 계산하고, (2) 이 $c^*$를 사용하여 $\theta$를 개선하는 교대 절차로 이어집니다.
- **IM-REINFORCE:**
  - 현재 정책 $\pi_\theta$와 전문가 정책 $\pi_E$로부터 궤적을 생성합니다.
  - $\sup_{c \in \mathcal{C}} (\hat{E}_{\rho_{\pi_\theta}}[c(s,a)] - \hat{E}_{\rho_{\pi_E}}[c(s,a)])$를 해결하여 경험적 최적 비용 $\hat{c}$를 추정합니다.
  - 샘플 궤적을 사용하여 정책 경사 $\nabla_\theta \eta_{\hat{c}}(\pi_\theta)$를 추정합니다.
  - 경사 하강법을 통해 정책 매개변수 $\theta$를 업데이트합니다.
- **IM-TRPO (도제 학습을 위한 신뢰 영역 정책 최적화):**
  - `IM-REINFORCE`의 높은 분산 문제를 해결하기 위해 `TRPO`를 적용합니다.
  - `KL-divergence` 신뢰 영역 제약 조건 하에서 $\delta_\mathcal{C}$에 대한 대리 목적 함수를 최소화합니다. 즉, 다음을 해결합니다:
    $$
    \min_\theta \sup_{c \in \mathcal{C}} \left( L_c(\pi_\theta) - \eta_c(\pi_E) \right) \quad \text{s.t.} \quad D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \le \Delta
    $$
    여기서 $L_c(\pi_\theta)$는 $\eta_c(\pi_\theta)$의 국소 근사치이며, $\eta_c(\pi_E)$는 전문가의 비용입니다.
  - 목적 함수 내의 $c$에 대한 `sup`는 $\theta$에 의존하더라도 효율적으로 계산될 수 있습니다. 모든 경험적 기댓값은 $\pi_{\theta_{\text{old}}}$에 대해 취해지므로, 새로운 시뮬레이션이 필요 없습니다. 선형 비용 함수($\mathcal{C}_{\text{linear}}$)의 경우 이 `sup`는 닫힌 형식(closed-form)으로 해를 가집니다.
  - `KL` 제약은 정책 업데이트의 급격한 변화를 방지하여 학습의 안정성을 높입니다.

## 📊 Results

- **그리드월드(Gridworlds):** `IM-REINFORCE`는 `LPAL`(전역 최적 방법)의 성능의 98%를 달성하며 유사한 샘플 복잡도를 보였지만, 훈련 시간 면에서는 훨씬 더 효율적이었습니다 (65536개 상태에서 `LPAL`은 평균 10분, `IM-REINFORCE`는 약 4분). `BC`는 성능이 크게 떨어졌습니다.
- **오브젝트월드(Objectworld, 소규모 연속 환경):** `IM-TRPO`는 모델-프리 방식임에도 불구하고 `CIOC`(모델 기반 `IRL` 방법)의 성능에 필적하는 제로 초과 비용(excess cost)을 달성했습니다.
- **워터월드(Waterworld, 다양한 차원):** `IM-TRPO`는 관측 차원(27개에서 102개 특징)이 증가해도 거의 완벽한 모방 성능을 보여주며 견고함을 입증했습니다. `IM-REINFORCE`는 상당히 느렸습니다. `IM-TRPO`의 비용 함수($c^*$) 계산 오버헤드는 무시할 수 있는 수준이었습니다.
- **고속도로 주행(Highway Driving, 고차원 부분 관측):** `IM-TRPO`는 610차원 자기 중심적 관측(egocentric observation)을 사용하여 다양한 주행 스타일(공격적, 꼬리 물기, 회피)을 학습했습니다. 학습된 행동은 `CIOC`(전체 상태 특징 및 환경 모델 사용) 및 인간 시연과 질적, 양적으로 유사했습니다.

## 🧠 Insights & Discussion

본 논문에서 제안하는 모델-프리 접근 방식은 복잡하고 고차원적인 연속 환경에서 모방 학습을 위한 신경망 정책을 성공적으로 훈련할 수 있음을 보여줍니다. 최적 계획(optimal planning)이 불가능한 환경에서도 지역 최적해를 찾을 수 있다는 점은 반복적인 계획/강화 학습을 요구하는 경쟁 알고리즘에 비해 상당한 이점입니다. 제안된 방법은 완전히 모델-프리이며, 전문가 상호작용이나 추가적인 강화 신호를 요구하지 않습니다. 이 접근 방식은 전체 동역학 모델 및 미분을 사용하는 모델 기반 방법과 비교해도 경쟁력 있거나, 때로는 더 나은 성능을 보여줍니다.

**제한 및 향후 연구:** 본 연구는 정책 최적화 구성 요소에 중점을 두었으며, 적절한 비용 함수 클래스 설계에는 집중하지 않았습니다. `GAN(Generative Adversarial Network)`에서 영감을 받은 비선형 비용 함수 클래스를 탐색하면 정확한 모방 학습의 가능성을 열 수 있습니다. 또한, 전문가 상호작용이나 강화 학습 신호를 사용할 수 있는 경우, 더 샘플 효율적인 방법과 결합하는 연구도 흥미로운 방향입니다.

## 📌 TL;DR

복잡한 환경에서 모방 학습의 한계인 값비싼 계획(planning) 및 역강화 학습(IRL) 문제를 해결하고자 합니다. 본 논문은 도제 학습(apprenticeship learning)을 위한 모델-프리 정책 경사(policy gradient) 알고리즘인 `IM-REINFORCE`와 `IM-TRPO`를 제안합니다. `IM-TRPO`는 `TRPO`를 기반으로 하여 안정적이고 효율적인 정책 학습을 제공합니다. 이 방법들은 명시적인 비용 함수 학습 없이, 매개변수화된 정책을 직접 최적화하여 전문가 행동을 모방합니다. 결과적으로 이 방법들은 고차원 연속 환경에 확장 가능하며, 지역 최적화 보장과 함께 전문가 수준의 성능을 달성하고, 기존 방법 대비 우수한 계산 효율성을 보여줍니다.
