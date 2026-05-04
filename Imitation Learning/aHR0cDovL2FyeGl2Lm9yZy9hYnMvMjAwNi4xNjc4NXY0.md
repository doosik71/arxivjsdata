# LIPSCHITZNESS IS ALL YOU NEED TO TAME OFF-POLICY GENERATIVE ADVERSARIAL IMITATION LEARNING

Lionel Blondé, Pablo Strasser, Alexandros Kalousis (2023)

## 🧩 Problem to Solve

본 논문은 **Off-policy Generative Adversarial Imitation Learning (GAIL)** 방법론이 하이퍼파라미터에 극도로 민감하며, 학습 과정이 불안정하여 성공적인 성능을 내기 위해 과도한 엔지니어링 노력이 필요하다는 문제를 해결하고자 한다.

구체적으로, Off-policy GAIL은 **'Deadly Triad'**라 불리는 세 가지 요소(Function Approximation, Bootstrapping, Off-policy Learning)를 모두 포함하고 있어 발산 가능성이 크다. 여기에 더해, 학습 과정에서 보상 함수(Reward Function)가 함께 학습되므로 보상 신호 자체가 비정상성(Non-stationarity)을 띠게 되며, 이는 $\text{Q-value}$ 추정치의 분산을 증가시키고 과적합(Overfitting) 문제를 야기한다. 특히 Discriminator가 전문가의 데이터와 에이전트의 데이터를 너무 쉽게 구별하게 되면 보상 지형(Reward Landscape)에 날카로운 피크(Sharp Peaks)가 생기며, 이는 $\text{Q-value}$ 지형의 불안정성으로 이어져 정책 최적화를 방해한다.

따라서 본 연구의 목표는 Off-policy GAIL의 성능을 결정짓는 핵심 요소가 무엇인지 이론적·실험적으로 분석하고, 학습의 안정성을 보장하기 위한 충분조건과 필요조건을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **학습된 보상 함수가 Local Lipschitz-continuous(국소 립시츠 연속성)를 만족하도록 강제하는 것이 Off-policy GAIL의 성능 달성을 위한 필수 조건(Sine qua non)**이라는 점을 밝혀낸 것이다.

주요 기여 사항은 다음과 같다:
1. **보상 함수의 Lipschitzness와 안정성의 상관관계 규명**: 보상 함수에 Lipschitz 제약을 가하는 것이 $\text{Q-value}$ 함수의 변동성을 억제하여 학습의 안정성을 높임을 이론적으로 증명하고 실험적으로 입증하였다.
2. **Directed Regularization의 우수성 제시**: 다양한 Gradient Penalty(GP) 기법 중, 전문가 데이터와 에이전트 데이터를 잇는 경로 상에서 제약을 가하는 WGAN-GP 방식이 단순 국소 영역에 제약을 가하는 방식(DRAGAN 등)보다 우수함을 보였다. 이는 에이전트가 전문가를 향해 나아갈 수 있는 '자동 커리큘럼(Automatic Curriculum)' 역할을 하기 때문이다.
3. **PURPLE (Pessimistic Reward Preconditioning Enforcing Lipschitzness) 제안**: 보상 신호를 적절히 압축(Squashing)하여 $\text{Q-value}$의 Lipschitz 상수를 낮춤으로써, 시스템의 강건성(Robustness)을 이론적으로 보장하는 새로운 전처리기 기법을 제안하였다.

## 📎 Related Works

본 논문은 GAIL 및 그 발전 형태인 SAM, DAC와 같은 Off-policy adversarial IL 연구들을 기반으로 한다. 기존 연구들은 Discriminator의 과적합을 막기 위해 Spectral Normalization, Label Smoothing, Gradient Penalty 등을 사용해 왔으나, 이러한 기법들이 정확히 어떤 메커니즘으로 성능을 향상시키는지에 대한 심층적인 분석은 부족했다.

또한, RL의 일반적인 불안정성 원인인 'Deadly Triad'와 비정상적 보상(Non-stationary reward)에 관한 기존 문헌들을 검토하며, 본 연구가 단순한 정규화 기법의 적용을 넘어 $\text{Q-value}$의 Lipschitz 연속성이라는 관점에서 문제를 재정의하고 있음을 강조한다.

## 🛠️ Methodology

### 전체 시스템 구조
본 연구는 **SAM (Sample-efficient Adversarial Mimic)**을 기본 프레임워크로 사용한다. 이는 Actor-Critic 구조를 채택하여 샘플 효율성을 높인 Off-policy GAIL의 변형이다.

1. **Discriminator ($D_\phi$)**: 전문가 데이터($\pi_e$)와 에이전트 데이터($\beta$)를 구별하는 이진 분류기이다.
2. **Surrogate Reward ($r_\phi$)**: Discriminator의 출력을 이용해 보상을 정의한다. 본 논문에서는 **Minimax reward** 형태인 $r_\phi := -\log(1 - D_\phi)$를 사용한다.
3. **Critic ($Q_\omega$)**: TD-learning을 통해 $\text{Q-value}$를 추정하며, TD3의 기법(Clipped Double-Q learning, Target Policy Smoothing)을 도입하여 과대평가 편향을 줄인다.
4. **Actor ($\mu_\theta$)**: $\text{Q-value}$를 최대화하도록 결정론적 정책 경사법(Deterministic Policy Gradient)을 통해 업데이트된다.

### 핵심 방법론: Gradient Penalty (GP)
보상 함수의 급격한 변화를 막기 위해 Discriminator의 Jacobian norm에 제약을 가하는 Gradient Penalty를 손실 함수에 추가한다:

$$\ell_{GP}^\phi := \ell_\phi + \lambda \mathbb{E}_{s,a \sim \zeta} [(\|\nabla_{s,a} D_\phi(s,a)\| - k)^2]$$

여기서 $\zeta$는 정규화가 적용될 샘플 분포이며, $\lambda$는 정규화 강도, $k$는 목표 Lipschitz 상수이다.

### 이론적 분석 및 방정식
논문은 보상 함수 $r_\phi$가 $\delta\text{-Lipschitz}$일 때, 결과적으로 $\text{Q-value}$ 함수 $Q_\phi$의 Lipschitz 상수가 어떻게 결정되는지 증명한다. 무한한 호라이즌(Infinite-horizon) 설정에서 $\text{Q-value}$의 Lipschitz 상수 $\Delta_\infty$는 다음과 같다:

$$\|\nabla_{s,a} Q_\phi\|_F \le \Delta_\infty = \frac{\delta}{\sqrt{1 - \gamma^2 C}}$$

여기서 $C := A^2 \max(1, B^2)$이며, $A$와 $B$는 각각 환경의 다이내믹스($f$)와 정책($\mu$)의 Jacobian norm의 상한이다. 이는 보상의 변동성($\delta$)이 낮을수록, 그리고 시스템의 전이 및 정책 변동($C$)이 작을수록 $\text{Q-value}$가 더 강건해짐을 의미한다.

### PURPLE (Pessimistic Reward Preconditioning)
$\text{Q-value}$의 강건성을 더욱 높이기 위해, 보상 함수에 전처리기 $\kappa_t$를 곱하여 $\tilde{r}_\phi := \kappa_t r_\phi$로 변환한다. $0 < \kappa_t \le 1$인 이 전처리기는 보상 신호의 절대값을 줄여 $\text{Q-value}$의 Lipschitz 상수를 $\kappa \Delta_\infty$로 낮춤으로써 이론적으로 더 강건한 시스템을 구축한다.

## 📊 Results

### 실험 설정
- **환경**: MuJoCo 기반의 연속 제어 환경 (Hopper, Walker2d, HalfCheetah, Ant, Humanoid, InvertedDoublePendulum).
- **비교 대상**: NoGP (정규화 없음), NoGP-SN (Spectral Normalization만 적용), NoGP-SN-LS (SN + Label Smoothing 적용), GP (Gradient Penalty 적용).
- **측정 지표**: 에피소드당 누적 보상(Episodic Return).

### 주요 결과
1. **GP의 필수성**: 실험 결과, GP를 적용하지 않은 경우(NoGP, NoGP-SN, NoGP-SN-LS) 에이전트가 거의 학습되지 않거나 매우 낮은 성능을 보였다. 반면, GP를 적용했을 때만 전문가 수준의 성능에 도달할 수 있었다. 이는 **Local Lipschitz-continuity가 Off-policy GAIL의 성공을 위한 필요조건**임을 시사한다.
2. **GP 변형 비교**: WGAN-GP(전문가-에이전트 간 경로 정규화)가 DRAGAN이나 NAGARD(단순 국소 영역 정규화)보다 성능과 안정성 면에서 우수했다.
3. **C-validity와 성능의 상관관계**: 에이전트가 선택한 행동이 Lipschitz 제약을 얼마나 잘 만족하는지를 나타내는 지표($\text{b}_1^C$)와 실제 리턴 값 사이에 강한 양의 상관관계가 있음을 확인하였다.
4. **PURPLE의 효과**: PURPLE을 적용했을 때, 리턴 값은 비슷하게 유지되면서도 시스템의 변동성(Jacobian norm)이 유의미하게 감소하여 더 강건한 학습이 가능함을 보였다.

## 🧠 Insights & Discussion

### 보상 지형의 '날카로움'과 학습 실패
본 논문은 Discriminator의 과적합이 보상 지형에 '날카로운 피크'를 만들고, 이것이 $\text{Q-value}$ 지형의 불안정성을 초래하여 결국 정책 최적화가 실패하는 **'Overfitting Cascade'** 현상을 분석하였다. GP는 이 피크를 뭉툭하게 만들어(Smoothness) 최적화 경로를 안정화한다.

### Directed vs Isotropic Regularization
WGAN-GP가 우수한 이유는 정규화 영역이 고정되지 않고 에이전트와 전문가 사이의 '방향성'을 가지고 적응하기 때문이다. 이는 에이전트가 전문가를 향해 점진적으로 나아갈 수 있는 일종의 **보상 기반 커리큘럼**을 제공하는 효과를 준다.

### 한계 및 논의
- **전산 비용**: PURPLE이나 $\text{C-validity}$ 측정과 같이 Jacobian을 직접 계산해야 하는 기법들은 연산 비용이 매우 크다.
- **하이퍼파라미터 의존성**: 여전히 $\lambda, k, \tau$ 등의 하이퍼파라미터 튜닝이 필요하며, 특히 환경마다 적절한 값이 다를 수 있다는 점이 한계로 지적된다.
- **이론과 실제의 간극**: 이론적으로는 전역적 Lipschitzness를 가정하지만, 실제로는 미니배치 기반의 국소적 정규화만을 수행하므로, 실제 구현된 제약이 전 영역에서 보장되는지에 대해서는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 Off-policy GAIL의 고질적인 불안정성 원인이 보상 함수의 과적합으로 인한 '날카로운 보상 지형'에 있음을 밝히고, 이를 해결하기 위해 **보상 함수의 Local Lipschitz-continuity를 강제하는 Gradient Penalty(GP)가 필수적**임을 입증하였다. 특히 WGAN-GP 방식이 에이전트에게 효율적인 가이드를 제공하여 가장 우수한 성능을 보였으며, 추가적으로 보상 신호를 압축하는 **PURPLE** 기법을 통해 $\text{Q-value}$의 강건성을 이론적으로 보장하고 향상시킬 수 있음을 제시하였다. 이 연구는 Adversarial IL뿐만 아니라 일반적인 강화학습의 보상 설계 및 안정성 확보에 중요한 이론적 근거를 제공한다.