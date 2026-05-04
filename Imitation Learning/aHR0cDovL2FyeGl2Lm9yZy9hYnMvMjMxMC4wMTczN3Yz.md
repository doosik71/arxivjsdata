# Blending Imitation and Reinforcement Learning for Robust Policy Improvement

Xuefeng Liu, Takuma Yoneda, Rick L. Stevens, Matthew R. Walter, Yuxin Chen (2024)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)의 고질적인 문제인 높은 Sample Complexity를 해결하기 위해 모방 학습(Imitation Learning, IL)을 결합하되, 기존 방식들이 가진 전문가(Oracle) 의존성 문제를 해결하고자 한다.

일반적으로 IL은 전문가의 시연(Demonstration)을 통해 학습 효율을 높이지만, 이는 전문가의 성능이 매우 높다는 가정하에 유효하다. 현실적인 환경에서는 전문가가 최적이 아니거나(Suboptimal), 내부 동작을 알 수 없는 Black-box 형태로 제공되는 경우가 많다. 기존의 하이브리드 접근 방식인 MAMBA나 MAPS 등은 적어도 하나의 전문가가 특정 상태에서 최적의 행동을 제공한다는 가정을 전제로 한다. 그러나 모든 전문가가 특정 상태에서 낮은 성능을 보일 경우, 이러한 알고리즘들은 여전히 저성능 전문가를 맹목적으로 모방하게 되어 학습 효율이 떨어지거나 성능이 정체되는 문제가 발생한다.

따라서 본 연구의 목표는 전문가의 품질에 관계없이 강건하게 정책을 개선할 수 있는 알고리즘인 Robust Policy Improvement (RPI)를 제안하는 것이며, 이는 전문가의 조언과 학습자 스스로의 탐색(Exploration) 사이를 유연하게 오가는 메커니즘을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 학습자 정책(Learner Policy) 자체를 전문가 집합의 일부로 포함시키는 **Extended Oracle Set** 개념을 도입하여, 전문가의 성능이 학습자보다 낮을 때는 스스로의 가치 함수를 통해 학습하고, 전문가가 더 나은 성능을 보일 때만 모방 학습을 수행하도록 설계한 것이다.

구체적인 기여 사항은 다음과 같다:
1. **Robust Active Policy Selection (RAPS)**: 앙상블 가치 함수 모델을 통해 전문가와 학습자의 성능을 추정하고, 불확실성을 고려한 선택 전략을 통해 효율적으로 데이터를 수집한다.
2. **Robust Policy Gradient (RPG)**: 전문가의 조언과 RL의 자기 개선 능력을 결합한 새로운 Advantage function $\text{A}_{\text{GAE}}^+$를 기반으로 Actor-Critic 프레임워크를 구축하였다.
3. **$\text{max}^+$ Framework**: 전문가 집합에 학습자 정책을 추가하여 $\text{max}^+$-aggregation policy를 정의함으로써, 이론적으로 MAMBA보다 우수하거나 최소한 동등한 성능 하한선을 보장함을 증명하였다.

## 📎 Related Works

기존의 정책 개선 연구들은 주로 다음과 같은 한계를 가지고 있었다:
- **Pure RL (e.g., PPO)**: Sample complexity가 매우 높아 실제 로보틱스나 헬스케어 분야에 적용하기 어렵다.
- **Pure IL (e.g., Behavior Cloning, DAgger)**: 전문가가 최적(Near-optimal)이라는 가정에 의존하며, 전문가의 품질이 낮을 경우 성능 상한선이 전문가 수준으로 제한된다.
- **Hybrid Approaches (e.g., MAMBA, MAPS)**: 여러 전문가로부터 학습하여 성능을 높이려 했으나, 모든 전문가가 특정 상태에서 부적절한 조언을 제공하는 경우에도 모방 학습을 계속 수행하는 non-robustness 문제가 존재한다.

RPI는 전문가를 단순히 따라가는 것이 아니라, 온라인 성능 추정치를 기반으로 IL과 RL을 능동적으로 블렌딩(Blending)함으로써 이러한 한계를 극복하고 전문가의 성능을 초월하는 self-improvement를 가능하게 한다.

## 🛠️ Methodology

### 1. $\text{max}^+$ Framework 및 Extended Oracle Set
RPI는 전문가 집합 $\Pi^o = \{\pi_1, \dots, \pi_K\}$에 현재 학습자의 정책 $\pi_n$을 추가하여 확장된 전문가 집합 $\Pi^E = \Pi^o \cup \{\pi_n\}$를 정의한다.

이 집합을 바탕으로 베이스라인 가치 함수 $f^+(s)$를 다음과 같이 정의한다:
$$f^+(s) = \max_{k \in [|\Pi^E|]} V^k(s)$$
여기서 $V^k(s)$는 각 전문가(및 학습자)의 가치 함수이다. $\text{max}^+$-aggregation policy $\pi^\circledast$는 $f^+$에 대해 한 단계 개선(one-step improvement)을 수행하는 정책으로, 다음과 같은 $\text{A}^+$ Advantage function을 최대화하는 행동을 선택한다:
$$\text{A}^+(s, a) = r(s, a) + \mathbb{E}_{s' \sim P | \pi, s}[f^+(s')] - f^+(s)$$

### 2. Robust Active Policy Selection (RAPS)
전문가들의 가치 함수가 알려지지 않은 Black-box 상황에서, RPI는 앙상블 모델을 사용하여 평균 $\hat{V}_k^\mu(s)$와 불확실성 $\sigma_k(s)$를 추정한다.

RAPS는 전문가에게는 상한 신뢰 구간(UCB)을, 학습자에게는 하한 신뢰 구간(LCB)을 적용하여 데이터를 수집할 전문가 $k^\star$를 선택한다:
$$k^\star = \arg \max \left( \hat{V}_1^+(s), \dots, \hat{V}_K^+(s), \hat{V}_{K+1}^-(s) \right)$$
여기서 $\hat{V}^+ = \mu + \sigma$ (UCB)이고 $\hat{V}^- = \mu - \sigma$ (LCB)이다. 학습자에게 LCB를 적용하는 이유는 학습자가 확실히 전문가보다 우월하다고 판단될 때만 스스로의 데이터를 수집하고, 그렇지 않으면 전문가의 가이드를 통해 탐색(Exploration)을 촉진하기 위함이다.

### 3. Robust Policy Gradient (RPG)
RPG는 $\text{A}_{\text{GAE}}^+$라는 새로운 Advantage function을 사용하여 정책을 업데이트한다. 이는 Generalized Advantage Estimation (GAE)과 $\text{max}^+$ 베이스라인을 결합한 형태이다:
$$\hat{\delta}_t = r_t + \gamma \hat{f}^+(s_{t+1}) - \hat{f}^+(s_t)$$
$$\hat{\text{A}}_{\text{GAE}}^+_t = \hat{\delta}_t + (\gamma\lambda)\hat{\delta}_{t+1} + \dots + (\lambda\gamma)^{T-t-1}\hat{\delta}_{T-1}$$

이때 $\hat{f}^+(s)$는 전문가의 추정치가 너무 불확실할 경우($\sigma_k(s) > \Gamma_s$) 학습자의 가치 함수를 사용하도록 하는 confidence threshold $\Gamma_s$를 도입하여 강건성을 높였다:
$$\hat{f}^+(s) = \begin{cases} \hat{V}_{\pi_n}^\mu(s), & \text{if } \sigma_k(s) > \Gamma_s \\ \max_{k \in [|\Pi^E|]} \hat{V}_k^\mu(s), & \text{otherwise} \end{cases}$$

최종적으로 정책 그래디언트는 다음과 같이 계산된다:
$$\hat{g}_n = -\mathbb{E}_{s \sim d^{\pi_n}, a \sim \pi_n | s} \left[ \nabla \log \pi(a|s) \hat{\text{A}}_{\text{GAE}}^+_t(s, a) \right]$$

## 📊 Results

### 실험 설정
- **환경**: DeepMind Control Suite (Cheetah, Cartpole, Pendulum, Walker-walk) 및 Meta-World (Window-close, Faucet-open, Drawer-close, Button-press) 총 8개 태스크.
- **전문가**: PPO 및 SAC로 학습된 서로 다른 성능의 전문가 3명을 배치.
- **비교 대상**: PPO-GAE (Pure RL), Max-Aggregation (Pure IL), LOKI-variant, MAMBA, MAPS.
- **지표**: Best Return (최고 누적 보상).

### 주요 결과
1. **전반적 성능**: RPI는 모든 벤치마크에서 기존 baseline들을 능가하는 성능을 보였다. 특히 Dense reward 환경과 Sparse reward 환경 모두에서 강건한 성능 향상을 입증하였다.
2. **Sparse Reward 대응 능력**: Pendulum-swingup 및 Window-close와 같은 Sparse reward 태스크에서, Pure RL(PPO)은 학습이 매우 어려웠고, Pure IL 기반 방식들은 빠르게 성능이 정체되었다. 반면 RPI는 초기에는 전문가로부터 bootstrap하여 빠르게 학습하고, 이후 self-improvement 단계로 진입하여 전문가의 성능을 추월하였다.
3. **전문가 품질에 대한 강건성**: 전문가 세트가 비어있거나(No oracles) 단 한 명의 전문가만 있을 때도 RPI는 성능 저하 없이 작동하였다. 특히 전문가가 없을 때는 순수 RL(PPO-GAE)과 유사한 성능을 보이며 적응적으로 동작함을 확인하였다.

### Ablation Study
- **RAPS의 효과**: 단순한 APS(Active Policy Selection)보다 학습자를 선택지에 포함시킨 RAPS가 더 높은 성능을 보였다.
- **Confidence Threshold ($\Gamma_s$)**: $\Gamma_s=0.5$ 설정 시 가장 안정적인 성능을 보였으며, 이는 불확실한 전문가의 조언을 적절히 배제하는 것이 중요함을 시사한다.
- **LCB/UCB 전략**: 단순 평균($\text{MEAN}$)을 사용하는 것보다 LCB/UCB 기반의 정책 선택이 전반적으로 약 40% 더 높은 성능을 기록하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **IL의 효율성과 RL의 확장성을 적응적으로 결합**했다는 점이다. 학습 초기에는 학습자의 성능이 낮으므로 전문가의 가이드에 의존하는 IL의 성격이 강하게 나타나지만, 학습이 진행됨에 따라 $\hat{f}^+$가 학습자의 가치 함수로 수렴하며 자연스럽게 RL의 self-improvement 단계로 전이된다.

이론적 분석을 통해 RPI가 MAMBA보다 우수한 성능 하한선을 가짐을 증명한 점과, 실제 실험에서 Sparse reward 환경의 난제를 IL $\to$ RL 전이 과정을 통해 해결한 점이 인상적이다. 다만, 가치 함수 $\hat{V}$를 추정하기 위해 앙상블 모델을 사용하므로, 계산 복잡도와 메모리 사용량이 증가한다는 점이 잠재적인 한계로 작용할 수 있다. 또한, $\Gamma_s$와 같은 하이퍼파라미터가 성능에 영향을 미치므로, 이를 자동화하거나 환경에 맞게 튜닝하는 방법론에 대한 추가 연구가 필요할 것으로 보인다.

## 📌 TL;DR

RPI(Robust Policy Improvement)는 학습자 정책을 전문가 집합에 포함시키는 $\text{max}^+$ 프레임워크를 통해, 전문가가 최적이 아니더라도 강건하게 성능을 개선하는 알고리즘이다. 불확실성을 고려한 전문가 선택(RAPS)과 하이브리드 Advantage function(RPG)을 사용하여, 초기에는 전문가를 모방하고 후기에는 스스로 학습하는 전략을 취한다. 이를 통해 Sparse reward 환경에서도 효율적으로 학습하며 기존 SOTA 방법론들을 능가하는 성능을 보여주었으며, 향후 저품질 전문가 데이터만 존재하는 환경에서의 강건한 RL 연구에 중요한 이정표가 될 것으로 보인다.