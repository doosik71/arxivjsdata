# Safe Imitation Learning via Fast Bayesian Reward Inference from Preferences

Daniel S. Brown, Russell Coleman, Ravi Srinivasan, Scott Niekum

## 🧩 Problem to Solve

베이즈 보상 학습(Bayesian Reward Learning)은 모방 학습(Imitation Learning)에서 엄격한 안전성 및 불확실성 분석을 가능하게 하지만, 일반적으로 복잡한 제어 문제에 대해서는 계산적으로 다루기 어렵습니다. 이러한 비실용성은 고차원 문제나 MDP(Markov Decision Process) 모델을 사용할 수 없는 경우 모방 학습에서 강력한 안전성 및 불확실성 분석을 방해합니다. 기존의 딥러닝 기반 IRL(Inverse Reinforcement Learning) 방법은 주로 보상 함수의 점 추정(point estimate)만을 제공하므로, 견고한 안전성 분석에 필수적인 불확실성 정량화가 부족하다는 한계가 있습니다.

## ✨ Key Contributions

* **고효율 베이즈 보상 학습 알고리즘 제안:** 선호도 기반의 베이즈 보상 추론(Bayesian Reward Inference)을 통해 고차원 시각 제어 문제에 확장 가능하며, 계산 효율성이 높은 `Bayesian Reward Extrapolation (Bayesian REX)` 알고리즘을 제안합니다.
* **고속 베이즈 추론 달성:** 자기 지도 학습(self-supervised learning)을 통해 저차원 특징 임베딩(feature encoding)을 사전 학습하고, 시연 궤적(demonstrations)에 대한 선호도(preferences)를 활용하여 베이즈 추론 속도를 크게 향상시킵니다. 개인 노트북에서 5분 만에 보상 함수의 사후 분포(posterior distribution)에서 10만 개의 샘플을 생성할 수 있습니다.
* **경쟁력 있는 모방 학습 성능:** Atari 게임 모방 학습에서 보상 함수의 점 추정만을 학습하는 최첨단 방법과 비교하여 경쟁적이거나 더 나은 성능을 달성하며, 시연자(demonstrator)보다 더 나은 성능을 보입니다.
* **고신뢰 정책 평가(High-Confidence Policy Evaluation) 가능:** 보상 함수 샘플에 접근하지 않고도 임의의 평가 정책에 대한 효율적인 고신뢰 성능 하한(high-confidence lower bounds)을 계산할 수 있도록 합니다.
* **보상 해킹(Reward Hacking) 감지:** 이 고신뢰 성능 경계(performance bounds)를 사용하여 다양한 평가 정책의 성능과 위험을 순위화하고, 보상 해킹 행동을 감지하는 방법을 제공합니다.

## 📎 Related Works

* **모방 학습(Imitation Learning):**
  * **행동 복제(Behavioral Cloning):** 계산 효율적이지만, 누적 오류에 취약합니다 (Pomerleau, Ross et al., Torabi et al.). DAgger (Ross et al.) 및 DART (Laskey et al.)와 같은 방법은 이 문제를 해결합니다.
  * **역강화 학습(Inverse Reinforcement Learning, IRL):** 시연자가 최적화하는 보상 함수를 추정합니다 (Ng & Russell, Abbeel & Ng, Ziebart et al.). Bayesian IRL (Ramachandran & Amir)은 사후 분포에서 샘플링하지만, MDP를 반복적으로 풀어야 하므로 복잡한 문제에는 비실용적입니다.
  * **딥러닝 기반 IRL:** GAN (Finn et al.) 관련 접근 방식 (Ho & Ermon, Fu et al., Ghasemipour et al.)으로 복잡한 제어 문제에 확장되지만, 주로 보상 함수의 점 추정만을 제공합니다.
  * **선호도 기반 IRL:** 랭킹된 시연을 사용하여 보상 함수를 학습합니다 (Brown et al., 2019b;a). T-REX (Brown et al., 2019b)는 효율적이지만 점 추정만 제공합니다.
* **안전한 모방 학습(Safe Imitation Learning):**
  * SafeDAgger (Zhang & Cho) 및 EnsembleDAgger (Menda et al.)는 정책의 액션 차이가 클 때 시연자에게 제어권을 넘깁니다.
  * Brown & Niekum (2018)은 베이즈 샘플링으로 고신뢰 성능 경계를 제공하지만, 내부 루프에서 MDP 솔버가 필요합니다. Hadfield-Menell et al. (2017), Brown et al. (2018), Huang et al. (2018)도 MDP 솔버가 필요합니다.
* **가치 정렬 및 능동 선호 학습(Value Alignment and Active Preference Learning):**
  * 저차원 특징에 대한 베이즈 보상 추론에 활성 쿼리를 사용했습니다 (Sadigh et al., Brown et al., Bıyık et al.).
  * Christiano et al. (2017) 및 Ibarz et al. (2018)은 딥 네트워크로 고차원 작업에 확장했지만, 많은 활성 쿼리가 필요하고 베이즈 보상 추론은 수행하지 않습니다.
* **안전한 강화 학습(Safe Reinforcement Learning):**
  * 안전한 탐색 전략 또는 기대 수익률 외의 최적화 목표에 중점을 둡니다 (Garcıa & Fern ́andez). VaR(Value at Risk) 및 CVaR(Conditional VaR)와 같은 위험 민감 측정치가 사용됩니다 (Tamar et al., Chow et al.). 고신뢰 비정책 평가(high-confidence off-policy evaluation)도 연구되었습니다 (Thomas et al., Hanna et al.).
* **베이즈 신경망(Bayesian Neural Networks):**
  * MCMC, 변분 추론, 앙상블, 드롭아웃 등을 사용하여 BNN 가중치에 대한 사후 분포를 근사화합니다 (MacKay, Sun et al., Lakshminarayanan et al., Gal & Ghahramani). 본 논문은 불확실한 보상 함수 하에서 정책 평가의 불확실성을 측정하는 데 중점을 둡니다.

## 🛠️ Methodology

1. **고신뢰 정책 평가 문제 공식화 (HCPE-IL):**
    * MDP$\setminus$R, 평가 정책 $\pi_{eval}$, 시연 집합 $D=\{\tau_1,...,\tau_m\}$, 신뢰 수준 $\delta$, 성능 통계량 $g: \Pi \times \mathcal{R} \to \mathbb{R}$이 주어졌을 때, 참 보상 함수 $R^*$에 대한 $\pi_{eval}$의 성능 $g(\pi_{eval}, R^*)$에 대해 $(1-\delta)$ 신뢰도의 하한 $\hat{g}(\pi_{eval}, D)$을 찾는 문제로 정의합니다.
2. **Bayesian IRL의 병목 해결:**
    * 표준 Bayesian IRL (Ramachandran & Amir, 2007)의 주요 병목은 최적 Q-값 계산($Q^*_R$)을 통해 우도 함수를 계산하는 것입니다. 이는 MDP를 반복적으로 풀어야 하므로 고차원 시각 도메인에는 비실용적입니다. 이 문제를 해결하기 위해 새로운 우도 함수를 공식화합니다.
3. **쌍별 랭킹 우도(Pairwise Ranking Likelihood) 활용:**
    * $m$개의 궤적 $D = \{\tau_1, ..., \tau_m\}$과 궤적에 대한 쌍별 선호도 $P = \{(i,j) : \tau_i \prec \tau_j\}$에 접근한다고 가정합니다.
    * 표준 브래들리-테리 모델(Bradley-Terry model)을 사용하여 다음 쌍별 랭킹 우도 함수를 정의합니다:
        $$ P(D,P | R_{\theta}) = \prod_{(i,j) \in P} \frac{e^{\beta R_{\theta}(\tau_j)}}{e^{\beta R_{\theta}(\tau_i)} + e^{\beta R_{\theta}(\tau_j)}} $$
        여기서 $R_{\theta}(\tau) = \sum_{s \in \tau} R_{\theta}(s)$는 보상 함수 $R_{\theta}$ 하의 궤적 $\tau$의 예측 수익(predicted return)이며, $\beta$는 선호도 라벨의 신뢰도를 모델링하는 역온도(inverse temperature) 파라미터입니다.
    * 이 우도 함수는 MDP를 풀거나 환경에 접근할 필요가 없어 계산 효율성이 높습니다.
4. **최적화 (Optimizations):**
    * **저차원 잠재 특징 임베딩:** 보상 함수 $R_{\theta}$를 딥 네트워크로 표현하되, MCMC 제안(proposal) 시 마지막 레이어 가중치($w$)만 변경합니다. 신경망의 마지막에서 두 번째 레이어의 활성화(activation)를 $k$차원 잠재 보상 특징 $\phi(s) \in \mathbb{R}^k$으로 사용하고, 보상은 $R(s) = w^T \phi(s)$의 선형 조합으로 표현합니다.
    * **$\Phi_{\tau}$ 사전 계산 및 캐싱:** 궤적 $\tau$의 특징 합계를 $\Phi_{\tau} = \sum_{s \in \tau} \phi(s)$로 사전 계산하여 캐싱합니다. 이를 통해 우도 함수 계산은 $\prod_{(i,j) \in P} \frac{e^{\beta w^T \Phi_{\tau_j}}}{e^{\beta w^T \Phi_{\tau_j}} + e^{\beta w^T \Phi_{\tau_i}}}$가 되어 MCMC 샘플링 시 매 제안마다 $O(mk)$ 연산으로 효율성을 극대화합니다. 초기 계산은 $O(mT|\mathcal{R}_{\theta}|)$입니다.
    * **효율적인 정책 가치 평가:** 정책 $\pi$의 가치 함수는 $V^{\pi}_R = w^T E_{\pi}[\sum_{t=0}^T \phi(s_t)] = w^T \Phi_{\pi}$로 표현될 수 있습니다. $\Phi_{\pi_{eval}}$을 한 번만 계산($O(|S|^3)$)하면, $N$개의 보상 함수 가설에 대해 $O(Nk)$ 연산으로 효율적인 정책 평가가 가능합니다. 총 복잡도는 $O(|S|^3 + Nk)$로 크게 감소합니다.
5. **보상 함수 네트워크 사전 학습:**
    * 잠재 임베딩 함수 $\phi(s)$를 시연 데이터만을 사용하여 자기 지도 학습 방식으로 사전 학습합니다. 이는 제한된 선호도 데이터에 과적합되는 것을 방지하기 위함입니다.
    * 활용된 자기 지도 학습 태스크는 다음과 같습니다:
        * **역동학 모델 (Inverse Dynamics Model):** $\phi(s_t)$와 $\phi(s_{t+1})$로 $a_t$를 예측합니다.
        * **전방 동학 모델 (Forward Dynamics Model):** $\phi(s_t)$와 $a_t$로 $s_{t+1}$를 예측합니다.
        * **시간 거리 (Temporal Distance):** $\phi(s_t)$와 $\phi(s_{t+x})$로 $x$를 예측합니다.
        * **변분 오토인코더 (Variational Autoencoder):** $\phi(s)$를 통해 $s_t$를 재구성합니다.
    * 이러한 보조 목적 함수들은 임베딩이 다양한 특징을 인코딩하도록 유도합니다.
6. **HCPE-IL 구현:** Bayesian REX로 얻은 $P(w|D,P)$ 분포의 샘플들을 사용하여 정책 성능 $g(\pi_{eval}, R)$의 사후 분포를 계산합니다. $(1-\delta)$ 신뢰 하한은 $g(\pi_{eval}, w)$ 값의 $\delta$-quantile로 결정합니다. 구체적으로, $V^{\pi_{eval}}_{R^*} = w^{*T}\Phi_{\pi_{eval}}$의 $1-\delta$ 신뢰 하한을 계산합니다. 이는 $\delta$-VaR(Value at Risk)에 해당합니다.

## 📊 Results

* **Bayesian IRL 대비 효율성:**
  * 낮은 차원 그리드월드(gridworld) 실험에서, 충분한 수의 순위가 매겨진 비최적 시연(>5개)이 주어졌을 때, Bayesian REX는 동일한 수의 최적 시연이 주어진 Bayesian IRL과 비슷하거나 더 나은 성능을 보였습니다.
  * 비최적 시연만 주어진 경우, Bayesian REX는 Bayesian IRL보다 일관되게 우수한 성능을 보였습니다.
  * 최적 시연만 주어진 경우, Bayesian IRL이 Bayesian REX보다 우수했지만, 시연 수가 적을 때는 Bayesian REX가 경쟁력 있는 성능을 보였습니다.
* **고차원 시각 모방 학습 (Atari 게임):**
  * Beam Rider, Breakout, Enduro, Seaquest, Space Invaders 등 5가지 Atari 게임에서 Bayesian REX는 시연자의 성능을 뛰어넘는 결과를 달성했습니다.
  * T-REX(점 추정 방식) 및 GAIL과 비교했을 때, Bayesian REX는 5개 게임 중 3개에서 더 나은 성능을 보이거나 경쟁력 있는 성능을 유지했습니다.
* **고신뢰 정책 성능 경계:**
  * 부분적으로 훈련된 RL 정책(A-D)에 대한 고신뢰 성능 경계를 성공적으로 계산했으며, 이 경계는 실제 성능과 높은 상관관계를 보였습니다.
  * **보상 해킹 감지:** `No-Op` 정책(아무 행동도 하지 않고 게임이 끝날 때까지 기다리는)과 같은 보상 해킹 정책은 높은 기대 수익을 예측받았지만, 0.05-VaR(5% 신뢰 하한)에서는 매우 낮은 값(높은 위험)을 보여 보상 해킹을 효과적으로 감지할 수 있음을 입증했습니다.
  * `Breakout` 게임에서 MAP(Maximum A Posteriori) 및 `No-Op` 정책에 대한 추가 랭킹 시연을 포함하여 MCMC를 다시 실행한 결과, 이들 정책의 위험 프로필이 실제 선호도와 일치하도록 정확하게 조정됨을 보여주었습니다.
* **인간 시연 평가:**
  * `Beam Rider`와 `Enduro`에서 다양한 인간 시연(좋은 플레이, 나쁜 플레이, 적대적 플레이 등)에 대한 고신뢰 성능 경계를 계산하여, Bayesian REX가 이들 행동의 성능과 위험을 정확하게 순위화할 수 있음을 보였습니다.

## 🧠 Insights & Discussion

* **베이즈 추론의 효율성 증대:** Bayesian REX는 선호도 기반의 우도 함수와 사전 학습된 저차원 특징 임베딩을 통해 복잡한 시각 제어 태스크에서 베이즈 보상 추론의 계산적 장벽을 해결했습니다. 이는 기존 Bayesian IRL의 핵심 병목인 MDP 반복적 해결 문제를 우회함으로써 달성되었습니다.
* **안전성 및 불확실성 분석:** 보상 함수의 사후 분포를 얻음으로써, 임의의 정책에 대한 불확실성과 위험을 정량화하는 고신뢰 성능 경계를 제공할 수 있게 되었습니다. 이는 모방 학습의 안전성(safety)을 높이는 데 중요한 진전입니다.
* **보상 해킹 및 가치 불일치 감지:** 정책의 기대 수익률뿐만 아니라 위험 지표(예: VaR)를 함께 고려함으로써, 학습된 보상 함수를 악용하여 의도치 않은 행동(보상 해킹)을 하는 정책을 효과적으로 식별할 수 있습니다. 이는 AI 시스템의 가치 정렬(value alignment) 문제에 대한 실용적인 해결책을 제시합니다.
* **시연자보다 나은 성능:** 랭킹된 시연을 활용하여, 심지어 시연자가 비최적일 경우에도 더 나은 보상 함수를 학습하고 최적화된 정책이 시연자보다 우수한 성능을 달성할 수 있음을 보여주었습니다.
* **한계점:** 제안된 안전 경계는 좋은 특징 사전 학습, 빠른 MCMC 혼합(mixing), 정확한 시연 선호도 등 특정 가정에 의존합니다. 불확실성 정량화를 위한 다른 방법들(앙상블, MC 드롭아웃)과 비교했을 때, Bayesian REX가 전반적으로 더 강력하거나 경쟁력 있는 성능을 보였으나, 특정 게임에서는 다른 방법들이 더 우수할 수 있습니다.
* **향후 연구:** 탐색적 궤적을 이용한 잠재 특징 임베딩의 개선, 학습된 잠재 공간에서 관련 특징 누락 감지 방법 개발, 고신뢰 성능 경계를 활용한 안전한 정책 최적화 수행 등이 포함됩니다.

## 📌 TL;DR

**문제:** 기존 베이즈 역강화 학습(Bayesian IRL)은 고차원 모방 학습에서 계산 비용이 높고, 정책의 성능 불확실성을 제대로 정량화하기 어렵습니다.
**방법:** 본 논문은 `Bayesian REX`를 제안합니다. 이는 자기 지도 학습으로 저차원 특징 임베딩을 사전 학습하고, 시연 궤적에 대한 인간의 선호도 랭킹을 사용하여 보상 함수의 빠르고 효율적인 베이즈 추론을 수행합니다. 이 방법을 통해 계산 부담이 큰 MDP 솔버 없이도 베이즈 추론이 가능해집니다.
**결과:** Bayesian REX는 Atari 게임에서 최첨단 모방 학습 방식보다 우수하거나 경쟁력 있는 성능을 달성합니다. 또한, 임의의 평가 정책에 대한 효율적인 고신뢰 성능 하한을 제공하여 보상 해킹 정책과 같은 위험한 행동을 효과적으로 감지할 수 있음을 입증하여 모방 학습의 안전성과 신뢰성을 크게 향상시켰습니다.
