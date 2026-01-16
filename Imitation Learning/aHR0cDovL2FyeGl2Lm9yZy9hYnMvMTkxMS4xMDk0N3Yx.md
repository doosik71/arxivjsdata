# STATE ALIGNMENT-BASED IMITATION LEARNING

Fangchen Liu, Zhan Ling, Tongzhou Mu, Hao Su

## 🧩 Problem to Solve

대부분의 기존 모방 학습(Imitation Learning, IL) 방법론은 전문가(expert)와 모방자(imitator)가 동일한 동역학 모델(dynamics model)을 공유한다는 강한 가정을 기반으로 합니다. 이 가정은 동일한 액션 공간을 가지고, 주어진 상태-액션 쌍에 대해 다음 상태가 확률적으로 동일하다는 것을 의미합니다. 하지만 저속 로봇이 고속 로봇의 움직임을 모방하는 것과 같이, 실제 시나리오에서는 이러한 가정이 성립하지 않아 기존 방법(행동 복제, 역 강화 학습 기반 GAIL 등)이 실패하는 한계가 있습니다. 본 연구는 동역학 모델이 다른 상황에서도 모방자가 전문가의 상태 시퀀스를 최대한 따르도록 학습하는 모방 학습 문제를 해결하고자 합니다. 기존 방법들은 동역학 불일치(dynamics mismatch)와 궤적 이탈 시의 자체 수정(deviation correction) 기능을 학습하지 못합니다.

## ✨ Key Contributions

- 전문가와 모방자의 동역학이 다른 모방 학습 문제에서 **상태 정렬(state alignment) 기반 방법**을 제안합니다.
- $\beta$-VAE를 기반으로 하는 **지역적 상태 정렬(local state alignment) 방법**과 Wasserstein 거리를 기반으로 하는 **전역적 상태 정렬(global state alignment) 방법**을 제안합니다.
- 지역적 정렬과 전역적 정렬 구성요소를 정규화된 정책 업데이트 목적 함수를 통해 강화 학습(Reinforcement Learning, RL) 프레임워크에 결합합니다.
- 제안된 방법이 표준 모방 학습 설정뿐만 아니라, 전문가와 모방자의 동역학 모델이 다른 더욱 어려운 설정에서도 기존 방법론보다 우수함을 MuJoCo 환경 실험을 통해 입증합니다.

## 📎 Related Works

- **행동 복제(Behavioral Cloning, BC)**: 감독 학습(supervised learning) 방식으로 정책을 학습하지만, 누적 오류(compounding errors) 문제에 취약합니다 (DAGGER 등으로 완화).
- **역 강화 학습(Inverse Reinforcement Learning, IRL)**: 주어진 데모 궤적을 유도하는 보상 함수를 찾아냅니다.
- **GAN 기반 IRL**: GAIL (Ho & Ermon, 2016) 및 그 변형들은 GAN(Generative Adversarial Network) 기반 보상을 사용하여 전문가와 모방자의 상태-액션 분포를 정렬합니다. 상태-액션 쌍 대신 상태 분포 정렬에 초점을 맞춘 연구들도 있습니다 (Torabi et al., Sun et al., Schroecker & Isbell).
- **관측치 기반 학습(Learning from Observation)**: 액션 없이 관측치(예: 비디오)만으로 학습하는 접근법으로, 상태에 보상을 정의하고 IRL을 사용하거나 (Aytar et al., Liu et al., Peng et al.), 역 모델(inverse model)이나 잠재 액션(latent actions)을 학습하여 액션을 복구합니다 (Torabi et al., Edwards et al.).
- 본 연구는 기존 BC 및 IRL의 장점을 결합하여, 지역적 상태 전이 정렬(local state transition alignment)과 전역적 상태 분포 정렬(global state distribution matching)을 새로운 프레임워크로 통합합니다.

## 🛠️ Methodology

본 논문은 지역적 및 전역적 관점에서의 상태 정렬을 기반으로 한 모방 학습 방법을 제안합니다.

1. **지역적 정렬 (Local Alignment by State Predictive VAE)**:

   - 목표: 데모의 상태 전이를 최대한 따르며, 모방자가 궤적에서 벗어났을 때 데모 궤적으로 되돌아올 수 있도록 합니다.
   - 방법: $\beta$-VAE를 사용하여 다음 방문할 상태($s_{t+1}$)를 예측합니다. 일반적인 신경망 대신 VAE를 사용하는 이유는 VAE가 아웃라이어에 강하고, 잠재 공간의 확률적 샘플링으로 인해 인접한 데이터 포인트에 대해 유사한 예측을 제공하여 자체 수정(self-correctable) 능력을 부여하기 때문입니다.
   - 액션 복구: 예측된 다음 상태 $s_{t+1}$과 현재 상태 $s_t$를 바탕으로, 학습된 역 동역학 모델 $g_{inv}(s_t, s_{t+1})$을 사용하여 해당 액션 $a_t$를 복구합니다.

2. **전역적 정렬 (Global Alignment by Wasserstein Distance)**:

   - 목표: 데모와 모방자의 상태 방문 분포 간의 차이를 최소화하는 전역적 제약을 제공합니다. 이는 지역적 정렬만으로는 해결할 수 없는, 데모에서 멀리 떨어진 상태로 이탈했을 때의 상황을 다룹니다.
   - 방법: IRL 접근 방식을 사용하여 강화 학습 문제를 설정합니다. 보상은 전문가와 모방자 궤적의 Wasserstein 거리를 최소화하도록 설계됩니다.
   - Wasserstein 거리 계산: Kantorovich-Rubenstein 이중성(duality)을 사용하여 계산되며, $\phi$ 함수(WGAN의 판별자 역할)는 $W(\tau_e, \tau) = \sup_{\phi \in L_1} E_{s \sim \tau_e}[\phi(s)] - E_{s \sim \tau}[\phi(s)]$를 최대화하도록 학습됩니다.
   - 보상 함수: $r(s_i, s_{i+1}) = \frac{1}{T}[\phi(s_{i+1}) - E_{s \sim \tau_e}[\phi(s)]]$로 정의하여, 데모에서 확률이 높은 상태를 방문하도록 장려합니다. 누적 보상 최대화는 Wasserstein 거리 최소화와 동일합니다.

3. **정규화된 PPO 정책 업데이트 목적 함수 (Regularized PPO Policy Update Objective)**:

   - 지역적 정렬과 전역적 정렬을 통합하기 위해 PPO(Proximal Policy Optimization) 알고리즘에 정규화된 정책 업데이트 목표를 사용합니다.
   - 목표 함수: $J(\pi_\theta) = L_{CLIP}(\theta) - \lambda D_{KL}(\pi_\theta(\cdot|s_t) \parallel p_a)$
   - $L_{CLIP}(\theta)$: PPO의 표준 클립된 대리 목적 함수로, Wasserstein 거리 기반 보상으로부터 계산된 Advantage $\hat{A}_t$를 사용하여 전역적 정렬을 반영합니다.
   - $D_{KL}$ 항: 정책 $\pi_\theta$가 VAE와 역 동역학 모델에서 파생된 정책 사전(policy prior) $p_a$에 가깝게 유지되도록 하는 정규화자(local alignment) 역할을 합니다. $p_a(a_t|s_t) \propto \exp\left(-\frac{\left\|g_{inv}(s_t, f(s_t)) - a_t\right\|^2}{\sigma^2}\right)$와 같이 정의됩니다.

4. **사전 학습 (Pre-training)**:
   - 전문가 궤적을 사용하여 상태 예측 $\beta$-VAE와 역 동역학 모델을 사전 학습합니다.
   - 이로부터 얻은 정책 사전 $p_a$를 사용하여 PPO의 가우시안 정책 $\pi$를 초기화합니다.

## 📊 Results

- **동역학이 다른 에이전트 간의 모방 학습**:
  - **물리적/기하학적 특성 수정**: Ant 및 Swimmer 환경에서 밀도나 신체 부위 길이를 변경하여 Heavy/Light/Disabled Ant/Swimmer 에이전트를 생성했습니다. 본 논문의 SAIL 방법은 모든 6개 환경에서 BC, GAIL, AIRL보다 일관되게 우수하고 안정적인 성능을 보였습니다. 특히 GAIL은 동역학 차이에 가장 민감하게 반응했으며, AIRL은 Swimmer 환경에서 일부 경쟁력을 보였습니다. BC는 몇몇 환경에서 실패하거나 비효율적인 움직임을 보였습니다.
  - **이종 액션 동역학 (Point to Ant in a maze)**: MuJoCo에서 완전히 다른 유형의 에이전트(Point mass와 Ant) 간의 미로 탐색 모방 학습을 수행했습니다. SAIL은 Point 로봇의 데모를 통해 Ant가 미로의 목표에 도달하는 데 0.8의 성공률을 달성하여, 이종 에이전트 간에도 공유되는 상태 공간(예: (x,y) 좌표)에 대한 모방이 가능함을 보였습니다.
- **동역학이 동일한 에이전트 간의 모방 학습 (표준 IL 설정)**:
  - Swimmer, Hopper, Walker, Ant, HalfCheetah, Humanoid 등 6개의 MuJoCo 제어 작업에서 평가했습니다.
  - SAIL은 기존 BC, GAIL, AIRL과 유사하거나 더 나은 성능을 달성했습니다. 특히 데이터 증강 역할을 하는 VAE의 견고성 덕분에 Hopper-v2(데모 10개) 및 HalfCheetah-v2에서 강점을 보였습니다.
- **어블레이션 스터디 (Ablation Study)**:
  - **$\beta$-VAE의 $\beta$ 계수**: $\beta \in [0.01, 0.1]$ 범위에서 성능이 가장 좋았으며, 특히 동역학이 다른 환경에서는 너무 작은 $\beta$가 데모 데이터에 과적합되어 성능 저하를 일으킬 수 있음을 발견했습니다. VAE가 일반 MLP보다 모든 설정에서 우수했습니다.
  - **액션 예측 $\beta$-VAE**: 다음 상태를 예측하는 VAE가 액션을 예측하는 VAE보다 우수했습니다. 이는 VAE의 강점이 상태 기반 접근 방식에서 비롯됨을 입증합니다.
  - **Wasserstein 거리 및 KL 정규화 효과**: 두 구성 요소 모두 필수적입니다. Wasserstein 거리 단독으로는 연속적인 상태에 대한 제약이 부족하여 성능이 저조했고, KL 정규화 단독으로는 에이전트가 데모에서 멀리 이탈했을 때 VAE가 상태를 외삽(extrapolate)하지 못해 실패하는 경향이 있었습니다. 두 가지를 결합했을 때 최고의 성능을 보였습니다.

## 🧠 Insights & Discussion

- SAIL은 지역적($\beta$-VAE를 통한 다음 상태 예측 및 이탈 수정) 및 전역적(Wasserstein 거리를 이용한 상태 분포 매칭) 관점의 상태 정렬을 결합하여 동역학 불일치 문제를 효과적으로 해결합니다.
- 액션이 아닌 상태를 모방하는 접근 방식은 동역학이 다른 시나리오에서 핵심적인 유연성을 제공합니다.
- $\beta$-VAE의 잠재 공간에서의 확률적 샘플링은 에이전트가 궤적에서 벗어났을 때 데모 궤적으로 스스로 되돌아오게 하는 강력한 자기 수정 및 견고성(robustness)을 제공합니다.
- 전역적 Wasserstein 거리 제약은 에이전트가 데모와 멀리 떨어진 상태로 벗어날 때 이를 데모 상태로 다시 이끌어주는 보상을 제공하여, 지역적 정렬이 한계를 가질 수 있는 상황을 보완합니다.
- PPO의 정책 업데이트에 KL 발산 기반 정규화자를 추가함으로써, 지역적 및 전역적 정렬이 단일 프레임워크 내에서 시너지를 발휘하여 행동 복제와 역 강화 학습의 장점을 모두 취합니다.
- Point mass와 Ant와 같이 작동 방식이 완전히 다른 에이전트 간의 모방 학습에서도 성공적인 결과를 보여, SAIL의 유연성과 실용성을 입증했습니다.

## 📌 TL;DR

- **문제**: 기존 모방 학습은 전문가와 모방자의 동역학 모델이 동일하다고 가정하여, 동역학이 다른 실제 시나리오에서는 작동하지 않습니다.
- **방법**: 본 논문은 동역학이 다른 에이전트 간 모방 학습을 위해 **상태 정렬 기반 모방 학습 (SAIL)**을 제안합니다. SAIL은 $\beta$-VAE를 이용한 **지역적 상태 예측**으로 궤적 이탈 시 자체 수정을 가능하게 하고, Wasserstein 거리를 이용한 **전역적 상태 분포 매칭**으로 전체적인 궤적을 전문가와 정렬합니다. 이 두 가지는 PPO의 정책 업데이트 목표에 KL 발산 기반 정규화자로 통합됩니다.
- **결과**: SAIL은 동역학이 다른 다양한 MuJoCo 환경(물리적 특성 변경, 이종 에이전트)에서 기존 BC, GAIL, AIRL보다 훨씬 우수하고 안정적인 성능을 보였으며, 표준 모방 학습 설정에서도 경쟁력 있는 결과를 달성했습니다. $\beta$-VAE의 상태 예측과 Wasserstein 거리 및 KL 정규화의 상호 보완적인 역할이 성공의 핵심임을 입증했습니다.
