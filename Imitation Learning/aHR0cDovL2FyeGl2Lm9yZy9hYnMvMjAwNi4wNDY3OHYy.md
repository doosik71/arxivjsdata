# PRIMAL WASSERSTEIN IMITATION LEARNING

Robert Dadashi, Léonard Hussenot, Matthieu Geist, Olivier Pietquin

## 🧩 Problem to Solve

강화 학습(RL)은 보상 함수를 정의하기 어렵거나 희소할 때 적용하기 어렵습니다. 모방 학습(IL)은 고정된 수의 전문가 시연으로부터 정책을 학습하여 이러한 문제를 해결하려 합니다. 기존의 IRL(Inverse Reinforcement Learning) 방법은 낮은 샘플 효율성을 가지며, 최근의 적대적 IL(Adversarial IL) 방법은 minmax 최적화 문제로 인해 훈련 불안정성, 하이퍼파라미터 민감성, 그리고 낮은 샘플 효율성을 겪습니다. 특히, $f$-divergence 대신 Wasserstein 거리를 활용하는 GAN(Generative Adversarial Network) 기반 접근 방식들조차 듀얼 공식화로 인해 여전히 가중치 클리핑과 같은 실용적인 문제에 직면합니다. 본 논문은 이러한 기존 IL 방법의 한계를 극복하는 것을 목표로 합니다.

## ✨ Key Contributions

- **Primal Wasserstein Imitation Learning (PWIL) 제안:** 전문가와 에이전트의 상태-행동 분포 간 Wasserstein 거리의 Primal form을 최소화하는 개념적으로 단순한 새로운 IL 방법을 제안합니다.
- **오프라인 보상 함수:** 환경과의 상호작용을 통해 보상 함수를 학습하는 적대적 IL과 달리, 오프라인으로 도출되며 미세 조정(fine-tuning)이 거의 필요 없는 보상 함수를 제시합니다. 이는 minmax 최적화 문제를 방지하여 학습 안정성과 하이퍼파라미터에 대한 낮은 민감성을 보장합니다.
- **샘플 효율적인 전문가 행동 복구:** 다양한 MuJoCo 연속 제어 작업에서 에이전트와 전문가의 상호작용 측면에서 샘플 효율적인 방식으로 전문가 행동을 복구할 수 있음을 보여줍니다.
- **Wasserstein 거리를 통한 행동 일치 입증:** 일반적인 성능 프록시 대신 Wasserstein 거리를 사용하여 에이전트의 행동이 전문가의 행동과 일치함을 입증합니다.
- **극단적인 저데이터 환경에서의 성능:** 단일 (서브샘플링된) 시연으로 Humanoid를 달리게 할 수 있는 최초의 방법입니다.
- **시각 기반 관측으로의 확장:** MDP metric 정의가 직관적이지 않은 픽셀 기반 관측 환경(예: 문 열기 작업)에서도 적용 가능함을 입증합니다.

## 📎 Related Works

- **Adversarial IL:** GAIL (Ho & Ermon, 2016) 및 AIRL (Fu et al., 2018)과 같이 $f$-divergence를 사용하거나, Wasserstein GANs (Arjovsky et al., 2017)의 듀얼 공식을 활용한 IL 방법들이 있습니다. 그러나 이들은 minmax 최적화 문제와 구현상의 어려움이 따릅니다. PWIL은 Wasserstein 거리의 Primal form을 사용하여 이러한 방법들과 차별화됩니다.
- **Expert Support Estimation:** 전문가의 상태-행동 서포트를 추정하고 에이전트가 이 서포트 내에 머물도록 보상을 정의하는 연구들 (예: Soft Q Imitation Learning, Random Expert Distillation)이 있습니다. PWIL은 "pop-outs"을 통해 에이전트가 전체 서포트를 방문하도록 강제하여 접근 방식이 다릅니다.
- **Trajectory-Based IL:** Aytar et al. (2018)은 오프라인 임베딩 학습 후 보상을 강화하고, Peng et al. (2018)은 시간적 보상 함수를 강화합니다. PWIL은 시공간 제약 없이 더 넓은 환경에 일반화 가능합니다.
- **Offline Reward Estimation:** Boularias et al. (2011) 등의 접근 방식은 환경과의 상호작용 없이 보상 함수를 도출하지만, 보상 구조에 대한 강한 가정을 필요로 합니다. PWIL은 구조적 가정 없이 에피소드 이력에 따라 달라지는 비정상적 보상 함수를 오프라인으로 도출합니다.
- **MMD-IL:** Generative Moment Matching Imitation Learning (Kim & Park, 2018)은 MMD를 최소화하지만, 전체 롤아웃이 필요하며 PWIL의 greedy coupling과 같은 자연스러운 완화 방식이 없습니다.

## 🛠️ Methodology

PWIL의 핵심은 에이전트 정책 $\pi$에 의해 생성된 상태-행동 분포 $\hat{\rho}_{\pi}$와 전문가의 상태-행동 분포 $\hat{\rho}_{e}$ 간의 1-Wasserstein 거리를 최소화하는 것입니다. 이 최적화 문제는 다음과 같습니다:
$$ \inf*{\pi \in \Pi} W*{1}(\hat{\rho}_{\pi}, \hat{\rho}_{e}) = \inf*{\pi \in \Pi} \sum*{i=1}^{T} c*{i,\pi}^{\*} $$
여기서 $c*{i,\pi}^{_}$는 각 타임스텝 $i$에서 정책 $\pi$에 대한 최적 커플링 $\theta\_{\pi}^{_}$을 통해 계산된 비용입니다. $c_{i,\pi}^{*}$는 에피소드 전체 궤적에 대한 지식이 필요하므로, 이 문제를 해결하기 위해 온라인으로 계산 가능한 비용 함수를 도입합니다.

1. **Greedy Coupling 도입:** 최적 커플링 대신 **Greedy Coupling** $\theta_{\pi}^{g}$를 사용하여 Wasserstein 거리의 상한을 정의합니다. Greedy Coupling은 에이전트의 상태-행동 쌍 $(s_{i}^{\pi}, a_{i}^{\pi})$이 순차적으로 나타날 때마다, 남은 전문가 상태-행동 쌍 중에서 가장 가까운 곳으로 "흙(dirt)"을 이동시키는 전략입니다. 이는 각 타임스텝에서 비용을 계산할 수 있게 합니다.
2. **비용 함수 정의:** 각 타임스텝 $i$에서 비용 $c_{i,\pi}^{g}$는 다음과 같이 계산됩니다:
   $$ c*{i,\pi}^{g} = \sum*{j=1}^{D} d((s*{i}^{\pi}, a*{i}^{\pi}), (s*{j}^{e}, a*{j}^{e})) \theta*{i,j}^{g} $$
    여기서 $d((s,a),(s',a'))$는 상태-행동 쌍 간의 거리 함수입니다. 이 비용 $c*{i,\pi}^{g}$는 이전 상태-행동 이력에 따라 달라지므로 비정상적이지만, 오프라인에서 도출됩니다.
3. **보상 함수:** 비용 $c_{i,\pi}^{g}$로부터 다음 보상 함수 $r_{i}$를 정의합니다:
   $$ r*{i} = \alpha \exp\left(-\beta \frac{T}{\sqrt{|S|+|A|}} c*{i}\right) $$
    여기서 $\alpha$와 $\beta$는 튜닝 가능한 하이퍼파라미터이며, $\frac{T}{\sqrt{|S|+|A|}}$는 상태 및 행동 공간의 차원과 시간 범위에 대한 정규화 요소입니다.
4. **MDP 거리 정의:**
   - **MuJoCo 로코모션 작업:** 관측 및 행동의 연결에 대해 전문가 시연의 역 표준 편차로 각 차원에 가중치를 부여한 표준화된 유클리드 거리 (L2 거리)를 사용합니다.
   - **시각 기반 관측 (문 열기):** Temporal Cycle-Consistency Learning (TCC)을 통해 시연 프레임의 임베딩을 오프라인으로 학습한 후, 이 임베딩 공간에서 L2 거리를 사용합니다.
5. **학습 절차 (Algorithm 1):** D4PG (또는 SAC)와 같은 직접 RL 에이전트는 위에서 정의된 보상 함수를 최대화하도록 정책 $\pi_{A}$를 학습합니다. 리플레이 버퍼는 전문가 상태-행동 쌍으로 미리 채워집니다.

## 📊 Results

- **MuJoCo 로코모션 작업 성능:** PWIL은 Hopper, Ant, HalfCheetah, Humanoid에서 최첨단 DAC보다 향상된 최종 성능을 보였습니다. Walker2d에서는 유사한 성능을, Reacher에서는 낮은 성능을 보였습니다.
- **Humanoid 학습 성공:** DAC가 저조한 성능을 보인 Humanoid 환경에서 PWIL은 단 한 번의 시연만으로도 평균 7000점 이상의 거의 최적의 성능을 달성하여 에이전트가 실제로 달릴 수 있게 하는 최초의 방법입니다 (5000점은 '서 있는 것'에 해당).
- **샘플 효율성:** 샘플 효율성 측면에서는 DAC가 HalfCheetah, Ant, Reacher에서 PWIL보다 우수하거나 유사했지만, DAC의 보상 함수는 많은 환경 상호작용을 통해 신중하게 튜닝해야 합니다. 반면 PWIL은 하이퍼파라미터가 적어 실제 튜닝 비용이 낮습니다.
- **Wasserstein 거리 감소:** PWIL은 학습 과정 동안 평가 정책과 전문가 시연 간의 Wasserstein 거리를 DAC보다 더 작게 감소시켰습니다. 제안된 상한이 "경험적으로 타이트함"이 입증되어 Greedy Coupling의 유효성이 검증되었습니다.
- **Ablation Study:**
  - **PWIL-state (액션 정보 미사용):** 액션 정보 없이도 모든 환경에서 비자명한 행동을 복구할 수 있었습니다.
  - **PWIL-L2 (비표준화 유클리드 거리):** Hopper-v2를 제외한 모든 환경에서 성능이 크게 저하되어 MDP metric의 품질이 중요함을 보여주었습니다.
  - **PWIL-nofill (리플레이 버퍼 미초기화):** Walker와 Ant에서 성능이 크게 저하되어 리플레이 버퍼를 전문가 전환으로 미리 채우는 것이 중요함을 나타냈습니다.
  - **PWIL-support (무한 용량 홀):** Ant, Walker, HalfCheetah에서 성능이 크게 저하되어 "pop-outs" 전략이 전문가 행동 복구에 필수적임을 강조했습니다.
- **시각 기반 환경 (Door Opening):** TCC로 학습된 임베딩 공간에서 PWIL 보상을 사용하여 문 열기 작업을 성공적으로 해결하여 시각 기반 환경으로의 확장 가능성을 입증했습니다.

## 🧠 Insights & Discussion

PWIL은 에이전트와 전문가의 상태-행동 분포 간 Wasserstein 거리의 Primal form을 최소화하는 개념적으로 단순하고 강력한 모방 학습 방법입니다. 이 방법의 가장 중요한 통찰은 오프라인에서 도출되는 보상 함수를 통해 기존 적대적 IL 방법의 고질적인 문제인 훈련 불안정성과 하이퍼파라미터 민감성을 효과적으로 회피한다는 점입니다. 단 두 개의 하이퍼파라미터만으로도 극도로 적은 시연 데이터(단일 시연)에서도 근접 전문가 성능을 달성할 수 있어 매우 데이터 효율적입니다.

특히, 일반적인 작업 보상(task reward)을 알 수 없는 실제 모방 학습 환경에서 PWIL은 에이전트와 전문가 행동 간의 직접적인 유사성 척도(Wasserstein 거리)를 제공하여 평가에 유용합니다. 또한, MDP metric을 학습해야 하는 어려운 시각 기반 설정까지 확장 가능함을 보여주었습니다.

하지만 PWIL의 성능은 MDP metric의 품질에 민감하며, Greedy Coupling에서 "pop-outs" 전략이 전문가 행동을 정확히 복구하는 데 필수적이라는 한계점도 발견되었습니다.

## 📌 TL;DR

PWIL은 기존 적대적 모방 학습의 불안정성과 비효율성을 해결하기 위해, 전문가와 에이전트의 상태-행동 분포 간 Wasserstein 거리의 Primal form을 최소화하는 새로운 모방 학습 방법이다. Greedy Coupling을 통해 오프라인에서 도출되는 보상 함수는 적은 하이퍼파라미터로도 높은 샘플 효율성을 보이며, 특히 단일 시연으로 Humanoid를 학습시키는 등 어려운 연속 제어 작업에서 뛰어난 성능을 입증하고 시각 기반 환경으로도 확장 가능하다.
