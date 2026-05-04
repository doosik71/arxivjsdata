# Imitating Opponent to Win: Adversarial Policy Imitation Learning in Two-player Competitive Games

The Viet Bui, Tien Mai, Thanh H.Nguyen (2022)

## 🧩 Problem to Solve

본 논문은 심층 강화학습(Deep Reinforcement Learning, RL) 에이전트의 취약성을 공격하는 Adversarial Policy(적대적 정책)의 효율성을 높이는 문제를 다룬다. 기존의 적대적 정책 학습 방식은 공격자 에이전트가 피해자(Victim) 에이전트와 직접 상호작용하며 얻은 경험만을 바탕으로 학습된다. 

이러한 접근 방식의 핵심적인 한계는 과거의 상호작용 데이터에 기반한 지식이 피해자 에이전트의 탐색되지 않은 정책 영역(unexplored policy regions)으로 제대로 일반화되지 않는다는 점이다. 결과적으로 학습된 적대적 정책은 특정 상황에서는 효과적일 수 있으나, 전반적인 공격 효율성이 떨어진다는 문제가 있다. 본 연구의 목표는 피해자 에이전트의 내재적 특성을 파악하여 의도를 예측함으로써, 보지 못한 상태에서도 효과적으로 대응할 수 있는 강력한 적대적 정책 학습 알고리즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 피해자 에이전트를 모방하는 **Imitator(모방자)**를 도입하는 것이다. 

1. **의도 예측 기반 공격**: Imitator가 피해자의 정책을 학습하여 다음 행동을 예측하게 하고, 적대적 에이전트가 이 예측 정보를 입력으로 받아 최적의 공격 행동을 결정하게 한다. 이를 통해 피해자의 탐색되지 않은 정책 영역에서도 효과적인 공격이 가능해진다.
2. **Enhanced Imitation Learning**: 단순히 모방하는 것을 넘어, 적대적 에이전트의 가치 함수(Value Function)의 반대 방향을 모방 목표에 통합함으로써, Imitator가 피해자를 닮으면서도 동시에 공격자에게는 적대적인(즉, 더 까다로운) 형태가 되도록 유도한다.
3. **이론적 보장**: 적대적 정책과 모방 정책 간의 상호 의존성을 분석하고, 적대적 정책이 안정화되었을 때 모방 정책이 원하는 수준으로 수렴함을 보장하는 수학적 바운드(provable bound)를 제공한다.

## 📎 Related Works

기존의 RL 공격 연구는 주로 피해자의 관측값(Observation)이나 행동(Action)에 섭동(Perturbation)을 가해 잘못된 행동을 유도하는 방식에 집중했다. 그러나 이는 현실 세계(예: 자율주행)에서 공격자가 피해자의 입력 데이터에 직접 개입하기 어렵다는 점에서 실용성이 떨어진다.

이에 대한 대안으로 최근에는 동일한 환경에서 플레이하는 적대적 에이전트를 학습시키는 Adversarial Policy 접근법(ADRL, APL 등)이 제시되었다. 하지만 기존의 ADRL이나 APL은 피해자와의 직접적인 상호작용 경험에만 의존하여 학습하므로, 피해자의 정책 분포가 넓은 경우 일반화 성능이 낮다는 한계가 있다. 본 논문은 Imitation Learning을 결합하여 이 일반화 문제를 해결함으로써 기존 접근 방식과 차별화를 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 연구는 피해자 에이전트($\pi_\nu$), 적대적 에이전트($\pi_{\text{adv}}$), 그리고 모방자(Imitator, $e\pi_\nu$)의 세 구성 요소로 이루어진 프레임워크를 제안한다. 

1. **Imitator**는 피해자의 궤적(Trajectory)을 관찰하여 피해자의 행동을 예측하는 정책 $e\pi_\nu$를 학습한다.
2. **Adversary**는 현재 상태 $s_t$와 Imitator가 예측한 피해자의 다음 행동 $ea_{\nu t}$를 모두 입력으로 받아 자신의 행동 $a_{\text{adv } t}$를 결정한다. 즉, 적대적 정책은 $\pi_{\text{adv}}(a_{\text{adv } t} | s_t, ea_{\nu t})$ 형태로 정의된다.

### 모방 학습 방법 (Enhanced GAIL)
기본적으로 Generative Adversarial Imitation Learning(GAIL)을 기반으로 하며, 판별자(Discriminator) $D$와 생성자(Generator) $e\pi_\nu$가 서로 경쟁하며 피해자의 정책을 학습한다. 본 논문에서는 이를 강화한 **Enhanced Imitation Learning** 모델을 제안하며, 그 목적 함수는 다음과 같다.

$$\max_{e\pi_\nu^\psi} \min_{D_\phi} \left( \mathcal{F}(e\pi_\nu^\psi, D_\phi) - V_{\pi_{\text{adv}}}(s_0 | q_{\text{adv}}(e\pi_\nu^\psi)) \right)$$

여기서 $\mathcal{F}$는 표준 GAIL의 목적 함수이며, $V_{\pi_{\text{adv}}}$는 적대적 에이전트의 가치 함수이다. 이는 Imitator가 피해자를 모방함과 동시에 적대적 에이전트의 보상을 최소화하는 방향으로 학습되게 하여, 결과적으로 더 강력한 모방자를 생성한다.

### 적대적 정책 학습 (Adversarial Policy Training)
적대적 에이전트는 자신의 보상을 최대화하고 피해자의 보상을 최소화하는 방향으로 학습된다. 목적 함수는 다음과 같이 정의된다.

$$\max_{\pi_{\text{adv}}} \left( V_{\pi_{\text{adv}}}(s_0) - V_{\pi_\nu}(s_0 | q_\nu(\pi_{\text{adv}})) \right)$$

이 문제는 다음과 같은 차분 보상(Differentiated Reward) $\Delta r(s_t) = r_{\text{adv}}(s_t) - r_\nu(s_t)$를 최대화하는 표준 강화학습 문제로 변환될 수 있다.

$$\max_{\pi_{\text{adv}}} \mathbb{E}_{\tau \sim \pi_{\text{adv}}} \left[ \sum_{t=0}^{\infty} \gamma^t \Delta r(s_t) \right]$$

### 학습 절차
1. 적대적 에이전트와 피해자 에이전트가 상호작용하며 데이터를 수집한다.
2. 수집된 궤적을 사용하여 Imitator의 판별자 $D$를 업데이트한다.
3. 판별자의 피드백과 적대적 에이전트의 가치 함수를 이용하여 Imitator의 정책 $e\pi_\nu$를 업데이트한다.
4. 업데이트된 Imitator의 예측 행동 $ea_\nu$를 상태 정보에 추가하여 적대적 정책 $\pi_{\text{adv}}$를 PPO(Proximal Policy Optimization) 등을 통해 업데이트한다.

## 📊 Results

### 실험 설정
- **환경**: MuJoCo 기반의 4가지 경쟁 게임 (Kick And Defend, You Shall Not Pass, Sumo Humans, Sumo Ants).
- **비교 대상**: Baseline agents, ADRL, APL.
- **지표**: 승률(Winning Rate), 승률 및 무승부 합계(Winning + Tie Rate).

### 정량적 결과
- **승률 향상**: 제안된 APIL 및 E-APIL 방식은 대부분의 환경에서 기존의 ADRL, APL보다 유의미하게 높은 승률을 기록했다. 특히 Sumo Humans 환경에서 기존 방식 대비 압도적인 성능 향상을 보였다.
- **E-APIL vs APIL**: 적대적 가치 함수를 통합한 E-APIL이 일반 APIL보다 Kick And Defend와 You Shall Not Pass에서 더 높은 성능을 보였으며, 이는 강화된 모방자가 더 정교한 공격을 가능하게 함을 시사한다.
- **특이 사항**: Sumo Ants 환경에서는 승률보다 무승부 비율이 높게 나타났는데, 이는 환경 특성상 피해자가 단순히 바닥으로 뛰어내려 무승부를 유도하는 전략이 효율적이기 때문으로 분석된다.

### 추가 분석
- **Blinding 실험**: 적대적 에이전트의 관측값에서 피해자 정보를 제거(zero-out)하고 오직 Imitator의 예측값만 제공했을 때도, 타 방법론 대비 높은 승률을 유지했다. 이는 Imitator의 예측 능력이 공격의 핵심임을 증명한다.
- **피해자 회복력(Resiliency) 향상**: E-APIL로 생성된 강력한 적대적 에이전트를 상대로 피해자를 재학습시켰을 때, 다른 공격 방식으로 학습된 피해자보다 훨씬 더 높은 회복력을 갖게 되었으며, 이 회복력은 다른 종류의 공격자에게도 전이(Transfer)되었다.

## 🧠 Insights & Discussion

### 강점
본 연구는 단순히 과거 데이터를 학습하는 것을 넘어 **'상대방의 의도를 예측'**하는 메커니즘을 적대적 학습에 도입했다는 점이 매우 강력하다. 특히 Imitator를 통해 피해자의 정책 분포를 더 넓게 파악함으로써, 기존 방식들이 도달하지 못한 최적의 공격 지점을 찾아낼 수 있었다.

### 한계 및 논의
- **상호 의존성 문제**: Imitator는 적대적 에이전트의 정책에 따라 변하는 환경 역학(Dynamics) 속에서 학습된다. 논문은 이에 대한 이론적 바운드를 제시하여 수렴성을 논했지만, 실제 구현 시 두 정책의 업데이트 속도(Learning rate) 조절이 학습 안정성에 큰 영향을 미칠 가능성이 있다.
- **계산 복잡도**: Imitator와 Discriminator를 동시에 학습시켜야 하므로, 표준 적대적 학습보다 더 많은 계산 자원과 시간이 소요될 수 있다.

### 비판적 해석
본 논문은 피해자 에이전트의 정책이 고정되어 있다는 가정을 사용했다. 하지만 현실에서는 피해자 또한 실시간으로 학습하며 대응할 가능성이 높다. 비록 논문 후반부에 피해자 정책이 변하는 경우(Unfixed Victim's Policy)에 대한 worst-case 분석을 포함했으나, 두 에이전트가 동시에 진화하는 완전한 동적 게임 환경에서의 안정성에 대해서는 추가적인 연구가 필요해 보인다.

## 📌 TL;DR

본 논문은 적대적 에이전트가 피해자의 행동을 예측하는 **Imitator(모방자)**를 함께 학습함으로써 공격 성능을 극대화하는 **APIL/E-APIL** 알고리즘을 제안한다. Imitator는 피해자의 정책을 모방함과 동시에 공격자에게 불리한 방향으로 강화되어, 적대적 에이전트가 피해자의 취약점을 더 정확히 공략하게 돕는다. 실험 결과, MuJoCo 경쟁 환경에서 기존 SOTA 방법론들을 상회하는 승률을 기록했으며, 이를 통해 학습된 피해자 에이전트의 강건성(Robustness) 또한 크게 향상시킬 수 있음을 입증했다. 이는 향후 강화학습 에이전트의 보안 테스트 및 강건한 정책 설계에 중요한 기여를 할 것으로 보인다.