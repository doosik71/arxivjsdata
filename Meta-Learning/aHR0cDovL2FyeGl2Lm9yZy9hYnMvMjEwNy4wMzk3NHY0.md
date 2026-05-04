# Offline Meta-Reinforcement Learning with Online Self-Supervision

Vitchyr H. Pong, Ashvin Nair, Laura Smith, Catherine Huang, Sergey Levine (2022)

## 🧩 Problem to Solve

본 논문은 Meta-Reinforcement Learning (Meta-RL)의 학습 비용 문제를 해결하기 위해 Offline Meta-RL을 적용하는 과정에서 발생하는 특유의 문제점을 다룬다. Meta-RL은 적은 양의 데이터로 새로운 작업에 빠르게 적응할 수 있는 정책을 학습시키지만, Meta-training 단계에서 방대한 양의 온라인 데이터가 필요하다는 단점이 있다. 이를 해결하기 위해 정적인 데이터셋만을 사용하는 Offline Meta-RL이 제안되었으나, 저자들은 여기서 **z-space에서의 분포 변화(Distribution Shift in z-space)**라는 핵심적인 문제를 발견하였다.

Offline Meta-RL에서 에이전트는 사전에 수집된 Behavior Policy ($\pi_\beta$)의 데이터를 통해 적응 절차를 학습한다. 그러나 실제 테스트 단계에서 에이전트는 자신이 학습한 Exploration Policy ($\pi_\theta$)를 사용하여 데이터를 수집하고 적응한다. 이때 $\pi_\theta$가 수집한 데이터의 분포가 $\pi_\beta$의 데이터 분포와 체계적으로 다르기 때문에, 인코더가 생성하는 컨텍스트 변수 $z$의 분포가 달라지게 된다. 결과적으로 Offline 데이터로만 학습된 정책은 자신이 직접 수집한 데이터로부터 유도된 $z$ 값에 익숙하지 않아, 적응 성능이 크게 저하되는 현상이 발생한다.

본 논문의 목표는 보상 라벨이 없는 추가적인 온라인 상호작용(unsupervised online data)을 활용하여 이러한 분포 변화를 완화하고, 온라인 Meta-RL에 근접하는 적응 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 아이디어는 **보상 라벨이 없는 온라인 데이터를 수집하고, 학습된 Reward Decoder를 통해 이를 스스로 라벨링하여 Meta-training을 지속하는 Semi-supervised 학습 체계**를 구축하는 것이다.

핵심 기여 사항은 다음과 같다:
1. Offline Meta-RL에서 발생하는 $z$-space의 분포 변화 문제를 정의하고, 실험적 증거를 통해 이것이 성능 저하의 주된 원인임을 밝혀냈다.
2. Reward-labeled offline 데이터를 사용하여 보상 함수를 생성하는 모델을 학습시키고, 이를 통해 라벨이 없는 온라인 데이터에 가상 보상을 부여하는 **SMAC (Semi-Supervised Meta Actor-Critic)** 알고리즘을 제안하였다.
3. 제안된 방법론이 단순한 Offline Meta-RL보다 월등하며, 보상 라벨이 완전히 제공되는 Online Oracle 성능에 근접함을 입증하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하며 SMAC의 차별점을 제시한다.

- **Online Meta-RL (PEARL 등):** 빠른 적응이 가능하지만 매 에피소드마다 보상 라벨이 필요하며 학습 비용이 매우 높다. SMAC은 오프라인 데이터와 보상 없는 온라인 데이터를 사용함으로써 이 비용을 획기적으로 줄인다.
- **Offline RL (AWAC 등):** 데이터 분포 변화 문제를 해결하기 위해 정책을 데이터셋에 가깝게 제약하는 방식을 사용한다. SMAC은 AWAC의 아이디어를 차용하여 Offline 단계의 과대평가(overestimation) 문제를 해결한다.
- **Offline Meta-RL (MACAW, BOReL):** 기존의 오프라인 메타 학습 방식들은 정적인 데이터셋만을 사용한다. SMAC은 이들과 달리 'Self-supervised online fine-tuning' 단계를 추가하여 $z$-space의 분포 변화를 직접적으로 해결한다.
- **Unsupervised Meta-Learning:** 스스로 보상을 생성하는 방식이 존재하지만, SMAC은 기존의 reward-labeled offline 데이터셋이 존재한다는 가정하에 이를 활용하여 더 정확한 보상 생성기를 학습시킨다는 점에서 차이가 있다.

## 🛠️ Methodology

SMAC은 크게 두 단계, **Offline Meta-Training**과 **Self-Supervised Online Meta-Training**으로 구성된다.

### 1. Offline Meta-Training
먼저, 제공된 오프라인 데이터셋 $\mathcal{D}$를 사용하여 기본적인 Meta-RL 구조를 학습한다. 

- **Critic 학습:** 벨만 오차(Bellman error)를 최소화하도록 학습하며, 수식은 다음과 같다.
$$L_{\text{critic}}(w) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}_i, z \sim q_{\phi_e}(z|h), a' \sim \pi_\theta(a'|s',z)} \left[ (Q_w(s,a,z) - (r + \gamma \bar{Q}_w(s',a',z)))^2 \right]$$
- **Actor 학습:** Offline RL의 고질적인 문제인 부트스트래핑 에러를 방지하기 위해, AWAC 방식을 도입하여 정책이 데이터셋의 행동 범위 내에 머물도록 제약한다.
$$L_{\text{actor}}(\theta) = -\mathbb{E}_{s,a,s' \sim \mathcal{D}, z \sim q_{\phi_e}(z|h)} \left[ \log \pi_\theta(a|s) \times \exp \left( \frac{Q(s,a,z) - V(s',z)}{\lambda} \right) \right]$$
- **Reward Decoder 학습:** 컨텍스트 $z$를 입력받아 보상 $r$을 복원하는 $r_{\phi_d}(s, a, z)$를 학습시킨다.
$$L_{\text{reward}}(\phi_d, \phi_e, h, z) = \sum_{(s,a,r) \in h} \| r - r_{\phi_d}(s,a,z) \|^2_2 + D_{KL}(q_{\phi_e}(\cdot|h) || p_z(\cdot))$$
여기서 $D_{KL}$ 항은 잠재 공간 $z$에 정보 병목(information bottleneck)을 제공하여 $z$가 의미 있는 변수가 되도록 정규화한다.

### 2. Self-Supervised Online Meta-Training
오프라인 학습 후, 에이전트는 환경과 직접 상호작용하며 데이터를 수집한다. 이때 실제 보상 라벨은 제공되지 않는다.

- **데이터 수집:** 현재 정책 $\pi_\theta$를 사용하여 궤적 $\tau$를 수집한다.
- **가상 보상 부여:** 오프라인 데이터셋에서 샘플링한 $h_{\text{offline}}$을 통해 $z \sim q_{\phi_e}(z|h_{\text{offline}})$를 얻고, 앞서 학습한 Reward Decoder를 이용해 수집한 데이터에 보상을 부여한다.
$$r_{\text{generated}} = r_{\phi_d}(s, a, z), \quad \text{where } z \sim q_{\phi_e}(z|h_{\text{offline}})$$
- **업데이트:** 이렇게 생성된 가상 보상을 포함한 데이터를 버퍼에 추가하고, Actor와 Critic을 다시 학습시킨다. 단, 이 단계에서는 ground-truth 보상이 없으므로 Reward Decoder와 Encoder는 업데이트하지 않는다.

## 📊 Results

### 실험 설정
- **데이터셋 및 환경:** MuJoCo 기반의 6가지 작업(Cheetah Velocity, Ant Direction, Humanoid, Walker Param, Hopper Param)과 더 복잡한 Sawyer Manipulation 작업에서 평가하였다.
- **데이터 수집:** 일부러 매우 서브옵티멀(suboptimal)한 데이터를 사용하여, 제안 방법이 부족한 데이터로부터 얼마나 개선될 수 있는지 테스트하였다.
- **비교 대상:** MACAW, BOReL(기존 Offline Meta-RL), Meta Behavior Cloning, 그리고 Online Oracle(정답 보상을 모두 받는 온라인 학습)을 비교 대상으로 설정하였다.

### 주요 결과
- **성능 향상:** 모든 환경에서 SMAC은 Offline 학습 직후보다 Self-supervised 단계 이후에 성능이 지속적으로 향상되는 모습을 보였다.
- **분포 변화 완화:** Ant Direction 작업에서 시각화 결과, Offline 학습 직후에는 $h_{\text{online}}$에 기반한 적응 성능이 매우 낮았으나, Self-supervised 학습 후에는 데이터 소스에 관계없이 일관되게 높은 성능을 보였다.
- **비교 우위:** MACAW와 BOReL 같은 기존 방식들은 온라인 데이터 활용 기제가 없으므로 오프라인 성능에 머무는 반면, SMAC은 이들을 크게 상회하며 Online Oracle의 성능에 근접하였다.
- **Sawyer Manipulation:** 가장 어려운 작업인 로봇 조작 환경에서도 SMAC은 타 방법론 대비 압도적인 성능 향상을 보였으며, 특히 AWAC 기반의 actor 업데이트가 필수적임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Offline Meta-RL의 성능 저하가 단순한 데이터 부족이 아니라, **적응 단계에서 사용하는 데이터의 분포가 학습 단계와 다르기 때문에 발생하는 $z$-space의 불일치**에 있음을 날카롭게 지적하였다.

**강점:**
- 보상 라벨링 비용이 높다는 현실적인 제약을 고려하여, '한 번의 라벨링 $\rightarrow$ 무한한 무라벨 데이터 활용'이라는 효율적인 프레임워크를 제시하였다.
- Reward Decoder를 통한 보상 예측 일반화가 정책 일반화보다 쉽다는 가설을 세우고 이를 실험적으로 증명하였다.
- $z$-space뿐만 아니라 state-space의 분포 변화 역시 이 방법론으로 완화될 수 있음을 추가 실험(Test task 상호작용)을 통해 보여주었다.

**한계 및 논의:**
- 에이전트가 환경과 자율적으로 상호작용할 수 있다는 가정이 필요하다. 실제 물리 로봇의 경우 안전 문제로 인해 무분별한 온라인 데이터 수집이 어려울 수 있다.
- Reward Decoder의 정확도가 성능에 미치는 영향에 대해 분석하였으나, 예상보다 낮은 정확도에서도 동작한다는 점은 흥미롭지만 더 깊은 이론적 분석이 필요해 보인다.

## 📌 TL;DR

Offline Meta-RL은 학습 데이터와 테스트 데이터 간의 컨텍스트 분포($z$-space) 불일치로 인해 성능이 저하되는 문제가 있다. SMAC은 오프라인 데이터를 통해 **Reward Decoder**를 학습시키고, 이를 이용해 **라벨 없는 온라인 데이터에 가상 보상을 부여**함으로써 이 분포 격차를 해소한다. 결과적으로 최소한의 보상 라벨만으로도 온라인 Meta-RL에 근접한 빠른 적응 능력을 갖춘 에이전트를 학습시킬 수 있음을 보였으며, 이는 향후 실용적인 로봇 학습 시스템 구축에 중요한 기여를 할 것으로 보인다.