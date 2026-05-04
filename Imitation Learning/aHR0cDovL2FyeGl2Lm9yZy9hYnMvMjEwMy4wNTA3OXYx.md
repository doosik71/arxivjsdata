# Domain-Robust Visual Imitation Learning with Mutual Information Constraints

Edoardo Cetin & Oya Celiktutan (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 **Observational Imitation Learning(관찰 기반 모방 학습)**에서 발생하는 도메인 차이(Domain Gap) 문제이다. 일반적인 Imitation Learning은 에이전트의 관점에서 기록된 상태(state)와 행동(action) 데이터가 필요하며, 이를 Agent-centric imitation이라 한다. 그러나 실제 환경에서 전문가의 행동을 관찰하여 배우려면 전문가와 에이전트의 외형(appearance)이나 신체 구조(embodiment)가 서로 다를 가능성이 높다.

이러한 도메인 차이가 존재할 때, 에이전트는 전문가의 관찰 영상에서 '전문가의 의도(목표 달성 정도)'와 '도메인 고유의 특성(배경, 신체 구조 등)'을 분리해내야 한다. 만약 이를 분리하지 못하면, 에이전트는 목표 달성 방법이 아니라 전문가의 외형적 특징을 모방하려 하게 되며, 이는 학습 실패로 이어진다. 따라서 본 논문의 목표는 전문가와 에이전트의 도메인이 서로 다르더라도 강건하게 동작하는 모방 학습 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문은 **DisentanGAIL**이라는 새로운 알고리즘을 제안하며, 그 핵심 직관은 Discriminator 내부에 **Mutual Information(상호 정보량) 제약 조건이 적용된 잠재 표현(Latent Representation) 병목 구간**을 두는 것이다.

핵심 설계 아이디어는 다음과 같다:

1. **도메인 불변 표현 학습**: 잠재 표현 $z$가 도메인 정보(전문가인지 에이전트인지)를 포함하지 않으면서, 오직 작업의 목표 달성 정도(goal-completion information)만을 인코딩하도록 강제한다.
2. **두 가지 MI 제약 조건**: 전문가 시연 데이터와 무작위로 수집된 Prior 데이터를 활용하여, 잠재 표현과 도메인 레이블 사이의 상호 정보량을 제한함으로써 도메인 불변성을 확보한다.
3. **도메인 정보 은닉(Domain Information Disguising) 방지**: 통계 네트워크를 이중화하고 Spectral Normalization을 적용하여, Discriminator가 MI 제약을 우회하여 도메인 정보를 몰래 학습하는 현상을 방지한다.

## 📎 Related Works

기존의 모방 학습은 주로 Behavior Cloning이나 Inverse Reinforcement Learning (IRL)에 기반하였다. 최근에는 GAIL(Generative Adversarial Imitation Learning)과 같이 적대적 학습을 통해 보상 함수를 명시적으로 설계하지 않고 전문가의 행동을 모방하는 방식이 주를 이룬다.

Observational Imitation 영역에서는 전문가와 에이전트 사이의 매핑을 학습하려는 시도가 있었으나, 대개 시간적 정렬(time-aligned) 데이터나 사전 지식, 혹은 특정 환경 가정에 의존하는 한계가 있었다. 특히 Stadie et al. (2017)의 TPIL은 Domain Confusion Loss를 사용하여 도메인 불변 표현을 학습하려 했으나, 이는 주로 저차원 제어 문제나 단순한 외형 차이가 있는 경우에만 효과적이었으며, 신체 구조(embodiment)가 다른 고차원 문제에서는 성능이 저하되는 한계가 있었다.

## 🛠️ Methodology

### 전체 시스템 구조

DisentanGAIL은 전처리기 $P_{\theta_1}$와 불변 판별기 $S_{\theta_2}$로 구성된 Discriminator $D_\theta$를 사용한다. 전체 구조는 $D_\theta = S_{\theta_2} \circ P_{\theta_1}$로 정의된다.

1. **Preprocessor ($P_{\theta_1}$)**: 입력 관찰값 $o_i$를 다변량 가우시안 분포 $\mathcal{N}(\mu_{\theta_1}(o_i), \Sigma_{\theta_1}(o_i))$로 투영하여 잠재 표현 $z_i$를 샘플링한다. 가우시안 표현을 사용하는 이유는 latent space의 서포트를 무한히 확장하고, 특정 차원의 분산을 높임으로써 정보 흐름을 직접적으로 제어하기 위함이다.
2. **Invariant Discriminator ($S_{\theta_2}$)**: $z_i$의 시퀀스를 입력받아 해당 행동이 전문가의 것인지 판별한다. 특히, 현재 시점 $t$를 포함해 최근 4개의 관찰값에 대한 잠재 표현을 연결(concatenate)한 $\hat{z}_t = \text{concat}(z_t, z_{t-1}, z_{t-2}, z_{t-3})$를 입력으로 사용한다. 이는 POMDP 환경에서 관찰되지 않은 상태 정보를 복원하고 목표 달성 진행 상황을 파악하기 위해서이다.

### 학습 목표 및 손실 함수

Discriminator는 기본적으로 GAIL의 목적 함수 $J_G$를 최대화하도록 학습된다:
$$\arg \max_{\theta} J_G(\theta, B_E, B_\pi) = \arg \max_{\theta} \mathbb{E}_{B_E, P_{\theta_1}} \log(S_{\theta_2}(\hat{z}_i)) + \mathbb{E}_{B_\pi, P_{\theta_1}} \log(1 - S_{\theta_2}(\hat{z}_i))$$

여기에 도메인 불변성을 위한 두 가지 Mutual Information(MI) 제약 조건이 추가된다. MI 추정에는 MINE(Mutual Information Neural Estimator)을 사용하며, 통계 네트워크 $T_\phi$가 이를 계산한다.

1. **Expert Demonstrations Constraint**: 전문가 데이터 $B_E$와 에이전트 데이터 $B_\pi$의 합집합에 대해, 잠재 표현 $z_i$와 도메인 레이블 $d_i$ 사이의 MI를 1비트 미만으로 유지한다 ($I_\phi(z_i, d_i | B_E \cup B_\pi) < 1$). 이는 판별기가 오직 도메인 정보만으로 정답을 맞히는 것을 방지하고, 목표 달성 정보 $c_i$를 찾도록 강제한다.
2. **Prior Data Constraint**: 목표와 무관하게 수집된 Prior 데이터 $B_{P.E}$와 $B_{P.\pi}$에 대해, MI를 0에 가깝게 유지한다 ($I_\phi(z_i, d_i | B_{P.E} \cup B_{P.\pi}) \approx 0$). 이는 두 도메인의 무작위 행동 분포를 latent space에서 일치시켜 불필요한 정보를 제거하는 효과를 준다.

최종 Discriminator 목적 함수는 다음과 같다:
$$\arg \max_{\theta} J_G(\theta, B_E \cup B_\pi) - L_\beta(\theta_1, B_E \cup B_\pi) - L_\lambda(\theta_1, B_{P.E} \cup B_{P.\pi})$$
여기서 $L_\beta$는 적응형 페널티(Adaptive Penalty)이며, $L_\lambda$는 라그랑주 승수(Lagrange multiplier)를 이용한 듀얼 페널티(Dual Penalty)이다.

### 도메인 정보 은닉 방지 및 정책 학습

- **Double Statistics Network**: 두 개의 독립적인 $T_\phi$를 학습시키고 그 중 최댓값을 MI 추정치로 사용하여, $D_\theta$가 통계 네트워크의 국소 최적점(local minimum)을 이용해 도메인 정보를 숨기는 것을 방지한다.
- **Spectral Normalization**: $S_{\theta_2}$에 적용하여 1-Lipschitz 연속성을 보장함으로써, MI 추정기가 잡지 못한 미세한 도메인 정보를 판별기가 이용하는 것을 억제한다.
- **Policy Learning**: 학습된 Discriminator로부터 얻은 의사 보상(pseudo-reward) $R^D$를 사용하여 **Soft Actor-Critic (SAC)** 알고리즘을 통해 에이전트의 정책 $\pi_\omega$를 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**: MuJoCo 시뮬레이터를 사용하여 Inverted Pendulum, Reacher (저차원) 및 Hopper, Half-Cheetah, 7DOF-Pusher, 7DOF-Striker (고차원) 등 6가지 환경 렐름(Realm)을 구축하였다.
- **도메인 차이**: 외형(Appearance) 차이, 신체 구조(Embodiment) 차이, 그리고 두 가지가 모두 존재하는 경우를 설정하여 테스트하였다.
- **지표**: 무작위 행동을 0, 전문가 행동을 1로 스케일링한 누적 보상(cumulative reward)을 측정하였다.

### 주요 결과

1. **전반적 성능**: DisentanGAIL은 모든 도메인 차이 설정에서 TPIL, DCL, 정규화 없는 모델보다 일관되게 높은 성능을 보였으며, 특히 전문가의 성능에 근접한 결과를 나타냈다.
2. **Prior 데이터의 중요성**: Prior 데이터 제약을 제거한 버전(No prior)은 외형 차이가 심한 경우 성능이 크게 하락하였다. 이는 무감독 데이터가 도메인 불변 특징 학습에 중요한 역할을 함을 시사한다.
3. **고차원 작업 확장성**: 7DOF-Pusher 및 Striker와 같은 고차원 조작 작업에서도 전문가 성능의 90% 이상을 달성하였다. 반면 TPIL과 정규화 없는 모델은 고차원 환경에서 거의 학습에 실패하였다.
4. **정성적 분석**: 잠재 표현 $z$를 기반으로 전문가와 에이전트의 관찰값을 매칭시킨 결과, 외형과 구조가 다름에도 불구하고 '목표 달성 정도'가 유사한 상태끼리 성공적으로 매칭됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 단순히 도메인 레이블을 분류하지 못하게 하는 기존의 Domain Confusion 방식에서 벗어나, **상호 정보량(MI)의 상한선을 정교하게 제어**함으로써 해결책을 제시하였다. 특히, MI를 완전히 0으로 만드는 것이 아니라 1비트 미만으로 유지하는 설정이 중요한데, 이는 전문가 데이터와 에이전트 데이터의 목표 달성 분포가 본질적으로 다르기 때문에, MI를 너무 강하게 제한하면 오히려 유용한 정보(목표 달성 정보)까지 삭제되어 학습이 불가능해지기 때문이다.

### 한계 및 논의사항

- **Locomotion 작업의 난이도**: Hopper나 Half-Cheetah 같은 이동(locomotion) 작업에서는 조작(manipulation) 작업보다 성능이 다소 낮게 나타났다. 저자들은 이동 작업이 특정 상태 도달이 아니라 연속적인 행동 스트림을 유지하는 것이 목표이므로, 신체 구조 변화가 Discriminator의 보상 신호에 더 큰 모호성을 주기 때문이라고 분석한다.
- **Prior 데이터 의존성**: 강한 시각적 차이가 존재할 때 Prior 데이터가 필수적이라는 점은, 실제 환경 적용 시 무작위 행동 데이터 수집이라는 추가적인 비용이 발생함을 의미한다.

## 📌 TL;DR

본 논문은 전문가와 에이전트 사이의 외형 및 신체 구조 차이가 존재하는 상황에서도 가능한 **Observational Imitation Learning**을 위해 **DisentanGAIL**을 제안한다. 이 알고리즘은 Discriminator 내부에 MI 제약 조건이 적용된 잠재 표현 병목을 두어, 도메인 정보는 버리고 목표 달성 정보만을 추출하도록 유도한다. 실험을 통해 고차원 제어 작업에서도 강건한 모방 성능을 입증하였으며, 이는 향후 사람이 로봇을 직접 조종하지 않고 단순히 관찰하는 것만으로 가르칠 수 있는 자연스러운 인간-로봇 상호작용 연구에 기여할 가능성이 크다.
