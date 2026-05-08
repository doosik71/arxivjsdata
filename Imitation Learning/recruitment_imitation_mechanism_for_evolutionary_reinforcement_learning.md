# Recruitment-imitation Mechanism for Evolutionary Reinforcement Learning

Shuai Lü, Shuai Han, Wenbo Zhou, Junwei Zhang (2019)

## 🧩 Problem to Solve

본 논문은 연속 제어 작업(continuous control tasks)을 해결하기 위해 강화학습(Reinforcement Learning, RL), 진화 알고리즘(Evolutionary Algorithms, EA), 그리고 모방 학습(Imitation Learning, IL)의 장점을 결합하고자 한다.

각 방법론은 다음과 같은 고유한 한계점을 가진다. 강화학습은 샘플 효율성이 높지만 하이퍼파라미터 설정에 민감하고 효율적인 탐색(exploration)이 필수적이다. 진화 알고리즘은 학습 과정이 안정적이지만 샘플 효율성이 매우 낮다. 모방 학습은 효율성과 안정성을 모두 갖추었으나, 반드시 전문가 데이터(expert data)라는 가이드가 필요하다.

특히 기존의 진화 강화학습(Evolutionary Reinforcement Learning, ERL) 방식에서는 다음과 같은 두 가지 구체적인 문제점이 존재한다.

1. 강화학습 에이전트가 경험(experience)을 통해서만 학습할 뿐, 현재 진화 집단(population) 내의 엘리트 개체들로부터 직접적인 가이드를 받을 수 없다.
2. 강화학습 에이전트와 진화 집단 내 개체들의 네트워크 구조가 다를 경우, 강화학습 에이전트를 집단에 직접 주입하여 진화 과정에 참여시키는 것이 불가능하다.

따라서 본 논문의 목표는 이러한 구조적, 방법론적 제약을 극복하기 위해 Recruitment-imitation Mechanism (RIM)이라는 확장 가능한 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 강화학습 에이전트가 진화 집단에서 우수한 개체를 '채용(Recruitment)'하여 학습에 활용하고, 반대로 집단의 저성능 개체들이 강화학습 에이전트를 '모방(Imitation)'하게 함으로써 두 프로세스를 유기적으로 결합하는 것이다.

이를 위해 본 연구는 다음과 같은 설계를 도입하였다.

- **Dual-actors and Single-critic 구조**: 강화학습 에이전트에 두 개의 액터(Gradient Policy Network와 Recruitment Policy Network)를 두어, 크리틱(Critic)이 더 높은 가치를 지닌 행동을 선택하게 함으로써 학습 성능을 높였다.
- **Off-policy 모방 학습**: 전문가 데이터를 별도로 수집하는 대신, Experience Replay Buffer에 저장된 데이터를 사용하여 집단 내 최하위 개체가 RL 에이전트의 행동 패턴을 학습하도록 설계하였다.
- **Soft Update 기반 채용**: 진화 집단의 챔피언을 채용할 때 Soft Update 전략을 적용하여 학습의 일관성과 안정성을 확보하였다.

## 📎 Related Works

본 논문은 강화학습과 진화 알고리즘, 또는 모방 학습을 결합하려는 기존 시도들을 다룬다. 전통적인 결합 방식은 모방 학습이나 진화 알고리즘을 강화학습의 하위 프로세스(예: 초기 가중치 최적화, 탐색 노이즈 개선)로 사용하는 경향이 있었다.

특히 ERL(Evolutionary Reinforcement Learning)은 집단과 환경의 상호작용 데이터를 재사용하고 RL 정책을 집단에 주입하는 패러다임을 제시하였다. RIM은 ERL의 확장선상에 있으나, 다음과 같은 차별점을 가진다.

- RL 관점에서 진화 알고리즘을 결합하여, 집단의 정책을 직접 RL 프로세스에 채용한다.
- Dual-policy 에이전트를 구축하여 $Q$-value의 추정 정확도를 높였다.
- 구조적 불일치 문제를 해결하기 위해 직접적인 주입 대신 Off-policy 모방 학습을 통해 정책을 동기화한다.

또한 CERL(Collaborative ERL)이나 CEM-RL과 같은 최신 연구들이 탐색 최적화나 하위 컴포넌트 교체에 집중한 반면, RIM은 채용-모방이라는 메커니즘을 통해 세 가지 학습 방법론의 효과적인 결합을 추구한다.

## 🛠️ Methodology

### 1. Dual Policy Reinforcement Learning Agent

RIM의 핵심은 두 개의 액터 네트워크와 하나의 크리틱 네트워크로 구성된 RL 에이전트이다.

- **Gradient Policy Network ($\pi_{pg}$)**: 경사 하강법을 통해 학습되는 일반적인 RL 정책이다.
- **Recruitment Policy Network ($\pi_{ea}$)**: 진화 집단에서 가장 우수한 개체의 파라미터를 복사하여 생성된 정책이다.

에이전트의 최종 행동 결정 정책 $\pi_{rl}$은 다음과 같이 결정된다.
$$\pi_{rl} = \begin{cases} \pi_{pg}, & Q(s_t, \pi_{pg}(s_t|\theta_{pg})|\theta_Q) \ge Q(s_t, \pi_{ea}(s_t|\theta_{ea})|\theta_Q) \\ \pi_{ea}, & \text{otherwise} \end{cases}$$
즉, 크리틱 네트워크가 판단하기에 더 높은 보상을 기대할 수 있는 정책의 행동을 선택한다.

### 2. $Q$-value 추정의 정확성 (Theorem 1)

논문은 이러한 Dual-policy 구조가 $Q'$의 추정치를 더 정확하게 만든다는 것을 수학적으로 증명한다. DDPG의 행동 정책 $\pi_{ddpg}$보다 $\pi_{rl}$을 사용했을 때, 최적 정책 $\pi^*$에 의한 가치 $Q^*$에 더 가까운 기댓값을 가짐을 보였다.
$$E_{ddpg}(\hat{Q}') \le E_{rim}(\hat{Q}') \le Q^*$$
이는 RL 컴포넌트가 충분히 수렴하지 않았더라도, 진화 집단의 우수한 정책($\pi_{ea}$)이 더 나은 행동을 제시한다면 $Q'$ 추정치가 상향 보정되어 학습 속도가 빨라짐을 의미한다.

### 3. Off-policy Imitation Learning

RL 에이전트와 진화 집단 개체의 구조가 다를 경우 직접 주입이 불가능하므로, 모방 학습을 통해 행동 패턴을 전달한다. DAgger의 반복적인 데이터 라벨링 과정 대신, Experience Replay Buffer를 직접 활용하는 Off-policy 방식을 사용한다. 집단 내 최악의 성능을 가진 개체 $\pi_{wt}$는 다음 손실 함수를 통해 RL 에이전트 $\pi_{rl}$을 모방하여 학습한다.
$$J(\theta_{wt}) = \frac{1}{K} \sum_{k=1}^{K} L(\pi_{wt}(s_k|\theta_{wt}), \pi_{rl}(s_k))$$
이 과정은 데이터 버퍼 내의 수많은 오류 상태 데이터를 포함하고 있어, 결과적으로 에이전트가 오류 상태에서 복구하는 방법까지 학습하게 하는 효과를 준다.

### 4. 전체 학습 및 진화 절차

1. **진화 단계**: 집단 내 개체들의 적합도를 평가하여 생존자를 선택하고, 변이(mutation)와 교차(crossover)를 통해 다음 세대를 생성한다.
2. **채용 단계**: 집단의 챔피언을 $\pi_{ea}$로 채용한다. 이때 target recruitment network를 도입하여 Soft Update를 적용함으로써 학습 안정성을 높인다.
   $$\theta_{ea}' \leftarrow (1-\tau_0)\theta_{ea}' + \tau_0\theta_{ea}$$
3. **RL 학습 단계**: $\pi_{rl}$을 통해 행동을 결정하고 경험을 버퍼에 저장하며, DDPG 방식으로 $\pi_{pg}$와 $Q$ 네트워크를 업데이트한다.
4. **모방 단계**: 주기적으로 집단 내 최하위 개체가 $\pi_{rl}$을 모방 학습하여 RL의 성과를 집단에 전파한다.

## 📊 Results

### 실험 설정

- **환경**: Mujoco의 4가지 연속 제어 작업 (Walker2d-v2, Hopper-v2, HalfCheetah-v2, Swimmer-v2).
- **비교 대상**: EA, DDPG, ERL.
- **평가 지표**: 최대 보상(Max), 평균(Mean), 중앙값(Median), 표준편차(Std).

### 주요 결과

1. **정량적 성능**: RIM은 모든 환경에서 기존 방법론(EA, DDPG, ERL)보다 높은 평균 성능을 보였다. 특히 Walker2d와 Hopper 환경에서 우월성이 두드러졌는데, 이는 전통적인 Off-policy RL이 탐색에 어려움을 겪는 환경에서 RIM의 채용 메커니즘이 정체기(stagnation period)를 빠르게 돌파했기 때문이다.
2. **컴포넌트 분석**: RIM의 RL 컴포넌트 성능이 ERL의 RL 에이전트보다 우수했으며, 모방 학습을 통해 주입된 개체들이 ERL에서 RL 에이전트를 직접 주입했을 때보다 더 좋은 성능을 보였다.
3. **채용 전략 비교**: Hard Update보다 Soft Update를 통한 채용이 $y_t$ 계산 시 더 안정적인 값을 제공하여 학습 속도와 성능을 약간 더 향상시켰다.
4. **Ablation Study**:
   - **RIM-IL (모방 학습 제외)**: 성능이 크게 하락하여, RL의 성과를 집단에 전달하는 모방 학습의 중요성이 입증되었다.
   - **RIM-EA / RIM-PG (Dual-policy 구성 변경)**: $Q'$ 추정을 위해 $\pi_{ea}$나 $\pi_{pg}$ 중 하나만 사용한 경우 성능이 하락하여, Dual-policy를 통한 정확한 $Q$-value 추정이 핵심임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 RL, EA, IL이라는 세 가지 서로 다른 패러다임을 '채용'과 '모방'이라는 개념으로 연결하여 시너지를 냈다. 특히, RL 에이전트가 단순히 데이터를 생성하는 역할에 그치지 않고, 진화 집단의 정수를 직접 흡수($\pi_{ea}$ 채용)함으로써 학습의 가이드라인을 얻는 구조는 매우 효율적이다.

**강점**:

- 구조적 제약(네트워크 형태 차이)을 모방 학습으로 우회하여 유연한 프레임워크를 구축하였다.
- $Q$-value 추정 오차 문제를 해결하기 위한 Dual-policy 구조를 이론적(Theorem 1)으로 뒷받침하였다.

**한계 및 논의**:

- 본 연구에서는 표준 EA를 사용하였으나, NES(Natural Evolution Strategy)나 NEAT와 같은 더 효율적인 진화 알고리즘을 적용한다면 추가적인 성능 향상이 가능할 것으로 보인다.
- DDPG 기반의 구조를 제안하였으나, SAC나 TD3와 같은 최신 Off-policy 알고리즘을 Dual-policy 구조에 이식한다면 더욱 강력한 성능을 낼 수 있을 것이다.
- RL 업데이트와 모방 학습 업데이트가 순차적으로 이루어지므로, 실제 훈련 시간이 길어질 수 있어 병렬 처리 방법의 도입이 필요하다.

## 📌 TL;DR

본 논문은 진화 알고리즘의 안정성과 강화학습의 효율성, 모방 학습의 가이드를 통합한 **Recruitment-imitation Mechanism (RIM)**을 제안한다. 핵심은 **Dual-actor RL 에이전트**가 진화 집단의 엘리트를 채용하여 $Q$-value 추정 정확도를 높이고, 집단의 하위 개체들이 RL 에이전트를 모방하여 집단 전체의 수준을 끌어올리는 상호보완적 구조이다. Mujoco 벤치마크 실험 결과, RIM은 기존 ERL 및 DDPG 대비 월등한 성능을 보였으며, 특히 학습 정체기를 빠르게 극복하는 능력을 입증하였다.
