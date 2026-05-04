# Active Hierarchical Imitation and Reinforcement Learning

Yaru Niu, Yijun Gu (2020)

## 🧩 Problem to Solve

본 연구는 로봇이 희소 보상(sparse rewards) 환경에서 복잡한 작업을 효율적으로 학습하기 위해 계층적 구조를 활용하는 방안을 다룬다. 인간은 복잡한 문제를 하위 작업(sub-tasks)으로 분할하여 해결하는 능력이 있으며, 이를 모방 학습(Imitation Learning, IL)과 강화 학습(Reinforcement Learning, RL)의 결합을 통해 로봇에게 구현하려는 시도가 지속되어 왔다.

기존의 계층적 모방 및 강화 학습(Hierarchical Imitation and Reinforcement Learning, HIRL) 연구들은 주로 이산적인(discrete) 액션 공간과 단순한 2D 게임 환경에 국한되어 있었다. 또한, 많은 모방 학습 연구들이 사람이 아닌 하드코딩된 전문가 정책이나 RL로 학습된 정책에 의존했다. 실제 인간-로봇 상호작용(Human-Robot Interaction, HRI) 시나리오에서는 사람이 직접 시연(demonstration)을 제공해야 하므로, 전문가의 물리적·정신적 노력을 줄이기 위해 학습 효율을 높이는 것이 매우 중요하다. 따라서 본 논문의 목표는 연속적인(continuous) 상태 및 액션 공간에서 작동하는 계층적 학습 프레임워크를 구축하고, 능동 학습(Active Learning, AL) 기법을 도입하여 데이터 효율성을 높이고 전문가의 비용을 절감하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 연속 공간에서의 계층적 구조를 기반으로, 전문가의 시연 효율을 극대화하기 위한 능동 학습 알고리즘을 설계하고 이를 인간 대상 실험으로 검증한 것이다. 구체적인 설계 아이디어는 다음과 같다.

1. **연속 공간을 위한 HIRL 프레임워크**: 고수준 제어기(High-level controller)는 DAgger를 통해 모방 학습을 수행하고, 저수준 제어기(Low-level controller)는 강화 학습(DDPG)을 통해 하위 목표를 달성하도록 설계하여 학습 효율을 높였다.
2. **세 가지 능동 학습(Active Learning) 전략 제안**:
    * **Noise-Based AL**: 상태 공간에 노이즈를 추가하여 정책의 분산이 큰 지점을 찾아내고, 해당 지점을 초기 상태로 설정하여 학습하는 방식이다.
    * **Multi-Policy AL**: 앙상블 학습과 유사하게 여러 정책을 학습시킨 후, 정책 간의 의견 불일치(disagreement)가 큰 상태를 선택하여 집중적으로 학습하는 방식이다.
    * **Reward-Based AL**: 환경의 보상 정보를 활용하여, 목표 달성에 실패한 에피소드의 데이터 중 가치가 낮은 데이터를 경험 재현 버퍼(experience replay buffer)에서 제거함으로써 유용한 데이터만 유지하는 방식이다.
3. **인간 중심의 성능 평가**: 단순한 정량적 지표뿐만 아니라 NASA-TLX 기반의 설문을 통해 전문가가 느끼는 물리적·정신적 부하를 측정하여 제안 방법론의 실용성을 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들의 한계점을 지적하며 차별성을 둔다.

* **Interactive Imitation Learning**: DAgger와 AggreVaTeD 같은 알고리즘이 제안되었으나, AggreVaTeD의 경우 환경의 보상 정보에 의존하여 정책 경사(policy gradient)를 업데이트하므로 보상이 희소한 환경에서는 효율이 떨어진다는 한계가 있다.
* **Hierarchical Reinforcement Learning (HRL)**: Options 프레임워크나 FeUdal Network 등이 제안되었으나, 저수준 정책이 변함에 따라 고수준 정책 학습이 불안정해지는 Non-stationary 문제가 존재한다. 본 연구는 이를 해결하기 위해 Hindsight Experience Replay (HER) 개념을 차용한 구조를 활용한다.
* **Active Learning**: 분류 문제나 단순 내비게이션 작업에서의 AL 연구는 많았으나, 다중 작업 학습(Multi-task learning)과 AL을 결합하여 계층적 구조에 적용한 사례는 드물다.
* **Combining RL and LfD**: DQfD와 같이 시연 데이터를 사전 학습(pre-training)에 사용하는 방식이 존재하지만, 본 연구는 고수준과 저수준 제어기 간의 상호작용 형태로 결합하여 연속적인 피드백 환경에 적용했다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (HIRL Framework)

본 시스템은 2단계 계층 구조로 설계되었다.

* **고수준 제어기(High-level Meta-controller)**: 최종 목표 $\text{g}$를 달성하기 위해 중간 단계인 하위 목표(subgoal) $\text{g}_{LO}$를 생성한다. 이 제어기는 DAgger를 통해 인간 전문가의 시연을 모방하여 학습한다.
* **저수준 제어기(Low-level Controller)**: 고수준 제어기가 제시한 $\text{g}_{LO}$에 도달하기 위한 구체적인 액션을 수행한다. 이는 수정된 DDPG(Deep Deterministic Policy Gradient) 알고리즘을 통해 학습된다.

### 2. 능동 학습(Active Learning) 상세 방법론

#### (1) Noise-Based AL

상태 벡터 $\text{s}$를 목표 상태와 일치하는 부분 공간 $\text{s}_g$와 그 외의 부분 공간 $\text{s}_o$로 나눈다. $\text{s}_o$에 무작위 균등 노이즈를 추가했을 때, 현재 정책 $\pi_\theta$의 출력값 변화(분산)가 가장 큰 $\text{s}_g$를 선택하여 초기화 지점으로 설정한다.
$$\text{s}^*_g = \arg \max_{\text{s}_g \in \text{S}_g} \sum_{i=1}^n [\pi_\theta(\text{s}_i(\text{s}_g)) - \bar{\pi}_\theta(\text{s}_i(\text{s}_g))]^2$$
여기서 $\text{s}_i(\cdot)$는 $\text{s}_o$에 노이즈를 추가하여 생성된 상태를 의미한다.

#### (2) Multi-Policy AL

동일한 데이터셋으로 학습된 $n$개의 서로 다른 정책 $\{\pi_{\theta,i}\}$를 생성한다. 특정 상태 $\text{s}$에서 각 정책들의 출력값이 서로 가장 많이 다른(분산이 큰) 지점을 찾아 $\text{s}^*_g$로 선택한다.
$$\text{s}^*_g = \arg \max_{\text{s}_g \in \text{S}_g} \sum_{i=1}^n [\pi_{\theta,i}(\text{s}(\text{s}_g)) - \bar{\pi}_{\theta,i}(\text{s}(\text{s}_g))]^2$$

#### (3) Reward-Based AL

에피소드가 종료되었을 때, 최종 목표 달성 여부에 따라 데이터를 처리한다. 목표 달성에 실패한 경우, 에피소드 끝 지점으로부터의 시간적 거리 $\text{t}_d$에 기반하여 데이터를 버퍼에서 삭제할 확률 $\text{p}$를 계산한다.
$$\text{p} = e^{-\text{t}_d}$$
즉, 목표 지점에서 멀리 떨어진(에피소드 초반의) 데이터일수록 삭제될 확률이 높아지며, 상대적으로 목표에 근접했던 유용한 데이터만 남겨 학습 효율을 높인다.

## 📊 Results

### 1. 실험 설정

* **환경**: MuJoCo 기반의 3D 미로 내비게이션. Agent는 Ant 로봇을 사용하며, 조감도(Bird's-eye view) 픽셀 데이터를 입력으로 받는다.
* **작업**: 무작위로 생성된 미로의 한 구석에서 무작위 목표 지점까지 500 스텝 이내에 도달하는 것.
* **비교 대상**: Vanilla DAgger, DAgger + Multi-Policy AL, DAgger + Reward-Based AL.
* **측정 지표**: 성공률(Success rate), 전문가 비용(Expert cost, 에피소드당 평균 시연 횟수), 시도 횟수(Attempts, 목표 달성까지 생성된 하위 목표 수).

### 2. 정량적 결과

* **성공률**: **Reward-Based AL**이 가장 우수한 성능을 보였다. 약 60회 에피소드 이후부터 일관되게 가장 높은 성공률을 기록했으며, 100회 미만 학습 시 성공률 60%를 초과하였다. 반면 Multi-Policy AL은 Vanilla DAgger보다 성능이 낮고 변동성이 컸다.
* **전문가 비용**: 성공률 50% 이상 구간에서 **Reward-Based AL**이 가장 적은 전문가 시연 횟수로 목표를 달성하여 비용 절감 효과가 뚜렷했다.
* **시도 횟수**: 세 알고리즘 모두 학습 초반 20회 이후 급격히 감소하는 경향을 보였으며, 알고리즘 간의 유의미한 차이는 나타나지 않았다.

### 3. 정성적 결과 (User Study)

5명의 참가자를 대상으로 NASA-TLX 기반 설문을 진행한 결과:

* **Reward-Based AL** 사용 시, 참가자들은 물리적 요구량(Physical demand)과 정신적 노력(Effort)이 유의미하게 낮다고 느꼈으며, 작업 성공에 대한 만족도는 더 높게 나타났다 ($p < .001$).
* **Multi-Policy AL** 사용 시, 오히려 Vanilla DAgger보다 물리적 요구량과 노력이 더 많이 든다고 느꼈으며 성공률에 대한 체감 수치도 낮았다.

## 🧠 Insights & Discussion

본 연구는 계층적 구조에 능동 학습을 결합했을 때, 단순히 데이터의 양을 늘리는 것보다 **어떤 데이터를 유지하고 어떤 상태에서 학습하느냐**가 훨씬 중요하다는 것을 보여준다.

특히 **Reward-Based AL**의 성공은 매우 흥미롭다. Noise-based나 Multi-policy 방식처럼 '불확실성'이 높은 지점을 찾는 방식은 이론적으로 타당해 보이지만, 실제 인간-로봇 상호작용 환경에서는 오히려 전문가에게 부자연스럽거나 불필요한 시연을 요구하게 되어 물리적/정신적 부하를 높였을 가능성이 크다. 반면, 보상을 기반으로 실패한 데이터 중 불필요한 부분을 쳐내는 방식은 전문가의 시연 데이터를 효율적으로 정제하여 학습에 활용함으로써 성능 향상과 비용 절감을 동시에 달성했다.

다만, 본 연구는 5명의 소규모 인원을 대상으로 한 실험이라는 점에서 일반화에 한계가 있으며, Multi-Policy AL이 왜 예상과 달리 성능이 저하되었는지에 대한 심층적인 분석이 부족하다는 점이 아쉽다.

## 📌 TL;DR

본 논문은 연속적인 공간의 로봇 제어를 위해 **고수준(모방 학습) $\rightarrow$ 저수준(강화 학습)** 구조의 계층적 프레임워크를 제안하고, 여기에 세 가지 능동 학습(AL) 기법을 적용하였다. 실험 결과, **보상 기반의 능동 학습(Reward-Based AL)**이 가장 높은 성공률을 기록함과 동시에 인간 전문가의 시연 비용과 심리적 부하를 유의미하게 줄였음을 확인하였다. 이는 향후 인간-로봇 협동 학습 시스템에서 데이터 효율성을 극대화하는 핵심 전략이 될 가능성이 높다.
