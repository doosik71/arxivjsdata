# Imitation Bootstrapped Reinforcement Learning

Hengyuan Hu, Suvir Mirchandani, Dorsa Sadigh (2024)

## 🧩 Problem to Solve

본 논문은 로봇 제어 작업에서 강화학습(Reinforcement Learning, RL)이 가진 고질적인 문제인 **샘플 효율성(Sample Efficiency)**과 **탐색(Exploration)**의 어려움을 해결하고자 한다.

로봇 제어와 같은 연속적인 제어 문제에서는 보상 신호가 매우 희소(Sparse Reward)한 경우가 많아, 무작위로 초기화된 정책으로는 성공적인 경험을 얻기 매우 어렵다. 이를 해결하기 위해 모방 학습(Imitation Learning, IL)이 대안으로 사용되지만, 모든 시나리오를 커버하는 광범위한 전문가 시연 데이터를 수집하는 것은 비용이 매우 많이 들며, 학습 데이터와 실제 환경 간의 **분포 변화(Distribution Shift)**가 발생할 경우 성능이 급격히 저하되는 한계가 있다.

따라서 본 연구의 목표는 모방 학습의 효율적인 시작 지점과 강화학습의 자가 개선(Self-improvement) 능력을 결합하여, 적은 양의 시연 데이터만으로도 효율적으로 학습하고 전문가 이상의 성능을 달성할 수 있는 **Imitation Bootstrapped Reinforcement Learning (IBRL)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

IBRL의 핵심 아이디어는 독립적으로 학습된 IL 정책을 RL의 **추론(Inference)** 단계와 **학습(Training)** 단계 모두에서 '고품질 액션 제안자'로 활용하는 것이다.

1. **Actor Proposal (추론 단계):** 온라인 상호작용 시, RL 정책이 제안하는 액션과 IL 정책이 제안하는 액션 중 현재의 Q-함수가 더 높게 평가하는 액션을 선택하여 탐색의 질을 높인다.
2. **Bootstrap Proposal (학습 단계):** TD-학습의 타겟 값을 계산할 때, RL 정책의 액션뿐만 아니라 IL 정책의 액션 중 더 높은 가치를 가진 것을 사용하여 가치 추정의 정확도를 높이고 수렴 속도를 가속화한다.
3. **모듈형 구조:** IL 정책을 RL 네트워크와 분리함으로써, 각 작업에 최적화된 서로 다른 네트워크 아키텍처를 사용할 수 있으며, RL과 IL 사이의 가중치를 조절하기 위한 복잡한 하이퍼파라미터 튜닝(예: 정규화 손실 계수)이 필요 없다.

## 📎 Related Works

본 논문은 기존의 RL-IL 결합 방식들을 다음과 같이 분류하고 차별점을 제시한다.

- **시연 데이터 오버샘플링 (예: RLPD, Hybrid RL):** 리플레이 버퍼에 전문가 데이터를 넣고 높은 확률로 샘플링하는 방식이다. 단순하지만 IL 정책이 가진 '어느 정도 보장된 액션 품질'이라는 정보를 직접적으로 활용하지 못한다.
- **사전 학습 후 미세 조정 (예: RFT, ROT):** IL로 정책을 먼저 학습시킨 후 RL로 파인튜닝하며, 이때 기존 지식을 잊지 않기 위해 BC(Behavior Cloning) 손실 함수를 정규화 항으로 추가한다. 이 방식은 정규화 계수 $\alpha$를 정밀하게 튜닝해야 하며, IL과 RL이 동일한 아키텍처를 공유해야 한다는 제약이 있다.
- **모델 기반 RL (예: MoDem):** 세계 모델(World Model)을 학습하여 샘플 효율성을 높인다. 성능은 좋으나 계산 비용이 매우 높아 실시간 고주파 제어에는 부적합하다.

**IBRL의 차별점:** IBRL은 IL 정책을 별도의 독립된 참조 정책(Reference Policy)으로 유지하며, 이를 통해 정규화 손실 없이도 자연스럽게 IL의 지식을 RL에 주입한다. 또한, 탐색과 타겟 업데이트 양쪽 모두에 IL을 통합함으로써 최대의 샘플 효율성을 달성한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

IBRL은 크게 두 단계로 진행된다. 먼저 제공된 전문가 시연 데이터 $D$를 사용하여 독립적인 IL 정책 $\mu_\psi$를 학습시킨다. 이후 이 정책을 고정시킨 채, TD3(Twin Delayed DDPG)를 백본으로 하는 RL 학습을 진행한다.

### 2. 상세 방법론

#### A. IL 정책 학습

Behavior Cloning (BC)을 사용하여 전문가 데이터의 액션을 모방하는 정책 $\mu_\psi$를 학습한다. 손실 함수는 다음과 같은 평균 제곱 오차(MSE) 형태를 가진다.
$$L(\psi) = \mathbb{E}_{(s,a)\sim D} \|\mu_\psi(s) - a\|_2^2$$

#### B. Online Interaction: Actor Proposal

에이전트가 환경과 상호작용할 때, RL 정책 $\pi_\theta$와 IL 정책 $\mu_\psi$가 각각 액션을 제안한다. 에이전트는 타겟 Q-네트워크 $Q_{\phi}'$를 사용하여 더 높은 가치를 가진 액션을 최종적으로 선택한다.
$$a^* = \text{argmax}_{a \in \{a_{IL}, a_{RL}\}} Q_{\phi}'(s, a)$$
여기서 $a_{IL} \sim \mu_\psi(s)$ 이고 $a_{RL} \sim \pi_\theta(s)$ 이다.

#### C. RL Training: Bootstrap Proposal

Q-함수를 업데이트하기 위한 타겟 값(Target Value)을 계산할 때, 미래 상태 $s_{t+1}$에서 RL 정책과 IL 정책 중 더 높은 Q-값을 가진 액션을 선택하여 부트스트랩한다.
$$Q_{\phi}(s_t, a_t) \leftarrow r_t + \gamma \max_{a' \in \{a_{IL}, a_{RL}\}} Q_{\phi}'(s_{t+1}, a')$$
이는 미래의 롤아웃이 항상 IL과 RL 중 최선의 선택을 하는 정책에 의해 수행될 것이라고 가정하는 것이다.

#### D. Soft IBRL 변형

Greedy한 $\text{argmax}$ 선택이 지역 최적해(Local Optimum)에 빠질 가능성을 방지하기 위해, 볼츠만 분포(Boltzmann distribution)를 이용한 확률적 샘플링 방식을 제안한다.
$$p_Q(a) \propto \exp(\beta Q(s, a)) \quad \text{for } a \in \{a_{IL}, a_{RL}\}$$
여기서 $\beta$는 분포의 가파른 정도를 조절하는 역온도(Inverse Temperature) 파라미터이다.

### 3. 추가적인 아키텍처 개선

- **Actor Dropout:** 정책 네트워크(Actor)에 Dropout을 적용하여 학습 안정성과 샘플 효율성을 높였다.
- **ViT-based Q-network:** 픽셀 입력의 경우, 기존의 얕은 ConvNet 대신 얕은 Vision Transformer(ViT) 스타일의 인코더를 사용하여 복잡한 조작 작업에서의 표현력을 높였다.

## 📊 Results

### 1. 실험 설정

- **시뮬레이션:** Meta-World(4개 작업), Robomimic(PickPlaceCan, NutAssemblySquare) 작업 수행. 모든 작업은 성공 시 1, 실패 시 0을 주는 희소 보상(Sparse Reward) 환경이다.
- **실제 로봇:** Lift(블록 들어올리기), Drawer(서랍 열기), Hang(천 걸기) 세 가지 작업 수행.
- **비교 대상:** RLPD(오버샘플링), RFT(정규화 파인튜닝), MoDem(모델 기반 RL).

### 2. 주요 결과

- **시뮬레이션 결과:** Meta-World에서는 IBRL이 MoDem과 RLPD를 능가했으며, RFT와 유사한 성능을 보였다. 특히 가장 어려운 작업인 Robomimic의 'Square' 작업에서 IBRL은 유일하게 0.5M 샘플 내에 작업을 해결했으며, 타 방법론 대비 압도적인 성공률을 기록했다.
- **실제 로봇 결과:** 모든 실제 작업에서 IBRL이 가장 빠른 학습 속도와 높은 최종 성공률을 보였다. 특히 가장 어려운 'Hang' 작업에서 기존 RL 방법론들은 BC 베이스라인조차 넘지 못했으나, IBRL은 BC보다 20% 더 높은 성공률을 달성했다.
- **분포 변화 대응:** 'Lift' 작업의 Hard Eval 설정(블록이 카메라 경계에 위치)에서 BC는 성공률이 0%로 떨어졌으나, IBRL은 95%의 성공률을 유지하여 RL을 통한 자가 개선이 분포 변화 문제를 효과적으로 해결함을 입증했다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **자동 균형 조절:** 별도의 정규화 계수 튜닝 없이도, 학습 초기에는 IL 정책의 비중이 높고 학습 후기에는 RL 정책의 비중이 높아지며 자동으로 최적의 균형을 찾아간다.
- **아키텍처의 유연성:** 실험을 통해 BC는 깊은 ResNet-18에서 성능이 좋고, RL은 얕은 ViT에서 성능이 좋음을 확인했다. IBRL은 이 둘을 분리하여 각각 최적의 네트워크를 사용할 수 있게 함으로써 성능을 극대화했다.
- **부트스트랩의 중요성:** Ablation 연구를 통해 'Actor Proposal'보다 'Bootstrap Proposal'이 성능 향상에 더 결정적인 역할을 함을 확인했다. 이는 IL 정책이 타겟 가치 추정치의 분산을 줄여 학습을 가속화하기 때문이다.

### 2. 한계 및 논의사항

- **환경 리셋:** 실제 로봇 실험에서 수동 리셋(Manual Reset)에 의존했다는 점이 한계로 지적되었으며, 향후 자율 리셋(Autonomous Reset) 시스템과의 결합이 필요하다.
- **IL 정책의 품질:** suboptimal한 IL 정책을 사용했을 때 초기 성능은 낮아지지만, 결국 RL의 자가 개선을 통해 최적의 정책으로 수렴할 수 있음을 확인했다. 이는 IBRL이 매우 강건한 프레임워크임을 시사한다.

## 📌 TL;DR

본 논문은 독립적으로 학습된 모방 학습(IL) 정책을 강화학습(RL)의 **액션 제안(Inference)**과 **가치 업데이트(Training)** 단계에 직접 통합한 **IBRL** 프레임워크를 제안한다. 이를 통해 하이퍼파라미터 튜닝의 어려움 없이 샘플 효율성을 극대화했으며, 특히 보상이 희소하고 난이도가 높은 로봇 조작 작업에서 기존 SOTA 방법론들을 크게 상회하는 성능을 보였다. 이 연구는 실제 로봇 환경에서 적은 양의 시연 데이터만으로 효율적인 정책 개선을 이룰 수 있는 실용적인 방향성을 제시한다.
