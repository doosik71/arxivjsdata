# Accelerating Training in Pommerman with Imitation and Reinforcement Learning

Hardik Meisheri, Omkar Shelke, Richa Verma, Harshad Khadilkar (2019)

## 🧩 Problem to Solve

본 논문은 멀티 에이전트 경쟁 환경인 Pommerman(고전 게임 Bomberman의 시뮬레이션 버전)의 2x2 팀 모드에서 에이전트를 효율적으로 학습시키는 문제를 다룬다. Pommerman 환경은 다음과 같은 복잡한 특성을 가지고 있어 기존 강화학습(Reinforcement Learning, RL) 적용에 어려움이 있다.

첫째, 에이전트 관점에서 환경이 비정상성(Non-stationarity)을 띠므로, 보상이 에이전트 자신의 정책 변화만으로 설명되지 않는다. 둘째, 보상이 매우 희소하고(Sparse) 지연되어 나타나며, 특히 팀 단위의 공통 보상이 주어질 때 개별 에이전트의 기여도를 평가하는 신용 할당(Credit Assignment) 문제가 발생한다. 셋째, 부분 관측성(Partial Observability)과 통신 제한으로 인해 중앙 집중식 비평가(Centralised Critic) 구조를 사용할 수 없다.

결과적으로 순수 RL 방식은 샘플 효율성이 매우 낮아 상당한 양의 학습 데이터와 계산 자원이 필요하며, 학습 과정에서 기본 기술을 잊어버리는 정책 퇴화(Policy Degeneration) 현상이 빈번하게 발생한다. 따라서 본 연구의 목표는 Imitation Learning(IL)과 RL을 결합하여 학습 시간을 단축하고, 안정적인 성능 향상을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 노이즈가 포함된 전문가 정책을 먼저 모방 학습(Imitation Learning)하여 기본 기능을 습득하게 한 뒤, Proximal Policy Optimization(PPO) 알고리즘을 통해 이를 정교화하는 하이브리드 접근 방식을 사용하는 것이다. 특히, 모방 학습 단계에서 RL 단계로 전환될 때 발생할 수 있는 정책 망각(Policy Forgetting)을 방지하기 위해 다음과 같은 설계 아이디어를 도입하였다.

1. **안정적 학습 패러다임**: 보상 형성(Reward Shaping), 휴리스틱 기반의 액션 필터(Action Filter), 그리고 커리큘럼 학습(Curriculum Learning)을 통해 IL에서 RL로의 부드러운 전이를 구현하였다.
2. **학습 시간의 획기적 단축**: 트리 탐색(Tree Search) 기반의 방법이나 순수 RL 방식에 비해 훨씬 적은 횟수(총 150,000 게임)의 학습만으로도 우수한 성능을 확보하였다.
3. **행동 제약 메커니즘**: 에이전트가 무의미하게 반복 행동을 하는 Jitter 현상을 수정하고, 자살 행위(Suicidal actions)를 방지하는 필터를 적용하여 학습 효율을 높였다.

## 📎 Related Works

기존의 Pommerman 연구는 크게 모델 프리 RL(Model-free RL)과 트리 탐색 기반 RL(Tree-search-based RL)로 나뉜다. MCTS(Monte Carlo Tree Search)와 같은 탐색 기법은 완전 관측 모드(FFA)에서 강력한 성능을 보이지만, 계산 비용이 매우 높다. RHEA(Rolling Horizon Evolutionary Algorithm)와 같은 방식은 공격적인 전략이 자살로 이어질 위험이 있다는 한계가 지적되었다.

최근에는 A2C(Advantage Actor-Critic)를 이용한 지속적 학습(Continual Learning)이나 self-attention 메커니즘 기반의 Relevance Graphs, 그리고 학습 속도를 높이는 Backplay 기법 등이 제안되었다. 특히 PPO를 사용한 Skynet 에이전트가 높은 성능을 보였으나, 이는 순수 RL 방식으로서 막대한 계산 자원과 학습 시간을 요구한다. 본 논문은 이러한 기존 연구들의 높은 계산 비용 문제를 해결하기 위해 IL을 초기화 단계로 활용함으로써 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 및 MDP 모델링

본 문제는 마르코프 결정 과정(Markov Decision Process, MDP)으로 모델링되며, 이는 $(S, A, T, R, \gamma)$로 정의된다. 여기서 $S$는 부분 관측 상태, $A$는 6가지 이산 행동(상하좌우 이동, 폭탄 설치, 대기), $T$는 전이 확률, $R$은 보상, $\gamma$는 할인율이다.

### 네트워크 아키텍처

에이전트는 CNN 기반의 Actor-Critic 구조를 가진다.

- **입력($S$)**: $11 \times 11$ 크기의 보드 정보를 19개의 채널로 구성하여 입력한다. 여기에는 통로, 벽, 폭탄, 화염, 에이전트 위치, 파워업 상태뿐만 아니라, 각 타일의 '바람직함(Desirability)'을 나타내는 스칼라 값 채널이 포함되어 안전한 위치로 이동하도록 유도한다.
- **구조**: 3개의 Convolutional Layer(각 층마다 Max Pooling 및 Dropout 적용) $\rightarrow$ 2개의 Fully Connected Layer $\rightarrow$ Softmax 출력층(6개 유닛)으로 구성된다.

### 학습 절차

학습은 크게 두 단계로 진행된다.

1. **Imitation Learning (IL) 단계**:
    - 기본 휴리스틱 에이전트인 `SimpleAgent`가 플레이한 50,000 게임의 데이터를 수집한다.
    - 수집된 상태-행동 쌍을 사용하여 Cross-entropy loss를 통해 지도 학습(Supervised Learning)을 수행한다.
    - $$\text{Loss} = -\sum y \log(\hat{y})$$

2. **Reinforcement Learning (RL) 단계**:
    - IL로 학습된 가중치를 Policy Network의 초기값으로 사용한다. Value Network는 별도로 생성하여 처음부터 학습시킨다.
    - **PPO 알고리즘**을 사용하여 정책을 업데이트하며, 정책의 급격한 변화를 막기 위해 Clipping 기법을 사용한다.
    - **커리큘럼 학습**: `StaticAgent` $\rightarrow$ `SimpleAgent (NoBomb)` $\rightarrow$ `SimpleAgent` 순으로 점진적으로 어려운 상대와 대결하며 학습한다.

### 주요 최적화 기법

- **Reward Shaping**: 팀원 간의 신용 할당 문제를 해결하기 위해, 학습 시 팀원을 강제로 자살시켜 1:2 상황을 만든다. 터미널 보상은 다음과 같이 재설정된다.
  - 양쪽 적이 모두 살아있는 경우: $-1$
  - 적 중 한 명만 살아있는 경우: $0.5$
  - 승리한 경우(적 모두 제거): $1$
- **Jitter Correction**: 에이전트가 두 칸 사이를 무한 반복해서 이동하는 현상을 방지하기 위해, 특정 패턴이 감지되면 일시적으로 전문가 정책(Expert Policy)의 행동을 따르게 한다.
- **Action Filter**: 에이전트가 폭탄 경로로 들어가는 등 즉각적인 사망이 예상되는 행동을 선택하면 이를 거부하고, 사망하지 않는 다른 행동을 무작위로 선택하게 하여 학습 효율을 높인다.

## 📊 Results

### 실험 설정

- **비교 대상**: `SimpleAgent`(휴리스틱), `Imitation Agent`(순수 모방 학습), `PPOAgent_Cautious`(커리큘럼 및 필터가 없는 Vanilla PPO), `Skynet`(기존 고성능 RL 에이전트).
- **측정 지표**: 승리(Win), 패배(Lost), 무승부(Tie) 비율.

### 정량적 결과

실험 결과, 제안된 `PPOAgent`는 단순 모방 학습 에이전트보다 승률이 비약적으로 상승하였다. 특히, 2018년 대회에서 우수한 성적을 거둔 `Skynet`을 상대로 8게임 중 7게임을 승리하거나 무승부로 이끄는 성과를 보였다.

| 상대 에이전트 | Imitation (Vanilla) Win | PPO (Jitter+Action Filter) Win |
| :--- | :---: | :---: |
| StaticAgent | 0.111 | 0.904 |
| SimpleAgent | 0.331 | 0.778 |
| Skynet | - | 0.451 |

### 분석 결과

- **필터의 효과**: Jitter Correction은 무승부를 줄이고 승률을 높였으며, Action Filter는 패배율을 유의미하게 낮추었다. 두 필터를 모두 적용했을 때 최적의 성능이 나타났다.
- **커리큘럼의 중요성**: 커리큘럼 없이 학습한 `PPOAgent_Cautious`는 폭탄 설치 자체를 두려워하여 구석에 머무는 경향을 보였으며, 이는 기본 기술의 망각이 발생했음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 전문가 정책이 완벽하지 않더라도(Noisy Expert), 이를 초기값으로 사용하는 것이 RL의 수렴 속도를 획기적으로 높일 수 있음을 증명하였다. 특히, 단순히 IL 이후 RL을 수행하는 것이 아니라, 커리큘럼 학습과 행동 필터링이라는 '안전장치'를 통해 정책의 급격한 붕괴를 막은 점이 핵심적인 성공 요인이다.

비판적으로 해석하자면, Action Filter와 Jitter Correction은 일종의 하드코딩된 휴리스틱이다. 이는 학습 속도를 높여주지만, 순수한 RL의 관점에서는 에이전트가 스스로 최적의 경로를 탐색할 기회를 제한하는 요소가 될 수 있다. 또한, 팀원을 강제로 자살시켜 1:2 상황을 만든 보상 형성 방식은 실제 팀 협력(Coordination) 능력을 학습하는 데 한계가 있을 수 있다.

그럼에도 불구하고, 본 논문은 부분 관측성과 희소 보상이라는 가혹한 환경에서 어떻게 하면 RL 에이전트를 효율적으로 Cold-start 시킬 수 있는지에 대한 실무적인 통찰을 제공한다.

## 📌 TL;DR

본 논문은 Pommerman 환경에서 **Imitation Learning $\rightarrow$ PPO RL**로 이어지는 단계적 학습 파이프라인을 제안한다. **커리큘럼 학습, 보상 형성, 그리고 행동 필터(Jitter/Action Filter)**를 도입하여 학습 시간을 대폭 단축하면서도, 기존의 고성능 에이전트인 Skynet을 압도하는 성능을 달성하였다. 이 방법론은 보상이 희소하고 환경이 복잡한 멀티 에이전트 강화학습 문제에서 매우 유용한 가속화 전략이 될 수 있다.
