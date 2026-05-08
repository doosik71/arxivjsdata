# Curriculum Learning with a Progression Function

Andrea Bassich, Francesco Foglino, Matteo Leonetti, Daniel Kudenko (2021)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)에서 복잡한 작업을 학습시키기 위해 사용되는 Curriculum Learning(CL)의 효율성과 유연성을 높이는 문제를 해결하고자 한다. 일반적으로 강화학습 에이전트가 매우 복잡한 최종 목표(Final Task)를 처음부터 학습하는 것은 매우 어렵고 시간이 오래 걸린다. 이를 위해 중간 단계의 작업들을 순차적으로 학습시키는 커리큘럼을 구성하지만, 기존의 접근 방식들은 다음과 같은 한계점을 가진다.

첫째, 많은 기존 방법론들이 유한한 작업 집합(Finite set of tasks)을 전제로 하거나, 특정 작업 간의 전이 가능성에 의존한다. 둘째, 커리큘럼의 생성 단계가 실행 전(Offline)에 결정되거나, 에이전트의 내부 상태에 접근해야 하는 등 알고리즘 의존적인 경우가 많다.

따라서 본 연구의 목표는 학습 알고리즘에 구애받지 않고(Learning-algorithm agnostic), 무한한 작업 집합에서도 적용 가능하며, 에이전트의 숙련도에 따라 실시간으로 환경의 복잡도를 조절할 수 있는 새로운 커리큘럼 생성 패러다임을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 커리큘럼 생성을 **Progression Function(진행 함수)**과 **Mapping Function(매핑 함수)**이라는 두 가지 독립적인 구성 요소로 분리하는 것이다.

- **Progression Function ($\Pi$):** 주어진 시점에서 에이전트에게 적절한 '복잡도(Complexity)' 수치를 결정한다. 이는 "얼마나 빨리 다음 단계로 넘어갈 것인가"에 대한 스케줄링을 담당한다.
- **Mapping Function ($\Phi_D$):** 결정된 복잡도 수치를 실제 환경의 파라미터로 변환하여 구체적인 MDP(Markov Decision Process)를 생성한다. 이는 "특정 복잡도가 실제 환경에서 어떤 모습인가"에 대한 도메인 지식을 담당한다.

이러한 설계를 통해 도메인 전문가의 고수준 지식을 Mapping Function에 쉽게 주입할 수 있으며, 에이전트의 성능에 따라 복잡도를 유연하게 조절하는 Adaptive Curriculum을 구현할 수 있다.

## 📎 Related Works

논문에서는 기존의 Curriculum Learning 접근 방식을 크게 두 가지 범주로 나누어 설명한다.

1. **경험 재배치 방식 (Experience Ordering):** Prioritized Experience Replay와 같이 단일 MDP 내에서 샘플의 순서를 조정하는 방식이다.
2. **작업 레벨 방식 (Task-level CL):** 에이전트에게 서로 다른 MDP의 시퀀스를 제공하는 방식이다.

기존의 Task-level 방법론들과의 차별점은 다음과 같다.

- **Reverse Curriculum Generation (RCG):** 목표 지점에서 시작 지점을 점진적으로 멀리 떨어뜨리는 방식이다. 본 제안 방법은 시작 지점뿐만 아니라 환경의 다양한 파라미터를 수정할 수 있다는 점에서 더 일반적이다.
- **HTS-CR:** 작업 시퀀싱을 조합 최적화 문제로 정의하여 최적의 순서를 찾는 방식이다. 하지만 HTS-CR은 유한한 작업 집합을 필요로 하는 반면, 본 제안 프레임워크는 무한한 작업 집합에서도 작동 가능하다.
- **Teacher-Student CL:** 교사가 학생의 진전을 관찰하며 작업을 선택한다. 본 논문은 중간 작업들이 최종 작업을 위한 디딤돌 역할을 하도록 설계하여 최종 작업의 학습 시간을 최소화하는 데 집중한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

전체 시스템은 $\Pi \rightarrow \Phi_D \rightarrow \text{MDP}$ 순으로 작동한다. Progression Function $\Pi$가 복잡도 값 $c_t \in [0, 1]$를 계산하면, Mapping Function $\Phi_D$가 이를 받아 실제 학습 환경인 $M_i$를 생성하고, 에이전트는 이 환경에서 학습을 진행한다.

### Progression Functions

논문은 두 가지 종류의 진행 함수를 제안한다.

**1. Fixed Progression (고정 진행)**
에이전트의 성능과 관계없이 시간에 따라 복잡도가 결정된다.

- **Linear Progression:** 시간이 흐름에 따라 복잡도가 선형적으로 증가한다.
  $$\Pi^l(t, t_e) = \min\left(\frac{t}{t_e}, 1\right)$$
- **Exponential Progression:** 파라미터 $s$에 따라 초기에 빠르게 증가하거나 후반에 빠르게 증가하도록 조절 가능하다.
  $$\Pi^e(t, \{t_e, s\}) = \frac{(e^{t/t_e})^{\alpha-1}}{e^{\alpha-1}}, \quad \alpha = \frac{1}{s}$$

**2. Adaptive Progression (적응형 진행)**
에이전트의 성능 함수 $p_t$를 기반으로 실시간으로 복잡도를 조절한다. 특히 **Friction-based progression**은 물리적인 '마찰력' 모델을 차용한다.

- **직관:** 평면 위를 미끄러지는 상자의 속도 $s_t$를 정의하고, 복잡도를 $c_t = 1 - \text{speed}$로 정의한다. 에이전트의 성능 향상도($\mu_t$)가 마찰력으로 작용하여 상자의 속도를 늦추고, 결과적으로 복잡도를 높이는 방식이다.
- **주요 방정식:**
  - 성능 변화율(마찰 계수): $\mu_t = \frac{p_t - p_{t-i}}{i}$
  - 속도 업데이트: $s_t = \max(0, \min(1, s_{t-1} - m \cdot g \cdot \mu_t))$
  - 복잡도 결정: $\Pi^{fu} = 1 - \text{Uniform}(s_t, s_{min})$
  여기서 $i$는 성능을 측정하는 시간 간격이며, $\text{Uniform}$ 샘플링을 통해 성능 저하시 복잡도를 약간 낮추어 에이전트가 너무 어려운 과제에 빠져 학습이 망가지는 것을 방지한다.

### Mapping Functions

Mapping Function $\Phi_D$는 $c_t$ 값을 받아 환경 파라미터 $a_t$로 변환한다. 도메인 지식을 활용하여 단조 증가/감소 관계를 정의하며, 다음과 같은 선형 보간 식을 사용한다.

- $a_h > a_e$ 인 경우: $a_t = a_e + (a_h - a_e) \cdot c_t$
- $a_e > a_h$ 인 경우: $a_t = a_e - (a_e - a_h) \cdot c_t$
($a_e$: 가장 쉬운 값, $a_h$: 가장 어려운 값)

### Parallel Environments

여러 개의 환경을 병렬로 운영하며, 각 환경마다 서로 다른 $\Pi$ 파라미터(예: 서로 다른 $s$ 값이나 $i$ 간격)를 부여한다. 이를 통해 에이전트가 다양한 난이도의 경험을 동시에 쌓게 하여 학습의 안정성을 높인다.

## 📊 Results

### 실험 설정

- **데이터셋(도메인):** Grid World Maze, Point Mass Maze, Directional Point Maze, Ant Maze, Predator-Prey, Half Field Offense (HFO) 총 6개 도메인.
- **비교 대상 (Baselines):** Reverse Curriculum Generation (RCG), HTS-CR.
- **학습 알고리즘:** PPO (Proximal Policy Optimization).
- **측정 지표:** 도메인별로 상이 (성공률, 생존 시간, 누적 보상 등).

### 주요 결과

1. **Friction-based Progression의 우수성:** 거의 모든 도메인에서 Friction-based progression이 가장 높은 성능을 보였으며, 특히 복잡한 제어가 필요한 MuJoCo Maze와 HFO에서 압도적이었다.
2. **Uniform Sampling의 효과:** 단순 속도 기반($1-s_t$)보다 Uniform 샘플링을 적용한 방식이 더 안정적이었다. 이는 급격한 난이도 상승 시 에이전트가 일시적으로 쉬운 과제를 다시 학습하며 적응할 기회를 주기 때문이다.
3. **병렬 학습의 이점:** 멀티프로세싱을 통해 서로 다른 진행 속도를 가진 환경에서 동시에 학습했을 때, 단일 프로세스보다 학습 초기 성능이 빠르게 상승하고 최종 성능 또한 향상되는 경향을 보였다.
4. **강건성(Robustness):** Mapping Function에 의도적으로 노이즈(Local, Global noise)를 섞었을 때도 Friction-based progression은 성능 저하가 적어, 매핑 함수 설계가 완벽하지 않더라도 적응형 진행 함수가 이를 어느 정도 보완함을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 커리큘럼의 '속도'와 '내용'을 분리함으로써 얻는 유연성이다. 기존 방법론들이 "어떤 작업을 다음에 수행할 것인가"라는 이산적인 선택 문제에 집중했다면, 본 연구는 이를 연속적인 복잡도 제어 문제로 변환하여 무한한 작업 공간을 효율적으로 탐색하게 하였다.

특히 Friction-based progression에서 도입한 물리적 모델은 에이전트의 성능 향상 속도와 환경 난이도 상승 속도 사이의 동적인 균형을 맞추는 효과적인 메커니즘을 제공한다.

다만, Mapping Function을 정의할 때 여전히 "어떤 파라미터가 난이도를 결정하는가"에 대한 최소한의 도메인 지식이 필요하다는 점은 한계로 볼 수 있다. 만약 난이도 정의 자체가 매우 모호한 도메인이라면, 본 프레임워크의 효율성은 떨어질 가능성이 있다. 또한, 병렬 환경에서의 파라미터 설정($s$ 값의 범위 등)에 대한 최적의 기준이 명확히 제시되지 않아 사용자의 튜닝이 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 강화학습의 커리큘럼 생성을 **복잡도 결정(Progression Function)**과 **환경 생성(Mapping Function)**으로 분리한 새로운 프레임워크를 제안한다. 특히 에이전트의 성능 변화를 마찰력으로 모델링한 **Friction-based progression**을 통해 실시간으로 난이도를 조절하는 적응형 커리큘럼을 구현하였으며, 이는 6개의 다양한 도메인에서 기존 SOTA 방법론(RCG, HTS-CR)보다 뛰어난 성능과 강건성을 보였다. 이 연구는 도메인 지식을 간단한 매핑 함수로 주입하고 알고리즘에 상관없이 적용할 수 있어, 향후 매우 복잡한 RL 환경의 학습 속도를 높이는 데 중요한 역할을 할 것으로 기대된다.
