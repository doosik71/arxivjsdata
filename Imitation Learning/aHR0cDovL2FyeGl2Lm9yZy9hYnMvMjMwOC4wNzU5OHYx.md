# Generating Personas for Games with Multimodal Adversarial Imitation Learning

William Ahlberg, Alessandro Sestini, Konrad Tollmar, Linus Gisslén (2023)

## 🧩 Problem to Solve

게임 개발 과정에서 플레이테스트(Playtesting)는 버그 발견뿐만 아니라 게임 디자인의 질을 결정하는 필수적인 품질 관리 단계이다. 그러나 대규모 AAA급 게임의 경우, 사람이 직접 수행하는 수동 플레이테스트는 막대한 비용과 시간이 소모되어 효율성이 떨어진다. 이를 해결하기 위해 자동화된 테스트 에이전트의 필요성이 제기되었으며, 특히 단순한 성능 최적화를 넘어 다양한 사용자의 플레이 스타일, 즉 '페르소나(Persona)'를 모사하는 에이전트가 필요하다.

기존의 강화 학습(Reinforcement Learning, RL) 기반 에이전트는 인간 수준의 성능을 낼 수 있으나, 페르소나와 같은 정성적인 특성을 보상 함수(Reward Function)로 정의하는 '보상 엔지니어링(Reward Engineering)' 과정이 매우 복잡하고 어렵다는 한계가 있다. 또한, RL 에이전트의 정책(Policy)은 예측 불가능한 경우가 많아 게임 디자이너가 의도한 특정 스타일의 행동을 유도하기 어렵다. 본 논문의 목표는 보상 엔지니어링 없이 전문가의 시연(Demonstration) 데이터만을 이용해 여러 페르소나를 학습하고, 이를 자유롭게 전환하거나 혼합할 수 있는 단일 에이전트 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **MultiGAIL(Multimodal Generative Adversarial Imitation Learning)**이라는 새로운 모방 학습(Imitation Learning) 알고리즘을 제안한 것이다. 

MultiGAIL의 핵심 아이디어는 단일 정책 모델에 **보조 입력 파라미터(Auxiliary input parameter, $\bar{\alpha}$)**를 도입하여, 여러 페르소나의 특성을 하나의 모델에 통합하는 것이다. 이를 위해 각 페르소나별로 개별적인 판별기(Discriminator)를 두어 스타일 보상을 계산하고, 사용자가 입력하는 $\bar{\alpha}$ 값에 따라 각 판별기의 보상 비중을 조절함으로써 추론 시점에 실시간으로 플레이 스타일을 변경하거나 서로 다른 스타일을 블렌딩(Blending)할 수 있게 설계하였다.

## 📎 Related Works

### 1. 자동화된 플레이테스트 (Automated Playtesting)
기존 연구들은 주로 휴리스틱이나 MCTS(Monte-Carlo Tree Search)와 같은 고전적 AI 기법을 사용했다. 최근에는 심층 강화 학습(Deep RL)을 통해 게임의 상태 공간을 탐색하고 버그를 찾는 시도가 있었으나, RL은 상태-행동 공간 탐색에 막대한 계산량이 필요하며 최종 정책을 정밀하게 제어하기 어렵다는 단점이 있다.

### 2. 페르소나 모델링 (Personas)
일부 연구에서 게임 메트릭(Game-metrics)과 RL/MCTS를 결합해 특정 페르소나를 부여하려 했으나, 이는 여전히 복잡한 보상 함수 설계에 의존한다. 정성적인 행동 특성을 정량적인 보상 함수로 변환하는 과정에서 의도치 않은 행동이 발생하는 문제가 빈번하다.

### 3. 모방 학습 (Imitation Learning)
Behavioral Cloning(BC)은 분포 변화(Distributional shift) 문제로 인해 학습되지 않은 상태에서 오류가 누적되는 경향이 있다. GAIL(Generative Adversarial Imitation Learning)은 적대적 학습을 통해 이를 해결하려 했으나 학습 불안정성이 존재한다. AMP(Adversarial Motion Priors)는 GAIL을 확장하여 스타일이 있는 캐릭터 제어에 성공했다. 본 논문은 AMP를 기반으로 하되, 다중 판별기와 보조 입력을 통해 '다중 페르소나'를 지원한다는 점에서 차별성을 가진다. 또한, 기존의 Policy Fusion(PF) 방식이 페르소나별로 개별 모델을 학습시켜 추론 시점에 결합하는 것과 달리, MultiGAIL은 단일 모델만 사용하므로 추론 시간이 훨씬 짧고 연속적인 스타일 블렌딩이 가능하다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
MultiGAIL은 전문가의 시연 데이터셋 $M = \{M_i\}_{i=1}^n$ (여기서 $n$은 페르소나의 수)을 기반으로 학습한다. 시스템은 크게 **보상 모델 학습(Reward Models Learning)**과 **정책 학습(Policy Learning)**의 두 부분으로 나뉜다.

### 2. 보상 모델 및 스타일 보상
에이전트의 전체 보상 $r$은 환경의 목표를 달성하기 위한 '작업 보상(Task-reward, $r^G$)'과 행동의 스타일을 결정하는 '스타일 보상(Style-reward, $r^S$)'의 가중 합으로 정의된다.

$$r(s_t, a_t, s_{t+1}) = w^G r^G(s_t, a_t, s_{t+1}, g) + w^S r^S(s_t, a_t)$$

여기서 스타일 보상 $r^S$는 각 페르소나 $M_i$에 대응하는 $n$개의 판별기 $\{D_i\}_{i=1}^n$의 출력값과 보조 입력 $\alpha_i$의 곱으로 계산된다.

$$r^S(s_t, a_t) = \sum_{i=1}^n \alpha_i \max \left[ 0, 1 - 0.25(D_i(s_t, a_t) - 1)^2 \right]$$

- **판별기($D_i$):** 입력된 상태-행동 쌍이 전문가 데이터 $M_i$에서 왔는지 아니면 에이전트의 정책에서 왔는지 판별하며, 유사도를 확률 값으로 반환한다.
- **보조 입력($\alpha_i$):** 특정 페르소나의 영향력을 조절하는 가중치이다. 예를 들어 $\bar{\alpha} = (1, 0)$이면 첫 번째 페르소나만 모사하고, $\bar{\alpha} = (0.5, 0.5)$이면 두 스타일을 혼합한다.

### 3. 학습 절차 및 손실 함수
- **판별기 학습:** 학습 안정성을 위해 LSGAN(Least-Square GAN) 손실 함수와 Gradient Penalty를 사용한다.
$$\mathcal{L}_{AMP}^i = \arg \min_{D_i} \mathbb{E}_{d^{M_i}(s,a)} [(D_i(s,a)-1)^2] + \mathbb{E}_{d^\pi(s,a)} [(D_i(s,a)+1)^2] + w_{gp} \text{ (Gradient Penalty)}$$
- **정책 학습:** PPO(Proximal Policy Optimization) 알고리즘을 사용하여 위에서 정의된 전체 보상 $r$을 최대화하는 방향으로 정책 $\pi$를 업데이트한다.
- **학습 루프:** 매 에피소드마다 $\alpha_i$ 값을 $[0, 1]$ 범위에서 무작위로 샘플링하여 에이전트가 다양한 스타일 조합을 경험하도록 유도한다.

### 4. 신경망 아키텍처
정책 $\pi$와 판별기 $D_i$는 동일한 구조를 공유한다.
1. **입력 처리:** 에이전트-목표 벡터 정보는 Linear 레이어를 통해 self-embedding($x_a$)으로 변환된다. 주변 엔티티(적, 물체 등) 정보는 공유 가중치 Linear 레이어를 거쳐 Transformer Encoder에 입력된 후 average pooling되어 $x_t$가 된다.
2. **지역 인지:** $5 \times 5 \times 5$ 크기의 3D 시맨틱 맵(Semantic map)을 3D CNN에 통과시켜 $x_M$을 추출한다.
3. **최종 출력:** $x_t$와 $x_M$을 결합한 후 MLP를 통과시킨다. 정책 $\pi$는 행동 확률 분포를 출력하고, 판별기 $D_i$는 Sigmoid 함수를 통해 확률 값을 출력한다.

## 📊 Results

### 1. 실험 설정
- **Racing Game (연속 행동 공간):** 'Careful'(조심스러운 주행)과 'Reckless'(난폭한 주행) 두 페르소나 학습.
- **Navigation Game (이산 행동 공간):** 'Jump', 'Zigzag', 'Strafe' 세 페르소나 학습.
- **평가 지표:** 전문가의 행동 분포와 에이전트의 행동 분포 간의 거리(KL Divergence, JS Divergence, $\chi^2$ test, Wasserstein distance)를 측정하여 모사 성능을 정량화하였다.

### 2. 주요 결과
- **단일 모델 다중 작업 성능:** MultiGAIL은 각 페르소나별로 개별 모델을 학습시킨 AMP(Baseline)보다 오히려 더 정확하게 전문가의 행동 분포를 복제하는 경향을 보였다(특히 Racing Game). 이는 여러 판별기를 동시에 최적화하는 과정이 Generator의 Mode Collapse를 방지하고 학습을 더 견고하게 만들기 때문으로 분석된다.
- **페르소나 보간(Interpolation):** $\bar{\alpha}$ 값을 조정함에 따라 에이전트의 행동이 연속적으로 변화함을 확인하였다. Racing Game에서 $\alpha$ 값의 변화에 따라 가속도와 조향(Steering)의 분포가 부드럽게 전이되는 것이 Kernel Density Estimation(KDE) 그래프를 통해 입증되었다.
- **이산 행동 공간에서의 비교:** Navigation Game에서 MultiGAIL을 Policy Fusion(PF)과 비교한 결과, MultiGAIL이 여러 페르소나를 혼합할 때 더 자연스럽고 강건한(Robust) 전이를 보였으며, 단일 모델만으로 동일한 효과를 낼 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 복잡한 보상 함수 설계 없이 시연 데이터와 $\bar{\alpha}$ 파라미터만으로 다중 페르소나를 구현했다는 점에서 실용성이 매우 높다. 특히 추론 시점에 단일 파라미터 변경만으로 행동 스타일을 즉각적으로 바꿀 수 있어, 게임 디자이너가 직관적으로 테스트 에이전트를 제어할 수 있는 환경을 제공한다.

### 한계 및 비판적 해석
1. **비선형적 관계:** 실험 결과, 보조 입력 $\alpha$와 실제 구현되는 페르소나 간의 관계가 항상 선형적이지 않다는 점이 발견되었다. 특히 페르소나가 3개 이상일 때 $\bar{\alpha} = (0.5, 0.5, 0.5)$가 정확히 절반의 혼합을 보장하지 않는 문제가 있으며, 이는 향후 연구가 필요한 부분이다.
2. **환경의 단순성:** 검증에 사용된 내비게이션 작업들이 비교적 단순하다. 실제 복잡한 게임 메카닉이 포함된 환경에서도 이 방식이 유효할지는 추가적인 검증이 필요하다.
3. **제로 입력 시 행동:** 모든 $\alpha_i$가 0일 때 정책이 무작위(Random)가 될 것이라 예상했으나, 실제로는 학습된 스타일들의 평균적인 행동이 나타났다. 이는 모델이 $\alpha$가 없을 때의 기본 행동(Default behavior)을 어떻게 처리하는지에 대한 이론적 분석이 부족함을 시사한다.

## 📌 TL;DR

본 논문은 전문가의 시연 데이터를 이용해 여러 플레이 스타일(페르소나)을 학습하는 **MultiGAIL** 알고리즘을 제안하였다. 단일 모델 내에 다중 판별기를 배치하고 보조 입력 파라미터 $\bar{\alpha}$를 통해 각 스타일의 가중치를 조절함으로써, **보상 엔지니어링 없이도 페르소나의 전환 및 블렌딩이 가능**함을 보였다. 이 연구는 게임 테스트 자동화 과정에서 디자이너가 직관적으로 에이전트의 행동을 제어할 수 있는 도구를 제공하며, 향후 다양한 사용자 행동 모델링 연구에 기여할 가능성이 크다.