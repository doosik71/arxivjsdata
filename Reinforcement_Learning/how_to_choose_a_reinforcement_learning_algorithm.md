# How to Choose a Reinforcement-Learning Algorithm

Fabian Bongratz, Vladimir Golkov, Lukas Mautner, Luca Della Libera, Frederik Heetmeyer, Felix Czaja, Julian Rodemann, Daniel Cremers

## Problem to Solve

강화 학습(**RL**) 분야는 순차적 의사결정 문제를 해결하기 위한 다양한 개념과 방법을 제공하지만, 그 수가 너무 많아 특정 작업에 적합한 알고리즘을 선택하는 것이 매우 어렵습니다. 기존 설문조사(**survey**)들은 **RL** 방법론과 응용 분야에 대한 개요를 제공하지만, 어떤 상황에서 어떤 방법을 선택해야 하는지에 대한 **구조화된 지침이 부족**합니다. 또한, **RL** 알고리즘 설계에 대한 실용적인 권장 사항(**recommendations**)은 주로 시뮬레이션된 로코모션 벤치마크와 같은 **몇 가지 환경 조건에만 집중**되어 있습니다.

## Key Contributions

이 연구는 **RL** 알고리즘 및 행동 분포(**action-distribution**) 계열을 선택하는 과정을 간소화합니다.

- **기존 방법론과 그 속성에 대한 구조화된 개요**를 제공하고, 특정 상황에서 어떤 방법을 선택해야 하는지에 대한 지침을 제시합니다.
- **실용적인 고려 사항**과 **기존 문헌**을 기반으로 특정 환경에서 특정 방법이 다른 방법보다 유리한 이유를 설명합니다.
- 이러한 지침의 **대화형 온라인 버전**(<https://rl-picker.github.io/>)을 제공하여 사용자가 쉽게 알고리즘을 선택할 수 있도록 돕습니다.
- 새로운 알고리즘을 제안하기보다는 **기존 지식을 체계화하고 통합**하여 **RL** 알고리즘 선택을 위한 **실용적인 가이드라인**을 제시합니다.

## Methodology

이 논문은 **RL** 알고리즘 선택을 위한 **체계적인 의사결정 과정**을 제시하며, 이는 다양한 특성과 상황을 고려한 **조건부 지침**으로 구성됩니다. 주요 방법론은 다음과 같습니다.

1. **모델 기반(**Model-based**) vs. 모델 프리(**Model-free**) 강화 학습 선택**:

   - 환경 동역학(**dynamics**)의 가용성(**availability**) 및 학습 가능성, 학습 과정의 주요 목표(**안정성**, **데이터 효율성**, **전이 학습**, **안전성**, **설명 가능성**)를 기반으로 선택합니다(**Table 2** 참조).

2. **계층적(**Hierarchical**) **RL** 방법론 적용 여부 결정**:

   - 행동 시퀀스(**action sequences**)의 복잡성(**complexity**)에 따라 계층적 접근 방식의 적합성을 평가합니다(**Table 3** 참조).

3. **모방 학습(**Imitation Learning**) 포함 여부**:

   - 전문가(**expert**) 행동 데이터의 가용성 여부를 고려하여 모방 학습의 필요성을 판단합니다(**Table 4** 참조).

4. **분산 알고리즘(**Distributed Algorithms**) 선택**:

   - 사용 가능한 계산 자원(**computational resources**)과 병렬 실행(**parallel execution**)의 가능성 여부를 기준으로 결정합니다(**Table 5** 참조).

5. **분포형 알고리즘(**Distributional Algorithms**) 선택**:

   - 특정 행동의 "위험"(**risk**)을 추정해야 하는 요구 사항 유무에 따라 결정합니다(**Table 6** 참조).

6. **온-정책(**On-policy**) vs. 오프-정책(**Off-policy**) 학습 선택**:

   - 탐색(**exploration**)의 중요성, 훈련 안정성(**training stability**), 경험(**experience**) 획득 비용(**cost**)을 고려하여 결정합니다(**Table 8** 참조).

7. **가치 기반(**Value-based**) vs. 정책 기반(**Policy-based**) vs. 액터-크리틱(**Actor-critic**) 알고리즘 선택**:

   - 참 가치 함수(**true value functions**)와 최적 정책(**optimal policy**) 간의 관계, 그리고 대상 정책(**target policy**)의 확률적(**stochastic**) 또는 결정론적(**deterministic**) 특성을 고려합니다(**Table 9** 참조).
   - 가치 기반의 경우 상태 공간(**state space**)과 행동 공간(**action space**)의 크기에 따라 세부 알고리즘(**Table 11**)을, 정책 기반 또는 액터-크리틱의 경우 학습된 크리틱(**critic**) 사용 여부와 에피소드(**episode**) 길이를 고려합니다(**Table 10** 참조).

8. **가치 함수 학습(**Value-function learning**)을 위한 서브 알고리즘 선택**:

   - 마르코프 가정(**Markov assumption**) 위반 여부, 보상의 밀도(**dense**)와 희소성(**sparse**) 또는 기만성(**deceptive**), 에피소드 길이를 고려하여 `Monte Carlo`, `Temporal-Difference`(**TD**), `Eligibility Traces` 또는 `n-step methods` 중에서 선택합니다(**Table 12** 참조).

9. **엔트로피 정규화(**Entropy Regularization**) 적용 여부**:

   - 보상 특성과 정책의 확률적 특성을 고려하여 `per-state entropy regularization`, `soft Q-learning`, `Kullback–Leibler divergence regularization` 또는 `Mutual information`(**MI**) regularization을 선택합니다(**Table 13** 참조).

10. **행동 분포 계열(**Action-distribution families**) 선택**:
    - 가치 기반 알고리즘의 경우 `greedy`, `ε-greedy`, `Boltzmann exploration`, `randomized value functions`, `noisy nets` 중 선택합니다(**Table 14** 참조).
    - 정책 기반 또는 액터-크리틱 알고리즘의 경우 `categorical distribution`, `Gaussian distribution`, `Gaussian mixture distribution`, `normalizing flows`, `stochastic networks` 및 `black-box policies`, `tanh` 함수를 따르는 연속 분포, `Beta distribution`, `deterministic`, `added noise` 중에서 선택합니다(**Table 15** 참조). 이때 행동 분포의 표현력(**expressiveness**)(`Table 16`)과 행동 공간(**action space**)을 고려합니다.

**추가적인 고려 사항**:

- **신경망(**Neural-network**) 아키텍처**: `Fully Connected NNs`, `Time-Recurrent NNs`, `CNNs`, `Dueling NNs`, `Parameter sharing`, `Broadcasting`, `Seq2Seq` 등 다양한 아키텍처가 사용될 수 있습니다(**Section 5** 참조).
- **훈련 안정성 및 성능 개선 기술**: `Trust regions`, `Clipping parts of the objective functions`, `Clipping the gradients`, `Target networks`, `Implicit parameterization`, `Learned state- or time-dependent baselines`, `Weight averaging` (**안정성** 개선). `Diverse set of policies`, `Intrinsic motivation`, `Data augmentation`, `Low-level design choices` (**테스트 결과** 개선) 등 다양한 기법이 설명됩니다(**Section 6** 참조).
- **최종 선택**: 여러 옵션이 남은 경우, 실험을 통해 비교하거나 유사한 문제에서 좋은 성능을 보인 방법을 참고할 것을 권장합니다(**Section 4** 참조).

## Results

이 논문은 새로운 **RL** 알고리즘을 제안하거나 실험적 결과를 제시하는 대신, **RL** 알고리즘 선택을 위한 **포괄적이고 체계적인 가이드라인**을 제공하는 것을 주요 성과로 합니다.

- **RL** 알고리즘 선택의 복잡성을 줄이고, 사용자에게 **명확하고 구조화된 의사결정 프로세스**를 제공합니다.
- **모델 프리 RL** 알고리즘을 제어(**control**) 작업에 적용할 때 고려해야 할 핵심 속성들을 **테이블과 설명을 통해 상세히 분류**합니다.
- 각 환경적 특성과 목표에 따라 어떤 알고리즘 속성이 적합한지 **"If ... then ... because ..."** 형식으로 명시하여 **실용적인 지침**을 제공합니다.
- 이러한 접근 방식은 **RL** 분야의 광범위한 방법론을 **효과적으로 탐색하고 이해**하는 데 도움을 줍니다.
- 궁극적으로, 특정 상황에 **최적의 "만능" 알고리즘은 존재하지 않음**을 강조하며, 여러 개념을 통합하거나 특정 상황에서 유리한 알고리즘을 선택하기 위한 **가이드라인과 온라인 도구**(<https://rl-picker.github.io/>)를 제공함으로써 **RL** 알고리즘 선택 과정을 **간소화**하는 데 기여합니다.
