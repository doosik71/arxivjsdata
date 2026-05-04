# RAIL: A modular framework for Reinforcement-learning-based Adversarial Imitation Learning

Eddy Hudson, Garrett Warnell, and Peter Stone (2021)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning (AIL) 분야의 연구가 체계적인 모듈화 없이 개별적인(ad-hoc) 방식으로 진행되어 왔다는 점을 지적한다. 컴퓨터 비전이나 자연어 처리 분야에서는 최적화 알고리즘과 네트워크 아키텍처가 서로 독립적인 모듈로 정의되어 각자의 발전이 전체 성능 향상으로 이어지는 선순환 구조를 가지고 있다. 반면, AIL 연구는 어떤 설계 결정이 성능에 구체적으로 어떤 영향을 미치는지 명확하지 않은 상태에서 다양한 알고리즘들이 제안되어 왔다.

따라서 본 연구의 목표는 RL을 사용하는 AIL 알고리즘들을 체계적으로 정리할 수 있는 모듈형 프레임워크인 RAIL (Reinforcement-learning-based Adversarial Imitation Learning)을 제안하고, 이를 통해 새로운 알고리즘을 설계하여 그 유효성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AIL 알고리즘의 하위 집합을 'RL 백본(RL backbone)'과 '판별자 입력 표현(discriminator's input representation)'이라는 두 가지 핵심 모듈로 정의한 RAIL 프레임워크를 제시한 것이다. 이러한 모듈화 관점을 통해 연구자는 기존 알고리즘의 구성 요소를 자유롭게 조합하여 새로운 변형 모델을 쉽게 설계할 수 있다.

이 프레임워크를 바탕으로 저자들은 최신 Off-policy RL 알고리즘인 SAC (Soft Actor Critic)를 RL 백본으로 채택하고, 관찰 기반 모방 학습(IfO, Imitation from Observation)을 위해 판별자 입력으로 $(s_t, s_{t+1})$를 사용하는 **SAIfO (SAC-based Adversarial Imitation from Observation)** 알고리즘을 제안하였다.

## 📎 Related Works

논문에서는 GAIL, GAIfO, DAC, OPOLO와 같은 기존의 AIL 알고리즘들을 언급한다. AIL은 기본적으로 GAN (Generative Adversarial Networks)의 구조를 차용하여, 전문가의 궤적(trajectory)과 모방자의 궤적을 구분하려는 판별자($D$)와 판별자를 속이려는 생성자(정책 $\pi$) 사이의 min-max 게임을 수행한다.

기존 연구들의 한계와 차별점은 다음과 같다.
- **GAIL 및 GAIfO**: 주로 TRPO나 PPO 같은 On-policy RL 알고리즘을 사용하여 샘플 효율성이 낮다는 한계가 있다.
- **DAC 및 OPOLO**: Off-policy RL을 도입하여 샘플 효율성을 크게 개선하였으나, 여전히 전체적인 설계 구조가 모듈화되어 있지 않아 최적의 조합을 찾기 어렵다.
- **ValueDICE 및 ASAF**: 이들은 판별자로부터 직접 역전파를 수행하거나 가중치를 복사하는 방식을 사용하므로, RL 알고리즘을 통해 정책을 최적화하는 RAIL의 정의에 부합하지 않는다.

## 🛠️ Methodology

### RAIL 프레임워크 구조
RAIL은 AIL 알고리즘을 다음 두 가지 모듈의 조합으로 정의한다.

1. **RL Backbone**: 판별자가 제공하는 보상을 바탕으로 정책을 최적화하는 RL 알고리즘이다. (예: TRPO, PPO, TD3, SAC 등)
2. **Discriminator's Input Representation**: 판별자가 전문가와 모방자를 구분하기 위해 입력받는 데이터의 형태이다.
    - 상태-행동 쌍 $(s_t, a_t)$: 일반적인 IL에서 사용한다.
    - 상태-다음 상태 쌍 $(s_t, s_{t+1})$: 행동 정보가 없는 IfO 상황에서 사용한다.
    - 임의의 부분 시퀀스: 예컨대 $(s_t, s_{t+3})$ 등 다양한 형태가 가능하다.

### SAIfO (SAC-based Adversarial Imitation from Observation)
SAIfO는 위 RAIL 프레임워크의 설계 공간에서 다음과 같은 조합을 선택한 알고리즘이다.
- **RL Backbone**: SAC (Soft Actor Critic) $\rightarrow$ 최신 Off-policy RL 알고리즘을 사용하여 샘플 효율성과 안정성을 높였다.
- **Discriminator Input**: $(s_t, s_{t+1}) \rightarrow$ 행동 데이터 없이 상태 변화만을 관찰하여 학습하는 IfO를 구현하였다.

### 전반적인 학습 절차 및 목적 함수
AIL의 일반적인 목적은 판별자 $D$와 정책 $\pi$를 동시에 학습시키는 것이다.
- **판별자 ($D$)**: 전문가의 궤적 $\tau^E$에 대해서는 1에 가깝게, 모방자의 궤적 $\tau$에 대해서는 0에 가깝게 출력하도록 학습한다.
  $$\mathbb{E}_{o \sim \tau^E}[D(o)] \rightarrow 1, \quad \mathbb{E}_{o \sim \tau}[D(o)] \rightarrow 0$$
  여기서 $o$는 궤적의 세그먼트(SAIfO의 경우 $(s_t, s_{t+1})$)를 의미한다.
- **생성자/정책 ($\pi$)**: 판별자 $D$가 전문가의 데이터라고 믿게끔(즉, $D$의 출력이 커지도록) 행동을 생성한다. SAIfO에서는 SAC의 최대 엔트로피 RL 프레임워크를 사용하여 이 최적화를 수행한다.

### SILEM (추가 제안 알고리즘)
논문은 embodiment mismatch(전문가와 학습자의 신체 구조 차이) 문제를 해결하기 위해 SILEM을 언급한다. SILEM은 판별자의 입력값에 학습 가능한 아핀 변환(affine transform) $T$를 적용하여 신체 구조 차이를 보정한다.
- 입력 형태: $T(s_t, s_{t+1}, s_{t+2}, s_{t+3})$

## 📊 Results

### 실험 설정
- **환경**: OpenAI Gym의 Mujoco 환경 중 locomotion 태스크인 `HalfCheetah-v2`와 `Hopper-v2`를 사용하였다.
- **비교 대상 (Baselines)**: GAIfO, OPOLO, DACfO.
- **평가 지표**: 환경과의 상호작용 횟수(Timesteps)에 따른 누적 보상(Reward)을 측정하였다.

### 정량적 결과
실험 결과, SAIfO는 모든 태스크에서 기존의 IfO 기반 RAIL 알고리즘들(GAIfO, OPOLO, DACfO)보다 우수한 성능을 보였다.
- 특히 `HalfCheetah-v2`와 `Hopper-v2` 모두에서 전문가 수준의 성능(Expert level performance)에 더 빠르게 도달하며, 최종 보상 값 또한 더 높게 나타났다.
- 저자들은 공정한 비교를 위해 베이스라인 알고리즘들에 대해 하이퍼파라미터 그리드 서치와 웜업(warm-up) 단계를 추가하여 최적화한 후 실험을 진행하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구의 가장 큰 강점은 AIL이라는 복잡한 분야를 '모듈' 단위로 분해하여 해석했다는 점이다. 이를 통해 단순히 새로운 알고리즘을 제안하는 것을 넘어, "어떤 RL 백본이 더 효율적인가?" 혹은 "판별자의 입력 형태를 어떻게 바꾸면 IfO가 가능해지는가?"와 같은 연구 질문을 체계적으로 던질 수 있게 되었다.

특히 SAC가 TD3나 AlgaeDICE보다 우수한 성능을 보인 이유에 대해, 저자들은 SAC가 제공하는 stochastic update(확률적 업데이트)와 커널 트릭(kernel trick)의 영향이 중요했을 것이라고 추측한다.

### 한계 및 향후 과제
1. **입력 표현의 탐색 부족**: 본 논문에서는 주로 $(s_t, s_{t+1})$ 형태의 입력을 사용하였으나, $(s_t, s_{t+4})$ 또는 상태 차이 $(s_{t+1} - s_t)$와 같은 다른 입력 형태가 성능에 미치는 영향은 아직 탐구되지 않았다.
2. **이론적 분석의 부재**: SAC가 왜 더 좋은 성능을 내는지에 대한 정밀한 이론적 증명보다는 실험적 결과에 의존하고 있다.

## 📌 TL;DR

본 논문은 RL 기반의 Adversarial Imitation Learning 알고리즘들을 **RL 백본**과 **판별자 입력 표현**이라는 두 가지 모듈로 체계화한 **RAIL 프레임워크**를 제안한다. 이를 통해 설계된 **SAIfO** 알고리즘은 SAC를 백본으로 사용하여 기존 IfO 알고리즘들보다 빠른 학습 속도와 높은 성능을 달성하였다. 이 연구는 AIL 분야의 설계를 모듈화함으로써 향후 더 효율적인 모방 학습 알고리즘을 설계할 수 있는 기반을 마련했다는 점에서 가치가 있다.