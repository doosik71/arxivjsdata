# Diffusion-Reward Adversarial Imitation Learning

Chun-Mao Lai, Hsiang-Chun Wang, Ping-Chun Hsieh, Yu-Chiang Frank Wang, Min-Hung Chen, Shao-Hua Sun (2024)

## 🧩 Problem to Solve

본 논문은 보상 신호(reward signal)가 없는 환경에서 전문가의 시연(demonstration)만을 통해 정책을 학습하는 모방 학습(Imitation Learning)의 문제를 다룬다. 특히, 생성적 적대 모방 학습(Generative Adversarial Imitation Learning, GAIL)이 이론적 보장에도 불구하고 실제 학습 과정에서 매우 불안정하고 취약(brittle)하다는 점에 주목한다.

GAIL의 핵심은 전문가의 상태-행동 분포와 에이전트의 분포를 구분하는 판별자(discriminator)를 학습시키고, 이를 통해 얻은 보상을 이용해 정책을 업데이트하는 것이나, 이 과정에서 생성되는 보상 함수가 매끄럽지 않거나 과적합되는 경향이 있어 정책 학습의 효율성과 안정성을 저해한다. 따라서 본 연구의 목표는 Diffusion Model을 GAIL 프레임워크에 통합하여, 더 견고하고 매끄러운 보상 함수를 제공함으로써 안정적인 정책 학습을 가능하게 하는 **DRAIL(Diffusion-Reward Adversarial Imitation Learning)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Diffusion Model의 역과정(reverse process)에서 발생하는 **Diffusion Loss(노이즈 예측 오차)가 데이터의 타겟 분포 적합도를 나타낸다는 점**을 이용하는 것이다.

기존의 GAIL 판별자 대신, 전문가 데이터와 에이전트 데이터 각각에 대해 조건부 Diffusion Model을 학습시켜 판별자로 활용한다. 구체적으로, 전체 디노이징 과정을 거치지 않고 단 한 번의 디노이징 단계에서 계산된 Diffusion Loss의 차이를 통해 상태-행동 쌍의 "실제성(realness)"을 판별하는 **Diffusion Discriminative Classifier**를 설계하였다. 이를 통해 기존 GAIL보다 일반화 성능이 높고 형태가 매끄러운 보상 함수를 생성하여 정책 학습의 안정성을 획기적으로 높였다.

## 📎 Related Works

- **Behavioral Cloning (BC):** 전문가의 행동을 지도 학습 방식으로 직접 모방한다. 단순하고 효과적이지만, 전문가 데이터에 없는 상태에 도달했을 때 오류가 누적되는 compounding error 문제로 인해 일반화 능력이 떨어진다.
- **Inverse Reinforcement Learning (IRL):** 전문가의 행동을 가장 잘 설명하는 보상 함수를 추론한 뒤 이를 통해 정책을 학습한다. 그러나 동일한 행동을 유도하는 보상 함수가 여러 개 존재할 수 있는 ill-posed problem 특성이 있으며, 제약 조건 설정에 따라 일반화 성능이 제한될 수 있다.
- **Adversarial Imitation Learning (AIL):** GAIL을 포함하여 에이전트와 전문가의 분포를 적대적으로 일치시키는 방식이다. 효과적이지만 학습이 매우 불안정하다는 단점이 있다. 이를 해결하기 위해 Gradient Penalty(GAIL-GP)나 Wasserstein 거리(WAIL)를 도입하는 시도가 있었다.
- **DiffAIL:** Diffusion Model을 AIL에 통합한 선행 연구이다. 하지만 DiffAIL은 비조건부(unconditional) Diffusion Model을 사용하여 전문가와 에이전트의 행동을 명시적으로 구분하는 능력이 부족하며, 이는 보상 신호의 명확성을 떨어뜨리는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

DRAIL은 판별자(Diffusion Discriminative Classifier)와 정책(Policy)을 교대로 업데이트하는 적대적 학습 구조를 따른다.

1. **판별자 업데이트:** 에이전트가 수집한 데이터와 전문가 데이터를 사용하여 Diffusion Discriminative Classifier를 학습시킨다.
2. **보상 계산:** 학습된 판별자를 통해 현재 상태-행동 쌍에 대한 Diffusion Reward를 계산한다.
3. **정책 업데이트:** PPO(Proximal Policy Optimization)와 같은 RL 알고리즘을 사용하여 해당 보상을 최대화하도록 정책을 업데이트한다.

### Diffusion Discriminative Classifier

단순히 Diffusion Model을 통해 라벨(0 또는 1)을 생성하는 방식은 $T$번의 디노이징 단계가 필요하여 계산 비용이 너무 크다. 이를 해결하기 위해 본 논문은 단일 디노이징 단계의 loss를 이용한다.

먼저, 조건 $c \in \{c^+, c^-\}$ (전문가 $c^+$, 에이전트 $c^-$)에 따른 Diffusion Loss를 다음과 같이 정의한다.
$$L_{diff}(s, a, c) = \mathbb{E}_{t \sim T} \left[ \| \epsilon_\phi(s, a, \epsilon, t | c) - \epsilon \|^2 \right]$$
여기서 $\epsilon_\phi$는 예측된 노이즈이며, $\epsilon$은 주입된 노이즈이다. 전문가 데이터에 적합할수록 $L_{diff}(s, a, c^+)$는 작아지고, 에이전트 데이터에 적합할수록 $L_{diff}(s, a, c^-)$는 작아진다.

이 두 loss의 차이를 이용하여 [0, 1] 범위의 확률값을 출력하는 판별자 $D_\phi$를 다음과 같이 구성한다.
$$D_\phi(s, a) = \frac{e^{-L_{diff}(s, a, c^+)}}{e^{-L_{diff}(s, a, c^+)} + e^{-L_{diff}(s, a, c^-)}} = \sigma(L_{diff}(s, a, c^-) - L_{diff}(s, a, c^+))$$
여기서 $\sigma$는 시그모이드 함수이다.

### 훈련 목표 및 손실 함수

판별자 $D_\phi$는 Binary Cross-Entropy (BCE) 손실 함수를 통해 최적화된다.
$$L_D = \mathbb{E}_{(s, a) \in \tau_E} [-\log(D_\phi(s, a))] + \mathbb{E}_{(s, a) \in \tau_i} [-\log(1 - D_\phi(s, a))]$$
$\tau_E$는 전문가 궤적, $\tau_i$는 에이전트 궤적을 의미한다.

### Diffusion Reward 및 정책 학습

정책 $\pi_\theta$는 다음의 Diffusion Reward를 최대화하도록 학습된다.
$$r_\phi(s, a) = \log(D_\phi(s, a)) - \log(1 - D_\phi(s, a))$$
이 보상 함수는 $D_\phi$의 출력이 1에 가까울수록(전문가와 유사할수록) 높은 값을 가지며, 로그-오즈(log-odds) 형태를 취함으로써 정책 학습에 더 안정적인 신호를 제공한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업:** MAZE(내비게이션), FETCHPUSH(조작), HANDROTATE(정밀 조작), ANTREACH(내비게이션), WALKER(보행), CARRACING(이미지 기반 레이싱) 등 총 6가지의 다양한 연속 제어 환경에서 평가하였다.
- **비교 대상:** BC, Diffusion Policy, GAIL, GAIL-GP, WAIL, DiffAIL.
- **측정 지표:** Success Rate(성공률) 및 Return(총 보상).

### 주요 결과

- **전반적 성능:** DRAIL은 거의 모든 환경에서 기존 방법론보다 우수하거나 경쟁력 있는 성능을 보였다. 특히 DiffAIL과의 비교에서 6개 작업 중 5개에서 더 높은 성능을 기록하며, 조건부 Diffusion Model의 유효성을 입증하였다.
- **일반화 능력 (Generalizability):** FETCHPUSH 작업에서 초기 상태와 목표 위치에 노이즈를 추가한 실험 결과, DRAIL은 노이즈 수준이 $2.0\times$인 극한 환경에서도 95% 이상의 성공률을 유지하며 가장 강력한 강건성을 보였다. 반면 Diffusion Policy는 79.2%, DiffAIL은 매우 불안정한 결과를 보였다.
- **데이터 효율성 (Data Efficiency):** 전문가 데이터의 양을 줄였을 때, DRAIL은 다른 방법론보다 훨씬 적은 데이터로도 빠르게 학습하는 모습을 보였다. 특히 WALKER 작업에서 단 1개의 궤적만으로도 높은 리턴 값을 유지하였다.
- **보상 함수 시각화:** SINE 환경 실험을 통해 GAIL의 보상은 전문가 데이터에 과적합되어 뾰족한 형태를 띠는 반면, DRAIL의 보상은 더 넓고 매끄러운(smooth) 분포를 가져, 에이전트가 전문가 궤적에서 벗어났을 때도 올바른 방향으로 가이드할 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문의 가장 큰 성과는 Diffusion Model의 score-matching 특성(Loss가 분포의 밀도와 연관됨)을 판별자의 메커니즘으로 영리하게 전환한 점이다. 이를 통해 GAIL의 고질적인 문제인 '불안정한 보상 신호'를 '매끄러운 Diffusion 보상'으로 대체하여 학습 안정성을 확보하였다. 또한, 조건부 라벨($c^+, c^-$)을 도입함으로써 전문가와 에이전트의 분포 경계를 명시적으로 학습하게 하여, DiffAIL과 같은 비조건부 방식보다 훨씬 명확한 분류 신호를 생성하였다.

### 한계 및 향후 과제

논문에서 명시적으로 언급된 한계는 적으나, 미래 연구 방향으로 Wasserstein 거리나 $f$-divergence와 같은 다른 거리 측정 지표를 탐색하여 학습 안정성을 더욱 높일 가능성을 제시하고 있다. 또한, 실제 로봇 환경이나 자율 주행과 같은 더 복잡한 도메인으로의 확장 가능성을 언급하였다.

비판적으로 해석하자면, Diffusion Model 기반의 판별자가 연산량 측면에서 단순 MLP 판별자보다 무거울 수 있으나, 본 논문은 단일 디노이징 스텝만을 사용함으로써 이 비용 문제를 효율적으로 해결하였다고 판단된다.

## 📌 TL;DR

DRAIL은 GAIL의 불안정한 학습 문제를 해결하기 위해 **조건부 Diffusion Model을 판별자로 통합**한 모방 학습 프레임워크이다. Diffusion Loss의 차이를 이용해 매끄럽고 견고한 보상 함수를 생성하며, 이를 통해 **높은 일반화 성능, 우수한 데이터 효율성, 그리고 학습 안정성**을 달성하였다. 특히 데이터가 부족하거나 노이즈가 많은 환경에서 기존 AIL 방법론 대비 압도적인 성능 향상을 보여, 향후 로보틱스 및 자율 주행 분야의 모방 학습에 중요한 기여를 할 것으로 기대된다.
