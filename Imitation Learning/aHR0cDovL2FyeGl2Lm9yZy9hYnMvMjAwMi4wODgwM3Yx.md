# Support-weighted Adversarial Imitation Learning

Ruohan Wang, Carlo Ciliberto, Pierluigi Amadori, Yiannis Demiris (2020)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning (AIL) 알고리즘들이 직면한 두 가지 주요 실무적 문제인 **학습 불안정성(Training Instability)**과 **암시적 보상 편향(Implicit Reward Bias)**을 해결하고자 한다.

AIL은 전문가의 시연(demonstration)을 통해 보상 함수를 학습하고 이를 강화학습(RL)에 활용하는 강력한 방법론이지만, 적대적 학습(adversarial training) 특성상 기울기 소실(vanishing gradients)이나 전문가 데이터에 대한 과적합 문제가 발생하며, 이는 학습의 불안정성으로 이어진다. 또한, 목표 지향적 작업에서 에이전트가 작업을 완료하는 대신 목표 주변에서 계속 머물며 보상을 쌓으려는 '생존 편향(survival bias)' 문제가 발생한다.

반면, 전문가 정책의 Support Estimation을 통해 고정된 보상 함수를 구축하는 방법(예: RED)은 적대적 학습의 불안정성을 피할 수 있지만, 전문가 데이터가 희소(sparse)할 경우 성능이 급격히 저하되는 한계가 있다. 따라서 본 논문은 이 두 가지 접근 방식의 장점을 결합하여, 적은 양의 전문가 데이터로도 안정적이고 효율적으로 모방 학습을 수행하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **적대적 보상(Adversarial Reward)에 전문가 정책의 Support Estimation으로부터 유도된 신뢰도 점수(Confidence Score)를 가중치로 적용**하는 것이다.

구체적으로, 전문가의 데이터 분포가 집중된 영역(Support)에서는 적대적 보상을 적극적으로 활용하여 탐색을 촉진하고, 전문가 데이터가 없는 영역에서는 보상을 억제함으로써 에이전트가 의도치 않은 행동으로 발산하는 것을 방지한다. 이러한 설계를 통해 SAIL은 기존 AIL 알고리즘의 탐색 능력과 Support Estimation의 안정성을 동시에 확보하며, 기존 AIL 프레임워크(GAIL, DAC 등) 위에 쉽게 적용할 수 있는 일반적인 프레임워크를 제공한다.

## 📎 Related Works

### 1. Behavioral Cloning (BC)
전문가 궤적을 지도 학습(supervised learning) 방식으로 직접 모방하는 방법이다. 구현이 간단하지만, 에이전트의 상태 분포가 전문가의 분포에서 벗어나는 **분포 드리프트(Distributional Drift)** 문제에 취약하며, 이를 해결하기 위해 방대한 양의 데이터가 필요하다.

### 2. Adversarial Imitation Learning (AIL) 및 GAIL
Inverse Reinforcement Learning (IRL)과 GAN의 구조를 결합하여 전문가와 에이전트의 상태-행동 분포를 일치시키는 방식이다. GAIL은 적은 양의 데이터로도 높은 성능을 보이지만, 적대적 학습의 고질적인 문제인 학습 불안정성과 보상 편향 문제가 존재한다.

### 3. Support Estimation 기반 모방 학습 (RED)
Random Network Distillation (RND)을 활용하여 전문가 정책의 Support를 추정하고 이를 기반으로 보상을 생성하는 RED(Random Expert Distillation) 방식이 제안되었다. 이는 적대적 학습을 배제하여 매우 안정적이지만, 전문가 데이터가 부족할 경우 보상 함수의 품질이 떨어져 성능이 저하되는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인
SAIL은 전문가 데이터로부터 Support Estimation을 수행하여 고정된 가중치 함수 $r^{red}$를 먼저 학습한 뒤, 이를 적대적 보상 $\hat{r}^{gail}$과 결합하여 최종 보상을 산출하고, 이를 통해 정책을 업데이트하는 구조를 가진다.

### 2. 주요 구성 요소 및 방정식

#### (1) Support Estimation 보상 ($r^{red}$)
전문가 데이터의 Support를 추정하기 위해 RED 방식을 사용한다. 무작위로 초기화된 네트워크 $f_\theta$와 전문가 데이터로 학습된 네트워크 $f_{\hat{\theta}}$ 사이의 거리(embedding distance)를 측정한다.
$$\min_{\hat{\theta}} \mathbb{E}_{s,a \sim \pi^E} \|f_{\hat{\theta}}(s,a) - f_\theta(s,a)\|_2^2$$
이를 통해 다음과 같은 보상 함수를 정의한다.
$$r^{red}(s,a) = \exp(-\sigma \|f_{\hat{\theta}}(s,a) - f_\theta(s,a)\|_2^2)$$
여기서 $\sigma$는 하이퍼파라미터이며, $r^{red}$는 상태-행동 쌍 $(s,a)$가 전문가의 Support 내에 있을 때 높은 값을 가진다.

#### (2) 적대적 보상 ($\hat{r}^{gail}$)
일반적인 $\log$ 형태의 보상 대신, 학습 안정성을 위해 범위가 제한된(bounded) 보상을 사용한다.
$$\hat{r}^{gail}(s,a) = 1 - D(s,a) \in [0,1]$$
여기서 $D(s,a)$는 전문가와 에이전트를 구분하는 판별자(Discriminator)이다.

#### (3) SAIL 최종 보상 함수 ($r^{sail}$)
위의 두 신호를 곱하여 최종 보상을 정의한다.
$$r^{sail}(s,a) = r^{red}(s,a) \cdot \hat{r}^{gail}(s,a)$$
이 수식에서 $r^{red}$는 적대적 보상 $\hat{r}^{gail}$의 신뢰도에 대한 가중치 역할을 한다. 즉, 전문가 데이터가 없는 영역($r^{red} \approx 0$)에서는 적대적 보상이 아무리 높더라도 최종 보상을 낮게 유지하여 에이전트의 무분별한 탐색을 억제한다.

### 3. 학습 절차
1. 전문가 궤적 $\tau^E$를 사용하여 $r^{red}$를 사전 학습한다.
2. 판별자 $D$와 정책 $\pi$를 교대로 업데이트한다.
   - 판별자 $D$는 전문가 분포와 에이전트 분포를 구분하도록 학습된다.
   - 정책 $\pi$는 $r^{sail}$을 최대화하도록 TRPO(Trust Region Policy Optimization) 알고리즘을 통해 업데이트된다.

## 📊 Results

### 1. Mujoco Control Tasks 실험
- **환경 및 지표**: Hopper, Reacher, HalfCheetah, Walker2d, Ant, Humanoid 등 6개 작업에서 에피소드당 보상을 측정하였다. 전문가 데이터는 20개 샘플 간격으로 서브샘플링하여 난이도를 높였다.
- **결과**:
    - **성능**: SAIL-b(Bounded reward 버전)가 대부분의 작업에서 GAIL, RED, BC보다 우수한 성능을 보였다.
    - **안정성**: 특히 Humanoid 작업에서 SAIL-b는 다른 방법론들에 비해 표준 편차가 현저히 낮았다. 이는 GAIL-b가 평균 성능은 비슷하더라도 가끔씩 심각하게 추락(crash)하는 현상이 발생하는 반면, SAIL-b는 일관되게 전문가를 모방함을 의미한다.
    - **표본 효율성**: Reacher, Ant, Humanoid 작업에서 SAIL-b는 더 빠른 수렴 속도와 높은 샘플 효율성을 보였다.

### 2. Lunar Lander 실험 (보상 편향 검증)
- **설정**: 환경의 조기 종료(early termination) 기능을 제거한 'no-terminal' 환경을 구축하여, 에이전트가 단순히 공중에 떠서 보상을 얻으려는 생존 편향(survival bias)이 발생하는지 확인하였다.
- **결과**:
    - GAIL은 목표 지점에서 착륙하지 않고 주변을 맴도는(hovering) 경향을 보였으나, SAIL-b는 성공적으로 착륙을 수행하였다.
    - **분석**: 목표 상태(goal states)에서 'no op'(아무것도 하지 않음) 행동에 대해 SAIL-b가 다른 알고리즘보다 훨씬 높은 보상을 할당함을 확인하였다. 이는 전문가가 목표 지점에서 정지하는 행동이 Support 내에 존재하기 때문에 $r^{red}$가 이를 강화했기 때문이다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 근거
SAIL은 적대적 학습의 '탐색 능력'과 Support Estimation의 '안정성'을 결합한 시너지 효과를 낸다. 이론적으로 SAIL은 전문가 데이터에 대한 샘플 복잡도 측면에서 RED와 GAIL 중 더 빠른 학습 속도를 가진 쪽의 성능을 보장하며, 최소한 GAIL만큼 효율적임을 입증하였다.

### 2. 보상 범위 제한의 중요성
본 논문에서는 $\log D(s,a)$ 대신 $1-D(s,a)$와 같은 bounded reward를 사용했을 때 성능과 안정성이 향상됨을 보였다. 이는 Support Estimation 점수($r^{red}$)와 적대적 보상 점수가 동일한 범위([0, 1])를 가짐으로써 서로 대등하게 보상 함수에 기여할 수 있게 하기 때문이다.

### 3. 한계 및 비판적 해석
SAIL이 생존 편향을 상당 부분 완화했지만, 완전히 제거하지는 못했다. 저자들은 여전히 보상이 비음수(non-negative)라는 점 때문에 에이전트가 착륙과 호버링 사이에서 진동하는 현상이 발생한다고 분석하였다. 이는 향후 보상 함수의 부호 설계나 추가적인 제약 조건 연구가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 적대적 모방 학습(AIL)의 불안정성과 보상 편향 문제를 해결하기 위해, **전문가 정책의 Support Estimation 결과($r^{red}$)를 적대적 보상($\hat{r}^{gail}$)의 가중치로 사용하는 SAIL 프레임워크**를 제안한다. SAIL은 전문가 데이터가 희소한 영역에서의 잘못된 보상 신호를 마스킹함으로써 학습의 안정성을 획기적으로 높였으며, 특히 Mujoco와 Lunar Lander 환경에서 기존 GAIL 및 RED보다 더 강건하고 일관된 성능을 입증하였다. 이 연구는 서로 다른 성격의 보상 신호를 결합하는 것이 모방 학습의 효율성을 높이는 유망한 방향임을 보여준다.