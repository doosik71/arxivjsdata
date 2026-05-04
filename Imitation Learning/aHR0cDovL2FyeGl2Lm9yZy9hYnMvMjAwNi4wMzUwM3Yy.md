# Wasserstein Distance guided Adversarial Imitation Learning with Reward Shape Exploration

Ming Zhang, Yawei Wang, Xiaoteng Ma, Li Xia, Jun Yang, Zhiheng Li, Xiu Li (2020)

## 🧩 Problem to Solve

본 논문은 고차원 연속 제어 작업(high-dimensional continuous tasks)에서 전문가의 시연(demonstrations)을 통해 정책을 학습하는 Imitation Learning(IL)의 효율성과 안정성 문제를 해결하고자 한다. 기존의 대표적인 프레임워크인 Generative Adversarial Imitation Learning(GAIL)은 다음과 같은 세 가지 주요 한계점을 가진다.

첫째, GAIL은 두 분포 간의 거리를 측정하기 위해 Jensen-Shannon (JS) divergence를 사용하는데, 이는 수학적 특성상 Gradient Vanishing 문제를 야기하여 판별자(discriminator)가 제공하는 보상 신호를 불안정하게 만든다. 둘째, 환경과의 상호작용 횟수가 지나치게 많아 Sample Efficiency가 낮고 학습 속도가 느리다는 단점이 있다. 셋째, 기존의 GAIL 및 그 확장 모델들은 주로 고정된 로그 형태(logarithmic form)의 보상 함수를 사용하는데, 이는 특정 복잡한 환경에서 보상 편향(reward bias)을 일으켜 최적이 아닌 정책(sub-optimal behavior)으로 유도하는 경향이 있다.

따라서 본 연구의 목표는 Wasserstein distance를 도입하여 학습의 안정성을 높이고, PPO 알고리즘을 통해 효율성을 개선하며, 다양한 보상 함수 형태(reward shape)를 탐색하여 작업별 최적의 보상 구조를 찾는 WDAIL 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 적대적 모방 학습(Adversarial Imitation Learning) 과정에서 분포 측정 지표와 최적화 알고리즘, 그리고 보상 함수의 형태를 개선하는 것이다.

1. **Wasserstein Distance 도입**: JS divergence 대신 $L_1$-Wasserstein distance를 사용하여 판별자와 생성자(정책) 간의 학습 안정성을 확보하고 Gradient Vanishing 문제를 완화하였다.
2. **PPO (Proximal Policy Optimization) 적용**: 기존 GAIL에서 사용하던 TRPO 대신 구현이 더 간단하고 샘플 효율성이 높은 PPO를 정책 최적화 단계에 적용하였다.
3. **Reward Shape Exploration**: 고정된 로그 형태의 보상 함수에서 벗어나, 다양한 수학적 형태(양수, 음수, 선형 등)의 보상 함수를 설계하고 실험함으로써 각 작업에 가장 적합한 보상 구조가 무엇인지 분석하였다.

## 📎 Related Works

기존의 모방 학습은 크게 Behavior Cloning(BC)과 Inverse Reinforcement Learning(IRL)으로 나뉜다. BC는 지도 학습 방식으로 접근하여 데이터가 충분할 때 유용하지만, 고차원 환경에서는 전문가의 행동을 정확히 복제하지 못하는 경우가 많다. IRL은 전문가의 행동을 설명할 수 있는 보상 함수를 찾는 방식이지만, 동일한 행동에 대해 여러 보상 함수가 존재할 수 있는 Ill-posed 문제이며 계산 비용이 매우 높다.

GAIL은 GAN의 구조를 차용하여 모델 없이(model-free) 직접 정책을 추출함으로써 BC와 IRL의 단점을 극복하고 뛰어난 성능을 보였다. 그러나 앞서 언급한 JS divergence 기반의 불안정성과 샘플 효율성 문제가 지속적으로 제기되었다. 또한, Adversarial Inverse Reinforcement Learning(AIRL)과 같은 연구들이 특정 보상 함수 형태를 제안했으나, 이는 생존 보너스(survival bonus)가 있는 환경에서 부적절한 정책을 유도하는 등 보상 편향 문제가 존재했다. 본 논문은 이러한 기존 연구들의 한계를 지적하며, 보상 함수의 형태가 학습 성능에 결정적인 영향을 미친다는 점에 주목하여 이를 체계적으로 탐색하였다.

## 🛠️ Methodology

### 전체 파이프라인
WDAIL은 판별자 $D$와 정책 $\pi$가 서로 경쟁하는 적대적 학습 구조를 가진다. 판별자는 전문가의 궤적과 정책이 생성한 궤적을 구분하려 하며, 정책은 판별자가 구분하지 못하도록 전문가의 행동을 모방하며 보상을 최대화한다.

### 1. Wasserstein Distance 기반의 적대적 학습
본 논문은 정책의 점유 측정치(occupancy measure) $\rho^\pi$와 전문가의 점유 측정치 $\rho^E$ 사이의 거리를 $L_1$-Wasserstein distance로 정의한다. 목적 함수는 다음과 같다.

$$\min_{\pi \in \Pi} \{ -H(\pi) + W_1^d(\rho^\pi, \rho^E) \}$$

여기서 $H(\pi)$는 정책의 엔트로피이며, $W_1^d$는 Wasserstein distance이다. Kantorovich-Rubinstein duality에 의해 이는 다음과 같이 표현된다.

$$W_1^d(\rho^\pi, \rho^E) = \max_{\|d\|_L \le 1} \{ \mathbb{E}_{\pi^E}[d(s,a)] - \mathbb{E}_\pi[d(s,a)] \}$$

여기서 $d(s,a)$는 1-Lipschitz 조건을 만족해야 하는 판별자 함수이다. 실제 구현에서는 이 조건을 강제하기 위해 Gradient Penalty ($L_{gp}$)를 사용한다.

$$L_{gp} = (\|\nabla_{(\hat{s},\hat{a})} D((\hat{s},\hat{a}))\|_2 - 1)^2$$

### 2. 정책 최적화 (PPO)
정책 $\pi_\theta$를 업데이트하기 위해 Clipped PPO 목적 함수를 사용한다. 이는 정책 업데이트 폭을 제한하여 학습의 안정성을 높인다.

$$L^{CLIP}(\theta) = \hat{\mathbb{E}} [ \min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t) ]$$

여기서 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$는 확률 비율이며, $\hat{A}_t$는 Advantage 추정치이다.

### 3. 보상 함수 형태 설계 (Reward Shaping)
판별자의 출력 $x = D(s,a)$를 입력으로 하여 다양한 보상 함수 $f_{rew}(x)$를 정의한다. (여기서 $\sigma(x)$는 시그모이드 함수이다.)

- **양수 보상 형태**: $\sigma(x)$, $e^x$, $-\log(1-\sigma(x))$
- **음수 보상 형태**: $-e^{-x}$, $\log(\sigma(x))$
- **선형/무편향 형태**: $x$ (이는 $\log(\sigma(x)) - \log(1-\sigma(x))$와 동등함)

### 4. 학습 절차
전체 과정은 다음과 같은 Minimax 게임으로 정의된다.

$$\min_\theta \max_w L_{wdail}(\theta, w) = L_{wd} - \lambda L_{gp}$$

1. **판별자 업데이트**: 정책 궤적 버퍼 $B^\pi$와 전문가 데이터셋 $B^E$에서 샘플을 추출하여 $L_{wd}$와 $L_{gp}$를 통해 판별자 파라미터 $w$를 업데이트한다.
2. **보상 생성**: 업데이트된 판별자를 통해 각 상태-행동 쌍에 대한 보상 $r_t = f_{rew}(D_w(s_t, a_t))$를 계산한다.
3. **정책 업데이트**: 계산된 보상을 바탕으로 Advantage $\hat{A}$를 구하고, PPO의 $L^{CLIP}$ 목적 함수를 통해 정책 파라미터 $\theta$를 업데이트한다.

## 📊 Results

### 실험 설정
- **환경**: MuJoCo의 고차원 연속 제어 작업 (Hopper, Walker2d, HalfCheetah).
- **비교 대상**: BC, GAIL, Random Policy.
- **지표**: 전문가 성능과 랜덤 정책 성능 사이로 정규화된 $[0, 1]$ 범위의 점수.
- **변수**: 전문가 궤적의 수 (1, 5, 10, 50개) 및 보상 함수의 형태.

### 주요 결과
1. **보상 형태의 영향**:
    - $\sigma(x)$, $e^x$, $-\log(1-\sigma(x))$와 같은 **양수 보상 형태**가 세 가지 모든 작업에서 가장 효과적이었으며, 전문가 데이터의 양에 관계없이 강건한 성능을 보였다.
    - 무편향 보상 함수 $x$ 및 $\log(\sigma(x))$는 HalfCheetah에서는 양호했으나, 다른 환경에서는 랜덤 수준의 낮은 성능을 보였다.
    - $-e^{-x}$ 형태의 음수 보상은 데이터가 매우 적을 때만 일부 효과가 있었으며 전반적으로 성능이 낮았다.

2. **알고리즘 간 비교**:
    - WDAIL은 전문가 데이터가 적은 상황에서도 BC나 GAIL보다 우수한 성능을 보였으며, 특히 학습 곡선이 매우 매끄럽고 빠르게 수렴하는 양상을 보였다.
    - BC는 데이터가 매우 많을 때만 성능이 향상되었으며, GAIL은 WDAIL에 비해 학습 과정이 불안정했다.

3. **샘플 효율성**: WDAIL은 1백만 번의 환경 상호작용 이내에 수렴하는 높은 샘플 효율성을 보여주었다.

## 🧠 Insights & Discussion

본 논문은 적대적 모방 학습에서 단순히 판별자를 학습시키는 것뿐만 아니라, 그 판별자의 출력을 어떤 형태의 보상으로 변환하여 정책에 전달하느냐가 성능에 결정적인 영향을 미친다는 점을 실증적으로 증명하였다. 특히 '양수 보상' 형태가 MuJoCo 환경에서 일종의 survival bonus 역할을 하여 에이전트가 더 오래 생존하며 전문가의 행동을 학습하도록 유도했음을 알 수 있다.

또한, JS divergence를 Wasserstein distance로 대체함으로써 적대적 학습의 고질적인 문제인 Gradient Vanishing을 해결하고 학습의 안정성을 확보하였다. PPO의 도입은 구현의 단순함과 효율성을 동시에 잡은 적절한 선택이었다고 판단된다.

다만, 한계점으로는 전문가 시연 데이터의 품질이 낮거나 변동성(variance)이 매우 큰 경우, 데이터의 양이 많아질수록 오히려 성능이 저하되는 현상이 HalfCheetah 환경에서 관찰되었다. 이는 모델이 불완전한 전문가 데이터를 과하게 모방하려 할 때 발생하는 문제로 보이며, 향후 연구에서 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

본 논문은 GAIL의 불안정성과 샘플 효율성, 보상 편향 문제를 해결하기 위해 **Wasserstein distance**와 **PPO**, 그리고 **다양한 보상 함수 형태 탐색(Reward Shaping)**을 결합한 **WDAIL** 알고리즘을 제안한다. 실험 결과, 양수 형태의 보상 함수를 사용한 WDAIL이 MuJoCo의 복잡한 연속 제어 작업에서 기존 GAIL 및 BC보다 훨씬 안정적이고 빠른 학습 성능을 보임을 확인하였다. 이는 향후 적대적 모방 학습에서 보상 함수의 설계가 매우 중요한 요소임을 시사한다.