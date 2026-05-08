# How to Leverage Diverse Demonstrations in Offline Imitation Learning

Sheng Yue, Jiani Liu, Xingyuan Hua, Ju Ren, Sen Lin, Junshan Zhang, Yaoxue Zhang (2024)

## 🧩 Problem to Solve

본 논문은 전문가 데이터가 부족하고 불완전한 시연(imperfect demonstrations) 데이터가 섞여 있는 환경에서의 Offline Imitation Learning (IL) 문제를 해결하고자 한다. 실제 환경에서는 완벽한 전문가 데이터를 대량으로 수집하는 것이 매우 비용이 많이 들기 때문에, 다양한 품질의 데이터가 포함된 데이터셋을 활용하는 것이 필수적이다.

기존의 접근 방식들은 주로 전문가 데이터와 상태-행동(state-action)의 유사성을 기준으로 데이터를 선택하여 학습하였다. 그러나 이러한 방식은 전문가의 시연과는 다르지만 결과적으로는 유익한 행동을 유발하는 '다양한 행동(diverse behaviors)'에 포함된 귀중한 정보를 무시한다는 한계가 있다. 특히, 에이전트가 전문가 데이터에 없는 상태(out-of-distribution state)에 진입했을 때, 다시 전문가 상태로 복귀하는 방법을 배우지 못해 발생하는 Error Compounding(오차 누적) 문제는 BC(Behavior Cloning) 계열 알고리즘의 고질적인 문제이다.

따라서 본 논문의 목표는 불완전한 데이터셋 내에서 노이즈를 제거하고, 전문가의 행동뿐만 아니라 전문가 상태로 유도하는 유익한 다양성 행동을 효과적으로 추출하여 정책의 강건성과 일반화 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 행동의 가치를 판단하는 기준을 '전문가 행동과의 유사성'이 아닌, 그 행동을 수행한 후 도달하게 되는 **'결과 상태(resultant state)'**로 변경하는 것이다.

즉, 어떤 행동이 전문가의 행동과 닮지 않았더라도, 그 행동을 통해 도달한 상태가 전문가 데이터 집합 내에 존재한다면, 이는 에이전트를 다시 정상 궤도로 복귀시키는 유익한 행동으로 간주한다. 이러한 직관을 바탕으로 dynamics 정보를 명시적으로 활용하여 데이터셋을 구성하고, 추출된 데이터 간의 간섭을 최소화하는 가중치 기반의 가벼운 BC 알고리즘인 **ILID (Offline Imitation Learning with Imperfect Demonstrations)**를 제안한다.

## 📎 Related Works

기존의 Offline IL 연구들은 크게 세 가지 방향으로 나뉜다.

1. **Behavior Cloning (BC) 및 확장 모델:** 단순 BC는 Covariate Shift에 취약하며, 이를 해결하기 위해 DWBC나 ISWBC 같은 방법들이 제안되었다. 이들은 판별자(discriminator)나 중요도 샘플링(importance sampling)을 통해 전문가 행동과 유사한 데이터에 가중치를 두어 학습한다. 하지만 이들은 상태-행동 유사성에 의존하므로, 전문가 데이터의 커버리지가 좁을 경우 복구 능력을 학습하지 못한다.
2. **Offline Inverse Reinforcement Learning (IRL):** 보상 함수를 직접 추론하여 Offline RL 과정으로 해결하려는 시도이다. 하지만 보상 함수를 정확히 정의하기 어렵고, 학습된 World Model의 불확실성으로 인해 고차원 환경에서 학습이 불안정하며 하이퍼파라미터 민감도가 높다는 단점이 있다.
3. **Offline RL 기반 접근:** 레이블이 없는 데이터를 활용하려는 시도가 있으나, 대량의 데이터가 필요하거나 명시적인 보상 신호가 있어야 한다는 제약이 있다.

본 논문은 보상 함수 학습이라는 복잡한 과정 없이, 상태 판별자와 결과 상태 기반의 데이터 선택이라는 단순하고 효율적인 구조를 통해 기존 방식들의 한계를 극복한다.

## 🛠️ Methodology

### 1. 불완전한 행동의 선택 (Selection of Imperfect Behaviors)

본 논문은 "전문가 데이터에 없는 상태에 있을 때, 무작위 행동을 하는 것보다 전문가 상태로 전이되는 행동을 취하는 것이 더 이득이다"라는 가설(Hypothesis 4.1)을 세운다. 이를 위해 다음과 같은 절차로 데이터를 선택한다.

**가. 상태 판별자 학습**
먼저, 상태 $s$가 전문가 상태인지 아닌지를 구분하는 state-only discriminator $d$를 학습시킨다.
$$\max_{d} \mathbb{E}_{s \sim \mathcal{D}_e}[\log d(s)] + \mathbb{E}_{s \sim \mathcal{D}_u}[\log(1-d(s))]$$
여기서 $\mathcal{D}_e$는 전문가 데이터, $\mathcal{D}_u$는 전체 데이터(전문가+불완전)이다. 학습된 $d^*(s) > \sigma$ (임계값)인 상태를 전문가 상태로 식별한다.

**나. 보완 데이터셋 $\mathcal{D}_s$ 구축**
불완전 데이터셋 $\mathcal{D}_b$에서 위에서 식별된 전문가 상태 $s_{i,h}$로 도달하게 만든 원인 행동들을 추출한다. 구체적으로, rollback step $K$를 설정하여 해당 상태 이전의 $K$개 상태-행동 쌍을 수집하여 보완 데이터셋 $\mathcal{D}_s$를 구성한다.
$$\mathcal{D}_s \leftarrow \mathcal{D}_s \cup \{(k, s_{i,h-k}, a_{i,h-k})\}, \quad k=1: \min\{h-1, K\}$$

### 2. 정책 학습 (Policy Learning)

추출된 $\mathcal{D}_s$는 전문가 데이터보다 품질이 낮을 수 있으므로, 단순 합산 학습 시 행동 간 간섭(interference)이 발생할 수 있다. 이를 방지하기 위해 다음과 같은 가중치 기반 BC 목적 함수를 제안한다.

$$\max_{\pi} \mathbb{E}_{\mathcal{D}_e}[\log(\pi(a|s))] + \mathbb{E}_{\mathcal{D}_s}[\mathbb{1}(\mathcal{D}_e(s) = 0) \log(\pi(a|s))]$$

이 식은 전문가 상태에서는 전문가의 행동을 정확히 따르고, 전문가 상태가 아닌 영역에서만 $\mathcal{D}_s$의 행동을 모방하도록 강제한다. 실제 구현에서는 다음과 같은 가중치 $\alpha$와 $\beta$를 사용한다.

$$\max_{\pi} J(\pi) = \mathbb{E}_{\mathcal{D}_u}[\alpha(s,a) \log(\pi(a|s))] + \mathbb{E}_{\mathcal{D}_s}[\beta(s,a) \log(\pi(a|s))]$$

- **$\alpha(s,a)$**: 전문가 데이터의 지지도를 높이기 위한 중요도 샘플링 가중치이다. 별도의 판별자 $D^*$를 통해 $\alpha(s,a) = \frac{D^*(s,a)}{1-D^*(s,a)}$로 계산한다.
- **$\beta(s,a)$**: $\mathbb{1}(\mathcal{D}_e(s) = 0)$를 근사하며, 상태 판별자 $d^*$의 출력이 임계값 $\sigma$ 이하일 때만 1이 된다 ($\beta(s,a) = \mathbb{1}(d^*(s) \leq \sigma)$).

## 📊 Results

### 실험 설정

- **벤치마크:** MuJoCo, Adroit, AntMaze, FrankaKitchen, vision-based Robomimic 등 총 6개 도메인, 21개 작업에서 평가하였다.
- **비교 대상:** BCE, BCU, DWBC, ISWBC, CSIL, MLIRL.
- **측정 지표:** Normalized score를 사용하여 성능을 측정하였다.

### 주요 결과

- **정량적 성능:** ILID는 21개 벤치마크 중 20개에서 SOTA 성능을 달성하였으며, 기존 방법들보다 통상적으로 2~5배 높은 성능을 보였다.
- **데이터 효율성:** 매우 적은 수의 전문가 궤적(예: MuJoCo에서 단 1개)만으로도 기존 방법들보다 훨씬 빠르게 전문가 성능에 도달하였다.
- **강건성:** 불완전 데이터의 품질이 낮거나 양이 변하더라도 일관되게 높은 성능을 유지하였다. 특히, Rollback step $K$ 값이 커질수록 초기 성능이 향상되다가 일정 수준에서 수렴하는 모습을 보여 가설 4.1을 뒷받침하였다.
- **계산 효율성:** 학습 시간이 단순 BC와 거의 유사하여(약 30~40분), 계산 비용 증가 없이 성능을 대폭 향상시켰다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구의 가장 큰 성과는 행동의 유사성이 아닌 **'결과 상태'**라는 새로운 기준을 도입하여 Offline IL의 고질적인 문제인 Error Compounding을 해결했다는 점이다. 이론적 분석(Theorem 4.2)을 통해 $\mathcal{D}_s$를 활용한 정책 $\tilde{\pi}$가 전문가 데이터만 사용한 BC보다 suboptimality 바운드가 낮음을 증명하였으며, 이는 실제 실험에서 에이전트가 전문가 궤적에서 벗어났을 때 다시 복귀하는 능력으로 나타났다.

### 한계 및 비판적 해석

논문에서 언급되었듯이, 본 방법론은 전문가 데이터와 불완전 데이터 사이에 어느 정도의 **상태 중첩(state overlap)**이 존재한다는 가정을 전제로 한다. 만약 불완전 데이터셋의 어떤 행동도 전문가 상태에 도달하지 못한다면, $\mathcal{D}_s$를 구축할 수 없으므로 본 방법론은 작동하지 않는다. 즉, "완전히 무작위인 데이터"보다는 "어느 정도 목표 지점 근처까지는 가본 데이터"가 있을 때 효과적이다.

또한, $\beta(s,a)$ 가중치를 통한 간섭 제거가 필수적임을 ablation study를 통해 보여주었는데, 이는 $\mathcal{D}_s$ 내부의 데이터 품질이 균일하지 않음을 시사하며, 향후에는 $\mathcal{D}_s$ 내부에서도 더 정밀한 필터링 메커니즘이 필요할 수 있다.

## 📌 TL;DR

본 논문은 Offline IL에서 불완전한 시연 데이터를 활용하기 위해, 전문가 행동과 닮았는지가 아니라 **'전문가 상태로 이끄는가'**를 기준으로 데이터를 선택하는 **ILID** 알고리즘을 제안한다. 이 방법은 결과 상태 기반의 데이터 추출과 가중치 기반 BC를 통해 21개 벤치마크 중 20개에서 압도적인 성능 향상을 보였으며, 특히 데이터 효율성과 복구 능력이 뛰어나다. 이는 전문가 데이터가 극도로 부족한 실제 환경에서 매우 실용적인 해결책이 될 가능성이 높다.
