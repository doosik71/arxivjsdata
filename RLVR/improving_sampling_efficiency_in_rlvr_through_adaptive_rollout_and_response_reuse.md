# IMPROVING SAMPLING EFFICIENCY IN RLVR THROUGH ADAPTIVE ROLLOUT AND RESPONSE REUSE

Yuheng Zhang, Wenlin Yao, Changlong Yu, Yao Liu, Qingyu Yin, Bing Yin, Hyokun Yun, Lihong Li (2025)

## 🧩 Problem to Solve

본 논문은 LLM의 추론 능력을 향상시키기 위한 Reinforcement Learning with Verifiable Rewards (RLVR) 과정에서 발생하는 **Vanishing Advantage** 문제를 해결하고자 한다.

RLVR의 대표적인 알고리즘인 Group Relative Policy Optimization (GRPO)는 하나의 프롬프트에 대해 여러 개의 응답 그룹을 생성하고, 그룹 내 보상(Reward)을 정규화하여 Advantage를 계산한다. 그러나 그룹 내의 모든 응답이 전부 정답이거나 전부 오답인 경우, 보상의 분산이 0이 되어 Advantage가 사라지게 되며, 결과적으로 모델 업데이트를 위한 기울기(Gradient) 신호가 생성되지 않는 문제가 발생한다.

기존의 DAPO(Dynamic Sampling Policy Optimization)는 모든 그룹이 0이 아닌 보상 분산을 가질 때까지 새로운 프롬프트와 응답을 계속해서 샘플링하는 방식으로 이 문제를 해결하려 했다. 하지만 이 방식은 응답 생성 단계에서 막대한 계산 비용을 초래하며, 특히 모델의 크기가 커질수록 생성 단계가 전체 학습의 병목 현상이 되는 심각한 효율성 문제를 야기한다. 따라서 본 연구의 목표는 샘플링 효율성을 극대화하면서 Vanishing Advantage 문제를 해결하는 새로운 RLVR 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문은 샘플링 효율성을 높이기 위해 **Adaptive Rollout**과 **Response Reuse**라는 두 가지 핵심 아이디어를 결합한 **AR3PO** 알고리즘을 제안한다.

1. **Adaptive Rollout (적응형 롤아웃)**: 모든 프롬프트에 고정된 수의 응답을 생성하는 대신, 난이도에 따라 생성 예산을 동적으로 할당한다. 쉬운 프롬프트는 빠르게 정답을 찾으면 생성을 중단하여 자원을 아끼고, 어려운 프롬프트에는 더 많은 생성 기회를 부여하여 정답을 찾을 확률을 높인다.
2. **Response Reuse (응답 재사용)**: 현재 단계에서 정답을 하나도 생성하지 못한 어려운 프롬프트의 경우, 과거 학습 단계에서 생성했던 정답 응답을 Replay Buffer에서 가져와 재사용함으로써 학습 신호가 완전히 사라지는 것을 방지한다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적하며 차별성을 둔다.

- **GRPO**: Value Network 없이 그룹 내 정규화만으로 Advantage를 계산하여 안정성을 높였으나, 보상이 균일할 때 Advantage가 0이 되는 취약점이 있다.
- **DAPO**: 보상 분산이 발생할 때까지 반복 샘플링하여 Vanishing Advantage를 해결하지만, 응답 생성 비용이 GRPO 대비 최소 3배 이상 높다는 치명적인 단점이 있다.
- **기존의 Rollout Replay 및 Adaptive 전략**: 일부 연구들이 생성 예산을 조정하거나 과거 데이터를 재사용하는 시도를 했으나, 본 논문은 특히 RLVR 설정에서 발생하는 Importance Ratio의 변동성 문제를 해결하기 위한 구체적인 기술(토큰 확률 재계산 및 Gradient Stop)을 제안함으로써 차별화된다.

## 🛠️ Methodology

### 1. GRPO의 기초 및 문제 정의

GRPO는 프롬프트 $x$에 대해 $G$개의 응답 $\{o_i\}_{i=1}^G$를 생성하고, 다음과 같이 Advantage $A^i$를 계산한다.

$$A^i = \frac{R^i - \text{mean}(\{R^i\}_{i=1}^G)}{\text{std}(\{R^i\}_{i=1}^G)}$$

여기서 $R^i$는 정답 여부에 따른 이진 보상이다. 모든 $R^i$가 동일하면 $A^i=0$이 되어 학습이 불가능해진다.

### 2. Adaptive Rollout

AR3PO는 생성 과정을 $S$개의 단계(Stage)로 나눈다.

- 각 단계에서 프롬프트당 $k$개의 응답을 생성한다.
- 하나 이상의 정답 응답이 생성된 프롬프트는 즉시 생성 풀($U$)에서 제거한다.
- 정답을 찾지 못한 프롬프트만 다음 단계로 넘어가 추가 응답을 생성한다.
이 방식은 쉬운 문제는 적은 샘플로 해결하고, 어려운 문제에 더 많은 계산 자원을 집중시키는 효과를 준다.

### 3. Response Reuse

Adaptive Rollout 이후에도 정답이 없는 경우, 과거의 정답 응답을 저장해둔 Replay Buffer $B$에서 무작위로 하나($o_c$)를 선택해 그룹 내 오답 응답 하나를 대체한다. 이때 과거 정책 $\pi_{\theta_{old}}$와 현재 정책 $\pi_\theta$ 사이의 분포 차이(Distribution Shift)로 인해 Importance Ratio $r_{c,t}(\theta')$가 매우 커지거나 작아지는 문제가 발생한다. 이를 해결하기 위해 두 가지 옵션을 제안한다.

- **Option I (Off-policy learning)**: 재사용된 응답 $o_c$의 토큰 확률을 현재 정책 $\pi_\theta$를 사용하여 다시 계산한다. 이는 편향(Bias)을 유발할 수 있으나 분산(Variance)을 크게 줄여 학습을 안정화한다.
- **Option II (Negative sample training)**: 재사용된 응답 $o_c$에 대해서는 Gradient를 계산하지 않고(Stop Gradient), 오직 현재 정책이 생성한 on-policy 샘플들만 업데이트에 사용한다. 이때 $o_c$는 오직 Advantage 계산을 위한 기준점(정답 예시)으로만 활용되며, 나머지 오답 샘플들에 대해 음의 Advantage를 부여하여 잘못된 방향으로의 탐색을 억제하는 효과를 준다.

### 4. 전체 학습 절차

1. 프롬프트 배치 샘플링 및 Adaptive Rollout 수행.
2. 정답이 없는 프롬프트에 대해 Replay Buffer에서 정답 응답 $o_c$를 가져와 교체.
3. 정규화된 Advantage $A^i$ 계산.
4. 선택한 옵션(Option I 또는 II)에 따라 Gradient를 계산하여 정책 $\pi_\theta$ 업데이트.
5. 새로 생성된 정답 응답을 Replay Buffer $B$에 저장.

## 📊 Results

### 실험 설정

- **데이터셋**: DAPO-Math (영어 프롬프트 14K개).
- **평가 벤치마크**: Math500, Minerva Math, Olympiad Bench, AIME 24.
- **사용 모델**: Qwen2.5-7B, Llama-3.1-8B-Instruct, Qwen2.5-32B.
- **비교 대상**: GRPO, DAPO.

### 주요 결과

- **성능 및 효율성**: AR3PO는 7B 및 8B 모델에서 GRPO보다 우수한 성능을 보였으며, DAPO와 비슷하거나 더 높은 성능을 달성했다. 특히 **응답 생성 비용(Rollout Cost)을 DAPO 대비 최대 4.2배 절감**했다.
- **모델 크기 확장성**: Qwen2.5-32B 모델에서도 DAPO와 유사한 성능을 유지하면서 훨씬 낮은 생성 비용을 기록했다.
- **전략 분석**:
  - Adaptive Rollout은 쉬운 문제의 생성 횟수를 줄이고 어려운 문제(성공률 0.0-0.2 그룹)에 더 많은 응답(평균 6.95개)을 할당함으로써 효율성을 입증했다.
  - Response Reuse는 정답이 하나도 없는 프롬프트의 비율을 약 0.3에서 0.2 이하로 낮추어 학습 신호를 지속적으로 제공했다.
  - 재사용 전략 중 Option II(Gradient Stop)가 가장 좋은 성능을 보였으며, 단순한 Rollout Replay보다 우수했다.

## 🧠 Insights & Discussion

본 논문은 RLVR에서 샘플링 효율성이 단순히 학습 속도의 문제가 아니라, 계산 자원의 효율적 배분을 통해 모델의 성능까지 끌어올릴 수 있음을 보여준다.

특히 주목할 점은 **Bias-Variance Trade-off**에 대한 접근이다. 중요도 샘플링(Importance Sampling)에서 발생하는 높은 분산을 해결하기 위해, 엄밀한 수학적 무편향성보다는 실질적인 분산 감소(토큰 확률 재계산 또는 Gradient Stop)를 택한 것이 LLM의 대규모 학습에서 더 효과적임을 입증했다.

다만, 본 연구는 수학적 추론 작업에 집중되어 있으며, 정답 여부가 명확한 binary reward 설정 하에 수행되었다. 보상이 연속적이거나 검증기(Verifier)가 불완전한 일반적인 텍스트 생성 작업에서도 동일한 효율성 향상이 나타날지는 추가적인 연구가 필요하다. 또한, 32B 모델 실험에서 최대 응답 길이를 4096으로 제한했는데, 이를 더 늘렸을 때의 성능 향상 가능성이 남아 있다.

## 📌 TL;DR

AR3PO는 GRPO의 고질적인 문제인 **Vanishing Advantage**를 해결하기 위해 **Adaptive Rollout(난이도별 동적 샘플링)**과 **Response Reuse(과거 정답 재사용)**를 도입한 효율적인 RLVR 알고리즘이다. 이를 통해 DAPO 수준의 높은 추론 성능을 유지하면서도 생성 비용을 최대 4.2배 낮추었으며, 이는 대규모 모델의 RL post-training 비용을 획기적으로 줄일 수 있는 실용적인 방법론을 제시한 것이다.
