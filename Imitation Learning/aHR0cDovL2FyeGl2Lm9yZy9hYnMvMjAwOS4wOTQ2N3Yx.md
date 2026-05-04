# Addressing reward bias in Adversarial Imitation Learning with neutral reward functions

Rohit Jena, Siddharth Agrawal, Katia Sycara (2020)

## 🧩 Problem to Solve

본 논문은 Generative Adversarial Imitation Learning(GAIL) 알고리즘에서 발생하는 근본적인 문제인 **Reward Bias(보상 편향)** 문제를 해결하고자 한다. GAIL과 같은 Adversarial Imitation Learning(AIL) 방식은 판별자(Discriminator)가 생성하는 보상을 사용하여 정책을 학습시키는데, 이때 선택하는 보상 함수의 형태에 따라 에이전트의 행동에 편향이 발생한다.

논문은 환경을 크게 Survival-based(생존 기반)와 Task-based(과업 기반) 환경으로 구분한다. 특히, 과업 기반 환경 중에서도 **복수의 종료 상태(Multiple Terminal States)**가 존재하는 경우, 기존의 보상 함수들이 다음과 같은 심각한 문제를 야기함을 지적한다.

1.  **Survival Bias (생존 편향):** 양수 보상(Positive Reward)을 사용할 때, 에이전트가 과업을 완료하여 에피소드를 종료하기보다 환경 내에서 계속 루프를 돌며 보상을 수집하려는 경향이 나타난다.
2.  **Termination Bias (종료 편향):** 음수 보상(Negative Reward)을 사용할 때, 에이전트가 누적 음수 보상을 최소화하기 위해 과업 성공 여부와 상관없이 가능한 한 빨리 에피소드를 종료시키려는 경향이 나타난다.

따라서 본 연구의 목표는 이러한 생존 편향과 종료 편향을 모두 극복하여, 단일 또는 복수의 종료 상태가 존재하는 과업 기반 환경에서도 전문가의 정책을 정확하게 모방할 수 있는 새로운 보상 함수를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Neutral Reward Function(중립 보상 함수)**이라는 개념을 도입하여 보상 편향 문제를 해결한 것이다.

핵심 직관은 보상 함수의 범위를 $(-\infty, \infty)$로 확장하여 실수 값(Real-valued)을 갖게 함으로써, 전문가의 행동을 따르지 않을 때 강력한 음수 보상을 부여하는 것이다. 이를 통해 에이전트가 보상을 얻기 위해 불필요하게 생존하려는 루프 행동을 억제(Survival Bias 해결)하는 동시에, 전문가의 궤적을 따를 때 얻는 양수 보상이 종료로 인한 보상 손실보다 크도록 설계하여 조기 종료를 방지(Termination Bias 해결)한다.

## 📎 Related Works

논문은 기존의 Imitation Learning 연구들을 다음과 같이 분류하고 한계를 설명한다.

1.  **Inverse Reinforcement Learning (IRL):** 전문가의 시연으로부터 보상 함수를 먼저 복구하고 이를 최적화하는 방식이다. 하지만 내부 루프에서 RL 문제를 풀어야 하므로 학습 속도가 매우 느리다는 단점이 있다.
2.  **Adversarial Imitation Learning (AIL/GAIL):** 보상 함수를 명시적으로 복구하지 않고 판별자와 생성자(정책)를 동시에 학습시켜 전문가의 상태 분포를 맞추는 방식이다. Behavior Cloning보다 샘플 효율성이 높고 Distributional Shift 문제를 완화하지만, 판별자가 제공하는 보상 함수의 편향 문제에 취약하다.
3.  **Discriminator-Actor-Critic (DAC):** GAIL의 보상 편향을 해결하기 위해 흡수 상태(Absorbing state)에 대한 보상을 명시적으로 학습하는 방식을 제안했다. 하지만 저자들은 DAC가 다음과 같은 한계가 있음을 지적한다.
    - 환경을 수정하여 추가적인 종료 상태를 만들어야 하므로 API가 고정된 환경이나 실제 환경에서는 적용이 어렵다.
    - 복수의 종료 상태가 존재할 때, 목표 달성으로 인한 종료와 실패로 인한 종료를 구분하지 못한다.
    - 종료 상태의 보상까지 학습해야 하므로 GAIL에 비해 샘플 효율성이 떨어진다.

## 🛠️ Methodology

### 전체 파이프라인
본 논문은 기본적으로 GAIL의 프레임워크를 유지하며, 정책 $\pi$와 판별자 $D$의 목적 함수는 다음과 같다.

$$\max_{\pi} \min_{D} \left[ \mathbb{E}_{a \sim \pi(s)} [\log(D(s,a))] + \mathbb{E}_{a \sim \pi^E(s)} [\log(1-D(s,a))] + \lambda H(\pi) \right]$$

여기서 $D(s,a)$는 상태-행동 쌍이 전문가의 궤적에서 나왔을 확률이다. 본 논문의 핵심은 이 판별자의 출력값을 정책 학습을 위한 보상으로 변환하는 함수를 변경하는 것이다.

### 보상 함수의 종류 및 비교
논문은 세 가지 형태의 보상 함수를 분석한다.

1.  **Positive Reward:** $R(s,a) = -\log(1-D(s,a))$
    - 항상 양수이며, 에이전트가 환경에 오래 머물며 보상을 수집하려는 Survival Bias를 유발한다.
2.  **Negative Reward:** $R(s,a) = \log(D(s,a))$
    - 항상 음수이며, 에이전트가 보상 감소를 피하기 위해 빨리 종료하려는 Termination Bias를 유발한다.
3.  **Neutral Reward (Proposed):** $R(s,a) = \log(D(s,a)) - \log(1-D(s,a))$
    - 실수 전체 범위 $(-\infty, \infty)$를 가지며, 전문가의 행동과 일치하면 양수, 불일치하면 음수 보상을 준다.

### 이론적 근거 (Theoretical Sketch)
저자들은 Oracle Discriminator(전문가 행동에만 보상 $R$을 주고 나머지에 $0$ 또는 $-R$을 주는 판별자)를 가정하여 이론적 분석을 수행했다.

- **Survival Bias 분석:** 양수 보상 환경에서 전문가 궤적의 보상 $R^E$보다, 전문가 행동을 일부 따라하다가 루프를 도는 궤적의 보상 $R^{loop}$가 $\gamma \ge 0.618$일 때 더 커질 수 있음을 수식으로 증명했다.
- **Neutral Reward의 이점:** 중립 보상 함수에서는 루프를 도는 순간 음수 보상 $-R$이 발생하므로, 어떠한 경우에도 $R^{loop} < R^E$가 성립함을 보였다. 또한, 조기 종료를 하더라도 전문가 궤적의 마지막 단계에서 얻는 양수 보상을 포기해야 하므로 $R^{term} < R^E$가 되어 Termination Bias 역시 극복 가능하다.

## 📊 Results

### 실험 설정
- **데이터셋:** 전문가(PPO 정책)의 롤아웃 1,000개를 사용하였다.
- **환경:** Gym-Minigrid 패키지를 사용하였다.
    - 단일 종료 상태 환경: `Empty`, `DoorKey`
    - 복수 종료 상태 환경: `RedBlueDoors`, `GoToDoor`, `DistShift` (라바/용암 지대 존재)
- **지표:** 성공률(Success Rate), 에피소드 보상, 평균 에피소드 길이.
- **비교 대상:** Positive Reward GAIL, Negative Reward GAIL, DAC, Neutral Reward GAIL.

### 주요 결과
1.  **단일 종료 상태 환경:** Negative Reward와 Neutral Reward, DAC 모두 높은 성능을 보였다. 특히 Negative Reward는 종료 편향이 오히려 빠른 과업 완수를 유도하여 효율적으로 작동했다.
2.  **복수 종료 상태 환경:** 
    - **Positive Reward**는 Survival Bias로 인해 과업을 수행하지 못했다.
    - **Negative Reward**는 Termination Bias로 인해 목표 지점이 아닌 가장 가까운 종료 지점(예: `GoToDoor`에서 잘못된 색상의 문)으로 이동하여 에피소드를 끝내버리는 경향을 보였다.
    - **DAC**는 종료 상태 간의 구분을 하지 못해 성능이 저하되었다.
    - **Neutral Reward**는 모든 환경에서 가장 높은 성공률을 기록하며, 생존 편향과 종료 편향을 모두 효과적으로 극복함을 입증하였다.

| 환경 | Positive Reward | Negative Reward | DAC | Neutral Reward |
| :--- | :---: | :---: | :---: | :---: |
| Empty | 0.03 | 1.00 | 1.00 | 1.00 |
| DoorKey | 0.00 | 1.00 | 0.83 | 1.00 |
| GoToDoor | 0.00 | 0.93 | 0.26 | 0.97 |
| RedBlueDoors | 0.00 | 0.83 | 0.70 | 0.91 |
| DistShift | 0.00 | 0.85 | 0.76 | 0.97 |

## 🧠 Insights & Discussion

본 논문은 매우 단순한 보상 함수의 변경만으로도 GAIL의 고질적인 문제인 보상 편향을 해결할 수 있음을 보여주었다. 

**강점 및 시사점:**
- **이론적 분석:** 단순한 실험 결과 제시뿐만 아니라, $\gamma$ 값에 따른 루프 보상과 전문가 보상의 관계를 수식으로 증명하여 보상 편향의 원인을 명확히 규명하였다.
- **실용적 가치:** DAC와 달리 환경 수정이 필요 없으며, 구현이 매우 간단하여 즉시 적용 가능하다. 특히 로봇 공학에서 장애물 회피(종료 상태)와 목표 도달(성공 상태)이 공존하는 시나리오에서 중립 보상 함수가 매우 유용할 것으로 판단된다.

**한계 및 논의:**
- **표본의 제한:** 실험이 Gym-Minigrid라는 상대적으로 단순한 그리드 월드 환경에서 진행되었다. 더 복잡한 고차원 연속 제어 환경(Continuous Control)에서도 동일한 효과가 나타나는지에 대한 검증이 추가적으로 필요하다.
- **판별자 안정성:** $\log(D) - \log(1-D)$ 형태의 보상은 $D$가 $0$ 또는 $1$에 매우 가깝게 수렴할 때 수치적 불안정성(Numerical Instability)을 초래할 가능성이 있다. 논문에서는 이에 대한 명시적인 클리핑(Clipping)이나 정규화 전략이 상세히 서술되지 않았다.

## 📌 TL;DR

이 논문은 GAIL에서 발생하는 **생존 편향(Positive Reward 사용 시 루프 발생)**과 **종료 편향(Negative Reward 사용 시 조기 종료 발생)** 문제를 지적하고, 이를 해결하기 위해 실수 값 범위를 갖는 **Neutral Reward Function** ($\log(D) - \log(1-D)$)을 제안한다. 실험 결과, 특히 복수의 종료 조건이 존재하는 복잡한 과업 기반 환경에서 기존 GAIL 변형 및 DAC보다 월등한 성공률을 보였으며, 이는 향후 실제 로봇의 안전한 경로 계획 및 모방 학습 연구에 중요한 기초가 될 수 있다.