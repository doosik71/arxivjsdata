# A reinforcement learning agent for maintenance of deteriorating systems with increasingly imperfect repairs

Alberto Pliego Marugán, Jesús María Pinar-Pérez, and Fausto Pedro García Márquez (2024)

## 🧩 Problem to Solve

본 논문은 시간이 지남에 따라 성능이 저하되는 시스템(deteriorating systems)의 효율적인 유지보수 전략을 수립하는 문제를 다룬다. 특히, 실제 산업 현장에서는 수리를 반복할수록 수리의 효율이 떨어지는 '점진적으로 불완전한 수리(increasingly imperfect repairs)' 현상이 발생하는데, 기존의 많은 모델은 이를 충분히 반영하지 못했다.

유지보수 비용은 전체 생산 비용의 15%에서 70%에 달할 정도로 경제적 비중이 크며, Industry 4.0 시대에 접어들면서 복잡한 시스템의 가동률을 높이고 비용을 최적화하는 새로운 유지보수 패러다임이 필요하다. 따라서 본 연구의 목표는 시스템의 연속적인 성능 저하 상태를 고려하고, 수리 횟수에 따라 효과가 감소하는 현실적인 제약 조건을 반영하여 장기적인 유지보수 비용을 최소화하는 Reinforcement Learning (RL) 기반의 에이전트를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같이 요약할 수 있다.

첫째, 수리를 수행할수록 그 효과가 감소하는 '점진적으로 불완전한 수리' 모델을 제안하였다. 이는 수리 후의 상태가 이전 수리 시점의 상태보다 더 나빠질 수밖에 없다는 메모리 효과(memory effect)를 truncated normal distribution을 통해 모델링함으로써 현실성을 높였다.

둘째, Double Deep Q-Network (DDQN) 아키텍처를 사용하여 유지보수 정책을 생성하는 에이전트를 구현하였다. 이 에이전트는 기존의 Condition-Based Maintenance (CBM)에서 필수적이었던 '예방 정비 임계값(preventive threshold)'을 미리 정의할 필요 없이, 학습을 통해 최적의 정비 시점을 스스로 결정한다.

셋째, 상태 공간을 이산화(discretization)하지 않고 연속적인 성능 저하 상태 공간(continuous degradation state space)에서 직접 작동하는 에이전트를 설계하여, 모델의 정밀도를 높이고 유연한 대응이 가능하게 하였다.

## 📎 Related Works

기존 연구들은 주로 확률적 성능 저하 프로세스(Stochastic Degradation Processes, SDP)를 모델링하기 위해 Gamma process, Inverse Gaussian process, Wiener process 등을 사용해 왔다. 특히 Gamma process는 단조 증가하는 성능 저하를 모델링하는 데 널리 사용된다.

유지보수 최적화를 위해 Value iteration, stochastic filtering, multi-objective optimization 등 다양한 기법이 제안되었으며, 최근에는 RL을 활용한 연구가 급증하고 있다. 기존의 RL 기반 유지보수 연구들은 주로 상태 공간을 몇 가지 단계(예: 4단계)로 이산화하여 Markov Decision Process (MDP)로 해결하거나, 특정 임계값을 기준으로 한 CBM 정책을 최적화하는 데 집중하였다.

본 논문은 기존 연구와 달리 다음과 같은 차별점을 가진다. 첫째, 수리 횟수가 증가함에 따라 수리 효율이 떨어진다는 점을 명시적으로 모델링하였다. 둘째, 성능 저하 상태를 이산화하지 않고 연속적인 값으로 처리함으로써 더 정밀한 의사결정을 가능하게 하였다.

## 🛠️ Methodology

### 1. 성능 저하 모델 및 유지보수 모델
시스템의 성능 저하는 homogeneous gamma process로 모델링한다. 시간 $t$에서의 성능 저하 상태를 $X_t$라 할 때, 시간 $\Delta t$ 동안의 저하 증분 $\Delta X$는 다음과 같은 Gamma 분포를 따른다.

$$\Delta X \sim \Gamma(v(t, \Delta t), \beta)$$

여기서 $v$는 shape parameter, $\beta$는 scale rate이다.

유지보수 작업은 두 가지로 나뉜다:
- **교체 (Replacements, R):** 시스템을 완전히 새 상태인 "As Good as New" (AGAN) 상태로 되돌린다 ($X^R_{T_n} = 0$).
- **수리 (Repairs, P):** 불완전한 수리로, 현재 상태에서 일정 양 $Z_n$만큼 저하도를 감소시킨다 ($X^P_{T_n} = X^-_{T_n} - Z_n$).

이때 수리 이득 $Z_n$은 truncated normal distribution을 따르며, 시스템은 이전 수리 후의 상태 $X_M$보다 더 좋은 상태로 돌아갈 수 없다는 제약이 있다. $Z_n$의 밀도 함수 $g(x)$는 다음과 같다.

$$g_{\mu, \sigma, X^-_{T_n}}(x) = \frac{1}{\sigma} \phi \left( \frac{x-\mu}{\sigma} \right) \frac{\Phi \left( \frac{X^-_{T_n}-\mu}{\sigma} \right) - \Phi \left( \frac{X_M-\mu}{\sigma} \right)}{\dots} I_{[X_M, X^-_{T_n}]}(x)$$

여기서 $\mu$와 $\sigma$는 현재 상태 $X^-_{T_n}$과 이전 수리 상태 $X_M$의 평균 및 표준편차에 기반하여 결정된다.

### 2. RL 에이전트 설계 (DDQN)
에이전트는 DQN의 Q-value 과대평가(overestimation) 문제를 해결하기 위해 DDQN을 사용한다. DDQN은 행동 선택과 행동 평가를 분리하여 더 안정적인 학습을 가능하게 한다.

- **상태 공간 (State Space):** $S^T_n = \{X^T_n, X_M\}$ (현재 성능 저하도, 이전 유지보수 후의 저하도)
- **행동 공간 (Action Space):** $A = \{a_0, a_1, a_2\}$
    - $a_0$: 아무 작업도 하지 않음
    - $a_1$: 예방 수리 (Preventive Repair)
    - $a_2$: 교체 (Replacement - 예방 또는 사후 교체)
- **보상 함수 (Reward):** 유지보수 비용을 최소화하는 것이 목표이므로, 비용 발생 시 음의 보상을 부여한다.
    - $a_0$ (정상): $0$
    - $a_1$ (수리): $-C_p$
    - $a_2$ (교체, 예방): $-C_R$
    - $a_2$ (교체, 고장 후): $-C_R - C_{down}$ (고장 임계값 $L$ 초과 시 가동 중단 비용 추가)

DDQN의 Bellman 방정식은 다음과 같이 정의된다.

$$Q(s, a; \theta) = r + \gamma Q(s', \text{arg max}_{a'} Q(s', a'; \theta); \theta')$$

여기서 $\theta$는 메인 네트워크의 파라미터, $\theta'$는 타겟 네트워크의 파라미터이다.

## 📊 Results

### 1. 실험 설정
7가지 서로 다른 시나리오(Case 1~7)를 설정하여 에이전트의 성능을 분석하였다. 각 시나리오는 수리 비용, 고장 임계값 $L$, 가동 중단 비용 $C_{down}$, 성능 저하 속도, 점검 주기 $\Delta t$ 등을 다르게 설정하여 에이전트의 유연성을 테스트하였다. 성능 지표로는 장기 비용 비율(long-run cost rate) $\mathbb{E}C_\infty$를 사용하였으며, 이는 다음과 같이 계산된다.

$$\mathbb{E}C_\infty = \frac{C_P \mathbb{E}[N_P(S_1)] + C_R \mathbb{E}[N_{PR}(S_1)] + (C_R + C_{down}) \mathbb{E}[N_{CR}(S_1)]}{\mathbb{E}[S_1]}$$

### 2. 주요 결과
- **시나리오 적응력:** 수리 비용이 저렴한 Case 1에서는 수리 횟수를 늘리고, 비용이 비싼 Case 3에서는 수리 횟수를 50%까지 줄이는 등 에이전트가 비용 구조에 따라 유연하게 정책을 변경함을 확인하였다.
- **임계값 영향:** 고장 임계값 $L$이 높아지면(Case 4) 유지보수 빈도가 줄어들어 전체 비용이 감소하였다.
- **점검 주기 영향:** 점검 주기 $\Delta t$가 길어지면(Case 7) 고장 위험이 커지므로, 에이전트는 더 낮은 저하도 상태에서 예방 정비를 수행하는 보수적인 정책을 학습하였다.
- **기존 정책과의 비교:** Case 2* (Baseline) 기준, DDQN 에이전트는 Fail Replacement (FR), Threshold-based (TBM), Age-based, ATBM 정책보다 장기 비용을 각각 약 41%, 28%, 31%, 17% 절감하였다.

## 🧠 Insights & Discussion

본 연구의 결과는 RL 에이전트가 복잡하고 연속적인 성능 저하 환경에서도 최적의 유지보수 시점을 매우 효과적으로 찾아낼 수 있음을 보여준다. 특히, 사람이 수동으로 설정해야 하는 '예방 정비 임계값' 없이도 비용 구조와 성능 저하 속도를 스스로 학습하여 최적의 균형점을 찾는다는 점이 강력한 강점이다.

또한, '점진적으로 불완전한 수리' 모델을 도입함으로써, 무한히 수리만 반복해서는 시스템을 유지할 수 없으며 결국 교체가 필요하다는 현실적인 유지보수 사이클을 성공적으로 모사하였다.

다만, 본 모델은 유지보수 비용의 최소화라는 경제적 관점에만 집중하였다. 따라서 항공기나 원자력 발전소와 같이 실패가 치명적인 결과(catastrophic failure)를 초래하는 safety-critical 시스템에는 현재의 보상 함수 구조를 그대로 적용하기 어려우며, 신뢰도(reliability)를 극대화하는 제약 조건이나 보상 설계가 추가로 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 수리를 반복할수록 효율이 떨어지는 '점진적으로 불완전한 수리' 환경에서, DDQN 에이전트를 이용해 유지보수 비용을 최소화하는 최적 정책을 학습시킨 연구이다. 이 에이전트는 상태 공간의 이산화나 사전 임계값 설정 없이도 연속적인 저하 상태를 처리하며, 기존의 전통적인 유지보수 전략(TBM, Age-based 등)보다 장기 비용을 최대 41%까지 절감하는 성능을 보였다. 이는 향후 Industry 4.0의 지능형 유지보수 시스템 구축에 중요한 기초 연구가 될 가능성이 높다.