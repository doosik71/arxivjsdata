# Constrained Meta Agnostic Reinforcement Learning

Karam Daaboul, Florian Kuhm, Tim Joseph, Marius J. Zöllner (2024)

## 🧩 Problem to Solve

본 논문은 Meta-Reinforcement Learning(Meta-RL) 환경에서 **빠른 적응성(Rapid Adaptability)과 환경적 제약 준수(Adherence to Constraints) 사이의 균형**을 맞추는 문제를 해결하고자 한다.

전통적인 Meta-RL의 목표는 다양한 태스크에 빠르게 적응할 수 있는 메타 지식을 습득하는 것이지만, 실제 환경에 이를 적용할 때 에이전트가 자신이나 주변 환경에 위험을 초래하는 행동을 할 수 있다는 안전성 문제가 존재한다. 특히, 메타 학습 단계뿐만 아니라 새로운 태스크에 적응하는 Fine-tuning 단계에서도 안전 제약 조건을 준수하는 것이 필수적이다.

따라서 본 연구의 목표는 메타 학습 프레임워크 내에 제약 최적화(Constrained Optimization)를 통합하여, 새로운 태스크에 빠르게 적응하면서도 안전성을 보장하는 **C-MAML(Constrained Model Agnostic Meta Learning)** 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Inner loop(태스크별 적응)와 Outer loop(메타 파라미터 최적화) 모두에 안전 제약 조건을 통합**하는 것이다.

1. **Inner Loop의 제약 통합**: 각 태스크를 Constrained Markov Decision Process(CMDP)로 모델링하고, 태스크별 제약 조건을 직접 반영하여 안전한 적응을 유도한다.
2. **Outer Loop의 전역 안전성 확보**: 모든 태스크에 공통적으로 적용되는 전역 안전 제약 조건을 도입하고, 이를 위해 **Global Safety Critic**과 라그랑주 승수 $\eta$를 사용하여 메타 초기 파라미터 자체가 안전한 영역에 위치하도록 설계하였다.
3. **계산 효율성 및 범용성**: 2차 미분 계산의 복잡성을 피하기 위해 First-Order MAML(FoMAML) 방식을 채택하였으며, Inner loop의 최적화 알고리즘에 구애받지 않는 Model-Agnostic 특성을 갖추어 다양한 Safe RL 방법론과 결합할 수 있도록 하였다.

## 📎 Related Works

### Meta Learning

MAML, FoMAML, REPTILE 등 기존의 gradient 기반 메타 학습 알고리즘들은 최소한의 데이터로 새로운 태스크에 빠르게 적응하는 초기 파라미터를 찾는 데 집중하였다. 그러나 이러한 방법론들은 학습 및 적응 과정에서의 **안전성(Safety) 고려가 결여**되어 있다는 한계가 있다.

### Safe Reinforcement Learning

CPO(Constrained Policy Optimization), TRPO-Lagrangian 등 Safe RL 연구들은 특정 태스크를 수행할 때 제약 조건을 준수하며 보상을 최대화하는 방법에 집중하였다. 하지만 이러한 접근 방식은 **동적인 환경에서 새로운 태스크에 빠르게 적응해야 하는 능력(Adaptability)**에 대해서는 충분히 다루지 않았다.

### 차별점

C-MAML은 기존 Meta-RL의 '빠른 적응력'과 Safe RL의 '제약 준수' 능력을 결합하여, 메타 학습 단계부터 안전한 초기 파라미터를 생성함으로써 Fine-tuning 과정에서의 위험을 최소화한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

C-MAML은 두 단계의 최적화 루프로 구성된다. Inner loop에서는 주어진 메타 정책 $\pi$를 바탕으로 태스크 $p$에 특화된 안전한 정책 $\pi^p$를 도출하고, Outer loop에서는 이러한 태스크별 정책들이 전반적으로 높은 보상을 얻으면서도 안전 제약을 준수하도록 메타 파라미터 $\pi$를 업데이트한다.

### 1. Inner Loop: 태스크별 제약 최적화

각 태스크는 CMDP로 정의되며, 보상 함수 $r^p$와 비용 함수 $c^p$를 가진다. 목표는 비용의 기댓값이 임계값 $d^p$ 이하인 조건에서 보상을 최대화하는 것이다. 본 논문에서는 이를 위해 **TRPOLag(Trust Region Policy Optimization with Lagrangian)**를 사용하며, 목적 함수는 다음과 같이 정의된다.

$$L_{inner} = L^p(\pi^p, \lambda^p) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \frac{\pi^p(a_t|s_t)}{\pi(a_t|s_t)} A_{\pi^p}(s_t, a_t) \right] - \lambda^p \left( \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \frac{\pi^p(a_t|s_t)}{\pi(a_t|s_t)} A_{\pi^C}^p(s_t, a_t) \right] + J^C(\pi) - d \right)$$

여기서 $\lambda^p$는 태스크별 라그랑주 승수이며, $\text{D}_{KL}(\pi^p \| \pi) \leq \epsilon$ 제약을 통해 정책 업데이트의 안정성을 보장한다.

### 2. Outer Loop: 메타 파라미터 최적화

메타 정책 $\pi$는 모든 태스크에 대해 평균적으로 안전하고 효율적인 초기값이어야 한다. 이를 위해 다음과 같은 메타 목적 함수를 최적화한다.

$$\max_{\pi} \mathbb{E}_{p \sim T} \left[ \mathbb{E}_{\tau \sim \pi^p} (R^p(\tau)) \right] \quad \text{s.t.} \quad \mathbb{E}_{p \sim T} \left[ \mathbb{E}_{\tau \sim \pi^p} (C^p(\tau)) - d^p \right] \leq 0$$

### 3. Safety Critic 및 전역 제약 강화

계산 효율성을 위해 FoMAML을 사용하되, 메타 정책 자체가 안전하도록 **Global Safety Critic** $V_C^\pi(s)$와 라그랑주 승수 $\eta$를 도입한 최종 Outer loop 목적 함수는 다음과 같다.

$$L_{outer}(\pi, \lambda, \eta) = \mathbb{E}_{p \sim T} \left[ \mathbb{E}_{\tau \sim \pi^p} (R^p(\tau)) - \lambda (\mathbb{E}_{\tau \sim \pi^p} (C(\tau)) - d) - \eta (V_C^\pi(s) - d) \right]$$

- **$\lambda$ 항**: 적응 후의 정책 $\pi^p$가 안전한지 평가한다.
- **$\eta$ 항**: 메타 정책 $\pi$ 그 자체(초기값)가 전역적으로 안전한지 평가하여, 적응 시작 전부터 안전한 파라미터 공간에 위치하게 한다.

### 4. 학습 절차 및 그라디언트

메타 정책의 업데이트를 위한 그라디언트는 다음과 같이 근사된다.

$$\nabla_{\pi} L_{outer} \approx I \cdot \mathbb{E}_{p \sim T} (\nabla_{\pi} L_{outer}(\pi^p)) - \frac{\partial}{\partial \pi} \eta \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \log(\pi(a_t|s_t)) \cdot V_C^\pi(s) \right]$$

여기서 Hessian 행렬을 단위 행렬 $I$로 근사하여 계산 복잡도를 낮추었으며, 두 번째 항은 Safety Critic을 통해 비용이 높은 행동의 확률을 낮추고 안전한 행동의 확률을 높이는 역할을 한다.

## 📊 Results

### 실험 설정

- **환경**: 고차원 연속 상태 및 행동 공간을 가진 모바일 로봇 시뮬레이션 (Point Robot).
- **태스크**: 장애물(Hazards)과 꽃병(Vases)을 피해 목표 지점에 도달하는 작업.
- **비교 대상**: Random Initialization, TRPOLag Pretrained Policy, unconstrained MAML.
- **측정 지표**: Mean Episode Return(평균 에피소드 보상), Mean Episode Costs(평균 에피소드 비용).

### 주요 결과

1. **안전성 및 적응 속도**: C-MAML은 Random 초기화보다 적응 속도가 훨씬 빠르며, 일반 MAML이나 Pretrained 정책보다 비용 제약($d=5$)을 훨씬 더 일관되게 준수하였다.
2. **$\eta$ 및 Safety Critic의 효과**: $\eta=0$인 경우(Safety Critic 미사용)보다 $\eta$를 적응적으로 학습시켰을 때 메타 학습 및 Fine-tuning 단계에서 비용 변동성이 낮고 제약 준수율이 높았다.
3. **Model Agnosticity 검증**: Inner loop 알고리즘을 TRPOLag에서 CPO(Constrained Policy Optimization)로 교체하여 실험한 결과, 동일하게 높은 적응성과 안전성을 보였다. 이는 제안한 프레임워크가 특정 Safe RL 알고리즘에 종속되지 않음을 입증한다.
4. **난이도 영향**: 장애물 수를 조절한 Environment 1(쉬움)과 Environment 2(어려움) 모두에서 C-MAML이 다른 베이스라인 대비 우수한 성능을 보였으며, 특히 어려운 환경에서도 안전성을 유지하며 빠르게 적응하는 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 Meta-RL의 고질적인 문제인 '적응 과정에서의 안전성 부족'을 체계적으로 해결하였다. 특히 **전역 안전 제약($\eta$ 항)**을 통해 메타-초기값이 안전한 영역에 머물게 함으로써, Fine-tuning 초기 단계에서 발생할 수 있는 치명적인 사고를 방지할 수 있다는 점이 매우 강력한 강점이다.

### 한계 및 논의사항

- **제약 조건의 가정**: 본 논문은 구현의 단순화를 위해 모든 태스크에 대해 동일한 전역 제약 조건 $C$와 임계값 $d$가 존재한다고 가정하였다. 하지만 실제 환경에서는 태스크마다 안전 기준이 다를 수 있으므로, 이를 일반화하는 연구가 필요하다.
- **계산 효율성**: FoMAML을 통해 2차 미분을 피했지만, 여전히 Inner loop의 반복적인 최적화와 Safety Critic의 학습으로 인해 전체 학습 시간이 증가할 가능성이 있다.
- **비판적 해석**: 실험이 시뮬레이션 환경에서 수행되었으므로, 실제 물리 로봇의 하드웨어 제약(예: 모터 토크 제한, 센서 노이즈) 하에서도 동일한 수준의 안전성이 보장될지는 추가적인 검증이 필요하다.

## 📌 TL;DR

C-MAML은 Meta-RL 프레임워크에 Safe RL의 제약 최적화를 통합하여, **새로운 태스크에 빠르게 적응하면서도 안전 제약 조건을 엄격히 준수**하는 메타 정책을 학습하는 방법론이다. Inner loop의 태스크별 제약과 Outer loop의 전역 Safety Critic 및 $\eta$ 조절기를 통해 **'안전한 메타 초기값'**을 찾아내며, 이는 다양한 Safe RL 알고리즘(TRPOLag, CPO 등)과 결합 가능한 범용적 구조를 가진다. 이 연구는 안전성이 필수적인 실제 로봇 제어 시스템의 Meta-RL 적용에 있어 중요한 기반을 제공할 것으로 기대된다.
