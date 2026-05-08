# TRAIL: NEAR-OPTIMAL IMITATION LEARNING WITH SUBOPTIMAL DATA

Mengjiao Yang, Sergey Levine, Ofir Nachum (2021)

## 🧩 Problem to Solve

Imitation Learning(IL)의 핵심 목표는 전문가의 시연(expert demonstrations) 데이터를 활용하여 효과적인 정책(policy)을 학습하는 것이다. 그러나 고품질의 전문가 데이터는 수집 비용이 매우 높으며, 대량으로 확보하기 어렵다는 현실적인 제약이 있다. 반면, 성능이 낮은 suboptimal 데이터나 작업과 무관한 task-agnostic trajectory는 상대적으로 대량으로 수집하기 쉽다.

이러한 suboptimal 데이터는 직접적으로 모방 학습에 사용하기에는 부적합하지만, 환경의 역학적 구조(dynamical structure), 즉 환경에서 '무엇을 할 수 있는지'에 대한 정보를 포함하고 있다. 본 논문은 이러한 suboptimal offline 데이터셋을 활용하여, 이후 진행될 downstream imitation learning의 샘플 효율성을 이론적으로 보장하며 향상시킬 수 있는지를 탐구한다. 즉, 전문가 데이터가 부족한 상황에서 보조적인 non-expert 데이터를 통해 학습 난이도를 낮추는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 suboptimal 데이터셋으로부터 **잠재 행동 공간(latent action space)**을 추출하고, 이 공간 상에서 모방 학습을 수행하는 것이다. 단순히 시간적 추상화(temporal abstraction)를 통해 스킬을 학습하는 기존의 계층적 RL 방식과 달리, 본 연구는 단일 단계의 행동 표현(single-step action representation)을 재매개변수화(reparameterize)함으로써 학습 효율을 높이는 데 집중한다.

주요 기여 사항은 다음과 같다.

1. 시간적 추상화 없이도 행동 표현 학습이 downstream IL에 이점을 준다는 것을 증명하는 이론적 상한선(upper bound)을 도출하였다.
2. 위 이론을 바탕으로, Energy-Based Model(EBM)을 통해 전이 모델(transition model)을 대조적으로 학습하고 이를 통해 행동 공간을 재매개변수화하는 **TRAIL(Transition-Reparametrized Actions for Imitation Learning)** 알고리즘을 제안하였다.
3. 다양한 내비게이션 및 로코모션 작업에서 TRAIL이 기존의 Behavioral Cloning(BC) 및 시간적 스킬 추출 방식보다 월등히 높은 성능을 보임을 실험적으로 입증하였다.

## 📎 Related Works

기존의 행동 추상화 연구는 주로 계층적 강화학습(Hierarchical RL) 관점에서 시간적으로 확장된 스킬(temporally extended skills)을 학습하는 데 집중하였다. OPAL, SkiLD, SPiRL과 같은 방법들은 offline 데이터에서 잠재 변수를 통해 스킬을 추출하고, 이를 통해 제어 빈도를 낮춤으로써 샘플 복잡도를 줄이려 하였다.

그러나 이러한 접근 방식은 다음과 같은 한계가 있다.

- **시간적 추상화 의존성:** 성능 향상의 주된 요인이 행동 공간의 효율적 표현이 아니라, 단순히 시간적 빈도를 낮춘 것(temporal abstraction)에 기인한다.
- **데이터 품질 민감도:** 데이터가 단일 모드(unimodal)이거나 완전히 무작위(random)인 경우, 유의미한 스킬을 추출하기 어려워 성능이 저하된다.

TRAIL은 시간적 추상화에 의존하지 않고 행동 공간 자체를 재매개변수화하므로, 매우 낮은 품질의 데이터나 단일 모드 데이터에서도 환경의 역학을 학습하여 downstream IL을 가속화할 수 있다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

TRAIL은 크게 **사전 학습(Pretraining)** 단계와 **Downstream 모방 학습(Downstream Imitation)** 단계로 구성된다.

1. **Pretraining:** suboptimal 데이터셋 $D_{off}$를 사용하여 factored transition model $T = T_Z \circ \phi$와 행동 디코더(action decoder) $\pi_\alpha$를 학습한다.
2. **Downstream Imitation:** 전문가 데이터셋 $D_{\pi^*}$의 행동을 학습된 $\phi$를 통해 잠재 공간으로 투영하고, 잠재 정책 $\pi_Z$를 Behavioral Cloning으로 학습한다.
3. **Inference:** $\pi_Z$에서 잠재 행동 $z$를 샘플링하고, $\pi_\alpha$를 통해 실제 행동 $a$로 디코딩하여 환경에 적용한다.

### 상세 방법론 및 이론적 배경

#### 1. 이론적 성능 상한 (Theorem 1)

본 논문은 학습된 정책 $\pi_\alpha \circ \pi_Z$와 전문가 정책 $\pi^*$ 사이의 거리(Total Variation divergence)를 다음과 같은 세 가지 오차 항으로 분해하여 정의한다.

$$\text{Diff}(\pi_\alpha \circ \pi_Z, \pi^*) \le C_1 \sqrt{J_T(T_Z, \phi)} + C_2 \sqrt{J_{DE}(\pi_\alpha, \phi)} + C_3 \sqrt{J_{BC, \phi}(\pi_Z)}$$

- $J_T$: 전이 표현 오차(Transition representation error). suboptimal 데이터에서 전이 모델이 얼마나 정확하게 학습되었는가를 나타낸다.
- $J_{DE}$: 행동 디코딩 오차(Action decoding error). 잠재 행동 $z$로부터 원래 행동 $a$를 얼마나 잘 복원하는가를 나타낸다.
- $J_{BC, \phi}$: 잠재 행동 공간에서의 BC 오차. 전문가 데이터를 잠재 공간에서 얼마나 잘 모방하는가를 나타낸다.

여기서 $J_T$와 $J_{DE}$는 대량의 suboptimal 데이터 $D_{off}$로 학습하며, 오직 $J_{BC, \phi}$만이 소량의 전문가 데이터 $D_{\pi^*}$를 필요로 한다. 잠재 공간의 차원 $|Z|$가 원래 행동 공간 $|A|$보다 작다면, $J_{BC, \phi}$를 줄이는 데 필요한 샘플 수가 획기적으로 감소한다.

#### 2. TRAIL EBM (Theorem 1 기반)

복잡하고 다중 모드(multi-modal)인 전이 역학을 캡처하기 위해 Energy-Based Model(EBM)을 사용하여 전이 모델을 파라미터화한다.

$$T_Z(s'|s, \phi(s, a)) \propto \rho(s') \exp(-\|\phi(s, a) - \psi(s')\|^2)$$

여기서 $\rho(s')$는 $D_{off}$에서의 상태 분포이다. 학습은 다음과 같은 대조적 손실 함수(contrastive loss)를 최소화하는 방식으로 진행된다.
$$\mathcal{L} = \mathbb{E}_{d_{off}} [\|\phi(s, a) - \psi(s')\|^2] + \log \mathbb{E}_{\tilde{s}' \sim \rho} [\exp(-\frac{1}{2} \|\phi(s, a) - \psi(\tilde{s}')\|^2)]$$

#### 3. TRAIL Linear (Theorem 3 기반)

연속적인 행동 공간에서 결정론적(deterministic) 잠재 정책을 사용하기 위해, Random Fourier Features(RFF)를 사용하여 EBM을 근사함으로써 선형 전이 모델(linear transition model)을 구현한다.

- $\bar{\phi}(s, a) = \cos(W f(s, a) + b)$ 형태로 행동 표현을 추출하며, 여기서 $W, b$는 학습되지 않는 고정된 파라미터이다.
- downstream 단계에서는 $\pi_\theta(s)$와 $\bar{\phi}(s, a)$ 사이의 평균 제곱 오차(MSE)를 최소화하는 단순한 회귀 문제로 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업:**
  - AntMaze (D4RL): 내비게이션 작업.
  - MuJoCo Ant: 로코모션 작업.
  - DeepMind Control Suite (DMC): 1-DoF(Cartpole)부터 21-DoF(Humanoid)까지 다양한 난이도의 작업.
- **비교 대상:** Vanilla BC, OPAL, SkiLD, SPiRL (시간적 스킬 추출 방식), CRR (Offline RL).
- **평가 지표:** 작업 성공률(Success Rate) 및 평균 보상(Average Reward).

### 주요 결과

1. **내비게이션 (AntMaze):** TRAIL은 시간적 추상화를 사용하지 않음에도 불구하고($t=1$), $t=10$을 사용하는 기존 스킬 추출 방법들과 대등하거나 더 높은 성능을 보였다. 특히 AntMaze-Large와 같은 어려운 작업에서 Vanilla BC 대비 성능이 크게 향상되었다.
2. **로코모션 (MuJoCo Ant):** 데이터가 unimodal하거나 random-policy로 생성된 매우 낮은 품질의 데이터셋에서도 TRAIL은 일관되게 BC보다 높은 성능을 보였다. 반면, 기존 스킬 추출 방식들은 데이터 품질이 낮을 때 성능이 급격히 저하되었다.
3. **DMC Suite:** 다양한 DoF를 가진 작업들에서 TRAIL은 기존 방법들을 압도하였으며, 일부 작업에서는 보상 라벨을 사용하는 Offline RL 방법인 CRR과 비슷하거나 더 나은 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 행동 공간의 **재매개변수화(reparameterization)**가 imitation learning의 샘플 효율성을 높이는 강력한 도구가 될 수 있음을 입증하였다. 특히, 기존 연구들이 성능 향상의 원인을 '시간적 추상화'에서 찾았던 것과 달리, TRAIL은 단일 단계의 표현 학습만으로도 충분한 이득을 얻을 수 있음을 보여주었다. 이는 suboptimal 데이터가 단순히 '스킬'의 저장소가 아니라, 환경의 '역학'을 학습하기 위한 풍부한 소스임을 시사한다.

### 한계 및 논의사항

- **전이 모델의 정확도:** 이론적 상한선에서 볼 수 있듯이, 사전 학습 단계의 전이 모델 오차($J_T$)와 디코딩 오차($J_{DE}$)가 크다면 downstream 성능이 제한될 수 있다.
- **계산 비용:** EBM 학습 시 대조적 손실 함수를 사용하므로 샘플링 과정에서 계산 비용이 발생할 수 있다.
- **확장성:** 본 연구는 주로 BC에 집중하였으나, 제안된 잠재 행동 공간을 강화학습(RL)의 탐색 효율성을 높이는 데 활용할 가능성이 열려 있다.

## 📌 TL;DR

본 논문은 대량의 suboptimal offline 데이터를 활용해 **잠재 행동 공간(latent action space)**을 학습하고, 이를 통해 소량의 전문가 데이터만으로도 고성능 정책을 학습할 수 있는 **TRAIL** 알고리즘을 제안한다. 이론적으로 행동 표현 학습이 IL의 샘플 효율성을 높임을 증명하였으며, 실험을 통해 시간적 추상화 없이도 무작위 데이터셋만으로 BC 성능을 최대 4배까지 향상시킬 수 있음을 보였다. 이는 전문가 데이터 확보가 어려운 실제 환경에서 suboptimal 데이터를 효율적으로 활용할 수 있는 새로운 경로를 제시한다.
