# On Combining Expert Demonstrations in Imitation Learning via Optimal Transport

Ilana Sebag, Samuel Cohen, Marc Peter Deisenroth (2021)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)에서 여러 전문가의 시연(Expert Demonstrations) 데이터를 어떻게 최적으로 결합하여 에이전트에게 학습시킬 것인가라는 문제를 다룬다. 

모방 학습의 핵심은 에이전트의 궤적(Trajectory)과 전문가의 궤적 사이의 거리를 정의하고 이를 최소화하는 정책을 찾는 것이다. 최근 최적 운송(Optimal Transport, OT) 방법론이 궤적 간의 의미 있는 거리를 측정하는 도구로 널리 사용되고 있으나, 전문가가 여러 명일 때 이들의 데이터를 결합하는 표준적인 방법은 단순히 상태-행동 궤적을 연결(Concatenate)하는 것이다. 

하지만 이러한 단순 연결 방식은 전문가들의 궤적이 서로 다른 모드(Multi-modal)를 가지거나 매우 다양할 경우, 이러한 다양성을 유의미한 정보가 아닌 노이즈(Noise)로 처리하게 되어 학습 효율이 떨어지는 문제가 발생한다. 따라서 본 논문의 목표는 여러 전문가의 다양한 시연 데이터를 기하학적으로 더 타당하게 결합하여 학습할 수 있는 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단순한 데이터 연결 대신 **Multi-Marginal Optimal Transport(MMOT)** 도구를 사용하여 여러 전문가의 궤적을 동시에 고려하는 것이다.

구체적으로는 **Sliced Multi-Marginal Monge-Wasserstein** 거리를 도입하여, 에이전트의 궤적과 여러 전문가 궤적 사이의 불일치를 측정한다. 이는 전문가들의 시연 데이터에 대한 **기하학적 평균(Geometric Average)**인 Wasserstein Barycenter를 구하는 것과 이론적으로 동일하며, 이를 통해 단순히 데이터를 샘플링하는 것보다 훨씬 매끄럽고(Smooth) 타당한 전문가의 기준 궤적을 생성하여 에이전트에게 제공할 수 있다.

## 📎 Related Works

모방 학습은 크게 행동 복제(Behavioral Cloning, BC)와 역강화학습(Inverse Reinforcement Learning, IRL)으로 나뉜다. BC는 상태에서 행동으로의 매핑을 직접 학습하는 지도 학습 방식이며, IRL은 전문가의 궤적과 유사한 궤적을 생성하는 보상 함수를 추론하는 방식이다.

최근 연구들은 궤적 간의 거리 측정 도구로 Optimal Transport(OT)를 활용하여 IRL의 성능을 높이려 했다. 특히 Primal Wasserstein Imitation Learning(PWIL)과 같은 알고리즘이 제안되었으며, 고차원 데이터에서 OT 계산의 복잡도를 줄이기 위해 데이터를 1차원으로 투영하여 계산하는 Sliced Wasserstein distance가 사용되었다. 

그러나 기존의 다수 전문가 데이터 처리 방식은 단순히 궤적들을 연결하고 그중 일부를 서브샘플링하여 단일 전문가 시연처럼 사용하는 방식에 의존해 왔으며, 이는 전문가 간의 다양성이 클 때 성능 저하를 야기한다는 한계가 있다.

## 🛠️ Methodology

본 연구는 PWIL 알고리즘을 기반으로 하되, 전문가 데이터를 처리하는 방식에 따라 두 가지 모델인 **SCOTIL**과 **SMMOTIL**을 제안하고 비교한다.

### 1. Sliced Optimal Transport (SOT) 기초
고차원 확률 분포 $\mu, \nu$ 사이의 거리를 효율적으로 계산하기 위해, 이를 다양한 방향 $\theta$로 투영하여 1차원 Wasserstein 거리의 평균을 구한다.
$$SW_2^2(\mu, \nu) = \int_{S^{d-1}} W_2^2(P_{\theta \#} \mu, P_{\theta \#} \nu) d\theta$$
여기서 $P_\theta(x) = \langle \theta, x \rangle$는 선형 투영 연산자이며, 1차원에서의 $W_2^2$는 정렬된 샘플 간의 유클리드 거리 제곱의 합으로 간단히 계산된다.

### 2. SCOTIL (Sliced Concatenated Optimal Transport IL)
기존의 표준 방식을 따르는 베이스라인 모델이다.
- **절차**: 여러 전문가의 궤적 $\mu_{e1}, \dots, \mu_{eP}$를 모두 연결하여 하나의 거대한 집합을 만든 후, 여기서 다시 샘플링하여 단일한 '평균 전문가 궤적' $\mu_e$를 생성한다.
- **보상 함수**: 에이전트의 상태 $s^a_t$와 샘플링된 전문가 상태 $s^e_{\eta_{k,t}}$ 사이의 Sliced Wasserstein 거리를 기반으로 의사 보상(Pseudo-reward)을 계산한다.
$$r(s^a_t) = \frac{1}{K} \sum_{k=1}^K |\langle s^a_t - s^e_{\eta_{k,t}}, \theta_k \rangle|$$

### 3. SMMOTIL (Sliced Multi-Marginal OT IL)
본 논문에서 제안하는 방법론으로, 데이터를 연결하지 않고 여러 전문가를 동시에 고려한다.
- **절차**: Multi-Marginal OT를 사용하여 에이전트 궤적 $\mu_a$와 $P$개의 전문가 궤적 $\mu_{e1}, \dots, \mu_{eP}$ 사이의 거리를 직접 측정한다. 이는 이론적으로 전문가들의 Wasserstein Barycenter와 에이전트 사이의 거리를 최소화하는 것과 같다.
- **보상 함수**: 모든 전문가의 상태들에 대한 기하학적 평균을 반영한 보상을 계산한다.
$$r_{t,p}(s^a_t, S) = \frac{1}{PK} \sum_{k=1}^K \left| \left\langle s^a_t - \frac{1}{P} \sum_{j=1}^P s^{(j)}_{\eta_{p,j,k}(t)}, \theta_k \right\rangle \right|^2$$

### 4. 학습 절차
두 방법 모두 에이전트로 **DQN(Deep Q-Network)**을 사용한다. 에이전트는 환경의 실제 보상 대신, 위에서 정의한 OT 기반의 의사 보상을 최대화하도록 학습된다.

## 📊 Results

### 실험 설정
- **환경**: OpenAI Gym의 `Pendulum-v0` 및 `CartPole-v0`.
- **데이터셋**: 전문가 시연 데이터를 두 가지 다양성 시나리오로 구성하였다.
    1. **Diverse Lengths**: 서로 다른 길이(Length)를 가진 전문가 5명.
    2. **Diverse Masses**: 서로 다른 질량(Mass)을 가진 전문가 5명.
- **지표**: 에피소드당 평균 보상(Mean moving reward per episode).

### 정량적 결과
실험 결과, 모든 시나리오에서 **SMMOTIL이 SCOTIL보다 지속적으로 높은 평균 보상을 기록**하였다. 특히 SCOTIL은 보상의 변동성(Variance)이 매우 크고 학습 과정이 불안정한 모습을 보인 반면, SMMOTIL은 안정적으로 수렴하며 더 높은 성능을 달성하였다.

## 🧠 Insights & Discussion

### 분석 및 강점
실험 결과는 전문가들의 데이터가 다양할 때(예: 물리적 특성이 다른 환경에서 수집된 데이터), 단순히 데이터를 연결하고 샘플링하는 방식(SCOTIL)은 전문가의 특성을 '노이즈'로 인식하게 만든다는 것을 보여준다. 반면 SMMOTIL은 Multi-Marginal OT를 통해 전문가들의 공통된 기하학적 중심(Barycenter)을 찾아내므로, 훨씬 더 강건하고 매끄러운 가이드라인을 에이전트에게 제공할 수 있다.

### 한계 및 향후 연구
- **차원 확장성**: 본 실험은 상대적으로 낮은 차원의 OpenAI Gym 환경에서 진행되었다. MuJoCo와 같은 더 복잡하고 고차원인 환경에서의 검증이 필요하다.
- **거리 척도**: 현재는 유클리드 거리 기반의 Sliced Wasserstein을 사용하였으나, 에이전트와 전문가가 서로 다른 상태 공간에 존재할 경우를 대비해 Gromov-Wasserstein 거리와 같은 더 일반적인 척도를 적용하는 연구가 가능할 것이다.

## 📌 TL;DR

본 논문은 다수의 전문가 시연 데이터를 결합할 때 발생하는 노이즈 문제를 해결하기 위해 **Sliced Multi-Marginal Optimal Transport**를 이용한 모방 학습 방법인 **SMMOTIL**을 제안한다. 단순히 데이터를 연결하는 기존 방식과 달리, 전문가들의 궤적을 기하학적으로 평균 내는 Barycenter 개념을 도입하여, 전문가 간의 다양성이 큰 상황에서도 에이전트가 안정적이고 효율적으로 학습할 수 있음을 증명하였다. 이는 향후 복잡한 환경에서 다양한 전문가의 데이터를 통합하여 학습시켜야 하는 강화학습 및 로보틱스 분야에 기여할 가능성이 높다.