# Latent Policies for Adversarial Imitation Learning

Tianyu Wang, Nikhil Karnwal, Nikolay Atanasov (2022)

## 🧩 Problem to Solve

본 논문은 로봇의 보행(locomotion) 및 조작(manipulation) 작업에서 전문가의 시연(expert demonstrations)으로부터 효율적으로 정책을 학습하는 문제를 다룬다. 기존의 Generative Adversarial Imitation Learning (GAIL)은 전문가와 에이전트의 전이(transition)를 구분하는 Discriminator를 학습시키고, 여기서 생성된 보상을 통해 Generator인 정책을 최적화하는 강력한 방법론이다.

그러나 고차원 상태-행동 공간을 가진 복잡한 환경에서 GAIL은 다음과 같은 심각한 문제에 직면한다. 첫째, Discriminator가 과적합(overfitting)되기 쉽다. 둘째, 작업과 무관한 특징(task-irrelevant features)과 클래스 라벨 사이의 가짜 상관관계(spurious associations)를 이용함으로써 Generator에게 유용한 정보를 제공하지 못하는 경우가 많다. 결과적으로 Generator와 Discriminator 사이의 섬세한 균형을 맞추는 것이 매우 어려우며, 이는 학습의 불안정성과 수렴 속도 저하로 이어진다. 따라서 본 연구의 목표는 고차원 환경에서도 안정적으로 작동하는 모방 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 원본 행동 공간(original action space)이 아닌, 적절하게 설계된 **저차원 잠재 작업 공간(low-dimensional latent task space)**에서 모방 학습을 수행하는 것이다.

연구진은 로봇 팔의 관절 토크(joint torques)와 같은 고차원 제어 신호는 복잡하지만, 이를 추상화한 말단 장치(end-effector)의 포즈 변화와 같은 잠재 공간에서의 표현은 상대적으로 단순하다는 직관에 주목하였다. 이를 위해 **LAtent Policy using Adversarial imitation Learnin(LAPAL)**을 제안하며, Conditional Variational Autoencoder (CVAE)를 통해 행동 인코더-디코더 모델을 학습시켜 효율적인 잠재 행동 표현을 획득한다. 이 방식은 학습 과정의 안정성을 높이고, 고차원 환경에서도 전문가 수준의 성능에 빠르게 도달하게 하며, 더 나아가 서로 다른 로봇 구성 간의 정책 전이(transfer learning)를 가능하게 한다.

## 📎 Related Works

기존의 Inverse Reinforcement Learning (IRL)은 전문가의 시연으로부터 보상 함수를 추론하고 이를 최적화하는 방식을 사용한다. GAIL과 같은 Adversarial Imitation Learning (AIL) 방법론들은 IRL과 GAN의 연결 고리를 활용하여 보상 함수를 명시적으로 설계하지 않고도 전문가의 분포를 모사한다.

잠재 표현 학습(latent representation learning)에 관한 기존 연구들은 주로 부분 관측 마르코프 결정 과정(POMDP)에서 시각적 관측값으로부터 잠재 상태(latent state)를 학습하는 것에 집중해 왔다. 또한 RL 분야에서 latent action space를 학습하려는 시도가 있었으나, 이는 주로 효율적인 탐색이나 오프라인 RL의 성능 향상을 목적으로 하였다. 반면, 본 논문은 완전 관측 MDP 환경임에도 불구하고 **행동 공간의 고차원성**으로 인해 발생하는 AIL의 학습 불안정성 문제를 해결하기 위해 잠재 행동 공간을 도입했다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 잠재 공간 정의

본 논문은 원본 MDP $M=\{S, A, T, r, \gamma\}$를 잠재 MDP $\bar{M}=\{S, \bar{A}, \bar{T}, \bar{r}, \gamma\}$로 확장한다. 여기서 $\bar{A}$는 저차원의 잠재 행동 공간이다. 행동 인코더 $g: A \to \bar{A}$와 디코더 $h: \bar{A} \to A$가 존재하여 $\bar{T}(s, g(a)) = T(s, a)$와 $\bar{r}(s, g(a)) = r(s, a)$를 만족한다고 가정한다.

### 2. Action Encoder-Decoder 학습 (CVAE)

잠재 행동 표현을 학습하기 위해 CVAE를 사용한다. 인코더 $g_\omega^1$은 상태 $s$가 주어졌을 때 행동 $a$를 잠재 분포 $E(\bar{a}|s)$로 매핑하고, 디코더 $h_\omega^2$는 상태 $s$와 잠재 행동 $\bar{a}$를 다시 원본 행동 $a$로 복원한다. 학습을 위한 손실 함수는 다음과 같다.

$$L_{CVAE}(s, a) = \|a - h_\omega^2(s, g_\omega^1(s, a))\|^2 + \beta D_{KL}[E(\bar{a}|s) || p(\bar{a})]$$

여기서 첫 번째 항은 재구성 손실(reconstruction loss)이며, 두 번째 항은 잠재 분포를 정규화하는 KL 발산 항이다. 이 모델은 전문가의 시연 데이터만을 사용하여 오프라인으로 학습된다.

### 3. LAPAL의 두 가지 학습 모드

#### (1) Task-agnostic LAPAL

인코더-디코더 파라미터 $\omega_1, \omega_2$를 고정시킨 상태에서 잠재 공간 $\bar{A}$ 상에서 GAIL을 수행한다.

- **Discriminator ($D_\phi$):** $(s, \bar{a})$ 쌍이 전문가의 것인지 에이전트의 것인지 분류한다.
- **Generator ($\bar{\pi}_\theta$):** 상태 $s$에서 잠재 행동 $\bar{a}$를 예측하며, 보상 $\bar{r}(s, \bar{a}) = -\log(1 - D_\phi(s, \bar{a}))$를 최대화하도록 Soft Actor-Critic (SAC)을 통해 학습된다.
- **실행:** 예측된 $\bar{a}$는 디코더 $h_{\omega_2}$를 통해 원본 행동 $a$로 변환되어 환경에 적용된다.

#### (2) Task-aware LAPAL

인코더-디코더 파라미터 $\omega_1, \omega_2$를 고정하지 않고 Discriminator 및 Generator와 함께 동시에 최적화한다.

- **Discriminator 업데이트:** $\phi$와 $\omega_1$을 동시에 업데이트하여 전문가와 에이전트의 잠재 행동 표현을 더 잘 구분하도록 한다.
- **Generator 업데이트:** $\bar{\theta}$와 $\omega_2$를 동시에 업데이트하여 보상을 최대화하는 잠재 행동과 그에 맞는 디코딩 방식을 학습한다.

## 📊 Results

### 1. 실험 설정

- **환경:** MuJoCo의 보행 환경 4종(HalfCheetah, Walker2d, Ant, Humanoid) 및 robosuite의 조작 환경 1종(Door).
- **비교 대상:** GAIL (Baseline).
- **데이터:** 각 환경당 64개의 전문가 시연 데이터 사용.
- **지표:** 누적 보상(Return) 및 학습 곡선의 수렴 속도.

### 2. 주요 결과

- **고차원 환경에서의 우위:** Ant, Humanoid, Door와 같은 고차원 시스템에서 LAPAL은 GAIL보다 훨씬 빠른 수렴 속도를 보였다. 특히 Humanoid-v3 환경에서 GAIL은 별도의 정규화(Gradient Penalty 등) 없이는 최적 정책을 찾지 못했으나, LAPAL은 안정적으로 전문가 수준의 성능에 도달하였다.
- **저차원 환경에서의 특성:** HalfCheetah-v3와 같은 저차원 환경에서는 Task-agnostic LAPAL이 다소 낮은 성능을 보였는데, 이는 잠재 공간으로의 투영 과정에서 정보 손실이 발생하여 원래의 모방 학습 목적 함수의 하한(lower bound)을 최적화했기 때문으로 분석된다.
- **전이 학습 (Transfer Learning):** Panda 로봇으로 학습한 잠재 정책 $\bar{\pi}_s$를 Sawyer 로봇에 적용하는 실험을 진행했다. Sawyer 로봇에 대해서는 새로운 디코더 $h_t$만 학습시켜 결합한 결과, 추가적인 환경 상호작용 없이도 전문가 성능의 상당 부분(평균 리턴 736 / 전문가 863)을 달성하여 잠재 정책의 일반화 가능성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 행동 공간을 추상화하여 저차원 잠재 공간에서 학습하는 것이 고차원 제어 문제에서 AIL의 고질적인 문제인 학습 불안정성을 해결하는 효과적인 방법임을 보여주었다. 특히, 로봇의 구체적인 하드웨어 구성(joint configuration)과 상관없이 작업의 본질적인 구조(task structure)를 잠재 공간에 담아낼 수 있다는 점이 인상적이다.

다만, 이론적 한계로서 데이터 처리 정리(Data Processing Theorem)에 의해 잠재 공간에서의 모방 학습 목적 함수가 원본 공간 목적 함수의 하한이 된다는 점이 언급되었다. 이는 행동 차원이 이미 낮은 환경에서는 오히려 성능 저하를 일으킬 수 있음을 시사하며, 잠재 공간의 차원을 결정하는 하이퍼파라미터 설정이 중요함을 의미한다. 또한, 현재는 행동 공간만을 잠재화하였으나, 상태 공간까지 함께 잠재화한다면 로봇 기종 간의 전이 학습 성능을 더욱 극대화할 수 있을 것으로 기대된다.

## 📌 TL;DR

본 연구는 고차원 행동 공간을 가진 로봇 제어 작업에서 GAIL의 학습 불안정성을 해결하기 위해, CVAE를 이용해 행동 공간을 저차원으로 압축한 **LAPAL** 프레임워크를 제안하였다. 실험 결과, 고차원 환경에서 GAIL보다 훨씬 빠르고 안정적으로 수렴하며, 학습된 잠재 정책을 다른 로봇으로 전이(transfer)할 수 있음을 확인하였다. 이는 복잡한 로봇 제어 시스템에서 행동 추상화를 통해 모방 학습의 효율성을 획기적으로 높일 수 있음을 시사한다.
