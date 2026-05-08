# Complex Skill Acquisition Through Simple Skill Imitation Learning

Pranay Pasula(2020)

## 🧩 Problem to Solve

본 논문은 복잡한 기술(complex skills)을 학습할 때, 이를 구성하는 단순한 하위 기술(simple subskills)들의 조합으로 이해하고 학습하는 인간의 인지 방식을 모방하고자 한다. 기존의 계층적 강화학습(Hierarchical Reinforcement Learning) 연구들은 주로 하위 작업들이 순차적으로 일어나는 **Sequential subtasks**에 집중해 왔다. 하지만 실제의 복잡한 동작(예: 백플립)은 점프하기, 무릎 굽히기, 뒤로 구르기, 팔 아래로 뻗기 등 여러 하위 동작이 동시에 일어나는 **Concurrent subtasks**의 조합으로 이루어져 있다.

따라서 본 연구의 목표는 복잡한 작업을 독립적이고 해석 가능한 동시적 하위 작업들로 분해하여 학습할 수 있는 알고리즘을 제안하는 것이다. 이를 통해 에이전트가 복잡한 동작을 더 효율적으로 학습하게 하고, 의미론적으로 해석 가능한 잠재 공간(latent space)을 구축하여 새로운 동작의 생성이나 세밀한 동작 조정이 가능하게 하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **잠재 공간의 구조화(Latent Space Restructuring)**이다. 구체적으로, 복잡한 동작의 임베딩(embedding)이 해당 동작을 구성하는 하위 기술들의 임베딩 합과 같아지도록 유도하는 것이다.

즉, 복잡한 동작 $A$의 임베딩을 $z_A$라 하고, 이를 구성하는 하위 기술들의 임베딩을 $z_a, z_b, z_c, z_d$라고 한다면, 다음과 같은 관계가 성립하도록 잠재 공간을 형성한다.
$$z_A = z_a + z_b + z_c + z_d$$

이러한 설계를 통해 에이전트는 단순한 기술들을 먼저 학습함으로써 구축된 잠재 공간을 활용하여, 학습하기 어려운 복잡한 기술의 모방 학습(imitation learning) 속도를 가속화하고 성능을 향상시킬 수 있다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 차별점을 제시한다.

- **계층적 강화학습 (Hierarchical RL):** 기존 연구들은 주로 순차적 하위 작업의 조합에 집중하였으나, 본 논문은 동시적(concurrent) 하위 작업의 조합이라는 관점을 제시한다.
- **해석 가능한 표현 학습 (Interpretable Representation Learning):** InfoGAN이나 $\beta$-VAE와 같이 잠재 공간의 disentanglement를 추구하는 연구들의 직관을 차용하여, 동작 임베딩 간의 가산적(additive) 관계를 유도한다.
- **CVAE 기반 궤적 학습:** 기존의 CVAE를 사용하여 궤적을 인코딩하고 복원하는 방식을 사용하지만, 단순히 재구성 오차를 줄이는 것을 넘어 잠재 공간에 구체적인 의미론적 구조를 강제한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (CVAE Architecture)

본 논문은 궤적(trajectory)을 인코딩하고 생성하기 위해 Conditional Variational Autoencoder (CVAE)를 사용한다. 구조는 다음과 같다.

- **Encoder $q_\phi(z|s_{1:T})$:** Bi-directional LSTM(BiLSTM)과 Attention 모듈을 사용하여 상태 시퀀스를 저차원의 잠재 변수 $z$의 분포로 매핑한다.
- **State Decoder $P_\psi(s_{t+1}|s_t, z)$:** Conditional WaveNet을 사용하여 다음 상태를 예측하는 dynamics model 역할을 수행하며, 다봉 분포(multi-modal dynamics)를 정밀하게 모델링한다.
- **Action Decoder $\pi_\theta(a_t|s_t, z)$:** MLP를 사용하여 주어진 상태와 잠재 변수로부터 행동 $a_t$를 샘플링하는 정책(policy) 역할을 수행한다.

기본적인 CVAE의 학습 목적 함수는 다음과 같다.
$$L(\theta,\phi,\psi;\tau_i) = -\mathbb{E}_{z \sim q_\phi(z|s_{i1:T}^i)} \left[ \sum_{t=1}^{T_i} \log \pi_\theta(a_{it}|s_{it}, z) + \log P_\psi(s_{it+1}|s_{it}, z) \right] + D_{KL}(q_\phi(z|s_{i1:T}^i) \| p(z))$$

### 2. 잠재 공간 형성 (Shaping the Latent Space)

복잡한 동작 $V$가 $M$개의 하위 기술 $\{\tau^{(1)}, \dots, \tau^{(M)}\}$로 구성된다고 할 때, 하위 기술들의 임베딩 합 $V = z_1 + z_2 + \dots + z_M$ (여기서 $z_i \sim q_\phi(z|s^{(i)}_{1:T^{(i)}})$)와, 이 $V$를 조건으로 생성된 궤적 $\tilde{\tau}$ 사이의 **상호 정보량(Mutual Information, MI)**을 최대화한다.

$$I(V; \tilde{\tau}) = H(V) - H(V|\tilde{\tau})$$

### 3. 상호 정보량의 하한선 도출 (Approximation of MI)

실제 사후 분포 $p(V|\tilde{\tau})$를 알 수 없으므로, 본 논문은 변분 추론(variational inference) 없이도 계산 가능한 하한선(lower bound)을 유도하였다.

- **Variational Approximation 방식:** 분포 $Q(V|\tilde{\tau})$를 도입하여 하한선을 구하는 방식이다.
- **Non-variational Approximation 방식:** 재귀적 엔트로피 전개를 통해 다음과 같은 근사식을 도출하였다.
$$I(V; \tilde{\tau}) \geq -\mathbb{E}_{V \sim p(V)}[\log p(V)] + \sum_{i=1}^M \mathbb{E}_{z_i \sim q_\phi(z_i|\tilde{\tau})}[\log q_\phi(z_i|\tilde{\tau})]$$
여기서 $p(V)$와 $q_\phi$가 가우시안 분포를 따른다고 가정하면, 엔트로피 $H(V)$는 닫힌 형태(closed-form)의 수식으로 간단히 계산될 수 있다.

### 4. 최종 손실 함수

최종적으로 기본 VAE 목적 함수에 상호 정보량 정규화 항을 추가한다.
$$L_{final} = L_{VAE} + \lambda L_I$$
여기서 $\lambda$는 잠재 공간의 구조화 정도를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **환경:** Bullet 시뮬레이터 기반의 Humanoid (상태 공간 197차원, 행동 공간 34차원).
- **대상 작업:** 'Spin Kick' (Kick, Spin, Jump의 조합) 및 'Backflip' (Jumping, Tucking knees, Rolling backwards, Thrusting arms downwards의 조합).
- **비교 대상:** 기본 VAE 목적 함수를 사용한 baseline과 제안된 정규화 방법(Variational 및 Non-variational 방식)을 비교하였다.
- **지표:** 생성된 궤적의 상태와 실제 시연(demonstration) 상태 간의 평균 제곱 오차(MSE).

### 결과

- **정량적 결과:** Figure 3에 따르면, 제안된 정규화 방법(Regularized)이 baseline(Original)보다 MSE가 낮으며, 더 빠르게 수렴하는 경향을 보였다.
- **결론:** 하위 기술들에 대해 먼저 학습하고 잠재 공간을 계층적 구조로 유도함으로써, 학습하기 어려운 복잡한 작업의 학습을 가속화하고 성능을 높일 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 연구는 복잡한 동작을 단순한 동작들의 '합'으로 정의하고, 이를 잠재 공간의 기하학적 구조로 강제함으로써 모방 학습의 효율성을 높였다는 점에서 독창적이다. 특히, 계산 비용이 높은 변분 추론을 피할 수 있는 근사식을 유도하여 실용성을 높였다.

### 한계 및 향후 과제

- **동시성 가정의 엄격함:** 제안된 방법이 모든 복잡한 동작의 '동시적 조합'이라는 가정에 의존하고 있으나, 실제 동작들이 얼마나 엄격하게 이 관계를 따르는지에 대한 추가 검증이 필요하다.
- **확장성:** 실험이 소수의 동작(Spin kick, Backflip)으로 제한되었다. 더 많은 종류의 동작 데이터셋을 통해 잠재 공간의 일반화 능력을 평가할 필요가 있다.
- **분해능 제어:** $\beta$-CVAE 등을 도입하여 잠재 변수 $z$의 disentanglement 정도를 더욱 세밀하게 조절할 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 복잡한 동작을 여러 단순 하위 동작의 동시적 조합으로 보고, **복잡한 동작의 임베딩이 하위 동작 임베딩들의 합이 되도록 잠재 공간을 구조화**하는 CVAE 기반의 모방 학습 알고리즘을 제안한다. 이를 통해 상호 정보량(Mutual Information)을 최대화함으로써 학습 속도와 정확도를 모두 향상시켰으며, 이는 향후 로봇의 정교한 동작 생성 및 제어 연구에 기여할 가능성이 크다.
