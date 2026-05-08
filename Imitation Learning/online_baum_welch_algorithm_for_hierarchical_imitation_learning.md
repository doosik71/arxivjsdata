# Online Baum-Welch algorithm for Hierarchical Imitation Learning

Vittorio Giammarino and Ioannis Ch. Paschalidis (2021)

## 🧩 Problem to Solve

본 논문은 계층적 강화학습(Hierarchical Reinforcement Learning, HRL)의 핵심 과제인 옵션 발견(Option Discovery) 문제를 해결하고자 한다. HRL은 복잡한 작업을 하위 문제로 나누어 처리함으로써 강화학습의 확장성(Scalability) 문제를 해결하지만, 적절한 옵션 구조를 초기화하거나 발견하는 것이 매우 어렵다는 단점이 있다.

전문가(Expert)의 데이터가 존재할 때, 전문가의 시연(Demonstration)으로부터 계층적 정책을 직접 학습하는 계층적 모방 학습(Hierarchical Imitation Learning, HIL)을 통해 이 문제를 해결할 수 있다. 기존의 HIL 방식은 주로 Baum-Welch(BW) 알고리즘과 같은 Expectation-Maximization(EM) 계열의 알고리즘을 사용하여 은닉 마르코프 모델(Hidden Markov Model, HMM)의 추론 문제로 접근해 왔다.

그러나 기존의 Batch BW 알고리즘은 매 반복(Iteration)마다 전체 데이터셋에 대해 전방-후방 재귀(Forward-Backward Recursion)를 수행해야 하므로, 학습 데이터가 많아질수록 계산 비용과 메모리 소모가 극심해지는 한계가 있다. 따라서 본 논문의 목표는 데이터를 실시간으로 처리하여 계산 효율성을 높인 온라인(Online) BW 알고리즘을 제안하고, 이를 통해 end-to-end HIL을 효율적으로 수행하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

첫째, 옵션 확률 그래프 모델(Options Probabilistic Graphical Model, OPGM)에 적용 가능한 최초의 온라인 end-to-end HIL 알고리즘을 개발하였다. 이는 기존 Batch 방식의 데이터 전체 스캔 과정을 제거하고, 데이터를 온더플라이(on-the-fly)로 처리하는 온라인 재귀 구조를 도입한 것이다.

둘째, 기존의 온라인 HMM 학습 알고리즘들이 가졌던 정책 파라미터화의 제약을 완화하였다. 이를 통해 신경망(Neural Networks)과 같은 비매끄러운 함수 근사치(Non-smooth function approximations)를 사용하여 계층적 정책을 파라미터화할 수 있도록 설계하였다.

셋째, 다양한 RL 벤치마크 환경에서 제안한 온라인 알고리즘과 기존 Batch 알고리즘의 성능 및 효율성을 정량적으로 비교 분석하여, 특정 조건(데이터 양이 탐색된 상태-행동 공간보다 클 때)에서 온라인 방식이 우월함을 입증하였다.

## 📎 Related Works

기존 연구들은 주로 Batch 버전의 BW 알고리즘을 사용하여 HIL을 수행하였다. 이들은 전방-후방 분해(Forward-Backward Decomposition)를 통해 잠재 변수의 평활화 분포(Smoothing distributions)를 계산하는데, 이는 데이터셋의 크기에 비례하는 계산 복잡도를 가진다.

본 논문이 다루는 HIL은 계층적 보상 함수를 추론하는 Hierarchical Inverse Reinforcement Learning(HIRL)과는 차이가 있다. HIRL이 보상 함수를 통해 간접적으로 정책을 유도하는 반면, HIL은 전문가의 정책 구조를 직접적으로 학습한다. 제안된 방법론은 OPGM을 HMM의 특수한 사례로 간주하여 EM 알고리즘을 적용함으로써, 기존의 복잡한 보상 설계 없이도 계층적 구조를 직접 추론할 수 있다는 차별점을 가진다.

## 🛠️ Methodology

### 1. Options Probabilistic Graphical Model (OPGM)

본 논문은 에이전트의 의사결정 과정을 다음과 같은 세 가지 정책의 조합으로 정의한다.

- **고수준 정책 ($\pi_{hi}$):** 현재 상태 $s_t$에서 어떤 옵션 $o_t$를 선택할지를 결정한다.
- **저수준 정책 ($\pi_{lo}$):** 선택된 옵션 $o_t$ 하에서 어떤 구체적인 행동 $a_t$를 취할지를 결정한다.
- **종료 정책 ($\pi_b$):** 현재 수행 중인 옵션을 종료하고 새로운 옵션을 선택할지($b_t=1$) 아니면 계속 유지할지($b_t=0$)를 결정한다.

### 2. Batch BW 알고리즘

Batch BW는 다음의 두 단계를 반복한다.

- **E-step:** 현재 파라미터 $\theta^{old}$를 이용하여 데이터 전체에 대한 평활화 분포를 계산하고, Baum의 보조 함수 $Q(\theta|\theta^{old})$를 도출한다.
- **M-step:** $\theta = \arg \max Q(\theta|\theta^{old})$를 통해 파라미터를 업데이트한다.

### 3. Online BW 알고리즘

본 논문은 전체 데이터를 스캔하는 대신, 새로운 상태-행동 쌍 $(s_T, a_T)$이 들어올 때마다 업데이트되는 충분 통계량(Sufficient Statistic) $\phi_\theta^T$를 도입한다.

$\phi_\theta^T$는 다음과 같이 정의된다:
$$\phi_\theta^T(o', b, o, s, a) = \frac{1}{T} E_{\theta} \left[ \sum_{t=1}^{T} \mathbb{1}[O_{t-1}=o', B_t=b, O_t=o, S_t=s, A_t=a] \mid (s_t, a_t)_{1:T} \right]$$

이를 위해 두 가지 필터를 사용하여 온라인으로 업데이트를 수행한다:

1. **필터링 분포 $\chi_\theta^T(o)$:** 시간 $T$에서 옵션 $O_T=o$일 확률을 계산한다.
2. **조인트 기대값 $\rho_\theta^T$:** 옵션 전이 및 행동 발생에 대한 누적 기대값을 계산한다.

최종적으로 온라인 BW는 새로운 샘플이 들어올 때마다 E-step(필터 업데이트)을 수행하고, 일정 수 이상의 샘플($T > T_{min}$)이 쌓이면 M-step(정책 업데이트)을 수행하는 구조를 가진다.

### 4. Regularization Penalties

학습된 옵션이 해석 가능하고 전이 가능하도록 하기 위해 M-step의 목적 함수에 다음과 같은 규제항을 추가한다.

- **고수준 정책 규제:** 옵션 활성화의 희소성(Sparsity)을 강제하는 $L_b$와 옵션 간의 변별력을 높이는 분산 기반 규제 $L_v$를 도입한다.
- **저수준 정책 규제:** 서로 다른 옵션이 서로 다른 행동을 취하도록 유도하기 위해 옵션 간 $\pi_{lo}$의 KL-divergence($L_{D_{KL}}$)를 최대화한다.

최종 최적화 문제는 다음과 같다:
$$\theta^{(T)} \in \arg \max_{\theta \in \Theta} Q^T( \theta | \theta^{old}) - \lambda_b L_b + \lambda_v L_v + \lambda_{D_{KL}} L_{D_{KL}}$$

## 📊 Results

### 실험 설정

- **데이터셋 및 환경:** OpenAI Gym의 Cartpole, Pendulum, Lunar Lander(연속 상태 공간)와 Discrete Grid-world(이산 상태 공간 및 높은 확률성)를 사용하였다.
- **비교 대상:** Batch BW 알고리즘 vs 제안하는 Online BW 알고리즘.
- **측정 지표:** 전문가의 성과를 1로 정규화한 평균 보상(Average Reward)을 측정하였다.
- **모델 구조:** 모두 1개의 은닉층을 가진 Fully Connected Neural Network를 사용하였다.

### 주요 결과

- **연속 상태 공간 환경:** Lunar Lander, Pendulum, Cartpole에서는 두 알고리즘의 성능이 거의 유사하게 나타났다. 이는 해당 환경들에서 데이터의 양 $T$와 탐색된 상태-행동 공간의 크기 $|\tilde{S}| \times |\tilde{A}|$가 비슷했기 때문으로 분석된다.
- **이산 상태 공간 환경 (Grid-world):** Online BW가 Batch BW보다 우수한 성능을 보였다. 이는 Grid-world의 경우 $T \gg |\tilde{S}| \times |\tilde{A}|$ 관계가 성립하여, 온라인 업데이트 방식이 훨씬 더 효율적으로 파라미터를 수렴시켰기 때문이다.

## 🧠 Insights & Discussion

### 계산 복잡도 분석

- Batch BW의 E-step 복잡도는 $O(T \times |O|^2 \times |B|)$이며, 데이터 크기 $T$에 직접적으로 의존한다.
- Online BW의 E-step 복잡도는 $O(|\tilde{S}| \times |\tilde{A}| \times |O|^3 \times |B|^2)$이며, 탐색된 상태-행동 공간의 크기에 의존한다.

따라서 에이전트가 방문하는 상태-행동의 조합보다 전문가의 시연 데이터가 훨씬 많은 경우($T \gg |\tilde{S}| \times |\tilde{A}|$), 온라인 알고리즘이 압도적인 계산 효율성을 가진다.

### 한계 및 비판적 해석

본 논문은 온라인 알고리즘의 효율성을 입증하였으나, 실험에 사용된 환경들이 비교적 단순한 제어 작업이라는 한계가 있다. 상태 공간이 매우 거대하여 $|\tilde{S}| \times |\tilde{A}|$가 $T$보다 훨씬 커지는 환경에서는 오히려 온라인 방식의 복잡도가 증가할 수 있다. 하지만 저자들은 신경망을 통한 함수 근사를 도입함으로써 이 문제를 일부 완화하였으며, 향후 더 복잡하고 현실적인 환경에서의 검증이 필요하다.

## 📌 TL;DR

본 논문은 계층적 모방 학습(HIL)에서 계산 비용이 높은 Batch Baum-Welch 알고리즘을 대체하기 위해, 데이터를 실시간으로 처리하는 **Online Baum-Welch 알고리즘**을 제안하였다. OPGM을 HMM으로 정식화하고 충분 통계량의 온라인 업데이트 재귀식을 도입함으로써, 대규모 데이터셋에서도 효율적인 옵션 학습이 가능함을 보였다. 특히 데이터 양이 상태 공간의 크기보다 큰 환경에서 성능과 효율성 모두 향상됨을 입증하였으며, 이는 향후 대규모 전문가 데이터를 활용한 계층적 정책 학습의 확장성을 높이는 데 중요한 기여를 할 것으로 보인다.
