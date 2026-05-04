# f-GAIL: Learning f-Divergence for Generative Adversarial Imitation Learning

Xin Zhang, Yanhua Li, Ziming Zhang, Zhi-Li Zhang (2020)

## 🧩 Problem to Solve

모방 학습(Imitation Learning, IL)의 핵심 목표는 전문가의 시연(expert demonstrations)으로부터 학습자(learner)의 행동과 전문가의 행동 사이의 차이를 최소화하는 정책(policy)을 학습하는 것이다. 기존의 많은 IL 알고리즘들은 두 분포 사이의 차이를 측정하기 위해 KL-divergence, Jensen-Shannon (JS) divergence, Reverse KL (RKL) divergence 등 미리 정의된(pre-determined) divergence measure를 사용해 왔다.

그러나 특정 작업(task)에서 어떤 divergence measure를 선택하느냐에 따라 전문가 정책을 복원하는 정확도와 데이터 효율성이 크게 달라진다. 예를 들어, KL-divergence는 다중 모드를 포괄하는 mode-covering 특성이 있는 반면, RKL-divergence는 단일 모드에 집중하는 mode-seeking 특성을 가진다. 기존 방식처럼 수동으로 고정된 divergence를 선택하는 것은 잠재적인 모든 divergence 공간을 활용하지 못하게 하며, 결과적으로 최적보다 낮은 성능의(sub-optimal) 정책을 학습하게 만드는 원인이 된다. 따라서 본 논문은 전문가 시연 데이터가 주어졌을 때, 해당 작업에 가장 적합한 discrepancy measure를 $f$-divergence family 내에서 자동으로 학습하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 고정된 divergence를 사용하는 대신, **학습 가능한 $f$-divergence**를 도입하여 전문가 행동 분포와 학습자 행동 분포 사이의 차이를 측정하는 것이다. 이를 통해 각 작업의 특성에 맞는 최적의 divergence measure를 스스로 찾아내어 정책 학습의 가이드로 활용한다.

핵심적인 기술적 기여는 다음과 같다.
1. **Learnable Divergence Modeling**: 모방 학습 과정에서 $f$-divergence 공간으로부터 최적의 discrepancy measure를 자동으로 학습하는 $\text{f-GAIL}$ 프레임워크를 제안하였다.
2. **$f^*$-network 설계**: 유효한 $f$-divergence를 표현하기 위해 두 가지 제약 조건인 **볼록성(convexity)**과 **제로 갭(zero gap, $f(1)=0$)** 조건을 강제하는 신경망 구조를 설계하였다.
3. **성능 검증**: 6가지 물리 기반 제어 작업에서 기존의 고정 divergence 기반 베이스라인들보다 높은 데이터 효율성과 정책 복원 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 모방 학습 연구들은 주로 다음과 같은 접근 방식을 취했다.
- **Behavior Cloning (BC)**: 전문가 데이터를 지도 학습(supervised learning) 방식으로 학습하며, 이는 본질적으로 KL-divergence를 최소화하는 것과 같다. 하지만 covariate shift 문제로 인해 데이터 효율성이 낮고 오차가 누적되는 경향이 있다.
- **GAIL (Generative Adversarial Imitation Learning)**: GAN 구조를 사용하여 학습자 정책(generator)과 판별자(discriminator)를 함께 학습하며, JS-divergence를 최소화한다.
- **AIRL 및 RKL-VIM**: JS-divergence 대신 Reverse KL (RKL) divergence를 사용하여 보상 함수를 복원하거나 정책을 학습한다.

이러한 기존 연구들의 한계는 분석적 형태가 명확한 몇 가지 유명한 divergence measure에만 의존한다는 점이다. $\text{f-GAIL}$은 이러한 제한된 선택지를 넘어, $f$-divergence family라는 광범위한 함수 공간 내에서 데이터에 기반해 최적의 함수를 탐색한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 목적 함수
$\text{f-GAIL}$은 정책 네트워크 $\pi_\theta$, 보상 신호 네트워크 $T_\omega$, 그리고 $f$-divergence의 convex conjugate 함수인 $f^*_\phi$ 네트워크의 세 가지 구성 요소로 이루어진다. 이들의 상호작용은 다음과 같은 minimax 최적화 문제로 정의된다.

$$\min_{\pi} \max_{f^* \in \mathcal{F}^*, T} \mathbb{E}_{\pi^E}[T(s,a)] - \mathbb{E}_\pi[f^*(T(s,a))] - H(\pi)$$

여기서 $\pi^E$는 전문가 정책, $H(\pi)$는 정책의 엔트로피이며, $\mathcal{F}^*$는 유효한 $f$-divergence를 표현하는 함수 공간이다. $T$는 전문가와 학습자의 상태-행동 쌍을 구분하는 판별자 역할을 하며, $f^*$는 이 판별자의 출력을 통해 divergence를 계산하는 정규화 항으로 작용한다.

### 2. $f^*_\phi$ 네트워크의 제약 조건 구현
함수 $f^*$가 유효한 $f$-divergence가 되기 위해서는 다음 두 가지 조건이 필수적이다.

**가. 볼록성 제약 (Convexity Constraint)**
$f^*(u)$가 입력 $u$에 대해 볼록 함수여야 한다. 이를 위해 본 논문은 **Fully Input Convex Neural Network (FICNN)** 구조를 채택하였다. 
- 입력 $u$에서 모든 후속 레이어로 이어지는 shortcut 연결을 구성한다.
- 가중치 $W^{(z)}$를 양수로 제한(clipping)하고, 활성화 함수 $g$를 ReLU와 같은 볼록하고 비감소하는 함수로 설정하여 전체 네트워크 출력이 입력 $u$에 대해 볼록함을 보장한다.

**나. 제로 갭 제약 (Zero Gap Constraint)**
$\inf_{u} \{f^*(u) - u\} = 0$ 조건을 만족해야 한다. 이를 위해 매 에포크마다 다음 절차를 수행한다.
1. 경사 하강법을 통해 $\delta = \inf_{u} \{f^*(u) - u\}$를 추정한다.
2. 추정된 $\delta$를 이용하여 다음과 같이 함수를 이동(shift)시킨다.
   $$f^{*\prime}_\phi(u) = f^*_\phi(u - \frac{\delta}{2}) - \frac{\delta}{2}$$
이 연산을 통해 $f^*(u)$와 $u$ 사이의 최소 거리가 0이 되도록 강제한다.

### 3. 학습 절차 (Algorithm)
학습은 교대 경사법(alternating gradient method)을 통해 진행된다.
1. **판별자 업데이트**: 보상 신호 $T_\omega$와 $f^*_\phi$ 네트워크를 동시에 업데이트하여 목적 함수를 최대화한다. 이후 $\delta$를 추정하여 제로 갭 제약을 적용한다.
2. **정책 업데이트**: TRPO(Trust Region Policy Optimization)를 사용하여 목적 함수를 최소화하도록 정책 $\pi_\theta$를 업데이트한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 작업**: CartPole 및 MuJoCo 기반의 5개 복잡한 작업(HalfCheetah, Hopper, Reacher, Walker, Humanoid)을 사용하였다.
- **비교 대상 (Baselines)**: BC, GAIL, BC+GAIL, AIRL, RKL-VIM.
- **평가 지표**: 전문가 성능을 1, 랜덤 정책 성능을 0으로 정규화한 기대 보상(Expected Return)을 측정하였다.

### 2. 주요 결과
- **정책 복원 성능**: 모든 작업에서 $\text{f-GAIL}$이 모든 베이스라인을 압도하는 성능을 보였다. 특히 Hopper, Reacher, Walker, Humanoid와 같은 복잡한 작업에서 격차가 뚜렷했으며, 모든 데이터셋 규모에서 전문가 성능의 최소 80% 이상을 달성하였다.
- **데이터 효율성**: 전문가의 궤적(trajectory) 수가 적은 환경에서도 $\text{f-GAIL}$은 안정적으로 높은 성능을 유지하였다. 반면 BC나 BC+GAIL은 데이터가 부족할 때 성능이 급격히 저하되는 covariate shift 문제를 보였다.
- **학습된 $f^*$의 특성**: 실험 결과, $\text{f-GAIL}$은 각 작업에 맞는 고유한 $f^*_\phi(u)$ 함수를 학습하였다. 유사한 성격의 작업(예: CartPole과 Reacher)은 유사한 형태의 $f^*$ 함수를 학습하는 경향을 보였다.
- **입력 분포 분석**: 학습된 정책이 전문가에 가까울수록 $T_\omega(s,a)$의 분포가 제로 갭 영역($f^*(u) \approx u$)에 집중되고 표준편차가 작아진다. $\text{f-GAIL}$은 베이스라인들보다 이 두 가지 기준을 훨씬 더 잘 만족시키는 것으로 나타났다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 divergence measure를 고정된 상수로 취급하지 않고 학습 가능한 파라미터로 확장함으로써, 모방 학습의 유연성을 크게 높였다. 특히 FICNN과 shifting operation을 통해 수학적 제약 조건을 딥러닝 구조로 성공적으로 구현하여 $f$-divergence 공간을 효과적으로 탐색하였다는 점이 돋보인다. 학습된 $f^*$ 함수가 작업의 유사성에 따라 비슷하게 형성된다는 결과는, 각 작업마다 최적의 "차이 측정 방식"이 존재하며 이를 데이터로부터 추출할 수 있음을 시사한다.

### 한계 및 향후 과제
본 논문은 $f$-divergence family 내에서의 탐색에 집중하였으나, 이 공간에는 Wasserstein distance와 같은 중요한 거리 척도가 포함되지 않는다. 저자들은 향후 연구에서 $f$-divergence와 Wasserstein distance를 모두 포괄하는 $c$-Wasserstein distance family로 확장할 가능성을 언급하였다. 또한, 이 구조를 $\text{f-GAN}$이나 $\text{f-EBM}$에 결합하여 이미지나 오디오 생성 모델의 품질을 높이는 데 활용할 수 있을 것으로 기대된다.

## 📌 TL;DR

$\text{f-GAIL}$은 모방 학습 시 전문가와 학습자 사이의 차이를 측정하는 **divergence measure 자체를 데이터로부터 학습**하는 모델이다. 볼록성(Convexity)과 제로 갭(Zero-gap) 제약을 가진 신경망 구조를 통해 유효한 $f$-divergence를 학습하며, 이를 통해 기존의 고정된 divergence(JS, KL 등)를 사용하는 방식보다 **더 높은 데이터 효율성과 뛰어난 정책 복원 성능**을 달성하였다. 이는 향후 로봇 제어 및 시스템 자동화 분야에서 전문가의 소수 시연만으로 정교한 정책을 학습하는 데 기여할 수 있다.