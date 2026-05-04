# OLLIE: Imitation Learning from Offline Pretraining to Online Finetuning

Sheng Yue, Xingyuan Hua, Ju Ren, Sen Lin, Junshan Zhang, Yaoxue Zhang (2024)

## 🧩 Problem to Solve

본 논문은 정적인 데모 데이터로부터 정책을 먼저 학습하는 Offline Imitation Learning(IL)과, 이후 환경과의 상호작용을 통해 정책을 정교화하는 Online IL을 결합하는 'Offline-to-Online IL' 문제를 다룬다. 일반적으로 Offline IL로 사전 학습된 정책을 Online IL(예: GAIL)로 미세 조정(Finetuning)하면, 기대와 달리 성능이 초기에 급격히 하락하거나 사전 학습된 지식을 잃어버리는 **Unlearning** 현상이 발생한다.

이러한 문제의 핵심 원인은 **Discriminator Misalignment**에 있다. Online IL에서 사용되는 Discriminator는 보통 무작위로 초기화되는데, 이는 사전 학습된 정책의 초기 상태와 일치하지 않는다. 결과적으로 Discriminator가 잘못된 로컬 보상 신호를 제공하여 정책 최적화 방향을 오도하게 되고, 이는 결국 사전 학습 단계에서 획득한 유용한 지식을 파괴하는 결과를 초래한다. 따라서 본 논문의 목표는 사전 학습된 정책과 정렬된(Aligned) Discriminator를 동시에 학습하여, 성능 저하 없이 빠르고 부드럽게 온라인 미세 조정으로 전환할 수 있는 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **OLLIE(OffLine-to-onLine Imitation lEarning)**라는 원칙적인 Offline-to-Online IL 프레임워크를 제안한 것이다. 

OLLIE의 중심 아이디어는 Offline 단계에서 단순히 정책만 학습하는 것이 아니라, **정책의 상태-행동 분포와 정렬된 Discriminator의 초기값을 함께 도출**하는 것이다. 이를 위해 논문은 표준 IL 목적 함수를 보조 목적 함수(Surrogate Objective)로 변환하고, Convex Conjugate를 이용하여 이를 Convex-Concave Stochastic Saddle Point(SSP) 문제로 재정의한다. 이 과정을 통해 추가적인 환경 상호작용이나 계산 비용 없이, 최적의 정책 초기값과 이에 정렬된 Discriminator 초기값을 동시에 얻을 수 있으며, 이를 통해 Online 단계에서 GAIL을 통한 효율적인 미세 조정이 가능해진다.

## 📎 Related Works

### 기존 연구 및 한계
1.  **Online IL (예: GAIL):** 환경과의 지속적인 상호작용을 통해 전문가의 분포를 모방한다. 성능은 우수하지만, 상호작용 비용이 매우 크고 초기 단계의 무작위 행동으로 인해 안전성이 낮은 환경에서는 적용이 어렵다.
2.  **Offline IL (예: BC, Offline IRL):** 정적 데이터만 사용하므로 안전하고 경제적이다. 하지만 전문가 데이터의 부족으로 인한 **Covariate Shift** 문제와 이로 인한 오차 누적(Error Compounding) 문제에 취약하다.
3.  **Offline-to-Online RL:** Offline RL로 사전 학습 후 Online RL로 미세 조정하는 패러다임이 성공을 거두었으나, IL에서는 보상 함수가 주어지지 않으므로 보상 함수를 먼저 학습해야 하는 Offline IRL 과정이 필요하며, 이 과정에서 보상 외삽 오류(Reward Extrapolation Error)가 빈번하게 발생한다.

### OLLIE의 차별점
OLLIE는 중간 단계의 보상 추론 과정(Intermediate IRL)을 우회한다. 기존의 Offline IL 방법들이 단순히 정책 추출에 집중하거나 편향된 그래디언트 추정치에 의존했던 것과 달리, OLLIE는 수학적으로 정밀하게 설계된 SSP 문제를 통해 **편향 없는(Unbiased) 오프라인 최적화**를 수행하며, 특히 온라인 단계로의 매끄러운 전환을 위한 '정렬된 Discriminator'를 직접 도출한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인
OLLIE는 크게 두 단계로 구성된다.
- **Offline Phase:** 전문가 데이터($D_e$)와 불완전한 데이터($D_s$)를 모두 활용하여 최적의 정책 $\pi^*$와 정렬된 Discriminator $D^0$를 학습한다.
- **Online Phase:** 학습된 $\pi^*$와 $D^0$를 GAIL의 초기값으로 설정하여 환경 상호작용을 통해 빠르게 미세 조정을 수행한다.

### 2. 주요 구성 요소 및 상세 방법론

#### (1) Surrogate Objective (보조 목적 함수)
표준 IL의 목적 함수인 역 KL-발산(Reverse KL-divergence) 최소화 문제를 다음과 같은 형태로 변환하여 불완전한 데이터($D_s$)를 활용할 수 있게 한다.
$$\max_{\pi} \mathbb{E}_{(s,a)\sim\rho^\pi} [\tilde{R}(s,a)] - D_{KL}(\rho^\pi \| \tilde{\rho}_o)$$
여기서 $\tilde{\rho}_o$는 전문가 데이터와 불완전한 데이터의 합집합 분포이며, $\tilde{R}(s,a) = \log \frac{\tilde{\rho}_e(s,a)}{\tilde{\rho}_o(s,a)}$는 보조 보상 함수로 작용한다.

#### (2) Stochastic Saddle Point (SSP) 문제로의 변환
위의 목적 함수를 직접 최적화하면 편향된 그래디언트 문제가 발생한다. 이를 해결하기 위해 라그랑주 승수법과 **Convex Conjugate**를 적용하여 다음과 같은 Min-Max 형태의 SSP 문제로 변환한다.
$$\min_{\nu} \max_{y} \tilde{F}(\nu, y) = \mathbb{E}_{(s,a,s')\sim D_o} [\tilde{\delta}_\nu(s,a,s')y(s,a) - y(s,a) \log y(s,a)] + (1-\gamma)\mathbb{E}_{s\sim D_o(s_0)} [\nu(s)]$$
여기서 $\tilde{\delta}_\nu(s,a,s') = \tilde{R}(s,a) + \gamma\nu(s') - \nu(s)$이며, $\nu$는 가치 함수, $y$는 지수적 이득(Exponential Advantage)과 관련된 변수이다. 이 형태는 편향 없는 확률적 그래디언트를 사용할 수 있게 하여 수렴성과 일반화 성능을 보장한다.

#### (3) Offline 정책 추출 (Policy Extraction)
최적의 $y^*$를 얻으면, 복잡한 계산 없이 **Weighted Behavior Cloning**을 통해 최적 정책 $\pi^*$를 추출할 수 있다.
$$\max_{\pi} \mathbb{E}_{(s,a)\sim D_o} [y^*(s,a) \log \pi(a|s)]$$
이는 전문가 분포에 더 가까운 샘플에 더 높은 가중치를 두어 학습하는 방식이다.

#### (4) 정렬된 Discriminator 도출 (Aligned Discriminator)
OLLIE의 가장 핵심적인 부분으로, 온라인 학습의 시작점인 Discriminator $D^0$를 다음과 같이 도출한다.
$$D^0(s,a) = \left( 1 + \frac{d^*(s,a)}{1-d^*(s,a)} \cdot \frac{1}{y^*(s,a)} \right)^{-1}$$
여기서 $d^*$는 오프라인 단계에서 학습한 밀도 비율 추정기이다. 이 수식은 사전 학습된 정책 $\pi^*$의 정상 상태 분포 $\rho^*$와 GAIL의 최적 Discriminator 형태를 일치시킴으로써, 온라인 학습 시작 시점의 Misalignment를 완전히 제거한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 작업:** AntMaze, Adroit, MuJoCo, FrankaKitchen, Vision-based Robomimic 등 총 5개 도메인, 21개 작업에서 평가를 진행하였다.
- **비교 대상:** BC, NBCU, CSIL, DWBC, MLIRL, ISWBC 등 최신 Offline IL 방법론 및 GAIL(from scratch)과 비교하였다.
- **평가 지표:** 정규화된 리턴(Normalized Return), 수렴 속도, 데모 효율성 등을 측정하였다.

### 2. 주요 결과
- **Offline IL 성능:** OLLIE는 거의 모든 작업에서 기존 베이스라인을 압도하였다. 특히 전문가 데이터가 매우 적은 상황에서도 기존 방법 대비 2~4배 높은 성능을 보였으며, 수렴 속도가 매우 빨랐다.
- **Online Finetuning 성능:** 
    - 기존 Offline IL 방법들로 사전 학습 후 GAIL로 미세 조정을 하면 초기에 성능이 급락하는 현상이 나타났으나, OLLIE는 이러한 **Unlearning 없이** 즉시 성능이 상승하는 곡선을 보였다.
    - 매우 적은 상호작용(종종 10 에피소드 이내)만으로도 전문가 수준의 성능에 도달하였다.
    - GAIL을 처음부터 학습시켰을 때 실패하는 고차원 비전 작업(can, square 등)에서도 OLLIE의 사전 학습 덕분에 성공적인 수행이 가능했다.
- **강건성:** 불완전한 데이터의 품질이 낮더라도(Random 수준) 일관된 성능 향상을 보였으며, 고차원 환경에서도 확장성이 입증되었다.

## 🧠 Insights & Discussion

### 강점 및 해석
OLLIE는 Offline-to-Online IL에서 발생하는 고질적인 문제인 '초기 성능 하락'을 수학적인 정렬(Alignment)을 통해 해결하였다. 특히, 복잡한 World Model을 구축하지 않는 Model-free 방식임에도 불구하고, SSP 문제로의 변환을 통해 Offline 단계에서 Dynamics 정보를 효과적으로 활용했다는 점이 고무적이다. 또한, 사전 학습된 정책과 Discriminator를 동시에 제공함으로써 Online 단계의 탐색 비용을 획기적으로 줄였다.

### 한계 및 향후 과제
본 논문의 한계는 미세 조정 단계에서 여전히 GAIL이라는 Adversarial 방식에 의존한다는 점이다. GAIL의 특성상 샘플 효율성이 낮고 학습이 불안정할 수 있는 잠재적 위험이 있다. 저자들은 향후 연구로 Non-adversarial 방식이나 Off-policy IL 방법론과의 결합 가능성을 제시하였다. 또한, 레이블이 없는 데이터(Unlabeled data)를 어떻게 더 효율적으로 활용할 것인가에 대한 탐구가 필요하다.

## 📌 TL;DR

OLLIE는 Offline IL로 사전 학습된 정책을 Online IL로 미세 조정할 때 발생하는 **지식 망각(Unlearning)** 문제를 해결하기 위해, **정책과 정렬된 Discriminator 초기값을 동시에 학습**하는 프레임워크이다. SSP(Stochastic Saddle Point) 최적화를 통해 편향 없이 오프라인 정책을 추출하고, 이를 바탕으로 정렬된 Discriminator를 직접 도출함으로써 온라인 학습의 시작점을 최적화한다. 실험 결과, 고차원 비전 작업과 복잡한 로봇 제어 작업 모두에서 기존 방법론 대비 월등한 데모 효율성과 빠른 수렴 속도를 입증하였으며, 이는 실제 로봇 시스템처럼 상호작용 비용이 매우 큰 분야에 적용될 가능성이 매우 높다.