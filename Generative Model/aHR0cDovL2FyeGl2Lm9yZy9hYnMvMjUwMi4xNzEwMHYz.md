# Generative Models in Decision Making: A Survey

Yinchuan Li, Xinyu Shao, Jianping Zhang, Haozhi Wang, Leo Maxime Brunswic, Kaiwen Zhou, Jiqian Dong, Kaiyang Guo, Xiu Li, Zhitang Chen, Jun Wang, Jianye Hao (2025)

## 🧩 Problem to Solve

본 논문은 최근 이미지 및 텍스트 생성 분야에서 비약적인 발전을 이룬 생성 모델(Generative Models)을 의사 결정(Decision Making) 과정에 통합하려는 시도와 그 방법론을 체계적으로 분석하는 것을 목표로 한다.

전통적인 순차적 의사 결정(Sequential Decision Making) 방식인 동적 계획법(Dynamic Programming, DP)과 강화 학습(Reinforcement Learning, RL)은 주로 보상 기반의 시행착오(Trial-and-Error)나 미리 정의된 상태에 의존한다. 이러한 방식은 탐색(Exploration)의 효율성이 떨어지고, 새로운 환경에 적용할 때마다 재학습이 필요하며, 계산 비용이 매우 높다는 한계가 있다.

따라서 본 연구는 복잡한 데이터 분포를 학습하고 새로운 샘플을 생성할 수 있는 생성 모델의 능력을 의사 결정 시스템에 도입하여, 에이전트가 고보상 영역(High-reward regions)이나 중간 하위 목표(Intermediate sub-goals)를 향하는 궤적(Trajectory)을 생성하게 함으로써 기존 방식의 한계를 극복하고자 한다. 구체적으로는 단순한 행동 모방을 넘어 환경과의 상호작용을 통한 정책 학습, 정책의 생성, 다양한 환경에 대한 적응성, 그리고 장기적인 추론 능력을 확보하는 것이 핵심 문제이다.

## ✨ Key Contributions

본 논문의 주요 기여는 생성 모델을 이용한 의사 결정 방법론에 대한 포괄적인 분류 체계(Taxonomy)를 제안하고 분석한 것에 있다.

1. **종합적인 분류 체계 제안**: 생성 모델을 7가지 모델 패밀리(EBM, GAN, VAE, NF, DM, GFN, AM)로 구분하고, 의사 결정 과정에서의 역할을 세 가지(Controller, Modeler, Optimizer)로 정의하여 체계화하였다.
2. **다양한 실전 응용 사례 분석**: 로봇 제어(Robot Control), 구조 생성(Structural Generation), 게임(Games), 자율 주행(Autonomous Driving), 최적화(Optimization) 등 5가지 핵심 도메인에서 생성 모델이 어떻게 적용되는지 상세히 검토하였다.
3. **미래 연구 방향 제시**: 기존 접근 방식의 강점과 한계를 분석하여, 고성능 알고리즘, 대규모 일반화 의사 결정 모델, 자가 진화 및 적응형 모델이라는 세 가지 차세대 발전 방향을 제시하였다.

## 📎 Related Works

논문은 순차적 의사 결정의 기본 틀인 마르코프 결정 과정(Markov Decision Process, MDP)과 부분 관측 마르코프 결정 과정(Partially Observed MDP, POMDP)을 기초로 하며, 다음과 같은 기존 접근 방식들을 소개한다.

- **동적 계획법(Dynamic Programming)**: 벨만 방정식(Bellman Equation)을 통해 최적 정책을 찾으나, 환경 모델이 정확히 알려져 있어야 한다는 전제가 필요하다.
- **강화 학습(Reinforcement Learning)**:
  - **가치 기반 방법(Value-based)**: $Q$-함수 등을 추정하여 최적 행동을 결정한다.
  - **정책 경사 방법(Policy Gradients)**: 기대 보상을 직접 최적화하는 방향으로 정책 $\pi_\theta$를 업데이트한다.
  - **Actor-Critic**: 가치 함수와 정책 함수를 결합하여 학습의 안정성을 높인다.
  - **모델 기반 RL(Model-based RL)**: 환경의 전이 확률 $P(s'|s, a)$를 학습하여 계획(Planning)에 활용한다.

**기존 방식과의 차별점**: 전통적인 RL이 환경과의 상호작용을 통한 '최적 행동의 학습'에 집중한다면, 생성 모델 기반 접근 방식은 데이터의 '기저 분포 학습 및 샘플링'에 집중한다. 즉, 보상을 최대화하는 단일 경로를 찾는 것이 아니라, 고성능 궤적들의 분포를 학습하여 더 다양하고 효율적인 탐색과 생성이 가능하다는 점이 핵심적인 차이이다.

## 🛠️ Methodology

본 논문은 생성 모델을 의사 결정 시스템 내에서 수행하는 **역할(Function)**과 사용된 **모델 패밀리(Family)**라는 두 가지 축으로 방법론을 설명한다.

### 1. 생성 모델의 7가지 패밀리와 수식적 기초

- **Energy-Based Models (EBM)**: 데이터 $x$에 에너지 함수 $E_\theta(x)$를 할당하며, 확률 밀도는 다음과 같이 정의된다.
    $$p_\theta(x) = \frac{1}{Z_\theta} \exp(-E_\theta(x))$$
    여기서 $Z_\theta$는 정규화 상수(Partition function)이다. 에너지가 낮을수록 해당 데이터의 발생 확률이 높다고 판단한다.
- **Generative Adversarial Networks (GAN)**: 생성자($G$)와 판별자($D$)의 적대적 학습을 통해 실제 데이터 분포를 모방한다.
    $$\min_\theta \max_\phi \mathbb{E}_{x \sim p_{data}}[\log D(x;\phi)] + \mathbb{E}_{\hat{x} \sim G(\theta)}[\log(1-D(\hat{x};\phi))]$$
- **Variational Autoencoders (VAE)**: 잠재 공간 $z$를 통해 데이터를 생성하며, ELBO(Evidence Lower Bound)를 최대화하는 방향으로 학습한다.
    $$\mathcal{L}(\phi, \theta) = \mathbb{E}_{z \sim p(z|x;\phi)}[\log p(x|z;\theta)] - KL(q(z) \| p(z))$$
- **Normalizing Flows (NF)**: 가역 함수(Invertible transformation)의 연속적인 적용을 통해 단순 분포를 복잡한 데이터 분포로 변환한다.
    $$x = f_K(f_{K-1}(\dots f_0(z_0) \dots)) \text{ where } z_0 \sim \mathcal{N}(0, I)$$
- **Diffusion Models (DM)**: 데이터에 점진적으로 노이즈를 추가하는 순방향 과정(Forward process)과 이를 다시 복원하는 역방향 과정(Reverse process)을 학습한다.
  - 순방향: $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$
  - 역방향: $p_\theta(x_{t-1} | x_t)$를 학습하여 노이즈로부터 데이터를 생성한다.
- **GFlowNets (GFN)**: 보상 $R(s \to s_f)$에 비례하는 확률로 상태를 샘플링하도록 흐름(Flow)을 학습한다. 특히 다중 경로가 존재하는 구조에서 다양한 고보상 샘플을 효율적으로 생성하는 데 강점이 있다.
- **Autoregressive Models (AM)**: 이전 요소들에 조건부로 다음 요소를 순차적으로 예측한다.
    $$p(y_1, \dots, y_T) = \prod_{t=1}^T p(y_t | y_1, \dots, y_{t-1})$$

### 2. 의사 결정에서의 세 가지 역할 (Functions)

- **Controller (제어기)**: 생성 모델이 직접 정책($\pi$)의 역할을 수행한다. 고보상 궤적이나 상태-행동 쌍의 분포를 학습하여 최적의 행동을 생성한다. (예: Decision Transformer, Diffusion Policy)
- **Modeler (모델러)**: 데이터의 기저 패턴을 학습하여 새로운 데이터를 생성한다. 주로 데이터 증강(Data Augmentation), 시뮬레이션, 프라이버시 보호 등을 통해 후속 최적화 과정을 돕는다. (예: S2P, MTDiff)
- **Optimizer (최적화 도구)**: 고차원 공간에서 솔루션을 샘플링하고 반복적으로 정제하여 목적 함수를 최적화한다. 전통적인 경사 하강법이 어려운 비볼록(Non-convex) 공간에서 효율적인 탐색을 가능하게 한다. (예: DDOM, MOGFNs)

## 📊 Results

본 논문은 서베이 논문으로서 개별 실험 결과보다는 광범위한 문헌 분석을 통한 정성적/정량적 경향성을 제시한다.

- **모델별 특성 비교**:
  - **Diffusion Models & Normalizing Flows**: 샘플 다양성과 안정성이 매우 높지만, 계산 비용이 커서 실시간 의사 결정에는 제약이 있다.
  - **VAEs & GANs**: 학습 및 생성 속도가 빠르고 효율적이지만, 샘플 다양성이 부족하여 모드 붕괴(Mode collapse)나 과적합 문제가 발생할 수 있다.
- **응용 분야별 성과**:
  - **로봇 제어**: Diffusion Policy 등이 시각-운동(Visuomotor) 정책 학습에서 기존 RL보다 높은 정밀도를 보였다.
  - **자율 주행**: TrafficGen과 같은 모델이 엣지 케이스(Edge cases) 시나리오를 생성하여 데이터 부족 문제를 해결하고 있다.
  - **최적화**: GFlowNets가 조합 최적화(Combinatorial Optimization) 문제에서 기존 방식보다 더 다양한 파레토 최적해(Pareto-optimal solutions)를 찾는 성과를 보였다.
- **연구 트렌드**: 2000년대 이후 완만하게 성장하다가, 최근 Diffusion Model과 Transformer 기반의 AM이 등장하면서 의사 결정 분야로의 적용 사례가 폭발적으로 증가하고 있음을 Google Scholar 데이터 기반의 바 차트로 증명하였다.

## 🧠 Insights & Discussion

### 강점 및 기회

생성 모델은 특히 **고차원 상태 공간**과 **희소한 보상(Sparse reward)** 환경에서 전통적인 RL보다 강력한 탐색 능력을 보여준다. 특히 Diffusion 모델의 반복적 정제 과정은 복잡한 제약 조건을 만족하는 궤적을 생성하는 데 유리하며, GFlowNets는 보상 분포에 따른 직접적인 샘플링을 통해 최적해의 다양성을 보장한다.

### 한계 및 미해결 과제

1. **추론 속도**: Diffusion 모델의 반복적인 디노이징 단계는 실시간 제어가 필수적인 로봇이나 자율 주행 시스템에서 치명적인 지연 시간을 초래한다. (Consistency Models 등이 대안으로 제시됨)
2. **일반화 능력**: 현재 대부분의 모델이 특정 작업(Task-specific)에 최적화되어 있어, 완전히 새로운 환경에 적용하기 위해서는 여전히 상당한 튜닝이 필요하다.
3. **데이터 의존성**: 생성 모델의 성능은 학습 데이터의 품질과 양에 크게 의존하며, 특히 오프라인 RL 설정에서 데이터 분포 외(Out-of-distribution) 샘플 생성 시의 안전성 문제가 남아 있다.

### 비판적 해석

본 논문은 매우 방대한 분류 체계를 제시하고 있으나, 각 모델 패밀리가 실제 성능 지표(예: 성공률, 보상 값) 면에서 어떤 정량적 우위를 가지는지에 대한 직접적인 벤치마크 비교 데이터는 부족하다. 하지만 이는 각 연구의 환경이 너무나 다양하기 때문으로 보이며, 오히려 이러한 다양성을 '역할(Controller, Modeler, Optimizer)'이라는 관점에서 통합한 점이 학술적으로 매우 가치 있다.

## 📌 TL;DR

본 논문은 생성 모델을 의사 결정에 적용하는 방법론을 **7가지 모델 패밀리**와 **3가지 역할(Controller, Modeler, Optimizer)**로 체계화한 종합 서베이 보고서이다. 전통적인 RL의 시행착오 방식에서 벗어나 데이터 분포 학습을 통한 효율적인 정책 생성과 최적화를 가능하게 함을 역설하며, 향후 **대규모 일반화 모델**과 **자가 진화형 적응 모델**로의 발전 방향을 제시한다. 이 연구는 복잡한 환경에서의 로봇 제어, 자율 주행, 조합 최적화 등 다양한 실무 분야에 생성형 AI를 접목하려는 연구자들에게 필수적인 로드맵을 제공한다.
