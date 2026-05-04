# Invariant Causal Imitation Learning for Generalizable Policies

Ioana Bica, Daniel Jarrett, Mihaela van der Schaar (2021)

## 🧩 Problem to Solve

본 논문은 여러 환경에서 수집된 전문가의 시연 데이터(Demonstrations)를 통해 모방 정책(Imitation Policy)을 학습하고, 이를 학습 시 보지 못한 새로운 환경(Unseen Environment)에 배포했을 때도 성능이 유지되는 일반화(Generalization) 문제를 해결하고자 한다.

특히, 본 연구는 상호작용이 불가능하고 오직 기록된 데이터만 사용할 수 있는 **Strictly Batch Imitation Learning** 설정에 집중한다. 이 설정에서 발생하는 핵심적인 문제는 각 환경의 관측치(Observations)에 전문가의 행동과 상관관계가 있지만 인과 관계는 없는 **가짜 상관관계(Spurious Correlations)**가 포함되어 있다는 점이다. 만약 정책이 이러한 가짜 상관관계에 의존하여 학습된다면, 환경이 바뀌어 상관관계의 양상이 달라질 때 정책의 성능이 급격히 저하되는 일반화 실패 문제가 발생한다. 

따라서 본 논문의 목표는 환경 간에 변하지 않는 **불변의 인과적 구조(Invariant Causal Structure)**를 추출하여, 환경 변화에 강건하고 일반화 능력이 뛰어난 모방 정책을 학습하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 관측치를 **환경 간에 공유되는 불변의 인과적 상태 표현(Invariant Causal Representation, $s$)**과 **환경마다 서로 다른 노이즈 표현(Environment-specific Noise Representation, $\eta^e$)**으로 분리하여 학습하는 것이다.

전문가의 정책은 오직 불변의 인과적 상태 $s$에만 의존한다고 가정하며, 이를 통해 가짜 상관관계(노이즈)를 제거하고 핵심적인 인과 관계만을 학습하여 새로운 환경에서도 동작하는 정책을 구축한다. 또한, 배치 설정에서의 고질적인 문제인 분포 편향(Distribution Shift)을 해결하기 위해, 에너지 기반 모델(Energy-Based Model, EBM)을 도입하여 모방 정책이 생성하는 다음 상태가 전문가의 상태 분포 범위 내에 있도록 규제하는 메커니즘을 제안한다.

## 📎 Related Works

### 기존 연구 및 한계
1. **Strictly Batch Imitation Learning**: Behaviour Cloning(BC)은 단순하지만, 전문가의 데이터 분포를 벗어난 상태에 진입했을 때 오류가 누적되는 compounding error 문제에 취약하다. 이를 해결하기 위해 VDICE나 EDM 같은 분포 매칭 방식이 제안되었으나, 이들은 단일 환경에서의 최적화에 집중했을 뿐 환경 간 일반화 문제는 다루지 않았다.
2. **Invariant Risk Minimization (IRM)**: 여러 도메인에서 불변의 예측자를 찾는 기법이다. 하지만 IRM을 그대로 모방 학습에 적용하면 순차적 의사결정 과정에서 액션이 다음 상태에 미치는 영향(Dynamics)을 고려하지 못한다는 한계가 있다.
3. **Generalization in IL**: 도메인 적응(Domain Adaptation) 연구들이 존재하지만, 대개 타겟 환경의 데이터에 접근 가능하거나 시뮬레이터와의 온라인 상호작용이 필요하다는 가정을 전제로 한다.

### 차별점
ICIL은 타겟 환경의 데이터 없이도 학습이 가능하며, 단순한 불변성 추구를 넘어 **인과적 표현 학습(Causal Representation Learning)**과 **다이내믹스 보존(Dynamics Preserving)**, 그리고 **에너지 기반의 상태 분포 매칭**을 동시에 수행한다는 점에서 기존 방법론과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
ICIL은 관측치 $x_t^e$를 인과적 표현 $s_t$와 노이즈 표현 $\eta_t^e$로 분해하는 인코더 $\phi$와 $\mu_e$를 학습한다. 전체 파이프라인은 크게 (1) 불변 인과 표현 학습, (2) 전문가 행동 모방 정책 학습의 두 단계로 구성된다.

### 1. 불변 인과 표현 학습 (Learning Invariant Causal Representations)
인과적 표현 $s_t = \phi(x_t^e; \theta_s)$와 노이즈 표현 $\eta_t^e = \mu_e(x_t^e; \theta_\eta^e)$를 학습하기 위해 세 가지 조건을 강제한다.

- **환경 불변성(Invariance)**: 환경 분류기 $c_s$를 도입하여, $s_t$만으로는 어떤 환경인지 구별할 수 없도록 적대적 손실 함수를 사용한다.
  $$L_{inv}(\theta_s) = \sum_{e \in E_{train}} \mathbb{E}_{x_t^e \sim \rho_e^D} [-H(c_s(\phi(x_t^e; \theta_s); \theta_c))]$$
- **다이내믹스 보존(Dynamics Preserving)**: 상태 전이 모델 $g_s$와 노이즈 전이 모델 $g_\eta^e$, 그리고 이를 다시 관측치로 복원하는 디코더 $\psi$를 학습하여 표현이 시스템의 물리적 변화를 보존하게 한다.
  $$L_{dyn} = \sum_{e \in E_{train}} \mathbb{E}_{x_{t+1}^e, a_t, x_t^e \sim \rho_e^D} \|x_{t+1}^e - \psi(g_s(s_t, a_t), g_\eta^e(\eta_t^e, a_t))\|^2$$
- **상호 독립성(Independence)**: $s_t$와 $\eta_t^e$가 서로 독립적이어야 하므로, MINE(Mutual Information Neural Estimation)를 사용하여 두 표현 사이의 상호 정보량(Mutual Information)을 최소화한다.
  $$L_{mi} = \sum_{e \in E_{train}} \mathbb{E}_{x_t^e \sim \rho_e^D} I(\phi(x_t^e), \mu(x_t^e); \theta_m)$$

### 2. 모방 정책 학습 및 분포 매칭 (Matching Expert Behaviour)
학습된 불변 표현 $s_t$를 입력으로 받는 정책 $\pi(a_t | s_t)$를 학습한다.

- **행동 모방**: 전문가의 액션을 최대 우도 추정(MLE) 방식으로 학습한다.
  $$L_\pi(\theta_\pi, \theta_s) = \sum_{e \in E_{train}} -\mathbb{E}_{x_t^e, a_t \sim \rho_e^D} \log \pi(a_t | \phi(x_t^e); \theta_\pi)$$
- **상태 분포 규제**: 모방 정책이 전문가의 상태 분포를 벗어나지 않도록, 전문가의 관측치 분포를 에너지 기반 모델(EBM) $\bar{E}_\theta(x)$로 근사한다. 정책 $\pi$가 선택한 액션 $\bar{a}_t$로 인해 도달하게 될 다음 상태 $\bar{x}_{t+1}$의 에너지를 최소화함으로써 전문가의 고밀도 영역에 머물게 한다.
  $$L_{energy} = \sum_{e \in E_{train}} \mathbb{E}_{x_t^e \sim \rho_e^D, \bar{a}_t \sim \pi(\cdot|s_t)} \bar{E}_\theta(\bar{x}_{t+1})$$
  여기서 $\bar{x}_{t+1} = \psi(g_s(s_t, \bar{a}_t), g_\eta^e(\eta_t^e, \bar{a}_t))$이다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**: 
    - OpenAI Gym (Acrobot, Cartpole, LunarLander, BeamRider): 상태 공간에 가짜 상관관계를 추가하여 여러 환경을 구축하고, 새로운 상관관계 계수를 가진 테스트 환경에서 평가하였다.
    - MIMIC-III (ICU 데이터셋): 인공호흡기(Mechanical Ventilator) 적용 여부를 결정하는 의료 전문가의 정책을 모방하며, 병원별 선택 편향(Selection Bias)을 가짜 상관관계로 모델링하였다.
- **비교 대상**: BC, RCAL, VDICE, EDM 및 이들에 IRM-v1 목적 함수를 추가한 버전(BC-IRM 등).
- **지표**: OpenAI Gym에서는 평균 리턴(Average Return), MIMIC-III에서는 액션 일치도(ACC, AUC, APR)를 측정하였다.

### 주요 결과
- **일반화 성능**: 모든 작업에서 ICIL이 기존 벤치마크보다 월등한 성능을 보였다. 특히 학습 데이터의 양이 적을 때도 불변 표현 학습을 통해 안정적인 일반화 성능을 유지하였다.
- **IRM-v1의 한계**: 기존 배치 모방 학습 방법론에 단순히 IRM-v1 손실 함수를 추가하는 것만으로는 성능 향상이 뚜렷하지 않았으며, 오히려 학습이 불안정해지는 경향이 확인되었다.
- **의료 데이터 적용**: MIMIC-III 데이터셋에서 ICIL은 ACC 0.855, AUC 0.856, APR 0.789를 기록하며, 가짜 상관관계를 배제하고 전문가의 인과적 결정 기준을 가장 잘 포착했음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **인과적 분리의 효과**: Ablation Study 결과, $L_{inv}$(불변성 손실)가 테스트 환경의 성능 향상에 가장 결정적인 역할을 하는 것으로 나타났다. 이는 단순한 데이터 증강보다 인과적 부모(Causal Parents)를 명시적으로 학습하는 것이 일반화에 훨씬 유리함을 시사한다.
- **강건성**: 가짜 상관관계의 개수가 증가하더라도 BC나 EDM에 비해 성능 저하가 훨씬 적었으며, 이는 노이즈 표현 $\eta^e$를 별도로 분리하여 처리한 설계의 유효성을 보여준다.

### 한계 및 비판적 해석
- **데이터 요구사항**: 불변 표현을 학습하기 위해서는 최소 두 개 이상의 서로 다른 환경 데이터가 필요하다. 단일 환경 데이터만 존재할 경우 본 방법론을 적용할 수 없다는 근본적인 제약이 있다.
- **이론적 근거 부족**: 실험적으로는 우수한 성능을 보였으나, 일반화 오차(Generalization Error)에 대한 엄밀한 이론적 상한(Error Bound)이나 수렴성 증명이 부족하다.
- **전문가 편향**: 모방 학습의 특성상, 전문가의 정책 자체가 편향되어 있거나 오류가 있을 경우 ICIL 역시 그 오류를 그대로 학습하여 배포하게 되는 위험이 존재한다.

## 📌 TL;DR

본 논문은 배치 모방 학습에서 환경 간의 가짜 상관관계로 인해 발생하는 일반화 실패 문제를 해결하기 위해 **Invariant Causal Imitation Learning (ICIL)**을 제안한다. ICIL은 관측치를 불변의 인과적 표현과 환경 특이적 노이즈 표현으로 분리 학습하고, 에너지 기반 모델을 통해 상태 분포를 규제함으로써 새로운 환경에서도 강건하게 동작하는 정책을 학습한다. 이 연구는 특히 데이터 수집이 제한적이고 환경 변화가 심한 의료 및 제어 시스템의 모방 학습 분야에 중요한 기여를 할 가능성이 크다.