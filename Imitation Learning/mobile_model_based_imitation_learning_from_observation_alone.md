# MobILE: Model-Based Imitation Learning From Observation Alone

Rahul Kidambi, Jonathan D. Chang, Wen Sun (2021)

## 🧩 Problem to Solve

본 논문은 전문가의 행동(action) 정보 없이 방문한 상태(state) 시퀀스만 제공되는 **Imitation Learning from Observation alone (ILFO)** 문제를 해결하고자 한다.

일반적인 모방 학습(Imitation Learning, IL)은 전문가의 상태-행동 쌍$(s, a)$을 제공받아 지도 학습(Supervised Learning) 방식으로 행동을 복제하는 Behavior Cloning(BC)이나 DAgger와 같은 기법을 사용할 수 있다. 그러나 실제 환경에서는 전문가와 학습자의 행동 공간(action space)이 다르거나, Sim-to-Real 전이 과정, 혹은 단순한 비디오 데이터만을 통해 학습해야 하는 경우가 많아 전문가의 행동 데이터에 접근할 수 없는 경우가 빈번하다.

따라서 본 연구의 목표는 전문가의 상태 관측치만을 이용하여, 전문가 수준의 성능을 내는 정책 $\pi$를 효율적으로 학습할 수 있는 모델 기반 프레임워크인 **MobILE**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **전략적 탐색(Strategic Exploration)과 모방(Imitation) 사이의 정교한 트레이드-오프**를 구현하는 것이다. 이를 위해 강화학습의 '불확실성 앞에서의 낙관주의(Optimism in the face of uncertainty)' 개념을 분포 매칭(Distribution Matching) 기반의 모방 학습 프레임워크에 통합하였다.

주요 기여 사항은 다음과 같다:

1. **MobILE 프레임워크 제안**: 모델 기반 학습, 탐색을 위한 낙관주의, 그리고 적대적 모방 학습을 결합하여, 특정 구조적 복잡성을 가진 MDP dynamics에 대해 이론적으로 효율적인 성능 보장(Regret bound)을 제공한다.
2. **IL과 ILFO의 이론적 차이 규명**: 전문가의 행동 정보가 있을 때(IL)와 없을 때(ILFO)의 샘플 복잡도(Sample Complexity) 사이에 지수적 차이(Exponential gap)가 존재함을 증명하여, ILFO가 근본적으로 더 어려운 문제임을 보였다.
3. **실전적 구현 및 검증**: 신경망 앙상블(Neural Network Ensemble)과 불일치 기반 보너스(Disagreement-based bonus)를 사용하여 MobILE을 구현하였으며, OpenAI Gym 벤치마크에서 기존 ILFO 알고리즘보다 우수한 성능을 입증하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

1. **Model-free ILFO (예: FAIL)**: ILFO 문제를 일련의 1단계 분포 매칭 문제로 환원하여 해결한다. 그러나 각 타임스텝별로 문제를 풀기 때문에 데이터를 효율적으로 재사용하지 못하며, 긴 호라이즌(Long horizon) 작업에서 샘플 효율성이 매우 떨어진다.
2. **Inverse/Forward Dynamics 기반 방식**: 전문가의 행동을 추론하기 위해 역동역학(Inverse dynamics) 모델을 학습하여 행동을 복구한 뒤 BC를 적용한다. 하지만 이는 MDP의 전이 동역학이 단사 함수(Injective)여야 한다는 강한 가정이 필요하며, 그렇지 않은 경우 모델이 정의되지 않는 문제가 발생한다.
3. **Hand-crafted Cost Functions**: 도메인 지식을 이용해 비용 함수를 직접 설계하는 방식이나, 이는 엔지니어링 비용이 높고 일반화 능력이 떨어진다.

### MobILE의 차별점

MobILE은 항상 유일하게 정의되는 **순방향 전이 동역학(Forward transition dynamics)** 모델을 학습하며, 모델의 불확실성을 이용한 낙관적 보너스를 통해 탐색과 모방을 자동으로 조절한다. 이는 전문가의 행동 정보가 없더라도 체계적인 탐색을 통해 환경을 이해하고 전문가의 상태 분포를 효과적으로 추종하게 한다.

## 🛠️ Methodology

### 전체 파이프라인

MobILE은 다음의 세 단계를 반복하는 루프 구조를 가진다:

1. **Dynamics Model Learning**: 온라인 상호작용을 통해 수집한 $(s, a, s')$ 데이터를 버퍼에 저장하고, 전이 모델 $\hat{P}$를 학습한다.
2. **Bonus Design**: 학습된 모델 $\hat{P}$의 불확실성이 높은 영역(방문 횟수가 적은 곳)에서는 큰 보너스 $b(s, a)$를 부여하고, 확실한 영역에서는 작은 보너스를 부여한다.
3. **Optimistic Model-based Min-Max IL**: 학습된 모델 $\hat{P}$와 보너스 $b$를 사용하여 전문가의 상태 분포와 학습자의 상태 분포를 일치시키는 정책 $\pi$를 최적화한다.

### 주요 구성 요소 및 방정식

#### 1. 낙관적 목적 함수 (Optimistic IPM Objective)

MobILE은 Integral Probability Metric (IPM)을 사용하여 상태 분포를 매칭하며, 이때 보너스 항을 추가하여 다음과 같은 목적 함수를 푼다:
$$\pi_{t+1} \leftarrow \arg \min_{\pi \in \Pi} \max_{f \in \mathcal{F}} L(\pi, f; \hat{P}, b, D_e) := \mathbb{E}_{(s, a) \sim d^\pi_{\hat{P}}} [f(s) - b(s, a)] - \mathbb{E}_{s \sim D_e} [f(s)] \quad (1)$$
여기서 $d^\pi_{\hat{P}}$는 모델 $\hat{P}$ 하에서의 상태-행동 분포, $D_e$는 전문가 데이터셋, $f$는 판별자(Discriminator)이다.

**핵심 직관**: 보너스 $b(s, a)$는 모델이 불확실한 영역에서 판별자 $f$의 페널티를 상쇄시킨다. 즉, 모델이 정확하지 않은 곳에서는 상태 매칭 제약을 완화하여 학습자가 자유롭게 탐색하도록 유도하고, 모델이 정확한 곳에서는 전문가의 상태를 엄격하게 따라가도록 강제한다.

#### 2. 불확실성 측정 및 보너스 (Bonus)

실제 구현에서는 신경망 앙상블을 사용하여 모델 간의 예측 불일치(Disagreement)를 측정한다. 두 모델 $h_{\theta_1}, h_{\theta_2}$의 예측값 차이를 보너스로 설정한다:
$$\delta(s, a) = \|h_{\theta_1}(s, a) - h_{\theta_2}(s, a)\|^2, \quad b(s, a) = \lambda \cdot \min\left(\frac{\delta(s, a)}{\delta^D}, 1\right)$$
여기서 $\delta^D$는 버퍼 내 최대 불일치 값으로 정규화에 사용된다.

#### 3. 학습 절차

- **판별자 업데이트**: 현재 정책 $\pi$에 대해 목적 함수 (1)을 최대화하는 $f$를 찾는다. 구현에서는 RBF 커널을 이용한 Maximum Mean Discrepancy (MMD)를 사용하며, 이는 닫힌 형태(Closed-form)의 해를 가진다.
- **정책 업데이트**: 고정된 $f$와 $b$에 대해, 모델 $\hat{P}$ 상에서 $f(s) - b(s, a)$를 비용 함수로 하여 TRPO(Trust Region Policy Optimization)와 같은 Policy Gradient 방법을 통해 $\pi$를 업데이트한다.

## 📊 Results

### 실험 설정

- **데이터셋**: OpenAI Gym의 MuJoCo 환경 (Cartpole-v1, Reacher-v2, Swimmer-v2, Hopper-v2, Walker2d-v2).
- **비교 대상**: BC, GAIL (전문가 행동 정보 사용-상한선), BC-O, ILPO, GAIFO (ILFO 방식).
- **평가 지표**: 전문가 성능 대비 정규화된 점수(Normalized Score) 및 학습 곡선.

### 주요 결과

1. **성능 우위**: MobILE은 전문가의 행동 정보 없이도, 행동 정보를 사용하는 BC나 GAIL의 성능에 근접하거나 때로는 능가하는 결과를 보였다. 특히 기존 ILFO 방식인 BC-O, ILPO, GAIFO보다 월등히 높은 성능을 보였다.
2. **낙관주의의 중요성**: 보너스 $b(s, a)$를 제거한 경우(No optimism), 학습 속도가 현저히 느려지거나 아예 전문가 수준에 도달하지 못했다. 이는 ILFO 문제에서 체계적인 탐색이 필수적임을 시사한다.
3. **샘플 효율성**: 모델 기반 방식의 이점으로 인해, GAIL이나 GAIFO 같은 Model-free 방식보다 훨씬 적은 온라인 샘플만으로도 빠르게 수렴하였다.
4. **전문가 샘플 수의 영향**: 전문가 궤적(Trajectory) 수가 증가할수록 정책의 분산이 줄어들고 성능이 안정적으로 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 해석

본 논문은 단순히 알고리즘을 제안한 것에 그치지 않고, $\text{Regret} \le O(H^{2.5}\sqrt{I_T/\sqrt{T}} + \dots)$ 형태의 이론적 상한선을 제시하였다. 여기서 $I_T$는 Information Gain으로, 모델이 환경에 대해 얼마나 많은 정보를 얻었는지를 나타낸다. 이는 모델 기반 학습과 낙관적 탐색이 결합되었을 때 이론적으로 최적의 정책에 수렴할 수 있음을 수학적으로 뒷받침한다.

또한, IL과 ILFO 사이의 지수적 샘플 복잡도 차이를 증명함으로써, 행동 정보가 없는 상황에서는 단순한 복제가 불가능하며 반드시 '전략적 탐색'이 수반되어야 함을 명시적으로 드러냈다.

### 한계 및 비판적 해석

- **모델 의존성**: 모든 성능이 학습된 전이 모델 $\hat{P}$의 정확도에 의존한다. 만약 모델 클래스가 실제 dynamics를 표현하기에 너무 단순하거나(Misspecification), 학습이 불안정할 경우 보너스 기반 탐색이 오히려 잘못된 방향으로 유도될 위험이 있다.
- **하이퍼파라미터 민감도**: 보너스의 스케일을 결정하는 $\lambda$ 값에 따라 탐색과 모방의 균형이 달라지므로, 환경에 맞는 정교한 튜닝이 필요하다.
- **관측 공간의 확장성**: 본 논문은 저차원의 상태 공간을 가정한다. 고차원 이미지(Video) 데이터로 확장할 경우, Forward dynamics 모델을 정확하게 학습하는 것이 매우 어려워지므로 이에 대한 추가적인 연구가 필요하다.

## 📌 TL;DR

**MobILE**은 전문가의 행동 정보 없이 상태 관측치만으로 학습하는 **ILFO** 문제를 해결하기 위해, **전이 모델 학습(Model-based)**과 **낙관적 탐색 보너스(Optimism)**를 결합한 프레임워크이다. 모델의 불확실성이 높은 곳에서는 탐색을 장려하고, 확실한 곳에서는 전문가를 모방하는 전략을 통해 ILFO의 근본적인 어려움(샘플 복잡도 증가)을 극복하였다. 이 연구는 향후 행동 라벨이 없는 대규모 비디오 데이터로부터 로봇 제어 정책을 학습하는 연구에 중요한 이론적/실천적 토대를 제공한다.
