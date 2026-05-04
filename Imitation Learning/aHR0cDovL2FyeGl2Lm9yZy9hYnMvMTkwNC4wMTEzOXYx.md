# GENERATIVE PREDECESSOR MODELS FOR SAMPLE-EFFICIENT IMITATION LEARNING

Yannick Schroecker, Mel Vecerik & Jonathan Scholz (2019)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning)에서 발생하는 고질적인 문제인 **오차 누적(Compounding Errors)**과 **낮은 샘플 효율성(Sample Efficiency)**을 해결하고자 한다.

전형적인 행동 복제(Behavioral Cloning, BC) 방식은 지도 학습(Supervised Learning)으로 접근하여 상태에서 행동으로의 매핑을 학습한다. 하지만 에이전트가 학습 데이터에 없는 상태에 진입하게 되면, 작은 실수가 누적되어 전문가가 탐색하지 않은 영역으로 벗어나게 되며, 결국 시스템이 완전히 붕괴되는 문제가 발생한다. 이를 해결하기 위해 GAIL과 같은 분포 매칭 방식이 제안되었으나, 적대적 학습(Adversarial Learning)의 특성상 학습이 불안정하고, 특히 환경 상호작용 데이터가 많이 필요하다는 한계가 있다.

따라서 본 연구의 목표는 적은 수의 전문가 시연(Expert Demonstrations)과 효율적인 자기 지도 상호작용(Self-supervised Interactions)만으로도 전문가의 상태-행동 분포를 정확히 따라가며, 경로를 벗어났을 때 다시 복귀할 수 있는 강건한 정책을 학습하는 알고리즘인 **GPRIL(Generative Predecessor Models for Imitation Learning)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'전문가가 보여준 상태에 도달하게 만들 확률이 높은 상태-행동 쌍'을 생성 모델로 학습하여 훈련 데이터를 증강**하는 것이다.

핵심 직관은 에이전트가 전문가의 상태에서 벗어났을 때, 어떻게 하면 다시 그 상태로 돌아올 수 있는지를 학습시키는 것이 corrective behavior(교정 행동)를 익히는 가장 효과적인 방법이라는 점이다. 이를 위해 미래의 특정 상태 $\bar{s}$가 주어졌을 때, 그 상태에 도달하기 전의 상태 $s$와 행동 $a$의 분포인 **Long-term Predecessor Distribution**을 생성 모델을 통해 추론하고, 이를 정책 학습에 활용한다.

## 📎 Related Works

- **Behavioral Cloning (BC):** 전문가 데이터를 지도 학습으로 모방하지만, i.i.d. 가정을 위반하여 오차가 누적되는 문제가 있다.
- **Inverse Reinforcement Learning (IRL):** 전문가의 보상 함수를 추론하여 학습하지만, 동일한 행동을 유발하는 보상 함수가 너무 많아 정의가 불분명(ill-defined)한 경우가 많다.
- **Generative Adversarial Imitation Learning (GAIL):** 상태-행동 분포를 직접 매칭하는 적대적 방식을 사용한다. 성능은 뛰어나지만, 학습 과정이 불안정하며 Mode Collapse와 같은 GAN 특유의 문제에 취약하다.
- **State Aware Imitation Learning (SAIL):** 상태-행동 분포의 그래디언트를 학습하여 매칭하지만, 파라미터 수가 적은 정책에서만 적용 가능하다는 한계가 있다.

GPRIL은 GAIL처럼 분포 매칭을 지향하면서도, 적대적 학습 대신 생성 모델(MAF)을 사용하여 학습 안정성을 높이고 샘플 효율성을 극대화했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
GPRIL의 학습 루프는 다음과 같은 세 단계로 구성된다.
1. **환경 상호작용:** 현재 정책으로 환경과 상호작용하며 $(s, a, s_{future})$ 데이터를 수집하고 리플레이 버퍼에 저장한다.
2. **전구체 모델 학습:** 수집된 데이터를 사용하여, 미래 상태 $\bar{s}$가 주어졌을 때 과거의 $(s, a)$를 생성하는 조건부 생성 모델(Conditional Generative Model)을 학습시킨다.
3. **정책 업데이트:** 전문가 시연 데이터와 더불어, 전구체 모델이 생성한 '전문가 상태로 되돌아오는 경로' 데이터를 사용하여 정책 $\pi_\theta$를 지도 학습 방식으로 업데이트한다.

### 상세 방법론 및 수식 설명

#### 1. 상태 분포의 그래디언트 추정
에이전트의 정상 상태 분포(Stationary State Distribution) $d^\pi_\theta(\bar{s})$의 로그 그래디언트는 다음과 같이 Long-term Predecessor Distribution $B^\pi_\theta$에 대한 기대값으로 근사할 수 있다.

$$\nabla_\theta \log d^\pi_\theta(\bar{s}) \propto \mathbb{E}_{s, a \sim B^\pi_\theta(\cdot, \cdot | \bar{s})} [\nabla_\theta \log \pi_\theta(a|s)]$$

여기서 $B^\pi_\theta(s, a | \bar{s})$는 정책 $\pi_\theta$ 하에서 미래에 $\bar{s}$에 도달하게 될 상태-행동 쌍들의 가중 합으로 정의된다.

$$B^\pi_\theta(s, a | \bar{s}) := (1-\gamma) \sum_{j=0}^{\infty} \gamma^j q^\pi_\theta(s_t=s, a_t=a | s_{t+j+1}=\bar{s})$$

$\gamma$는 할인 인자(Discount factor)로, 미래 상태와 현재 상태 사이의 시간적 거리가 가까울수록 더 큰 가중치를 부여하여 분산을 줄이는 역할을 한다.

#### 2. 생성 모델: Masked Autoregressive Flows (MAF)
본 논문은 전구체 분포 $B^\pi_\theta$를 모델링하기 위해 **Masked Autoregressive Flows (MAF)**를 사용한다. MAF는 복잡한 분포를 가우시안 분포의 가역적 변환(Invertible Transformation)으로 표현하여 정확한 밀도 추정(Density Estimation)이 가능하다.

모델은 다음과 같이 요인화(Factored)된 구조를 가진다.
$$B^\pi_\omega(s, a | \bar{s}) := B^s_{\omega_s}(s | \bar{s}) B^a_{\omega_a}(a | s, \bar{s})$$
즉, 먼저 미래 상태 $\bar{s}$로부터 과거 상태 $s$를 생성하고, 다시 $\bar{s}$와 $s$를 조건으로 행동 $a$를 생성한다.

#### 3. 정책 학습 및 손실 함수
최종적으로 에이전트는 상태-행동 분포 $\rho^\pi_\theta(\bar{s}, \bar{a})$를 전문가의 분포와 매칭시키기 위해 다음의 그래디언트를 따라 업데이트한다.

$$\nabla_\theta \log \rho^\pi_\theta(\bar{s}, \bar{a}) \approx \beta^\pi \nabla_\theta \log \pi_\theta(\bar{a}|\bar{s}) + \beta^d \mathbb{E}_{s, a \sim B^\pi_\omega(\cdot, \cdot | \bar{s})} [\nabla_\theta \log \pi_\theta(a|s)]$$

- $\beta^\pi \nabla_\theta \log \pi_\theta(\bar{a}|\bar{s})$: 전문가의 행동을 그대로 따라하게 하는 지도 학습 성분(Behavioral Cloning)이다.
- $\beta^d \mathbb{E}_{s, a \sim B^\pi_\omega} [\dots]$: 전구체 모델을 통해 생성된 샘플을 이용하여, 전문가 상태 $\bar{s}$로 복귀하는 방법을 학습하게 하는 교정 성분이다.

## 📊 Results

### 실험 설정
- **태스크:** 
    1. **Clip Insertion:** 탄성 클립을 플러그에 삽입하는 작업 (정밀한 조작 필요).
    2. **Peg Insertion:** 시뮬레이션 및 실제 로봇을 이용한 핀 삽입 작업.
- **비교 대상:** Behavioral Cloning (BC), GAIL.
- **측정 지표:** 성공률(Success Rate), 샘플 효율성(환경 상호작용 횟수), 전문가 데이터의 양.

### 주요 결과
1. **성능 및 강건성:** Clip Insertion 태스크에서 GPRIL은 BC보다 훨씬 높고, GAIL과 비슷하거나 더 높은 성공률을 기록했다. 특히 경로를 이탈했을 때 다시 시도하는 복구 능력이 뛰어났다.
2. **샘플 효율성:** GPRIL은 GAIL보다 수 차례의 **Order of Magnitude(수십~수백 배)** 적은 환경 상호작용만으로도 태스크를 해결했다 (그림 1c).
3. **데이터 효율성:** 전문가의 행동(Action) 데이터 없이 **상태(State) 데이터만으로도** 학습이 가능함을 보였다. 이는 Kinesthetic Teaching(직접 로봇을 움직여 가르치는 방식)과 같이 상태만 기록되는 환경에서 매우 유용하다.
4. **실제 로봇 적용:** 시뮬레이션에서 검증된 효율성을 바탕으로 실제 로봇 시스템에 적용하여, 단 몇 시간의 상호작용만으로 가변적인 위치의 소켓에 핀을 삽입하는 정책을 학습시켰다.

## 🧠 Insights & Discussion

### 강점 및 기여
GPRIL은 전구체 모델이라는 개념을 통해 "어떻게 돌아오는가"에 대한 정답지를 생성 모델로 만들어 낸 점이 매우 혁신적이다. 특히 MAF를 사용하여 적대적 학습의 불안정성을 제거하고, 안정적인 Maximum Likelihood 학습을 통해 샘플 효율성을 극대화한 점이 돋보인다.

### 한계 및 논의사항
- **생성 모델의 의존성:** 전구체 모델 $B^\pi_\omega$가 실제 분포를 얼마나 정확하게 근사하느냐가 전체 성능을 결정한다. 만약 생성 모델이 잘못된 경로를 생성한다면 정책 학습에 악영향을 줄 수 있다.
- **$\gamma$의 영향:** 논문의 부록에서 언급되었듯 $\gamma$ 값은 정확한 분포 매칭과 분산 감소 사이의 트레이드-오프를 결정한다. $\gamma$가 낮을수록 더 빠르게 복귀하는 정책을 학습하지만, 장기적인 분포 매칭 정확도는 떨어질 수 있다.

### 비판적 해석
본 논문은 GAIL과의 비교를 통해 우수성을 입증했으나, 최신 Diffusion-based imitation learning 모델들과 비교했을 때 생성 모델의 표현력이 충분한지에 대한 추가 검증이 필요해 보인다. 하지만 2019년 당시 기준으로는 매우 효율적인 분포 매칭 프레임워크를 제시했다고 평가할 수 있다.

## 📌 TL;DR

GPRIL은 전문가의 상태로 되돌아오는 경로를 생성하는 **전구체 모델(Predecessor Model)**을 MAF로 학습시켜, 이를 통해 훈련 데이터를 증강함으로써 오차 누적 문제를 해결하는 모방 학습 알고리즘이다. GAIL 대비 압도적인 **샘플 효율성**과 **학습 안정성**을 보이며, 전문가의 행동 데이터 없이 상태만으로도 학습이 가능하여 **실제 로봇 시스템에 적용 가능함**을 증명했다.