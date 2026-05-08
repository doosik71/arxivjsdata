# Bootstrapping Reinforcement Learning with Imitation for Vision-Based Agile Flight

Jiaxu Xing, Angel Romero, Leonard Bauersfeld, Davide Scaramuzza (2024)

## 🧩 Problem to Solve

본 논문은 시각 정보만을 이용한 쿼드로터(Quadrotor)의 고속 비행을 위한 visuomotor policy 학습 문제를 다룬다. 특히 자율 드론 레이싱 환경에서 다음과 같은 핵심적인 난제들을 해결하고자 한다.

첫째, 고차원 시각 입력으로 인한 Reinforcement Learning(RL)의 낮은 샘플 효율성(Sample Efficiency)이다. RL은 시행착오를 통해 최적의 정책을 찾지만, 시각 데이터의 차원이 매우 높기 때문에 효율적인 탐색(Exploration)이 어렵고 막대한 계산 자원이 소모된다.

둘째, Imitation Learning(IL)의 성능 한계와 Covariate Shift 문제이다. IL은 전문가의 시연을 통해 빠르게 학습할 수 있으나, 학습된 정책의 성능이 전문가의 수준을 초과할 수 없으며, 훈련 데이터와 실제 환경 간의 분포 차이로 인해 발생하는 Covariate Shift로 인해 강건성(Robustness)이 떨어진다는 단점이 있다.

결과적으로 본 연구의 목표는 RL의 고성능 달성 능력과 IL의 높은 샘플 효율성을 결합하여, 명시적인 상태 추정(State Estimation)이나 IMU 데이터 없이 오직 RGB 이미지 또는 게이트 코너 정보만을 이용하여 고속 비행을 수행하는 정책을 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'상태 기반 RL $\rightarrow$ 시각 기반 IL $\rightarrow$ 적응형 RL 미세 조정'**으로 이어지는 3단계 부트스트래핑(Bootstrapping) 프레임워크를 제안하는 것이다.

가장 중심적인 설계 아이디어는 IL을 통해 학습된 초기 정책을 시작점으로 삼아 RL로 미세 조정하되, 학습 과정에서 발생하는 **'안정성-가소성 딜레마(Stability-Plasticity Dilemma)'** 즉, 새로운 탐색 과정에서 기존에 학습된 유용한 행동을 잊어버리는 치명적 망각(Catastrophic Forgetting) 문제를 해결하기 위해 **성능 기반 적응형 학습률(Performance-Adaptive Learning Rate)** 기법을 도입한 것이다. 이를 통해 정책의 현재 성능에 따라 학습률과 PPO(Proximal Policy Optimization)의 클립 범위($\epsilon$)를 동적으로 조절함으로써 안정적인 성능 향상을 도모한다.

## 📎 Related Works

기존의 시각 기반 로봇 학습은 크게 RL과 IL 두 방향으로 진행되어 왔다. RL은 보상 함수를 통해 강건하고 일반화된 성능을 얻을 수 있으나, 모바일 로봇 환경에서는 샘플 효율성 문제로 인해 주로 시뮬레이션이나 고정 기반 조작 작업에 국한되는 경향이 있었다. 반면 IL은 행동 복제(Behavior Cloning) 등을 통해 적은 데이터로도 학습이 가능하여 실제 로봇에 적용된 사례가 많지만, 앞서 언급한 Covariate Shift와 전문가 성능의 한계라는 명확한 제약이 존재한다.

최근에는 BC(Behavior Cloning)로 초기화한 후 RL로 미세 조정하는 방식이 제안되었으나, 온라인 미세 조정 과정에서 탐색으로 인한 데이터 분포 변화가 기존 지식을 파괴하는 문제가 지속적으로 제기되었다. 본 논문은 이러한 기존의 단순 미세 조정 방식과 달리, 정책의 수행 성능과 네트워크 업데이트 속도를 연동시키는 적응형 메커니즘을 통해 차별점을 둔다.

## 🛠️ Methodology

본 논문이 제안하는 전체 파이프라인은 다음의 세 단계로 구성된다.

### Phase I: State-based Teacher Policy Training

먼저, 특권 정보(Privileged state information)를 모두 사용할 수 있는 Teacher 정책 $\pi_{\text{teacher}}$를 RL(PPO)로 학습시킨다. 여기서 상태 $s$는 위치 $p$, 회전 $\tilde{R}$, 선속도 $v$, 각속도 $\omega$, 그리고 다음 게이트까지의 거리 $d$ 등을 포함한다. 보상 함수 $r_t$는 다음과 같이 정의된다.
$$r_t = r_{\text{prog}}^t + r_{\text{perc}}^t + r_{\text{act}}^t + r_{\text{br}}^t + r_{\text{pass}}^t + r_{\text{crash}}^t$$
각 항은 게이트로의 전진 정도($r_{\text{prog}}$), 카메라의 시선이 게이트 중심을 향하는지($r_{\text{perc}}$), 제어 입력의 급격한 변화 방지($r_{\text{act}}$), 모션 블러 감소를 위한 각속도 제한($r_{\text{br}}$), 게이트 통과 성공($r_{\text{pass}}$), 충돌 패널티($r_{\text{crash}}$)를 의미한다.

### Phase II: Imitation Learning using Visual Input

Teacher 정책의 지식을 시각 입력 기반의 Student 정책 $\pi_{\text{student}}$로 전이(Distillation)한다. Student 정책은 $H$ 길이의 시각 관측치 시퀀스 $[o_{t-H+1}, \dots, o_t]$를 입력으로 받으며, Temporal Convolutional Network(TCN)를 통해 시간적 특징을 추출하고 MLP를 통해 제어 명령을 출력한다. 학습에는 DAgger 알고리즘을 사용하여 Covariate Shift를 완화하며, 손실 함수는 Teacher와 Student의 출력값 간의 평균 제곱 오차(MSE)를 사용한다.
$$\mathcal{L}^A(D, \theta_{\text{student}}) = \mathbb{E}_D [\|\pi_{\text{student}}([o_{t-N+1}, \dots, o_t]; \theta_{\text{student}}) - \pi_{\text{teacher}}(s_t)\|]$$

### Phase III: Performance-Adaptive Online Fine-Tuning

IL로 학습된 정책을 시작점으로 하여 RL을 통해 성능을 극대화한다. 이때 치명적 망각을 방지하기 위해 다음과 같은 적응형 업데이트 방식을 채택한다.

1. **Warm-up**: Actor를 고정하고 Critic 네트워크를 먼저 학습시켜 가치 함수를 안정화한다.
2. **Adaptive Update**: 현재 롤아웃 보상 $r_{\text{rollout}}$과 초기 보상 $r_{\text{init}}$의 비율 $\alpha = r_{\text{rollout}} / r_{\text{init}}$를 계산한다.
3. **Parameter Adjustment**: $\alpha$ 값에 따라 정책 학습률 $\text{LR}_\pi$, 가치 함수 학습률 $\text{LR}_V$, 그리고 PPO의 클립 범위 $\epsilon$을 동적으로 업데이트한다. 성능 개선이 적을 때는 Critic의 학습률을 높이고, 성능이 일정 수준 이상 올라오면 Actor의 학습률과 $\epsilon$을 높여 더 공격적인 탐색을 허용한다.

또한, 학습 효율을 높이기 위해 **Asymmetric Actor-Critic** 구조를 사용한다. Actor는 시각 정보만을 사용하지만, Critic은 특권 상태 정보($s$)를 함께 입력받아 더 정확한 가치 평가를 수행함으로써 Actor의 학습을 가이드한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**: 'SplitS', 'Figure 8', 'Kidney' 세 가지 복잡도의 레이싱 트랙에서 평가하였다.
- **입력 방식**: (1) ResNet50 기반의 RGB 이미지 임베딩, (2) 투영된 게이트 코너의 픽셀 좌표.
- **측정 지표**: 성공률(Success Rate, SR), 평균 게이트 통과 오차(Mean-Gate-Passing-Error, MGE), 랩 타임(Lap Time, LT).

### 주요 결과

1. **정량적 성능**: 모든 트랙과 입력 방식에서 제안 방법이 BC, DAgger, RL-from-scratch, PIRLNav 등 모든 베이스라인보다 우수한 성능을 보였다. 특히 RL-from-scratch의 경우 고차원 입력으로 인해 성공률이 0%에 머물렀다.
2. **적응형 미세 조정의 효과**: 단순한 Vanilla Bootstrap(적응형 메커니즘 없는 미세 조정)의 경우 초기에는 성능이 오르다 시간이 흐를수록 다시 하락하는 모습을 보였으나, 제안된 적응형 방법은 지속적으로 성능이 향상되었다.
3. **강건성 테스트**: 프레임 블랙아웃(센서 실패 시뮬레이션) 및 갑작스러운 위치 점프(강풍 시뮬레이션) 상황에서도 DAgger 대비 월등히 높은 성공률과 낮은 오차를 기록하였다.
4. **실제 환경 검증**: HIL(Hardware-in-the-Loop) 시뮬레이션 및 실제 비행 실험 결과, DAgger보다 더 타이트한 궤적과 빠른 랩 타임을 달성하였으며, 특히 급격한 회전 구간에서 최적화된 기동을 보여주었다.
5. **한계 돌파**: 최대 추력 제한을 해제한 실험에서 랩 타임 5.54초를 기록하며, 인간 세계 챔피언 수준의 성능에 도달할 수 있음을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 RL의 탐색 효율성 문제를 IL로 해결하고, 다시 IL의 성능 한계를 적응형 RL 미세 조정으로 극복하는 유기적인 파이프라인을 구축했다는 점이다. 특히 Asymmetric Critic을 통해 부분 관측성(Partial Observability) 문제를 완화하고, 성능 기반 학습률 조절을 통해 딥러닝의 고질적인 문제인 치명적 망각을 효과적으로 억제하였다.

다만, 본 연구는 통제된 실험실 환경 및 디지털 트윈 기반의 시뮬레이션에서 검증되었다는 한계가 있다. 실제 야외(in-the-wild) 환경에서는 조명 변화, 복잡한 배경 등 더 많은 Out-of-Distribution(OOD) 케이스가 발생하므로, 현재의 ResNet 기반 인코더보다는 더 특화된 비전 인코더의 도입이 필요할 것으로 보인다.

비판적으로 해석하자면, 본 방법론은 특권 정보(Privileged state)를 가진 Teacher 정책의 존재를 전제로 한다. 만약 Teacher를 학습시키기 어려운 환경이라면 본 프레임워크의 적용이 제한될 수 있다. 하지만 드론 레이싱과 같이 물리 엔진이 잘 구축된 환경에서는 매우 강력한 도구가 될 수 있다.

## 📌 TL;DR

본 논문은 시각 정보만을 이용한 고속 드론 비행을 위해 **'상태 기반 RL $\rightarrow$ 시각 기반 IL $\rightarrow$ 적응형 RL 미세 조정'**의 3단계 학습 프레임워크를 제안하였다. 이를 통해 RL의 샘플 효율성 문제와 IL의 성능 한계를 동시에 해결하였으며, 특히 성능-연동형 적응형 학습률 조절을 통해 치명적 망각 없이 정책을 최적화하였다. 실험 결과, 시뮬레이션과 실제 환경 모두에서 기존 IL/RL 방법론을 압도하였으며, 최종적으로는 인간 챔피언 수준의 비행 성능을 달성하였다. 이 연구는 고차원 시각 입력을 사용하는 다른 모바일 로봇의 제어 정책 학습에도 널리 적용될 가능성이 높다.
