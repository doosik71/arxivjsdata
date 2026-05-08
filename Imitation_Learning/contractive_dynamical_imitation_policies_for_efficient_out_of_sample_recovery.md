# CONTRACTIVE DYNAMICAL LIMITATION POLICIES FOR EFFICIENT OUT-OF-SAMPLE RECOVERY

Amin Abyaneh, Mahrokh G. Boroujeni, Hsiu-Chin Lin, Giancarlo Ferrari-Trecate (2025)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning)에서 발생하는 Out-of-Sample(OOS) 지역에서의 불안정성 문제를 해결하고자 한다. 모방 학습은 전문가의 행동을 데이터 기반으로 학습하지만, 학습 데이터에 포함되지 않은 상태(OOS state)에 진입하거나 외부 섭동(perturbation)이 발생했을 때 로봇의 궤적이 신뢰할 수 없게 되는 문제가 있다.

기존의 안정적 동역학 시스템(Stable Dynamical Systems) 연구들은 시스템이 최종적으로 목표 상태에 수렴한다는 점은 보장하지만, 수렴하기까지의 과정인 과도 응답(transient behavior)에 대해서는 고려하지 않는다. 이로 인해 OOS 상태에서 시작한 로봇이 결국 목표지점에는 도달하더라도, 그 경로가 전문가의 궤적과 크게 달라지는 현상이 발생한다. 따라서 본 논문의 목표는 수렴성뿐만 아니라 과도 응답 단계에서도 전문가의 행동을 밀접하게 모방하며, 어떤 섭동 상황에서도 효율적으로 복구할 수 있는 Contractive Dynamical System(축약 동역학 시스템) 기반의 정책 학습 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 정책을 설계 단계부터 **축약성(Contractivity)**을 갖도록 구조화하여, 별도의 제약 조건 최적화 없이도 모든 궤적이 지수적으로 서로에게 수렴하도록 보장하는 것이다.

1. **SCDS(State-only Contractive Dynamical System) 프레임워크**: 전문가의 속도 데이터 없이 오직 상태(state) 측정값만으로 정책을 학습하여, 기존 방법론에서 나타나던 누적 오차(cumulative error) 문제를 해결하였다.
2. **제약 없는 최적화 가능 구조**: Recurrent Equilibrium Networks(RENs)와 Coupling Layers를 결합하여, 파라미터 $\theta$의 선택과 관계없이 시스템의 축약성이 보장되는 아키텍처를 설계하였다. 이를 통해 계산 비용이 높은 제약 조건 최적화 대신 효율적인 무제약 최적화를 사용할 수 있다.
3. **OOS 복구의 이론적 보장**: OOS 초기 상태에서 전문가 궤적과의 최대 편차에 대한 이론적 상한선(upper bound)을 제시하여 배포 시의 신뢰성을 수학적으로 입증하였다.
4. **유연한 잠재 공간 학습**: 상태 공간을 저차원 또는 고차원 잠재 공간(latent space)으로 매핑하여 학습함으로써 고차원 상태 공간에서도 효율적인 학습이 가능하도록 하였다.

## 📎 Related Works

기존의 축약성 모방 학습 접근 방식은 크게 두 가지로 나뉜다.

첫째는 **제약 조건 최적화(Constrained Optimization)** 방식이다. 이는 학습 과정에서 축약성 조건을 만족시키기 위해 수학적 제약을 추가하는 방식이지만, 복잡한 전문가 행동이나 고차원 상태 공간에서는 계산 비용이 매우 높으며, 제약 조건을 만족시키기 위해 모방 정확도를 희생하는 트레이드오프가 발생한다.

둘째는 **내장된 보장 모델(Built-in Guarantees)** 방식이다. 파라미터 설정과 관계없이 구조적으로 축약성이 보장되는 모델을 사용하는 방식이다. 최근 Mohammadi et al. (2024) 등이 제안한 방식이 이에 해당하나, 이는 2차 미분 방정식(second-order ODE)을 해결해야 하는 복잡성이 있으며 저차원 환경에서 성능이 저하되는 한계가 있다.

SCDS는 RENs를 통해 1차 미분 방정식 수준에서 축약성을 보장하면서도, Coupling Layers를 도입하여 표현력을 높임으로써 기존 방식들의 계산 복잡성과 표현력 한계를 동시에 해결하였다.

## 🛠️ Methodology

### 전체 시스템 구조

SCDS 정책 $\phi_\theta$는 다음과 같은 연속 시간 자율 동역학 시스템(autonomous DS)으로 정의된다.
$$
\phi_\theta : \begin{cases} \hat{y}(t) = g_\theta(z(t)) & \text{(output transformation)} \\ z(0) = h_\theta(y_0) & \text{(initial condition)} \\ \dot{z}(t) = f_\theta(z(t)) & \text{(latent dynamics)} \end{cases}
$$
여기서 $y_0$는 초기 상태, $z(t)$는 잠재 상태, $\hat{y}(t)$는 정책이 생성한 계획 상태이다.

### 주요 구성 요소 및 역할

1. **Latent Dynamics ($f_\theta$)**: 잠재 상태의 동역학은 **Recurrent Equilibrium Networks(RENs)**로 모델링된다. RENs는 구조적으로 모든 파라미터 $\theta$에 대해 축약성을 보장하며, 그 수식은 다음과 같다.
    $$ \begin{bmatrix} \dot{z}(t) \\ v(t) \end{bmatrix} = \Omega(\theta, \gamma) \begin{bmatrix} z(t) \\ \sigma(v(t)) \end{bmatrix} $$
    여기서 $v(t)$는 내부 변수, $\gamma$는 조절 가능한 축약 속도(contraction rate), $\sigma$는 활성화 함수이다.

2. **Output Transformation ($g_\theta$)**: 잠재 공간의 축약성을 실제 상태 공간 $\hat{y}$에서도 유지하기 위해, 학습 가능한 선형 투영(linear projection) $P_\theta$와 $K$개의 **Coupling Layers**를 순차적으로 적용한다.
    $$ \hat{y}(t) = g_{\theta,1} \circ \dots \circ g_{\theta,K}(P_\theta z(t)) $$
    Coupling Layers는 가역적(bijective) 매핑으로, 잠재 공간의 축약 특성을 보존하면서 모델의 비선형 표현력을 극대화한다.

3. **Initial Condition ($h_\theta$)**: 초기 상태 $y_0$를 잠재 공간 $z(0)$으로 매핑하기 위해, Coupling Layers의 역함수와 $P_\theta$의 유사 역행렬(pseudoinverse) $P_\theta^\dagger$를 사용한다.
    $$ z(0) = P_\theta^\dagger g_{\theta,K}^{-1} \circ \dots \circ g_{\theta,1}^{-1}(y_0) $$

### 학습 절차 및 손실 함수

본 모델은 **Neural ODE** 프레임워크를 사용하여 미분 가능한 궤적을 생성하고, 전문가 데이터와 비교하여 $\theta$를 최적화한다.

- **Soft-DTW Loss**: 궤적의 길이나 속도가 다르더라도 공간적 유사성을 측정할 수 있도록 미분 가능한 soft-DTW 손실 함수를 사용한다.
- **가중치 부여 방식**: 여러 전문가 시연이 있을 때, 현재 초기 상태 $\hat{y}_0$와 전문가의 초기 상태 $y_{m,0}$ 사이의 거리의 역수에 비례하여 가중치 $\lambda_m$을 부여한다.
    $$ \lambda_m(\hat{y}_0) = \frac{\| \hat{y}_0 - y_{m,0} \|^{-2}}{\sum_{m'=1}^M \| \hat{y}_0 - y_{m',0} \|^{-2}} $$
    이는 현재 상태에서 가장 가까운 전문가 궤적을 우선적으로 모방하게 하여 학습의 일관성을 높인다.

## 📊 Results

### 실험 설정

- **데이터셋**: LASA(2D, 4D), Robomimic(6D, 14D)
- **비교 대상(Baselines)**: SNDS(Stable Neural DS), SDS-EF(Euclideanizing Flows), BC(Behavioral Cloning)
- **지표**: Mean Squared Error(MSE), soft-DTW
- **평가 방법**: 학습 데이터 내의 상태에서 시작하는 In-sample 궤적과, 학습 데이터 외의 무작위 상태에서 시작하는 OOS 궤적을 모두 생성하여 오차를 측정하였다.

### 주요 결과

1. **OOS 복구 성능**: SCDS는 모든 데이터셋에서 OOS 궤적 오차를 획기적으로 낮추었다. 특히 Robomimic 고차원 데이터셋에서 기존 베이스라인 대비 최대 2.5배 낮은 오차를 기록하며 강력한 일반화 능력을 보였다.
2. **안정성 비교**: BC는 안정성 보장이 없어 일부 궤적이 발산하였고, SNDS와 SDS-EF는 목표 지점에는 도달(asymptotic stability)하지만 전문가 궤적으로의 빠른 복구(contractivity) 능력이 부족하여 오차가 크게 나타났다. 반면 SCDS는 과도 응답 단계부터 빠르게 전문가 궤적으로 수렴하였다.
3. **이론적 상한선 검증**: Corollary 4.1.1에서 제시한 MSE 손실의 이론적 상한선($L_{MSE}^{ub}$)이 실제 측정된 OOS 손실보다 항상 크면서도 비교적 타이트하게 유지됨을 확인하여 이론의 타당성을 입증하였다.
4. **로봇 배포**: Isaac Lab 시뮬레이터 상에서 Franka Panda 암과 Clearpath Jackal 모바일 로봇에 적용한 결과, 무작위 OOS 상태에서도 신속하게 전문가 궤적으로 복구하며 작업을 완수하는 것을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

- **구조적 보장의 이점**: 축약성을 모델 아키텍처 내에 내장함으로써, 복잡한 제약 조건 최적화 없이 일반적인 경사하강법만으로 안정적인 정책을 학습할 수 있다는 점이 매우 효율적이다.
- **표현력의 확장**: 잠재 공간의 차원 $N_z$를 상태 공간 차원 $N_y$보다 크게 설정함으로써 복잡한 전문가 행동을 더 잘 캡처할 수 있으며, Coupling Layers가 REN의 내부 깊이를 늘리는 것보다 학습 속도와 표현력 측면에서 더 효율적임을 밝혔다.

### 한계 및 논의사항

- **계산 복잡도**: 전문가의 데이터가 급격하게 변하는 경우, 이를 정확히 모방하기 위해 ODE solver의 시뮬레이션 수평선(horizon $H$)을 늘려야 하며, 이는 계산 비용의 증가로 이어진다.
- **가정 사항**: 이론적 보장을 위해 초기 상태가 특정 다초점 타원 영역(multi-focal ellipse region) 내에 존재한다는 가정을 세웠는데, 실제 환경에서 이 영역의 크기 $R$이 매우 커질 경우 복구 성능이 저하될 가능성이 있다.

## 📌 TL;DR

본 논문은 모방 학습에서 OOS 상태의 불안정성을 해결하기 위해 **SCDS(State-only Contractive Dynamical System)**를 제안한다. RENs와 Coupling Layers를 결합하여 구조적으로 축약성(Contractivity)을 보장하는 정책을 설계함으로써, 무제약 최적화만으로도 어떤 섭동 상황에서든 전문가 궤적으로 빠르게 복구되는 강건한 정책을 학습할 수 있다. 이 방법론은 특히 고차원 로봇 조작 및 내비게이션 작업에서 기존 안정성 기반 방법론들보다 뛰어난 OOS 복구 성능을 보이며, 향후 안전성이 필수적인 실시간 로봇 제어 시스템에 적용될 가능성이 높다.
