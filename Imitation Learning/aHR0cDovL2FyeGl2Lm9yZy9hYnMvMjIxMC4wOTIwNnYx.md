# Model Predictive Control via On-Policy Imitation Learning

Kwangjun Ahn, Zakaria Mhammedi, Horia Mania, Zhang-Wei Hong, Ali Jadbabaie (2022)

## 🧩 Problem to Solve

본 논문은 제약 조건이 있는 선형 시스템(constrained linear systems)을 제어하기 위한 Model Predictive Control(MPC)의 계산 복잡도 문제를 해결하고자 한다. MPC는 매우 유연하고 강력한 제어 기법이지만, 매 시간 단계마다 온라인으로 최적화 문제를 해결해야 하므로 지연 시간(latency) 제약이 엄격하거나 계산 자원이 제한적인 고차원 시스템에 적용하기 어렵다는 단점이 있다.

이를 해결하기 위해 최근 데이터 기반의 모방 학습(Imitation Learning, IL)을 통해 MPC의 동작을 모사하는 명시적 제어기(explicit controller)를 학습시키려는 시도가 있었다. 하지만 가장 단순한 형태의 모방 학습인 Behavior Cloning(BC)은 학습 데이터와 실제 실행 시의 상태 분포가 달라지는 Distribution Shift 문제로 인해 오차가 누적(error compounding)되어 시스템이 불안정해지거나 제약 조건을 위반하는 문제가 발생한다. 따라서 본 논문의 목표는 Distribution Shift 문제를 해결하면서도, 시스템의 안정성(stability)과 제약 조건 만족(constraint satisfaction)을 이론적으로 보장하는 샘플 효율적인 데이터 기반 MPC 제어기를 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 On-policy 모방 학습 방법인 Forward Training 알고리즘을 MPC의 특성에 맞게 변형한 **Forward-Switch** 알고리즘을 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **Forward-Switch 알고리즘 제안**: 기존의 Forward Training은 시간 단계별로 서로 다른 제어기를 학습하므로 무한한 시간 지평(infinite horizon)에 적용하기 어렵다. 본 논문은 MPC 제어기가 특정 상태 영역(positively invariant set)에 진입하면 선형 이차 조절기(Linear Quadratic Regulator, LQR)와 동일하게 동작한다는 특성을 이용하여, 일정 단계 이후에는 학습된 제어기 대신 LQR로 전환하는 메커니즘을 도입하였다.
2.  **Robust MPC의 전문가 활용**: 단순한 MPC 대신 Robust MPC를 전문가(expert) 정책으로 사용하여, 학습 과정에서 발생하는 작은 오차를 외란(disturbance)으로 간주하고 이를 견딜 수 있는 강건한 제어기를 학습하도록 설계하였다.
3.  **이론적 보장 제공**: 학습된 제어기가 일정 수준 이상의 데이터가 확보되었을 때 시스템을 안정화시키고 제약 조건을 만족함을 이론적으로 증명하였다. 또한, 샘플 복잡도(sample complexity)와 비용 최적성(cost suboptimality)에 대한 상한선을 제시하였다.
4.  **시뮬레이션을 통한 검증**: 고차원 선형 시스템에 대해 BC 대비 Forward-Switch가 Distribution Shift에 강건하며 훨씬 적은 데이터로도 최적 성능에 도달함을 입증하였다.

## 📎 Related Works

기존의 접근 방식은 크게 두 가지로 나뉜다. 첫째는 최적화 문제를 사전에 계산하여 저장하는 Explicit MPC이며, 둘째는 신경망 등을 통해 MPC를 근사하는 데이터 기반 방식이다. 

특히 최근의 데이터 기반 방식들은 주로 Behavior Cloning(BC)을 사용하였다. BC는 전문가가 생성한 궤적 데이터를 수집하고 이를 지도 학습(supervised learning) 방식으로 모사한다. 그러나 BC는 상호작용 없는(non-interactive) 학습 방식이므로, 학습 시 보지 못한 상태에 진입했을 때 발생하는 작은 오차가 다음 상태의 분포를 변화시키고, 이것이 다시 더 큰 오차를 유발하는 Distribution Shift 문제가 치명적이다. 본 논문은 이러한 한계를 극복하기 위해 전문가와 상호작용하며 단계적으로 학습하는 On-policy 방식의 Forward Training을 채택하여 기존 BC 기반 접근 방식과 차별화를 두었다.

## 🛠️ Methodology

### 1. 시스템 모델 및 전문가 정의
본 논문은 다음과 같은 제약 조건이 있는 선형 동역학 시스템을 다룬다.
$$x_{t+1} = Ax_t + Bu_t, \quad x_t \in X, u_t \in U$$
여기서 $X$와 $U$는 상태 및 입력 제약 집합이다. 전문가는 Robust MPC ($\pi$)로 설정되며, 이는 외란 $w \in W$가 존재하더라도 제약 조건을 만족하며 시스템을 안정화하도록 설계된 제어기이다.

### 2. Forward Training 알고리즘
Forward Training은 시간 단계 $t$에 대해 제어기 $\hat{\pi}_t$를 귀납적으로 학습한다.
- **Stage 0**: 초기 상태 분포 $D$에서 샘플을 추출하여 $\hat{\pi}_0$를 학습한다.
- **Stage $t$**: 이전 단계까지 학습된 제어기 $\hat{\pi}_{0:t-1}$를 사용하여 시스템을 구동하고, $t$ 시점에 도달한 상태 $\hat{x}_t$들을 수집한다. 이 상태들에서 전문가의 출력 $\pi^?(\hat{x}_t)$를 쿼리하여 $\hat{\pi}_t$를 학습한다.
이 방식은 배포 시 $\hat{\pi}_t$가 학습 때와 동일한 분포의 상태에서 평가되므로 Distribution Shift 문제를 근본적으로 회피한다.

### 3. Forward-Switch 알고리즘
Forward Training의 샘플 복잡도가 시간 지평 $T$에 비례하여 증가하는 문제를 해결하기 위해 다음과 같은 절차를 수행한다.
1.  **Inductive Learning**: $\hat{\pi}_0$부터 순차적으로 학습한다.
2.  **Early Termination**: 매 단계마다 샘플 궤적들이 Robust MPC의 양의 불변 집합(positively invariant set)인 $\bar{O}_\infty$에 진입했는지 확인한다.
3.  **Switching**: 모든 샘플이 $\bar{O}_\infty$에 진입한 시점을 $\hat{\tau}_\infty$로 설정하고, $t \ge \hat{\tau}_\infty$ 이후에는 더 이상 학습하지 않고 명시적인 LQR 제어기 $\pi_{lqr}$를 사용한다.

최종 출력 정책 $\tilde{\pi}_t$는 다음과 같이 정의된다.
$$\tilde{\pi}_t = \begin{cases} \hat{\pi}_t, & \text{if } t < \hat{\tau}_\infty \\ \pi_{lqr}, & \text{if } \hat{\tau}_\infty \le t \le T-1 \end{cases}$$

### 4. 모델 클래스 및 학습
제어기 $\Pi$는 ReLU 활성화 함수를 가진 신경망을 사용한다. 이는 MPC의 최적 제어기가 구간별 아핀(piecewise affine) 함수 형태를 띠며, ReLU 네트워크가 이를 효율적으로 표현할 수 있다는 기존 연구 결과에 근거한다. 학습은 경험적 위험 최소화(ERM)를 통해 수행되며, 손실 함수는 전문가 출력과의 $L_2$ 노름 차이를 최소화하는 방향으로 설정된다.

## 📊 Results

### 실험 설정
- **시스템**: 개루프 불안정(open-loop unstable) 선형 시스템, 차원 $d \in \{3, 5\}$.
- **제약 조건**: 상태 $x_t \in [-100, 100]^d$, 입력 $u_t \in [-10, 10]$.
- **비교 대상**: Behavior Cloning(BC), 기본 Forward Training, Forward-Switch.
- **평가 지표**: 정규화된 비용(Normalized cost-to-go), 제약 조건 만족 비율(Constraint satisfaction ratio).

### 주요 결과
1.  **BC의 한계**: $d=3$ 및 $d=5$ 모든 케이스에서 BC는 Distribution Shift로 인해 궤적이 발산하며, 매우 많은 데이터를 사용하더라도 비용이 급증하고 제약 조건을 빈번하게 위반하였다.
2.  **Forward-Switch의 효율성**: 
    - $d=5$ 실험에서 Forward-Switch는 단 180개의 MPC 데모만으로도 정규화된 비용 $\approx 1.034$를 달성하였다.
    - 반면, 기본 Forward 알고리즘은 210개의 데모를 사용했음에도 비용이 $\approx 35$로 매우 높게 나타났다. 이는 LQR로의 전환(Switching)이 샘플 효율성을 극도로 높였음을 보여준다.
3.  **안정성**: Figure 2의 궤적 분석 결과, BC는 시간이 지날수록 전문가의 궤적에서 벗어나 발산하는 반면, Forward-Switch는 전문가의 궤적을 매우 유사하게 추종하며 시스템을 안정적으로 원점으로 수렴시켰다.

## 🧠 Insights & Discussion

본 논문은 제어 이론(MPC, LQR)과 기계 학습(Imitation Learning)을 성공적으로 결합하였다. 특히, 단순히 신경망으로 MPC를 모사하는 것에 그치지 않고, **"MPC $\rightarrow$ LQR"**로 이어지는 제어 이론적 특성을 학습 알고리즘의 조기 종료 조건으로 활용한 점이 매우 영리한 설계이다.

또한, 학습된 제어기의 오차를 단순한 노이즈가 아닌 외란(disturbance)으로 해석하여 Robust MPC를 전문가로 설정함으로써, 이론적 안정성 보장과 실제 성능 향상을 동시에 달성하였다. 다만, 본 연구는 선형 시스템과 볼록 제약 조건이라는 제한적인 환경에서 수행되었으므로, 비선형 시스템이나 비볼록(non-convex) 제약 조건이 있는 복잡한 환경에서도 동일한 이론적 보장과 효율성이 유지될지는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 MPC의 온라인 계산 부담을 줄이기 위해 On-policy 모방 학습 기반의 **Forward-Switch** 알고리즘을 제안한다. 이 알고리즘은 Distribution Shift 문제를 해결하기 위해 단계별 학습을 수행하며, 상태가 특정 영역에 진입하면 LQR로 전환하여 샘플 효율성을 극대화한다. 실험 결과, 기존 Behavior Cloning보다 훨씬 적은 데이터로도 시스템 안정성과 제약 조건을 보장하는 고성능 제어기를 학습할 수 있음을 입증하였다. 이 연구는 고차원 시스템의 실시간 제어를 위한 데이터 기반 제어기 설계에 중요한 이론적/실천적 가이드라인을 제공한다.