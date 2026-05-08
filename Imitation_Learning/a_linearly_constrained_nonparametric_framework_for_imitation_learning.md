# A Linearly Constrained Nonparametric Framework for Imitation Learning

Yanlong Huang and Darwin G. Caldwell (2019)

## 🧩 Problem to Solve

본 논문은 로봇의 모방 학습(Imitation Learning) 과정에서 발생하는 **선형 제약 조건(Linear Constraints)의 처리 문제**를 해결하고자 한다.

일반적인 모방 학습 방법론들은 제약 조건이 없는 환경에서의 기술 습득에 집중해 왔으나, 실제 로봇 시스템에서는 물리적인 환경이나 하드웨어의 한계로 인해 다양한 제약 조건이 발생한다. 예를 들어, 로봇이 칠판에 글씨를 쓰는 작업에서는 말단 장치(end-effector)의 궤적이 칠판이라는 평면 제약 조건(plane constraint)을 따라야 하며, 모든 관절 궤적은 관절 한계(joint limits)를 준수해야 한다.

따라서 본 연구의 목표는 다음과 같은 두 가지 핵심 요구사항을 동시에 충족하는 범용적인 프레임워크를 개발하는 것이다.

1. 다수의 시연 데이터 학습, 궤적의 재현 및 재생성, 그리고 위치와 속도 관점에서의 경유점(via-points) 및 종점(end-points)에 대한 적응(adaptation)이라는 모방 학습의 핵심 기능을 유지할 것.
2. 실제 환경에서 흔히 발생하는 임의의 선형 등식 제약(예: 평면 제약) 및 부등식 제약(예: 액션 컴포넌트의 선형 결합이 특정 값보다 크거나 작아야 함)을 처리할 수 있을 것.

## ✨ Key Contributions

본 논문의 핵심 기여는 **LC-KMP(Linearly Constrained Kernelized Movement Primitives)**라는 비매개변수적(non-parametric) 프레임워크를 제안한 것이다.

중심적인 아이디어는 다수 시연 데이터의 확률적 특성을 먼저 추출하고, 이를 **선형 제약이 포함된 최적화 문제(linearly constrained optimization problem)**로 정식화하는 것이다. 특히, 커널 트릭(kernel trick)을 적용함으로써 매개변수의 수에 제한받지 않는 비매개변수적 해를 도출하였다. 이를 통해 로봇은 시연된 동작의 특징을 유지하면서도, 사용자가 정의한 물리적 제약 조건을 엄격히 준수하는 궤적을 생성할 수 있다.

## 📎 Related Works

논문에서는 Dynamical Movement Primitives (DMP), Task-parameterized Gaussian Mixture Model (GMM), 그리고 Kernelized Movement Primitives (KMP)와 같은 기존의 모방 학습 접근 방식을 언급한다.

기존 연구들의 한계점은 대부분 외부 또는 내부 제약 조건을 무시한 채 인간의 기술을 학습하는 데 집중했다는 점이다. 일부 연구에서 등식 제약(equality constraints)이나 관절 한계 회피(joint limit avoidance)를 다루었으나, 이는 특정 사례에 국한된 해결책이었다.

제안된 LC-KMP는 기존 KMP의 장점(시간 및 고차원 입력에 대한 학습 및 적응 능력)을 그대로 계승하면서, 등식과 부등식 제약을 모두 처리할 수 있는 일반적인 프레임워크를 제공한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 시연 데이터의 확률적 특성 추출

먼저 $M$개의 시연 데이터 $D$를 확률적으로 모델링한다. 출력 $\xi \in \mathbb{R}^O$와 그 일차 미분 $\dot{\xi}$를 결합하여 $\eta = [\xi^T, \dot{\xi}^T]^T$로 정의하고, GMM을 통해 결합 확률 분포 $P(t, \eta)$를 모델링한다.

$$P(t, \eta) \sim \sum_{c=1}^{C} \pi_c \mathcal{N}(\mu_c, \Sigma_c)$$

이후 Gaussian Mixture Regression (GMR)을 통해 확률적 참조 궤적 $D_r = \{t_n, \hat{\mu}_n, \hat{\Sigma}_n\}_{n=1}^N$을 추출하며, 이는 시연 데이터의 평균과 공분산 정보를 캡슐화한다.

### 2. 선형 제약 기반의 최적화 문제 정식화

궤적 $\eta(t)$를 $\eta(t) = \Theta^T(t)w$라는 매개변수 형태로 표현한다. 여기서 $\Theta(t)$는 기저 함수(basis function) 벡터이며, $w$는 학습해야 할 매개변수이다.

학습 목표는 확률 밀도를 최대화하는 $w$를 찾는 것이며, 동시에 다음과 같은 선형 제약 조건을 만족해야 한다.

$$\text{argmax}_w \sum_{n=1}^N P(\eta(t_n) | \hat{\mu}_n, \hat{\Sigma}_n)$$
$$\text{s.t. } g_{n,f}^T \eta(t_n) \ge c_{n,f}, \quad \forall f \in \{1, \dots, F\}, \forall n \in \{1, \dots, N\}$$

이 문제는 로그 변환과 정규화 항 $\frac{1}{2}\lambda w^T w$를 추가하여 다음과 같은 최소화 문제로 재구성된다.

$$\text{argmin}_w \sum_{n=1}^N \frac{1}{2}(\Theta^T(t_n)w - \hat{\mu}_n)^T \hat{\Sigma}_n^{-1}(\Theta^T(t_n)w - \hat{\mu}_n) + \frac{1}{2}\lambda w^T w$$
$$\text{s.t. } g_{n,f}^T \eta(t_n) \ge c_{n,f}$$

### 3. 라그랑주 승수법 및 커널 트릭 적용

위의 제약 최적화 문제를 풀기 위해 라그랑주 승수 $\alpha_{n,f} \ge 0$를 도입하여 라그랑주 함수 $L(w, \alpha)$를 구성한다. $w$에 대해 미분하여 $0$이 되는 지점을 찾으면, 최적의 $w^*$는 $\alpha$에 대한 함수로 표현된다.

최종적으로 $\alpha$를 결정하는 문제는 다음과 같은 이차 계획법(Quadratic Programming, QP) 문제로 변환된다.

$$\max_{\alpha} \tilde{L}(\alpha) = \alpha^T B_1 \alpha + B_2 \alpha, \quad \text{s.t. } \alpha \ge 0$$

여기서 커널 트릭 $\phi(t_i)^T \phi(t_j) = k(t_i, t_j)$을 적용하여, 매개변수 $w$를 직접 구할 필요 없이 커널 행렬 $K$를 이용해 예측값 $\eta(t^*)$를 바로 계산한다.

$$\eta(t^*) = k^*(K + \lambda \Sigma)^{-1}(\mu + \Sigma G \alpha^*)$$

### 4. 알고리즘 흐름 요약

1. **초기화**: $\lambda$, 커널 함수 $k(\cdot, \cdot)$, 선형 제약 조건 및 시연 데이터를 설정한다.
2. **모델링**: GMM/GMR을 통해 참조 궤적 $D_r$을 추출한다.
3. **최적화**: QP 문제를 풀어 최적의 라그랑주 승수 $\alpha^*$를 찾는다.
4. **예측**: 새로운 입력 $t^*$에 대해 $\eta(t^*)$를 예측한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 작업**: 2D 문자 'G' 쓰기(운동 제한 포함), 3D 문자 'G' 쓰기(평면 제약 포함), 휴머노이드 로봇의 안정적 보행 궤적 생성.
- **비교 대상**: Vanilla KMP (제약 조건이 없는 표준 KMP).
- **지표**: 궤적이 정의된 제약 범위(limits)를 준수하는지 여부 및 목표 지점(desired points)에 도달하는지 여부를 정성적/정량적으로 평가.

### 2. 주요 결과

- **2D 문자 'G' 적응**: 위치 및 속도 제한($-4 \le x \le 10, y \ge -4, \dot{x} \ge -32, \dot{y} \ge -20$) 하에서 LC-KMP는 목표 지점으로 궤적을 적응시키면서도 제한 범위를 엄격히 준수하였다. 반면, Vanilla KMP는 제약 조건을 무시하고 궤적을 생성하였다.
- **평면 제약 적응**: 3D 공간에서 임의의 평면 방정식($a_x x + b_y y + c_z z = d$)을 제약으로 주었을 때, LC-KMP는 다양한 평면 위에서 시연 데이터를 적절히 투영하고 적응시키는 능력을 보였다.
- **휴머노이드 보행 안정성**: CoM(Center of Mass) 궤적에 'Capture Region'이라는 안정성 제약 조건을 적용하였다. 실험 결과, Vanilla KMP는 Y방향 제약 범위를 벗어나 불안정한 궤적을 생성한 반면, LC-KMP는 모든 제약 조건을 만족하며 안정적인 보행 궤적을 생성하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 유연성

LC-KMP는 단순히 제약을 거는 것에 그치지 않고, 다음과 같은 유연한 운용이 가능하다.

- **부분적 제약 학습(Partially-constrained Learning)**: 특정 시간 구간에서만 제약을 활성화하고 다른 구간에서는 비활성화함으로써, 제약과 목표 지점이 충돌할 때 전략적인 선택이 가능하다.
- **등식 제약 처리**: 부등식 제약 두 개($g^T \eta \ge c - \epsilon$ 및 $-g^T \eta \ge -c - \epsilon$)를 조합하여 등식 제약을 근사적으로 구현할 수 있다.

### 2. MPC(Model Predictive Control)와의 연결성

논문은 LC-KMP와 선형 MPC 사이의 수학적 유사성을 분석한다. 두 방법 모두 궤적 최적화 문제의 형태를 띠고 있으나, 결정적인 차이점은 다음과 같다.

- **모델의 차이**: MPC는 시스템 동역학 모델($\eta_{t+1} = A\eta_t + Bu_t$)을 사용하여 미래를 예측하지만, LC-KMP는 매개변수 모델($\eta(t) = \Theta^T(t)w$) 및 커널 기반의 비매개변수적 접근법을 사용한다.
- **최적화 대상**: MPC는 최적의 제어 입력($u$)을 찾는 것이 목적이지만, LC-KMP는 최적의 궤적 매개변수($w$) 또는 그에 대응하는 $\alpha$를 찾는 것이 목적이다.

### 3. 비판적 해석 및 한계

본 논문은 시간 기반(time-driven) 궤적 학습에 집중하고 있다. 커널 트릭을 통해 고차원 입력으로 확장 가능하다는 가능성을 제시했으나, 실제 고차원 입력 환경에서의 제약 조건 처리 성능에 대한 실험적 검증은 부족한 상태이다. 또한, 선형 제약 조건만을 다루고 있어, 비선형 제약 조건이 강하게 작용하는 환경에서의 적용 가능성에 대해서는 추가적인 연구가 필요하다.

## 📌 TL;DR

본 논문은 로봇 모방 학습 시 물리적/환경적 제약 조건을 준수하기 위해, GMM/GMR의 확률적 모델링과 커널 트릭 기반의 제약 최적화를 결합한 **LC-KMP** 프레임워크를 제안하였다. 이 연구는 특히 안정적인 보행이나 정밀한 평면 작업과 같이 **안전성과 물리적 정밀도가 필수적인 로봇 제어 분야**에서 기존의 단순 모방 학습이 가진 한계를 극복하고, 제약 조건 하에서도 유연한 궤적 적응이 가능함을 입증하였다.
