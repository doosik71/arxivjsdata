# Localized active learning of Gaussian process state space models

Alexandre Capone, Gerrit Noske, Jonas Umlauft, Thomas Beckers, Armin Lederer, Sandra Hirche (2020)

## 🧩 Problem to Solve

본 논문은 동적 시스템의 상태 공간(state space)이 무한하거나 매우 넓은 경우, 시스템 전체에 대해 정확한 모델을 구축하려는 기존의 탐색(exploration) 기법들이 가지는 비효율성 문제를 해결하고자 한다. 일반적으로 비매개변수 모델(non-parametric model)을 사용할 때 전역적으로 정확한 모델을 얻기 위해서는 잠재적으로 무한한 수의 데이터 포인트가 필요하며, 이는 현실적으로 불가능하다.

또한, 실제 많은 제어 응용 분야(예: 국소 안정화 작업, local stabilization tasks)에서는 시스템 전체가 아닌 특정 관심 영역(region of interest) 내에서만 모델이 정확하면 충분하다. 따라서 본 연구의 목표는 상태-제어 공간(state-action space)의 유한한 부분 집합 내에서 모델의 정확도를 효율적으로 높일 수 있는 국소적 능동 학습(Localized Active Learning, LocAL) 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 탐색 궤적과 관심 영역의 이산화(discretization) 지점들 간의 상호 정보량(Mutual Information)을 최대화하는 것이다. 이를 달성하기 위해 저자들은 다음과 같은 설계를 도입하였다.

1. **정보량 기반의 목표 지점 설정**: 관심 영역 내에서 가장 정보 가치가 높은 단일 데이터 포인트 $\xi^*$를 먼저 계산한다.
2. **MPC를 통한 궤적 생성**: 계산된 $\xi^*$를 목표 지점으로 설정하고, Model Predictive Control(MPC)를 사용하여 시스템을 해당 지점으로 유도한다.
3. **계산 복잡도 해소**: 상호 정보량을 직접 MPC 최적화 단계에 통합하는 것은 계산적으로 불가능(intractable)하므로, '가장 정보가 많은 점의 선택'과 'MPC 최적화' 단계를 분리하여 병렬적으로 해결 가능한 구조로 만들었다.

## 📎 Related Works

기존의 데이터 기반 제어 및 시스템 식별 연구에서는 Gaussian Process(GP)가 우수한 일반화 성능과 불확실성 추정 능력 덕분에 널리 사용되어 왔다. 탐색 전략 측면에서는 다음과 같은 접근 방식들이 존재한다.

- **무작위 탐색(Random Exploration)**: 일정 확률로 무작위 제어 입력을 선택하는 방식이나, 이미 불확실성이 낮은 영역을 반복적으로 방문하게 되어 매우 비효율적이다.
- **전역적 능동 학습(Global Active Learning)**: 정보 획득량을 최대화하는 궤적을 선택하여 전역적으로 정확한 모델을 얻으려는 시도이다. 하지만 이는 상태 공간이 유한한 경우에만 유효하며, 무한한 상태 공간에서는 적용이 어렵다.

본 논문은 이러한 전역적 접근 방식과 달리, 특정 유효 영역(bounded subset)에 집중함으로써 데이터 효율성을 극대화하고 무한한 상태 공간 문제에 대응한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 시스템 모델링
본 논문은 다음과 같은 이산 시간 비선형 시스템을 가정한다.
$$x_{t+1} = f(\tilde{x}_t) + g(\tilde{x}_t) + w_t$$
여기서 $\tilde{x}_t = [x_t^T, u_t^T]^T$는 상태와 제어 입력의 결합 벡터이며, $f(\cdot)$는 물리 법칙 등으로 알려진 성분, $g(\cdot)$는 학습해야 할 미지의 성분, $w_t \sim \mathcal{N}(0, \Sigma_w)$는 가우시안 노이즈이다.

### Gaussian Process (GP) 기반 학습
미지 함수 $g(\cdot)$를 GP로 모델링하며, 훈련 입력으로는 $\tilde{x}_t$를, 타겟 값으로는 실제 측정된 전이 값인 $x_{t+1} - f(\tilde{x}_t)$를 사용한다. GP의 사후 평균 $\mu_n(\tilde{x}_t)$와 분산 $\sigma_n^2(\tilde{x}_t)$는 다음과 같이 정의된다.
$$\mu_n(\tilde{x}_t) = k^T(\tilde{x}_t)(K + \sigma_w^2 I)^{-1} y_{\tilde{X}}$$
$$\sigma_n^2(\tilde{x}_t) = k(\tilde{x}_t, \tilde{x}_t) - k^T(\tilde{x}_t)(K + \sigma_w^2 I)^{-1} k(\tilde{x}_t)$$
여기서 $k(\cdot, \cdot)$는 커널 함수이며, $K$는 훈련 데이터 간의 공분산 행렬이다.

### LocAL 알고리즘의 동작 원리
LocAL 알고리즘은 다음의 단계를 반복적으로 수행한다.

1. **최적 정보 지점 $\xi^*$ 탐색**: 관심 영역 $\tilde{X}_{ref}$ 내에서 상호 정보량 $I(y_\xi, y_{\tilde{X}_{ref}})$를 최대화하는 지점 $\xi^*$를 찾는다. 이때 계산 효율을 위해 다음 식의 최소값을 구한다.
$$\xi^* = \arg \max_{\xi \in \tilde{X}} \frac{1}{2} \log \frac{(k(\xi, \xi) + \sigma_w^2) |K_{\tilde{X}_{ref}} + \sigma_w^2 I|}{|K_{\tilde{X}_{ref} \cup \xi} + \sigma_w^2 I|}$$
2. **MPC 제어**: 계산된 $\xi^*$로 시스템을 유도하기 위해 다음 비용 함수를 최소화하는 제어 입력 시퀀스 $U^*$를 계산한다.
$$\min_{U \in \mathcal{U}} \sum_{t=1}^{N_H} (\xi^* - \tilde{x}_t)^T Q (\xi^* - \tilde{x}_t)$$
제약 조건으로 GP 평균 모델 $\mu_t(\cdot)$를 이용한 시스템 동역학을 사용한다.
3. **데이터 수집 및 업데이트**: 최적 제어 입력을 적용하여 새로운 상태를 측정하고, 이를 GP 모델에 추가하여 $\mu_t, \sigma_t^2$를 업데이트한다.

### 이론적 분석 (Sensitivity Analysis)
본 논문은 Theorem 1을 통해 상태 $\tilde{x}_t$와 최적 정보 지점 $\xi^*$ 사이의 거리가 줄어들수록 상호 정보량의 차이가 줄어듦을 수학적으로 증명하였다. 이는 MPC를 통해 $\xi^*$에 가깝게 접근하는 전략이 실제로 모델의 정확도를 점진적으로 향상시킬 수 있음을 정당화한다.

## 📊 Results

### 실험 설정
- **대상 시스템**: Toy Problem, Surface Exploration, Pendulum, Cart-pole 총 4가지 시스템.
- **평가 지표**: 관심 영역 $\tilde{X}_{ref}$ 내의 500개 샘플 포인트에 대한 Root Mean Square Error (RMSE).
- **비교 대상**: Greedy entropy-based exploration (탐욕적 엔트로피 기반 탐색).
- **공통 설정**: Squared-exponential kernel 사용, MPC horizon $N_H = 10$, 하이퍼파라미터 온라인 최적화.

### 주요 결과
1. **Toy Problem**: LocAL은 시스템을 관심 영역 내에 머물게 하여 RMSE를 빠르게 낮춘 반면, 엔트로피 기반 방식은 불필요하게 넓은 영역을 탐색하여 효율성이 떨어졌다.
2. **Surface Exploration**: 모든 상태 변수가 무한한 환경에서, 엔트로피 기반 방식은 관심 영역에 도달하지 못해 모델 개선이 거의 없었으나, LocAL은 효과적으로 영역을 탐색하여 RMSE를 크게 낮추었다.
3. **Pendulum & Cart-pole**: 두 시스템 모두에서 LocAL이 엔트로피 기반 방식보다 낮은 RMSE와 적은 표준 편차를 보였다. 이는 관심 영역(예: 진자의 수직 상단 위치)을 더 정밀하게 탐색했기 때문이다.

## 🧠 Insights & Discussion

### 강점 및 의의
본 연구는 무한한 상태 공간을 가진 시스템에서 '전역적 최적화'라는 불가능한 목표 대신 '국소적 최적화'라는 현실적인 목표를 설정하여 능동 학습의 효율성을 극대화하였다. 특히 상호 정보량 계산과 MPC 제어를 분리함으로써 계산 복잡도 문제를 해결하고 실시간 적용 가능성을 높인 점이 돋보인다.

### 한계 및 논의사항
- **MPC 모델 의존성**: MPC 단계에서 GP의 평균 모델 $\mu_t$만을 사용하고 불확실성을 전파(uncertainty propagation)하지 않았다. 비록 논문에서 moment-matching 등의 기법으로 확장 가능하다고 언급하였으나, 모델 불확실성이 매우 큰 초기 단계에서 $\xi^*$로의 유도 성능이 얼마나 보장되는지에 대한 추가 분석이 필요하다.
- **관심 영역 설정**: $\tilde{X}_{ref}$를 사전에 정의해야 한다는 점은 사용자에게 도메인 지식을 요구한다. 만약 관심 영역이 동적으로 변해야 하는 상황이라면 본 알고리즘의 수정이 필요할 것이다.

## 📌 TL;DR

본 논문은 무한한 상태 공간을 가진 동적 시스템에서 특정 관심 영역 내의 모델 정확도를 효율적으로 높이기 위한 **LocAL(Localized Active Learning)** 알고리즘을 제안한다. 이 방법은 GP를 통해 관심 영역 내에서 가장 정보 가치가 높은 지점 $\xi^*$를 찾고, MPC를 통해 시스템을 해당 지점으로 유도하는 방식을 취한다. 실험 결과, 전역적 탐색을 수행하는 기존 엔트로피 기반 방식보다 관심 영역 내에서 훨씬 빠르고 정확하게 모델을 학습함을 입증하였다. 이 연구는 로봇 제어나 시스템 식별과 같이 특정 작동 영역의 정밀한 모델이 필요한 실무 응용 분야에 매우 중요한 기여를 할 것으로 보인다.