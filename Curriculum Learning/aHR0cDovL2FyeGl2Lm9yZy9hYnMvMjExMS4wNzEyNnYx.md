# On the Statistical Benefits of Curriculum Learning

Ziping Xu, Ambuj Tewari (2021)

## 🧩 Problem to Solve

본 논문은 머신러닝 학습 전략으로 널리 사용되는 Curriculum Learning(CL)의 통계적 이점에 대한 이론적 이해가 부족하다는 문제점에서 출발한다. 기존의 CL 연구들은 주로 최적화(Optimization) 관점에서 학습의 난이도를 조절하여 수렴 속도를 높이는 'Optimization Benefit'에 집중해 왔다. 그러나 데이터의 샘플 할당량이나 태스크 선택 순서가 모델의 최종 성능(통계적 오차)에 어떤 영향을 미치는지에 대한 'Statistical Benefit' 관점의 엄밀한 분석은 부족한 상태이다.

따라서 본 논문의 목표는 다중 태스크 선형 회귀(Multi-task Linear Regression) 문제에서 CL의 통계적 이점을 정량적으로 분석하는 것이다. 특히, 태스크 간의 구조가 없는 Unstructured setting과 저차원 표현을 공유하는 Structured setting 두 가지 환경에서, 최적의 커리큘럼을 알고 있는 Oracle 시나리오와 학습자가 스스로 커리큘럼을 결정해야 하는 Adaptive learning 시나리오 간의 Minimax Rate를 도출하여 그 차이를 규명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 CL의 통계적 이점을 수학적으로 공식화하고, 다음과 같은 직관적인 설계 아이디어를 통해 분석했다는 점이다.

1. **통계적 이점의 공식화**: CL을 단순히 학습 순서의 문제가 아니라, 각 태스크에서 얼마나 많은 샘플을 추출할 것인가에 대한 최적화 문제로 정의하였다.
2. **Adaptive vs Oracle Gap 분석**: Unstructured setting에서는 적응적 학습(Adaptive learning)이 Oracle 학습보다 근본적으로 어려울 수 있음을 보였으며, Structured setting에서는 그 차이가 매우 작음을 입증하였다.
3. **OFU 기반 적응적 스케줄러 제안**: Representation learning 환경에서 태스크의 다양성(Diversity)을 확보하기 위해 '불확실성 하에서의 낙관주의(Optimism in the Face of Uncertainty, OFU)' 원칙을 적용한 태스크 스케줄링 알고리즘을 제안하였다.
4. **Prediction Gain 방법론의 이론적 정당화**: 실무에서 자주 사용되는 '로컬 예측 이득(Local prediction gain)이 높은 태스크를 선택하는 탐욕적 방법'이 실제로 Minimax lower bound에 근접하는 성능을 낸다는 것을 수학적으로 증명하였다.

## 📎 Related Works

논문은 CL의 이점을 두 가지 관점으로 구분하며 기존 연구를 검토한다.

- **Optimization Benefit**: Bengio et al. (2009) 등은 쉬운 데이터부터 학습하는 것이 목적 함수의 convex-ness를 높여 비볼록(non-convex) 최적화 문제에서 전역 최적해(global optima)에 더 빠르게 도달하게 한다고 주장하였다. 이는 주로 학습 순서에 의존하는 이점이다.
- **Statistical Benefit**: 샘플 할당량의 최적화를 통해 노이즈를 줄이고 일반화 성능을 높이는 이점이다. Weinshall and Amir (2020)가 노이즈 수준이 다른 샘플들의 수렴 속도를 분석한 바 있으나, 본 논문은 이를 다중 태스크 환경으로 확장하여 분석한다.

기존의 전이 학습(Transfer Learning)이나 다중 태스크 학습(Multitask Learning) 연구들은 주로 고정된 샘플 수를 가정하지만, 본 논문은 **학습자가 동적으로 샘플 수를 결정할 수 있는 커리큘럼 설계 능력**에 초점을 맞추어 기존 접근 방식과 차별화를 둔다.

## 🛠️ Methodology

### 1. Unstructured Linear Regression
각 태스크 $t$의 응답 변수 $Y_t$는 다음과 같이 생성된다.
$$Y_t = X_t^T \theta^*_t + \epsilon_t$$
여기서 $\epsilon_t$는 분산이 $\sigma^2_t$인 가우시안 노이즈이며, $\theta^*_t$는 태스크 $t$의 진정한 파라미터이다. 타겟 태스크 $T$에 대한 오차를 최소화하는 것이 목표이다.

- **Transfer Distance**: 두 태스크 간의 유사성을 $\Delta_{t_1, t_2} = \|\theta^*_{t_1} - \theta^*_{t_2}\|_2$로 정의한다.
- **Oracle Rate**: 최적의 커리큘럼을 알고 있을 때, 초과 리스크(Excess Risk)의 하한선은 다음과 같이 나타난다.
$$R^N_T(\Theta(Q)) \gtrsim C_0 \min_t \{ Q_t^2 + \frac{d\sigma^2_t}{N} \}$$
이는 전이 거리($Q_t$)와 노이즈 수준($\sigma^2_t$) 사이의 균형을 맞추는 최적의 태스크를 찾는 것이 핵심임을 시사한다.

### 2. Structured Linear Regression (Representation Learning)
모든 태스크가 공통의 저차원 표현 행렬 $B^* \in \mathbb{R}^{d \times k}$를 공유하며, 각 태스크는 고유의 계수 $\beta^*_t \in \mathbb{R}^k$를 갖는 구조이다.
$$f_t(x) = x^T B^* \beta^*_t$$

- **Diversity**: 표현 학습의 성능은 선택된 태스크들의 $\beta^*_t$가 공간을 얼마나 잘 커버하는지, 즉 $\lambda_k(\sum \beta^*_{t_i} \beta^{*T}_{t_i})$(최소 고윳값)가 얼마나 큰지에 달려 있다.
- **OFU Scheduler**: 학습자는 각 태스크에 대한 신뢰 집합(Confidence set) $B_{i,t}$를 구성하고, 이 집합 내에서 최소 고윳값 $\lambda_k$를 최대화할 수 있는 태스크를 낙관적으로 선택한다.

### 3. Prediction Gain driven Scheduler
로컬 예측 이득 $G(A, H_{i+1}) = L_T(\theta_i) - L_T(\theta_{i+1})$을 최대화하는 태스크를 선택하는 방식이다. 논문은 이를 SGD 업데이트 식과 연결하여, 이 방법이 통계적으로 최적의 태스크 $t^*$를 선택하는 것과 일치함을 보인다.

## 📊 Results

### 1. Unstructured Setting의 결과
- **Adaptive Learning의 한계**: 전이 거리를 모르는 상태에서 적응적으로 학습할 경우, $\frac{\sigma^2_T \log(T)}{N}$라는 피할 수 없는 추가 손실이 발생한다. 이는 타겟 태스크의 데이터 없이 소스 태스크만으로는 어떤 태스크가 최적인지 판별할 수 없기 때문이다.
- **이득의 존재**: 그럼에도 불구하고, 적절한 소스 태스크를 찾았을 때의 오차는 단일 태스크 학습 시의 $d/N$ 항보다 $\log(T)/N$ 수준으로 낮아질 수 있어, 잠재적인 성능 향상 폭은 $d/\log(T)$ 배에 달한다.

### 2. Structured Setting의 결과
- **적응적 다양성 확보**: 제안한 OFU 알고리즘을 통해, 태스크의 실제 파라미터를 모르더라도 적응적으로 다양한 태스크를 선택하여 Oracle에 근접한 다양성을 확보할 수 있음을 보였다.
- **수렴 속도**: $N$이 충분히 클 때($dkT \ll N$), 적응적 스케줄링으로 인한 추가 오차는 무시할 수 있는 수준이 된다.

### 3. Prediction Gain의 결과
- **정당성 입증**: 정확한 Prediction Gain을 기반으로 태스크를 선택하고 Averaging SGD를 사용할 경우, 그 결과가 Theorem 1에서 제시한 Minimax 하한선과 일치함을 보였다.
$$G_T(\bar{\theta}_N) \approx \Delta^2_{t^*, T} + \frac{(d\sigma^2_{t^*} + C_5) \log(N)}{N}$$

## 🧠 Insights & Discussion

본 논문은 커리큘럼 러닝이 단순히 '학습의 편의성'을 위한 도구가 아니라, 데이터 할당의 최적화를 통한 '통계적 효율성' 도구임을 이론적으로 증명하였다.

**강점 및 통찰**:
- **Unstructured vs Structured의 대비**: 구조가 없는 문제에서는 적응적 학습의 비용이 크지만, 공유 표현이 존재하는 구조적 문제에서는 적응적 스케줄링이 매우 효율적임을 밝혀, 표현 학습에서 CL의 중요성을 강조하였다.
- **실무 방법론의 이론적 뒷받침**: Prediction Gain 기반의 탐욕적 선택이 통계적으로 최적의 선택과 맞닿아 있음을 증명하여, 휴리스틱한 방법론에 학술적 근거를 제공하였다.

**한계 및 논의사항**:
- **$\sqrt{1/N}$ 의존성**: Structured setting에서 다양성 확보를 위한 하한선과 상한선 사이에 여전히 $d$ 정도의 간극이 존재하며, $\sqrt{1/N}$ 의존성이 불가피하다는 점은 향후 연구 과제로 남았다.
- **정확한 Gain 측정의 어려움**: 이론적으로는 '정확한' Prediction Gain을 가정하지만, 실제 환경에서는 이를 추정(estimation)해야 하며 이 추정 오차가 전체 성능에 미치는 영향은 상세히 다루어지지 않았다.

## 📌 TL;DR

이 논문은 다중 태스크 선형 회귀 모델을 통해 Curriculum Learning(CL)의 통계적 이점을 분석하였다. **Unstructured setting**에서는 최적의 태스크를 적응적으로 찾는 것이 어렵지만, **Structured setting**에서는 OFU 기반 스케줄러를 통해 효율적인 표현 학습이 가능함을 보였다. 또한, 실무에서 쓰이는 **Prediction Gain 기반의 태스크 선택 방식이 통계적 하한선에 도달함**을 증명함으로써 CL의 이론적 토대를 마련하였다. 이 연구는 향후 효율적인 데이터 샘플링 및 태스크 스케줄링 알고리즘 설계에 중요한 가이드라인을 제공할 것으로 기대된다.