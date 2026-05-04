# Data Retrieval with Importance Weights for Few-Shot Imitation Learning

Amber Xie, Rahul Chand, Dorsa Sadigh, Joey Hejna (2025)

## 🧩 Problem to Solve

본 논문은 로봇의 Few-shot Imitation Learning(IL)에서 새로운 환경이나 보지 못한 작업(unseen tasks)에 배포할 때 발생하는 데이터 부족 문제를 해결하고자 한다. 일반적으로 딥러닝 기반의 모방 학습은 단일 작업을 학습하는 데에도 수백에서 수천 개의 전문가 시연(demonstrations)이 필요하며, 이는 새로운 작업을 수행할 때마다 막대한 데이터 수집 비용을 발생시킨다.

이를 해결하기 위해 기존의 Retrieval-based Imitation Learning 방식은 대규모의 사전 데이터셋($D_{prior}$)에서 현재 목표 작업($D_t$)과 유사한 샘플을 추출하여 학습 데이터를 증강하는 방식을 사용한다. 그러나 기존 방식들은 잠재 공간(latent space)에서 target 데이터와 prior 데이터 간의 L2 거리(minimum distance)를 기준으로 데이터를 선택하는 휴리스틱한 방식을 사용해 왔으며, 이는 다음과 같은 두 가지 핵심적인 문제를 야기한다.

첫째, 최근접 이웃(Nearest Neighbor) 추정치는 분산이 매우 높아 노이즈에 취약하다. 둘째, target 데이터의 분포뿐만 아니라 prior 데이터 자체의 분포를 고려하지 않고 데이터를 선택하기 때문에 선택된 데이터의 분포가 편향(bias)될 가능성이 크다. 따라서 본 논문의 목표는 확률론적 관점에서 데이터 추출 과정을 재정의하여, 더 정확하고 강건한 데이터 선택 메커니즘인 Importance Weighted Retrieval(IWR)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 데이터 추출 과정을 **Importance Sampling(중요도 샘플링)**의 관점에서 해석하는 것이다. 단순히 Target 데이터와 가깝다는 이유만으로 데이터를 가져오는 것이 아니라, Target 분포($p_t$)와 Prior 분포($p_{prior}$)의 비율인 **Importance Weight(중요도 가중치)**를 계산하여 데이터를 선택하는 것이다.

이를 위해 저자들은 Gaussian Kernel Density Estimation(KDE)을 도입하였다. Gaussian KDE를 통해 target과 prior의 확률 밀도 함수를 부드럽게 추정함으로써, 단일 최근접 이웃에 의존하는 기존 방식의 고분산 문제를 해결하고, Prior 분포의 밀도를 분모로 둠으로써 Target 분포에 더 근접한 데이터를 효과적으로 추출할 수 있게 설계하였다.

## 📎 Related Works

기존의 retrieval-based IL 연구들인 BehaviorRetrieval, FlowRetrieval, SAILOR, STRAP 등은 주로 어떤 잠재 표현(latent representation)을 학습할 것인가에 집중해 왔다. 이들은 VAE나 skill-based representation learning 등을 통해 state-action 쌍을 인코딩하고, L2 거리를 기반으로 유사한 데이터를 추출한다. 하지만 이러한 방식들은 거리 측정 지표 자체에 대한 이론적 근거가 부족하며 휴리스틱한 설계에 의존한다는 한계가 있다.

또한, Importance Sampling 기법은 언어 모델의 데이터 선택이나 강화 학습의 우선순위 샘플링 등 다른 분야에서는 널리 사용되어 왔으나, 로봇의 Few-shot 모방 학습에서의 데이터 추출 단계에 이를 적용하여 분포의 편향을 해결하려 시도한 연구는 본 논문이 처음이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
IWR의 전체 파이프라인은 다음과 같은 단계로 구성된다.
1. **Representation Learning**: VAE 등을 통해 state-action 쌍을 저차원 잠재 벡터 $z$로 인코딩하는 함수 $f_\phi$를 학습한다.
2. **Importance Weight Estimation**: Gaussian KDE를 사용하여 target 데이터의 분포 $p_t$와 prior 데이터의 분포 $p_{prior}$를 추정한다.
3. **Data Retrieval**: 계산된 중요도 가중치 $p_t / p_{prior}$가 높은 상위 데이터를 $D_{prior}$에서 선택하여 $D_{ret}$을 구성한다.
4. **Policy Learning**: target 데이터 $D_t$와 추출된 데이터 $D_{ret}$을 함께 사용하여 정책 $\pi_\theta$를 co-training 한다.

### 2. 주요 방정식 및 이론적 배경

**훈련 목표 (Weighted Behavior Cloning)**
학습하고자 하는 정책 $\pi_\theta$의 목적 함수는 다음과 같다.
$$\max_{\theta} \alpha \frac{1}{|D_t|} \sum_{(s,a) \in D_t} \log \pi_\theta(a|s) + (1-\alpha) \frac{1}{|D_{ret}|} \sum_{(s,a) \in D_{ret}} \log \pi_\theta(a|s)$$
여기서 $\alpha$는 target 데이터와 retrieved 데이터 간의 가중치 계수(통상 0.5)이다.

**Gaussian KDE를 통한 밀도 추정**
데이터셋 $D$에 대한 확률 밀도 함수 $p^{KDE}(z)$는 다음과 같이 모든 데이터 포인트의 가우시안 합으로 정의된다.
$$p^{KDE}(z) = \frac{1}{|D|} \sum_{z' \in f_\phi(D)} \left( (2\pi)^{d/2} |h^2 \Sigma|^{1/2} \right)^{-1} \exp \left\{ -\frac{1}{2}(z-z')^\top (h^2 \Sigma)^{-1} (z-z') \right\}$$
여기서 $h$는 bandwidth, $\Sigma$는 샘플 공분산 행렬, $d$는 잠재 공간의 차원이다. $h$는 Scott's rule을 기반으로 설정하여 추정치의 분산을 낮추고 부드러운 밀도 함수를 얻는다.

**중요도 가중치 (Importance Weights)**
최종적으로 retrieval을 위한 점수는 다음과 같은 밀도 비율로 결정된다.
$$\text{Weight}(z) = \frac{p_t^{KDE}(z)}{p_{prior}^{KDE}(z)}$$
이 식은 Target 분포의 기댓값을 Prior 샘플을 통해 추정하려는 중요도 샘플링의 원리 $\mathbb{E}_{p_{prior}}[ \frac{p_t}{p_{prior}} \log \pi ] = \mathbb{E}_{p_t} [ \log \pi ]$에 근거한다.

## 📊 Results

### 실험 설정
- **시뮬레이션 환경**: Robomimic Square (10개 target demos), LIBERO-10 (각 작업당 5개 target demos).
- **실제 환경**: Bridge V2 Dataset (Corn, Carrot, Eggplant 작업).
- **비교 대상**: BC (Baseline), Behavior Retrieval (BR), Flow Retrieval (FR), SAILOR (SR).
- **평가 지표**: 성공률(Success Rate, %), long-horizon 작업의 경우 부분 성공률(Partial Success, PS).

### 주요 결과
1. **성능 향상**: IWR은 모든 환경에서 기존 retrieval 방법들보다 일관되게 높은 성능을 보였다. 특히 LIBERO 벤치마크에서 SAILOR 대비 5.8%, Flow Retrieval 대비 4.4%, Behavior Retrieval 대비 5.8%의 평균 성공률 향상을 기록했다.
2. **실제 환경에서의 효과**: Bridge V2 데이터셋을 이용한 실제 로봇 실험에서 IWR의 효과가 더욱 두드러졌으며, 특히 Behavior Retrieval 대비 평균 성공률이 **30% 상승**하였다. Long-horizon 작업인 Eggplant의 경우, IWR은 모든 시도(100%)에서 부분 성공을 거두었으나 다른 방법들은 50%에 그쳤다.
3. **데이터 분포의 최적화**: 분석 결과(Fig 4), 기존 BR 방식은 target과 일부만 겹치는 'Mixed' 작업이나 초기 단계(Reach/Pick up)의 샘플에 편향되어 추출하는 경향이 있었으나, IWR은 Target 작업에 직접적으로 관련된 'Relevant' 데이터를 더 많이 추출하고 전체 타임스텝에 걸쳐 균형 잡힌 데이터를 선택함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
IWR은 기존의 단순 L2 거리 기반 추출 방식이 사실상 Gaussian KDE의 대역폭(bandwidth) $h \to 0$인 특수한 경우(Limit case)임을 수학적으로 증명하였다. 이는 기존 방식이 지나치게 제한적인 가정을 사용했음을 시사한다. IWR은 $p_{prior}$를 분모에 배치함으로써 Prior 데이터셋에 흔하게 존재하는 샘플보다 Target 작업에 상대적으로 더 중요한 샘플을 우선적으로 선택하게 하여, 데이터의 다양성과 관련성을 동시에 확보하였다.

### 한계점 및 비판적 해석
1. **잠재 공간의 특성 의존성**: IWR은 VAE와 같이 L2 smoothness 제약이 있는 잠재 공간에서는 효과적이지만, BYOL과 같이 이러한 제약이 없는 공간에서는 성능 향상이 미미하거나 오히려 저하되는 모습을 보였다. 이는 IWR이 잠재 공간의 기하학적 구조에 의존적임을 의미한다.
2. **차원의 저주**: Gaussian KDE는 잠재 공간의 차원이 높아질 경우 계산 복잡도가 증가하고 수치적으로 불안정해지는 문제가 있다. 따라서 매우 고차원의 표현력을 사용해야 하는 복잡한 작업에서는 적용에 한계가 있을 수 있다.
3. **작업 범위의 제한**: 실험이 주로 Pick-and-place와 같은 단순 조작 작업에 한정되어 있어, 더 정교하고 복잡한 dexterity가 필요한 작업에서의 일반화 성능은 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 Few-shot Imitation Learning에서 Prior 데이터를 추출할 때 사용하는 휴리스틱한 L2 거리 방식을 대체하여, **Target 분포와 Prior 분포의 밀도 비율($p_t / p_{prior}$)**을 기반으로 데이터를 선택하는 **Importance Weighted Retrieval(IWR)**을 제안한다. Gaussian KDE를 통해 계산된 중요도 가중치를 사용하여 데이터 선택의 편향과 분산을 줄였으며, 이는 시뮬레이션과 실제 환경 모두에서 기존 retrieval 기반 방법들보다 월등한 성능 향상을 가져왔다. 특히 기존의 다양한 표현 학습 방법(BR, FR, SR 등)에 플러그인 형태로 쉽게 적용 가능하다는 점에서 실용성이 매우 높다.