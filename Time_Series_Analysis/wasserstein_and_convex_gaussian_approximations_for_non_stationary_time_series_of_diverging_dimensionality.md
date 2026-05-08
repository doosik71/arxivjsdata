# Wasserstein and Convex Gaussian Approximations for Non-stationary Time Series of Diverging Dimensionality

Miaoshiqi Liu, Jun Yang, and Zhou Zhou (2025)

## 🧩 Problem to Solve

본 논문은 고차원 비정상성(High-Dimensional Non-stationary, HDNS) 시계열 분석에서 표본 평균의 분포를 가우시안 분포로 근사하는 Gaussian Approximation (GA)의 한계를 극복하고자 한다.

기존의 고차원 시계열 GA 연구들은 주로 $L_\infty$ 노름(norm)을 통한 최대값 분석이나 하이퍼-렉탱글(hyper-rectangles) 영역에서의 근사에 집중되어 왔다. 그러나 실제 통계적 추론 문제에서 다루는 이차 형식(quadratic forms), U-통계량, 고유값 분석 및 임계값 처리(thresholding)와 같은 문제들은 하이퍼-렉탱글의 범위를 벗어난 훨씬 복잡한 집합 구조를 가진다. 특히 차원 $d$가 표본 크기 $n$과 함께 발산하는 상황에서, 하이퍼-렉탱글보다 훨씬 넓은 범위인 모든 볼록 집합(convex sets)과 2-Wasserstein 거리에서의 가우시안 근사 이론을 정립하는 것이 본 논문의 핵심 목표이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 고차원 비정상성 시계열에 대해 다음과 같은 이론적, 실무적 도구를 제공한 것이다.

1. **범용적 GA 이론 정립**: 하이퍼-렉탱글을 넘어 모든 유클리드 볼록 집합(convex sets)과 2-Wasserstein 거리 하에서의 GA 경계(bound)를 수립하였다. 이는 고차원 비정상성 시계열 분석에서 다룰 수 있는 통계적 추론 문제의 범위를 획기적으로 확장한 것이다.
2. **최적에 가까운 수렴 속도 증명**: 충분히 약한 의존성(weak dependence)과 가벼운 꼬리(light tail) 조건을 만족하는 HDNS 시계열에 대해, 차원 $d$와 시계열 길이 $n$에 대해 거의 최적(nearly optimal)이거나 독립 데이터의 최적 GA 속도와 거의 동일한 수렴 속도를 입증하였다.
3. **Multiplier Bootstrap 절차의 이론적 정당화**: 이론적으로 도출된 GA를 실제 구현하기 위해 고차원 확장형 multiplier bootstrap 절차를 제안하고, 이것이 볼록 집합 상에서 일관되게 작동함을 이론적으로 검증하였다.

## 📎 Related Works

기존의 GA 연구들은 주로 정상성 시계열(stationary time series)이나 하이퍼-렉탱글 영역으로 제한되어 있었다. Zhang and Wu (2017), Zhang and Cheng (2018), Wu and Zhou (2024) 등의 연구가 비정상성 시계열의 GA를 다루었으나, 대부분 $L_\infty$ 노름에 국한되었다. 최근 Chang et al. (2024)이 단순 볼록 집합으로 범위를 확장했으나, 이는 여전히 하이퍼-렉탱글의 변형으로 근사 가능한 수준이었다.

본 연구는 독립 데이터에 대한 볼록 GA 연구(Bentkus 2003, Fang and Koike 2024)와 2-Wasserstein 거리 연구(Eldan et al. 2020)의 성과를 비정상성 시계열 영역으로 확장하였다. 특히, 기존의 비정상성 시계열 GA가 고정된 차원을 가정하거나 매우 제한적인 집합에서만 유효했던 것과 달리, 본 논문은 $d$가 $n$과 함께 발산하는 상황에서도 유효한 이론적 토대를 마련함으로써 기존 연구와의 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 모델 및 의존성 측정

본 논문은 HDNS 시계열 $\{x_i^{[n]}\}$을 시변 베르누이 시프트(time-varying Bernoulli shifts)의 삼각 배열로 모델링한다.
$$x_i^{[n]} := G_i^{[n]}(F_i^{[n]}), \quad i=1, \dots, n$$
여기서 $F_i^{[n]}$은 i.i.d. 랜덤 요소들에 의해 생성된 필트레이션이다. 시계열의 시간적 의존성을 측정하기 위해 다음과 같은 물리적 의존성 척도(physical dependence measure) $\theta_{k,j,q}$를 사용한다.
$$\theta_{k,j,q} := \sup_n \max_{1 \le i \le n} \|G_{i,j}^{[n]}(F_i^{[n]}) - G_{i,j}^{[n]}(F_{i,i-k}^{[n]})\|_q$$
이는 $k$ 단계 전의 혁신(innovation)이 현재 관측치에 미치는 영향력을 측정하며, 이를 통해 시계열의 기억력(memory) 정도를 정량화한다.

### 2. 가우시안 근사(GA) 이론

논문은 두 가지 거리 척도에 대해 GA 경계를 제시한다.

- **2-Wasserstein 거리 ($W_2$)**: 두 확률 분포의 물리적 거리를 제어한다.
$$W_2(X_n/\sqrt{n}, Y_n/\sqrt{n}) = \sqrt{\inf_{\tilde{X} \overset{D}{=} X_n, \tilde{Y} \overset{D}{=} Y_n} E[|\tilde{X}/\sqrt{n} - \tilde{Y}/\sqrt{n}|^2]}$$
- **볼록 변동 거리 (Convex Variation Distance, $d_c$)**: 모든 볼록 집합 $\mathcal{A}$에 대해 확률 차이의 상한을 구한다.
$$d_c(X_n, Y_n) := \sup_{A \in \mathcal{A}} |P(X_n/\sqrt{n} \in A) - P(Y_n/\sqrt{n} \in A)|$$

### 3. Multiplier Bootstrap 절차

실제 구현을 위해 다음과 같은 multiplier bootstrap 통계량 $\tau_{n,L}$을 정의한다.
$$\tau_{n,L} := \frac{n-L+1}{\sqrt{n-L+1}} \sum_{i=1}^{n-L+1} B_i \psi_{i,L}, \quad \psi_{i,L} := \sum_{j=i}^{i+L-1} x_j / \sqrt{L}$$
여기서 $B_i$는 i.i.d. 표준 정규분포를 따르는 랜덤 변수이며, $L$은 사용자가 설정하는 윈도우 크기(bandwidth)이다. 이 절차는 단기 블록 합 $\psi_{i,L}$을 통해 시계열의 상관 구조를 캡처하며, 데이터가 주어졌을 때 $\tau_{n,L}$은 가우시안 벡터가 되므로 $X_n$의 분포를 효과적으로 근사할 수 있다.

## 📊 Results

### 1. 이론적 수렴 속도

- **Wasserstein GA**: 유한 $p$차 모멘트와 다항식 의존성 감소 조건 하에서 $W_2$ 거리는 $O(dn^{1/r-1/2}(\log n))$의 속도로 수렴하며, 지수적 감소 조건 하에서는 $O(dn^{-1/2}(\log n)^5)$까지 빠르게 수렴한다. 이는 $d$가 $o(n^{1/2})$까지 커지더라도 근사가 유효함을 의미한다.
- **Convex GA**: 볼록 집합 상에서의 거리 $d_c$는 $d$가 $O(n^{2/5-\delta})$ 수준으로 발산하더라도 0으로 수렴하며, 이는 독립 데이터에 대해 알려진 최적 속도와 거의 일치한다.

### 2. 실제 적용 사례

- **Combined $L_2$ and $L_\infty$ Inference**: 회귀 계수 $\beta$에 대해 $L_2$ 노름(전반적인 편차)과 $L_\infty$ 노름(특정 차원의 날카로운 편차)을 결합한 테스트 통계량 $T$를 제안하였다. 이 통계량은 볼록 집합(구와 하이퍼-렉탱글의 교집합) 상의 추론 문제로 귀결되며, 제안된 bootstrap을 통해 유효한 임계값을 산출할 수 있음을 보였다.
- **Thresholded Inference**: Soft-thresholding 함수 $\delta_\lambda(x)$를 적용한 추론 문제를 다루었다. 이 함수는 비볼록(non-convex) 특성을 가지므로 Wasserstein GA 이론이 필수적으로 사용되었으며, 이를 통해 고차원 비정상성 시계열에서도 임계값 기반 추론이 가능함을 입증하였다.

### 3. 시뮬레이션 결과

AR 모델, 벡터 AR 모델, 국소 정상성(locally stationary) 모델, 헤비-테일(heavy-tail) 모델 등 5가지 시나리오(M1~M5)에서 테스트를 수행하였다. 실험 결과, 제안된 bootstrap 기반 테스트의 1종 오류(Type I error)가 설정한 유의 수준 $\alpha$와 잘 일치하였으며, 특히 Combined 테스트가 $L_2$와 $L_\infty$ 테스트 각각의 강점을 모두 가지며 강건한 검정력을 보임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 고차원 비정상성 시계열 분석에서 가우시안 근사의 적용 범위를 '하이퍼-렉탱글'에서 '모든 볼록 집합' 및 'Wasserstein 거리'로 확장함으로써 이론적 돌파구를 마련하였다.

**강점 및 의의**:
가장 큰 강점은 이론적 수렴 속도가 독립 데이터의 경우와 거의 차이가 없을 정도로 최적에 가깝다는 점이다. 이는 비정상성으로 인한 복잡성에도 불구하고, 적절한 의존성 조건과 모멘트 조건이 있다면 고차원 데이터에서도 가우시안 근사를 매우 신뢰할 수 있음을 시사한다. 또한, 이론에 그치지 않고 Multiplier Bootstrap이라는 구체적인 구현 방법을 제시하여 실제 통계 분석에 적용 가능하게 하였다.

**한계 및 논의**:
본 논문의 결과는 '충분히 짧은 기억(short memory)'과 '가벼운 꼬리(light tail)'라는 가정에 기반하고 있다. 만약 시계열이 장기 기억(long memory) 특성을 갖거나 매우 두꺼운 꼬리를 가진 분포를 따른다면, 제시된 수렴 속도는 보장되지 않을 수 있다. 또한, 윈도우 크기 $L$의 선택이 실제 성능에 영향을 미칠 수 있으며, 본 논문에서는 plug-in 방식을 사용하였으나 최적의 $L$을 찾는 문제는 여전히 도전적인 과제로 남아 있다.

## 📌 TL;DR

본 연구는 차원이 표본 크기와 함께 증가하는 고차원 비정상성(HDNS) 시계열에 대해, 모든 볼록 집합과 2-Wasserstein 거리 상에서의 가우시안 근사(GA) 이론을 정립하였다. 특히 독립 데이터 수준의 최적 수렴 속도를 증명하였으며, 이를 실무적으로 구현할 수 있는 multiplier bootstrap 절차를 제안하였다. 이 결과는 고차원 시계열의 결합 노름 추론 및 임계값 기반 추론과 같이 기존의 하이퍼-렉탱글 기반 GA로는 해결할 수 없었던 복잡한 통계적 문제들을 해결할 수 있는 강력한 이론적 도구를 제공한다.
