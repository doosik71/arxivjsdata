# Weighted Meta-Learning

Diana Cai, Rishit Sheth, Lester Mackey, Nicolo Fusi (2020)

## 🧩 Problem to Solve

본 논문은 Meta-learning, 특히 Gradient-based meta-learning 알고리즘들이 가지고 있는 핵심적인 가정의 한계를 해결하고자 한다. Model-Agnostic Meta-Learning(MAML)과 같은 기존의 대중적인 알고리즘들은 모든 소스 작업(source tasks)들이 타겟 작업(target task)과 동일한 분포에서 추출되었다고 가정하며, 메타 학습 과정에서 모든 소스 작업에 동일한 가중치를 부여하여 균등하게 샘플링한다.

그러나 실제 환경에서는 타겟 작업이 모든 소스 작업과 유사한 것이 아니라, 단 몇 개의 소스 작업과만 유사하거나 심지어 단 하나의 작업과만 유사할 가능성이 높다. 이러한 상황에서 모든 소스 작업에 동일한 가중치를 적용하는 것은 오히려 성능을 저하시키는 요인이 된다. 따라서 본 연구의 목표는 타겟 샘플을 학습 과정에 직접 활용하여 소스 작업들의 가중치를 동적으로 조절함으로써, 타겟 작업에 최적화된 초기 모델(initialization)을 찾는 일반적인 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 소스 작업들의 손실 함수(loss)에 가중치를 부여하되, 이 가중치가 타겟 샘플에 의존하도록 설계하는 것이다. 즉, 타겟 작업과 더 유사한 소스 작업에 더 높은 가중치를 부여하여 메타 학습을 진행함으로써 타겟 작업에 더 빠르게 적응할 수 있는 초기화를 학습하는 것이다.

이를 위해 저자들은 Integral Probability Metric(IPM)을 도입하여 소스 분포와 타겟 분포 간의 유사도를 측정하고, 이를 기반으로 기대 타겟 리스크(expected target risk)의 상한선(upper bound)을 수학적으로 증명하였다. 또한, 이론적으로 도출된 상한선을 최소화하는 계산 가능한 알고리즘인 $\alpha$-MAML과 $\alpha$-ERM을 제안하여 실증적으로 그 효용성을 입증하였다.

## 📎 Related Works

기존의 Gradient-based meta-learning 연구들은 주로 온라인 볼록 최적화(online convex optimization) 관점에서 보장(guarantees)을 제공하거나, 모델 파라미터 간의 거리로 작업 유사도를 정의하였다. 또한, MAML의 확장판들이 계층적 작업 분포나 비정상성(non-stationarity)을 다루려 시도했으나, 이는 대개 복잡한 메타 학습기를 설계하는 방향이었다.

Domain Adaptation 분야에서는 $H$-divergence나 population IPM을 사용하여 여러 소스 도메인을 결합하는 연구가 진행되었으며, 일부 연구는 task embedding distance와 같은 프록시 지표를 통해 유사도를 측정했다. 본 논문은 이러한 기존 방식과 달리, 모델 클래스와 손실 함수가 결합된 형태의 IPM을 사용하여 '성능' 관점에서의 작업 유사도를 명시적으로 정의하고, 이를 통해 계산 가능한 커널 거리(kernel distance)로 연결했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 목적 함수
본 논문은 $J$개의 소스 작업 $\{\mathcal{Z}^{(j)}\}_{j=1}^J$과 하나의 타겟 작업 $\mathcal{Z}^T$가 존재한다고 가정한다. 각 작업의 데이터는 $\hat{S}^{(j)}$(소스)와 $\hat{T}$(타겟)라는 경험적 분포로 표현된다. 이때 가중치 $\alpha \in \Delta^{J-1}$ (단, $\sum \alpha_j = 1$)를 사용하여 가중치가 적용된 소스 혼합 분포 $\hat{S}_\alpha$를 다음과 같이 정의한다.
$$\hat{S}_\alpha := \sum_{j=1}^J \alpha_j \hat{S}^{(j)}$$

최종적인 메타 학습의 목적 함수는 다음과 같이 정의된다.
$$\sum_{j=1}^J \alpha_j \mathbb{E}_{\hat{S}^{(j)}} g(z)$$
여기서 $g$는 모델 클래스와 손실 함수가 결합된 함수이다. Joint Training의 경우 $\alpha_j = 1/J$이며 $g(x,y) = \ell(y, f(x; \theta))$가 된다. MAML의 경우 $\alpha_j = 1/J$이며 $g(x,y) = \ell(y, f(x; U(\theta)))$가 되는데, 여기서 $U(\theta)$는 빠른 적응(fast adaptation)을 위한 업데이트 함수이다.

### 2. 이론적 배경: IPM 및 일반화 오차 상한
두 분포 $P$와 $Q$ 사이의 Integral Probability Metric(IPM)은 함수 클래스 $\mathcal{G}$에 대해 다음과 같이 정의된다.
$$\gamma_{\mathcal{G}}(P, Q) := \sup_{g \in \mathcal{G}} | \mathbb{E}_P g(z) - \mathbb{E}_Q g(z) |$$

본 논문의 핵심 이론인 Theorem 3.4는 $\alpha$-혼합 소스 분포의 경험적 리스크와 타겟 리스크 사이의 거리 상한을 다음과 같이 제시한다.
$$\gamma_{\mathcal{G}}(\hat{S}_\alpha, T) \le \gamma_{\mathcal{G}}(\hat{S}_\alpha, \hat{T}) + 2R(\mathcal{G} | z_1, \dots, z_{N(T)}) + 3\sqrt{\frac{(b-a)^2 \log(2/\epsilon)}{2N(T)}}$$
여기서 $R(\mathcal{G} | \dots)$는 타겟 샘플에 대한 경험적 Rademacher complexity를 의미한다. 이 식은 타겟 리스크를 최소화하기 위해서는 경험적 IPM $\gamma_{\mathcal{G}}(\hat{S}_\alpha, \hat{T})$를 최소화하는 $\alpha$를 찾아야 함을 시사한다.

### 3. 계산 가능한 알고리즘: Kernel Distance 활용
IPM은 일반적인 함수 클래스 $\mathcal{G}$에 대해 계산이 불가능하므로, 저자들은 Reproducing Kernel Hilbert Space(RKHS)의 단위 볼 $\mathcal{G}_{RKHS}$를 사용하여 이를 근사한다. $\mathcal{G} \subseteq \mathcal{G}_{RKHS}$ 조건을 만족하는 커널 $k$를 찾으면, IPM은 Maximum Mean Discrepancy(MMD)인 커널 거리 $\gamma_k(\hat{S}_\alpha, \hat{T})$로 상한이 결정된다.

구체적으로 제곱 손실(Square loss)과 힌지 손실(Hinge loss)에 대해 $\mathcal{G} \subseteq \mathcal{G}_{RKHS}$를 만족하는 특성 맵(feature map) $\phi$를 다음과 같이 설계하였다.
- **Square loss**: $\phi((\psi(x), y)) = [\text{vec}(\psi(x)\psi(x)^\top), \sqrt{2}y\psi(x), y^2]^\top$
- **Hinge loss**: $\phi((\psi(x), y)) = [y\psi(x), 1]^\top$

이를 통해 가중치 $\alpha$를 찾는 문제는 다음과 같은 이차 계획법(Quadratic Program) 문제로 변환되어 효율적으로 해결된다.
$$\hat{\alpha} := \arg \min_{\alpha \in \Delta^{J-1}} \gamma_k(\hat{S}_\alpha, \hat{T})$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 합성 선형 회귀(Synthetic linear regression), 사인파 회귀(Sine wave regression), 실제 데이터셋(Diabetes, Boston house prices, Sales data).
- **비교 대상**: MAML, Joint Training(ERM), $\alpha$-MAML, $\alpha$-ERM, Threshold method(가장 유사한 단일 소스만 선택).
- **평가 지표**: RMSE (Root Mean Squared Error).

### 2. 주요 결과
- **합성 사인파 회귀**: $\alpha$-MAML은 균등 가중치 MAML보다 훨씬 적은 수의 gradient step(약 100회 vs 더 많은 횟수)만으로도 타겟 데이터에 빠르게 적응하였다. 특히 10-shot 시나리오에서 $\alpha$-MAML의 RMSE가 MAML보다 낮게 나타났다.
- **실제 데이터(Diabetes, Boston)**: $\alpha$-weighted 방식이 균등 가중치 방식보다 일관되게 낮은 RMSE를 기록하였다. 분석 결과, 타겟 작업의 연령대와 유사한 소스 작업들에 더 높은 가중치가 부여되었음을 확인하였다.
- **판매 데이터(Sales data)**: 타겟 데이터만으로 학습했을 때는 성능이 매우 낮았으나, $\alpha$-MAML과 $\alpha$-ERM을 적용했을 때 RMSE가 비약적으로 감소하였다(MAML $\approx 12.69 \rightarrow \alpha\text{-MAML} \approx 2.41$). 이는 타겟 제품과 유사한 패턴을 가진 소스 제품들을 효과적으로 가중치 부여했기 때문이다.

## 🧠 Insights & Discussion

본 연구는 Meta-learning에서 막연하게 가정하던 '작업 분포의 동일성' 문제를 수학적으로 정의하고, 이를 IPM이라는 도구를 통해 해결함으로써 이론적 근거와 실무적 알고리즘을 동시에 제공하였다. 특히 유사도를 단순한 임베딩 거리가 아닌, 모델의 성능(loss) 관점에서 정의하여 RKHS로 연결한 점이 매우 정교하다.

**강점**:
- 단순한 휴리스틱이 아니라 Rademacher complexity와 IPM을 이용한 일반화 오차 상한선을 통해 가중치 최적화의 당위성을 부여하였다.
- MAML과 같은 기존 구조를 그대로 유지하면서 가중치 최적화 단계만 추가하면 되므로 적용 가능성이 높다.

**한계 및 논의**:
- 제안된 커널 구성은 제곱 손실과 힌지 손실에 국한되어 있으며, 다중 클래스 분류나 더 복잡한 손실 함수로의 확장은 추가적인 연구가 필요하다.
- 타겟 샘플을 학습 과정에 직접 사용하므로, 타겟 샘플의 수가 극단적으로 적을 경우 $\alpha$ 값의 추정이 불안정해질 가능성이 있다.
- 직접적으로 일반화 상한선을 최적화하는 방식(Direct bound optimization)이 제안되었으나, 본 논문의 2단계 알고리즘(가중치 먼저 결정 후 모델 학습)보다 수렴 속도가 느리다는 점이 관찰되었다.

## 📌 TL;DR

이 논문은 모든 소스 작업을 동일하게 취급하는 기존 Meta-learning의 한계를 극복하기 위해, 타겟 작업과의 유사도(IPM 기반)에 따라 소스 작업의 가중치를 다르게 부여하는 **Weighted Meta-Learning** 프레임워크를 제안한다. 커널 거리를 이용하여 계산 가능한 가중치 최적화 알고리즘($\alpha$-MAML, $\alpha$-ERM)을 구현하였으며, 이를 통해 합성 및 실제 회귀 데이터셋에서 기존 MAML보다 훨씬 빠른 적응 속도와 더 낮은 오차를 달성하였다. 이 연구는 작업 간의 이질성이 큰 실제 환경에서 Meta-learning을 적용하는 데 중요한 이론적/실천적 토대를 제공한다.