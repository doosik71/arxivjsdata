# Querying Easily Flip-Flopped Samples for Deep Active Learning

Seong Jin Cho, Gwangsu Kim, Junghyun Lee, Jinwoo Shin, Chang D. Yoo (2024)

## 🧩 Problem to Solve

Active Learning (AL)의 핵심은 모델의 성능을 전략적으로 향상시키기 위해 가장 정보량이 많은(informative) 미라벨링 데이터를 선택하여 쿼리하는 것이다. 기존의 많은 AL 알고리즘은 모델의 예측 불확실성(predictive uncertainty)을 활용하며, 직관적으로 데이터 샘플이 결정 경계(decision boundary)에 가까울수록 불확실성이 높다고 판단한다.

그러나 딥러닝 기반의 다중 클래스 분류(multiclass classification) 작업에서 형성되는 복잡한 결정 경계와 샘플 사이의 거리를 직접적으로 계산하는 것은 계산적으로 불가능(intractable)하다. 따라서 본 논문은 기존의 유클리드 거리 기반 접근법에서 벗어나, 결정 경계와의 근접성을 측정하는 새로운 지표를 제안하여 효율적이고 이론적 근거가 확실한 샘플 선택 전략을 구축하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 **Least Disagree Metric (LDM)**이라는 새로운 불확실성 측정 지표를 도입하는 것이다. LDM은 특정 샘플의 라벨이 모델의 작은 섭동(perturbation)에 의해 얼마나 쉽게 바뀔 수 있는가(flip-flopped)를 통해 결정 경계와의 거리를 측정한다.

주요 기여 사항은 다음과 같다:

- 샘플의 결정 경계 근접성을 측정하기 위한 LDM의 정의 및 이론적 정립.
- 파라미터 섭동(parameter perturbation)을 통해 LDM을 효율적으로 계산할 수 있는 점근적 일관성(asymptotically consistent)을 가진 추정기(estimator) 제안.
- LDM의 불확실성과 배치 다양성(batch diversity)을 동시에 고려한 AL 알고리즘인 **LDM-S** 제안 및 다양한 벤치마크 데이터셋에서의 SOTA 성능 입증.

## 📎 Related Works

기존의 AL 연구들은 주로 다음과 같은 접근 방식을 취했다:

- **Uncertainty Sampling**: Entropy나 Margin 기반 방법이 대표적이다. 단순하고 효율적이지만, 많은 경우 견고한 이론적 기초가 부족한 휴리스틱 근사치에 의존하며, 특히 다중 클래스 분류에서 Entropy는 덜 중요한 클래스의 확률에 과도하게 영향을 받는 한계가 있다.
- **Bayesian Active Learning**: MC-dropout을 사용하는 DBAL이나 BatchBALD 등이 있다. 하지만 계산 비용이 높거나 쿼리 사이즈가 커질 경우 적절하지 않은 경우가 많다.
- **Diversity-based Approach**: Coreset이나 BADGE와 같이 데이터의 분포나 그래디언트 임베딩의 다양성을 추구하는 방법이다. 하지만 이는 순수한 불확실성 측정과는 거리가 있을 수 있다.

본 논문은 거리 기반의 불확실성을 측정하되, 계산 불가능한 유클리드 거리 대신 모델 파라미터 공간에서의 '불일치 확률'이라는 관점에서 접근함으로써 기존 방법론들과 차별화된다.

## 🛠️ Methodology

### 1. Least Disagree Metric (LDM) 정의

LDM은 두 가설 $h_1, h_2$ 사이의 불일치 측정치(disagree metric) $\rho$에서 출발한다. $\rho$는 전체 데이터 분포 $D_X$에 대해 두 모델의 예측이 서로 다를 확률로 정의된다:
$$\rho(h_1, h_2) := P_{X \sim D_X}[h_1(X) \neq h_2(X)]$$

특정 가설 $g$와 샘플 $x_0$에 대해, $x_0$에서의 예측값이 $g(x_0)$와 다른 모든 가설의 집합을 $H_{g, x_0}$라고 할 때, **LDM**은 다음과 같이 정의된다:
$$L(g, x_0) := \inf_{h \in H_{g, x_0}} \rho(h, g)$$
즉, $x_0$의 라벨을 바꿀 수 있는 가장 '가까운'(불일치 확률이 가장 낮은) 가설과의 거리라고 볼 수 있다. LDM 값이 작을수록 해당 샘플은 결정 경계에 매우 가깝고 불확실성이 높음을 의미한다.

### 2. LDM 추정기 (Estimator)

실제 환경에서는 $\rho$를 계산하기 위한 데이터 분포를 모르고 가설 집합 $H$가 무한하므로, 다음과 같은 근사 추정기 $L^{N,M}$를 제안한다:

- **Monte-Carlo 방법**: 확률 $P$를 $M$개의 샘플을 이용한 경험적 확률로 대체한다.
- **Finite Hypothesis Set**: 무한한 $H$를 $N$개의 유한한 가설 집합 $H_N$으로 대체한다.

$$L^{N,M}(g, x_0) := \inf_{h \in H_{g, x_0}^N} \left( \frac{1}{M} \sum_{i=1}^M I[h(X_i) \neq g(X_i)] \right)$$

실제 구현에서는 SGD로 학습된 모델 $g$의 파라미터 $v$를 중심으로 가우시안 섭동(Gaussian perturbation) $w \sim \mathcal{N}(v, I\sigma^2)$을 주어 가설 $h$를 샘플링한다. 이때 $\sigma^2$의 범위를 다양하게 설정하여 LDM을 최적화한다.

### 3. LDM-S: Active Learning 알고리즘

단순히 LDM이 가장 작은 샘플들만 선택하면 정보의 중복이 발생하는 샘플링 편향(sampling bias)이 나타난다. 이를 해결하기 위해 **LDM-Seeding** 전략을 사용한다.

1. **가중치 계산**: 풀(pool) $P$를 LDM이 작은 집합 $P_q$와 나머지 $P_c$로 나누고, 지수적으로 감쇠하는 가중치 $\gamma_x$를 부여한다:
   $$\eta_x = (L_x - L_q)_+ / L_q, \quad \gamma_x = \frac{e^{-\eta_x}}{\sum e^{-\eta_{x_j}}}$$
2. **다양성 고려 샘플링**: 첫 번째 샘플은 LDM이 가장 작은 샘플로 선택하고, 이후 샘플 $x_n$은 다음 확률 분포에 따라 선택한다:
   $$P(x) \propto \gamma_x^2 \cdot \min_{x' \in Q_{n-1}} d_{\cos}(z_x, z_{x'})$$
   여기서 $d_{\cos}$는 마지막 레이어 피처 간의 코사인 거리이다. 즉, **불확실성(LDM)**과 **다양성(Cosine Distance)**을 동시에 고려하여 배치를 구성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: OpenML 3종, MNIST, CIFAR10/100, SVHN, Tiny ImageNet, FOOD101, ImageNet 등 매우 광범위한 데이터셋 사용.
- **모델 아키텍처**: MLP, S-CNN, K-CNN, Wide-ResNet, ResNet-18 등.
- **비교 대상**: Random, Entropy, Margin, Coreset, ProbCov, DBAL, ENS, BADGE, BAIT.

### 주요 결과

- **성능 우위**: Dolan-Moré plot(성능 프로필)과 Penalty Matrix 분석 결과, LDM-S가 거의 모든 데이터셋과 아키텍처에서 다른 알고리즘들을 압도하거나 대등한 성능을 보였다.
- **효율성**: 계산 시간 측면에서 Entropy 기반 방법과 유사한 수준(약 3~6% 증가)으로 매우 효율적이며, 특히 BAIT나 Ensemble 방법보다 훨씬 빠르다.
- **강건성**: 배치 사이즈의 변화(1부터 5,000까지)에도 불구하고 일관되게 높은 성능을 유지하여 배치 크기에 대한 강건함이 입증되었다.

## 🧠 Insights & Discussion

- **LDM의 유효성**: 이론적으로 LDM은 결정 경계까지의 거리와 밀접한 관련이 있으며, 실험적으로도 LDM 순서와 예측 불확실성 순서 사이에 강한 음의 상관관계가 있음이 확인되었다.
- **다양성의 필수성**: LDM이 매우 작은 샘플들만 선택했을 때 오히려 성능이 저하되는 구간이 발견되었다(Figure 3). 이는 결정 경계 근처의 특정 영역에 샘플이 뭉치는 현상 때문이며, 이를 해결하기 위한 Seeding 전략이 필수적임을 시사한다.
- **구현의 실용성**: LDM 추정 시 정지 조건 $s$를 작게 설정하더라도 샘플 간의 상대적 순위(rank order)가 잘 유지된다는 점을 발견하였다. 이는 실제 구현 시 계산 비용을 획기적으로 줄이면서도 효과적으로 샘플을 선택할 수 있음을 의미한다.

## 📌 TL;DR

본 논문은 모델 파라미터의 작은 변화에 의해 라벨이 쉽게 바뀌는 샘플이 가장 불확실하다는 직관을 바탕으로 **Least Disagree Metric (LDM)**을 제안하였다. 이를 효율적으로 계산하기 위한 가우시안 섭동 기반 추정기와, 다양성을 보완한 **LDM-S** 알고리즘을 통해 다양한 이미지 및 정형 데이터셋에서 SOTA 성능을 달성하였다. 특히, 이론적 근거를 갖추면서도 계산 효율성이 매우 높아 실제 대규모 딥러닝 기반 Active Learning 시스템에 적용 가능성이 매우 높다.
