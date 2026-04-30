# ACTIVE LEARNING BY QUERY BY COMMITTEE WITH ROBUST DIVERGENCES

Hideitsu Hino, Shinto Eguchi (2022)

## 🧩 Problem to Solve

본 논문은 지도 학습(Supervised Learning)에서 발생하는 높은 데이터 라벨링 비용 문제를 해결하기 위한 Active Learning(능동 학습), 그 중에서도 Query by Committee(QBC) 알고리즘의 취약성을 개선하는 것을 목표로 한다.

QBC는 여러 개의 예측 모델(Committee)을 구성하고, 이들 간의 의견 불일치(Disagreement)가 가장 큰 샘플을 선택하여 라벨링하는 방식이다. 기존의 QBC 방식은 의견 불일치를 측정하기 위해 Kullback–Leibler(KL) Divergence를 주로 사용한다. 그러나 KL Divergence는 이상치(Outlier)에 매우 민감하다는 단점이 있다. 특히 Active Learning의 초기 단계에서는 학습 데이터가 매우 적기 때문에, 일부 커미티 멤버가 매우 부정확한 예측을 수행하는 이상치로 작용할 가능성이 높으며, 이는 잘못된 샘플 선택으로 이어져 전체 학습 효율을 떨어뜨린다.

따라서 본 연구의 목표는 KL Divergence를 대체하여 이상치에 강건한(Robust) Divergence 척도를 도입함으로써, 커미티 멤버의 신뢰도가 낮은 상황에서도 안정적으로 최적의 샘플을 선택할 수 있는 강건한 QBC 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 정보 기하학(Information Geometry)의 관점에서 KL Divergence를 Bregman Divergence의 특수 사례로 파악하고, 이를 확장하여 이상치에 강건한 $\beta$-divergence와 dual $\gamma$-power divergence를 도입하는 것이다.

주요 기여 사항은 다음과 같다:
1. **강건한 Divergence 도입**: KL Divergence 대신 $\beta$-divergence 및 dual $\gamma$-power divergence를 사용하여 커미티의 합의 모델(Consensus Model)과 획득 함수(Acquisition Function)를 재정의하였다.
2. **이론적 강건성 증명**: 영향 함수(Influence Function, IF) 분석을 통해, KL Divergence 기반의 방법은 이상치에 대해 IF가 유계되지 않지만(unbounded), 제안된 $\beta$ 및 $\gamma$ 기반 방법은 IF가 유계됨(bounded)을 수학적으로 증명하여 이론적 강건성을 입증하였다.
3. **명시적 합의 모델 제시**: Bregman Divergence 기반의 $u$-mixture와 dual $\gamma$-power divergence 기반의 dual $\gamma$-mixture 모델을 정의하여, 실제 계산 가능한 형태의 합의 모델을 도출하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 배경으로 한다:
- **Active Learning 및 QBC**: 데이터 라벨링 비용을 줄이기 위해 불확실성이 높은 샘플을 선택하는 기법이다. 특히 QBC는 여러 모델의 합의 과정을 통해 불확실성을 측정한다.
- **정보 기하학(Information Geometry)**: 확률 분포의 매개변수 공간을 기하학적으로 분석하는 방법론으로, Divergence 함수는 두 분포 사이의 거리나 차이를 측정하는 핵심 도구로 사용된다.
- **강건 통계학(Robust Statistics)**: 이상치가 포함된 데이터셋에서도 안정적인 추정치를 얻기 위한 연구 분야이다.

기존의 QBC 접근 방식은 계산의 편의성과 이론적 기초 때문에 KL Divergence에 의존하였으나, 이는 통계학적으로 이상치에 취약하다는 한계가 있다. 본 논문은 이를 해결하기 위해 강건 통계학에서 사용되는 Power Divergence 개념을 Active Learning의 샘플 선택 과정에 접목했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
본 논문은 Generalized Linear Model(GLM) 프레임워크 내에서 다음과 같은 순차적 Active Learning 절차를 따른다:
1. 초기 학습 데이터 $S_0$를 사용하여 커미티 멤버 $\theta^{c, t-1} (c=1, \dots, C)$를 학습시킨다.
2. 제안된 강건한 Divergence를 사용하여 합의 모델 $p_{robust}$를 생성한다.
3. 획득 함수 $a_{robust}(x)$를 최대화하는 샘플 $x_t$를 풀(Pool) 데이터셋 $X_p$에서 선택한다.
4. 선택된 $x_t$에 대한 라벨 $y_t$를 획득하여 데이터셋을 업데이트하고 모델을 재학습한다.

### Generalized Linear Model (GLM) 설정
출력 변수 $y$의 확률 밀도 함수는 다음과 같은 지수 가족(Exponential Family) 형태로 정의된다:
$$p(y|\xi(x)) = \exp\left(\frac{y\xi(x) - \psi(\xi(x))}{\phi} + c(y, \phi)\right)$$
여기서 $\xi(x) = \langle \theta, x \rangle$는 선형 예측자이며, $\psi$는 cumulant generating function, $\phi$는 분산 매개변수이다.

### 강건한 Divergence 및 합의 모델
#### 1. $\beta$-divergence 및 $u$-mixture
$\beta$-divergence는 Bregman Divergence의 일종으로, 다음과 같이 정의된다:
$$D_\beta(p, q) = \frac{1}{\beta(1 + \beta)} \int p^{\beta+1} d\Lambda - \frac{1}{\beta} \int pq^\beta d\Lambda + \frac{1}{\beta+1} \int q^{\beta+1} d\Lambda$$
이를 이용한 합의 모델($u$-mixture) $p^u$는 다음과 같다:
$$p^u(y|x; w) = \left[ \sum_{c=1}^C w_c \exp\left(\beta \frac{y\xi^c(x) - \psi(\xi^c(x))}{\phi} + \beta c(y, \phi)\right) - \beta b(x) \right]^{1/\beta}$$
여기서 $b(x)$는 정규화 상수이다.

#### 2. dual $\gamma$-power divergence 및 dual $\gamma$-mixture
$\beta$-divergence의 정규화 문제를 해결하기 위해, 첫 번째 인자에 대해 스케일 불변성을 갖는 dual $\gamma$-power divergence를 도입한다. 이에 따른 합의 모델은 다음과 같이 명시적으로 정의된다:
$$p_\gamma(y; w) = \frac{1}{z(w)} \left( \sum_{c=1}^C w_c p_c(y)^\gamma \right)^{1/\gamma}, \quad z(w) = \int \left( \sum_{c=1}^C w_c p_c(y)^\gamma \right)^{1/\gamma} d\Lambda(y)$$

### 강건한 획득 함수 (Acquisition Function)
합의 모델 $p_{robust}$ (여기서는 $p^u$ 또는 $p_\gamma$)를 사용하여, 커미티 멤버들과의 불일치 정도를 측정하는 획득 함수를 정의한다:
- $\beta$-divergence 기반: $a_\beta(x; w) = \sum_{c=1}^C w_c D_\beta(p(Y|\xi^c(x)), p^u(Y|x))$
- $\gamma$-divergence 기반: $a_\gamma(x; w) = \sum_{c=1}^C w_c D^*_\gamma(p(Y|\xi^c(x)), p_\gamma(Y|x))$

이 함수 값이 큰 샘플일수록 커미티 간의 의견 차이가 크므로, 정보 가치가 높다고 판단하여 선택한다.

## 📊 Results

### 실험 설정
- **모델**: Logistic Regression
- **커미티 구성**: $C=10$ (Bagging 등을 통해 생성), 가중치 $w_c = 1/10$
- **비교 대상**: Random Sampling, KL Divergence 기반 QBC, $\beta$-divergence 기반 QBC ($\beta=1.0$), dual $\gamma$-power divergence 기반 QBC ($\gamma=1.0$)
- **데이터셋**: 인공 데이터셋 1종 및 LIBSVM 실세계 데이터셋 6종 (Adult, Breast-cancer, Diabetes, Mushrooms, IJCNN, Titanic)
- **지표**: 획득한 데이터 수에 따른 예측 오차(Prediction Error) 및 영향 함수(IF) 값

### 정량적 및 정성적 결과
1. **예측 성능**: 대부분의 데이터셋에서 $\beta$ 및 $\gamma$ 기반 방법이 KL 기반 방법보다 더 적은 샘플로 더 낮은 예측 오차를 달성하였다. 특히 Active Learning의 **초기 단계**에서 KL 기반 방법은 불안정한 모습을 보였으나, 제안 방법들은 안정적으로 오차를 감소시켰다.
2. **강건성 확인**: 영향 함수(IF)를 측정한 결과, KL 기반 방법은 이상치 $\xi_{out}$에 대해 IF 값이 매우 높게 나타난 반면, $\beta$ 및 $\gamma$ 기반 방법은 IF 값이 낮게 유지되며 유계(bounded)됨을 확인하였다. 이는 이론적 예측과 일치한다.
3. **특이 사항**: Adult 데이터셋의 경우 $\beta$-divergence 기반 방법의 성능이 다소 낮게 나타났으며, 이는 강건한 Divergence의 하이퍼파라미터 설정이 데이터셋 특성에 따라 영향을 줄 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 이론적 해석
본 논문은 단순한 알고리즘 제안을 넘어 정보 기하학적 분석을 통해 강건성의 근거를 제시하였다. 특히, 지수 가족 모델(Exponential Family)과 강건한 Divergence 간의 '불일치'가 오히려 이상치에 대한 내성을 제공한다는 통찰을 제시한다. 즉, 모델의 구조와 추정 방식이 서로 다른 기하학적 구조를 가질 때 이상치의 영향력이 억제된다는 점이 핵심이다.

### 한계 및 미해결 과제
- **파라미터 선택**: $\beta$나 $\gamma$ 값을 어떻게 최적으로 선택할 것인가에 대한 명확한 기준이 제시되지 않았다. 이는 향후 연구 과제로 남아 있다.
- **적응적 선택**: 학습 초기에는 강건한 Divergence를 사용하고, 충분한 데이터가 수집되어 모델이 안정화된 후에는 효율적인 KL Divergence로 전환하는 적응적(Adaptive) 전략의 필요성이 제기되었다.
- **계산 복잡도**: $\beta$-divergence 기반 합의 모델의 경우 정규화 상수 $b(x)$의 계산이 분석적으로 어려워 수치적 근사가 필요하다.

## 📌 TL;DR

본 논문은 Active Learning의 QBC 알고리즘이 KL Divergence 사용으로 인해 이상치(Outlier)에 취약하다는 점을 지적하고, 이를 해결하기 위해 **$\beta$-divergence와 dual $\gamma$-power divergence를 도입한 강건한 QBC**를 제안한다. 영향 함수(Influence Function) 분석을 통해 이론적 강건성을 증명하였으며, 다양한 데이터셋 실험을 통해 특히 학습 초기 단계에서 기존 KL 기반 QBC보다 더 안정적이고 우수한 성능을 보임을 입증하였다. 이 연구는 정보 기하학을 통해 Active Learning의 샘플 선택 과정을 최적화하고 강건성을 확보하는 새로운 방향성을 제시한다.