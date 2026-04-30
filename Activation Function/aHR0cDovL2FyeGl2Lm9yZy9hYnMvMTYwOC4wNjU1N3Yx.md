# Neural Networks with Smooth Adaptive Activation Functions for Regression

Le Hou, Dimitris Samaras, Tahsin M. Kurc, Yi Gao, Joel H. Saltz (2016)

## 🧩 Problem to Solve

본 논문은 회귀(Regression) 문제에 적용되는 신경망(Neural Networks, NN)의 모델 편향(Model Bias)을 줄이면서도 과적합(Overfitting)을 방지할 수 있는 새로운 활성화 함수를 제안한다.

일반적으로 회귀 문제는 분류(Classification) 문제와 달리 출력값이 넓은 범위의 실수값을 가져야 하므로, 고정된 구조의 활성화 함수를 사용하는 신경망은 상대적으로 더 큰 편향을 가지는 경향이 있다. 이를 해결하기 위해 학습 가능한 파라미터를 가진 적응형 활성화 함수(Adaptive Activation Functions, AAF)를 사용할 수 있으나, 기존의 AAF들은 다음과 같은 한계를 가진다.
1. **단순한 AAF**: 표현력이 부족하여 최적의 활성화 함수를 충분히 근사하지 못하며, 이는 높은 모델 편향으로 이어진다.
2. **복잡한 AAF**: 표현력은 높으나 파라미터 수가 급격히 증가하여 심각한 과적합 문제를 야기하며, 이를 체계적으로 제어할 방법이 부족하다.

따라서 본 연구의 목표는 낮은 모델 편향과 제어 가능한 모델 복잡도(Model Complexity)를 동시에 달성할 수 있는 **Smooth Adaptive Activation Function (SAAF)**을 설계하고, 이를 회귀 신경망의 마지막 전 단계인 회귀 레이어(Regression Layer)에 적용하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 조각마다 다항식(Piecewise Polynomial) 형태를 띠는 SAAF를 설계하여, 함수 근사 능력은 극대화하되 $\text{L}_2$ 규제(Regularization)를 통해 함수의 매끄러움(Smoothness)과 모델 복잡도를 이론적으로 보장하는 것이다.

주요 기여 사항은 다음과 같다.
- **SAAF 제안**: 임의의 1차원 연속 함수를 원하는 정밀도로 근사할 수 있으면서, 파라미터 크기를 제한함으로써 모델 복잡도를 낮게 유지할 수 있는 SAAF를 제안하였다.
- **이론적 분석**: Lipschitz 연속성(Lipschitz continuity)을 기반으로 회귀 모델의 모델 복잡도를 측정하는 지표인 $\text{fat-shattering dimension}$의 상한선(Upper-bound)을 증명하였다.
- **회귀 레이어 적용**: SAAF를 신경망의 모든 레이어가 아닌 회귀 레이어(second-to-last layer)에만 적용함으로써, 적은 파라미터 증가만으로도 효율적으로 편향을 줄일 수 있음을 보였다.

## 📎 Related Works

기존의 AAF 연구들은 크게 두 가지 방향으로 나뉜다.
1. **사전 정의된 함수 기반**: Sigmoid나 Exponential 함수의 기울기 등을 조정하는 방식이다. 수렴 속도는 빠르나 형태의 제약이 많아 근사 능력이 떨어진다.
2. **복잡한 파라미터화 방식**: Splines나 조각별 선형/이차 함수 등을 사용하는 방식이다. 표현력은 좋으나 훈련 과정이 복잡하고, 불연속성(Discontinuity)이나 미분 불가능성 등의 문제가 발생하며, 과적합을 막기 위한 체계적인 원칙이 부족하다.

특히 PReLU나 Maxout, APLU와 같은 최신 기법들이 분류 작업에서 우수한 성능을 보였으나, 이들은 각 뉴런마다 개별적인 파라미터를 학습시키므로 파라미터 수가 급증하며 과적합 위험이 크다. SAAF는 이러한 복잡도 문제를 Lipschitz 상수를 통한 규제로 해결한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 회귀 신경망의 구조적 분석
본 논문은 회귀 신경망의 최종 예측값 $y$를 다음과 같은 1차원 함수들의 합으로 분해하여 설명한다.
$$y = \sum_{i=1}^{m} f_i(x_i) + b$$
여기서 $f_i$는 $i$번째 뉴런의 활성화 함수이며, $x_i$는 해당 뉴런의 입력값이다. 논문에서는 $\text{Theorem 1}$을 통해, $x_i$들이 서로 독립적이라고 가정할 때 각 $f_i(x_i)$가 타겟 값 $t$의 조건부 기대값 $E[t|x_i]$를 근사하게 됨을 증명한다. 이는 회귀 레이어의 활성화 함수 $f_i$가 정교할수록 모델 전체의 편향이 줄어듦을 의미한다.

### 2. SAAF의 정의 및 구조
SAAF는 다음과 같이 조각별 다항식 형태로 정의된다.
$$f(x) = \sum_{j=0}^{c-1} v_j p_j(x) + \sum_{k=1}^{n} w_k b_k^c(x)$$
- $p_j(x) = \frac{x^j}{j!}$: 기본 다항식 기저 함수(Basis function)이다.
- $b_k^c(x)$: 구간 $[a_k, a_{k+1})$에서 정의된 Boxcar 함수 $b_k^0$를 $c$번 적분한 함수이다.
- $c$: 다항식 세그먼트의 차수(Degree)를 결정하는 하이퍼파라미터이다.
- $a_k$: 조각을 나누는 브레이크 포인트(Break points)이다.
- $v_j, w_k$: 학습 과정에서 최적화되는 파라미터이다.

이 구조의 특징은 $c$차 미분값은 $w_k$에 의해 결정되지만, 그보다 낮은 차수의 미분값들은 전 구간에서 연속성을 유지한다는 점이다. 따라서 $\text{L}_2$ 규제를 통해 $w_k$의 크기를 제한하면 함수의 곡률이 제어되어 매끄러운(Smooth) 함수가 된다.

### 3. 모델 복잡도 제어 (Lipschitz Continuity)
SAAF를 사용한 신경망은 파라미터의 크기가 유계(Bounded)일 때 Lipschitz 연속성을 가진다. Lipschitz 상수 $L$은 다음과 같이 파라미터들의 적분 형태로 유도된다.
$$L = \max_x \left| \int \dots \int w(\alpha) d\alpha^{c-1} + \sum_{j=1}^{c-1} \int \dots \int v_j d\alpha^{j-1} \right|$$
논문은 모든 Lipschitz 연속 회귀 모델에 대해 $\text{fat-shattering dimension}$의 상한선을 다음과 같이 증명한다.
$$\text{fat}_F(\gamma) \le d + \frac{L^d}{d! \gamma^d \sqrt{2^d} (d+1)}$$
결과적으로, $\text{L}_2$ 규제를 통해 Lipschitz 상수 $L$을 줄이면 모델의 복잡도가 다항식 수준으로 감소하여 과적합을 체계적으로 방지할 수 있다.

## 📊 Results

### 실험 설정
- **비교 대상**: ReLU, LReLU, PReLU, APLU 및 제안 방법인 SAAF (선형 $c=1$, 이차 $c=2$).
- **적용 범위**: 전체 레이어 적용 방식과 회귀 레이어에만 적용하는 방식(prefix $\text{R-}$)을 구분하여 실험하였다.
- **데이터셋**:
    - Pose Estimation: LSP, Volleyball
    - Age Estimation: Adience, ICCV 2015 ChaLearn-AgeGuess
    - Facial Attractiveness: hotornot.com 기반 자체 데이터셋
    - Pathology Images: 핵(Nuclei)의 원형도(Circularity) 측정

### 주요 결과
1. **Pose Estimation**: R-SAAFc2를 적용한 결과, 기존 SOTA 모델보다 우수한 PCP(Percentage of Correctly estimated Parts)를 기록하였다. 특히 단순한 뉴런 추가보다 SAAF를 적용하는 것이 훨씬 더 효율적으로 성능을 향상시켰다.
2. **Age Estimation**: Adience 데이터셋에서 SOTA 성능을 달성하였으며, AgeGuess 챌린지에서도 단일 CNN 모델 중 가장 낮은 오차를 기록하였다.
3. **기타 작업**: 얼굴 매력도 예측 및 핵 원형도 측정 작업에서도 SAAF(특히 $\text{R-SAAFc1, c2}$)가 타 활성화 함수 대비 RMSE를 낮추고 상관계수(Correlation)를 높이는 등 일관된 성능 향상을 보였다.
4. **정량적 향상**: 전반적으로 ReLU, LReLU, PReLU, APLU 대비 에러율을 약 $4.2\% \sim 25.0\%$ 가량 감소시켰다.

## 🧠 Insights & Discussion

### 강점
본 연구는 단순히 "성능이 좋다"는 결과 제시를 넘어, **회귀 레이어의 활성화 함수가 타겟 값의 조건부 기대값을 근사한다**는 수학적 근거를 제시하였다. 또한, 복잡한 AAF를 사용하면서도 $\text{L}_2$ 규제가 어떻게 모델 복잡도($\text{fat-shattering dimension}$)를 이론적으로 제어하는지 증명함으로써, 실무적인 튜닝과 이론적 보장 사이의 간극을 메웠다.

### 한계 및 해석
- **하이퍼파라미터 의존성**: 브레이크 포인트 $a_k$의 위치와 개수($n$), 다항식 차수($c$)를 사전에 정의해야 한다. 본 논문에서는 $n=22, a_k \in [-1.1, 1.1]$로 설정하였으나, 데이터셋마다 최적의 설정이 다를 수 있으며 이에 대한 자동화된 탐색 방법은 제시되지 않았다.
- **범용성**: 본 논문은 회귀 작업에 집중하고 있으며, 분류 작업에서의 효용성은 향후 과제로 남겨두었다. 다만, 회귀 레이어의 특성을 이용한 접근법이므로 분류 문제에 그대로 적용하기에는 구조적 차이가 있을 수 있다.

## 📌 TL;DR

이 논문은 회귀 신경망의 모델 편향을 줄이기 위해 조각별 다항식 형태의 **Smooth Adaptive Activation Function (SAAF)**을 제안한다. SAAF는 임의의 함수를 근사할 수 있는 강력한 표현력을 가지면서도, $\text{L}_2$ 규제를 통해 Lipschitz 상수를 제어함으로써 모델 복잡도를 이론적으로 억제하고 과적합을 방지한다. 특히 이를 회귀 레이어에만 적용했을 때 적은 파라미터 증가로도 포즈 추정, 연령 예측 등 다양한 회귀 작업에서 SOTA 성능을 달성하였다. 이 연구는 적응형 활성화 함수의 설계 원칙과 복잡도 제어 방안을 제시했다는 점에서 향후 정교한 회귀 모델 설계에 중요한 기여를 한다.