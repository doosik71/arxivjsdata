# Theory of Curriculum Learning, with Convex Loss Functions

Daphna Weinshall, Dan Amir (2018)

## 🧩 Problem to Solve

본 논문은 기계 학습에서 쉬운 예제부터 어려운 예제로 점진적으로 학습시키는 Curriculum Learning(CL)의 이론적 근거를 제시하는 것을 목표로 한다. CL은 여러 실험적 연구를 통해 성능 향상이 입증되었음에도 불구하고, 단순한 케이스에 대해서조차 체계적인 이론적 분석이 부족한 상태였다.

특히, 기존 CL 연구들의 핵심적인 한계점은 '예제의 어려움(difficulty score)'에 대한 공식적이고 일반적인 정의가 없다는 점이다. 이로 인해 도메인 전문가가 수동으로 커리큘럼을 설계해야 하는 한계가 있었으며, 이는 데이터 규모가 커질수록 불가능에 가깝다. 또한, 쉬운 데이터부터 학습하라는 CL의 직관과, 현재 모델이 틀린 어려운 데이터를 집중 학습하라는 Hard Data Mining 및 Boosting의 직관이 서로 충돌한다는 점이 이론적으로 해결되지 않은 문제로 남아 있었다.

## ✨ Key Contributions

본 논문의 핵심 기여는 '어려움'의 정의를 두 가지 관점(Global 및 Local)으로 분리하여 정의하고, 이를 통해 CL의 효과를 수학적으로 증명한 것이다.

1. **Ideal Difficulty Score (IDS) 정의**: 예제의 전역적 어려움을 최적 가설(Optimal Hypothesis)에서의 손실 값으로 정의하여, 객관적인 Global Difficulty의 기준을 제시하였다.
2. **전역적 어려움과 수렴 속도의 관계 증명**: 선형 회귀(Linear Regression)와 힌지 손실(Hinge Loss) 기반의 이진 분류라는 두 가지 볼록(Convex) 문제에서, 전역적 어려움 $\Psi$가 낮을수록(즉, 쉬운 예제일수록) SGD의 기대 수렴 속도가 단조적으로 증가함을 증명하였다.
3. **전역적 vs 지역적 어려움의 구분**: 현재 가설에서의 손실 값인 Local Difficulty $\Upsilon$와 IDS인 Global Difficulty $\Psi$를 구분하였다. 분석 결과, 전역적 어려움이 고정된 상태에서는 지역적 어려움이 높을수록 수렴 속도가 증가함을 보였다. 이를 통해 CL과 Hard Data Mining이라는 상반된 두 휴리스틱이 실제로는 서로 다른 층위의 어려움을 다루고 있음을 이론적으로 설명하였다.

## 📎 Related Works

논문은 Curriculum Learning과 관련된 기존 접근 방식들의 특성과 한계를 다음과 같이 설명한다.

- **Bengio et al. (2009)**: CL의 개념을 제안하고 실험적으로 성능 향상을 보였으나, 커리큘럼을 도메인 지식에 기반해 수동으로 설계해야 한다는 한계가 있다.
- **Self-Paced Learning (SPL)**: 학습자의 현재 상태(Local Loss)를 이용하여 스스로 커리큘럼을 생성하는 방식이다. 외부의 사전 정의된 커리큘럼이 필요 없다는 장점이 있으나, 최적화 문제가 더 복잡해지고 과적합(Overfitting) 및 학습 불안정성 문제가 발생할 수 있다.
- **Hard Example Mining 및 Boosting**: 모델이 현재 예측하기 어려워하는(Local Loss가 높은) 데이터에 가중치를 두는 방식이다. 이는 낮은 손실의 데이터를 선호하는 SPL이나 CL의 직관과 정면으로 충돌한다.
- **MentorNet 및 Active Bias**: 데이터 기반으로 적응형 커리큘럼을 생성하거나 예측 분산(Variance)을 활용하는 방식이다. 하지만 이들은 '어려움'의 정의를 직접적으로 다루지 않으며, 생성된 편향이 반드시 이론적인 어려움 기반 커리큘럼과 일치하는지는 알 수 없다.

## 🛠️ Methodology

### 1. 어려움의 정의 (Difficulty Score)

본 논문은 어려움을 전역적 관점과 지역적 관점으로 나누어 정의한다.

- **Global Difficulty (Ideal Difficulty Score, $\Psi$):** 최적 가설 $\bar{h}$에 대한 손실 값으로 정의한다.
  $$\Psi(X) = g(L(X, \bar{h}))$$
- **Local Difficulty ($\Upsilon$):** 현재 반복 단계 $t$에서의 가설 $h_t$에 대한 손실 값으로 정의한다.
  $$\Upsilon(X) = g(L(X, h_t))$$
여기서 $g(\cdot)$는 단조 증가 함수이다.

### 2. 분석 대상 모델 및 학습 절차

본 논문은 이론적 분석을 위해 다음 두 가지 Convex Loss 함수를 사용하며, 기본 SGD 업데이트 규칙을 따른다.
$$w_{t+1} = w_t - \eta \frac{\partial L(X_t, w)}{\partial w} \Big|_{w=w_t}$$

#### A. 선형 회귀 (Linear Regression)

손실 함수로 최소제곱법(Least Squares)을 사용한다.
$$L(X, w) = (x \cdot w - y)^2$$
여기서 $\Psi(X) = \sqrt{L(X, \bar{w})}$로 정의한다. 분석을 위해 파라미터 공간에서 최적 해 $\bar{w}$를 원점으로 하는 구면 좌표계를 도입하고, 기대 수렴 속도 $\Delta(\Psi)$를 다음과 같이 정의하여 분석한다.
$$\Delta(\Psi) = E[\|w_t - \bar{w}\|^2 - \|w_{t+1} - \bar{w}\|^2 | \Psi]$$

#### B. 힌지 손실 기반 분류 (Hinge Loss Minimization)

소프트 마진 SVM과 유사한 형태의 힌지 손실을 사용하며, 가중치 벡터의 크기를 $\|w\|=1$로 제한하는 제약 조건을 둔다.
$$L(X, w) = \max(1 - (x \cdot w)y, 0)$$
수렴 지표로는 두 벡터 간의 코사인 유사도(Cosine Similarity)의 변화량을 사용한다.
$$\Delta(\Psi) = E \left[ \frac{w_{t+1} \cdot \bar{w}}{\|w_{t+1}\| \|\bar{w}\|} - \frac{w_t \cdot \bar{w}}{\|w_t\| \|\bar{w}\|} \Big| \Psi \right]$$

## 📊 Results

본 논문은 실험적 수치보다는 수학적 증명을 통한 정리를 결과로 제시한다.

### 1. Global Difficulty와 수렴 속도

- **선형 회귀**: Theorem 3.1을 통해 $\frac{\partial \Delta(\Psi)}{\partial \Psi} \le 0$임을 증명하였다. 즉, 전역적 어려움 $\Psi$가 낮을수록 기대 수렴 속도가 단조적으로 증가한다.
- **힌지 손실**: Theorem 4.1을 통해 $\bar{w}$와 $w_t$가 양의 상관관계를 가질 때, $\Psi > 1 - \cos\theta$ 범위에서 수렴 속도가 $\Psi$에 따라 단조 감소함을 증명하였다. 이는 대부분의 유의미한 데이터 범위에서 CL이 유효함을 시사한다.

### 2. Local Difficulty와 수렴 속도

- **선형 회귀**: 전역적 어려움 $\Psi$가 고정되었을 때, 지역적 어려움 $\Upsilon$가 증가할수록 기대 수렴 속도가 단조적으로 증가함을 보였다(Theorem 3.2).
- **힌지 손실**: $\cos\theta \ge 0$인 조건에서 $\Psi$가 고정되었을 때, $\Upsilon > 0$인 모든 범위에서 수렴 속도가 $\Upsilon$에 따라 단조 증가함을 증명하였다(Theorem 4.3).

## 🧠 Insights & Discussion

본 논문은 CL의 이론적 토대를 마련함으로써 다음과 같은 통찰을 제공한다.

첫째, **전역적 어려움과 지역적 어려움의 상충 관계를 해결**하였다. 분석 결과에 따르면, 학습자는 기본적으로 전역적으로 쉬운 예제부터 시작하는 커리큘럼을 따라야 하지만($\Psi$ 최소화), 동시에 현재 모델이 이미 마스터하여 지역적으로 너무 쉬운 예제에 시간을 낭비해서는 안 된다($\Upsilon$ 최대화).

둘째, **SPL과 Hard Example Mining의 선택 기준을 제시**한다.

- 학습 데이터에 노이즈가 많을 경우, Local Score와 Global Score의 상관관계가 높을 가능성이 크다. 이 경우 Local Score가 낮은 데이터를 선호하는 SPL 방식이 결과적으로 Global Score가 낮은(깨끗한) 데이터를 선택하게 되어 수렴을 도울 수 있다.
- 반면 두 스코어의 상관관계가 낮다면, 현재 모델이 틀린 데이터를 집중 학습하는 Hard Example Mining이 더 효율적일 수 있다.

셋째, 본 연구의 한계는 분석 범위를 Convex 문제로 제한했다는 점이다. 하지만 저자들은 딥러닝과 같은 Non-convex 문제에서도 이러한 경향성이 관찰된다는 이전 연구(Weinshall et al., 2018)를 언급하며 이론의 확장 가능성을 시사하였다.

## 📌 TL;DR

본 논문은 Curriculum Learning의 핵심인 '어려움'을 최적 가설 기준의 **Global Difficulty ($\Psi$)**와 현재 가설 기준의 **Local Difficulty ($\Upsilon$)**로 정의하고, 볼록 최적화 문제에서 이들의 수렴 속도 관계를 수학적으로 증명하였다. 결론적으로 **전역적으로는 쉬운 데이터를 우선하되, 지역적으로는 현재 모델이 어려워하는 데이터를 학습하는 것이 가장 효율적**임을 밝혀, CL과 Hard Example Mining의 이론적 통합 가능성을 제시하였다. 이는 향후 데이터의 특성(노이즈 등)에 따라 어떤 샘플링 전략을 취해야 하는지에 대한 이론적 가이드라인이 될 수 있다.
