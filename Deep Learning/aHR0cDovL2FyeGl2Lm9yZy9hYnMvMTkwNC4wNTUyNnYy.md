# A Selective Overview of Deep Learning

Jianqing Fan, Cong Ma, Yiqiao Zhong (2019)

## 🧩 Problem to Solve

본 논문은 최근 컴퓨터 비전과 자연어 처리 등 다양한 분야에서 엄청난 성공을 거둔 Deep Learning(심층 학습)을 통계학 및 과학적 관점에서 분석하고 정리하는 것을 목표로 한다. 구체적으로 해결하고자 하는 핵심 질문은 다음과 같다.

1. **Deep Learning의 본질**: Deep Learning이란 무엇이며, 고전적인 통계 방법론과 비교했을 때 어떤 새로운 특징을 가지는가?
2. **이론적 근거**: 왜 Deep Learning은 이미지 인식과 같은 복잡한 작업에서 기존의 얕은 모델(Shallow models)보다 뛰어난 성능을 보이는가?
3. **일반화의 미스터리**: 수많은 파라미터를 가진 Over-parametrization(과잉 매개변수화) 상태임에도 불구하고, 왜 과적합(Overfitting)되지 않고 새로운 데이터에 대해 좋은 일반화 성능을 보이는가?

이 논문은 이러한 질문들에 답하기 위해 신경망 모델의 구조, 학습 알고리즘, 근사 이론(Approximation theory), 그리고 일반화 능력(Generalization power)을 체계적으로 검토한다.

## ✨ Key Contributions

본 논문의 중심적인 직관은 Deep Learning의 성공이 단순히 데이터 양의 증가나 계산 능력의 향상 때문만이 아니라, **Depth(깊이)**와 **Over-parametrization(과잉 매개변수화)**이라는 두 가지 핵심 특성에서 기인한다는 것이다.

- **Depth의 역할**: 현실 세계의 데이터(예: 이미지)는 계층적 구조를 가지고 있다. 깊은 층을 통해 하위 수준의 특징(edges, corners)을 결합하여 상위 수준의 특징(wheels, eyes)을 추출하는 함수 합성(Function composition) 구조가 복잡한 비선형성을 효율적으로 모델링할 수 있게 한다.
- **Over-parametrization의 역설**: 파라미터 수가 샘플 수보다 훨씬 많은 상태에서도, Stochastic Gradient Descent(SGD)와 같은 최적화 알고리즘이 암묵적인 정규화(Implicit regularization) 효과를 제공하여 단순한 모델과 유사한 일반화 능력을 갖게 한다는 관점을 제시한다.

## 📎 Related Works

논문은 Deep Learning 이전의 고전적인 통계적 머신러닝 방법론들을 언급하며 차이점을 설명한다.

- **기존 접근 방식**: Linear classifiers(선형/로지스틱 회귀), Kernel methods(SVM), Tree-based methods(Decision trees, Random forests), Nonparametric regression 등이 있다. 이들은 주로 사람이 직접 설계한 특징(Handcrafted features)에 의존하거나, 특정 함수 클래스 $\mathcal{F}$ 내에서 최적의 함수를 찾는 방식이다.
- **한계 및 차별점**: 고전적 방법들은 데이터의 복잡한 비선형 의존성을 모델링하는 데 한계가 있으며, 특히 고차원 데이터에서 '차원의 저주(Curse of dimensionality)' 문제에 직면한다. 반면, Deep Learning은 데이터로부터 직접 특징을 학습하는 Feature representation learning을 수행하며, 깊은 층을 통해 이를 효율적으로 달성한다.

## 🛠️ Methodology

### 1. 신경망 모델 구조
#### Feed-forward Neural Networks (MLP)
가장 기본적인 구조로, 여러 층의 선형 변환과 비선형 활성화 함수 $\sigma(\cdot)$의 합성을 통해 다음과 같이 정의된다.
$$f(x; \theta) = W^L \sigma^L(W^{L-1} \dots \sigma^2(W^2 \sigma^1(W^1 x)))$$
여기서 $\sigma(\cdot)$로는 ReLU ($\max\{z, 0\}$)가 널리 사용된다. 학습을 위해 주로 Multinomial logistic loss(교차 엔트로피 손실)를 최소화한다.

#### Convolutional Neural Networks (CNN)
이미지와 같은 공간적 구조를 가진 데이터에 특화된 모델이다.
- **Convolutional Layer**: Filter(커널)를 사용하여 국소적 특징을 추출하며, 모든 위치에서 동일한 필터를 사용하는 Weight sharing(가중치 공유) 특성을 가진다.
- **Pooling Layer**: Max-pooling 등을 통해 데이터를 다운샘플링하여 계산량을 줄이고 불필요한 중복을 제거한다.

#### Recurrent Neural Networks (RNN)
시계열 및 시퀀스 데이터를 처리하기 위한 모델로, 이전 단계의 은닉 상태 $h_{t-1}$을 현재 단계의 입력 $x_t$와 함께 처리하는 재귀적 구조를 가진다.
$$h_t = f_\theta(h_{t-1}, x_t)$$
장기 의존성(Long-range dependency) 문제를 해결하기 위해 게이트 구조를 도입한 LSTM과 GRU가 사용된다.

#### Unsupervised Learning (AE, GAN)
- **Autoencoders (AE)**: 입력 $x$를 저차원 코드 $h$로 압축하는 Encoder $f(\cdot)$와 이를 다시 복원하는 Decoder $g(\cdot)$로 구성되며, $\|x - g(f(x))\|^2$를 최소화한다.
- **Generative Adversarial Networks (GAN)**: 가짜 데이터를 만드는 Generator $G$와 진짜/가짜를 판별하는 Discriminator $D$가 서로 경쟁하는 Min-max game 구조를 가진다.
$$\min_{\theta_G} \max_{\theta_D} \mathbb{E}_{x \sim P^X} [\log(d(x))] + \mathbb{E}_{z \sim P^Z} [\log(1 - d(g(z)))]$$

### 2. 학습 절차 및 최적화
- **SGD (Stochastic Gradient Descent)**: 전체 데이터가 아닌 mini-batch를 사용하여 그래디언트를 계산함으로써 계산 효율성을 높인다.
- **Back-propagation**: 연산 그래프(Computational graphs) 위에서 Chain rule을 적용하여 출력층부터 입력층 방향으로 그래디언트를 효율적으로 전달한다.
- **수치적 안정화 기법**: 
    - **ReLU**: Sigmoid의 Gradient Vanishing 문제를 해결한다.
    - **Skip connections (Residual connections)**: $\sigma(x + F(x))$ 구조를 통해 매우 깊은 망에서도 그래디언트 흐름을 원활하게 하여 학습을 가능하게 한다.
    - **Batch Normalization**: 각 층의 입력을 평균 0, 분산 1로 표준화하여 학습 속도를 높이고 하이퍼파라미터 민감도를 낮춘다.

## 📊 Results

본 논문은 새로운 실험 결과보다는 기존의 벤치마크 결과(ImageNet Challenge)를 통해 Deep Learning의 우수성을 논증한다.

- **정량적 결과**: Table 1에 제시된 바와 같이, 2012년 전의 Shallow 모델은 Top-5 error가 25%였으나, AlexNet(8층), VGG(19층), ResNet-152(152층)로 층이 깊어짐에 따라 에러율이 3.6%까지 급격히 감소하였다.
- **분석**: 이는 모델의 깊이(Depth)와 파라미터 수의 증가가 이미지 인식 작업에서 성능 향상과 직접적인 상관관계가 있음을 보여준다.

## 🧠 Insights & Discussion

### 1. 근사 이론(Approximation Theory)과 깊이의 이점
얕은 신경망(One-hidden-layer)도 충분한 뉴런이 있다면 모든 연속 함수를 근사할 수 있다는 Universal Approximation Theorem이 존재하지만, 이는 뉴런 수가 차원에 따라 지수적으로 증가해야 하는 '차원의 저주'를 수반한다. 반면, 깊은 망은 함수 합성을 통해 다항식(Monomial)과 같은 복잡한 함수를 훨씬 적은 수의 뉴런($O(d)$)으로 표현할 수 있음을 수학적으로 보여준다.

### 2. 일반화 능력에 대한 통찰
- **Algorithm-independent control**: 가중치의 Frobenius norm 등을 제한하면 모델의 복잡도(Rademacher complexity)가 제어되어 일반화 오차가 줄어든다.
- **Algorithm-dependent control**: SGD 자체의 Stability(안정성)와 Implicit regularization(암묵적 정규화) 효과가 중요하다. 예를 들어, 정규화 항이 없더라도 GD는 L2-norm이 최소인 해(Max-margin solution)를 찾는 경향이 있다.

### 3. 비판적 해석 및 한계
- **이론과 실제의 괴리**: 많은 이론적 증명이 ReLU가 아닌 특정 조건의 활성화 함수나 단순화된 선형 모델(Linear networks)에서 이루어지고 있다.
- **강건성(Robustness) 문제**: 높은 성능에도 불구하고, 아주 작은 적대적 섭동(Adversarial perturbations)에 모델이 쉽게 무너지는 취약점이 있으며 이는 향후 해결해야 할 과제이다.

## 📌 TL;DR

이 논문은 Deep Learning을 통계적 관점에서 분석한 종합 보고서이다. 핵심은 **Depth**를 통한 효율적인 함수 표현(차원의 저주 극복)과 **Over-parametrization** 상태에서도 SGD가 제공하는 암묵적 정규화를 통해 뛰어난 일반화 성능을 달성한다는 것이다. 이 연구는 딥러닝의 실무적 성공을 수학적으로 뒷받침하려는 시도로서, 향후 데이터 분포의 특성 규명 및 모델의 강건성 향상 연구에 중요한 이론적 토대를 제공한다.