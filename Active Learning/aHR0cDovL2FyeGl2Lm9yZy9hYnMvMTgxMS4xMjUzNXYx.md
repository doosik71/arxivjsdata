# The Relevance of Bayesian Layer Positioning to Model Uncertainty in Deep Bayesian Active Learning

Jiaming Zeng, Adam Lesnikowski, Jose M. Alvarez (2018)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델이 가진 고질적인 문제인 모델 불확실성(Model Uncertainty) 측정의 어려움을 해결하고자 한다. 이를 위해 Bayesian Deep Learning이 대안으로 제시되었으나, 완전한 Bayesian Neural Networks(BNNs)는 결정론적(Deterministic) 네트워크에 비해 학습 시간이 오래 걸리고 계산 비용이 매우 높다는 단점이 있다. 특히 네트워크의 크기와 파라미터 수가 증가함에 따라 학습의 복잡도가 급격히 증가하는 문제가 발생한다.

따라서 본 연구의 목표는 모델의 불확실성을 성공적으로 캡처하기 위해 반드시 모든 레이어가 Bayesian 레이어여야 하는지를 탐구하는 것이다. 즉, 네트워크 내에서 Bayesian 레이어의 수와 위치를 조정함으로써, 계산 효율성을 유지하면서도 결정론적 네트워크의 속도와 Bayesian 네트워크의 불확실성 추정 능력을 동시에 확보할 수 있는 최적의 구성을 찾는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"네트워크의 모든 레이어를 Bayesian으로 만들 필요 없이, 출력단에 가까운 일부 레이어만 Bayesian으로 구성해도 충분한 모델 불확실성을 확보할 수 있다"**는 것이다. 

저자들은 Bayesian 레이어의 위치와 수에 따른 능력을 Active Learning(AL) 환경에서 실험적으로 검증하였으며, 특히 출력층 근처의 Dense 레이어를 Bayesian으로 설정하는 것이 모델의 불확실성을 캡처하는 데 결정적인 역할을 한다는 직관을 제시한다.

## 📎 Related Works

본 논문은 고차원 데이터에 대해 네트워크 불확실성을 추정하기 위해 Monte Carlo(MC) Dropout을 사용하는 기존 연구$[2]$를 주요 비교 대상으로 삼는다. MC Dropout은 효율적인 불확실성 추정 방법으로 알려져 있으나, Bayesian CNN 역시 네트워크 규모가 커질수록 학습 시간과 복잡도가 증가한다는 한계가 명시되어 있다.

본 연구는 MC Dropout 대신 Gaussian approximate variational inference를 사용한 Bayesian CNN을 채택하였다. 기존 연구들이 주로 전체 네트워크를 Bayesian으로 구축하거나 특정 기법(Dropout 등)에 의존했다면, 본 논문은 **'레이어의 위치(Positioning)'**라는 관점에서 Bayesian 레이어의 배치가 불확실성 추정에 미치는 영향을 분석했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 구조 및 파이프라인
연구에 사용된 네트워크 아키텍처는 다음과 같다:
`Input` $\rightarrow$ `Conv1` $\rightarrow$ `ReLU` $\rightarrow$ `Conv2` $\rightarrow$ `ReLU` $\rightarrow$ `Max Pooling` $\rightarrow$ `Dropout` $\rightarrow$ `Dense1` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout` $\rightarrow$ `Dense2` $\rightarrow$ `Output`

저자들은 이 구조에서 어떤 레이어를 Bayesian으로 설정하고 어떤 레이어를 Deterministic으로 유지할지를 조합하여 총 8가지의 아키텍처(BNN, BNN-1~3, BNN1~3, CNN)를 구성하여 실험하였다.

### 2. Bayesian CNN의 수학적 기초
Bayesian CNN은 모델 파라미터 $w$에 대해 사전 확률 분포(Prior distribution) $p(w) \sim \mathcal{N}(\mu, \Sigma)$를 정의한다. 데이터 $D=\{X_i, y_i\}_{i=1}^N$가 주어졌을 때, 사후 분포(Posterior distribution)는 다음과 같이 정의된다.
$$p(w|X,y) = \frac{p(y|X,w)p(w)}{p(y|X)}$$

최종 예측(Likelihood)은 다음과 같은 적분 형태로 나타난다.
$$p(y^*|x^*,X,y) = \int p(y^*|x^*,w)p(w|X,y)dw$$

### 3. 학습 및 추론 절차
사후 분포 $p(w|X,y)$를 직접 계산하는 것은 매우 어렵기 때문에, 본 논문에서는 변분 추론(Variational Inference)을 사용하여 이를 단순한 가우시안 분포 $q(w) \sim \mathcal{N}(\mu_q, \sigma_q)$로 근사한다.

- **손실 함수**: KL Divergence를 최소화하는 것은 Negative Evidence Lower Bound(ELBO)를 최소화하는 것과 동일하며, 이를 손실 함수로 사용한다.
$$-ELBO = -\mathbb{E}(\log p(X|w)) + \mathbb{E}\left(\log \frac{q(w)}{p(w)}\right)$$
- **최적화**: Mini-batch 내의 그라디언트 상관관계를 제거하여 효율적인 추정치를 얻기 위해 Flipout 기법을 사용하였으며, ADAM 옵티마이저를 적용하였다.
- **불확실성 추정**: 추론 시에는 Monte Carlo 적분을 사용하여 $T$번의 forward pass를 수행하고 이를 평균내어 예측 분포를 근사한다.
$$p(y=c|x,D) \approx \frac{1}{T} \sum_{t=1}^T p(y=c|x, \hat{w}_t)$$

### 4. Active Learning 설정
- **데이터셋**: MNIST
- **절차**: 초기 20장의 이미지로 시작하여, 사이클마다 10장씩 추가로 획득하여 총 1,000장까지 학습한다.
- **Acquisition Functions**:
    - **Random**: 무작위 선택
    - **Max Entropy**: $H[y|x,D] := -\sum_c p(y=c|x,D) \log p(y=c|x,D)$
    - **Variation Ratios**: $VR[x] := 1 - \max_y p(y|x,D)$

## 📊 Results

### 1. 실험 설정 및 지표
- **비교 대상**: 8가지 Bayesian/Deterministic 조합 아키텍처 및 MC Dropout
- **지표**: 테스트 에러(Test Error, 낮을수록 좋음)

### 2. 주요 결과
- **레이어 위치의 영향**: Bayesian 레이어가 입력단에 가까운 경우(BNN1, BNN2, BNN3)보다 출력단에 가까운 경우(BNN-1, BNN-2, BNN-3)가 불확실성을 훨씬 더 잘 캡처하여 테스트 에러가 낮게 나타났다.
- **최적 구성**: 특히 **Dense2 레이어 하나만 Bayesian으로 설정한 BNN-1**이 전체를 Bayesian으로 설정한 BNN보다 더 우수한 성능을 보였다. 이는 불필요하게 많은 Bayesian 레이어를 추가하는 것이 오히려 정확도를 떨어뜨릴 수 있음을 시사한다.
- **Bayesian-ness의 영향**: 사전 분포의 분산 $\mu$ 값을 조정하여 '얼마나 Bayesian하게' 초기화할지를 실험한 결과, 출력층(Dense2)의 $\mu$ 값이 클수록(더 Bayesian할수록) 불확실성 캡처 능력이 향상되어 성능이 개선되었다.

| Acquisitions | MC Dropout | BNN | **BNN-1 (Best)** | BNN-2 | BNN-3 | BNN1 | BNN2 | BNN3 | CNN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Random | 4.66% | 6.62% | **5.58%** | 5.71% | 6.37% | 6.62% | 6.44% | 6.59% | 6.90% |
| Max Ent | 1.74% | 3.67% | **2.63%** | 3.22% | 3.28% | 7.50% | 4.62% | 3.58% | 10.03% |
| Var Ratios | 1.64% | 3.56% | **2.40%** | 3.00% | 3.34% | 2.70% | 2.84% | 3.32% | 6.48% |

*(위 표는 Table 2의 데이터를 요약한 것임)*

## 🧠 Insights & Discussion

### 강점 및 발견
본 연구는 Bayesian Neural Network의 막대한 계산 비용 문제를 해결하기 위해 "부분적 Bayesian 접근법"이라는 실용적인 방향을 제시하였다. 실험 결과, 모델의 불확실성은 네트워크 전체에 퍼져 있기보다 출력층 근처의 파라미터에 의해 더 효과적으로 결정된다는 점을 입증하였다. 이를 통해 결정론적 네트워크의 학습 속도와 Bayesian 네트워크의 불확실성 추정 능력을 결합할 수 있는 가능성을 보여주었다.

### 한계 및 비판적 해석
1. **데이터셋의 제한성**: 실험이 MNIST라는 상대적으로 단순하고 작은 데이터셋에서만 진행되었다. 더 복잡하고 고차원인 실제 데이터셋(예: ImageNet, 의료 영상 등)에서도 동일한 레이어 위치의 유효성이 유지될지는 명확하지 않다.
2. **아키텍처의 단순함**: 사용된 CNN 구조가 매우 단순하여, 최신 아키텍처(ResNet, Transformer 등)의 깊은 계층 구조에서도 출력층만 Bayesian으로 설정하는 것이 유효할지는 추가 검증이 필요하다.
3. **가정**: 본 논문은 Gaussian approximate variational inference를 사용하였으나, 다른 종류의 Prior나 추정 방식을 사용했을 때도 레이어 위치의 영향이 동일하게 나타날지는 언급되지 않았다.

## 📌 TL;DR

이 논문은 Bayesian CNN의 모든 레이어를 Bayesian으로 만들 필요 없이, **출력층(Dense2)과 같이 출력에 가까운 일부 레이어만 Bayesian으로 구성하는 것이 계산 효율성과 불확실성 캡처 능력 면에서 더 유리함**을 증명하였다. 이는 향후 Bayesian 딥러닝을 실제 대규모 시스템에 적용할 때, 학습 비용을 획기적으로 줄이면서도 신뢰할 수 있는 불확실성 추정치를 얻을 수 있는 설계 지침을 제공한다.