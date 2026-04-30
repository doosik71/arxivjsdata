# Meta-free few-shot learning via representation learning with weight averaging

Kuilin Chen, Chi-Guhn Lee (2022)

## 🧩 Problem to Solve

본 논문은 Few-shot Learning (FSL) 분야에서 기존의 두 가지 주류 접근 방식인 Episodic Meta-learning과 Transfer Learning이 가진 한계점을 해결하고자 한다.

첫째, Episodic Meta-learning 알고리즘은 전이 가능한 지식을 추출하기 위해 에피소드 방식으로 여러 태스크를 학습하지만, 최근 연구에 따르면 고정된 임베딩을 사용하는 단순한 Transfer Learning 방식이 유사하거나 더 나은 성능을 보인다는 점이 밝혀졌다. 즉, Meta-learning의 성능 향상이 빠른 태스크 적응 능력이 아니라, 단순히 고품질의 Representation을 재사용하는 것에서 기인한다는 분석이 있다.

둘째, 기존 Transfer Learning 방식은 주로 Few-shot Classification에 국한되어 있으며, Few-shot Regression 문제로의 확장이 부족하다.

셋째, 대부분의 FSL 모델은 점 추정(Point estimation) 정확도 향상에만 집중하여, 예측 결과에 대한 불확실성(Uncertainty)을 적절히 측정하는 Calibration 능력이 부족하다. 이는 의료 진단과 같이 리스크 회피가 중요한 실제 응용 분야에서 치명적인 문제가 될 수 있다.

따라서 본 논문의 목표는 Meta-learning의 복잡한 에피소드 학습 과정 없이도, Regression과 Classification 모두에 적용 가능하며 예측 불확실성이 잘 보정된(Well-calibrated) 고성능의 Meta-free Representation Learning (MFRL) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Stochastic Weight Averaging (SWA)**를 통한 Representation의 일반화 성능 향상과, **Probabilistic Linear Model**을 이용한 예측 불확실성의 보정이다.

중심적인 직관은 SWA가 모델을 손실 함수(Loss function)의 평탄한 최솟값(Flat minimum)으로 유도하며, 이것이 결과적으로 Low-rank Representation을 형성하게 하여 새로운 태스크에 대한 샘플 효율성(Sample efficiency)을 높인다는 것이다. 또한, 메타 테스트 단계에서 단순한 선형 층을 확률론적 모델로 미세 조정(Fine-tuning)함으로써 추가적인 메타 학습 없이도 신뢰할 수 있는 불확실성 추정을 가능하게 한다.

## 📎 Related Works

**1. Episodic Meta-learning**
- **Metric-based methods:** 데이터를 특징 벡터로 투영하고 유사도를 측정하여 예측한다 (예: ProtoNet, MatchingNet). 하지만 Regression 문제에 직접 적용하기 어렵다.
- **Optimization-based methods:** 새로운 태스크에 빠르게 적응할 수 있는 초기 파라미터를 찾는다 (예: MAML). 하지만 하이퍼파라미터 튜닝이 까다롭고 계산 비용이 높으며, 학습 과정이 불안정하다는 한계가 있다.

**2. Transfer Learning**
- 표준 지도 학습으로 특징 추출기(Feature extractor)를 학습한 후, 새로운 태스크에서 선형 예측기를 미세 조정한다.
- 일부 최신 방법론(Baseline++, Knowledge Distillation 등)이 성능을 높였으나, 여전히 Classification에 치중되어 있으며 Regression에 대한 대응책이 부족하다. 또한, 보정된 불확실성을 제공하는 메커니즘이 결여되어 있다.

## 🛠️ Methodology

제안된 MFRL은 크게 'Representation Learning'과 'Fine-tuning'의 두 단계로 구성된다.

### 1. Representation Learning (Meta-free)
에피소드 방식이 아닌, 모든 훈련 데이터를 하나의 데이터셋 $D_{tr}$로 통합하여 일반적인 지도 학습을 수행한다.

- **손실 함수:** 태스크의 성격에 따라 다음과 같이 정의한다.
  - **Regression:** Mean Squared Error (MSE) 사용
    $$\mathcal{L}_{MSE}(\theta) = \frac{1}{2N'} \sum_{\tau=1}^{T} \sum_{j=1}^{N_{\tau}} (y_{\tau,j} - w_{\tau}^\top h(x_{\tau,j}))^2$$
  - **Classification:** Cross-Entropy (CE) 사용
    $$\mathcal{L}_{CE}(\theta) = -\sum_{j=1}^{N'} \sum_{c=1}^{C} y_{j,c} \log \frac{\exp(w_c^\top h(x_j))}{\sum_{c'=1}^{C} \exp(w_{c'}^\top h(x_j))}$$
  여기서 $h(x)$는 특징 추출기로, $\theta$는 추출기와 선형 층의 파라미터이다.

- **SWA를 통한 정규화:** SGD 학습 후, 마지막 $s$ 에포크 동안의 가중치를 평균 내어 $\theta_{SWA}$를 얻는다.
  $$\theta_{SWA} = \frac{1}{s} \sum_{i=T+1}^{T+s} \theta_i$$
  이 과정은 모델을 Flat minimum으로 유도하여 Low-rank Representation을 생성하며, 이는 새로운 태스크에 대한 일반화 성능을 높이는 암시적 정규화(Implicit regularization) 역할을 한다.

### 2. Fine-tuning (Probabilistic Top Layer)
학습된 특징 추출기 $\theta_f$를 고정하고, 메타 테스트 태스크의 소수 샘플을 사용하여 새로운 최상위 층을 학습한다.

- **Few-shot Regression:** 
  Hierarchical Bayesian linear model을 사용한다. 가중치 $w$에 대한 Gaussian prior $p(w|\lambda)$를 설정하고, 하이퍼파라미터 $\lambda$(정밀도)와 $\sigma^2$(노이즈 분산)에 대해 Gamma 분포의 Hyper-prior를 부여한다. Marginal likelihood를 최대화하는 반복적 최적화(Iterative optimization)를 통해 $\lambda$와 $\sigma^2$를 추정하며, 최종 예측 분포는 다음과 같이 계산된다.
  $$p(y^* | x^*, X, y, \lambda, \sigma^2) = \int p(y^* | x^*, w, \sigma^2) p(w | X, y, \lambda, \sigma^2) dw$$

- **Few-shot Classification:** 
  $L_2$ 정규화가 포함된 Logistic Regression을 학습한다. 
  $$\mathcal{L}(W) = -\sum_{i=1}^{nK} \sum_{c=1}^{K} y_{i,c} \log \frac{\exp(w_c^\top h(x_i))}{\sum_{c'=1}^{K} \exp(w_{c'}^\top h(x_i))} + \lambda \sum_{c=1}^{K} w_c^\top w_c$$
  불확실성 보정을 위해 **Temperature Scaling** 인자 $T$를 도입하여 Softmax 출력을 조정한다.
  $$p_c = \frac{\exp(w_c^\top h(x^*) / T)}{\sum_{c'=1}^{K} \exp(w_{c'}^\top h(x^*) / T)}$$
  $T$는 메타 검증 데이터셋에서 Expected Calibration Error (ECE)를 최소화하는 값으로 설정한다.

## 📊 Results

### 1. Few-shot Regression
- **데이터셋:** Sine waves, Head pose estimation.
- **결과:** MFRL은 MAML, Bayesian MAML 등 기존 Meta-learning 방법론보다 낮은 MSE를 기록하며 SOTA 성능을 달성했다. 특히 SWA를 적용했을 때 성능이 크게 향상되었으며, Hierarchical Bayesian 모델을 통해 10개의 샘플만으로도 정확한 예측 불확실성을 정량화함을 확인했다.

### 2. Few-shot Classification
- **데이터셋:** miniImageNet, tieredImageNet, CIFAR-FS, FC100 및 Cross-domain (miniImageNet $\rightarrow$ CUB).
- **결과:** ResNet-12, WRN-28-10 등 다양한 백본에서 대부분의 SOTA 방법론을 능가하는 정확도를 보였다. 특히 Cross-domain 태스크에서 강력한 성능을 보여, SWA로 학습된 Representation의 일반화 능력이 매우 뛰어남을 입증했다.

### 3. 분석 실험
- **Effective Rank:** Singular value의 감쇄 속도를 분석한 결과, SWA를 적용한 Representation이 적용하지 않은 경우보다 더 빠르게 감쇄(Faster decay)하는 것을 확인했다. 이는 SWA가 실제로 Low-rank Representation을 형성한다는 가설을 뒷받침한다.
- **Reliability:** Reliability Diagram과 ECE, MCE, BRI 지표를 통해 분석한 결과, Temperature Scaling을 적용한 MFRL이 MAML이나 ProtoNet보다 훨씬 더 잘 보정된(Well-calibrated) 확률 예측을 수행함을 보였다.
- **SWA의 범용성:** SWA를 ProtoNet, MAML 같은 기존 에피소드 기반 Meta-learning에 적용했을 때도 분류 정확도가 향상됨을 확인하여, SWA가 학습 패러다임에 관계없이 유효한 기법임을 입증했다.

## 🧠 Insights & Discussion

본 연구는 Meta-learning의 복잡한 구조 없이도 단순한 Transfer Learning과 SWA의 조합만으로 SOTA 성능을 낼 수 있음을 보여주었다. 특히 SWA가 단순한 가중치 평균을 넘어, Low-rank Representation을 유도함으로써 Few-shot 상황에서의 샘플 효율성을 극대화한다는 점을 발견한 것이 학술적으로 큰 의미가 있다.

또한, 불확실성 보정을 위해 도입한 Temperature Scaling과 Bayesian Linear Model이 메타 학습 단계의 복잡한 설계 없이도 추론 단계의 미세 조정만으로 충분히 작동한다는 점을 보였다.

**한계점 및 논의사항:**
- SWA가 Low-rank Representation을 형성한다는 점을 경험적으로는 증명했으나, 이에 대한 엄밀한 이론적 연결 고리는 여전히 연구 과제로 남아 있다.
- Explicit regularizer(L1, Nuclear norm 등) 대신 SWA라는 implicit regularizer를 사용함으로써 하이퍼파라미터 설정의 어려움을 줄이고 계산 효율성을 높였으나, SWA의 최적 에포크나 학습률에 대한 일반적인 가이드라인은 추가적인 연구가 필요할 수 있다.

## 📌 TL;DR

본 논문은 Meta-learning의 에피소드 학습 없이 SWA(Stochastic Weight Averaging)를 통해 일반화 성능이 높은 Low-rank Representation을 학습하고, 확률론적 선형 모델과 Temperature Scaling으로 예측 불확실성을 보정하는 **MFRL(Meta-Free Representation Learning)** 방법론을 제안한다. 이 방법은 Regression과 Classification 모두에서 SOTA 성능을 달성했으며, 특히 구현이 간단하면서도 높은 신뢰성의 예측 결과를 제공한다는 점에서 실용적 가치가 매우 높다.