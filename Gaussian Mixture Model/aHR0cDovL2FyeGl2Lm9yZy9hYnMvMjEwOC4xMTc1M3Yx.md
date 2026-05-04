# A survey on Bayesian inference for Gaussian mixture model

Jun Lu(2021)

## 🧩 Problem to Solve

본 논문은 클러스터링의 핵심 기술인 Gaussian Mixture Model(GMM)에서 발생하는 베이지안 추론의 문제점과 이를 해결하기 위한 방법론을 다룬다. 특히, 유한(Finite) 및 무한(Infinite) GMM에서 공통적으로 나타나는 **과적합(Overfitting) 및 과잉 클러스터링(Over-clustering)** 문제를 해결하는 것을 목표로 한다.

구체적으로는 다음과 같은 문제들에 집중한다:

1. **유한 GMM의 과적합**: 컴포넌트의 수 $K$를 실제보다 크게 설정했을 때, 위치가 유사한 중복 컴포넌트들이 생성되어 모델이 불필요하게 복잡해지고 결과의 정확도가 떨어지는 문제이다.
2. **무한 GMM(DPM)의 비식별성(Non-identifiability)**: Dirichlet Process Mixture(DPM) 모델은 데이터 샘플 수가 증가함에 따라 새로운 클러스터를 지속적으로 생성하는 경향이 있으며, 이로 인해 실제 필요한 수보다 더 많은 작은 클러스터(redundant mixture components)가 생성되는 과잉 클러스터링 문제가 발생한다.
3. **기존 해결책의 한계**: 기존의 Pruning 방식이나 단순한 상한선(Upper bound) 설정은 임계값(Threshold)을 정하기 어렵고 실세계 데이터에 적용하기에 최적의 설정값을 찾기 어렵다는 한계가 있다.

## ✨ Key Contributions

본 논문의 핵심 기여는 베이지안 GMM 추론을 위한 포괄적인 수학적 도구를 정리함과 동시에, 과잉 클러스터링을 억제하기 위한 새로운 메커니즘을 제안하고 분석한 것이다.

1. **Powered Chinese Restaurant Process (pCRP) 제안**: 기존 CRP의 'Rich-get-richer' 속성을 강화하여, 작은 클러스터가 생성되는 것을 억제하고 기존의 큰 클러스터로의 통합을 유도하는 pCRP 모델을 소개한다.
2. **Symmetric Dirichlet Prior를 위한 Hyperprior 도입**: 유한 GMM에서 집중 매개변수 $\alpha$에 대한 hyperprior를 설정함으로써, 데이터에 따라 적절한 $\alpha$ 값을 학습하여 중복 컴포넌트를 효율적으로 제거하는 방법을 제시한다.
3. **Pruning 방법론 분석**: Constrained Sampling(cSampling)과 Loss-based Sampling(lSampling)과 같은 가지치기 기법을 통해 샘플링 과정에서 불필요한 클러스터를 제거하는 전략을 상세히 설명한다.
4. **수학적 기반의 통합 가이드**: Conjugate Prior(NIW, Dirichlet)부터 MCMC, Gibbs Sampling, 그리고 Cholesky Decomposition을 이용한 계산 최적화까지 베이지안 GMM 추론에 필요한 전 과정을 체계적으로 정리하였다.

## 📎 Related Works

논문은 기존의 접근 방식과 베이지안 방식의 차이점을 다음과 같이 설명한다.

- **Frequentist EM 알고리즘**: 최대 우도 추정(MLE) 프레임워크를 사용하지만, 지역 최적점(Local maxima)에 빠질 위험이 크고 파라미터의 불확실성(Uncertainty)에 대한 추정치를 제공하지 않는다.
- **Standard Dirichlet Process Mixture (DPM)**: 유연한 모델링이 가능하지만, 샘플 크기가 커짐에 따라 로그 속도(log rate)로 새로운 클러스터를 추가하여 컴포넌트 수를 과다 추정하는 경향이 있다.
- **Pitman-Yor Process 등 확장 모델**: 새로운 클러스터 생성 속도를 멱법칙(power law)으로 변경하여 유연성을 높였으나, 여전히 너무 많은 클러스터가 생성되는 근본적인 문제를 완전히 해결하지 못했다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 기본 구조

본 논문은 베이지안 추론의 기본 원칙인 $\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$를 따른다. GMM의 경우, 각 클러스터의 할당 변수 $z$, 혼합 가중치 $\pi$, 그리고 각 가우시안 성분의 파라미터 $\{\mu, \Sigma\}$를 추론해야 한다.

### 2. 핵심 구성 요소 및 Conjugate Priors

계산 효율성을 위해 다음과 같은 켤레 사전분포(Conjugate Prior)를 사용한다:

- **혼합 가중치 $\pi$**: Multinomial 분포의 켤레 사전분포인 $\text{Dirichlet}(\alpha)$를 사용한다.
- **가우시안 파라미터 $\{\mu, \Sigma\}$**: Multivariate Gaussian 분포의 켤레 사전분포인 **Normal-Inverse-Wishart (NIW)** 분포를 사용한다.
  - $\Sigma \sim \text{Inverse-Wishart}(S_0, \nu_0)$
  - $\mu|\Sigma \sim \text{Normal}(m_0, \frac{1}{\kappa_0}\Sigma)$

### 3. 추론 절차: Gibbs Sampling

파라미터들의 결합 분포를 직접 샘플링하기 어려우므로, 조건부 분포를 이용해 순차적으로 샘플링하는 Gibbs Sampler를 사용한다.

- **Uncollapsed Gibbs Sampling**: $z \to \pi \to \{\mu, \Sigma\}$ 순으로 모든 변수를 샘플링한다.
- **Collapsed Gibbs Sampling**: 켤레 사전분포의 성질을 이용하여 $\pi, \mu, \Sigma$를 적분하여 제거하고, 오직 할당 변수 $z$만을 샘플링함으로써 차원을 축소하고 수렴 속도를 높인다.

### 4. Powered Chinese Restaurant Process (pCRP)

본 논문의 핵심 아이디어로, 기존 CRP의 할당 확률을 다음과 같이 수정한다.

기존 CRP에서는 데이터 $x_i$가 기존 클러스터 $k$에 할당될 확률이 $N_{k,-i}$ (해당 클러스터의 인원수)에 비례했다. pCRP에서는 여기에 지수 $r (>1)$을 적용한다:

$$p(z_i=k | z_{-i}, \alpha) \propto \begin{cases} N_{k,-i}^r & \text{if } k \text{ is occupied} \\ \alpha & \text{if } k \text{ is a new table} \end{cases}$$

이 수정은 $r$ 값이 커질수록 인원수가 많은 클러스터에 할당될 확률을 비약적으로 높여, 작은 클러스터들이 살아남지 못하고 큰 클러스터로 흡수되게 만드는 **강력한 'Rich-get-richer' 효과**를 유도한다.

### 5. Pruning 방법론

- **cSampling**: 매 $s$번의 반복마다 일정 임계값(예: 전체의 4%)보다 작은 클러스터를 제거하고 해당 데이터를 큰 클러스터로 재할당한다.
- **lSampling**: 클러스터를 제거했을 때 전체 손실 함수(Loss function)가 감소하는지 확인하여 제거 여부를 결정한다. 이때 손실 함수로 **Squared Inertia** (각 클러스터 내 제곱합의 제곱근의 합)를 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 합성 데이터(Sim 1, Sim 2), MNIST 숫자 데이터(1-4), Old Faithful Geyser 데이터.
- **비교 대상**: Standard CRP, CRP-Oracle (최적의 $\alpha$를 미리 알고 있는 경우), pCRP.
- **평가 지표**:
  - **NMI (Normalized Mutual Information)**: 정답 레이블과 예측 레이블 간의 유사도 (높을수록 좋음).
  - **VI (Variation of Information)**: 두 클러스터링 간의 정보 차이 (낮을수록 좋음).
  - **$K$ (Average cluster number)**: 추정된 평균 클러스터 수.

### 2. 주요 결과

- **클러스터 수 추정**: pCRP는 CRP와 CRP-Oracle에 비해 실제 정답 클러스터 수 $K_0$에 가장 가깝게 추정하는 경향을 보였다. 특히 샘플 수 $N$이 증가할 때, CRP는 $K$를 계속 과다 추정하는 반면, pCRP는 정답 수로 수렴하는 양상을 보였다.
- **정량적 지표**:
  - MNIST 데이터셋 실험에서 pCRP는 CRP보다 낮은 VI 값을 기록하며 더 정밀한 클러스터링 성능을 보였다.
  - Old Faithful Geyser 데이터에서 CRP가 4개의 클러스터를 생성한 반면, pCRP는 정답인 2개의 클러스터를 정확히 찾아냈다.
- **시각적 분석**: pCRP의 결과물은 CRP에서 나타나는 작은 '노이즈' 클러스터들이 제거되어 훨씬 깨끗한(cleaner) 군집화 결과를 보여주었다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석

- pCRP는 기존의 교환 가능성(Exchangeability) 가정을 일부 위반함으로써, 현재의 군집 패턴을 학습하여 새로운 군집 생성을 억제하는 **피드백 메커니즘**을 성공적으로 구현하였다.
- $r$이라는 단일 하이퍼파라미터를 통해 모델의 복잡도(Sparsity)를 직관적으로 제어할 수 있으며, 이는 Cross-Validation을 통해 데이터 기반으로 최적화 가능하다.

### 2. 한계 및 논의사항

- **비교환성(Non-exchangeability)**: pCRP는 데이터의 순서에 영향을 받을 수 있으므로, 이를 해결하기 위해 매 반복마다 데이터를 무작위로 섞는(Permutation) 과정이 필수적이다.
- **계산 비용**: $\alpha$에 대한 hyperprior를 적용하고 ARS(Adaptive Rejection Sampling)를 사용할 경우, 일반적인 Gamma 샘플링보다 계산 시간이 증가하는 트레이드-오프가 존재한다.

### 3. 비판적 해석

본 논문은 이론적 배경과 실용적 해결책을 잘 결합하였다. 특히, 단순히 "클러스터 수가 너무 많다"는 현상을 지적하는 데 그치지 않고, 확률 질량 함수 수준에서 $N_k^r$라는 수식을 통해 이를 수학적으로 억제한 점이 훌륭하다. 다만, $r$ 값의 선택이 결과에 매우 큰 영향을 미치는데, 이에 대한 이론적 하한/상한에 대한 논의가 더 보완되었다면 더 강력한 가이드라인이 되었을 것이다.

## 📌 TL;DR

본 논문은 베이지안 GMM 추론의 수학적 기초를 정리하고, 특히 무한 GMM(DPM)에서 발생하는 **과잉 클러스터링(Over-clustering)** 문제를 해결하기 위해 **Powered Chinese Restaurant Process (pCRP)**를 제안한다. pCRP는 할당 확률에 지수 $r$을 도입하여 작은 클러스터를 억제하고 큰 클러스터로의 통합을 유도함으로써, 실제 정답에 더 가까운 클러스터 수를 추정하고 노이즈를 효과적으로 제거한다. 이 연구는 대규모 데이터셋에서 해석 가능하고 간결한 클러스터링 결과를 얻고자 하는 향후 연구 및 실제 적용에 중요한 기여를 한다.
