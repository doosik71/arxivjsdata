# Deep Active Learning: Unified and Principled Method for Query and Training

Changjian Shui, Fan Zhou, Christian Gagné, Boyu Wang

## 🧩 Problem to Solve

Deep 신경망(DNN)은 뛰어난 성능을 보이지만, 많은 양의 레이블 데이터가 필요하다는 명확한 한계가 있습니다. Active Learning (AL)은 미레이블 데이터 풀에서 가장 정보 가치가 높은 샘플(배치)을 선별하여 레이블링 비용을 최소화하고 예측 성능을 극대화하는 것을 목표로 합니다. 그러나 기존 AL 방법은 다음과 같은 문제점을 안고 있습니다:

- **불확실성 기반 샘플링:** **샘플링 편향(sampling bias)**이 발생하여 현재 레이블링된 데이터가 전체 분포를 대표하지 못할 수 있습니다.
- **다양성 기반 샘플링:** **K-중심(K-center)** 또는 **코어셋(Core-set)**과 같은 방법은 계산 비용이 높고 대규모 데이터셋에 효율적이지 않을 수 있습니다.
- **하이브리드 전략:** 불확실성과 다양성을 모두 고려하지만, 경험적(heuristic) 접근 방식이 많고 **원칙적인(principled)** 통합 방법이 부족합니다.
- 미레이블 데이터를 활용하여 더 나은 특징 표현(feature representation)을 구축하고 DNN 가중치를 최적화하는 새로운 훈련 손실(training loss) 설계에 대한 질문이 남아 있습니다. 특히, **H-divergence**가 다양성 측정에 적합하지 않을 수 있습니다.

## ✨ Key Contributions

- **통합적이고 원칙적인 방법:** 딥 배치 Active Learning의 **질의(querying)** 및 **훈련(training)** 과정을 위한 **통합적이고 원칙적인 방법**인 WAAL(Wasserstein Adversarial Active Learning)을 제안합니다.
- **Wasserstein 거리 기반 분포 매칭:** AL의 상호작용 절차를 **분포 매칭(distribution matching)**으로 모델링하고 **Wasserstein 거리**를 사용하여 이론적 통찰력을 제공합니다. **H-divergence**에 비해 **Wasserstein 거리**가 다양성 측정에 더 적합함을 분석적으로 입증합니다.
- **새로운 훈련 손실 및 교대 최적화:** 이론적 분석에서 새로운 훈련 손실을 도출했으며, 이는 **심층 신경망(DNN) 파라미터 최적화**와 **배치 질의 선택**의 두 단계로 교대 최적화(alternative optimization)를 통해 분해됩니다.
- **최소-최대 최적화 기반 훈련:** DNN 훈련을 위한 손실은 미레이블 데이터를 활용하는 **최소-최대(min-max) 최적화 문제**로 자연스럽게 공식화됩니다. 이를 통해 더 나은 특징 표현을 학습할 수 있습니다.
- **명시적인 불확실성-다양성 트레이드오프:** 질의 배치 선택 시 **불확실성-다양성(uncertainty-diversity) 트레이드오프**를 명시적으로 고려하는 원칙을 제시합니다.
- **우수한 경험적 성능 및 효율성:** 다양한 벤치마크에서 기존 방법론 대비 **일관되게 더 나은 경험적 성능**과 **더 효율적인 질의 전략**을 달성했습니다. 특히 초기 훈련 단계에서 큰 성능 향상을 보입니다.

## 📎 Related Works

- **불확실성 기반 AL:** **Least confidence**, **Smallest margin**, **Maximum-Entropy sampling** [Settles, 2012], **Deep Bayesian AL (DBAL)** [Gal et al., 2017], **DeepFoolAL** [Mayer and Timofte, 2018].
- **다양성 기반 AL:** **K-Median** 및 **Core-set** 접근법 [Sener and Savarese, 2018].
- **하이브리드 전략:** 불확실성 및 다양성을 결합한 접근법 [Guo and Schuurmans, 2008; Wang and Ye, 2015; Yin et al., 2017; Ash et al., 2019].
- **생성 모델 활용 AL:** **GAN** 또는 **VAE**와 같은 생성 기술을 Active Learning에 적용한 연구 [Zhu and Bento, 2017; Mayer and Timofte, 2018; Tran et al., 2019; Sinha et al., 2019].
- **분포 매칭:** 레이블/미레이블 데이터셋을 구별하는 **H-divergence** 기반 접근법 [Gissin and Shalev-Shwartz, 2019] 및 **Maximum Mean Discrepancy (MMD)** 메트릭을 사용한 연구 [Chattopadhyay et al., 2013].

## 🛠️ Methodology

1. **Active Learning을 분포 매칭으로 모델링:**

   - AL의 상호작용 절차를 데이터 생성 분포 $D$와 질의 분포 $Q$ 간의 **분포 매칭** 문제로 정의합니다.
   - 손실 함수 $\mathcal{L}$이 대칭적이고 **L-Lipschitz**이며, 가설 $h \in \mathcal{H}$가 **H-Lipschitz** 함수이고, 레이블링 함수 $h^?$가 **$\phi(\lambda)$-(D,Q) Joint Probabilistic Lipschitz** 조건을 만족한다고 가정합니다.
   - 이때, $D$에 대한 예상 위험 $R_D(h)$가 질의 분포 $Q$에 대한 예상 위험 $R_Q(h)$와 **Wasserstein-1 거리** $W_1(D,Q)$에 의해 다음과 같이 상한이 결정됨을 이론적으로 증명합니다:
     $$R_D(h) \le R_Q(h) + L(H+\lambda)W_1(D,Q) + L\phi(\lambda)$$
   - **Wasserstein-1 거리**는 **H-divergence**보다 분포의 다양성을 더 잘 측정하며, 이는 샘플링 편향을 피하는 데 중요합니다.

2. **최소-최대(Min-Max) 최적화 문제로 공식화:**

   - 위의 이론적 분석을 바탕으로, 훈련 목표를 **Kantorovich-Rubinstein 쌍대성**을 활용한 최소-최대 최적화 문제로 재구성합니다:
     $$\min_{\theta_f, \theta_h, \hat{B}} \max_{\theta_d} \hat{R}(\theta_f, \theta_h) + \mu \hat{E}(\theta_f, \theta_d)$$
     여기서 $\theta_f$는 특징 추출기, $\theta_h$는 작업 예측기, $\theta_d$는 분포 비평가(critic) 파라미터를 나타냅니다. $\hat{R}$은 예측기 손실이고 $\hat{E}$는 adversarial (min-max) 손실입니다.

3. **두 단계 최적화(Two-stage Optimization):**
   - **3.1. DNN 파라미터 훈련 단계:**
     - 관찰된 모든 레이블 데이터($\hat{L}$)와 미레이블 데이터($\hat{U}$)를 사용하여 신경망 파라미터 $\theta_f, \theta_h, \theta_d$를 최적화합니다.
     - 이 과정은 예측 오차를 최소화하는 동시에, 비평가 $g(x)$가 샘플이 레이블 또는 미레이블 세트 중 어디에서 왔는지 구별하려고 노력하고(최대화), 특징 추출기 $\theta_f$는 이들을 혼란시키려 하는(최소화) **min-max adversarial training**을 포함합니다.
     - **Redundancy Trick**: 레이블/미레이블 데이터의 불균형을 해소하기 위해 레이블 데이터에 대해 **샘플링-with-replacement**를 적용하고 "bias coefficient" $C_0$를 도입하여 훈련 시 균형을 조절합니다.
   - **3.2. 질의 전략 단계:**
     - 미레이블 데이터 풀 $\hat{U}$에서 다음 질의 배치 $\hat{B}$를 선택합니다. 이 과정은 **불확실성(uncertainty)**과 **다양성(diversity)**의 트레이드오프를 명시적으로 고려합니다.
     - **불확실성 측정:** 예측 신뢰도가 낮은 샘플을 선택합니다. 다음 두 가지 상한(upper bounds) 중 하나 또는 조합을 사용합니다:
       - 가장 낮은 예측 신뢰도를 가진 샘플 (Highest least prediction confidence score).
       - 예측 신뢰도 분포가 균일한 샘플 (Uniformly of prediction confidence score).
     - **다양성 측정:** 비평가 함수 $g(x)$의 값이 높은 샘플을 선택합니다. $g(x)$ 값이 높다는 것은 해당 미레이블 샘플이 현재 레이블링된 샘플들과 **Wasserstein 거리** 측면에서 더 다르다는 것을 의미하며, 이는 더 높은 정보 가치와 다양성을 나타냅니다.
     - 최종적으로 불확실성 점수 $U(x_u)$와 비평가 출력 $g(x_u)$를 결합한 $U(x_u) - \mu g(x_u)$를 최소화하는 $B$개의 샘플을 **탐욕적(greedy)**으로 선택합니다.

## 📊 Results

- **벤치마크:** Fashion MNIST, SVHN, CIFAR-10 데이터셋에서 실험을 수행했습니다.
- **성능 우위:** 제안하는 **WAAL**은 **Random**, **Least Confidence**, **Margin**, **Entropy**, **K-Median**, **Core-set**, **DBAL**, **DeepFoolAL** 등 **모든 기준선(baselines)보다 일관되게 우수한 예측 정확도**를 보였습니다.
- **초기 훈련 단계의 개선:** 특히, 초기 훈련 단계에서 **5% 이상의 큰 성능 향상**을 보였는데, 이는 미레이블 데이터를 활용하여 효율적으로 좋은 특징 표현을 구성하는 WAAL의 능력 때문입니다.
- **효율적인 질의 전략:** **WAAL**의 질의 시간은 불확실성 기반 전략과 비슷한 **빠른 질의 속도**를 가지며, 특징 공간에서 거리 행렬 계산이 필요한 **Core-set**이나 **K-Median**과 같은 다양성 기반 방법보다 **훨씬 효율적**입니다.
- **Wasserstein 거리의 이점 확인:** **H-divergence** 기반 adversarial training을 사용한 **ablation study**에서도 **WAAL**이 여전히 우수한 성능을 보여, 딥 AL 문제에 **Wasserstein 거리**를 채택하는 것의 실질적인 이점을 재확인했습니다.

## 🧠 Insights & Discussion

- 이 연구는 **Active Learning**의 질의 및 훈련 과정을 **Wasserstein 거리**를 사용한 **분포 매칭**이라는 단일 **원칙적인 프레임워크**로 성공적으로 통합했습니다. 이는 딥 Active Learning의 이론적 기반을 강화하는 중요한 기여입니다.
- **H-divergence**가 아닌 **Wasserstein 거리**를 사용하여 **다양성**을 측정하는 것이 데이터 분포를 더 정확하게 반영하고 **샘플링 편향** 문제를 효과적으로 완화하는 데 기여함을 이론적 분석과 경험적 증거를 통해 보여주었습니다.
- 미레이블 데이터를 **min-max adversarial training**에 활용하는 방법은 모델이 **더 나은 특징 표현**을 학습하게 하여, 특히 데이터가 부족한 **초기 훈련 단계**에서 모델의 성능을 크게 향상시킬 수 있습니다.
- **불확실성**과 **다양성**의 **명시적인 트레이드오프**를 통합한 질의 전략은 정보 가치가 높으면서도 전체 데이터 분포를 잘 대표하는 샘플을 효율적으로 선택할 수 있도록 합니다. 이는 AL 시스템의 전반적인 효율성과 견고성을 높입니다.
- **한계 및 미래 연구 방향:**
  - 다양한 분포 발산 메트릭(divergence metrics)이 학습 시나리오에 미치는 영향을 더 깊이 이해하는 것이 필요합니다.
  - 적대적 훈련(adversarial training) 외에 **오토인코더(auto-encoder) 기반** 접근 방식과 같은 다른 실용적인 원칙을 탐색하여 딥 AL 알고리즘의 설계를 개선할 수 있습니다.

## 📌 TL;DR

이 논문은 **딥 Active Learning**의 **레이블링 효율성**과 **훈련 성능**을 개선하기 위한 **통합적이고 원칙적인 방법(WAAL)**을 제안합니다. WAAL은 AL의 상호작용을 **Wasserstein 거리** 기반 **분포 매칭**으로 모델링하여 새로운 **min-max 훈련 손실**을 도출하고, **불확실성-다양성 트레이드오프**를 명시적으로 고려하는 **질의 전략**을 개발했습니다. 실험 결과, WAAL은 미레이블 데이터를 효과적으로 활용하여 **일관되게 우수한 성능**을 달성하고 **빠른 질의 속도**를 제공하며, 특히 **초기 훈련 단계**에서 큰 성능 향상을 보였습니다.
