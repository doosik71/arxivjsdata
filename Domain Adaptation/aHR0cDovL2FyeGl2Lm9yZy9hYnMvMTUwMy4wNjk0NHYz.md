# PAC-Bayesian Theorems for Domain Adaptation with Specialization to Linear Classifiers

Pascal Germain, Amaury Habrard, François Laviolette, Emilie Morvant (2018)

## 🧩 Problem to Solve

본 논문은 소스 도메인(Source Domain)의 레이블된 데이터만을 이용하여, 레이블이 없는 타겟 도메인(Target Domain)에서 잘 작동하는 분류기를 학습시키는 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 문제를 해결하고자 한다.

일반적인 머신러닝은 학습 데이터와 테스트 데이터가 동일한 확률 분포에서 추출되었다는 가정을 전제로 한다. 그러나 실제 환경에서는 스팸 필터링 사용자마다 이메일 특성이 다르듯, 소스 분포 $P^S$와 타겟 분포 $P^T$가 서로 다른 경우가 많다. 이러한 분포의 차이는 모델의 일반화 성능을 급격히 떨어뜨리며, 특히 타겟 도메인의 레이블이 전혀 없는 상황에서 타겟 오차(Target Error)를 최소화하는 것은 매우 어려운 과제이다. 따라서 본 연구의 목표는 PAC-Bayesian 이론을 기반으로 도메인 간의 거리를 정량화하고, 이를 통해 타겟 오차에 대한 더 타이트한(tighter) 일반화 경계(Generalization Bound)를 도출하며, 이를 실제로 최적화할 수 있는 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 기존의 최악의 경우(worst-case)를 고려하던 도메인 거리 측정 방식에서 벗어나, **$\rho$-평균 불일치( $\rho$-average disagreement)**라는 새로운 유사-거리(pseudometric)를 도입하여 PAC-Bayesian 프레임워크 내에서 도메인 적응 문제를 공식화한 것이다.

1.  **새로운 도메인 거리 척도 제안**: 가설 공간 $H$ 위에서의 사후 분포 $\rho$에 기반하여, 두 도메인 간의 불일치 정도를 $\rho$-평균으로 측정하는 $\text{dis}_\rho(D_S, D_T)$를 정의하였다. 이는 기존의 $H\Delta H$-divergence보다 이론적으로 항상 낮거나 같으며, 샘플을 통해 추정하기가 훨씬 용이하다.
2.  **타이트한 일반화 경계 도출**: Stochastic Gibbs Classifier에 대해 기존 연구보다 개선되고 해석이 쉬운 새로운 PAC-Bayesian 도메인 적응 경계를 유도하였다.
3.  **PBDA 알고리즘 설계**: 제안된 이론적 경계를 직접 최적화하는 선형 분류기 전용 알고리즘인 **PBDA(PAC-Bayesian Domain Adaptation)**를 설계하였다. 이 알고리즘은 소스 리스크, 도메인 불일치, 모델 복잡도(KL-divergence)의 트레이드-오프를 동시에 최적화한다.
4.  **다중 소스 도메인 적응(Multisource DA)으로 확장**: 단일 소스가 아닌 여러 소스 도메인이 존재하는 경우로 이론을 확장하여, 소스 도메인들의 혼합 분포를 고려한 일반화 경계를 제시하였다.

## 📎 Related Works

### 1. 기존 도메인 적응 접근 방식
- **인스턴스 가중치 기반 방법**: Covariate-shift 문제(마진 분포만 다르고 레이블링 함수는 동일한 경우)를 해결하기 위해 사용된다.
- **자기 레이블링(Self-labeling) 및 표현 학습**: 타겟 데이터의 레이블을 전이하거나, 소스와 타겟이 공유하는 공통 표현(Common Representation)을 학습하여 간극을 줄이는 방식이다.
- **거리 기반 방법**: 소스와 타겟 분포 사이의 거리를 최소화하는 방식이다. 대표적으로 Ben-David 등이 제안한 $H\Delta H$-divergence와 Mansour 등이 제안한 Discrepancy distance가 있다.

### 2. 기존 접근 방식의 한계 및 차별점
기존의 $H\Delta H$-divergence나 Discrepancy distance는 가설 집합 $H$ 내의 두 분류기가 가질 수 있는 **최대 불일치(supremum)**를 측정한다. 이는 계산적으로 어렵고, 특정 가설 $h$의 성능과는 독립적인 특성을 가진다. 반면, 본 논문이 제안하는 $\text{dis}_\rho$는 **평균 불일치**를 측정하므로, 현재 학습 중인 사후 분포 $\rho$에 직접적으로 의존하며, 이를 통해 리스크 최소화와 도메인 거리 최소화를 하나의 목적 함수 내에서 동시에 최적화할 수 있다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. $\rho$-평균 도메인 불일치 ($\text{dis}_\rho$)
본 논문은 두 도메인의 마진 분포 $D_S, D_T$ 사이의 거리를 다음과 같이 정의한다.

$$\text{dis}_\rho(D_S, D_T) \stackrel{\text{def}}{=} |R^D_T(G_\rho, G_\rho) - R^D_S(G_\rho, G_\rho)|$$

여기서 $R^D(G_\rho, G_\rho)$는 분포 $D$ 상에서 $\rho$에 의해 추출된 두 분류기가 서로 다른 예측을 내놓을 확률(Expected Disagreement)을 의미한다.

### 2. PAC-Bayesian 도메인 적응 경계
Stochastic Gibbs Classifier $G_\rho$의 타겟 리스크 $R^{P_T}(G_\rho)$에 대한 경계는 다음과 같이 유도된다 (Theorem 9).

$$R^{P_T}(G_\rho) \le R^{P_S}(G_\rho) + \frac{1}{2}\text{dis}_\rho(D_S, D_T) + \lambda_\rho$$

여기서 $\lambda_\rho$는 타겟 도메인과 소스 도메인 간의 기대 공동 오차(expected joint error)의 편차를 의미하며, 비지도 학습 상황에서는 일반적으로 낮다고 가정하고 무시한다.

### 3. PBDA 알고리즘 (Linear Classifiers Specialization)
선형 분류기에 대해, 사전 분포 $\pi_0$와 사후 분포 $\rho_w$를 공분산 행렬이 단위 행렬인 구형 가우시안 분포로 설정한다. 이때 $\rho_w$에 의한 다수결 투표(Majority Vote) 결과는 중심 벡터 $w$를 사용하는 선형 분류기 $h_w$의 결과와 일치한다.

PBDA는 다음의 목적 함수 $G(w)$를 최소화하는 $w$를 찾는다.

$$G(w) = C \frac{1}{m} \sum_{i=1}^m \Phi_{\text{cvx}}(y^s_i \frac{w \cdot x^s_i}{\|x^s_i\|}) + A | \frac{1}{m} \sum_{i=1}^m \Phi_{\text{dis}}(\frac{w \cdot x^s_i}{\|x^s_i\|}) - \frac{1}{m} \sum_{i=1}^m \Phi_{\text{dis}}(\frac{w \cdot x^t_i}{\|x^t_i\|}) | + \frac{1}{2}\|w\|^2$$

**주요 구성 요소 설명:**
- **소스 리스크 항**: $\Phi_{\text{cvx}}$는 Gibbs 분류기의 오차를 근사하는 볼록 완화(convex relaxation) 손실 함수이다.
- **도메인 불일치 항**: $\Phi_{\text{dis}}(a) = 2\Phi(a)\Phi(-a)$이며, 소스와 타겟 샘플 간의 평균 불일치를 측정한다.
- **복잡도 항**: $\frac{1}{2}\|w\|^2$는 가우시안 분포 간의 KL-divergence $\text{KL}(\rho_w \| \pi_0)$에 해당하며, 모델의 과적합을 방지하는 정규화 역할을 한다.
- **학습 절차**: Gradient Descent를 통해 $w$를 최적화하며, 초기값으로는 소스 데이터만으로 학습된 PBGD3 모델의 가중치를 사용한다.

### 4. 다중 소스 확장
여러 소스 도메인이 있을 때, 소스 도메인들의 가중치 분포 $v$를 도입하여 혼합 소스 분포 $P^v_S$를 정의한다. 이 경우 $\text{dis}_\rho(D^v_S, D_T)$를 통해 타겟 리스크의 경계를 확장하여 정의하며, 소스 도메인 간의 가중치 $v$를 함께 학습하는 방향으로 알고리즘 확장이 가능하다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 
    - **Toy Problem**: 두 개의 초승달 모양(Two Inter-Twinning Moons) 데이터셋을 생성하고, 이를 10°에서 90°까지 회전시켜 타겟 도메인을 구축하였다.
    - **Real Dataset**: Amazon Reviews 데이터셋을 사용하여 4개 제품 카테고리 간의 12가지 도메인 적응 태스크를 수행하였다.
- **비교 대상 (Baselines)**: SVM (no adaptation), PBGD3 (no adaptation), DASVM, CODA.
- **평가 지표**: 타겟 테스트 셋에 대한 에러율(Error Rate).
- **하이퍼파라미터 튜닝**: 타겟 레이블이 없으므로 **Reverse/Circular Validation (RCV)** 기법을 사용하여 하이퍼파라미터를 선정하였다.

### 2. 주요 결과
- **Toy Problem**: 회전 각도가 커질수록(문제가 어려워질수록) PBDA가 타 방법론 대비 낮은 에러율을 기록하였다. 특히 가장 어려운 90° 회전 케이스에서 PBDA의 성능 우위가 두드러졌다.
- **Sentiment Analysis**: Amazon 데이터셋에서 PBDA는 CODA보다 평균적으로 우수했으며, DASVM과 경쟁 가능한 수준의 성능을 보였다. 
- **효율성**: PBDA는 반복적인 최적화 과정이 필요한 DASVM이나 CODA보다 훨씬 빠른 학습 속도를 보였다. 이는 단 한 번의 최적화 단계로 리스크와 거리를 동시에 줄이기 때문이다.
- **표현 학습과의 결합**: mSDA(Marginalized Stacked Denoising Autoencoders)로 학습된 공통 표현 공간 위에서 PBDA를 적용했을 때, 단순 SVM보다 더 나은 성능을 보이는 경우가 많음을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
본 논문의 가장 큰 강점은 도메인 적응이라는 난제를 **PAC-Bayesian의 평균적 관점**에서 접근하여, 이론적 정당성과 실용적 효율성을 동시에 확보했다는 점이다. 특히 $H\Delta H$-divergence와 같은 최악의 경우를 가정하는 척도보다 $\text{dis}_\rho$가 실제 데이터에서 더 타이트한 경계를 제공함을 보였으며, 이를 통해 소스 정확도를 일부 희생하더라도 타겟 일반화 성능을 높이는 적응적 행동(adaptive behavior)이 가능함을 실험적으로 입증하였다.

### 2. 한계 및 가정
- **$\lambda_\rho$의 무시**: 이론적 경계에서 $\lambda_\rho$(공동 오차의 편차) 항을 실제 알고리즘에서는 무시하였다. 이는 소스와 타겟의 레이블링 정보가 서로 관련이 있다는 강한 가정을 전제로 한다. 만약 두 도메인의 레이블링 함수가 완전히 다르다면, 본 방법론으로는 해결이 불가능하다.
- **하이퍼파라미터 의존성**: $A$와 $C$라는 하이퍼파라미터에 따라 성능 변화가 있으며, 타겟 레이블이 없는 상황에서 RCV에 의존하여 이를 튜닝해야 하는 번거로움이 있다.

### 3. 비판적 논의
본 논문은 선형 분류기에 특화된 분석을 제공하였으나, 최근의 딥러닝 기반 도메인 적응 모델들이 사용하는 신경망 구조로의 확장 가능성에 대해서는 구체적으로 다루지 않았다. 다만, mSDA와의 결합 실험을 통해 표현 학습 단계 이후의 분류기 단계에서 PBDA가 유효함을 보였다는 점은 고무적이다.

## 📌 TL;DR

본 논문은 비지도 도메인 적응 문제를 해결하기 위해, 가설 공간의 사후 분포 $\rho$를 이용한 **평균 불일치 척도 $\text{dis}_\rho$**를 제안하고 이를 바탕으로 타이트한 PAC-Bayesian 일반화 경계를 도출하였다. 이를 구현한 **PBDA 알고리즘**은 소스 리스크와 도메인 간 거리, 모델 복잡도를 동시에 최적화하며, 실험을 통해 기존 반복적 도메인 적응 방법들보다 빠르고 경쟁력 있는 성능을 보임을 증명하였다. 이 연구는 향후 딥러닝 모델의 정규화 항 설계나 다중 소스 학습 및 평생 학습(Lifelong Learning) 분야에 이론적 토대를 제공할 가능성이 높다.