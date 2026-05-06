# Unsupervised Anomaly Detection in Multivariate Time Series across Heterogeneous Domains

Vincent Jacob, Yanlei Diao (2025)

## 🧩 Problem to Solve

본 논문은 AIOps(Artificial Intelligence for IT Operations) 환경에서 발생하는 다변량 시계열 데이터의 이상치 탐지(Anomaly Detection, AD) 문제를 다룬다. 특히, 소프트웨어/하드웨어 구성 요소의 변경이나 운영 컨텍스트의 변화로 인해 '정상 동작(Normal Behavior)'의 분포가 변화하는 **Domain Shift** 문제를 해결하는 것을 목표로 한다.

기존의 비지도 학습 기반 이상치 탐지 방법들은 학습 데이터와 테스트 데이터의 정상 분포가 동일하다는 가정을 전제로 한다. 그러나 실제 AIOps 시나리오에서는 동일한 서비스 엔티티라 하더라도 실행 환경(컨텍스트)에 따라 정상 데이터의 분포가 크게 달라지며, 이는 기존 모델들이 테스트 단계에서 정상 데이터를 이상치로 오판하게 만들어 성능을 저하시키는 주요 원인이 된다. 따라서 본 연구의 목표는 서로 다른 도메인(컨텍스트) 간에 일반화될 수 있는 **Domain-Invariant Representation**을 학습하여, 학습 시 보지 못한 새로운 도메인에서도 효과적으로 이상치를 탐지하는 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **특징 분리(Feature Disentanglement)**를 통해 입력 데이터를 도메인에 의존적인 요소와 도메인에 무관한 요소로 분리하는 것이다.

저자들은 이상치가 도메인에 무관한(Domain-Invariant) 속성에 유의미한 영향을 미칠 것이라는 직관에서 출발한다. 즉, 메모리 사용량이나 처리 지연 시간과 같은 지표들은 도메인(컨텍스트)마다 다르게 나타날 수 있지만(Domain-Specific), 시스템이 정상적으로 작동하는지 여부를 결정짓는 핵심 성능 지표(KPI)의 패턴은 도메인과 관계없이 일정하게 유지된다는 점을 이용한다. 이를 위해 **DIVAD(Domain-Invariant VAE for Anomaly Detection)**라는 새로운 프레임워크를 제안하며, 이는 VAE를 기반으로 데이터를 도메인 특화 잠재 변수 $z_d$와 도메인 불변 잠재 변수 $z_y$로 분리하여 학습한다.

## 📎 Related Works

논문에서는 기존의 다변량 시계열 이상치 탐지 방법론을 다섯 가지 카테고리로 분류하여 설명한다.

1. **Forecasting methods**: LSTM-AD와 같이 미래 값을 예측하고 실제 값과의 오차를 기반으로 스코어링한다.
2. **Reconstruction methods**: PCA, Autoencoder, TranAD 등이 있으며, 데이터를 저차원 공간으로 투영 후 복원했을 때의 재구성 오차를 사용한다.
3. **Encoding methods**: Deep SVDD와 같이 데이터를 특정 초구(Hypersphere) 내로 매핑하고 중심점으로부터의 거리를 측정한다.
4. **Distribution methods**: VAE, OmniAnomaly 등이 있으며, 데이터의 분포를 추정하고 로그 가능도(Log-likelihood)를 기반으로 스코어링한다.
5. **Isolation tree methods**: Isolation Forest가 대표적이며, 데이터를 고립시키는 데 필요한 경로 길이를 측정한다.

**기존 방식의 한계 및 차별점**:
위 방법들은 모두 학습 및 테스트 데이터의 분포가 유사하다고 가정하므로 Domain Shift에 취약하다. Domain Generalization(DG) 연구가 이미지 분류 분야에서는 활발하나, 시계열 AD 분야, 특히 비지도 학습 기반의 다변량 데이터 설정에서는 거의 다루어지지 않았다. DIVAD는 이러한 공백을 메우기 위해 특성 분리 기반의 DG를 비지도 시계열 AD에 최초로 적용하였다.

## 🛠️ Methodology

### 전체 시스템 구조

DIVAD는 입력 데이터 $x$를 두 개의 독립적인 잠재 변수 $z_d$ (Domain-specific)와 $z_y$ (Domain-invariant)로 분리하는 **Multi-encoder architecture**를 가진다.

- **$z_d$ (Domain-specific factor)**: 관측된 도메인 $d$에 조건부로 결정되며, 도메인 간의 차이를 캡처한다.
- **$z_y$ (Domain-invariant factor)**: 도메인 $d$와 독립적이며, 정상/이상 여부를 판단하는 핵심 정보를 담고 있다.

### 학습 목표 및 손실 함수

모델은 최대 가능도 추정(Maximum Likelihood Estimation)을 위해 VAE의 **ELBO(Evidence Lower Bound)**와 **도메인 분류 손실**을 결합하여 최적화한다.

$$ \mathcal{L} = \mathcal{L}_{ELBO} + \alpha_d \mathcal{L}_d $$

1. **$\mathcal{L}_{ELBO}$**: 입력 $x$를 $z_d, z_y$로 인코딩한 후 다시 $x$로 복원하는 능력을 학습시킨다.
    $$ \mathcal{L}_{ELBO} = \mathbb{E}_{q}[ \log p_{\theta_{yd}}(x|z_d, z_y) ] - \beta KL(q_{\phi_y}(z_y|x) \| p(z_y)) - \beta KL(q_{\phi_d}(z_d|x) \| p_{\theta_d}(z_d|d)) $$
    - 첫 번째 항은 재구성 오차를 최소화한다.
    - 두 번째 항은 $z_y$를 도메인 불변 사전 분포 $p(z_y)$에 가깝게 정규화한다.
    - 세 번째 항은 $z_d$를 해당 도메인의 사전 분포 $p_{\theta_d}(z_d|d)$에 가깝게 정규화한다.

2. **$\mathcal{L}_d$ (Domain Classification Loss)**: $z_d$만으로 도메인 $d$를 정확히 분류할 수 있도록 하여, 도메인 특화 정보가 $z_d$에 집중되게 강제한다.
    $$ \mathcal{L}_d = \mathbb{E}_{q_{\phi_d}(z_d|x)} [ \log q_{\omega_d}(d|z_d) ] $$

### 추론 및 이상치 스코어링 절차

테스트 단계에서는 입력 $x$를 도메인 불변 공간 $z_y$로 매핑한 후, 학습된 정상 분포와의 거리를 기반으로 이상치 점수를 계산한다.

$$ f_w(x) = -\log p(z_y = f_y(x)) $$

여기서 $p(z_y)$를 모델링하는 세 가지 대안을 제시한다:

- **DIVAD-G**: $p(z_y)$를 고정된 표준 가우시안 분포 $\mathcal{N}(0, I)$로 가정한다.
- **DIVAD-GM**: $p(z_y)$를 학습 가능한 가우시안 혼합 모델(Gaussian Mixture Model, GMM)로 모델링하여 복잡한 정상 분포를 더 정밀하게 표현한다.
- **Aggregated Posterior**: 사전 분포 대신 학습 데이터의 인코딩 결과물들의 집합(Aggregated Posterior)을 직접 밀도 추정하여 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Exathlon (Spark streaming 애플리케이션 로그), ASD (Application Server Dataset).
- **지표**: Peak F1-score (정밀도-재현율 곡선에서 최적의 임계값을 선택했을 때의 F1-score).
- **비교 대상**: TranAD, OmniAnomaly, LSTM-AD, Deep SVDD 등 13가지 비지도 AD 방법론.

### 주요 결과

1. **성능 향상**: Exathlon 벤치마크에서 **Dense DIVAD-GM**은 SOTA 모델인 TranAD의 최대 Peak F1-score(0.66)보다 각각 **20% 및 15% 향상된 0.79 및 0.76**을 달성하였다.
2. **도메인 일반화 능력**: t-SNE 시각화를 통해 $z_d$ 공간에서는 도메인별 클러스터가 명확히 구분되는 반면, $z_y$ 공간에서는 도메인 간 경계가 사라진 것을 확인하여 성공적인 Feature Disentanglement가 이루어졌음을 입증하였다.
3. **ASD 데이터셋 적용**: 새로운 서버(unseen server)를 테스트 셋으로 사용하는 설정에서, Rec DIVAD-GM이 12개 서버 중 11개에서 TranAD보다 높은 성능을 보이며 광범위한 적용 가능성을 보여주었다.
4. **분석**: 분포 기반 방법(Distribution methods)들이 일반적으로 Domain Shift에 가장 취약했으나, DIVAD는 도메인 불변 공간에서 분포를 모델링함으로써 이 문제를 해결하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

DIVAD는 도메인 간의 공통된 '정상성'을 추출하여 모델링함으로써, 기존 모델들이 겪던 "정상 데이터의 분포 변화를 이상치로 오인하는 문제"를 효과적으로 해결하였다. 특히 가우시안 혼합 모델(GMM)을 사용한 DIVAD-GM이 더 좋은 성능을 보인 것은, 실제 시스템의 정상 동작이 단일 분포가 아닌 다봉분포(Multimodal distribution) 특성을 갖기 때문으로 해석된다.

### 한계 및 논의사항

- **Point vs Sequence Modeling**: Exathlon 데이터셋에서는 포인트 모델링(Point modeling)이 시퀀스 모델링(Sequence modeling)보다 더 좋은 성능을 보였다. 이는 해당 데이터셋의 이상치가 주로 특정 시점의 값들이 컨텍스트에 따라 변하는 'Contextual Anomaly' 형태였기 때문에, 복잡한 시퀀스 패턴을 도메인 불변하게 학습하는 것보다 단일 시점의 불변 특징을 찾는 것이 더 효율적이었기 때문이다.
- **학습 비용**: VAE 대비 인코더가 2개 필요하므로 학습 시간이 약 2배 정도 소요된다. 하지만 추론 시에는 $z_y$ 인코더만 사용하므로 VAE보다 오히려 빠르게 작동한다.

## 📌 TL;DR

본 논문은 AIOps 환경에서 정상 동작의 분포가 변하는 **Domain Shift** 문제를 해결하기 위해, 입력 데이터를 도메인 특화 요소와 불변 요소로 분리하는 **DIVAD** 프레임워크를 제안한다. VAE 기반의 특징 분리(Feature Disentanglement)를 통해 학습하지 않은 새로운 도메인에서도 강건한 이상치 탐지가 가능함을 보였으며, Exathlon 벤치마크에서 기존 SOTA 모델 대비 최대 20%의 F1-score 향상을 달성하였다. 이 연구는 동적인 IT 운영 환경에서 지속적으로 변화하는 정상 상태를 반영해야 하는 실무적인 이상치 탐지 시스템 구축에 중요한 기여를 할 것으로 보인다.
