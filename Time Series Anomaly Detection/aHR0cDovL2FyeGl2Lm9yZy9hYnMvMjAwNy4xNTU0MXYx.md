# Anomaly Detection at Scale: The Case for Deep Distributional Time Series Models

Fadhel Ayed, Lorenzo Stella, Tim Januschowski, Jan Gasthaus (2020)

## 🧩 Problem to Solve

본 논문은 대규모 분산 시스템 및 클라우드 환경에서 (마이크로) 서비스와 클라우드 리소스의 상태를 모니터링하기 위한 이상치 탐지(Anomaly Detection) 문제를 다룬다. 클라우드 모니터링 환경에서 효과적인 이상치 탐지 시스템을 구축하기 위해서는 다음과 같은 세 가지 주요 도전 과제를 해결해야 한다.

첫째, 모니터링 대상이 되는 메트릭의 수가 수백만 개에 달하고 데이터가 스트리밍 형태로 유입되기 때문에, 지도 학습(Supervised Learning)에 필요한 충분한 양의 레이블링된 데이터를 확보하기 어렵다. 또한 레이블링 자체가 주관적일 수 있어 비지도 학습(Unsupervised Learning) 방식의 접근이 필수적이다.

둘째, 수많은 시계열 데이터를 실시간에 가깝게 처리해야 하므로 모델의 계산 효율성과 확장성(Scalability)이 매우 중요하다. 특히 수백만 개의 메트릭을 사람이 일일이 튜닝할 수 없으므로, 수동 개입 없이도 견고한 성능을 내는 Out-of-the-box 성능이 요구된다.

셋째, CPU 사용량, 지연 시간(Latency), 에러율 등 서로 다른 성격의 시계열 데이터를 처리할 수 있는 유연성이 필요하며, 단순한 점 이상치(Point anomalies)뿐만 아니라 집단 이상치(Collective anomalies)나 문맥적 이상치(Contextual anomalies)를 모두 탐지할 수 있어야 한다.

특히, 기존의 방식들은 시계열 데이터를 고정된 시간 간격으로 집계할 때 평균이나 중앙값 같은 단일 통계량(Summary statistics)만을 사용한다. 그러나 이는 특정 분위수(Quantile)에서만 나타나는 이상 징후를 놓칠 위험이 크다는 치명적인 한계가 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시계열 데이터를 실수 값이나 벡터의 연속으로 모델링하는 대신, **실수 값에 대한 확률 분포(Probability Distributions)의 시계열**로 모델링하는 것이다.

중심적인 설계 직관은 다음과 같다. 데이터를 특정 시간 간격으로 집계할 때 단일 통계량만 추출하는 것이 아니라, 해당 구간 내 모든 관측치의 전체 분포를 유지함으로써 정보 손실을 최소화하는 것이다. 이를 위해 딥러닝 기반의 확률적 분포 시계열 모델을 제안하며, 구체적으로는 Autoregressive LSTM 기반의 순환 신경망(RNN)을 사용하여 분포의 시간적 진화를 예측한다. 이를 통해 기존 방식으로는 탐지가 불가능했던 '집단 이상치(예: 평균은 일정하지만 분산이 급격히 감소하는 경우)'를 효과적으로 탐지할 수 있게 되었다.

## 📎 Related Works

논문은 기존의 시계열 이상치 탐지 방법을 크게 세 가지 방향으로 설명하며 본 연구와의 차별점을 제시한다.

1. **전통적 시계열 모델 및 딥러닝 모델:** ARIMA와 같은 고전적 모델이나 일반적인 LSTM 기반 모델들은 주로 점 예측(Point prediction)에 집중한다. 이러한 모델들은 스트리밍 환경에서 유용하지만, 분포 전체를 고려하지 않으므로 분포의 형태가 변하는 집단 이상치를 탐지하는 데 한계가 있다.
2. **함수적 데이터 분석(Functional Data Analysis, FDA):** 함수적 시계열(Functional Time Series, FTS) 모델들은 데이터 포인트를 함수로 취급하여 예측한다. 본 연구는 이와 유사한 맥락에 있으나, 일반적인 함수가 아닌 '확률 분포'라는 제약 조건(비음수성, 적분값 1 등)을 가진 데이터를 딥러닝 모델과 결합하여 대규모 클라우드 환경에 적용했다는 점에서 차별화된다.
3. **분포 회귀(Distribution Regression):** 일부 연구들이 분포 간의 회귀를 다루지만, 이들은 주로 시계열의 자기상관성(Auto-correlation)을 무시하고 단순 회귀 문제로 접근한다. 반면, 본 제안 방법은 RNN을 통해 시계열의 시간적 의존성을 명시적으로 모델링한다.

## 🛠️ Methodology

### 1. 분포 시계열의 표현 (Distributional Time Series)

본 모델은 각 시간 단계 $t$에서의 확률 분포 $F_t$를 직접 다루기 위해, 이를 **빈드 밀도(Binned Densities)** 형태로 근사한다.

- **Binned Representation:** 관찰 영역 $Y = [y_{min}, y_{max}]$를 $d$개의 빈(bin)으로 나누고, 각 빈에 속할 확률을 원소로 가지는 확률 벡터 $p_t = (p_{t1}, \dots, p_{td})$로 표현한다. 이는 CDF(누적 분포 함수)를 구간별 선형 함수로 근사하는 것과 동일하다.
- **Dirichlet Distribution:** 확률 벡터 $p_t$의 분포를 모델링하기 위해 디리클레 분포(Dirichlet distribution)를 도입한다.
  $$p_t \sim \text{Dir}(\alpha_t)$$
  여기서 $\alpha_t \in \mathbb{R}^d_+$는 집중 매개변수(Concentration parameter)이며, 이 $\alpha_t$의 시간적 변화를 RNN이 학습하게 된다.

### 2. 데이터 설정에 따른 우도(Likelihood) 계산

데이터가 유입되는 방식(표본의 수 $n_t$)에 따라 두 가지 설정으로 우도를 계산한다.

- **Asymptotic Setting ($n_t \to \infty$):** 표본이 매우 많아 분포 $p_t$를 직접 관찰할 수 있는 경우, 디리클레 분포의 확률 밀도 함수를 사용한다.
  $$L_t = \text{Dir}(p_t; \alpha_t)$$
- **Finite $n_t$ Setting:** 표본의 수가 제한적인 경우, $p_t$를 주변화(marginalize)하여 디리클레-다항 분포(Dirichlet-Multinomial distribution)를 사용한다.
  $$L_t = \text{Dir-Mult}(m_t; n_t, \alpha_t)$$
  여기서 $m_t$는 각 빈에 들어간 관측치 수의 벡터이다.

### 3. RNN Temporal Dynamics Model

$\alpha_t$의 동역학을 학습하기 위해 Autoregressive LSTM을 사용한다.

- **구조:** 이전 시점의 은닉 상태 $h_{t-1}$, 이전 관측치 $z_{t-1}$, 그리고 외부 공변량(Covariates) $x_t$를 입력으로 받아 현재의 은닉 상태 $h_t$를 업데이트한다.
  $$h_t = r_\phi(h_{t-1}, z_{t-1}, x_t)$$
  $$\alpha_t = \theta_\phi(h_t)$$
- **학습 목표:** 음의 로그 우도(Negative Log Likelihood, NLL)를 최소화하는 방향으로 파라미터 $\phi$를 학습한다.
  $$\mathcal{L} = -\sum_{t=1}^T \log(L_t)$$

### 4. 이상치 탐지 절차 (Level Sets)

예측된 $\alpha_{T+1}$을 바탕으로 새로운 관측치 $z_{T+1}$의 이상 여부를 판별한다.

- **Credible Region:** 우도 함수의 레벨 셋(Level-set) $S_{T+1}(\eta) = \{z : L_{T+1}(z) \geq \eta\}$를 구성한다.
- **판별 기준:** $\mathbb{P}(Z_{T+1} \in S_{T+1}(\eta_{T+1})) = 1 - \epsilon$을 만족하는 임계값 $\eta_{T+1}$을 설정하고, 만약 $L_{T+1}(z_{T+1}) < \eta_{T+1}$이면 이상치로 판단한다.
- **Monte Carlo Approximation:** $\eta_{T+1}$을 정확히 계산하기 어려운 경우, $\alpha_{T+1}$로부터 $M$개의 샘플을 생성하여 그 우도 값들의 $\epsilon$-분위수를 통해 $\hat{\eta}_{T+1}$을 추정한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:**
  - 합성 데이터(Synthetic): 평균 이동 및 분산 붕괴(Collapse) 시나리오.
  - Yahoo Webscope: 공개 벤치마크 데이터셋(A1~A4).
  - AWS Internal Data: 실제 클라우드 메트릭 데이터(B1~B3).
- **평가 지표:** ROC-AUC를 사용하여 임계값에 독립적인 모델의 변별력을 측정한다.
- **비교 대상:** Luminol, TwitterAD, iForest, OCSVM, LOF, PCA, DeepAnT, FuseAD 등.

### 2. 주요 결과

- **합성 데이터:** 제안 방법은 분포의 변화(평균 및 분산 변화)를 매우 높은 정확도로 탐지하였다. 특히 기존의 통계량 기반 방식(TwitterAD, Luminol)이 분산 변화(DS1$\sigma$, DS2$\sigma$) 탐지에서 고전하는 반면, 제안 방법은 압도적인 AUC를 기록하였다.
- **Yahoo Webscope:** 저빈도 설정($n_t=1$)에서도 SOTA 모델들과 경쟁 가능한 성능을 보였으며, 특히 A2, A3 벤치마크에서 매우 높은 AUC를 달성하였다.
- **AWS 내부 데이터:** 실시간 스트리밍 환경을 가정한 실험에서 Luminol 및 TwitterAD 대비 월등한 성능 향상을 보였다. 일부 데이터셋에서는 평균적으로 최대 17%의 성능 개선이 관찰되었다.

### 3. 확장성 및 효율성

- **추론 속도:** 단일 데이터 포인트의 스코어링에 약 1ms가 소요된다.
- **처리량:** 표준 16코어 EC2 인스턴스 한 대당 분당 약 65,000개의 메트릭을 처리할 수 있으며, 수평적 확장이 용이하다.
- **메모리:** 메트릭당 약 80kb의 고정된 모델 상태 크기를 유지한다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **데이터 집계 단계에서 발생하는 정보 손실 문제를 확률 분포 모델링이라는 수학적 프레임워크로 해결**했다는 점이다. 기존의 점 예측 기반 모델들이 잡지 못하는 '분포의 모양 변화'를 포착함으로써, 실제 클라우드 환경에서 빈번하게 발생하는 복합적인 이상 징후를 탐지할 수 있게 되었다.

또한, 이론적 정교함뿐만 아니라 실제 배포 가능성(Deployability)을 충분히 고려하였다. LSTM의 컴팩트한 상태 표현을 통해 스트리밍 데이터 처리에 적합하도록 설계되었으며, MXNet/GluonTS 기반의 구현을 통해 대규모 확장성을 입증하였다.

다만, 한계점으로는 AWS 내부 데이터 실험에서 나타났듯이 레이블의 품질이 일정하지 않은 경우(Noisy labels) 성능 지표가 낮게 측정될 수 있다는 점이 언급된다. 이는 모델의 결함이라기보다 실제 현장 데이터의 레이블링 난이도에서 오는 문제이며, 향후 레이블을 학습 과정에 통합하는 지도/반지도 학습으로의 확장이 필요함을 시사한다.

## 📌 TL;DR

이 논문은 클라우드 리소스 모니터링을 위해 시계열 데이터를 단일 값이 아닌 **확률 분포의 시퀀스**로 모델링하는 딥러닝 기반 이상치 탐지 방법을 제안한다. Dirichlet 분포와 Autoregressive LSTM을 결합하여 분포의 시간적 진화를 예측하며, 이를 통해 기존 방식으로는 불가능했던 분산 변화 기반의 집단 이상치(Collective Anomaly)를 효과적으로 탐지한다. 합성 데이터, Yahoo, AWS 데이터셋 모두에서 SOTA 및 상용 도구 대비 우수한 성능을 보였으며, 수백만 개의 메트릭을 실시간으로 처리할 수 있는 높은 확장성을 갖추어 실제 대규모 시스템 적용 가능성이 매우 높다.
