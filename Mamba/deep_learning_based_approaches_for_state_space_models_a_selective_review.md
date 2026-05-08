# Deep Learning-based Approaches for State Space Models: A Selective Review

Jiahe Lin, George Michailidis (2024)

## 🧩 Problem to Solve

본 논문은 동역학 시스템(dynamical system) 분석을 위한 강력한 프레임워크인 상태 공간 모델(State Space Models, 이하 SSM)에 딥러닝 기술을 접목한 최신 연구 동향을 체계적으로 분석하고 리뷰하는 것을 목표로 한다.

전통적인 SSM은 시스템의 시간적 동역학이 잠재 상태(latent state)의 진화에 의해 결정되고, 이 상태가 관측값(observation)을 지배한다고 가정한다. 그러나 기존의 선형 가우시안 SSM(Linear Gaussian SSM)은 현실의 복잡한 비선형성이나 비가우시안 특성을 반영하기 어렵다는 한계가 있다. 특히 비선형 모델의 경우, 고차원 적분 계산의 복잡성으로 인해 최대 가능도 추정(Maximum Likelihood Estimation, MLE)과 같은 고전적 학습 방법론을 적용하는 데 큰 어려움이 따른다.

따라서 본 논문은 신경망을 통한 유연한 파라미터화와 변분 오토인코더(Variational Autoencoder, VAE) 기반의 학습 파이프라인을 통해 이러한 한계를 극복하려는 딥러닝 기반 SSM 접근 방식을 통합적인 관점에서 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 고전적 SSM부터 최신 딥러닝 기반의 시퀀스 모델링 모듈에 이르기까지의 흐름을 하나의 통일된 시각으로 정리한 것이다. 주요 기여 사항은 다음과 같다.

1. **통합적 관점 제시**: 이산 시간(discrete-time) 딥 SSM과 연속 시간(continuous-time) 모델(Latent Neural ODE/SDE 등)을 포괄하는 통합적인 리뷰를 제공한다.
2. **학습 패러다임의 전환 분석**: 고전적인 필터링(filtering) 및 스무딩(smoothing) 기법에서 VAE 기반의 인코더-디코더 구조로 전환되는 과정과 그 기술적 이점을 상세히 분석한다.
3. **구조적 모듈로서의 SSM 분석**: SSM을 단순한 생성 모델이 아닌, Transformer의 연산 복잡도 문제를 해결하기 위한 입력-출력 매핑 모듈(예: S4, Mamba)로서의 관점을 분석한다.
4. **실제 적용 사례 제시**: 혼합 빈도 데이터(mixed-frequency data) 및 불규칙하게 샘플링된 시계열 데이터 처리에서 SSM이 가지는 이점을 구체적인 사례를 통해 설명한다.

## 📎 Related Works

논문은 SSM의 발전 과정을 다음과 같은 관련 연구 흐름으로 설명한다.

- **고전적 SSM**: Kalman Filter를 중심으로 한 선형 가우시안 모델이 주를 이루었으며, 비선형 시스템을 위해 Extended Kalman Filter(EKF), Unscented Kalman Filter(UKF) 등이 제안되었다. 하지만 이들은 강한 비선형성 앞에서 성능이 저하되거나 계산 복잡도가 급격히 증가하는 한계가 있다.
- **샘플링 기반 접근법**: Particle Filtering과 같은 Sequential Monte Carlo(SMC) 방법론이 비가우시안/비선형 문제를 해결하려 했으나, 상태 차원이 높아질수록 성능이 떨어지는 '차원의 저주' 문제가 존재한다.
- **딥러닝 기반 SSM**: 신경망을 통해 상태 전이 함수와 관측 함수를 파라미터화하여 유연성을 확보하고, VAE를 통해 고차원 적분 문제를 최적화 문제로 변환하여 해결한다.
- **시퀀스 모델링 모듈**: 최근의 S4, Mamba 등은 SSM을 확률 모델이 아닌 신경망의 한 레이어로 사용하여, Transformer의 $\mathcal{O}(L^2)$ 복잡도를 $\mathcal{O}(L \log L)$ 또는 $\mathcal{O}(L)$로 낮추면서도 긴 의존성(long-range dependency)을 캡처하려 한다.

## 🛠️ Methodology

### 1. 기본 SSM 정식화 (General Formulation)

SSM은 다음과 같은 기본 구조를 가진다.

- **상태 방정식 (State Equation)**: $z_t = f_t(z_{t-1}, u_t; \theta)$ (잠재 상태의 진화)
- **관측 방정식 (Observation Equation)**: $x_t = g_t(z_t, \epsilon_t; \theta)$ (상태로부터 관측값 생성)
여기서 $z_t$는 잠재 상태, $x_t$는 관측값, $\theta$는 모델 파라미터이다.

### 2. VAE 기반 학습 파이프라인

딥러닝 기반 SSM은 대부분 VAE 구조를 사용하여 $\theta$와 $\phi$(인코더 파라미터)를 학습한다. 학습 목표는 증거 하한선(Evidence Lower Bound, ELBO)을 최대화하는 것이다.
$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) \parallel p_\theta(z))$$

- **인코더 ($q_\phi$)**: 관측 데이터 $x$를 통해 잠재 상태 $z$의 근사 사후 분포를 학습한다.
- **디코더 ($p_\theta$)**: 샘플링된 $z$를 통해 데이터를 재구성하며, 시스템의 동역학(dynamics)을 모사한다.

### 3. 연속 시간 모델 (Neural ODE & SDE)

- **Neural ODE**: 상태의 변화를 상미분 방정식(ODE)으로 정의한다.
    $$\frac{dz(t)}{dt} = f_\theta(z(t))$$
    ODE Solver를 통해 임의의 시간 $t$에서의 상태를 계산할 수 있어 불규칙한 샘플링 데이터 처리에 유리하다.
- **Neural SDE**: ODE에 확산 항(diffusion term)을 추가하여 확률적 변동성을 부여한다.
    $$dz(t) = f_\theta(z(t))dt + \sigma(z(t))dw(t)$$
    여기서 $dw(t)$는 위너 프로세스(Wiener process)이다.

### 4. 구조적 SSM (S4, Mamba)

시퀀스 모델링을 위해 SSM을 선형 상태 공간 레이어(LSSL)로 사용한다.

- **LSSL**: $\frac{dz(t)}{dt} = Az(t) + Bu(t), v(t) = Cz(t) + Du(t)$ 형태의 연속 시스템을 이산화하여 사용한다.
- **S4 (Structured State Spaces)**: 상태 행렬 $A$를 Diagonal-Plus-Low-Rank (DPLR) 구조로 설계하고, FFT(Fast Fourier Transform)를 이용해 합성곱(convolution) 형태로 계산하여 효율성을 극대화한다.
- **Mamba**: 기존 SSM의 시불변(time-invariant) 특성을 깨고, 입력 $u_t$에 따라 파라미터 $\bar{A}, \bar{B}, C$가 변하는 선택 메커니즘(Selection Mechanism)을 도입하여 문맥에 맞는 정보 선택이 가능하게 한다.

## 📊 Results

본 논문은 리뷰 논문으로서 개별 실험 결과보다는 각 모델군이 달성한 성과와 특성을 요약하여 제시한다.

- **시퀀스 모델링 효율성**: S4와 Mamba와 같은 구조적 SSM은 Transformer와 비교하여 메모리 및 계산 비용을 획기적으로 줄이면서도, 매우 긴 시퀀스(long-range dependencies) 처리 성능에서 우위를 보였다.
- **불규칙 시계열 처리**: Latent Neural ODE 모델은 고정된 시간 간격이 없는 의료 기록(EHR) 등의 데이터에서 전통적인 보간법(interpolation)보다 정확한 임퓨테이션(imputation) 및 예측 성능을 보였다.
- **혼합 빈도 데이터**: SSM 프레임워크를 사용한 Nowcasting(현재 시점 예측) 접근법은 고빈도 데이터(월간 지표)를 이용해 저빈도 데이터(분기 GDP)를 예측할 때, 데이터 빈도 불일치 문제를 일관성 있게 해결함을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 논문은 통계적 모델링으로서의 SSM과 신경망 아키텍처로서의 SSM 사이의 간극을 메웠다는 점에서 큰 의의가 있다. 특히, VAE 기반의 확률적 접근법이 어떻게 고전적 필터링의 한계를 극복했는지, 그리고 어떻게 이러한 구조가 Mamba와 같은 최신 LLM 대안 아키텍처로 진화했는지를 논리적으로 연결하였다.

### 한계 및 비판적 해석

1. **가우시안 가정이 여전함**: 많은 딥러닝 기반 SSM이 VAE를 사용함에도 불구하고, 계산의 편의를 위해 인코더와 디코더의 분포를 여전히 가우시안으로 가정하는 경우가 많다. 이는 진정한 의미의 비가우시안 특성 반영에는 한계가 있을 수 있다.
2. **모델 간의 분리**: 생성적 모델로서의 SSM(Section 3)과 모듈로서의 SSM(Section 4)은 수학적 형태는 유사하나, 실제 사용 목적과 학습 방식에서 큰 차이가 있다. 이 두 흐름이 어떻게 서로 영향을 주고받으며 통합될 수 있을지에 대한 논의가 더 필요하다.
3. **구현의 복잡성**: S4나 Mamba의 경우 이론적 효율성뿐만 아니라 GPU 커널 퓨전(Kernel Fusion)과 같은 하드웨어 최적화가 성능의 핵심인데, 이러한 구현 디테일이 논문의 이론적 분석만큼 중요하게 다뤄져야 한다.

## 📌 TL;DR

본 논문은 고전적인 상태 공간 모델(SSM)에서 최신 딥러닝 기반의 SSM 및 시퀀스 모델링 모듈(S4, Mamba 등)까지의 발전을 종합적으로 리뷰한다. 특히 VAE를 이용한 비선형 동역학 학습 방법과, 연산 효율성을 극대화한 구조적 SSM의 메커니즘을 상세히 분석한다. 이 연구는 긴 시퀀스 처리 효율화가 필요한 LLM 연구자와 불규칙한 시계열 데이터를 다루는 동역학 시스템 분석가들에게 중요한 기술적 지도를 제공한다.
