# BSSAD: Towards A Novel Bayesian State-Space Approach for Anomaly Detection in Multivariate Time Series

Usman Anjum, Samuel Lin, Jusin Zhan (2023)

## 🧩 Problem to Solve

본 논문은 다변량 시계열(Multivariate Time Series, MTS) 데이터에서의 이상치 탐지(Anomaly Detection) 문제를 해결하고자 한다. 다변량 시계열 데이터는 여러 개의 변수가 시간적 의존성뿐만 아니라 변수 간의 상호 의존성을 동시에 가지고 있어 분석이 매우 복잡하다.

기존의 접근 방식은 크게 두 가지로 나뉜다. 첫째, 잔차 기반(Residual-based) 방식은 주로 변수 간의 관계를 무시하고 개별 변수의 미래 값을 예측하여 실제 값과의 차이(잔차)를 통해 이상치를 탐지한다. 둘째, 밀도 기반(Density-based) 방식은 데이터의 분포를 학습하여 이 분포에서 벗어난 데이터를 이상치로 간주한다. 밀도 기반 방식은 더 높은 정확도를 보이지만, 시스템의 상태를 정확히 추정하기 위해서는 변수 간의 수학적 관계(물리 방정식 등)에 대한 사전 지식이 필요하며, 사후 분포(Posterior distribution)에 대한 정보가 부족하다는 치명적인 한계가 있다.

따라서 본 논문의 목표는 시스템의 수학적 모델을 미리 알지 못하더라도, 신경망을 통해 변수 간의 관계를 학습하고 베이지안 상태 공간(Bayesian State-Space) 모델을 통해 데이터의 분포를 정밀하게 추정함으로써 고성능의 이상치 탐지를 수행하는 BSSAD(Bayesian State-Space Anomaly Detection) 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 신경망의 강력한 표현 학습 능력과 베이지안 상태 공간 알고리즘의 상태 추정 능력을 결합하는 모듈형 설계를 도입하는 것이다.

1. **NNM과 BSSM의 결합**: Recurrent Neural Networks(RNN)와 Autoencoders(AE)를 통해 복잡한 다변량 데이터의 관계를 학습하여 상태 공간 모델의 파라미터를 추정하고, 이를 Ensemble Kalman Filter(EnKF) 및 Particle Filter(PF)와 결합하여 데이터의 사후 분포를 생성한다.
2. **고급 베이지안 필터의 도입**: 기존 연구(NSIBF)에서 사용된 Unscented Kalman Filter(UKF)보다 비선형성 및 비가우시안 노이즈 처리에 더 효과적인 EnKF와 PF를 다변량 시계열 이상치 탐지에 최초로 적용하였다.
3. **평가 지표의 확장**: 기존의 F1-score 외에도 이진 분류 작업에서 더 포괄적인 정확도를 제공하는 Matthews Correlation Coefficient(MCC)를 도입하여 모델의 성능을 다각도로 분석하였다.
4. **유연한 모듈 구조**: 데이터의 특성이나 요구되는 성능(속도 vs 정확도)에 따라 신경망 알고리즘이나 베이지안 필터를 쉽게 교체할 수 있는 유연한 구조를 설계하였다.

## 📎 Related Works

논문에서는 이상치 탐지 알고리즘을 다음과 같이 분류하고 기존 연구의 한계를 지적한다.

- **잔차 기반 및 재구성 기반 방법**: Isolation Forest, LSTM-AE, USAD 등이 이에 해당한다. 이들은 예측값과 실제값의 차이를 이용하지만, 다변량 데이터의 변수 간 상관관계를 충분히 반영하지 못하는 경우가 많다.
- **밀도 및 통계 기반 방법**: 데이터가 특정 분포를 따른다고 가정하며, 이 분포에서 크게 벗어난 데이터를 이상치로 판단한다.
- **베이지안 상태 공간 방법**: Kalman Filter 및 그 변형들이 상태 추정에 널리 사용되어 왔으나, 주로 물리 방정식이 존재하는 도메인에 국한되어 적용되었다.
- **NSIBF (Neural System Identification and Bayesian Filtering)**: AE와 LSTM을 사용하여 상태 공간 모델을 구축하고 UKF를 통해 분포를 추정하는 방식이다. 본 논문은 NSIBF의 구조를 계승하되, 필터 부분을 EnKF와 PF로 업그레이드하여 성능을 향상시켰다.

## 🛠️ Methodology

BSSAD는 크게 **Neural Network Module (NNM)**과 **Bayesian State-Space Module (BSSM)**의 두 단계로 구성된다.

### 1. Neural Network Module (NNM)

NNM은 상태 공간 모델의 핵심인 상태 전이 함수 $F$와 측정 함수 $H$를 학습하는 역할을 한다.

- **구조**:
  - **상태 전이 함수 $F$**: LSTM을 기반으로 하며, 이전 상태 $z_{t-1}$과 윈도우 크기 $\tau$ 동안의 관측값 $x_{t-\tau:t-1}$을 입력받아 다음 상태 $z_t$를 예측한다.
  - **측정 함수 $H$**: Autoencoder의 Encoder 부분을 사용하여 고차원 관측값 $x_t$를 저차원 잠재 공간(Latent space)의 상태 $z_t$로 매핑한다. Decoder $H^{-1}$는 이를 다시 원래 차원으로 복원한다.
- **학습 목표 및 손실 함수**:
    정상 데이터 $X^{(n)}$만을 사용하여 학습하며, 손실 함수 $L$은 다음과 같이 정의된다.
    $$L = \sum_{t=\tau}^{T} \left( \alpha_1 \|x_{t-1} - \hat{x}_{t-1}\|^2 + \alpha_2 \|x_t - \hat{x}_t\|^2 + \alpha_3 \|z_t - z_{t-1}\|^2 \right)$$
    여기서 첫 두 항은 재구성 및 예측 오차를 줄이기 위함이며, 마지막 항은 연속적인 hidden state $z$가 급격하게 변하지 않도록 평활화(Smoothing)하는 역할을 한다.
- **노이즈 추정**: 학습된 NNM을 검증 데이터셋에 적용하여 프로세스 노이즈 $q_t$와 측정 노이즈 $r_t$를 계산하고, 이를 통해 공분산 행렬 $Q$와 $R$을 도출한다.

### 2. Bayesian State-Space Module (BSSM)

BSSM은 학습된 $F, H, Q, R$을 이용하여 현재 데이터의 사후 분포를 추정하고 이상치를 판별한다. 상태 공간 방정식은 다음과 같다.

- 상태 전이: $z_t = F(z_{t-1}, x_{t-\tau:t-1}) + q_t$
- 측정: $x_t = H^{-1}(z_t) + r_t$

본 논문은 두 가지 필터를 구현하였다.

#### A. Neural Network Ensemble Kalman Filter (NN-EnKF)

EnKF는 몬테카를로 시뮬레이션을 통해 다수의 시그마 포인트($\chi$)를 생성하여 분포를 근사한다.

- **예측 단계**: 시그마 포인트를 $F$에 통과시켜 예측 상태 $\chi_f$를 얻는다.
- **업데이트 단계**: 실제 관측값 $x_t$와 예측값 사이의 차이를 이용하여 칼만 이득(Kalman Gain, $K$)을 계산하고 시그마 포인트를 업데이트한다. 최종적으로 업데이트된 포인트들의 평균 $\hat{\mu}_t$와 공분산 $\hat{P}_t$를 통해 사후 분포를 추정한다.

#### B. Neural Network SIR Particle Filter (NN-SIR)

PF는 가중치가 부여된 입자(Particle)들의 집합으로 분포를 표현한다.

- **절차**: 입자들을 샘플링하고 상태 전이 및 측정 함수를 통해 전파시킨다. 각 입자는 실제 관측값 $x_t$와의 거리(RBF 커널 사용)에 따라 가중치가 업데이트된다.
- **Degeneracy 문제 해결**: 일부 입자만 높은 가중치를 갖는 현상을 막기 위해, 유효 입자 수 $N_{eff}$가 임계값 $N_T$보다 낮아지면 체계적 재샘플링(Systematic Resampling)을 수행한다.

### 3. Anomaly Detection (이상치 판별)

추정된 평균 $\hat{\mu}_t$와 공분산 $\hat{P}_t$를 바탕으로 실제 관측값 $x_t$와의 **마할라노비스 거리(Mahalanobis distance)**를 계산하여 이상치 점수 $m$을 산출한다.
$$m = \sqrt{(x_t - \hat{\mu}_t)^T \hat{P}_t^{-1} (x_t - \hat{\mu}_t)}$$
이 거리 $m$이 설정된 임계값을 초과하면 해당 시점의 데이터는 이상치로 분류된다.

## 📊 Results

### 실험 설정

- **데이터셋**: SWaT, WaDi, PUMP, HTTP, ASD 등 5종의 다변량 시계열 데이터셋을 사용하였다.
- **비교 모델**: Isolation Forest, LSTM-AE, NSIBF(UKF 기반), Interfusion, Anomaly Transformer, BeatGAN, FGANomaly.
- **평가 지표**: F1-score 및 MCC. (Point-adjusted metric 적용)

### 주요 결과

- **성능 우위**: BSSAD는 대부분의 데이터셋에서 baseline 모델들보다 높은 F1-score와 MCC를 기록하였다. 특히 BSSAD-PF-20000 모델은 SWaT, WaDi, HTTP, ASD 데이터셋에서 최상위 성능을 보였다.
- **필터별 특성**:
  - **PF (Particle Filter)**: 계산 비용은 높지만, 대부분의 데이터셋에서 가장 높은 정확도를 보였다.
  - **EnKF (Ensemble Kalman Filter)**: PUMP 데이터셋에서 가장 좋은 성능을 보였으며, PF보다 계산 속도가 빨라 실시간 탐지에 유리하다.
- **MCC의 중요성**: 일부 모델의 경우 F1-score는 높았으나 MCC는 매우 낮게 나타나는 경우가 있었으며, BSSAD는 두 지표 모두에서 높은 수치를 기록하여 탐지 성능의 신뢰성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 다변량 시계열 데이터에서 물리적 모델 없이도 베이지안 필터를 적용할 수 있는 체계적인 프레임워크를 제시하였다.

- **강점**: 신경망을 통해 상태 공간 모델의 파라미터를 학습함으로써, 데이터 기반의 유연한 분포 추정이 가능하다는 점이다. 특히 EnKF와 PF의 도입을 통해 UKF보다 복잡한 비선형/비가우시안 분포를 더 잘 처리할 수 있게 되었다.
- **트레이드-오프**: 정확도와 계산 비용 사이의 명확한 관계가 관찰되었다. 정밀한 진단이 필요한 의료 분야 등에서는 PF가 적합하겠지만, 빠른 응답이 필요한 사이버 공격 탐지 분야에서는 EnKF나 UKF가 더 효율적일 것이다.
- **한계 및 향후 과제**: 현재는 기본 LSTM과 AE를 사용하고 있으나, 향후 Hierarchical VAE나 Variational Autoencoders 등을 NNM에 도입한다면 더 복잡한 상관관계를 학습할 수 있을 것으로 보인다. 또한, 입자 필터의 중요도 샘플링(Importance density) 전략을 최적화하는 연구가 필요하다.

## 📌 TL;DR

본 논문은 신경망(LSTM, AE)으로 다변량 시계열의 시스템 관계를 학습하고, 이를 기반으로 베이지안 상태 공간 모델(EnKF, PF)을 구축하여 이상치를 탐지하는 **BSSAD**를 제안한다. 제안 방법은 실제 관측값과 추정된 사후 분포 사이의 마할라노비스 거리를 측정하여 이상치를 판별하며, 실험 결과 기존의 잔차 기반 모델 및 UKF 기반 모델보다 우수한 탐지 성능(F1-score > 0.95)을 보였다. 이 연구는 도메인 지식이 부족한 환경에서도 고정밀 다변량 이상치 탐지를 가능하게 한다는 점에서 향후 다양한 산업 현장 및 의료 분야에 적용될 가능성이 높다.
