# Anomaly Detection of Time Series with Smoothness-Inducing Sequential Variational Auto-Encoder

Longyuan Li, Junchi Yan,Member, IEEE,, Haiyang Wang, and Yaohui JinMember, IEEE,

## 🧩 Problem to Solve

시계열 이상 탐지는 중요한 연구 과제이며, 특히 다차원 상관 시계열 데이터에서는 더욱 복잡합니다. 기존 심층 생성 모델(VAE 포함)은 시간적 종속성을 효과적으로 포착하지 못하거나, 훈련 데이터에 이상치가 포함될 경우 '정상' 패턴 대신 이상치에 과적합되어 밀도 추정이 왜곡될 수 있습니다. 또한, 많은 모델은 잡음이 시간에 따라 일정하다는(constant noise) 가정을 하지만, 실제 시계열 데이터의 분산은 비정상적(non-stationary)으로 변하는 경우가 많습니다. 본 논문은 이러한 비정상 잡음과 이상치 오염에 강건한 다차원 시계열 포인트 이상 탐지 모델을 개발하는 것을 목표로 합니다.

## ✨ Key Contributions

- **베이지안 평활도 사전 지식 도입:** 다차원 시계열 이상 탐지를 위한 심층 생성 모델 학습에 평활도 사전 지식(smoothness prior)을 베이지안 방식으로 통합하는 새로운 기술을 제안합니다. 이는 생성 모델의 비평활적인 재구성에 페널티를 부여하여(평균 및 분산 모두), 모델이 이상치에 강건하게 '정상' 패턴을 학습하도록 돕습니다.
- **비정상 잡음 모델링:** 각 타임스탬프의 평균과 분산을 개별적으로 신경망을 통해 파라미터화하여, 기존 마르코프 모델의 고정 잡음 가정 없이 동적인 이상 탐지 임계값 적응이 가능하도록 합니다.
- **효과적인 학습 및 이상 탐지 기준:** 새로운 확률적 경사 변분 베이즈(SGVB) 추정기를 통해 모델을 효율적으로 학습하며, 재구성 확률(reconstruction probability)과 재구성 오차(reconstruction error)라는 두 가지 이상 탐지 기준을 체계적으로 연구하여 재구성 확률이 비정상 시계열에 더 강건함을 보여줍니다.
- **실증적 성능 검증:** 합성 데이터셋 및 실제 벤치마크 데이터셋에 대한 광범위한 실험을 통해 제안된 모델이 최신 경쟁 모델보다 일관되게 우수한 성능을 보임을 입증합니다.

## 📎 Related Works

- **결정론적 모델(Deterministic models):** 시계열 디트렌딩(de-trending)과 같이 시간적 평활도를 위해 추가적인 정규화 항을 사용하지만, 시계열의 확률적 특성을 간과합니다.
- **확률론적 모델(Probabilistic models):** HMM(Hidden Markov Models), 행렬 분해(Matrix Factorizations) 등 데이터 생성 과정의 주변 우도 $p_{\theta}(x)$를 학습하며, 주로 재구성 확률을 임계값으로 사용하여 이상치를 탐지합니다. 하지만 복잡한 모델에 적용하기 어렵고 잡음이 일정하다는 가정을 하는 경우가 많습니다.
- **심층 생성 모델 (VAE 기반):**
  - **Vanilla VAE:** 데이터 포인트 간 i.i.d. 가정을 하여 시계열에 부적합합니다.
  - **Donut [3]:** 슬라이딩 윈도우를 사용하여 바닐라 VAE를 확장하지만, 시간적 구조를 완전히 모델링하지 못하며 이상치 라벨을 활용한 수정된 ELBO를 사용합니다.
  - **VRNN [16], STORN [30]:** RNN을 백본으로 사용하여 시간적 구조를 포착하는 시퀀스 버전 VAE이지만, 이상치의 존재를 목적 함수에서 명시적으로 고려하지 않습니다.
- **전통적인 통계 모델:** History Average (HA), Auto-regressive Moving Average (ARMA), Linear Dynamical System (LDS/Kalman filter) 등이 비교 대상으로 사용됩니다.

## 🛠️ Methodology

본 논문은 평활도 유도 시퀀스 변분 오토인코더(Smoothness-Inducing Sequential Variational Auto-Encoder, SISVAE) 모델을 제안합니다.

1. **시퀀스 VAE (Sequential VAE) 아키텍처:**
   - **백본:** GRU(Gated Recurrent Unit) 기반의 RNN을 사용하여 시계열의 잠재적인 시간 구조를 포착합니다.
   - **생성 모델($p_{\theta}(x_{1:T}, z_{1:T})$):** $p_{\theta}(x_t | z_{\le t}, x_{\lt t})p_{\theta}(z_t | x_{\lt t}, z_{\lt t})$로 분해됩니다.
     - 재구성 분포는 $\hat{x}_t \sim \mathcal{N}(\mu_{x,t}, \text{diag}(\sigma^2_{x,t}))$이며, 평균 $\mu_{x,t}$와 분산 $\sigma_{x,t}$는 신경망 $\varphi^{\text{dec}}_{\theta}$에 의해 이전 잠재 상태 $h_{t-1}$와 현재 잠재 변수 $z_t$에 따라 파라미터화됩니다.
     - 잠재 상태의 사전 분포는 $z_t \sim \mathcal{N}(\mu_{0,t}, \text{diag}(\sigma^2_{0,t}))$이며, $\mu_{0,t}$와 $\sigma_{0,t}$는 신경망 $\varphi^{\text{prior}}_{\theta}$에 의해 이전 $h_{t-1}$에 따라 파라미터화됩니다.
     - RNN 은닉 상태 $h_t$는 GRU 방정식을 통해 $h_t = f_{\theta}(\varphi^{\text{z}}_{\theta}(z_t), h_{t-1})$로 업데이트됩니다.
   - **추론 모델($q_{\varphi}(z_{1:T}|x_{1:T})$):** $z_t$의 근사 사후 분포는 $q_{\varphi}(z_t | x_{\le t}, z_{\lt t})$로 분해되며, $\mathcal{N}(\mu_{z,t}, \text{diag}(\sigma^2_{z,t}))$로 모델링되고, $\mu_{z,t}$와 $\sigma_{z,t}$는 신경망 $\varphi^{\text{enc}}_{\varphi}$에 의해 $x_t$와 $h_{t-1}$에 따라 파라미터화됩니다.
   - **비정상 잡음:** 각 타임스탬프의 평균과 분산을 신경망으로 개별 파라미터화하여 동적으로 변화하는 데이터 잡음을 모델링할 수 있습니다.
2. **평활도 사전 지식 모델링:**
   - 재구성된 시계열 $\hat{x}_{1:T}$의 확률 밀도 함수가 시간에 따라 부드럽게 변한다는 가정을 도입합니다.
   - 두 연속된 재구성 분포 $p_{\theta}(\hat{x}_{m,t-1})$와 $p_{\theta}(\hat{x}_{m,t})$ 사이의 거리 측정으로 KL-발산(KL-divergence)을 사용하여 누적 전환 비용(accumulative transition cost)을 $L_{\text{smooth}} = \sum_{t=1}^T \sum_{m=1}^M D_{KL}(p_{\theta}(\hat{x}_{m,t-1}) \| p_{\theta}(\hat{x}_{m,t}))$로 정의합니다.
   - 이 평활도 정규화는 모델이 이상치에 과적합되는 것을 방지하고 '정상' 패턴에 집중하도록 유도합니다.
3. **목적 함수 및 학습:**
   - SISVAE의 학습 목적 함수는 ELBO(Evidence Lower Bound)에 평활도 손실을 추가하여 구성됩니다:
     $$ \mathcal{L}_{\text{SISVAE}} = \sum_{t=1}^T \left\{ -D*{KL}(q*{\varphi}(z*t | x*{\le t}, z*{\lt t}) \| p*{\theta}(z*t | x*{\lt t}, z*{\lt t})) + \log p*{\theta}(x*t | z*{\le t}, x*{\lt t}) + \lambda D*{KL}(p*{\theta}(\hat{x}*{t-1} | \dots) \| p\_{\theta}(\hat{x}\_t | \dots)) \right\} $$
   - 확률적 경사 변분 베이즈(SGVB) 추정기와 역전파(backpropagation)를 사용하여 파라미터 $\theta$와 $\varphi$를 최적화합니다.
   - 긴 시계열은 슬라이딩 윈도우 기법으로 짧은 청크로 분할하여 학습 안정성을 높입니다.
4. **이상치 점수 계산:**
   - **재구성 오차(Reconstruction Error):** $e_{m,t} = \|x_{m,t} - \mu_{m,t}\|$. 비정상 분산 데이터에 취약합니다.
   - **재구성 확률(Reconstruction Probability):** 시퀀스 몬테 카를로(Sequential Monte Carlo, SMC)를 사용하여 $\log p_{\theta}(x) \approx \frac{1}{L} \sum_{l=1}^L \log p_{\theta}(x_t | \mu^{(l)}_x, \sigma^{(l)}_x)$를 추정합니다. 이는 시퀀스 종속성을 고려하며 비정상 잡음에 강건하게 이상치 점수를 계산합니다(SISVAE-p).

## 📊 Results

- **합성 데이터셋:**
  - 이상치 비율(0.5% ~ 10%) 변화에 대한 강건성을 평가했을 때, SISVAE-p는 비정규화된 모델(SISVAE-0, STORN)에 비해 성능 저하가 현저히 적었으며, 모든 이상치 비율 설정에서 우수한 성능을 보였습니다.
  - 평활도 정규화 하이퍼파라미터 $\lambda$는 0.5일 때 최적의 AUPRC 점수를 달성했습니다.
  - KL-정규화된 모델(SISVAE-p)은 비정규화된 모델과 결정론적 평균 정규화 모델에 비해 학습 중 더 안정적인 수렴과 높은 AUPRC를 보여, 제안된 확률적 평활도 정규화의 효과를 입증했습니다.
- **실제 데이터셋 (Yahoo S5 Webscope, Open$\mu$PMU):**
  - SISVAE-p는 AUROC, AUPRC, F1-score 등 대부분의 지표에서 Donut, STORN, HA, ARMA, LDS 등 다른 모델들을 지속적으로 능가했습니다.
  - 특히, 재구성 확률 기반 이상치 점수(SISVAE-p)가 재구성 오차 기반(SISVAE-e)보다 성능이 월등히 좋았으며, 이는 비정상 시계열에서 동적 분산 모델링의 중요성을 시사합니다.
  - 전통적인 통계 모델들은 비정상 데이터셋에서 제대로 작동하지 못하는 경우가 많았습니다.
  - Precision@K 평가에서 SISVAE-p는 K가 증가함에 따라 안정적인 높은 정밀도를 유지하여, 제한된 예산 내에서 효과적인 이상치 탐지가 가능함을 보여주었습니다.
- **사례 연구 (Yahoo-A1):**
  - **오탐(False Alarms):** 정상 시퀀스에서 SISVAE-p는 안정적으로 낮은 이상치 점수를 생성하여 STORN의 오탐을 방지했습니다.
  - **포인트 레벨 이상치:** SISVAE-p는 포인트 이상치를 성공적으로 탐지하면서도, STORN과 Donut에서 발생한 오탐을 피하고 기본 밀도를 잘 복구했습니다.
  - **하위 시퀀스 레벨 이상치:** SISVAE-p는 이상치 하위 시퀀스를 효과적으로 탐지하여, 다른 모델들이 놓치거나 오탐하는 문제를 해결했습니다.
- **Donut과의 추가 비교 (Mackey-Glass 시계열):** 단변량 시계열에서도 SISVAE-p는 Donut보다 우수한 이상 탐지 성능을 보였습니다.

## 🧠 Insights & Discussion

- 제안된 평활도 유도 사전 지식은 생성 모델이 훈련 데이터 내의 이상치에 과적합되지 않고 '정상' 시간 패턴을 학습하도록 효과적으로 유도하여 모델의 강건성을 크게 향상시킵니다.
- 신경망을 통해 시간에 따라 변하는 평균과 분산(비정상 잡음)을 모델링하는 능력은 실제 시계열 데이터에서 동적인 이상 탐지 임계값을 설정하는 데 결정적이며, SISVAE-p의 성능 우수성을 뒷받침합니다.
- 확률론적 프레임워크 내에서 KL-발산을 사용한 평활도 정규화는 결정론적인 평균 정규화보다 수학적으로 일관성이 있으며, 더 안정적인 학습과 뛰어난 이상 탐지 성능을 가져옵니다.
- 이상치 점수 계산 시 재구성 확률(SMC 기반)을 사용하는 것이 재구성 오차보다 비정상 데이터에 더 강건하고 효과적임을 입증했습니다.
- 향후 연구에서는 생성 모델의 출력 분포를 Student-t 분포와 같은 중꼬리 분포(heavy-tailed distribution)로 확장하여 극단적인 이상치에 대한 강건성을 더욱 높일 수 있습니다.

## 📌 TL;DR

**문제:** 이상치로 오염된 다차원 시계열 데이터에서 비정상 잡음을 고려한 강건한 비지도 이상 탐지가 어려움.
**방법:** 시퀀스 변분 오토인코더(VAE)에 RNN 백본을 활용하여 시계열 특성을 포착하고, 각 타임스탬프의 평균과 분산을 신경망으로 개별 파라미터화하여 비정상 잡음을 모델링. 여기에 KL-발산 기반의 새로운 평활도 사전 정규화(smoothness prior regularization)를 도입하여 이상치에 대한 과적합을 방지하고 '정상' 패턴 학습을 유도. 이상치 점수는 SMC 기반 재구성 확률로 계산 (SISVAE-p).
**결과:** SISVAE-p는 합성 및 실제 데이터셋 모두에서 기존 최신 모델들을 능가하는 이상 탐지 성능을 보였으며, 특히 비정상 시계열 데이터와 이상치 오염에 대한 강건함을 입증.
