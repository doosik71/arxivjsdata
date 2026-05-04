# Memory Consistent Unsupervised Off-the-Shelf Model Adaptation for Source-Relaxed Medical Image Segmentation

Xiaofeng Liu, Fangxu Xing, Georges El Fakhri, Jonghye Woo (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 **Unsupervised Domain Adaptation (UDA)**을 수행할 때 발생하는 데이터 접근 제한 문제를 해결하고자 한다. 일반적인 UDA는 레이블이 있는 소스 도메인(Source Domain)과 레이블이 없는 타겟 도메인(Target Domain)의 데이터를 동시에 사용하여 학습하지만, 실제 의료 환경에서는 환자의 개인정보 보호 및 지식재산권(IP) 문제로 인해 소스 도메인의 원본 데이터에 접근하는 것이 매우 어렵다.

따라서 본 연구의 목표는 소스 도메인의 데이터 없이, 이미 학습된 **Off-the-Shelf (OS)** 모델(사전 학습된 모델)만을 이용하여 타겟 도메인에 적응시키는 **OSUDA (Off-the-Shelf UDA)** 프레임워크를 개발하는 것이다. 특히, 기존의 Source-relaxed UDA 방식이 가진 클래스 비율(Class-ratio) 일관성 가정의 한계(예: 질병의 발생률이나 종양 크기가 도메인마다 다를 수 있음)를 극복하고, 추가적인 보조 네트워크 없이도 안정적인 도메인 적응을 달성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Batch Normalization (BN) 통계량의 도메인 특성과 메모리 기반의 일관성 학습을 결합하여 소스 데이터 없이도 도메인 간 간극을 줄이는 것이다.

1. **BN 통계량의 단계적 적응**: BN의 저차 통계량(Low-order statistics; 평균과 분산)은 도메인 특화적(Domain-specific)이며, 고차 통계량(High-order statistics; Scaling $\gamma$ 및 Shifting $\beta$ 인자)은 도메인 간 공유 가능(Shareable)하다는 점에 착안하여 이를 분리하여 처리한다.
2. **전이 가능성(Transferability) 기반 가중치 적용**: 각 채널별로 소스-타겟 간의 통계적 거리와 스케일링 인자($\gamma$)를 분석하여, 전이 가능성이 높은 채널에 더 높은 중요도를 부여하는 적응형 가중치 메커니즘을 제안한다.
3. **Queued Memory-Consistent Self-Training (MCSF)**: 타겟 도메인에서 생성된 의사 레이블(Pseudo-label)의 불안정성을 해결하기 위해, 최근 $H$번의 반복 학습 동안의 예측값을 저장하는 큐(Queue)를 도입하고, 이들의 역사적 일관성(Historical Consistency)이 높은 픽셀에 대해서만 학습을 수행함으로써 학습의 안정성을 높인다.

## 📎 Related Works

기존의 UDA 접근 방식과 본 논문의 차별점은 다음과 같다.

- **Conventional UDA**: 소스 데이터와 타겟 데이터를 동시에 사용하여 Adversarial learning이나 Self-training을 수행한다. 하지만 이는 소스 데이터 접근이 필수적이라는 한계가 있다.
- **Source-free/Relaxed UDA**: 소스 데이터 없이 모델만 사용하는 방식이다. 예를 들어 **CRUDA**는 소스-타겟 간의 클래스 비율이 동일하다는 가정하에 보조 모델을 학습시키지만, 의료 영상에서는 종양의 크기나 분포가 다르기 때문에 이 가정이 성립하지 않는 경우가 많다.
- **BN-based Adaptation**: BN 통계량을 조정하여 도메인 간극을 줄이는 시도가 있었으나, 대부분 소스 데이터와의 공동 학습을 전제로 했다. 본 논문은 이를 완전한 Source-free 설정으로 확장하였다.
- **Self-training**: 의사 레이블을 활용하는 방식은 소스 데이터의 교정 없이 사용할 경우 편향된 예측으로 인해 학습이 불안정해질 위험이 있다. 본 논문은 이를 **MCSF**라는 메모리 메커니즘으로 보완하였다.

## 🛠️ Methodology

### 1. Batch Normalization (BN) 통계량의 정의

BN 레이어에서 입력 특징 $f_{l,b,m,c}$는 다음과 같이 정규화되고 선형 변환된다.
$$\hat{f}_{l,b,m,c} = \frac{f_{l,b,m,c} - \mu_{l,c}}{\sqrt{\sigma^2_{l,c} + \epsilon}} \quad (1)$$
$$\tilde{f}_{l,b,m,c} = \gamma_{l,c} \hat{f}_{l,b,m,c} + \beta_{l,c} \quad (2)$$
여기서 $\mu, \sigma^2$는 저차 통계량(평균, 분산)이며, $\gamma, \beta$는 학습 가능한 고차 통계량(Scaling, Shifting)이다.

### 2. 적응형 BN 통계량 적응 (Adaptive BN Adaptation)

**A. 저차 통계량의 단계적 적응 (EMD)**
타겟 도메인의 평균과 분산을 소스 도메인의 값($\mu_K, \sigma^2_K$)에서 시작하여 점진적으로 타겟의 값으로 업데이트한다. 이때 Exponential Momentum Decay (EMD) 전략을 사용한다.
$$\mu^t_{l,c} = (1 - \eta_t) \mu^t_{l,c} + \eta_t \mu^K_{l,c} \quad (5)$$
$$\{\sigma^2\}^t_{l,c} = (1 - \eta_t) \{\sigma^2\}^t_{l,c} + \eta_t \{\sigma^2\}^K_{l,c} \quad (6)$$
여기서 $\eta_t = \eta_0 \exp(-t)$이며, 학습이 진행됨에 따라 소스 통계량의 비중을 줄이고 타겟 통계량을 반영한다.

**B. 고차 통계량 일관성 손실 ($L_{\gamma HBS}$)**
소스 모델의 $\gamma_K, \beta_K$와 현재 타겟 모델의 $\gamma_t, \beta_t$ 사이의 일관성을 강제한다. 이때 채널별 전이 가능성을 고려한 가중치를 적용한다.
$$L_{\gamma HBS} = \sum_{l} \sum_{c} \exp(-\gamma_{K,l,c}) (1 + \alpha_{l,c}) (|\gamma_{K,l,c} - \gamma_{t,l,c}| + |\beta_{K,l,c} - \beta_{t,l,c}|) \quad (10)$$

- $\alpha_{l,c}$: 저차 통계량의 차이($d_{l,c}$)가 작을수록(전이 가능성이 높을수록) 큰 값을 갖는 가중치이다.
- $\exp(-\gamma_{K,l,c})$: 스케일링 인자가 작은 채널은 영향력이 낮다고 판단하여 가중치를 낮춘다.

### 3. 타겟 도메인 Self-Entropy (SE) 최소화

타겟 데이터에 대한 예측의 확신도를 높이기 위해 엔트로피를 최소화하는 손실 함수를 사용한다.
$$L_{SE} = -\frac{1}{B \times H_0 \times W_0} \sum_{b} \sum_{m} \{ p_{b,m} \log p_{b,m} \} \quad (11)$$

### 4. Queued Memory-Consistent Self-Training (MCSF)

의사 레이블 $\hat{y}$을 생성하고, 이를 최근 $H$번의 반복 학습 동안의 예측값들과 비교하여 역사적 일관성 $\psi_{\tau}$를 계산한다.
$$\psi_{\tau}^{b,m} = 1 - \text{Sigmoid} \left( \frac{1}{H} \sum_{h=1}^{H} \| p_{\tau}^{b,m} - p_{\tau-h}^{b,m} \|_1 \right) \quad (13)$$
이 일관성 값 $\psi_{\tau}$를 가중치로 사용하여, 예측이 일관된 픽셀에 대해서만 학습을 수행하는 $L_{MCST}$를 정의한다.
$$L_{MCST} = -\frac{1}{B \times H_0 \times W_0} \sum_{b} \sum_{m} \psi_{\tau}^{b,m} \times \hat{y}_{\tau}^{b,m} \log p_{\tau}^{b,m} \quad (14)$$

### 5. 전체 학습 절차

최종 목적 함수는 다음과 같다.
$$\min_{w, \hat{y}} L = L_{\gamma HBS} + \lambda L_{SE} + \phi L_{MCST}$$
학습은 **Expectation $\to$ Classification $\to$ Maximization**의 3단계 루프로 진행되며, $\lambda$는 학습 과정에서 $10 \to 0$으로 선형적으로 감소시켜 초기 학습의 불안정성을 방지한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 작업**:
  - **Brain Tumor (BraTS2018)**: Cross-modality (T2 $\to$ T1, FLAIR, T1ce) 및 Cross-subtype (HGG $\to$ LGG) 분할.
  - **Cardiac (MM-WHS)**: MR $\to$ CT 분할.
- **지표**: Dice Similarity Coefficient (DSC) $\uparrow$, Hausdorff Distance (HD) $\downarrow$.
- **비교 대상**: CRUDA, SFKT, DPL 등 Source-free/relaxed 방법 및 CLS, DSFN 등 Source-available (Upper Bound) 방법.

### 2. 정량적 결과

- **Cross-modality Brain Tumor**: MCOSUDA는 기존의 Source-free 방법들(CRUDA, SFKT 등)보다 우수한 성능을 보였으며, 일부 타겟 도메인(T1, FLAIR)에서는 소스 데이터가 필요한 CycleGAN이나 SIFA보다 더 나은 성능을 기록했다. (평균 DSC 59.17%)
- **Cross-subtype Brain Tumor**: HGG $\to$ LGG 작업에서 CRUDA는 클래스 비율 차이로 인해 성능이 낮았으나, MCOSUDA는 소스 데이터 기반의 SEAT 모델에 근접하는 성능을 보였다. (평균 DSC 62.87%)
- **Cardiac MR $\to$ CT**: 매우 큰 도메인 간극이 존재하는 작업임에도 불구하고, MCOSUDA는 기존 Source-free 방식 대비 DSC를 약 4% 이상 향상시켰다. (평균 DSC 64.57%)

### 3. 절제 연구 (Ablation Study)

- $\alpha$ 기반의 채널 가중치(OSUDA-AC)와 SE 최소화(OSUDA-SE)를 제거했을 때 성능이 하락함을 확인하여, 각 구성 요소의 유효성을 입증하였다.
- MCSF를 적용했을 때 학습 곡선이 훨씬 안정적이며 최종 성능이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 의료 영상 데이터의 **프라이버시 문제**를 정면으로 다루며, 소스 데이터 없이 모델만으로 도메인 적응을 수행하는 실용적인 프레임워크를 제시했다. 특히 BN 통계량의 특성을 정밀하게 분석하여 저차/고차 통계량을 분리 적응시키고, 채널별 전이 가능성을 정량화하여 반영한 점이 돋보인다. 또한, 메모리 큐를 이용한 역사적 일관성 검증(MCSF)은 소스 데이터라는 '정답'이 없는 상황에서 의사 레이블의 신뢰도를 높이는 매우 효과적인 장치로 작용했다.

### 한계 및 논의사항

1. **백본 네트워크의 동일성**: 본 연구는 소스와 타겟 모델이 동일한 아키텍처를 사용한다고 가정한다. 다른 백본을 사용할 경우 Knowledge Distillation 등의 추가 기법이 필요할 것이다.
2. **하이퍼파라미터 민감도**: $\lambda, \phi, \alpha$ 등 조정해야 할 파라미터가 많으며, 최적의 성능을 내기 위해서는 세심한 튜닝이 필요하다는 점이 언급되었다.
3. **3D 데이터 활용**: BN 통계량의 안정적인 계산을 위해 2D 슬라이스 기반으로 실험을 진행했으나, 실제 의료 영상은 3D이므로 향후 3D 백본으로의 확장이 필요하다.

## 📌 TL;DR

이 논문은 소스 데이터에 접근할 수 없는 제한적인 환경에서 사전 학습된 모델(Off-the-Shelf)만을 이용하여 의료 영상을 분할하는 **MCOSUDA** 프레임워크를 제안한다. BN 통계량의 단계적 적응과 채널별 전이 가능성 가중치, 그리고 메모리 큐를 이용한 의사 레이블 일관성 학습(MCSF)을 통해 소스 데이터 없이도 기존 UDA 방식에 근접하거나 때로는 능가하는 성능을 달성하였다. 이는 환자 데이터 보안이 중요한 실제 의료 현장에서 모델을 배포하고 적응시키는 데 매우 중요한 역할을 할 수 있는 연구이다.
