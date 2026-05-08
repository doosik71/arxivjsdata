# Particle Filter Re-detection for Visual Tracking via Correlation Filters

Di Yuan, Xiaohuan Lu, Donghao Li, Yingyi Liang, Xinming Zhang (2018)

## 🧩 Problem to Solve

본 논문은 Correlation Filter(CF) 기반의 시각적 추적(Visual Tracking) 알고리즘이 가진 치명적인 결함을 해결하고자 한다. CF 기반 추적기들은 일반적으로 매우 빠른 계산 속도와 우수한 성능을 보이지만, 복잡한 추적 환경에서 타겟 객체의 위치를 부정확하게 지정하는 문제가 발생한다.

특히, 이러한 알고리즘들은 Response map의 최대 응답 값(Maximum response value)에 과도하게 의존하는 경향이 있다. 만약 배경 노이즈가 심하거나, 가려짐(Occlusion), 혹은 급격한 움직임으로 인해 Response map이 모호해지면 최대 응답 값이 작아지며, 이는 결국 추적 대상의 소실(Loss)이나 드리프트(Drift) 현상으로 이어진다. 따라서 본 연구의 목표는 CF의 빠른 속도를 유지하면서도, 추적 결과가 신뢰할 수 없을 때 객체를 효과적으로 다시 찾아낼 수 있는 재탐지(Re-detection) 메커니즘을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Particle Filter 기반의 재탐지 메커니즘 제안**: CF 추적 결과가 모호하거나 신뢰할 수 없는 상황(최대 응답 값이 특정 임계값보다 낮을 때)에서 Particle Filter(PF)의 리샘플링 전략을 사용하여 더 많은 타겟 후보군을 생성하고, 그중 최적의 위치를 재탐지함으로써 추적의 강건성(Robustness)을 높였다.
2. **새로운 스케일 평가(Scale Evaluation) 전략**: 연속된 프레임 간의 최대 응답 값의 변화율을 분석하여 객체의 크기 변화를 추정하는 간단하고 효율적인 메커니즘을 제안하였다. 이를 통해 타겟의 크기 변화가 성능에 미치는 영향을 줄이고 알고리즘의 안정성을 향상시켰다.

## 📎 Related Works

### Particle Filter Based Trackers

Particle Filter 기반 추적기는 비선형 및 비가우시안 가정하에서 상태 추정을 수행하며, 다중 상태 가설을 동시에 고려할 수 있어 강건한 추적이 가능하다. 그러나 파티클의 수를 결정하는 문제와 타겟 템플릿 선택의 정확도 문제가 있으며, 무엇보다 계산 복잡도가 매우 높아 실시간 추적에 제약이 크다는 단점이 있다.

### Correlation Filter Based Trackers

KCF(Kernelized Correlation Filter)와 같은 CF 기반 추적기는 공간 도메인의 컨볼루션 연산을 푸리에 도메인(Fourier domain)의 요소별 곱셈으로 변환하여 계산 속도를 획기적으로 높였다. 하지만 앞서 언급했듯이 최대 응답 값에 지나치게 의존하므로, 응답 맵이 불안정해지는 상황에서 객체를 놓치기 쉽다는 한계가 있다.

본 논문은 CF의 효율적인 계산 능력과 PF의 강건한 후보군 생성 능력을 결합하여, 평상시에는 CF로 빠르게 추적하고 위기 상황에서만 PF로 재탐지하는 하이브리드 방식을 채택함으로써 기존 방식들의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 시스템은 **CF-tracking 파트**와 **Re-detection 파트**의 두 가지 경로로 구성된다.

- **CF-tracking**: 일반적인 상황에서 사용되며, 현재 프레임의 최대 응답 값 $maxR$이 임계값 $\theta$보다 크거나 같을 때($maxR \ge \theta$) 작동한다.
- **Re-detection**: $maxR < \theta$인 경우 작동하며, PF를 통해 여러 후보 영역을 생성하고 최적의 위치를 다시 찾는다.

### 1. KCF 기반 추적 프레임워크

기본적으로 HOG 특징을 사용하며, 가우시안 커널을 통해 비선형 회귀 문제를 선형 문제로 변환한다. 필터 $w$는 다음의 비용 함수를 최소화하도록 학습된다.

$$w = \arg \min_{w} \sum_{m,n} |\langle \phi(x_{m,n}), w \rangle - y(m,n)|^2 + \lambda \|w\|^2$$

여기서 $\phi$는 커널 공간으로의 매핑, $\lambda$는 정규화 파라미터이다. 푸리에 변환을 이용한 최적해 $\alpha$는 다음과 같다.

$$\alpha = \mathcal{F}^{-1} \left( \frac{\mathcal{F}(y)}{\mathcal{F}(k_x) + \lambda} \right)$$

추적 과정에서 새로운 패치 $z$에 대한 응답 값 $f(z)$는 다음과 같이 계산된다.

$$f(z) = \mathcal{F}^{-1}(\mathcal{F}(k_z) \odot \mathcal{F}(\alpha))$$

### 2. Particle Filter 재탐지 모델

CF 결과가 불확실할 때, 이전 타겟 위치를 중심으로 정규분포를 따르는 $M$개의 파티클(후보 패치)을 생성한다. 각 파티클 $z_m$에 대해 HOG 특징을 추출하고 기존 CF 필터와 연산하여 개별 응답 맵 $R_m$을 구한다.

$$R_m = \mathcal{F}^{-1}(\mathcal{F}(\langle z_m, \hat{x} \rangle) \odot \mathcal{F}(\alpha))$$

최종적으로 $M$개의 파티클 중 최대 응답 값이 가장 큰 파티클을 새로운 타겟 위치로 결정한다.

$$maxR_{pf} = \max \{maxR_1, maxR_2, \dots, maxR_M\}$$

### 3. 스케일 평가 알고리즘

타겟의 크기 변화를 감지하기 위해 연속된 세 프레임의 최대 응답 값 변화율 $d_t$를 계산한다.

$$d_t = \frac{maxR_t}{maxR_{t-1}} - \frac{maxR_{t-1}}{maxR_{t-2}}$$

- $d_t > \phi$ 이면 타겟 크기가 작아진 것으로 판단 $\rightarrow$ 스케일 팩터 $s_t = 0.98$
- $d_t < \psi$ 이면 타겟 크기가 커진 것으로 판단 $\rightarrow$ 스케일 팩터 $s_t = 1.02$
- 그 외에는 크기 변화 없음 $\rightarrow$ $s_t = 1$

최종 크기는 $size_t = size_{t-1} * s_t$로 업데이트된다.

### 4. 모델 업데이트

현재 프레임의 타겟 $x_t$를 사용하여 선형 업데이트 모델을 적용한다.

$$H_t = \frac{\hat{y} \hat{x}_t * \hat{x}_t^*}{\|\hat{x}_t\|^2 + \lambda}, \quad W_t = (1-\gamma)W_{t-1} + \gamma H_t$$

여기서 $\gamma$는 학습률(learning rate)이다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB2013, OTB2015
- **평가 지표**:
  - Precision plot: 중심 위치 오차(CEL) 20픽셀 이내인 프레임의 비율.
  - Success plot: VOR(Overlap Ratio) 임계값 기준 성공 비율 및 AUC(Area Under the Curve) 측정.
  - OPE(One-Pass Evaluation), TRE(Temporal Robustness Evaluation), SRE(Spatial Robustness Evaluation) 수행.
- **구현 환경**: MATLAB, Intel Core-i3-4170 CPU, 8GB RAM (약 22 FPS)

### 주요 결과

1. **OTB2013 결과**:
    - 제안된 CFPFT 추적기는 Success plot에서 0.584, Precision plot에서 0.821의 점수를 기록하며 비교 대상 중 최상위 성능을 보였다.
    - KCF 대비 Success score는 13.62%, Precision score는 10.95% 향상되었으며, DSST와 비교해도 우수한 성능을 보였다.
    - 11가지 속성(Fast motion, Occlusion 등) 전반에서 최상위권의 성능을 기록했다.

2. **OTB2015 결과**:
    - 딥러닝 기반 추적기인 HDT와 CNN-SVM보다는 약간 낮지만, 그 외의 대부분의 최신 추적기(KCF, DSST, CNT 등)보다 우수한 성능을 보였다.
    - 특히 Fast motion, Background clutter, Occlusion, Scale variation 속성에서 강한 면모를 보였다.
    - 정량적 비교(Table 2) 결과, OTB2013(DPR 82.1%, OSR 58.4%), OTB2015(DPR 75.5%, OSR 54.9%)에서 경쟁력 있는 수치를 달성했다.

## 🧠 Insights & Discussion

### 강점

본 논문은 계산 효율성이 극대화된 CF와 강건성이 뛰어난 PF의 장점을 적절히 결합하였다. 특히 모든 프레임에서 PF를 사용하는 것이 아니라, CF의 응답 값이 임계값 이하로 떨어지는 '위기 상황'에서만 PF를 트리거함으로써 실시간성을 유지하면서도 타겟 소실 문제를 효과적으로 해결하였다. 또한, 복잡한 어파인 변환(Affine transformation) 대신 최대 응답 값의 변화 추이를 이용한 단순한 스케일 평가 방식을 제안하여 연산 비용을 최소화하면서도 스케일 변화에 대응하였다.

### 한계 및 비판적 해석

- **임계값 $\theta$의 의존성**: 시스템의 작동 여부가 임계값 $\theta$와 스케일 판단 임계값 $\phi, \psi$에 크게 의존한다. 이러한 하이퍼파라미터가 다양한 시퀀스에서 범용적으로 작동하는지에 대한 분석이 부족하다.
- **단순한 스케일 업데이트**: 스케일 팩터를 $0.98, 1, 1.02$라는 고정된 값으로 업데이트하는 방식은 매우 단순하다. 급격한 크기 변화가 일어나는 영상에서는 대응 속도가 느릴 가능성이 있다.
- **특징 추출의 한계**: HOG 특징만을 사용하였는데, 최근의 딥러닝 기반 특징 추출기(Deep feature extractor)를 결합했다면 더 높은 정밀도를 얻었을 것으로 판단된다.

## 📌 TL;DR

본 논문은 KCF 추적기가 최대 응답 값에 의존하여 객체를 놓치는 문제를 해결하기 위해, 응답 값이 낮아질 때만 동작하는 **Particle Filter 기반 재탐지 메커니즘**과 **응답 값 변화율 기반의 단순 스케일 평가 방식**을 제안하였다. 실험 결과 OTB2013/2015 데이터셋에서 기존 CF 기반 추적기보다 월등히 높은 강건성을 보였으며, 딥러닝 기반 추적기에 근접하는 성능을 달성하였다. 이 연구는 실시간 추적 시스템에서 '효율적 추적'과 '안정적 재탐지' 사이의 균형을 맞추는 실용적인 접근법을 제시했다는 점에서 의의가 있다.
