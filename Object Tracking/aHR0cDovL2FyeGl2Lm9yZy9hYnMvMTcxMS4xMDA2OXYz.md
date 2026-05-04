# Particle Filter Re-detection for Visual Tracking via Correlation Filters

Di Yuan, Xiaohuan Lu, Donghao Li, Yingyi Liang, Xinming Zhang (2018)

## 🧩 Problem to Solve

본 논문은 Correlation Filter (CF) 기반의 비주얼 트래킹 알고리즘이 가진 치명적인 결함인 '타겟 위치 추정의 부정확성' 문제를 해결하고자 한다. CF 기반 트래커는 연산 속도가 매우 빠르고 전반적인 성능이 우수하지만, 응답 맵(Response Map)의 최댓값(Maximum Response Value)에 지나치게 의존한다는 한계가 있다.

이러한 특성 때문에 배경 잡음이 심하거나, 타겟의 빠른 움직임, 혹은 가려짐(Occlusion)이 발생하는 복잡한 환경에서는 응답 맵이 모호해지며, 결과적으로 최댓값이 작아지거나 잘못된 위치에서 발생하게 된다. 이는 트래커가 타겟을 놓치거나(Lost) 엉뚱한 곳으로 튀는 드리프트(Drift) 현상으로 이어진다. 따라서 본 연구의 목표는 CF의 빠른 속도를 유지하면서도, 신뢰도가 떨어지는 상황에서 타겟을 정확하게 다시 찾아낼 수 있는 효율적인 재탐지(Re-detection) 메커니즘과 강건한 스케일 평가 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Correlation Filter(CF)의 효율성**과 **Particle Filter(PF)의 강건성**을 결합하는 것이다.

1. **Particle Filter 기반의 재탐지 메커니즘**: 평상시에는 KCF(Kernelized Correlation Filter)를 통해 빠르게 추적하지만, 응답 맵의 최댓값이 설정된 임계값보다 낮아져 신뢰도가 떨어질 경우, Particle Filter의 리샘플링 전략을 사용하여 여러 후보 영역을 탐색함으로써 타겟을 재탐지한다.
2. **새로운 스케일 평가 전략**: 복잡한 아핀 변환(Affine Transformation) 대신, 연속된 프레임 간의 최댓값 응답 변화율을 분석하여 타겟의 크기 변화(확대, 축소, 유지)를 판단하는 단순하고 효율적인 메커니즘을 제안한다.

## 📎 Related Works

### 1. Particle Filter 기반 트래커

Particle Filter는 비선형 및 비가우시안 가정을 통해 상태를 추정하므로 가려짐이나 복잡한 움직임에 강건하며, 여러 가설을 동시에 유지할 수 있다는 장점이 있다. 그러나 모든 입자(Particle)에 대해 계산을 수행해야 하므로 연산 복잡도가 매우 높아 실시간 추적에 어려움이 있다.

### 2. Correlation Filter 기반 트래커

CF 기반 트래커(KCF, DSST 등)는 푸리에 변환(FFT)을 통해 공간 도메인의 컨볼루션을 주파수 도메인의 요소별 곱셈으로 변환함으로써 연산 속도를 극대화했다. 하지만 앞서 언급했듯 최댓값 응답에 과도하게 의존하여, 응답 맵이 모호해지는 상황에서 타겟을 쉽게 놓치는 한계가 있다.

### 3. 차별점

제안 방법은 PF를 전체 추적 과정에 사용하는 것이 아니라, CF의 결과가 불확실한 '특정 상황'에서만 선택적으로 사용한다. 또한, CF의 빠른 연산을 PF의 후보군 평가에 활용함으로써 PF의 고질적인 문제인 연산량 문제를 해결하고 CF의 취약점인 드리프트 문제를 보완했다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 시스템은 크게 **CF-tracking part**와 **Re-detection part**의 두 부분으로 구성된다.

- 초기 프레임에서 타겟 특징을 추출하고 CF 모델을 학습시킨다.
- $t$번째 프레임에서 검색 윈도우로부터 특징을 추출하고 응답 맵을 계산한다.
- 최댓값 응답 $\max R$과 임계값 $\theta$를 비교하여 $\max R \ge \theta$이면 CF-tracking 결과를 그대로 사용하고, $\max R < \theta$이면 Re-detection part를 통해 타겟을 다시 찾는다.
- 결정된 타겟 위치를 바탕으로 CF 모델을 업데이트한다.

### 2. KCF 기반 추적 프레임워크

KCF는 HOG 특징을 사용하며, 가우시안 커널을 통해 비선형 회귀 문제를 선형 문제로 변환하여 해결한다. 필터 $w$는 다음의 최소화 문제를 통해 학습된다.
$$w = \arg \min_{w} \sum_{m,n} |\langle \phi(x_{m,n}), w \rangle - y(m,n)|^2 + \lambda \|w\|^2$$
여기서 $\phi$는 커널 공간으로의 매핑, $\lambda$는 정규화 파라미터이다. 추론 단계에서는 입력 패치 $z$에 대해 다음과 같이 응답 점수를 계산한다.
$$f(z) = \mathcal{F}^{-1}(\mathcal{F}(k_z) \odot \mathcal{F}(\alpha))$$
$\mathcal{F}$는 FFT, $\mathcal{F}^{-1}$은 IFFT, $\odot$은 요소별 곱셈을 의미한다.

### 3. Particle Filter 재탐지 모델

응답 맵이 신뢰할 수 없을 때, 이전 타겟 위치를 중심으로 정규분포를 따르는 $M$개의 입자(이미지 패치)를 생성한다. 각 입자 $z_m$에 대해 CF를 적용하여 응답 맵 $R_m$을 계산한다.
$$R_m = \mathcal{F}^{-1}(\mathcal{F}(\langle z_m, \hat{x} \rangle) \odot \mathcal{F}(\alpha))$$
최종적으로 $M$개의 입자 중 가장 큰 최댓값을 가진 입자를 타겟의 새 위치로 결정한다.
$$\max R_{pf} = \max \{ \max R_1, \max R_2, \dots, \max R_M \}$$

### 4. 스케일 평가 알고리즘

타겟의 크기 변화를 감지하기 위해 연속된 프레임 간의 응답 값 변화율 $d_t$를 계산한다.
$$d_t = \frac{\max R_t}{\max R_{t-1}} - \frac{\max R_{t-1}}{\max R_{t-2}}$$

- $d_t > \phi$ 이면: 타겟 크기가 작아짐 $\rightarrow$ 스케일 인자 $s_t = 0.98$
- $d_t < \psi$ 이면: 타겟 크기가 커짐 $\rightarrow$ 스케일 인자 $s_t = 1.02$
- 그 외: 크기 변화 없음 $\rightarrow$ 스케일 인자 $s_t = 1$
최종 크기는 $\text{size}_t = \text{size}_{t-1} \times s_t$로 갱신된다.

### 5. 모델 업데이트

현재 프레임의 타겟 $\hat{x}_t$를 이용하여 선형 업데이트 모델을 적용한다.
$$H_t = \frac{\hat{y} \hat{x}_t \odot \hat{x}_t^* + \lambda}{\odot \hat{x}_t}, \quad W_t = (1-\gamma)W_{t-1} + \gamma H_t$$
여기서 $\gamma$는 학습률(learning rate)이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: OTB2013 (51개 시퀀스), OTB2015 (100개 시퀀스)
- **지표**: 정밀도(Precision), 성공률(Success - VOR 기준), OPE, TRE, SRE
- **환경**: Intel Core-i3-4170 CPU, 8GB RAM (약 22 FPS로 동작)

### 2. 주요 결과

- **OTB2013**: 제안한 CFPFT 트래커는 정밀도와 성공률 모두에서 최상위권 성능을 기록하였다. 특히 KCF 대비 성공률은 13.62%, 정밀도는 10.95% 향상되었으며, DSST와 비교해서도 유의미한 성능 향상을 보였다.
- **OTB2015**: 딥러닝 기반 트래커(HDT, CNN-SVM)보다는 약간 낮지만, 그 외의 CF 기반 트래커(KCF, DSST) 및 대표적인 전통적 트래커(TGPR, SCM 등)보다 우수한 성능을 보였다.
- **속성별 분석**: 특히 빠른 움직임(Fast Motion), 배경 잡음(Background Clutter), 가려짐(Occlusion), 스케일 변화(Scale Variation) 속성에서 강건함이 입증되었다.

## 🧠 Insights & Discussion

본 논문은 CF의 속도와 PF의 강건함을 전략적으로 결합하여, 추적 실패 상황에서의 복구 능력을 극대화했다는 점에서 강점이 있다. 특히 복잡한 수학적 모델 없이 응답 값의 변화율만으로 스케일 변화를 추정하는 방식은 연산 효율성을 유지하면서도 실용적인 성능 향상을 가져왔다.

다만, 재탐지를 결정하는 임계값 $\theta$와 스케일 판단 임계값 $\phi, \psi$가 하이퍼파라미터로 설정되어 있어, 데이터셋이나 환경에 따라 최적값이 달라질 수 있다는 점이 한계로 보인다. 또한, 입자 생성 시 정규분포를 가정하므로, 타겟의 움직임이 매우 불규칙하거나 급격할 경우 입자가 타겟 영역을 벗어날 가능성이 존재한다.

결론적으로 본 연구는 CF 기반 트래커의 최대 약점인 '신뢰도 낮은 응답 맵으로 인한 드리프트' 문제를 PF라는 안전장치를 통해 효과적으로 해결하였음을 보여준다.

## 📌 TL;DR

이 논문은 CF 기반 트래커의 빠른 속도를 유지하면서, 응답 맵이 불확실할 때만 Particle Filter를 통해 타겟을 재탐색하는 **CFPFT** 알고리즘을 제안한다. 또한, 연속 프레임 간 응답 값의 변화율을 이용한 단순한 스케일 조정 메커니즘을 도입하였다. 실험 결과, OTB2013/2015 벤치마크에서 기존 CF 기반 방법론들을 크게 상회하는 강건성과 정확도를 보였으며, 이는 향후 실시간성이 중요한 환경에서 신뢰도 높은 객체 추적 시스템을 구축하는 데 중요한 기초가 될 수 있다.
