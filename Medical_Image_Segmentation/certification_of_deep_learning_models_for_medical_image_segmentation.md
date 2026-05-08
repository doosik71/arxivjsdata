# Certification of Deep Learning Models for Medical Image Segmentation

Othmane Laousy, Alexandre Araujo, Guillaume Chassagnon, Nikos Paragios, Marie-Pierre Revel, and Maria Vakalopoulou (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 모델의 **인증된 강건성(Certified Robustness)** 확보이다. 최근 딥러닝 기반의 분할 모델들은 임상 현장에서 널리 사용되고 있으나, 이미지에 인간이 인식하지 못하는 미세한 섭동(perturbation)을 추가하는 적대적 공격(Adversarial Attacks)에 취약하다는 치명적인 결함이 있다.

특히 의료 분야는 안전성 최우선(safety-critical) 영역이므로, 단순히 경험적인 방어(empirical defense)를 넘어 이론적인 보장을 제공하는 인증된 방어 기법이 필수적이다. 기존의 적대적 훈련(adversarial training)과 같은 경험적 방법은 더 강력한 적응형 공격(adaptive attacks)에 의해 무력화될 수 있다는 한계가 있다. 따라서 본 연구의 목표는 의료 영상 분할 모델에 대해 이론적 강건성 보장을 제공하는 첫 번째 인증된 베이스라인을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Randomized Smoothing** 프레임워크와 **Denoising Diffusion Probabilistic Models (DDPM)**를 결합하여 분할 모델의 인증된 강건성을 높이는 것이다.

중심적인 직관은 다음과 같다. Randomized Smoothing은 입력에 가우시안 노이즈를 추가하여 모델을 부드럽게(smooth) 만듦으로써 이론적 강건성 반경을 계산할 수 있게 하지만, 노이즈 수준이 높아질수록 모델의 정확도가 급격히 떨어지는 trade-off가 발생한다. 이를 해결하기 위해, 세그멘테이션 모델에 입력을 넣기 전, 최신 생성 모델인 DDPM을 사용하여 섭동이 추가된 이미지를 먼저 디노이징(denoising)함으로써 정확도 손실을 최소화하면서 강건성 인증을 달성하고자 한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 본 연구의 차별점을 제시한다.

1. **경험적 방어 (Empirical Defenses):** 적대적 훈련(Adversarial Training) 등이 대표적이며, 의료 영상 분야에서도 일부 적용되었다. 하지만 이러한 방법들은 이론적 보장이 없으며, 새로운 공격 기법에 의해 쉽게 깨질 수 있다는 한계가 있다.
2. **인증된 방어 (Certified Defenses):** 분류(Classification) 작업에서는 이미 많은 연구가 진행되었으며, 최근 Fischer 등이 제안한 SegCertify가 이미지 분할을 위한 첫 번째 인증 접근 방식을 제시했다.
3. **차별점:** 기존의 SegCertify는 가우시안 노이즈가 추가된 데이터로 직접 모델을 학습시켜야 하며, 이 과정에서 깨끗한 이미지에 대한 정확도가 저하되는 문제가 있다. 반면, 본 논문은 기학습된(off-the-shelf) DDPM과 세그멘테이션 모델을 결합한 파이프라인을 제안하여, 별도의 노이즈 학습 없이도 더 높은 인증된 성능을 달성한다.

## 🛠️ Methodology

### 전체 파이프라인

본 논문이 제안하는 시스템은 **"Denoise $\rightarrow$ Certify"**의 두 단계로 구성된다. 섭동이 추가된 입력 이미지 $x$에 대해 DDPM을 통해 노이즈를 제거한 후, 그 결과물을 세그멘테이션 모델에 입력하여 결과를 얻고, 이를 Randomized Smoothing 프레임워크 내에서 인증한다.

### Randomized Smoothing

기본적인 아이디어는 베이스 모델 $f$를 가우시안 분포 $\mathcal{N}(0, \sigma^2 I)$로 컨볼루션하여 새로운 부드러운 분류기 $g$를 만드는 것이다.
$$g(x) = \mathbb{P}_{\eta \sim \mathcal{N}(0, \sigma^2 I)} [f(x+\eta) = y]$$
Cohen 등은 $\Phi$를 표준 가우시안 분포의 누적분포함수(CDF)라고 할 때, 인증된 반경 $R$을 다음과 같이 정의하였다.
$$R = \sigma \Phi^{-1}(g(x))$$
이 반경 $R$ 내의 모든 섭동 $\delta$ ($\|\delta\|_2 \le R$)에 대해 $g(x+\delta) = y$가 유지됨이 이론적으로 보장된다.

### DDPM을 이용한 인증 강화

DDPM은 반복적인 디노이징 과정을 통해 이미지를 복원하는 모델이다. 본 논문은 이를 Randomized Smoothing의 노이즈 제거 단계에 활용한다.

1. **노이즈 매핑:** Randomized Smoothing에서 사용하는 노이즈 $\sigma^2$와 DDPM의 노이즈 스케줄 $\alpha_t$ 사이의 관계를 설정하여 적절한 타임스텝 $t^*$를 계산한다.
    $$\sigma^2 = \frac{1-\alpha_{t^*}}{\alpha_{t^*}}$$
2. **디노이징 절차:**
    - 입력 이미지 $x$에 $\delta \sim \mathcal{N}(0, \sigma^2 I)$를 추가하여 $x_{rs}$를 생성한다.
    - 계산된 $t^*$에서 시작하여 DDPM을 통해 이미지를 디노이징한다. 본 논문에서는 연산 속도와 아티팩트 방지를 위해 $t^*$에서 $t=0$으로 직접 예측하는 **Single-step denoising** 전략을 주로 사용한다.
3. **인증 및 평가:** 디노이징된 이미지를 세그멘테이션 모델에 입력하고, 몬테카를로 샘플링을 통해 각 픽셀의 클래스 확률을 추정한다. 픽셀 수준의 다중 테스트 교정을 위해 Holm-Bonferroni 방법을 적용하여 신뢰 수준 $1-\alpha$를 확보한다.

## 📊 Results

### 실험 설정

- **데이터셋:** 흉부 X-ray(JSRT, Montgomery, Shenzen), 피부 병변(ISIC 2018), 대장 내시경(CVC-ClinicDB) 등 총 5개의 공개 데이터셋을 사용하였다.
- **모델:** UNet, ResUNet++, DeeplabV2 세 가지 아키텍처를 실험하였다.
- **지표:** Certified Dice score, Certified mean IoU, 그리고 모델이 예측을 포기한 픽셀의 비율인 Abstention rate ($\% \varnothing$)를 측정하였다.

### 주요 결과

- **아키텍처 비교:** 세 모델 중 **ResUNet++**가 모든 노이즈 수준($\sigma$)과 반경($R$)에서 가장 강건한 성능을 보였다.
- **제안 방법 vs SegCertify:**
  - 저노이즈($\sigma=0.25$)에서는 노이즈로 학습된 SegCertify가 약간 우세할 수 있으나, 고노이즈($\sigma=1.0$) 환경에서는 SegCertify의 성능이 거의 0으로 떨어지는 반면, 제안 방법은 높은 Certified Dice score를 유지하였다.
  - 특히 제안 방법은 깨끗한 이미지에 대한 성능 저하 없이 강건성을 확보할 수 있다는 점이 큰 장점이다.
- **DDPM의 일반화 능력:** 일반적인 이미지로 학습된 off-the-shelf DDPM이 의료 영상 데이터셋의 디노이징 작업에서도 매우 효과적으로 작동함을 확인하였다.
- **디노이징 전략:** Multi-step denoising보다 Single-step denoising이 속도가 빠를 뿐만 아니라 생성적 아티팩트가 적어 인증 성능이 더 높게 나타났다.

## 🧠 Insights & Discussion

### 강점

본 연구는 의료 영상 분할 분야에서 이론적으로 보장된 강건성을 제공하는 첫 번째 베이스라인을 구축했다는 점에서 의의가 크다. 특히 DDPM을 활용함으로써, "강건성을 위해 정확도를 포기"해야 했던 기존의 trade-off 문제를 효과적으로 완화하였다. 또한, 특정 데이터셋에 맞춰 디노이저를 새로 학습시킬 필요 없이 기학습된 모델을 그대로 사용할 수 있다는 범용성을 입증하였다.

### 한계 및 해석

- **미세 구조의 취약성:** 실험 결과, 쇄골(clavicles)과 같이 크기가 작은 구조물은 노이즈 수준이 높아질 때 Abstention rate가 급격히 증가하는 경향을 보였다. 이는 디노이징 과정에서 미세한 디테일이 손실될 수 있음을 시사한다.
- **경계선 모호함:** 정성적 분석 결과, 디노이징 후 세그멘테이션 경계가 원본보다 덜 날카로워지는 현상이 관찰되었다.
- **가정:** 본 논문은 가우시안 노이즈 기반의 Randomized Smoothing에 의존하고 있으므로, 가우시안 분포를 벗어난 형태의 적대적 공격에 대해서는 별도의 분석이 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 분할 모델의 이론적 안전성을 보장하기 위해 **Randomized Smoothing**과 **DDPM(확산 모델)**을 결합한 인증된 방어 프레임워크를 제안한다. DDPM을 통해 섭동이 추가된 이미지를 먼저 정제함으로써, 기존 인증 방식(SegCertify)보다 훨씬 높은 강건성 반경에서도 높은 세그멘테이션 정확도를 유지할 수 있음을 보였다. 이 연구는 안전성이 필수적인 의료 AI 분야에서 인증된 강건성 벤치마크를 구축하는 기초가 될 것으로 기대된다.
