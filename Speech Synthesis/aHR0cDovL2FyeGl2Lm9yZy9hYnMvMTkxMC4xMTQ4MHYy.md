# PARALLEL WAVEGAN: A FAST WAVEFORM GENERATION MODEL BASED ON GENERATIVE ADVERSARIAL NETWORKS WITH MULTI-RESOLUTION SPECTROGRAM

Ryuichi Yamamoto, Eunwoo Song, Jae-Min Kim

## 🧩 Problem to Solve

- **느린 추론 속도:** WaveNet과 같은 자기회귀(autoregressive) 생성 모델은 고품질 음성 합성을 제공하지만, `autoregressive`한 특성 때문에 추론 속도가 매우 느려 실시간 응용에 제한이 있습니다.
- **복잡한 훈련 과정:** 교사-학생(teacher-student) 프레임워크 기반의 병렬 WaveNet (예: ClariNet)은 추론 속도 문제를 해결하지만, 잘 훈련된 교사 모델과 복잡한 밀도 증류(density distillation) 과정을 최적화하기 위한 시행착오 방식이 필요하여 훈련이 어렵고 시간이 오래 걸립니다.
- **품질 저하 가능성:** 기존 GAN 기반 접근 방식 중 일부(예: GELP)는 LP(Linear Prediction) 파라미터를 사용하여 음성 파형을 생성하는데, TTS 음향 모델에서 발생하는 불가피한 오류로 인해 품질 저하가 발생할 수 있습니다.

## ✨ Key Contributions

- **증류 없는(distillation-free) 병렬 파형 생성 방법 제안:** 기존의 복잡한 교사-학생 증류 프레임워크 없이 WaveNet 기반 생성자를 훈련하는 방법을 제시합니다.
- **다중 해상도 STFT 손실과 파형 영역 적대적 손실의 공동 최적화:** 현실적인 음성 파형의 시간-주파수 분포를 효과적으로 포착하기 위해 다중 해상도 STFT 손실($L_{\text{aux}}$)과 적대적 손실($L_{\text{adv}}$)을 함께 사용하여 모델을 훈련합니다.
- **훈련 및 추론 시간 대폭 단축:** 제안된 Parallel WaveGAN은 훈련 시간을 4.82배 (13.5일 $\rightarrow$ 2.8일), 추론 시간을 1.96배 (14.62 RTF $\rightarrow$ 28.68 RTF) 단축시킵니다 (24 kHz 음성, 단일 GPU 기준).
- **소형 모델 아키텍처:** 1.44M 파라미터의 작은 모델 크기에도 불구하고 고품질 음성 생성이 가능합니다.
- **Transformer 기반 TTS 프레임워크와의 결합 및 경쟁력 있는 MOS 달성:** Transformer 기반 TTS 모델과 결합하여 4.16 MOS를 달성, 최고의 증류 기반 Parallel WaveNet 시스템(ClariNet-GAN)과 비교해도 경쟁력 있는 지각적 품질을 제공합니다.

## 📎 Related Works

- **자기회귀 생성 모델:** WaveNet [4]은 고품질 음성을 생성하지만, 자기회귀적 특성으로 인해 추론 속도가 느립니다.
- **병렬 파형 생성 (Teacher-student framework):** WaveNet의 속도 문제를 해결하기 위해 등장했으며, 밀도 증류를 통해 자기회귀 교사 모델의 지식을 비자기회귀 학생 모델(예: Inverse Autoregressive Flow (IAF) 기반 ClariNet [10, 12])로 전달합니다 [9, 11].
- **GAN을 이용한 파형 생성:** 이전 연구에서 IAF 학생 모델을 생성자로 활용하여 KLD 및 보조 손실과 함께 적대적 손실을 최소화했지만, 복잡한 증류 훈련 단계가 한계로 지적되었습니다 [11]. GAN-excited Linear Prediction (GELP) [18]은 적대적 훈련을 통해 성대 여기(glottal excitations)를 생성하지만, LP 파라미터의 오류로 인한 품질 저하 가능성이 있습니다.

## 🛠️ Methodology

- **생성자 (Generator, G):**
  - 멜 스펙트로그램과 같은 보조 특징에 조건화된 비자기회귀(non-autoregressive) WaveNet 모델을 사용합니다.
  - 원래 WaveNet과 달리 비인과적(non-causal) 컨볼루션을 사용하며, 입력은 가우시안 분포에서 샘플링된 랜덤 노이즈 `z`입니다.
  - 훈련 및 추론 단계 모두에서 비자기회귀적으로 작동하여 병렬 생성이 가능합니다.
  - 30개 계층의 팽창 잔차(dilated residual) 컨볼루션 블록과 3개의 지수적으로 증가하는 팽창 주기(dilation cycle)로 구성됩니다.
- **판별자 (Discriminator, D):**
  - 생성된 샘플을 '가짜'로, 실제 샘플을 '진짜'로 분류하도록 훈련됩니다.
  - 10개 계층의 비인과적 팽창 1-D 컨볼루션과 Leaky ReLU 활성화 함수($\alpha = 0.2$)를 사용합니다.
- **손실 함수:**
  - **적대적 손실 ($L_{\text{adv}}$):** LSGAN (Least Squares GAN) [19]을 채택하여 훈련 안정성을 높였습니다.
    - 생성자 손실: $L_{\text{adv}}(G, D) = E_{z \sim N(0,I)}[(1-D(G(z)))^2]$
    - 판별자 손실: $L_D(G, D) = E_{x \sim p_{\text{data}}}[(1-D(x))^2] + E_{z \sim N(0,I)}[D(G(z))^2]$
  - **다중 해상도 STFT 보조 손실 ($L_{\text{aux}}$):** 적대적 훈련의 안정성과 효율성을 향상시키기 위해 제안되었으며, 세 가지 다른 FFT 크기, 윈도우 크기, 프레임 시프트를 사용하여 계산된 개별 STFT 손실($L_s$)의 합으로 구성됩니다.
    - 개별 STFT 손실: 스펙트럼 수렴 손실 ($L_{\text{sc}}$)과 로그 STFT 진폭 손실 ($L_{\text{mag}}$)의 합.
      $$L_s(G) = E_{z \sim p(z),x \sim p_{\text{data}}}[L_{\text{sc}}(x, \hat{x}) + L_{\text{mag}}(x, \hat{x})]$$
      $$L_{\text{sc}}(x, \hat{x}) = \frac{\||STFT(x)|-|STFT(\hat{x})|\|_{F}}{\||STFT(x)|\|_{F}}$$
      $$L_{\text{mag}}(x, \hat{x}) = \frac{1}{N}\||log|STFT(x)|-log|STFT(\hat{x})||\|_{1}$$
    - 총 다중 해상도 STFT 손실: $L_{\text{aux}}(G) = \frac{1}{M}\sum_{m=1}^{M}L_s^{(m)}(G)$ (여기서 $M=3$)
  - **생성자의 최종 손실 ($L_G$):** $L_{\text{aux}}$와 $L_{\text{adv}}$의 선형 결합입니다.
    $$L_G(G, D) = L_{\text{aux}}(G) + \lambda_{\text{adv}}L_{\text{adv}}(G, D)$$
    여기서 $\lambda_{\text{adv}}$는 4.0으로 설정됩니다.
- **훈련 절차:**
  - 처음 100K 스텝 동안은 판별자를 고정하고 생성자만 훈련하며, 이후 생성자와 판별자를 공동으로 훈련합니다.
  - RAdam 옵티마이저를 사용하며, 총 400K 스텝 동안 훈련됩니다.

## 📊 Results

- **음성 품질 (MOS):**
  - **분석/합성(Analysis/Synthesis) 환경:**
    - Parallel WaveGAN (다중 해상도 STFT + Adversarial): 4.06 MOS.
    - 최고 성능의 ClariNet (증류 + 다중 해상도 STFT): 4.21 MOS.
  - **Transformer 기반 TTS 프레임워크 환경:**
    - Transformer + Parallel WaveGAN (ours): **4.16 MOS**.
    - Transformer + ClariNet-GAN: 4.14 MOS.
    - 녹음 음성 (Recording): 4.46 MOS.
- **추론 속도 (RTF, Real-Time Factor):**
  - Parallel WaveGAN: 28.68배 실시간 빠름 (24 kHz 음성, 단일 NVIDIA Tesla V100 GPU).
  - 기존 ClariNet: 14.62배 실시간 빠름.
- **훈련 시간:**
  - Parallel WaveGAN: **2.8일** (2x NVIDIA Tesla V100 GPU).
  - ClariNet (교사 WaveNet 훈련 시간 포함): 12.7일.
  - ClariNet-GAN (교사 WaveNet 훈련 시간 포함): 13.5일.
- **모델 크기:**
  - Parallel WaveGAN: 1.44 M 파라미터.
  - ClariNet: 2.78 M 파라미터.

## 🧠 Insights & Discussion

- **다중 해상도 STFT 손실의 중요성:** 다중 해상도 STFT 손실을 사용함으로써 단일 STFT 손실보다 훨씬 높은 지각적 품질을 달성했습니다. 이는 음성 신호의 다양한 시간-주파수 특성을 효과적으로 학습하는 데 기여합니다.
- **증류 없는 훈련의 효율성:** Parallel WaveGAN은 복잡한 밀도 증류 과정 없이도 경쟁력 있는 음성 품질을 달성하며, 기존 ClariNet 대비 4.82배 빠른 훈련 시간을 보여 훈련 과정의 효율성을 크게 개선했습니다. 이는 교사 모델 훈련 및 복잡한 증류 과정 최적화가 필요 없기 때문입니다.
- **GAN의 견고성(Robustness) 향상:** 분석/합성 환경에서는 ClariNet-GAN과 유사한 성능을 보였지만, Transformer 기반 TTS 프레임워크와 결합했을 때, 적대적 손실의 사용이 음향 모델의 예측 오류에 대한 모델의 견고성을 향상시키는 데 유리함이 확인되었습니다.
- **실용적인 적용 가능성:** 1.44M 파라미터의 작은 모델 크기와 28.68배 실시간 빠른 추론 속도는 Parallel WaveGAN이 실시간 응용 및 경량화된 환경에서 사용될 수 있음을 시사합니다.
- **한계점 및 향후 연구:** 현재의 다중 해상도 STFT 보조 손실을 음성 특성을 더 잘 포착하도록 개선(예: 위상 관련 손실 도입)하고, 다양한 말뭉치(corpus)에 대한 성능 검증이 필요합니다.

## 📌 TL;DR

본 논문은 복잡한 밀도 증류(density distillation) 과정 없이 고속, 고품질, 소형 음성 파형 생성을 위한 Parallel WaveGAN을 제안합니다. 이 모델은 비자기회귀 WaveNet을 생성자로, GAN 기반 판별자를 사용하며, 다중 해상도 STFT 손실과 파형 영역 적대적 손실을 공동으로 최적화하여 사실적인 음성의 시간-주파수 분포를 학습합니다. 결과적으로 Parallel WaveGAN은 1.44M 파라미터의 경량 모델로 28.68배 실시간 빠른 24 kHz 음성 생성이 가능하며, Transformer 기반 TTS 시스템과 결합 시 4.16 MOS를 달성하여 기존의 복잡한 증류 기반 모델과 경쟁력 있는 성능을 보였습니다. 훈련 시간 또한 기존 방식 대비 4배 이상 단축하여 훈련 및 추론 효율성을 크게 높였습니다.
