# Multi-SpectroGAN: High-Diversity and High-Fidelity Spectrogram Generation with Adversarial Style Combination for Speech Synthesis

Sang-Hoon Lee, Hyun-Wook Yoon, Hyeong-Rae Noh, Ji-Hoon Kim, Seong-Whan Lee (2021)

## 🧩 Problem to Solve

기존의 GAN(Generative Adversarial Networks) 기반 신경망 텍스트 음성 합성(TTS) 시스템은 음성 합성의 품질을 크게 향상시켰으나, 여전히 생성된 멜-스펙트로그램(mel-spectrogram)과 실제 정답(ground-truth) 데이터 사이의 직접적인 재구성 손실(reconstruction loss)에 의존하여 학습하는 한계가 있다. 이는 adversarial feedback(적대적 피드백)만으로는 생성기(generator)를 충분히 학습시키기에 부족하기 때문이다.

또한, 대량의 고품질 텍스트-오디오 데이터 없이는 화자의 스타일을 정교하게 제어하거나 전이(transfer)하는 것이 어려우며, 특히 학습 데이터에 포함되지 않은 새로운 스타일(unseen speaking style)이나 텍스트에 대해 일반화 성능을 확보하는 것이 중요한 과제로 남아 있다. 따라서 본 논문의 목표는 정답 데이터와의 직접적인 비교 손실 없이 적대적 피드백만으로 학습 가능하며, 다양한 스타일의 음성을 고충실도로 합성할 수 있는 Multi-SpectroGAN(MSG)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 생성기의 자기지도 학습된 은닉 표현(self-supervised hidden representation)을 조건부 판별기(conditional discriminator)에 제공함으로써, 정답 데이터와의 직접적인 MSE/MAE 손실 없이도 생성기를 효과적으로 가이드하는 것이다.

또한, 여러 화자의 스타일 임베딩을 선형 결합하는 Adversarial Style Combination(ASC) 기법을 도입하였다. 이를 통해 모델은 학습 과정에서 보지 못한 스타일의 잠재 표현(latent representation)을 학습하게 되며, 결과적으로 다양한 화자 스타일의 보간(interpolation) 및 제어가 가능해져 합성 음성의 다양성과 일반화 성능을 극대화한다.

## 📎 Related Works

기존의 TTS 연구는 크게 두 가지 흐름으로 나뉜다. 첫째, Tacotron과 같은 자기회귀(autoregressive) 모델은 RNN 기반의 어텐션 메커니즘을 사용하여 멜-스펙트로그램을 생성하지만, 추론 속도가 느리고 단어 누락이나 반복과 같은 정렬(alignment) 문제가 발생한다. 둘째, FastSpeech 및 FastSpeech2와 같은 비자기회귀(non-autoregressive) 모델은 Transformer 구조와 길이 조절기(length regulator)를 사용하여 병렬 생성을 가능하게 함으로써 속도와 안정성을 높였다.

음성 품질을 높이기 위한 보코더(vocoder) 연구에서는 WaveNet 같은 자기회귀 모델에서 시작하여, 최근에는 Parallel WaveGAN과 같이 GAN을 이용해 실시간으로 고품질 오디오를 생성하는 방식이 주목받고 있다. 하지만 이러한 GAN 기반 모델들도 멜-스펙트로그램 생성 단계에서는 여전히 정답 데이터와의 재구성 손실에 의존하고 있다. 본 논문은 이러한 제약을 제거하고 적대적 학습만으로 고품질 스펙트로그램을 생성한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

Multi-SpectroGAN의 생성기는 FastSpeech2 아키텍처를 기반으로 하며, 크게 Phoneme Encoder, Style-conditional Variance Adaptor, 그리고 Decoder로 구성된다. 여기에 멜-스펙트로그램으로부터 고정 차원의 스타일 벡터를 추출하는 Style Encoder가 추가되었다.

### 주요 구성 요소 및 절차

1. **Style Encoder**: 6층의 1D Convolutional network와 GRU 레이어를 사용하여 입력 멜-스펙트로그램 $y$로부터 스타일 임베딩 $s$를 추출한다.
    $$s = E_s(y)$$
2. **Style-conditional Variance Adaptor**: 음소 인코더의 출력 $H^{pho}$에 스타일 임베딩 $s$를 결합하여, 각 화자의 고유한 특성이 반영된 지속 시간(duration), 피치(pitch), 에너지(energy)를 예측한다.
    - 지속 시간 예측: $\hat{D} = \text{DurationPredictor}(H^{pho}, s)$
    - 피치 및 에너지 예측: $\hat{P} = \text{PitchPredictor}(H^{mel}, s), \hat{E} = \text{EnergyPredictor}(H^{mel}, s)$
3. **Generator**: 예측된 분산 정보와 스타일 임베딩을 합산하여 최종 은닉 표현 $H^{total}$을 구성하고, 이를 디코더 $g(\cdot)$에 통과시켜 멜-스펙트로그램 $\hat{y}$를 생성한다.
    $$H^{total} = H^{mel} + s + p + e + PE(\cdot)$$
    $$\hat{y} = g(H^{total})$$

### Frame-level Conditional Discriminator

본 모델의 핵심은 정답 데이터와의 직접 비교 없이 학습하기 위해 도입된 조건부 판별기이다. 판별기는 생성기가 학습 과정에서 만든 프레임 레벨 조건 $c$를 입력으로 받는다.
$$c = H^{mel} + s + p + e$$
여기서 $c$는 언어적 정보($H^{mel}$), 스타일($s$), 피치($p$), 에너지($e$)의 합으로 구성된다. 판별기는 $D_k(y, c)$와 $D_k(\hat{y}, c)$를 구분하도록 학습하며, LSGAN(Least Squares GAN) 목적 함수를 사용한다.

### 손실 함수 및 학습 목표

생성기는 적대적 손실($L_{adv}$), 특징 매칭 손실($L_{fm}$), 그리고 분산 예측 손실($L_{var}$)의 합으로 학습된다.
$$\min_{f,g} L_{msg} = L_{adv} + \lambda L_{fm} + \mu L_{var}$$
특징 매칭 손실 $L_{fm}$은 판별기의 중간 레이어 feature map 사이의 MAE를 최소화하여 생성기가 더 정교한 특징을 학습하도록 돕는다.

### Adversarial Style Combination (ASC)

다양성을 높이기 위해 두 화자의 스타일 임베딩을 섞은 $s_{mix}$를 생성한다.
$$s_{mix} = \alpha s_i + (1-\alpha)s_j$$
여기서 $\alpha$는 베르누이 분포(Binary selection) 또는 균등 분포(Manifold mixup)에서 샘플링된다. 이렇게 섞인 스타일로 생성된 $\hat{y}_{mix}$를 판별기가 실제 데이터처럼 인식하도록 학습시켜, 모델이 보지 못한 새로운 스타일 영역을 탐색하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 단일 화자 모델은 LJ-speech, 다중 화자 모델은 VCTK 데이터셋을 사용하였다.
- **지표**: 주관적 평가인 MOS(Mean Opinion Score), 객관적 평가인 MCD(Mel-Cepstral Distortion), $F_0$ RMSE, 화자 분류 정확도(Top-1 acc.)를 측정하였다.
- **보코더**: 생성된 멜-스펙트로그램을 오디오로 변환하기 위해 사전 학습된 PWG(Parallel WaveGAN)를 사용하였다.

### 주요 결과

1. **단일 화자 합성**: MSG 모델은 정답(GT) 멜-스펙트로그램과 거의 동일한 수준의 MOS 점수를 기록하여, 적대적 학습만으로도 매우 높은 자연스러움을 확보했음을 입증하였다.
2. **다중 화자 합성**: FastSpeech2 대비- 보인 화자(seen speaker)에서는 0.08, 보이지 않은 화자(unseen speaker)에서는 0.13만큼 MOS 점수가 향상되었다.
3. **ASC의 효과**: ASC를 적용한 모델은 특히 보이지 않은 화자에 대해 더 나은 일반화 성능을 보였으며, 화자 분류 정확도와 $F_0$ RMSE에서도 긍정적인 결과가 나타났다.
4. **Ablation Study**: 판별기의 조건 $c$에서 언어적 정보($H^{mel}$)가 없을 경우 모델이 전혀 학습되지 않았으며, 이는 프레임 레벨의 언어 정보가 스펙트로그램 생성의 핵심 가이드임을 시사한다.

## 🧠 Insights & Discussion

본 연구는 TTS 시스템에서 정답 데이터와의 직접적인 픽셀 단위 비교(reconstruction loss) 없이도 고품질의 음성 합성이 가능하다는 것을 증명하였다. 이는 특히 정답 데이터가 부족하거나, 정답과는 다른 새로운 스타일의 음성을 생성해야 하는 상황에서 매우 강력한 이점을 가진다. 특히 프레임 레벨의 조건부 판별기를 통해 생성기가 학습해야 할 방향성을 명확히 제시한 점이 주효했다.

한계점으로는, 여전히 멜-스펙트로그램을 오디오로 변환하기 위해 외부 보코더(PWG)에 의존하고 있다는 점이 있다. 또한, ASC를 통해 스타일을 보간하는 과정에서 $\alpha$ 값에 따른 음성 변화의 선형성이 얼마나 엄격하게 유지되는지에 대한 정밀한 분석이 추가될 필요가 있다. 하지만 전반적으로 적대적 학습과 스타일 보간을 결합하여 TTS의 표현력과 다양성을 동시에 잡은 연구라고 평가할 수 있다.

## 📌 TL;DR

Multi-SpectroGAN은 정답 데이터와의 직접적인 재구성 손실($L_{rec}$) 없이, **프레임 레벨 조건부 판별기**를 통한 적대적 피드백만으로 고충실도 멜-스펙트로그램을 생성하는 모델이다. 특히 **Adversarial Style Combination(ASC)** 기법을 통해 여러 화자의 스타일을 보간함으로써, 학습 데이터에 없는 새로운 스타일의 음성까지 다양하고 자연스럽게 합성할 수 있게 하였다. 이 연구는 향후 소량의 데이터만으로 새로운 목소리를 구현하는 Few-shot 학습이나 교차 언어 스타일 전이 연구에 중요한 기반이 될 것으로 보인다.
