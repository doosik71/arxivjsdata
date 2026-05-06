# AN INVESTIGATION OF END-TO-END MODELS FOR ROBUST SPEECH RECOGNITION

Archiki Prasad, Preethi Jyothi, and Rajbabu Velmurugan (2021)

## 🧩 Problem to Solve

본 논문은 소음이 존재하는 환경에서도 강건한(robust) 자동 음성 인식(Automatic Speech Recognition, ASR)을 수행하기 위한 End-to-End(E2E) 모델의 전략을 분석한다. 기존의 E2E ASR 연구는 주로 깨끗한 음성(clean speech)에 집중되어 왔으며, 소음이 섞인 음성을 처리하기 위한 방법론에 대한 체계적인 비교 연구가 부족했다.

특히, 소음 문제를 해결하기 위해 입력 데이터를 먼저 정제하는 Speech Enhancement(SE) 기법과, 모델 자체가 소음에 적응하도록 만드는 Model-based Adaptation 기법 중 어떤 것이 더 효과적인지, 그리고 소음의 종류에 따라 어떤 전략을 선택해야 하는지를 규명하는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 소음 제거를 위한 전처리 과정(Front-end SE)과 모델의 내부 구조 및 학습 방식을 변경하는 적응 과정(Model-based Adaptation)을 다각도로 비교 분석하는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

- **Speech Enhancement 기반 접근**: 최신 SE 모델들을 전단에 배치하여 정제된 음성을 ASR 모델에 입력하는 Two-pass 방식을 검토한다.
- **Model-based Adaptation 기반 접근**: 데이터 증강(Data Augmentation), 다중 작업 학습(Multi-task Learning), 적대적 학습(Adversarial Learning)의 세 가지 기법을 통해 모델이 소음에 강건한 특징(noise-invariant features)을 학습하도록 유도한다.
- **소음 유형별 분석**: 정적인 소음(Stationary Noise)과 비정적인 소음(Non-stationary Noise)으로 구분하여 각 기법의 성능 차이를 분석함으로써, 소음 특성에 따른 최적의 방법론을 제시한다.

## 📎 Related Works

기존의 Robust ASR 연구는 주로 입력 음성을 SE 모듈에 통과시킨 후 표준 ASR 시스템에 전달하는 Two-pass 접근 방식을 사용하였다. 또한, 정적인 소음을 가정한 Mean Noise Estimate를 사용하는 Noise Aware Training(NAT)이나, 소음 제거를 위해 CNN을 활용하는 방식 등이 제안되었다.

E2E 모델의 경우, 데이터 증강과 미세 조정(Fine-tuning)을 결합하거나 Variational Autoencoders(VAE)를 이용해 깨끗한 음성 도메인에서 소음 도메인으로의 Domain Adaptation을 시도하는 연구들이 있었다. 하지만 본 논문은 이러한 개별 기법들을 넘어, 다양한 소음 유형에 대해 SE 기반 방식과 모델 적응 방식 간의 성능을 직접적으로 비교했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 기본 ASR 시스템

본 연구는 DeepSpeech 2(DS2)를 기본 E2E ASR 모델로 사용한다. DS2는 2개의 2D Convolutional layer, 5개의 Bidirectional LSTM layer, 그리고 마지막의 Fully-connected (FC) Softmax layer로 구성되며, Connectionist Temporal Classification(CTC) 목적 함수를 사용하여 학습된다.

### 2. Speech Enhancement (Front-End)

다음 세 가지 SE 기법을 적용하여 정제된 음성을 DS2에 입력한다.

- **SE-VCAE**: 시간 영역(time-domain)에서 잠재 특징의 분포를 학습하는 Variance Constrained Autoencoder 기반 모델이다.
- **DeepXi**: 스펙트로그램 상에서 SNR 추정치를 기반으로 마스크(mask)를 생성하여 깨끗한 스펙트로그램을 복원하는 방식이다.
- **DEMUCS**: 인코더-디코더 구조를 통해 파형(waveform) 수준에서 소음을 제거하는 모델이다.

### 3. Model-based Adaptation

- **Data Augmentation-based Training (DAT)**:
  - **Vanilla DAT**: 다양한 SNR 값의 소음을 섞어 학습시킨다.
  - **Soft-Freeze DAT**: 하위 레이어(특징 추출기)가 소음에 강건한 특징을 배우도록, 상위 레이어(FC 및 마지막 2개 LSTM)의 학습률을 0.5배로 낮추는 soft-freezing 기법을 적용한다.
- **Multi-task Learning (MTL)**:
  - ASR 작업과 동시에 소음의 종류를 분류하는 보조 분류기(Auxiliary Classifier)를 학습시킨다.
  - 손실 함수는 다음과 같이 하이브리드 형태로 정의된다:
    $$L_H = \lambda L_{CTC} + \eta(1-\lambda)L_{CE}$$
    여기서 $L_{CTC}$는 ASR 손실, $L_{CE}$는 소음 분류를 위한 Cross Entropy 손실이며, $\lambda$와 $\eta$는 스케일링 계수이다.
- **Adversarial Training (AvT)**:
  - MTL과 유사한 구조이나, 보조 분류기 앞에 Gradient Reversal Layer(GRL)를 추가한다.
  - GRL은 역전파 단계에서 그라디언트의 부호를 반전시켜, 모델이 소음 분류를 어렵게 만드는 방향, 즉 소음에 불변하는(noise-invariant) 표현을 학습하도록 강제한다.

## 📊 Results

### 실험 설정

- **데이터셋**: LibriSpeech(깨끗한 음성) 및 FreeSound에서 수집한 7가지 소음 유형(Babble, Airport/Station, Car, Metro, Cafe, Traffic, AC/Vacuum)을 사용한다.
- **평가 지표**: Word Error Rate(WER)를 사용하며, SNR 범위는 0dB에서 20dB까지 5dB 간격으로 측정한다.

### 주요 결과

- **SE 기법 비교**: DEMUCS가 객관적 SE 지표와 ASR 성능 모두에서 SE-VCAE와 DeepXi를 압도하였다.
- **소음 유형별 성능**:
  - **정적 소음 (Noise A: Car, Metro, Traffic)**: DEMUCS가 모든 모델 적응 기법보다 우수한 성능을 보였다.
  - **비정적 소음 (Noise B: Babble, Airport, Cafe, AC/Vacuum)**: AvT(적대적 학습)가 가장 낮은 WER을 기록하며 가장 강력한 성능을 보였다. MTL 또한 DEMUCS와 유사한 수준의 성능을 보였다.
- **Trade-off 분석**: AvT는 소음 환경(낮은 SNR)에서 매우 강력하지만, 깨끗한 음성에 대한 WER을 증가시키는 성능 저하(degradation)가 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 Robust ASR을 위한 전략 선택이 **소음의 특성**에 따라 달라져야 함을 시사한다.

첫째, 소음이 일정하고 정적인 환경에서는 정교한 SE 모델(예: DEMUCS)을 전단에 배치하는 것이 가장 효과적이다. 둘째, 예측 불가능하고 변동성이 큰 비정적 소음 환경에서는 모델이 소음에 무관한 특징을 추출하도록 하는 적대적 학습(AvT)이 더 유리하다.

다만, SE 기반 방식은 고성능의 사전 학습된 모델이 필요하며, 데이터가 부족할 경우 밑바닥부터 학습시키기 어렵다는 한계가 있다. 반면, 모델 적응 방식(MTL, AvT)은 상대적으로 적은 소음 데이터로도 유의미한 학습이 가능하다는 강점이 있다. 또한, AvT의 경우 소음 강건성과 깨끗한 음성에 대한 인식률 사이의 상충 관계(trade-off)가 존재하므로, 실제 적용 시에는 운용 환경의 SNR 분포를 고려한 하이퍼파라미터 튜닝이 필수적이다.

## 📌 TL;DR

본 연구는 E2E ASR 시스템에서 소음 제거(SE) 전처리와 모델 적응(DAT, MTL, AvT) 기법을 체계적으로 비교하였다. 실험 결과, **정적인 소음에는 DEMUCS 기반의 SE**가, **비정적인 소음에는 적대적 학습(AvT)**이 가장 효과적임을 밝혀냈다. 특히 AvT는 소음 환경에서 탁월하지만 깨끗한 음성 인식률을 일부 희생시킨다는 점을 확인하였으며, 이는 향후 소음 특성에 맞춘 맞춤형 ASR 설계의 중요성을 제시한다.
