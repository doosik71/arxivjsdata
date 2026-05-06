# UNIVERSAL NEURAL VOCODING WITH PARALLEL WAVENET

Yunlong Jiao, Adam Gabryś, Georgi Tinchev, Bartosz Putrycz, Daniel Korzekwa, Viacheslav Klimkov (2021)

## 🧩 Problem to Solve

본 논문은 텍스트 음성 변환(Text-to-Speech, TTS) 시스템의 두 번째 단계인 보코더(vocoder)의 확장성 문제를 해결하고자 한다. 기존의 신경망 기반 보코더들은 대개 학습 데이터에 과적합(overfitting)되는 경향이 있어, 학습 과정에서 보지 못한 새로운 화자의 목소리를 생성할 때 품질이 저하되는 문제가 발생한다.

이를 해결하기 위해 화자별로 개별 모델을 구축하는 Speaker-dependent 보코더를 사용해 왔으나, 이는 화자마다 막대한 양의 데이터와 계산 자원을 필요로 하므로 수많은 화자를 지원해야 하는 실제 서비스 환경에서는 비효율적이다. 따라서 본 연구의 목표는 다양한 연령, 성별, 언어, 스타일을 가진 화자들에 대해 범용적으로 적용 가능하면서도, 실시간 추론이 가능한 고품질의 Universal neural vocoder를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Parallel WaveNet(PW) 아키텍처에 **Audio Encoder**라는 추가적인 컨디셔닝 네트워크를 도입하여, 멜-스펙트로그램(mel-spectrogram)만으로는 부족한 화자의 전역적인 특징(global characteristics)을 명시적으로 모델링하는 것이다.

Audio Encoder는 참조 오디오(reference waveform)를 고정된 차원의 특징 벡터로 인코딩하며, 이를 통해 보코더가 특정 화자의 음색이나 스타일 정보를 전달받아 보다 자연스러운 음성을 생성할 수 있게 한다. 이를 통해 비자기회귀(non-autoregressive) 방식의 빠른 생성 속도를 유지하면서도, 새로운 화자에 대한 일반화 성능을 획기적으로 높였다.

## 📎 Related Works

기존의 신경망 보코더 연구들은 크게 두 가지 방향으로 진행되었다. 첫째는 WaveNet이나 WaveRNN과 같은 자기회귀(autoregressive) 모델로, 고품질의 음성을 생성하지만 샘플을 순차적으로 생성해야 하므로 추론 속도가 매우 느려 실시간 적용이 어렵다는 한계가 있다. 둘째는 Parallel WaveNet, WaveGlow, Parallel WaveGAN과 같은 비자기회귀 모델로, 병렬 생성이 가능해 속도는 매우 빠르지만 화자 독립적인(speaker-independent) 설정에서 품질이 저하되거나 일반화 능력이 부족한 경우가 많았다.

최근 Universal WaveRNN과 같은 범용 보코더 연구가 제시되었으나, 이는 여전히 자기회귀 방식의 속도 문제를 안고 있다. 본 논문은 비자기회귀 구조를 유지하면서도 범용성을 확보한 모델이 가능한지를 탐구하며, 기존의 단순한 Speaker-embedding 방식이 아닌 오디오 신호를 직접 인코딩하는 방식을 통해 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 **Universal Parallel WaveNet (UPW)**은 크게 두 가지 컨디셔닝 네트워크와 Parallel WaveNet 디코더로 구성된다.

1. **Audio Encoder**: 참조 오디오 파형을 입력받아 화자의 전역 특징을 추출하는 네트워크이다.
2. **Mel-spectrogram Conditioner**: 입력 멜-스펙트로그램을 처리하여 시간적 정보를 제공하는 네트워크이다.
3. **Parallel WaveNet**: 위 두 네트워크의 출력을 조건으로 하여 노이즈를 오디오 파형으로 변환한다.

### 주요 구성 요소 및 상세 설명

**1. Audio Encoder**
MelGAN의 판별자(discriminator) 설계에서 영감을 받은 멀티스케일 구조를 가진다.

- 3개의 동일한 오디오 인코딩 레이어로 구성되며, 각 레이어 사이에는 평균 풀링(average pooling)이 배치되어 서로 다른 시간 스케일의 특징을 추출한다.
- 각 인코딩 레이어는 Strided Convolutional layers, Weight Normalization, Leaky ReLU로 구성된다.
- 최종적으로 Global Max Pooling과 Dense layer를 거쳐 48차원의 오디오 특징 벡터 $e$를 생성한다.
- 정보 누출(information leakage)을 방지하기 위해 Amortized Variational Encoding을 적용하여 변분 오토인코더(VAE) 구조를 취한다.

**2. Mel-spectrogram Conditioner**

- 2개의 Bidirectional LSTM(hidden size 128)으로 구성된다.
- 입력으로 80개의 계수를 가진 멜-스펙트로그램(50 Hz ~ 12 kHz 범위)을 사용한다.

**3. 결합 및 추론 절차**

- Audio Encoder의 전역 벡터와 Mel-spectrogram Conditioner의 프레임 레벨 출력을 Broadcast-concatenate하고, 이를 반복(repetition)을 통해 샘플 레벨(24 kHz)로 업샘플링하여 PW의 입력으로 사용한다.
- **학습 시**: 타겟 오디오 자체를 참조 오디오로 사용하여 VAE 형태로 학습한다.
- **추론 시**: 참조 오디오를 입력하거나, 미리 생성된 특징 벡터를 사용한다. 특히, $\mu=0, \sigma=1$인 표준 정규 분포의 중심점($e=0$)을 사용했을 때 학습하지 않은 화자에 대해 더 높은 일반화 성능을 보임을 확인하였다.

### 학습 절차

본 모델은 "Teacher-Student" 학습 패러다임을 따른다.

1. **Teacher**: Universal WaveNet(자기회귀 모델)을 먼저 학습시킨다.
2. **Student**: 학습된 Teacher로부터 지식을 전수받는 Parallel WaveNet을 학습시킨다. 이때 Student는 Teacher가 이미 학습한 컨디셔닝 네트워크(Audio Encoder 포함)를 그대로 재사용하며, 이는 처음부터 학습시키는 것보다 성능이 우수함을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 78명의 화자, 28개 언어, 16개 스타일이 포함된 대규모 내부 데이터셋을 사용하였다.
- **평가 지표**: MUSHRA(MUltiple Stimuli with Hidden Reference and Anchor) 테스트를 통해 자연스러움(naturalness)과 오디오 품질을 0~100점으로 측정하였으며, 원본 녹음 대비 비율인 Relative MUSHRA(Rel.)를 계산하였다.
- **비교 대상**: Speaker-dependent PW(SDPW), Universal WaveRNN(UWRNN), Parallel WaveGAN(PWGAN), WaveGlow(WGlow).

### 주요 결과

**1. 화자-종속 모델(SDPW)과의 비교**

- 전체적으로 UPW의 Relative MUSHRA는 $84.24\%$로, SDPW의 $83.12\%$보다 통계적으로 유의미하게 높았다.
- 특히 노래(Singing)와 같이 표현력이 강한 스타일에서 UPW가 SDPW보다 월등한 성능 향상을 보였다. 이는 범용 모델이 다양한 화자의 데이터를 통해 더 풍부한 표현력을 학습했음을 시사한다.

**2. 타 범용 보코더와의 비교 (내부 데이터셋)**

- UPW는 전체 평균 Relative MUSHRA $94.82\%$를 기록하며 WGlow, PWGAN, UWRNN보다 우수한 성능을 보였다.
- 비자기회귀 모델인 WGlow와 PWGAN보다 일관되게 높은 품질을 보였으며, 자기회귀 모델인 UWRNN과 비교했을 때 일부 사례에서 품질은 비슷하거나 높으면서 추론 속도는 수십 배 더 빨랐다.

**3. 외부 데이터셋에 대한 강건성 테스트**

- LibriTTS-clean(고품질), LibriTTS-other(중간 품질), Common Voice(저품질) 데이터셋에서 평가하였다.
- 스튜디오 품질의 데이터(LibriTTS-clean)에서는 Relative MUSHRA $98.77\%$로 압도적인 성능을 보였으며, 저품질 데이터인 Common Voice에서도 WGlow와 대등한 수준의 강건함을 유지하였다.

## 🧠 Insights & Discussion

본 연구는 비자기회귀 보코더가 적절한 컨디셔닝 네트워크를 통해 범용성을 확보할 수 있음을 증명하였다. 특히 Audio Encoder를 통해 추출된 전역 특징 벡터가 화자의 정체성을 효과적으로 캡처하며, 추론 시 중심점($e=0$)을 사용하는 단순한 방법만으로도 보지 못한 화자에 대해 훌륭한 일반화 성능을 낼 수 있다는 점이 인상적이다.

또한, 단일 화자 모델(SDPW)보다 범용 모델(UPW)의 성능이 더 좋게 나타난 점은, 다수의 화자 데이터를 통해 학습된 모델이 음성 신호의 공통적인 구조와 복잡한 위상(phase) 특성을 더 잘 이해하게 되어, 결과적으로 개별 화자에게도 더 이득이 된다는 것을 보여준다.

다만, 본 논문에서는 전역 컨디셔닝(global conditioning)만을 다루었으며, 음성의 국소적인 변화를 캡처할 수 있는 local conditioning으로의 확장 가능성을 미래 과제로 남겨두었다. 또한, 겹치는 목소리(overlapping voices)나 비음성 발성(shouts, breath)과 같은 극한 상황에서의 성능에 대해서는 명시적인 분석이 이루어지지 않았다.

## 📌 TL;DR

본 논문은 Parallel WaveNet에 참조 오디오를 인코딩하는 **Audio Encoder**를 추가하여, 실시간 추론이 가능하면서도 다양한 화자와 언어에 적용 가능한 **Universal Parallel WaveNet(UPW)**을 제안한다. 실험 결과, UPW는 기존의 화자-종속 모델 및 다른 범용 보코더들보다 뛰어난 음질과 일반화 성능을 보였으며, 특히 표현력이 강한 음성 생성에서 강점을 나타낸다. 이 연구는 고품질의 다국어/다화자 TTS 시스템을 효율적으로 구축하는 데 핵심적인 역할을 할 수 있을 것으로 기대된다.
