# Robust Dual-Modal Speech Keyword Spotting for XR Headsets

Zhuojiang Cai, Yuhan Ma, and Feng Lu (2024)

## 🧩 Problem to Solve

본 논문은 Extended Reality (XR) 헤드셋에서 사용되는 기존의 Vocal Speech Keyword Spotting (KWS) 시스템이 가진 세 가지 주요 한계점을 해결하고자 한다. 첫째, 거리의 소음이나 대중교통과 같은 소음이 심한 환경에서는 키워드 인식 정확도가 현저히 떨어진다. 둘째, 타인이 작업 중이거나 휴식 중인 상황, 혹은 프라이버시 보호가 필요한 상황에서는 사용자가 음성으로 명령을 내릴 수 없어 시스템 활용이 불가능하다. 셋째, 주변에 다른 사람이 말하고 있을 때 시스템이 이를 사용자의 명령으로 오인하여 잘못 활성화되는 False Triggering 문제가 발생한다.

연구의 목표는 음성(Vocal) 정보와 입술 움직임(Lip movement) 정보를 결합한 Dual-modal KWS 시스템을 구축하여, 소음 환경에서의 강건성을 높이고, 무음 상태에서의 상호작용을 가능하게 하며, 주변 소음으로 인한 오작동을 줄이는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 음성 데이터와 초음파 에코(Ultrasonic echo) 데이터를 동시에 활용하는 Vocal-Echoic Dual-modal 시스템을 설계하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Vocal-Echoic Dual-modal KWS 시스템 제안**: 동일한 마이크로폰을 통해 저주파의 음성 신호와 고주파의 초음파 반사 신호를 동시에 획득하여 융합함으로써, 다양한 환경에서 강건한 키워드 스포팅을 가능하게 하였다.
2. **경량화된 Echoic 모델 설계**: 기존의 무거운 ResNet-18 모델을 기반으로 Ablation study를 수행하여, 성능 저하를 최소화하면서 파라미터 수를 100배 이상 줄인 경량 CNN 모델(ResNet-18-1/4-DS)을 도출하였다.
3. **다양한 시나리오 검증**: 소음 환경, 무음 상황, 주변 화자 간섭 상황 등 실제 XR 사용 환경에서 Dual-modal 시스템이 단일 모달 시스템보다 월등한 성능을 보임을 입증하였다.

## 📎 Related Works

### Vocal Speech Keyword Spotting

기존의 KWS는 ASR(Automatic Speech Recognition) 기반, HMM(Hidden Markov Models) 기반, 그리고 최근의 Deep KWS(CNN, RNN 기반)로 발전해 왔다. 특히 CNN 기반의 TC-ResNet, DC-ResNet 및 Broadcast Residual Learning 등의 구조가 제안되어 연산 효율성과 성능을 높였으나, 여전히 심한 소음 환경에서는 성능이 급격히 저하되는 한계가 있다.

### Silent Speech Interface (SSI)

무음 상태에서 말을 인식하는 SSI 연구는 크게 접촉식(EMG, 자력 센서, 스트레인 센서 등)과 비접촉식(카메라, 초음파 등)으로 나뉜다. 접촉식은 착용감이 불편하며, 카메라는 전력 소모와 프라이버시 문제가 있다. 초음파 기반의 EchoSpeech는 FMCW(Frequency-Modulated Continuous Wave)를 사용하여 낮은 전력으로 무음 인식을 구현하였으나, 저주파 음성 정보(Vocal modality)를 함께 활용하여 인식 범위를 확장하려는 시도는 부족했다.

## 🛠️ Methodology

### 전체 시스템 구조

전체 시스템은 입력 오디오 신호를 밴드패스 필터(Bandpass Filter)를 통해 Vocal modality($< 10\text{kHz}$)와 Echoic modality($> 17\text{kHz}$)로 분리한 후, 각각의 전용 파이프라인을 거쳐 최종적으로 융합(Fusion) 모듈에서 키워드를 예측하는 구조를 가진다.

### 주요 구성 요소 및 처리 절차

**1. Vocal Modal KWS**

- **전처리**: 10kHz 컷오프 저역 통과 필터를 적용한 후, Pre-emphasis, Framing, Windowing, FFT, Mel-frequency warping, Log scaling, DCT 과정을 거쳐 MFCC(Mel-frequency cepstral coefficients) 특징을 추출한다.
- **모델**: 추출된 MFCC는 Broadcast Residual 기반의 CNN 아키텍처에 입력되어 예측 벡터를 생성한다.

**2. Echoic Modal KWS**

- **신호 전송**: 스피커에서 두 가지 주파수 대역(17-20kHz, 20.5-23.5kHz)의 FMCW  chirp 신호를 방출한다.
- **에코 프로필 생성**: 마이크로폰으로 수신된 반사 신호와 전송 신호 간의 상호 상관(Cross-correlation)을 계산하여, 시간에 따른 입술 근처 피부의 거리 변화를 나타내는 Echo Profile을 생성한다. 이후 시간축으로 차분을 수행하여 Differential Echo Profile을 얻는다.
- **모델**: Differential Echo Profile을 입력으로 하여 경량화된 CNN(ResNet-18-1/4-DS)이 키워드를 예측한다.

**3. 융합 전략 (Fusion Strategies)**
본 논문은 두 가지 융합 방법을 제안한다.

**(1) Reliability-based Fusion**
각 모달리티 예측 결과의 신뢰도를 계산하여 가중치를 부여하는 방식이다. 신뢰도 지표로 $N$-best log-likelihood difference ($L_{m,t}$)와 dispersion ($D_{m,t}$)을 사용한다.

$$L_{m,t} = \frac{1}{N-1} \sum_{n=2}^{N} \log \frac{P(o_{m,t}|c_{m,t,1})}{P(o_{m,t}|c_{m,t,n})}$$
$$D_{m,t} = \frac{2}{N(N-1)} \sum_{n=1}^{N} \sum_{n'=n+1}^{N} \log \frac{P(o_{m,t}|c_{m,t,n})}{P(o_{m,t}|c_{m,t,n'})}$$

여기서 $P(o_{m,t}|c_{m,t,n})$은 클래스 $c$가 주어졌을 때 결과 $o$가 관찰될 확률이다. 계산된 지표가 특정 임계값을 넘어야 신뢰할 수 있는 것으로 간주하며, 두 모달리티가 모두 신뢰 가능할 때는 융합 지수 $\lambda_{v,t}$를 통해 선형 결합한다.

**(2) MLP-based Fusion**
수동으로 설계한 특징 대신 신경망을 이용하는 방식이다. Vocal과 Echoic 파이프라인에서 나온 예측 벡터를 단순히 연결(Concatenate)하여 MLP(Multi-Layer Perceptron) 모델의 입력으로 넣고, 최종 키워드 확률을 출력한다.

$$P(o_{f,t}|c) = \text{MLP}([P(o_{v,t}|c), P(o_{e,t}|c)])$$

## 📊 Results

### 실험 설정

- **데이터셋**: 15명의 참가자를 통해 수집한 자체 데이터와 Google Speech Commands 데이터셋을 혼합 사용하여 학습하였다.
- **평가 지표**: Word Error Rate (WER)를 사용하여 성능을 측정하였다.
- **하드웨어**: Microsoft HoloLens 2에 커스텀 ESP32 보드, 스피커 2개, 마이크로폰 2개를 장착하여 구현하였다.

### 주요 결과

**1. Echoic 모델 경량화 (Experiment 1)**
ResNet-18을 베이스라인으로 하여 너비를 줄이고 Depthwise Separable Convolution을 적용한 결과, ResNet-18-1/4-DS 모델은 WER 증가를 0.91%로 최소화하면서 파라미터 수를 100배 이상 줄여 헤드셋 탑재 가능성을 입증하였다.

**2. 소음 환경 성능 (Experiment 2)**
SNR (Signal-to-Noise Ratio) $-10\text{dB}$에서 $10\text{dB}$까지의 환경에서 실험한 결과, Dual-modal 시스템(특히 MLP fusion)이 단일 모달 시스템보다 일관되게 낮은 WER을 기록하였다. 특히 강한 소음($-10\text{dB}$)에서 MLP fusion은 Vocal 및 Echoic 단독 시스템 대비 WER을 각각 15.68%, 16.57% 감소시켰다.

**3. 무음 상호작용 (Experiment 3)**
Vocal 신호를 제거한 환경에서 실험한 결과, Vocal-only 시스템은 완전히 실패(모든 신호가 삭제 처리됨)한 반면, Dual-modal 시스템은 Echoic modality의 성능을 그대로 유지하며 정확하게 키워드를 인식하였다.

**4. 주변 화자 간섭 (Experiment 4)**
타인의 음성을 중첩시킨 환경에서 Vocal-only 시스템은 높은 Substitution 및 Insertion 에러를 보였으나, Dual-modal 시스템은 Echoic modality의 정보를 활용해 False Triggering을 획기적으로 줄였다.

## 🧠 Insights & Discussion

### 강점

본 연구의 Dual-modal 접근 방식은 기존 Vocal KWS가 가진 소음 및 간섭 문제를 해결함과 동시에, 무음 상태의 인터페이스(SSI) 기능까지 통합하였다는 점에서 매우 실용적이다. 특히 하드웨어적으로 기성품 마이크와 스피커만을 사용하여 추가 비용을 최소화하면서 두 가지 모달리티를 동시에 획득한 점이 돋보인다.

### 한계 및 논의사항

1. **물리적 활동의 영향**: 걷거나 머리를 흔드는 동작은 특히 Echoic modality의 성능을 크게 저하시킨다(WER 약 33~50% 증가). 이는 초음파 반사 경로가 물리적 움직임에 민감하기 때문이며, 이를 보정하기 위한 추가적인 연구가 필요하다.
2. **초음파 간섭**: 동일한 기기에서 방출되는 초음파가 주변에 있을 경우, 정지 상태에서는 차분(Differential) 처리로 완화되지만, 이동하는 초음파원(Moving source)이 있을 때는 성능이 저하되는 현상이 발견되었다.
3. **하드웨어 의존성**: 스피커와 마이크의 위치가 입술과의 거리 및 각도에 따라 성능이 결정되므로, 다양한 얼굴 형태에 대응하는 최적의 배치 설계가 필수적이다.

## 📌 TL;DR

본 논문은 XR 헤드셋을 위한 **Vocal(음성)과 Echoic(초음파 에코) 융합 기반의 Dual-modal Keyword Spotting 시스템**을 제안한다. 제안된 시스템은 소음이 심한 환경에서도 높은 강건성을 유지하며, 주변에 다른 사람이 말하고 있을 때의 오작동을 방지하고, 사용자가 말을 할 수 없는 무음 상황에서도 키워드 인식을 가능하게 한다. 특히 경량화된 CNN 모델과 효율적인 융합 전략(Reliability-based, MLP-based)을 통해 실제 웨어러블 기기 적용 가능성을 입증하였으며, 이는 향후 XR 기기의 자연스럽고 유연한 사용자 인터랙션을 구현하는 데 중요한 역할을 할 것으로 기대된다.
