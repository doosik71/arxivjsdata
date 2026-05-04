# A Comparison of Speech Data Augmentation Methods Using S3PRL Toolkit

Mina Huh, Ruchira Ray, Corey Karnei (2024)

## 🧩 Problem to Solve

본 연구는 음성 처리 작업에서 딥러닝 모델의 일반화 성능을 높이고 과적합(overfitting) 문제를 해결하기 위해 데이터 증강(Data Augmentation) 기법이 미치는 영향을 분석하고자 한다. 특히 실제 환경에서는 배경 소음이나 말하기 속도의 변화와 같은 변동성이 존재하지만, 많은 최신 모델들이 깨끗한(clean) 데이터셋으로만 학습되어 이러한 실세계 데이터에 대한 강건성(robustness)이 부족하다는 문제가 있다.

따라서 본 논문의 목표는 S3PRL 툴킷을 활용하여 음소 인식(Phoneme Recognition, PR)과 자동 음성 인식(Automatic Speech Recognition, ASR) 작업에서 서로 다른 데이터 증강 전략(SpecAugment, Gaussian Noise, Speed Perturbation)이 HuBERT와 wav2vec 모델의 성능 및 강건성에 어떠한 영향을 주는지를 정량적으로 비교 분석하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 자기지도학습(Self-Supervised Learning, SSL) 기반의 사전 학습 모델인 HuBERT와 wav2vec을 대상으로, 특성 증강(Feature Augmentation)과 데이터 증강(Data Augmentation) 기법이 다운스트림 작업의 성능에 미치는 영향을 체계적으로 실험했다는 점이다. 특히, 단순한 성능 향상을 넘어 특정 증강 기법으로 학습된 모델이 해당 환경의 테스트 데이터셋에서 더 높은 강건성을 보인다는 점을 입증함으로써, 실세계 데이터에 대응하기 위한 학습 전략의 방향성을 제시하였다.

## 📎 Related Works

음성 인식 분야에서는 데이터 부족 문제를 해결하고 일반화 능력을 키우기 위해 다양한 증강 기법이 제안되어 왔다. 기존 연구들은 크게 두 가지 접근 방식을 사용한다. 첫째는 원본 파형(waveform)을 변형하는 데이터 증강(Data Augmentation)이며, 둘째는 추출된 스펙트로그램(spectrogram)을 변형하는 특성 증강(Feature Augmentation)이다. 

SpecAugment는 스펙트로그램 상에서 시간과 주파수 영역을 마스킹하는 대표적인 특성 증강 기법으로 널리 사용되고 있다. 또한, 가우시안 노이즈 추가나 속도 섭동(Speed Perturbation)과 같은 기법들이 강건한 음성 인식 모델을 만들기 위해 제안되었다. 그러나 HuBERT나 wav2vec과 같은 최신 SSL 모델들의 경우, 사전 학습 단계에서는 데이터 증강이 충분히 검토되지 않았으며, 본 논문은 이러한 모델들을 파인튜닝(fine-tuning)하는 단계에서 증강 기법을 적용했을 때의 효과를 탐구함으로써 기존 연구의 공백을 메우고자 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조
본 연구는 S3PRL(Self-Supervised Speech Pre-training and Representation Learning) 툴킷을 사용하여 실험을 수행하였다. 전체 과정은 다음과 같은 단계로 진행된다.
1. **모델 초기화 및 기본 학습**: HuBERT 또는 wav2vec 모델을 무작위 가중치로 초기화하고, 원본 LibriSpeech 데이터셋으로 50,000 스텝 동안 학습시킨다.
2. **파인튜닝(Fine-tuning)**: 위에서 저장된 최적 모델을 기반으로, 서로 다른 증강 데이터셋(Baseline, SpecAugment, Speed Perturbation, Gaussian Noise)에 대해 각각 추가로 50,000 스텝 동안 파인튜닝을 수행한다.
3. **평가**: 학습된 모델들을 원본 테스트 세트와 증강된 테스트 세트에 적용하여 성능을 측정한다.

### 2. 데이터 증강 기법
본 연구에서 사용한 세 가지 증강 기법의 상세 내용은 다음과 같다.

- **SpecAugment**: 로그 멜 스펙트로그램(log-mel spectrogram)을 대상으로 하며, 다음 세 가지 전략을 사용한다.
    - **Frequency Masking**: 연속된 멜 주파수 채널에 마스크를 적용하여 특정 주파수 대역을 제거한다.
    - **Time Masking**: 연속된 시간 단계에 마스크를 적용하여 특정 시간 구간을 제거한다.
    - **Time Warping**: 이미지 중심을 지나는 수평선을 따라 무작위 지점을 특정 거리만큼 왜곡시킨다.
- **Gaussian Noise**: 원본 오디오 신호에 평균이 0인 가우시안 분포의 백색 잡음을 더한다. 신호 대 잡음비(Signal-to-Noise Ratio, $\text{SNR}$)를 10으로 설정하였으며, 이에 따라 표준편차(standard deviation)를 계산하여 노이즈를 생성한다.
- **Speed Perturbation**: 피치(pitch)를 변경하지 않고 오디오의 속도를 조절한다. $\text{torchaudio}$의 SoX 모듈을 사용하여 속도 계수를 $\{0.9, 1.0, 1.1, 0.5, 1.5\}$ 중 무작위로 선택하여 적용한 뒤 원래 샘플링 레이트로 재샘플링한다.

### 3. 학습 및 평가 설정
- **대상 작업**: 
    - 음소 인식(PR): 발화 내용을 음소 단위로 전사하며, 평가 지표로 $\text{Phone Error Rate (PER)}$를 사용한다.
    - 자동 음성 인식(ASR): 발화 내용을 단어 단위로 전사하며, 평가 지표로 $\text{Word Error Rate (WER)}$를 사용한다.
- **최적화**: HuBERT 모델의 경우 Adam 옵티마이저를 사용하며, PR 작업은 학습률 $0.01$, ASR 작업은 학습률 $0.0001$을 적용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: LibriSpeech corpus (1,000시간, 16kHz).
- **비교 모델**: wav2vec, HuBERT.
- **평가 지표**: $\text{PER}$ (for PR), $\text{WER}$ (for ASR).

### 2. 주요 결과
실험 결과는 다음과 같은 특징을 보인다.

- **SpecAugment의 효과**: SpecAugment로 학습한 모델은 원본 테스트 세트에서 성능이 약간 향상되거나 유지되는 경향을 보였다. 이는 특성 증강이 데이터의 전반적인 분포를 크게 변화시키지 않으면서 모델의 일반화 능력을 돕기 때문으로 분석된다.
- **데이터 증강의 강건성**: Gaussian Noise와 Speed Perturbation으로 학습한 모델은 원본 테스트 세트에서는 오히려 성능이 하락하였다. 그러나 해당 노이즈가 추가된 증강 테스트 세트에서는 압도적으로 낮은 에러율을 기록하였다.
    - **PR 작업**: $\text{HuBERT-Gaussian-Noise}$ 모델이 가우시안 노이즈 테스트 세트에서 가장 낮은 $\text{PER}$ ($13.10\%$)를 기록하였다.
    - **ASR 작업**: $\text{HuBERT-Speed-Perturbation}$ 모델이 속도 섭동 테스트 세트에서 가장 낮은 $\text{WER}$ ($21.63\%$)를 기록하였다.

## 🧠 Insights & Discussion

본 연구는 데이터 증강이 모델의 강건성과 일반화 성능 사이에서 서로 다른 영향을 미친다는 점을 시사한다. 

첫째, **분포 이동(Distribution Shift)** 문제이다. 가우시안 노이즈나 속도 섭동과 같은 데이터 증강 기법은 학습 데이터의 분포를 원본 데이터로부터 멀어지게 만든다. 이로 인해 깨끗한 데이터에 대해서는 성능이 저하되지만, 유사한 환경의 노이즈 데이터에 대해서는 매우 강력한 저항력을 갖게 된다.

둘째, **특성 증강의 이점**이다. SpecAugment는 오디오의 물리적 특성을 유지하면서 모델이 특정 시간-주파수 특징에 과도하게 의존하지 않도록 강제한다. 결과적으로 원본 데이터와 증강 데이터 모두에서 일관된 성능 향상을 보이며, 이는 데이터 증강이 반드시 분포를 변화시켜야 하는 것은 아님을 보여준다.

본 연구의 한계점으로는 다양한 공개 데이터셋이나 실제 야생(in-the-wild) 데이터에 대한 테스트가 부족하다는 점이 있다. 또한, 단일 증강 기법만을 적용하였으므로, 여러 기법을 혼합하여 적용했을 때의 시너지 효과에 대한 분석이 이루어지지 않았다.

## 📌 TL;DR

본 논문은 S3PRL 툴킷을 통해 HuBERT와 wav2vec 모델에 SpecAugment, Gaussian Noise, Speed Perturbation을 적용하여 PR 및 ASR 작업의 성능을 분석하였다. 실험 결과, SpecAugment는 전반적인 일반화 성능을 소폭 향상시키는 반면, Gaussian Noise와 Speed Perturbation은 원본 성능을 일부 희생하는 대신 해당 환경의 데이터에 대한 강건성을 극대화하는 것으로 나타났다. 이 연구는 향후 실제 환경의 소음과 변동성이 큰 음성 인식 시스템을 구축할 때, 목표 환경에 맞춘 전략적 데이터 증강의 중요성을 입증하였다.