# ENHANCING SYNTHETIC TRAINING DATA FOR SPEECH COMMANDS: FROM ASR-BASED FILTERING TO DOMAIN ADAPTATION IN SSL LATENT SPACE

Sebastião Quintas, Isabelle Ferrané, Thomas Pellegrini (2024)

## 🧩 Problem to Solve

최근 Text-to-Speech (TTS) 시스템을 이용한 데이터 증강(Data Augmentation)이 자동 음성 인식(ASR) 및 음성 분류 작업에서 널리 사용되고 있다. 특히 Voice Cloning 기술의 발전으로 적은 양의 오디오 세그먼트만으로도 다양한 화자의 음성을 생성할 수 있게 되었다. 그러나 이러한 시스템은 생성 과정에서 Hallucination(환각 현상)을 일으켜 품질이 낮거나 잘못된 데이터를 생성하는 경향이 있으며, 이는 결과적으로 다운스트림 태스크(Downstream Task)의 성능 저하로 이어진다.

또한, 최신 TTS 시스템으로 생성된 고품질의 합성 음성이라 할지라도 실제 인간의 음성과는 여전히 특성상의 차이(Domain Gap)가 존재한다. 본 논문은 특히 Speech Commands Classification (SCC) 작업에서 합성 데이터만을 사용하여 모델을 학습시킬 때 발생하는 성능 저하 문제를 해결하고자 한다. 연구의 최종 목표는 Hallucination이 제거된 고품질의 합성 데이터를 생성하고, Self-Supervised Learning (SSL) 잠재 공간(Latent Space)에서 도메인 적응(Domain Adaptation)을 수행하여 실제 데이터와의 간극을 좁히는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여와 설계 아이디어는 다음과 같다.

1. **ASR 기반 필터링 방법론 제안**: 두 개의 서로 다른 ASR 시스템을 사용하여 합성된 음성이 의도한 단어와 정확히 일치하는지 검증함으로써, Hallucination이 없는 깨끗한 합성 데이터셋을 구축한다.
2. **SSL 특성 기반의 도메인 분석**: WavLM과 같은 SSL feature를 사용할 때, 실제 음성과 합성 음성이 잠재 공간에서 명확하게 구분된다는 점을 PCA 시각화를 통해 입증하고, 이것이 성능 차이의 원인임을 제시한다.
3. **CycleGAN을 이용한 잠재 공간 도메인 적응**: 합성 음성의 SSL representation을 실제 음성의 분포로 변환하기 위해 CycleGAN을 도입하여 SCC 성능을 추가적으로 향상시킨다.

## 📎 Related Works

기존 연구들은 TTS를 ASR의 데이터 증강 수단으로 활용하여 저자원 시나리오에서 성능을 높이는 방향으로 진행되어 왔다. Keyword Spotting (KWS) 분야에서는 다중 화자 TTS로 생성한 키워드로 RNN-T 시스템을 미세 조정(Fine-tuning)하거나, 대규모 일반 음성 코퍼스를 이용한 SSL 사전 학습을 통해 실제 데이터 요구량을 줄이는 접근 방식이 시도되었다. 일부 연구에서는 실제 데이터와 합성 데이터를 혼합하여 사용하는 방식을 통해 높은 정확도를 달성하기도 하였다.

하지만 본 논문은 기존 연구들과 달리 **순수하게 합성 데이터만으로 학습시킨 모델**의 성능을 분석하며, 단순히 데이터를 늘리는 것이 아니라 합성 데이터 자체가 가지는 내재적인 품질 문제(Hallucination)와 실제 데이터와의 분포 차이(Domain Mismatch)를 직접적으로 해결하려는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 합성 음성 생성 및 ASR 필터링

합성 데이터 생성을 위해 XTTS v2 시스템과 Common Voice (CV) 데이터셋의 Voice Cloning 기능을 사용하였다. CV 데이터셋에서 무작위로 샘플링된 수만 명의 화자 정보를 이용해 다양한 목소리의 음성 명령어를 생성하였다.

이때 생성 과정에서 발생하는 Hallucination을 제거하기 위해 **ASR-based filtering** 루프를 적용한다. 구체적으로, 생성된 오디오를 두 가지 서로 다른 ASR 모델(Fast-Conformer Transducer 및 Jasper CTC)에 입력하고, 두 모델의 전사(Transcription) 결과가 모두 목표 단어와 정확히 일치하는 경우에만 해당 데이터를 유지한다.

### 2. SCC 모델 및 SSL representation

본 연구에서는 두 가지 모델 구조를 비교 분석한다.

- **MatchboxNet**: 1D Time-Channel Separable Convolutional 구조의 경량 모델로, MFCC 특성을 입력으로 사용한다.
- **WavLM-Base-Plus**: SSL 모델인 WavLM을 통해 768차원의 특성 벡터를 추출한다. 시간축에 대해 통계적 풀링(Statistic Pooling, 평균 및 표준편차 계산)을 수행하여 단일 벡터로 변환한 뒤, 단순한 Linear Layer 분류기를 통해 클래스를 판별한다.

### 3. CycleGAN 기반 도메인 적응

WavLM의 잠재 공간에서 실제 음성과 합성 음성의 분포가 분리되어 있다는 점을 해결하기 위해 CycleGAN을 사용한다. 이 모델은 합성 도메인($S$)과 실제 도메($R$) 사이의 상호 변환을 학습한다.

- **구조**: 두 개의 생성자 $G_A$ (Synth $\to$ Real), $G_B$ (Real $\to$ Synth)와 두 개의 판별자 $D_A, D_B$로 구성된다. 생성자는 3개의 Linear Layer와 ReLU, Tanh 활성화 함수를 사용한다.
- **손실 함수**: 전체 손실 함수 $L_{CC}$는 다음과 같이 정의된다.
$$L_{CC} = L_{A}^{gan} + L_{B}^{gan} + \lambda_c(L_{A}^c + L_{B}^c) + \lambda_{id}(L_{A}^{id} + L_{B}^{id})$$

여기서 각 항의 의미는 다음과 같다.

- $L^{gan}$: 판별자를 속이기 위한 적대적 손실(MSE Loss 사용)이다.
- $L^c$: 입력 데이터를 다른 도메인으로 변환했다가 다시 원래 도메인으로 복원했을 때의 일관성을 유지하는 Cycle-consistency loss (L1 Loss)이다.
- $L^{id}$: 입력이 이미 타겟 도메인인 경우, 생성자가 입력을 거의 그대로 유지하도록 하는 Identity loss (L1 Loss)이다.

학습 후에는 $G_A$만을 사용하여 합성 음성의 SSL feature를 실제 음성 분포로 변환한 뒤 분류기에 입력한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands (v2) 데이터셋의 10가지 기본 명령어 사용 (나머지는 unknown 클래스로 처리, silence 클래스는 제외).
- **지표**: Accuracy (%)

### 주요 결과

1. **MatchboxNet 결과**:
    - 실제 데이터 학습 시 약 $98.4\%$의 정확도를 보였으나, 필터링 없는 합성 데이터(Synth)는 $89 \sim 90\%$ 수준에 그쳤다.
    - ASR 필터링을 적용한 데이터(Synth (F))를 사용했을 때 정확도가 $92.6\%$까지 상승하여, Hallucination 제거의 중요성이 입증되었다.

2. **WavLM (SSL) 결과**:
    - 필터링된 합성 데이터(Synth (F))만으로 학습해도 $96.1\%$의 높은 정확도를 달성하여 MatchboxNet보다 훨씬 뛰어난 성능을 보였다.
    - 특이사항으로, 필터링 없는 데이터(Synth) 사용 시 정확도가 $83\%$까지 급락했는데, 이는 SSL feature가 MFCC보다 Hallucination에 더 민감하게 반응함을 시사한다.

3. **CycleGAN 적용 결과**:
    - CycleGAN을 통해 도메인 적응을 수행한 경우, 정확도가 $96.1\% \to 96.5\%$로 소폭 상승하였다. 비록 상승 폭은 작지만, 합성 데이터만으로 실제 데이터 학습 모델($98.0\%$)과의 간극을 유의미하게 좁혔음을 보여준다.

## 🧠 Insights & Discussion

본 연구의 PCA 분석 결과, MFCC 공간보다 SSL 잠재 공간에서 실제 음성과 합성 음성의 분리가 훨씬 더 명확하게 나타났다. 이는 WavLM과 같은 SSL 모델이 음성의 매우 세밀한 특성을 포착하며, 합성 음성이 실제 음성이 가진 다양성과 복잡성을 완전히 재현하지 못하고 있음을 의미한다.

CycleGAN을 통한 도메인 적응이 성능 향상을 가져온 것은 합성 데이터의 분포를 실제 데이터 쪽으로 이동시키는 것이 유효함을 입증한다. 다만, 성능 향상 폭이 제한적이라는 점은 단순히 분포를 맞추는 것 이상의 정보(예: 화자의 다양성, 환경 소음 등)가 필요함을 시사한다. 저자들은 향후 연구로 SSL feature 중 어떤 차원이 합성/실제 음성을 구분 짓는지 분석하여 해당 차원을 제거하거나, Flow Matching과 같은 최신 생성 모델을 이용한 도메인 적응을 시도할 필요가 있다고 논의한다.

## 📌 TL;DR

본 논문은 합성 음성 데이터를 이용한 음성 명령어 분류 성능을 높이기 위해 **(1) ASR 기반의 엄격한 필터링으로 Hallucination을 제거**하고, **(2) WavLM SSL feature를 활용**하며, **(3) CycleGAN으로 합성-실제 데이터 간의 잠재 공간 분포 차이를 보정**하는 방법론을 제안하였다. 이를 통해 순수 합성 데이터만으로도 실제 데이터 학습 모델에 근접한 성능($96.5\%$)을 달성하였으며, 이는 데이터 수집이 어려운 저자원 환경의 음성 인식 시스템 구축에 중요한 실마리를 제공한다.
