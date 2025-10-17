# SAMPLE EFFICIENT ADAPTIVE TEXT-TO-SPEECH

Yutian Chen, Yannis Assael, Brendan Shillingford, David Budden, Scott Reed, Heiga Zen, Quan Wang, Luis C. Cobo, Andrew Trask, Ben Laurie, Caglar Gulcehre, Aäron van den Oord, Oriol Vinyals, Nando de Freitas

## 🧩 Problem to Solve

기존의 Text-to-Speech (TTS) 모델은 각 화자(speaker)에 대해 수 시간 분량의 고품질 음성 데이터를 필요로 합니다. 이는 새로운 화자에 대한 TTS 시스템을 구축하는 데 엄청난 비용과 노력을 수반합니다. 이 연구는 **소량의 데이터(few-shot)만을 사용하여 새로운 화자의 음성 스타일을 빠르게 학습하고 적응하는 TTS 시스템**을 개발하는 것을 목표로 합니다. 즉, 고정된 파라미터를 가진 모델을 학습하는 대신, 배포 시 소량의 데이터로 새로운 화자에 빠르게 적응할 수 있는 "사전(prior)" TTS 네트워크를 학습하는 것이 핵심 문제입니다.

## ✨ Key Contributions

- **메타 학습(Meta-learning) 기반의 효율적인 적응형 TTS 방법론 제시**: 소량의 데이터만으로 새로운 화자에 적응할 수 있는 WaveNet 기반의 메타 학습 접근 방식을 제안했습니다.
- **세 가지 적응 전략 벤치마킹**:
  - **SEA-EMB**: WaveNet 코어는 고정하고 화자 임베딩($e_s$)만 학습하는 전략.
  - **SEA-ALL**: 초기 화자 임베딩을 최적화한 후 전체 아키텍처를 미세 조정(fine-tuning)하는 전략.
  - **SEA-ENC**: 보조 인코더 네트워크를 훈련하여 새로운 화자의 임베딩을 예측하는 전략.
- **최첨단 성능 달성**: 새로운 화자로부터 단 몇 분 분량의 오디오 데이터만으로도 음성 자연스러움(naturalness)과 음성 유사성(voice similarity) 측면에서 최첨단(state-of-the-art) 결과를 달성했습니다.
- **화자 검증 시스템 혼란**: 생성된 음성 샘플이 최첨단 텍스트 독립형 화자 검증(text-independent speaker verification) 시스템을 혼란시킬 정도로 실제 음성과 유사함을 입증했습니다.

## 📎 Related Works

이 연구는 다음을 포함한 다양한 관련 분야의 선행 연구들을 참조했습니다:

- **Few-shot Learning**: 소량의 데이터로 빠르게 학습하는 모델을 구축하는 연구 (Santoro et al., 2016; Vinyals et al., 2016 등).
- **Meta-learning**: "학습하는 방법"을 학습하는 프레임워크 (Thrun and Pratt, 2012; MAML: Finn et al., 2017a 등).
- **Generative Modeling for Few-shot Learning**: 생성 모델링 분야에서 소량 학습 문제를 다룬 연구들 (Rezende et al., 2016; Reed et al., 2018 등).
- **Neural TTS Models**: WaveNet (van den Oord et al., 2016), Tacotron (Wang et al., 2017), DeepVoice (Arık et al., 2017), Char2Wav (Sotelo et al., 2017), VoiceLoop (Taigman et al., 2018) 등 최신 신경망 기반 TTS 모델들.
- **동시 연구**: 소량 학습 TTS 문제를 다룬 Jia et al. (2018)의 Tacotron 확장, Nachmani et al. (2018)의 VoiceLoop 확장, Arik et al. (2018)의 DeepVoice 3 확장 연구와 비교 및 참조했습니다.

## 🛠️ Methodology

본 연구는 WaveNet 모델을 기반으로 메타 학습 접근 방식을 사용하여 소량의 데이터로 적응형 TTS를 구현합니다.

1. **WaveNet 아키텍처**:

   - WaveNet은 자기회귀(autoregressive) 생성 모델로, 오디오 파형 $x = \{x_1, \dots, x_T\}$의 결합 확률 분포를 조건부 확률의 곱으로 분해합니다:
     $$ p(x|h;w) = \prod*{t=1}^{T} p(x_t|x*{1:t-1}, h;w) $$
        여기서 $h$는 조건부 입력, $w$는 모델 파라미터입니다.
   - 멀티 스피커 TTS를 위해 조건부 입력 $h$는 화자 임베딩 벡터 $e_s$, 언어학적 특징 $l$, 기본 주파수 $f_0$ 값으로 구성됩니다.
   - 화자 임베딩 벡터 $e_s$는 WaveNet 파라미터와 함께 학습되는 임베딩 테이블에서 가져옵니다.
   - 언어학적 특징 $l$과 기본 주파수 $f_0$는 파형보다 낮은 샘플링 주기를 가지므로, 전치 합성곱 네트워크(transposed convolutional network)를 통해 업샘플링됩니다. $f_0$는 화자 독립적으로 평균 0, 단위 분산으로 정규화($\hat{f_0} := (f - E[f_s])/std(f_s)$)됩니다.

2. **메타 학습 3단계**:

   - **훈련(Training)**: 다수의 화자 데이터셋을 사용하여 공유 WaveNet 코어 파라미터 $w$와 각 화자의 임베딩 $e_s$를 공동으로 최적화하여 "사전(prior)" 모델을 학습합니다.
   - **적응(Adaptation)**: 새로운 화자에 대해 소량의 오디오 데이터(데모 데이터)를 사용하여 빠르게 화자별 파라미터를 학습합니다.
   - **추론(Inference)**: 적응된 모델을 사용하여 새로운 텍스트로부터 해당 화자의 음성을 생성합니다.

3. **세 가지 적응 전략**:
   - **SEA-EMB (Sample-Efficient Adaptive TTS - Embedding only)**:
     - 사전 훈련된 WaveNet 모델에서 공유 파라미터 $w$를 고정합니다.
     - 새로운 화자의 데모 데이터($x_{demo}$)로부터 $l_{demo}$와 $f_{0,demo}$ 특징을 추출하고, 새로운 임베딩 벡터 $e_{demo}$를 무작위로 초기화합니다.
     - $w$를 고정한 채 $e_{demo}$만을 최적화하여 데모 파형의 조건부 로그-가능도를 최대화합니다. 이 방법은 저차원 벡터만 최적화하므로 과적합(overfitting)에 덜 취약합니다.
   - **SEA-ALL (Sample-Efficient Adaptive TTS - All parameters)**:
     - SEA-EMB 방법으로 얻은 최적의 $e_{demo}$ 값을 사용하여 초기화합니다.
     - $e_{demo}$와 함께 _모든 모델 파라미터_ $w$를 데모 데이터에 대해 미세 조정합니다.
     - 과적합 방지를 위해 데모 데이터의 10%를 홀드아웃(hold-out)하여 조기 종료(early termination) 기준을 적용합니다.
   - **SEA-ENC (Sample-Efficient Adaptive TTS - Encoder)**:
     - 보조 인코더 네트워크 $e(\cdot)$를 훈련하여 새로운 화자의 데모 오디오 ($x_{demo}$)로부터 직접 화자 임베딩 벡터를 예측합니다.
     - WaveNet 모델과 인코더 네트워크 $e(\cdot)$는 처음부터 함께 훈련됩니다.
     - 적응 시간에 계산 비용이 거의 들지 않는 장점이 있지만, 인코더의 제한된 네트워크 용량으로 인해 편향(bias)을 도입할 수 있습니다. (인코더 네트워크는 사전 훈련된 TI-SV 모델과 1-D CNN 레이어로 구성됨)

## 📊 Results

- **데이터셋**: LibriSpeech (500시간), 독점 음성 코퍼스 (300시간)로 훈련. LibriSpeech 테스트 셋 (39화자, ~5분/화자) 및 VCTK (21화자, ~12분/화자)로 평가.
- **적응 데이터 크기**: 10초, 1분, 5분(LibriSpeech)/10분(VCTK)으로 변화시키며 평가.

1. **음성 자연스러움 (Mean Opinion Score, MOS)**:

   - **SEA-ALL**은 5분 미만의 데이터로 LibriSpeech에서 4.13점, VCTK에서 3.92점을 얻어 최첨단 성능을 달성했습니다. 이는 24시간 분량의 데이터를 사용한 기존 WaveNet (4.21점)에 필적하는 수준입니다.
   - **SEA-ALL**은 모든 경우에서 **SEA-EMB**보다 우수했으며, **SEA-ENC**가 가장 낮은 점수를 기록했습니다.
   - LibriSpeech에서는 더 많은 적응 데이터가 성능 향상에 기여했지만 VCTK에서는 그 효과가 덜했습니다.

2. **음성 유사성 (MOS)**:

   - **SEA-ALL**은 5분 미만의 데이터로 LibriSpeech에서 3.75점, 10분 미만의 데이터로 VCTK에서 3.97점을 얻어 다른 모델들을 크게 앞질렀습니다.
   - **SEA-ALL**의 성능 향상은 적응 데이터 양에 비례했습니다.
   - Jia et al. (2018)의 최첨단 시스템보다 우수한 성능을 보였으나, 인간은 여전히 실제 음성과 생성된 음성의 차이를 구별할 수 있었습니다.

3. **음성 유사성 (화자 검증 시스템 활용)**:
   - **d-vector 시각화**: t-SNE 투영 결과, **SEA-ALL**로 생성된 음성의 d-vector 임베딩이 실제 음성과 상당 부분 겹치는 클러스터를 형성하여, 화자 검증 시스템이 실제 음성과 생성된 음성을 구별하기 어렵다는 것을 시사했습니다.
   - **EER (Equal Error Rate) 기반 화자 식별**:
     - **SEA-ALL**은 가장 낮은 EER (LibriSpeech에서 5분 미만 데이터로 1.85%)을 달성했으며, 충분한 적응 데이터에서는 실제 음성보다도 EER이 낮게 나타나기도 했습니다. 이는 생성된 샘플이 실제 음성보다 화자 임베딩의 중심에 더 가깝게 집중될 수 있기 때문으로 해석됩니다.
     - **SEA-ALL > SEA-EMB > SEA-ENC** 순으로 성능이 우수했습니다.
   - **ROC/AUC 기반 실제-생성 음성 구별**:
     - 화자 검증 시스템을 사용하여 실제 음성과 생성된 음성을 구별하는 적대적 시나리오에서, **SEA-ALL**은 ROC 곡선이 대각선에 가장 가깝게 위치하고 AUC 값이 0.5에 가까워 (VCTK에서 0.56) 검증 시스템을 가장 효과적으로 혼란시켰습니다. 즉, 실제 음성과 구별하기 가장 어려웠습니다.

## 🧠 Insights & Discussion

- **SEA-ALL 방법론의 우수성**: 화자 임베딩을 먼저 최적화한 후 전체 모델을 미세 조정하는 **SEA-ALL** 전략은 새로운 화자에 대해 단 10초의 오디오만으로도 인상적인 성능을 보였으며, 몇 분의 데이터로는 최첨단 자연스러움과 우수한 음성 유사성을 달성했습니다.
- **d-vector 분석의 중요성**: 화자 검증 시스템의 d-vector를 사용한 분석은 생성된 음성이 실제 음성과 매우 유사한 음향 특성을 보이며, 심지어 특정 환경에서는 시스템을 혼란시킬 수 있음을 객관적으로 입증했습니다.
- **데이터 품질의 한계**: 본 연구의 모델은 깨끗하고 고품질의 훈련 및 적응 데이터에 의존합니다. 노이즈가 많은 데이터에서의 소량 학습은 여전히 어려운 문제로 남아있습니다.
- **윤리적 고려사항**: 소량의 데이터만으로도 음성을 합성할 수 있는 기술은 긍정적인 응용(예: 음성 손상 환자의 음성 복원)과 함께 오용(예: 합성 미디어 제작)의 가능성을 높입니다. 연구진은 오용을 방지하고 탐지하기 위한 추가 연구의 필요성을 강조했습니다.

## 📌 TL;DR

이 논문은 소량의 오디오 데이터(단 몇 분)만으로 새로운 화자에 빠르게 적응하는 TTS 시스템을 위한 메타 학습 접근 방식을 제시합니다. WaveNet 기반으로 **화자 임베딩만 미세 조정하는 SEA-EMB**, **전체 모델을 미세 조정하는 SEA-ALL**, 그리고 **인코더 네트워크로 임베딩을 예측하는 SEA-ENC** 세 가지 전략을 탐구합니다. 실험 결과, **SEA-ALL**은 음성 자연스러움과 화자 유사성에서 최첨단 성능을 달성했으며, 생성된 음성은 최첨단 화자 검증 시스템조차 실제 음성과 구별하기 어려울 정도로 사실적입니다.
