# Joint Training of Speech Enhancement and Self-supervised Model for Noise-robust ASR

Qiu-Shi Zhu, Jie Zhang, Zi-Qiang Zhang, Li-Rong Dai (2022)

## 🧩 Problem to Solve

자동 음성 인식(Automatic Speech Recognition, ASR) 시스템은 깨끗한 음성 환경에서는 높은 성능을 보이지만, 배경 소음이 심하거나 신호 대 잡음비(Signal-to-Noise Ratio, SNR)가 낮은 환경에서는 성능이 급격히 저하되는 문제가 있다. 이를 해결하기 위해 일반적으로 음성 향상(Speech Enhancement, SE) 모듈을 전처리 단계(front-end)로 사용하지만, SE 과정에서 발생하는 음성 왜곡(speech distortion)이 오히려 ASR의 단어 오류율(Word Error Rate, WER)을 높이는 결과가 초래되기도 한다.

최근 자기지도 학습(Self-supervised Learning, SSL) 기반의 사전 학습(pre-training)이 대규모의 라벨 없는 노이즈 데이터를 활용하여 ASR의 노이즈 강건성(noise robustness)을 높이는 데 효과적임이 밝혀졌다. 그러나 SE와 SSL 사전 학습을 최적으로 결합하여 SE의 왜곡 문제를 줄이면서 강건성을 극대화하는 방법론에 대해서는 연구가 미비한 상태이다. 따라서 본 논문의 목표는 SE 모듈과 SSL 모델을 공동 사전 학습(joint pre-training)하는 프레임워크를 제안하여 노이즈 환경에서의 ASR 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SE 모듈과 SSL 모델을 분리하여 학습시키는 대신, 하나의 통합된 프레임워크 내에서 공동으로 사전 학습시키는 것이다. 주요 기여 사항은 다음과 같다.

1. **공동 사전 학습 접근법 제안**: DEMUCS 기반의 시역(time-domain) SE 모듈과 Enhanced wav2vec 2.0(EW2) 기반의 SSL 모델을 함께 학습시킨다. 특히, 사전 학습 단계에서 양자화된 깨끗한 음성(quantized clean speech)을 타겟으로 설정하여, 모델이 노이즈 섞인 특징으로부터 깨끗한 음성 표현을 학습하도록 유도한다.
2. **Dual-attention Fusion 모듈 설계**: SE를 통해 얻은 향상된 특징($Z^{en}$)과 원래의 노이즈 특징($Z^{noisy}$)을 융합하는 이중 주의 집중(dual-attention) 메커니즘을 제안한다. 이는 SE 모듈 단독 사용 시 발생하는 정보 손실과 음성 왜곡을 상쇄하여 ASR 성능을 보완한다.
3. **범용적 프레임워크 입증**: 제안된 모델은 구성 요소(noisy, clean, enhanced branch)의 선택에 따라 기존의 EW2나 SE 기반 ASR 모델로 환원될 수 있는 일반화된 구조를 가진다. 또한, 합성 데이터뿐만 아니라 실제 노이즈 데이터셋인 CHiME-4에서 그 효과를 검증하였다.

## 📎 Related Works

논문에서는 노이즈 강건 ASR을 위한 기존 접근 방식을 두 가지 범주로 구분하여 설명한다.

1. **SE 모듈 기반 접근 방식**: SE 모듈을 ASR 전단에 배치하는 구조이다. SE와 ASR을 개별적으로 학습시킬 경우, SE의 목표가 주로 음질 지표(MSE, SNR 등)에 맞춰져 있어 ASR의 가독성(intelligibility)을 해치는 왜곡이 발생한다. 이를 해결하기 위해 공동 학습(joint training) 연구가 진행되었으나, 여전히 향상된 특징만을 입력으로 사용할 경우 왜곡 문제에서 완전히 자유롭지 못하며, 노이즈 특징과 향상된 특징을 융합하는 방식이 대안으로 제시되었다.
2. **자기지도 사전 학습(SSL) 기반 접근 방식**: wav2vec 2.0, HuBERT, WavLM 등 대규모 라벨 없는 데이터를 이용해 표현 학습(representation learning)을 수행하는 방식이다. 이러한 방법들은 SE 기반 방식보다 음성 왜곡 문제가 적고 데이터 효율성이 높다. 최근에는 노이즈 데이터에 특화된 사전 학습이나 일관성 학습(consistency learning)을 통해 강건성을 높이려는 시도가 있었다.

본 논문은 이 두 가지 접근 방식의 장점을 결합하되, 특히 SE의 왜곡 문제를 SSL의 문맥 표현 학습과 Dual-attention Fusion을 통해 해결하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Enhanced wav2vec 2.0 (EW2) 모델

EW2는 기본 wav2vec 2.0 구조(Feature Encoder $f$, Transformer Encoder $g$, Vector Quantization $VQ$)를 기반으로 한다.

- **특징 추출**: 노이즈 음성 $X^{noisy}$와 깨끗한 음성 $X^{clean}$에서 각각 $Z^{noisy} = f(X^{noisy})$와 $Z^{clean} = f(X^{clean})$를 추출한다.
- **양자화 타겟**: $Z^{clean}$은 VQ 모듈을 통해 이산적인 코드 $q^{clean} = VQ(Z^{clean})$으로 변환되어 사전 학습의 정답(target) 역할을 한다.
- **손실 함수**: 전체 손실은 다음과 같이 정의된다.
$$L = L_m + \alpha L_d + \beta L_f + \gamma L_c$$
여기서 $L_m$은 대조 손실(contrastive loss)로, 마스킹된 노이즈 특징에서 예측된 문맥 표현 $C^{noisy}$와 실제 $q^{clean}$ 사이의 유사도를 극대화한다. $L_d$는 코드북 사용량을 늘리는 다양성 손실, $L_f$는 $\ell_2$ 패널티, $L_c$는 노이즈 특징과 깨끗한 특징 사이의 유클리드 거리를 줄이는 일관성 손실(consistency loss)이다.
$$L_c = \lVert Z^{noisy}_t - Z^{clean}_t \rVert^2$$

### 2. Speech Enhancement (SE) 모듈

본 연구에서는 시역 기반의 **DEMUCS** 모델을 사용한다. DEMUCS는 인코더(CNN), LSTM, 디코더(Transposed CNN)로 구성되며, 스킵 연결(skip connection)을 포함한다. 손실 함수 $L^{SE}$는 시역의 $\ell_1$ 손실과 다중 해상도 STFT(Short-Time Fourier Transform) 손실의 합으로 구성된다.
$$L^{SE} = \frac{1}{T} \left( \lVert X - X^{en} \rVert_1 + \sum_{i=1}^M L_{stft}^{(i)}(X, X^{en}) \right)$$
STFT 손실은 스펙트럼 수렴(spectral convergence) 손실과 크기(magnitude) 손실을 포함하여 주파수 도메인의 세밀한 정보를 복원한다.

### 3. Dual-attention Fusion 모듈

향상된 특징 $Z^{en}$과 노이즈 특징 $Z^{noisy}$의 상호 보완적 정보를 학습하기 위해 제안되었다. 두 개의 브랜치로 구성되며, 각각 다른 입력을 쿼리(Query)로 사용하여 가중 합을 계산한다.
$$Z^{fusion} = \text{Linear}(\text{Multihead}(Z^{en}, Z^{noisy}, Z^{noisy})) + \text{Linear}(\text{Multihead}(Z^{noisy}, Z^{en}, Z^{en}))$$
이 구조를 통해 SE 과정에서 손실된 정보를 $Z^{noisy}$에서 보충하고, 노이즈 성분을 $Z^{en}$을 통해 억제함으로써 최적의 융합 특징 $Z^{fusion}$을 얻는다.

### 4. 공동 사전 학습 및 추론 절차

- **전체 파이프라인**: $X^{noisy}$는 SE 모듈을 통해 $X^{en}$이 되고, $X^{noisy}$와 $X^{en}$은 각각 Feature Encoder를 통해 $Z^{noisy}$와 $Z^{en}$이 된다. 이 둘은 Dual-attention Fusion을 통해 $Z^{fusion}$으로 합쳐진 후 Transformer Encoder를 거쳐 $C^{fusion}$이 된다.
- **전체 손실 함수**:
$$L_{total} = L_{contrastive} + \xi L^{SE}$$
이때 $L_{contrastive}$는 $C^{fusion}$과 $q^{clean}$ 사이의 대조 학습을 수행하며, 일관성 손실 $L_{cs}$는 $Z^{noisy}$와 $Z^{en}$ 모두가 $Z^{clean}$과 가까워지도록 강제한다.
- **미세 조정(Fine-tuning)**: 사전 학습이 끝나면 깨끗한 음성 브랜치를 제거하고, Transformer Encoder 출력단에 Linear layer를 추가하여 CTC(Connectionist Temporal Classification) 손실 함수를 통해 소량의 라벨링된 데이터로 학습시킨다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: LibriSpeech(합성 노이즈), CHiME-4(실제 노이즈).
- **지표**: Word Error Rate (WER).
- **비교 대상**: Baseline(DeepSpeech2), DEMUCS 단독, wav2vec 2.0, EW2, SEW2 등.

### 2. 주요 결과 및 분석

- **EW2의 우수성**: 깨끗한 음성을 타겟으로 사용하는 EW2가 일반 wav2vec 2.0보다 노이즈 환경에서 훨씬 낮은 WER을 기록하였다. 특히 다양한 노이즈 타입에 대해 더 강건한 모습을 보였다.
- **VQ 및 일관성 손실의 필요성**: 실험 결과, VQ 모듈을 제거하거나 일관성 손실 $L_c$를 제거했을 때 WER이 상승하였다. 이는 저수준 특징(low-level features)이 노이즈에 매우 취약하므로, 이를 깨끗한 특징과 일치시키는 제약 조건이 필수적임을 시사한다.
- **공동 학습의 효과 (EW2+SEW2)**:
  - 단순히 SE를 앞단에 붙인 SEW2보다, Dual-attention Fusion을 적용한 EW2+SEW2의 성능이 월등히 좋았다.
  - 이는 단순 결합(concatenation)보다 제안된 Attention 기반 융합이 SE의 왜곡 문제를 훨씬 효과적으로 완화함을 보여준다.
- **CHiME-4 실제 데이터 결과**: LM(Language Model) 없이도 EW2 대비 약 10%의 상대적 성능 향상을 보였으며, Transformer 기반 LM을 적용했을 때 최상의 성능을 달성하였다.
- **초기화의 중요성**: 랜덤 초기화보다 사전 학습된 모델로 초기화했을 때 손실 함수 표면(loss landscape)이 더 매끄럽고 최적 영역이 넓어짐을 시각적으로 확인하였으며, 이는 SE 모듈 추가 시 발생할 수 있는 최적화 어려움을 줄여준다.

## 🧠 Insights & Discussion

본 연구는 SE의 '음질 향상' 목표와 ASR의 '인식 정확도' 목표 사이의 괴리를 SSL의 표현 학습과 특징 융합 기법으로 해결하려 했다는 점에서 학술적 가치가 크다.

**강점**:

- SE 모듈의 왜곡 문제를 단순히 필터링하는 것이 아니라, SSL의 문맥 정보와 Attention 기반 융합을 통해 '보완'하는 관점으로 접근하였다.
- 수치적 최적화 관점에서 Loss Landscape를 분석하여 사전 학습된 모델 초기화의 이점을 이론적으로 뒷받침하였다.

**한계 및 논의사항**:

- 사전 학습 단계에서 깨끗한 음성-노이즈 음성 쌍(paired data)이 필요하다는 점은 실제 환경에서 데이터 수집의 제약이 될 수 있다.
- 본 논문에서는 MSE 기반의 SE 모듈(DEMUCS)을 사용하였으나, 저자들 또한 언급했듯이 향후 음성 가독성(intelligibility)을 직접 최적화하는 SE 모델을 결합한다면 왜곡 문제를 더욱 획기적으로 줄일 수 있을 것으로 보인다.

## 📌 TL;DR

이 논문은 **음성 향상(SE) 모듈과 자기지도 학습(SSL) 모델을 공동으로 사전 학습**하여 노이즈 환경에서 ASR 성능을 높이는 방법을 제안한다. 특히 **양자화된 깨끗한 음성을 학습 타겟**으로 설정하고, **Dual-attention Fusion**을 통해 SE의 왜곡 문제를 해결함으로써, 합성 및 실제 노이즈 데이터셋 모두에서 기존 방식보다 뛰어난 강건성과 인식 성능을 달성하였다. 이 연구는 향후 SE와 ASR의 통합 최적화 연구에 중요한 기준점을 제시한다.
