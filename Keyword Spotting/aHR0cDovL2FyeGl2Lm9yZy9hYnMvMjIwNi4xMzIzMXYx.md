# QbyE-MLPMixer: Query-by-Example Open-Vocabulary Keyword Spotting using MLPMixer

Jinmiao Huang, Waseem Gharbieh, Qianhui Wan, Han Suk Shim, Chul Lee (2022)

## 🧩 Problem to Solve

전통적인 Keyword Spotting (KWS) 시스템은 "Hey Siri"나 "Alexa"와 같이 사전에 정의된 고정된 키워드만을 인식하도록 학습된다. 그러나 실제 스마트 기기 환경에서는 사용자가 원하는 단어를 직접 설정하여 기기를 제어하는 개인화된 상호작용이 필수적이며, 이를 위해 Open-vocabulary KWS 기술이 요구된다.

Open-vocabulary KWS, 특히 사용자가 직접 키워드를 정의하는 User-defined KWS는 다음과 같은 기술적 난제가 존재한다. 첫째, 기기 탑재를 위한 낮은 지연 시간(low latency)과 작은 메모리 풋프린트(small memory footprint)가 필요하다. 둘째, 학습 데이터 분포에 포함되지 않은 새로운 단어(out-of-distribution utterances)에 대해서도 강건한 인식 성능을 보여야 한다. 본 논문의 목표는 MLPMixer 구조를 QbyE (Query-by-Example) 방식에 적용하여, 효율적이면서도 강건한 Open-vocabulary KWS 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Vision Transformer (ViT)의 Attention 메커니즘을 Multi-Layer Perceptrons (MLPs)로 대체한 MLPMixer 아키텍처를 오디오 도메인의 QbyE KWS 태스크에 적응시키는 것이다.

주요 기여 사항은 다음과 같다.

1. **MLPMixer의 오디오 적응**: MLPMixer를 QbyE KWS 문제에 맞게 단순하면서도 효과적으로 변형하여 제안하였다.
2. **입력 표현의 최적화**: 오디오 데이터의 경우, 기존 MLPMixer의 패치 분할(patching) 방식보다 MFCC 특징을 직접 입력하는 방식이 훨씬 효과적임을 입증하였다.
3. **효율성 및 성능 입증**: 제안 모델이 기존의 RNN 및 CNN 기반 SOTA 모델들보다 더 적은 파라미터 수와 연산량(MACs)을 가지면서도, 특히 소음이 심한 환경(10dB, 6dB)과 Far-field 환경에서 더 우수한 성능을 보임을 확인하였다.
4. **활성화 함수 최적화**: Hardswish 활성화 함수가 GELU 등 다른 함수보다 KWS 태스크에서 더 높은 성능을 제공함을 보였다.

## 📎 Related Works

기존의 KWS 연구는 DNN, CNN, RNN, 그리고 최근의 Transformer 기반 모델들로 발전해 왔으나, 이들은 대량의 타겟 키워드 데이터가 필요하다는 한계가 있다. Open-vocabulary KWS를 해결하기 위한 초기 접근법은 ASR (Automated Speech Recognition) 시스템의 텍스트나 음소(phoneme) 일치 여부를 확인하는 방식이었으나, 이는 연산 비용이 매우 높고 어휘집 외의 단어에 대해 성능이 급격히 저하되는 문제가 있었다.

이를 해결하기 위해 가변 길이의 오디오 신호를 고정 길이의 벡터 공간 임베딩으로 매핑하는 Query By Example (QbyE) 방식이 등장하였다. QbyE 시스템에서는 주로 LSTM이나 GRU-ATTN과 같은 RNN 계열의 인코더를 사용하여 키워드 임베딩을 추출한다. 최근에는 CNN 기반의 MobileNet이나 ViT와 같은 비전 모델을 오디오 태스크에 적용하려는 시도가 있었으나, 본 논문은 Attention 메커니즘 없이 MLP만으로 구성된 MLPMixer를 통해 연산 효율성과 성능의 균형을 맞추고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 시스템은 Encoder-Decoder 구조를 따른다. Encoder는 고차원의 오디오 데이터를 저차원의 임베딩으로 압축하며, Decoder는 학습 단계에서 동일 클래스의 임베딩은 가깝게, 다른 클래스는 멀게 배치하도록 유도하는 역할을 한다. 추론(Inference) 단계에서는 Decoder를 제거하고 Encoder가 생성한 임베딩 간의 거리를 비교하여 키워드 일치 여부를 결정한다.

### 입력 표현 (Input Representation)

입력 데이터로 81차원의 MFCC (Mel-frequency Cepstral Coefficients) 특징을 사용한다. 1초 길이의 오디오에서 12.5ms 간격으로 추출되어, 최종적으로 $81 \times 81$ 크기의 행렬이 생성된다. 이후 시간축에 대해 CMVN (Cepstral Mean and Variance Normalization)을 적용한다. 특이사항으로, 기존 MLPMixer의 패치 분할 방식을 사용하지 않고 MFCC 특징과 시간축 전체를 모델에 직접 입력하여 계산 효율성을 높이고 정보 손실을 방지하였다.

### MLPMixer 아키텍처

제안 모델은 Feature-mixing MLP 블록과 Time-mixing MLP 블록이 교차로 쌓인 구조이다.

1. **Feature-mixing**: 특징 차원($f$)을 은닉 차원($h$)으로 투영했다가 다시 $f$로 복원한다.
   $$U = X + W_2 \sigma(W_1 \text{LayerNorm}(X))$$
   여기서 $X \in \mathbb{R}^{f \times t}$이며, $W_1 \in \mathbb{R}^{h \times f}, W_2 \in \mathbb{R}^{f \times h}$이다. $\sigma$는 Hardswish 활성화 함수이다.

2. **Time-mixing**: 시간 차원($t$)을 은닉 차원($g$)으로 투영했다가 다시 $t$로 복원한다. 입력 $U$의 전치 행렬 $U^T$를 사용하여 연산한 후 다시 전치하여 원래 크기로 되돌린다.
   $$Y = U + (W_4 \sigma(W_3 \text{LayerNorm}(U^T)))^T$$
   여기서 $W_3 \in \mathbb{R}^{g \times t}, W_4 \in \mathbb{R}^{t \times g}$이다.

3. **Aggregation**: 마지막 MLP 블록의 출력 $O \in \mathbb{R}^{f \times t}$에 대해 시간 차원 방향으로 Average Pooling을 수행하여 최종 임베딩 $z \in \mathbb{R}^f$를 생성한다.
   $$z = \frac{1}{t} \sum_{i=1}^{t} O_{:,i}$$

### 추론 절차 (Inference)

1초의 이동 윈도우(stride 100ms)를 사용하여 임베딩을 생성한다. 등록된(enrollment) 임베딩과 쿼리 임베딩 간의 Cosine distance를 계산한다. 두 임베딩의 길이가 다를 경우, 짧은 쪽을 컨볼루션하거나 제로 패딩하여 길이를 맞춘 후 최소 거리를 구한다. 이 거리가 설정된 임계값(threshold)보다 작으면 키워드가 인식된 것으로 판단한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Librispeech(학습), Hey-Snips(평가), 내부 데이터셋(평가, 8개 키워드 및 400명 화자).
- **환경**: Clean, 10dB Noise, 6dB Noise 환경을 각각 Non Far-Field와 Far-Field 상황으로 나누어 총 6가지 시나리오에서 테스트하였다.
- **지표**: FA (False Acceptance) per hour가 0.3일 때의 FRR (False Rejection Rate)을 측정하였다.
- **비교 대상**: GRU-ATTN, MobileNetV2, MobileNetV3, EfficientNetB0, ViT.

### 주요 결과

1. **모델 효율성**: 제안 모델은 파라미터 수 0.25M, MACs 20.16M으로 모든 베이스라인 모델 중 가장 작은 크기와 연산량을 기록하였다. (표 1 참고)
2. **인식 성능**:
   - **Hey-Snips**: 소음이 심한 6dB 및 Far-field 환경에서 베이스라인 대비 FRR을 각각 8.57%, 6.43% 감소시키며 가장 우수한 성능을 보였다.
   - **내부 데이터셋**: Clean 환경에서는 MobileNetV3가 약간 우세했으나, 소음 및 Far-field 환경으로 갈수록 QbyE-MLPMixer의 FRR이 더 낮게 나타나 강건함이 입증되었다.
3. **Ablation Study**:
   - **입력 방식**: 패치 임베딩(PE)을 사용하는 것보다 MFCC를 직접 입력하는 것이 Non Far-Field 기준 에러율을 약 40% 감소시켰다.
   - **활성화 함수**: Hardswish가 GELU, ReLU, SiLU보다 전반적으로 가장 낮은 FRR을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 Vision 분야에서 주목받은 MLPMixer가 오디오 임베딩 추출 task에서도 매우 효율적일 수 있음을 보여주었다. 특히 주목할 점은 모델의 크기가 매우 작음에도 불구하고, 복잡한 Attention 메커니즘이나 깊은 CNN 구조 없이도 소음과 원거리 환경이라는 까다로운 조건에서 SOTA 모델들을 능가했다는 점이다.

또한, 비전 모델을 오디오에 적용할 때 흔히 사용하는 '패치 분할' 방식이 오디오의 시계열 특성상 오히려 독이 될 수 있음을 밝혀냈으며, MFCC 특징을 직접 활용하는 것이 훨씬 효과적이라는 통찰을 제공하였다. 다만, 내부 데이터셋의 Clean 환경에서는 MobileNetV3가 더 나은 성능을 보였다는 점에서, 극도로 깨끗한 환경보다는 실제 소음이 존재하는 환경에서 MLPMixer의 강점이 더 잘 드러난다고 해석할 수 있다.

## 📌 TL;DR

본 연구는 MLPMixer 아키텍처를 Open-vocabulary KWS의 QbyE 방식에 적용하여, **극소형 파라미터(0.25M)와 낮은 연산량으로도 소음 및 Far-field 환경에서 기존 RNN/CNN 모델보다 우수한 인식 성능**을 내는 모델을 제안하였다. 특히 오디오 데이터에 최적화된 입력 방식과 Hardswish 활성화 함수를 통해 효율성을 극대화하였으며, 이는 향후 화자 확인(Speaker Verification)이나 새로운 단어 감지(Novelty Detection)와 같은 저전력 임베딩 추출 시스템에 중요한 참고 자료가 될 것으로 보인다.
