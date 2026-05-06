# XLSR-Transducer: Streaming ASR for Self-Supervised Pretrained Models

Shashi Kumar, Srikanth Madikeri, Juan Zuluaga-Gomez, Esaú Villatoro-Tello, Iuliia Thorbecke, Petr Motlicek, Manjunath K E and Aravind Ganapathiraju (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 자기지도학습(Self-Supervised Learning, SSL) 기반의 사전학습 모델들이 자동 음성 인식(Automatic Speech Recognition, ASR)에서 뛰어난 성능을 보임에도 불구하고, **스트리밍 ASR(Streaming ASR)** 환경에는 적합하지 않다는 점이다.

기존의 대중적인 SSL 사전학습 모델들은 전체 오디오 컨텍스트를 한 번에 처리하는 Full Attention 메커니즘으로 학습된다. 그러나 스트리밍 ASR은 오디오 청크(chunk)가 순차적으로 입력될 때마다 부분적인 가설(partial hypotheses)을 실시간으로 생성해야 하므로, 전체 입력에 의존하는 기존 모델을 그대로 사용할 경우 심각한 성능 저하가 발생하거나 실시간 처리가 불가능하다.

따라서 본 연구의 목표는 SSL 사전학습 모델인 XLSR-53을 Transducer 구조의 인코더로 통합하고, 효율적인 어텐션 마스킹(Attention Masking) 전략과 Attention Sinks 기법을 도입하여 저자원(low-resource) 환경에서도 강력한 성능을 내는 스트리밍 ASR 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여 및 핵심 아이디어는 다음과 같다.

1. **XLSR-Transducer 제안**: 다국어 SSL 모델인 XLSR-53을 Transformer-Transducer (TT) 구조의 인코더로 사용하는 아키텍처를 제안하였다. 이를 통해 적은 양의 지도 학습 데이터만으로도 고성능의 ASR을 달성하였다.
2. **스트리밍을 위한 어텐션 마스킹 및 학습 전략**:
    * **Chunked Masking**: 특정 크기의 청크 단위로 어텐션을 제한하여 실시간 처리를 가능케 하였다.
    * **Variable Left Context**: 현재 청크 이전의 과거 청크들을 컨텍스트로 활용하여 인식 정확도를 높였다.
    * **Multi-chunk Training**: 학습 시 청크 크기를 무작위로 선택하여, 하나의 모델이 추론 시 다양한 지연 시간(latency) 요구사항에 대응할 수 있도록 하였다.
3. **Attention Sinks의 ASR 최초 도입**: 언어 모델(LM)에서 발견된 Attention Sinks 현상(초기 토큰에 비정상적으로 높은 어텐션 점수가 부여되는 현상)을 스트리밍 ASR에 적용하였다. 이를 통해 전체 연산량을 줄이면서도 단어 오류율(WER)을 개선하는 성과를 거두었다.

## 📎 Related Works

본 논문에서는 End-to-End (E2E) ASR의 세 가지 주요 아키텍처인 Encoder-Decoder (AED), Connectionist Temporal Classification (CTC), 그리고 Neural Transducer를 언급한다. 특히 Transformer-Transducer (TT)는 구조적으로 스트리밍에 적합하여 널리 사용된다.

기존 연구들의 한계점은 다음과 같다:

* **TT 모델**: 일반적으로 처음부터 학습(train from scratch)해야 하며, 이를 위해서는 방대한 양의 지도 학습 데이터가 필요하다.
* **SSL 모델 기반 ASR**: 대부분 CTC loss나 Encoder-Decoder 구조를 사용하며, 스트리밍 환경을 위해 설계되지 않은 경우가 많다.

본 연구는 XLSR-53이라는 강력한 다국어 사전학습 인코더를 Transducer 구조에 결합함으로써, 저자원 상황에서도 스트리밍 가능한 고성능 ASR을 구현했다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

XLSR-Transducer는 다음과 같은 세 가지 주요 네트워크로 구성된다.

1. **Encoder**: 사전학습된 **XLSR-53** 모델을 사용한다. 원시 오디오를 입력받아 25ms 프레임 길이, 20ms 스트라이드(stride)를 가진 음향 임베딩을 생성한다.
2. **Predictor**: 상태가 없는(stateless) 구조로, 임베딩 레이어와 하나의 1-D CNN 레이어로 구성되어 이전의 non-blank 토큰들을 처리한다.
3. **Joiner**: 인코더와 예측기의 출력을 결합하는 단일 선형 레이어(linear layer)로 구성되며, 최종적으로 어휘집(vocabulary)에 대한 확률 분포를 예측한다.

학습에는 메모리 효율적인 **Pruned-Transducer Loss**가 사용된다.

### 스트리밍 구현을 위한 마스킹 전략

SSL 모델의 Full Attention 문제를 해결하기 위해 다음과 같은 마스킹 패턴을 도입하였다.

* **Chunked Masking**: self-attention 계산 시 현재 처리 중인 청크 외부의 프레임은 마스킹하여 접근을 차단한다. 청크 크기는 $320\text{ms}, 640\text{ms}, 1280\text{ms}, 2560\text{ms}$로 설정하여 실험하였다.
* **Variable Left Context**: 현재 청크 $n$을 디코딩할 때, 이전의 $k$개 청크를 왼쪽 컨텍스트(left context)로 활용하여 어텐션을 수행한다.
* **Multi-chunk Training**: 고정된 청크 크기로 학습하면 추론 시 다른 크기를 사용할 때 성능이 저하된다. 이를 해결하기 위해 매 배치마다 predefined 리스트에서 청크 크기를 무작위로 선택하여 학습시킨다.

### Attention Sinks

Transformer 레이어가 초기 토큰들에 높은 어텐션 점수를 부여하는 특성을 이용한다. 추론 시, 현재 청크와 왼쪽 컨텍스트 외에 **입력 시퀀스의 최상위 초기 프레임 몇 개(initial frames)**에 항상 접근할 수 있도록 설정한다. 이를 통해 전체 컨텍스트 윈도우를 넓히지 않고도 성능을 향상시킬 수 있다.

## 📊 Results

### 실험 설정

* **데이터셋**:
  * **AMI**: 실제 회의 상황의 대화 데이터로, 음향적 복잡성과 발화 겹침이 존재하는 실전적인 데이터셋이다. (훈련 80시간)
  * **CommonVoice**: 저자원 설정을 위해 언어당 100시간의 부분 집합을 추출하여 5개 언어(CA, BE, ES, FR, IT)에 대해 검증하였다.
* **기준선(Baseline)**: Whisper large-v2, Zipformer-Transducer (from scratch), FastConformer.
* **지표**: 단어 오류율(Word Error Rate, WER).

### 주요 결과

1. **비-스트리밍 성능 (AMI)**:
    * XLSR-Transducer는 Whisper large-v2 대비 절대적 WER을 4% 개선하였으며, 처음부터 학습한 Zipformer-Transducer 대비 상대적으로 39% 향상된 성능을 보였다.
2. **스트리밍 성능 및 청크 크기 영향**:
    * 청크 크기가 커질수록($320\text{ms} \rightarrow 1280\text{ms}$) 컨텍스트가 많아져 WER이 단조 증가하며 개선된다.
    * Multi-chunk 학습을 적용한 모델은 비-스트리밍 디코딩 시에도 성능 저하가 매우 적어($0.2\%$ 차이), 하나의 모델로 두 모드 모두 사용 가능하다.
3. **Attention Sinks의 효과**:
    * AMI 데이터셋 실험 결과, 왼쪽 컨텍스트 청크를 1개만 사용하고 16개 프레임의 Attention Sink를 추가한 경우가, 왼쪽 컨텍스트를 2개 사용하는 것보다 상대적으로 $12\%$ 더 낮은 WER을 기록하였다. 즉, 연산량은 유지하면서 성능을 높이는 효율적인 trade-off를 제공한다.
4. **다국어 검증 (CommonVoice)**:
    * 5개 언어 모두에서 스트리밍 모델이 비-스트리밍 모델과 유사한 수준(최대 $1.5\%$ absolute WER 차이)의 성능을 보이며 강건함을 입증하였다.

## 🧠 Insights & Discussion

### 강점

* **저자원 효율성**: SSL 사전학습 모델을 활용함으로써, 대규모 지도 학습 데이터 없이도 높은 인식 성능을 확보하였다.
* **유연한 지연 시간 제어**: Multi-chunk 학습을 통해 단일 모델로 다양한 청크 크기(지연 시간)에 대응할 수 있는 실용적인 구조를 제안하였다.
* **효율적인 컨텍스트 활용**: Attention Sinks의 도입이 단순히 과거 컨텍스트 윈도우를 늘리는 것보다 성능 개선 측면에서 더 효율적임을 실험적으로 증명하였다.

### 한계 및 논의사항

* **학습 비용**: SSL 모델의 전체 사전학습을 다시 수행하는 것은 불가능하므로 파인튜닝 단계에서만 스트리밍 마스킹을 적용하였다. 이는 사전학습과 파인튜닝 간의 train-test mismatch를 유발할 수 있으나, 본 논문에서는 Multi-chunk 학습으로 이를 어느 정도 완화하였다.
* **Attention Sink 프레임 수**: Attention Sink로 사용하는 최적의 프레임 수에 대한 이론적 근거보다는 실험적 결과에 의존하고 있다.

## 📌 TL;DR

본 논문은 SSL 모델인 XLSR-53을 Transducer의 인코더로 사용하는 **XLSR-Transducer**를 제안하여, 특히 저자원 환경의 스트리밍 ASR 성능을 크게 향상시켰다. Multi-chunk 학습 전략을 통해 다양한 지연 시간 설정에 대응 가능하게 하였으며, ASR 분야에 처음으로 **Attention Sinks** 기법을 도입하여 연산 효율성과 인식 정확도 사이의 최적의 trade-off를 찾아냈다. 이 연구는 향후 실시간 다국어 음성 인식 시스템 구축에 있어 중요한 가이드라인을 제공한다.
