# Research on Modeling Units of Transformer Transducer for Mandarin Speech Recognition

Li Fu, Xiaoxiao Li, Libo Zi (발행 연도 미명시)

## 🧩 Problem to Solve

본 연구는 중국어(Mandarin) 자동 음성 인식(ASR) 시스템에서 Recurrent Neural Network Transducer(RNN-T)의 성능을 최적화하기 위한 두 가지 핵심 요소인 모델 아키텍처와 모델링 단위(Modeling Unit)의 영향을 분석하는 것을 목표로 한다.

중국어 ASR에서 기존의 RNN-T 기반 모델들은 주로 아키텍처 수정에 집중해 왔으며, 모델링 단위가 성능에 미치는 영향에 대한 연구는 상대적으로 부족했다. 또한, 실제 서비스 환경에서는 8kHz와 16kHz 등 서로 다른 샘플링 레이트의 오디오 데이터가 혼재되어 사용되는데, 단일 모델이 이러한 다양한 샘플링 레이트를 동시에 처리할 수 있는 일반화 성능을 갖추는 것이 중요한 과제로 제기되었다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

1. **하이브리드 Transformer Transducer 구조 제안**: Acoustic feature encoding에는 Self-attention Transformer를 사용하고, Label feature encoding에는 RNN(LSTM)을 결합한 새로운 아키텍처를 제안하여 정확도와 모델 크기, 계산 효율성을 동시에 확보하였다.
2. **Mix-bandwidth 훈련 방법 제시**: 8kHz와 16kHz 샘플링 레이트의 데이터를 동시에 학습할 수 있는 Mix-bandwidth 훈련 기법을 도입하여, 다양한 샘플링 레이트의 오디오에 대해 범용적인 인식 성능을 가진 모델을 구축하였다.
3. **모델링 단위에 따른 성능 비교 분석**: 중국어 인식에 사용되는 세 가지 모델링 단위(Syllable initial/final with tone, Syllable with tone, Chinese character)를 비교 실험하여, 'Syllable with tone' 단위가 가장 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 end-to-end ASR 프레임워크 중 RNN-T는 음향 정보와 언어 정보를 통합하여 학습하며, 프레임 동기화(frame-synchronous) 방식으로 정렬을 학습하기 때문에 스트리밍 ASR 애플리케이션에 적합하다는 장점이 있다. 최근에는 RNN-T의 RNN 구조를 Multi-head self-attention 메커니즘으로 대체한 Self-attention Transducer가 제안되어 병렬 연산 능력과 정확도가 향상되었다.

모델링 단위와 관련하여, 기존 연구에서는 Transformer 기반의 attention 모델에서 중국어 한자(Chinese character) 기반 모델이 가장 좋은 성능을 보였다는 결과가 보고된 바 있다. 그러나 본 논문은 RNN-T 구조 내에서 모델링 단위의 선택이 분류해야 할 클래스의 수와 레이블 시퀀스의 길이에 직접적인 영향을 미치며, 이것이 모델의 수렴 속도와 최종 인식 성능에 결정적인 역할을 한다는 점에 주목하여 차별화된 분석을 수행하였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 입력 음성 신호를 처리하여 텍스트 시퀀스로 변환하는 Transducer 구조를 따른다. 전체 파이프라인은 다음과 같은 순서로 구성된다:
$\text{Input Audio} \rightarrow \text{Mix-bandwidth Module} \rightarrow \text{Convolutional Layers} \rightarrow \text{Truncated Transformer Blocks} \rightarrow \text{Joint Network} \rightarrow \text{Softmax}$

### 주요 구성 요소 및 역할

1. **Acoustic Feature Encoder**:
    * **Convolutional Layers**: Transformer 블록 전단에 배치되어 입력 특징의 추출 성능을 향상시킨다.
    * **Truncated Transformer Blocks**: 무제한적인 Transformer 대신 truncated(절단된) 블록을 사용하여 모델을 스트리밍 가능하게 만들고, 지연 시간(latency)과 계산 비용을 줄인다. (Left context = 20, Right context = 5 설정)

2. **Label Feature Encoder**:
    * **Embedding Layer & LSTM**: Transformer 대신 Unidirectional LSTM을 사용하여 모델 크기를 줄이고 언어 특징 추출 효율을 높였다.

3. **Joint Network**:
    * 음향 인코딩 결과 $\mathbf{f}_t$와 레이블 인코딩 결과 $\mathbf{q}_u$를 입력으로 받아 결합된 로짓(logits)을 생성한다.
    * 수식은 다음과 같다:
        $$\mathbf{z}_{t,u} = \mathbf{W}_z \tanh(\mathbf{W}_p \mathbf{f}_t + \mathbf{W}_q \mathbf{q}_u + \mathbf{b}_z)$$
        $$\mathbf{y}_{t,u} = \text{softmax}(\mathbf{z}_{t,u})$$
    * 여기서 $\mathbf{W}_p, \mathbf{W}_q, \mathbf{W}_z$는 학습 가능한 파라미터이며, $\tanh$는 비선형 함수이다.

### Mix-bandwidth 훈련 방법

다양한 샘플링 레이트를 처리하기 위해 80-dimension Mel-filter bank(fbank)를 사용한다.

* **특징 계산**: 16kHz 데이터는 표준 프로세스로 계산하며, 8kHz 데이터는 512-point FFT를 수행한 후 고주파 영역을 0으로 패딩(zero-padding)하여 1024-dimension 출력으로 맞춘다.
* **특징 정규화(Normalization)**: 8kHz 데이터의 패딩 영역이 정규화 과정에서 정보 가중치를 떨어뜨리는 것을 방지하기 위해, 유효 대역폭 내에서만 평균 $\mu$와 표준편차 $\sigma$를 계산하여 적용한다.
* 구체적인 정규화 수식은 다음과 같다:
    $$\mathbf{s}_{n,d} = \frac{\mathbf{s}_{n,d} - \mu_{s,d}}{\sigma_{s,d}}$$
    여기서 $s$는 8kHz 또는 16kHz 데이터에 따라 서로 다른 통계값이 적용된다.

### 모델링 단위 (Modeling Units)

비교를 위해 다음 세 가지 단위를 사용하였다:

* **Syllable initial/final with tone**: 성모, 운모, 성조를 분리한 단위 (가장 적은 클래스 수, 가장 긴 시퀀스 길이)
* **Syllable with tone**: 성조가 포함된 음절 단위 (중간 수준의 클래스 수 및 시퀀스 길이)
* **Chinese character**: 중국어 한자 단위 (가장 많은 클래스 수, 짧은 시퀀스 길이)

## 📊 Results

### 실험 설정

* **데이터셋**: 총 12,000시간의 중국어 음성 데이터 (JD Digits 내부 데이터 및 AISHELL-1, AISHELL-2, THCHS-30 등 공공 데이터셋 포함).
* **학습 절차**: 먼저 CTC Loss를 사용하여 Acoustic Feature Encoder를 사전 학습(pre-train)한 후, Transformer Transducer로 미세 조정(fine-tuning)하였다.
* **지표**: Word Error Rate (WER) 및 Character Error Rate (CER).

### 정량적 결과

실험 결과, **Syllable with tone**을 모델링 단위로 사용했을 때 가장 우수한 성능을 보였다.

* **WER 관점**: 'Syllable with tone' 모델은 'Syllable initial/final with tone' 대비 평균 **14.4%**, 'Chinese character' 대비 평균 **44.1%**의 상대적 WER 감소를 달성하였다.
* **CER 관점**: 'Syllable initial/final with tone' 모델 대비 평균 **13.5%**의 상대적 CER 감소를 보였다.

| Modeling Units | Average WER (%) | Average CER (%) |
| :--- | :---: | :---: |
| Syllable initial/final with tone | 10.54 | 5.91 |
| **Syllable with tone** | **9.02** | **5.11** |
| Chinese character | 16.14 | N/A |

### 정성적 결과

학습 곡선(Learning Curve) 분석 결과, 'Syllable with tone' 단위를 사용한 모델이 다른 두 단위 모델에 비해 검증 데이터셋에서 손실 함수(loss function)가 가장 빠르게 감소하며 최적의 수렴 성능을 보였다.

## 🧠 Insights & Discussion

본 연구는 중국어 ASR 모델에서 모델링 단위의 선택이 단순한 레이블링 문제를 넘어 모델의 수렴 속도와 최종 정확도에 지대한 영향을 미친다는 것을 보여주었다.

**비판적 해석 및 논의**:

* **클래스 수와 시퀀스 길이의 트레이드-오프**: 한자 단위(Chinese character)는 시퀀스 길이는 짧지만 클래스 수가 너무 많아($7,228$개) 모델이 분류 학습에 어려움을 겪는 것으로 보인다. 반면, 성모/운모 분리 단위는 클래스 수($227$개)는 적지만 시퀀스 길이가 지나치게 길어져 정렬(alignment) 학습의 난이도가 높아진다. 'Syllable with tone'은 이 두 극단 사이에서 적절한 균형(클래스 수 $1,256$개, 짧은 시퀀스 길이)을 찾아내어 최적의 성능을 낸 것으로 해석된다.
* **강점**: 제안된 Mix-bandwidth 방법론은 실제 산업 현장에서 마주하는 다양한 샘플링 레이트 문제를 효과적으로 해결하여 모델의 범용성을 높였다.
* **한계**: 본 논문에서는 세 가지 단위만 비교하였으나, 중국어의 특성상 더 세분화된 단위나 하이브리드 단위의 가능성이 남아 있다. 또한, 최적의 단위를 찾는 과정이 수동적인 실험에 의존하고 있다.

## 📌 TL;DR

본 논문은 중국어 음성 인식을 위해 **Transformer(Encoder) + RNN(Label Encoder)** 기반의 하이브리드 Transducer 구조와 다양한 샘플링 레이트를 동시에 처리하는 **Mix-bandwidth 훈련법**을 제안하였다. 특히 다양한 모델링 단위를 비교 분석한 결과, **'Syllable with tone'** 단위가 클래스 수와 시퀀스 길이의 균형을 최적으로 맞추어 가장 낮은 WER/CER를 기록함을 확인하였다. 이 연구는 향후 ASR 모델 설계 시 언어적 특성에 맞는 최적의 모델링 단위를 선택하는 것이 아키텍처 수정만큼 중요하다는 점을 시사하며, 자동화된 모델링 단위 선택(AutoML)의 필요성을 제시한다.
