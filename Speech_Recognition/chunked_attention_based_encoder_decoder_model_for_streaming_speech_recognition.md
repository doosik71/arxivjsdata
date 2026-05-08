# Chunked Attention-Based Encoder-Decoder Model for Streaming Speech Recognition

Mohammad Zeineldeen, Albert Zeyer, Ralf Schluter, Hermann Ney (2024)

## 🧩 Problem to Solve

본 논문은 실시간 스트리밍 음성 인식(Streaming Speech Recognition)을 가능하게 하는 Attention-based Encoder-Decoder (AED) 모델의 구조적 단순화와 강건성 향상을 목표로 한다.

기존의 스트리밍 가능 AED 모델들은 설계가 지나치게 복잡하고 많은 휴리스틱(heuristics)에 의존하며, Transducer 모델에 비해 강건성이 떨어진다는 문제가 있었다. 특히, 기존의 Global AED 모델은 전체 입력 시퀀스를 한 번에 처리해야 하므로 실시간 처리가 불가능하며, 시퀀스 길이가 길어질수록 성능이 급격히 저하되는 문제와 길이에 대한 편향(length bias) 문제가 존재한다. 따라서 본 연구는 단순한 수정을 통해 AED 모델을 스트리밍 가능하게 만들면서도, 긴 음성 데이터(long-form speech)에 대해 높은 일반화 성능을 유지하는 모델을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인코더와 디코더 모두에 **청크(Chunk)** 단위의 처리를 도입하는 것이다.

가장 중추적인 설계 변경은 기존의 시퀀스 종료 심볼인 $\text{EOS}$ (End-of-Sequence)를 **$\text{EOC}$ (End-of-Chunk)** 심볼로 대체한 점이다. 디코더가 $\text{EOC}$ 심볼을 생성하면 다음 청크로 이동하도록 설계함으로써, 모델이 고정된 크기의 윈도우를 순차적으로 처리하게 한다. 이러한 구조적 변경은 AED 모델을 프레임 단위가 아닌 청크 단위로 작동하는 Transducer 모델과 수학적으로 동등한 관계로 위치시키며, 이를 통해 AED의 유연성과 Transducer의 스트리밍 강건성을 동시에 확보하고자 하였다.

## 📎 Related Works

음성 인식의 스트리밍 모델로는 전통적인 $\text{HMM}$, $\text{CTC}$, 그리고 최근의 $\text{Transducer}$가 사용되어 왔다. AED 모델을 스트리밍 가능하게 만들기 위해 다양한 시도들이 있었으나, 본 논문은 이들이 너무 복잡하거나 강건하지 못하다고 지적한다.

또한, 변수 경계를 가진 세그먼트 단위 처리(segmental attention)나 가변 위치의 고정 크기 윈도우 처리 방식이 제안된 바 있다. 하지만 본 논문은 고정된 위치의 고정 크기 청크를 사용함으로써, 인코더 학습 시 정렬(alignment)과 무관하게 계산을 병렬화할 수 있다는 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

전체 시스템은 입력 음성 신호를 고정된 크기의 청크로 나누어 처리하는 **Chunked Encoder**와, 해당 청크 내에서만 Attention을 수행하며 $\text{EOC}$ 심볼을 통해 청크 간 이동을 제어하는 **Chunked Decoder**로 구성된다.

### 2. Streamable Chunked Encoder

- **구조**: $\text{Conformer}$ 기반 인코더를 사용한다.
- **작동 방식**: 입력 시퀀스 $x_{1:T}$를 크기 $T_w$와 스트라이드(stride) $T_s$를 가진 청크로 분할한다.
- **Self-Attention**: 각 청크 내에서는 비인과적(non-causal)으로 작동하여 현재 청크의 모든 프레임과 오른쪽 컨텍스트($T_r$)를 참조한다. 또한, 이전 청크의 정보를 참조함으로써 히스토리 컨텍스트가 층(layer)을 거치며 누적되도록 설계하였다.
- **특징**: 학습 시 모든 청크를 병렬로 계산할 수 있어 효율적이며, 이는 $\text{Emformer}$의 look-ahead 누출 방지 방식과 수학적으로 동일하다.

### 3. Streamable Chunked Decoder

디코더는 현재 처리 중인 인코더 청크의 출력값에 대해서만 Attention을 수행한다.

- **$\text{EOC}$ 메커니즘**: 디코더는 출력 심볼로 $\text{EOC}$를 생성할 수 있다. $\text{EOC}$가 생성되면 현재 청크 인덱스 $k$를 $k+1$로 증가시켜 다음 청크로 진입한다.
- **수식 설명**:
  다음 라벨 $a_s$를 생성할 확률은 $\text{ZoneoutLSTM}$과 $\text{MLP cross-attention}$을 통해 계산된다.
  $$p(a_s | ...) = (\text{softmax} \circ \text{Linear} \circ \text{maxout} \circ \text{Linear})(g_s, c_s)$$
  여기서 $g_s$는 $\text{LSTM}$의 상태이며, $c_s$는 현재 청크 $k_s$ 내의 인코더 출력 $h'_{k_s, t}$에 대한 가중합(weighted sum)이다.
  $$c_s = \sum_{t=1}^{T'_w} \alpha_{s,t} \cdot h'_{k_s, t}$$
  가중치 $\alpha_{s,t}$는 현재 청크 범위 내에서만 계산되는 $\text{Softmax}$ attention 값이다.
- **청크 인덱스 업데이트**:
  $$k_s = \begin{cases} k_{s-1} + 1, & a_{s-1} = \text{EOC} \\ k_{s-1}, & a_{s-1} \neq \text{EOC} \end{cases}$$
  초기값은 $k_1 = 1$이며, $k_s = K$이고 $a_s = \text{EOC}$일 때 전체 시퀀스가 종료된다.

### 4. 학습 및 추론 절차

- **학습**: 기존의 프레임 단위 정렬(framewise alignment)을 청크 단위 정렬로 변환하고 $\text{EOC}$ 라벨을 추가한 후, 라벨 기반의 $\text{Cross-Entropy}$ 손실 함수를 사용하여 학습한다. $\text{Transducer}$와 달리 모든 정렬 경로를 합산하지 않고 특정 정렬을 따라 학습한다.
- **추론**: $\text{Alignment-synchronous beam search}$를 수행한다. 즉, 모든 가설이 $\text{EOC}$를 포함하여 동일한 수의 라벨을 가지도록 정렬하며 탐색한다. 외부 언어 모델($\text{LM}$) 및 내부 언어 모델($\text{ILM}$) 보정(prior correction)을 적용하여 성능을 최적화한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: $\text{LibriSpeech 960h}$, $\text{TED-LIUM-v2 200h}$ (BPE 라벨 사용).
- **기준선**: $\text{Global AED}$ 모델.
- **지표**: 단어 에러율 ($\text{WER}$, Word Error Rate).

### 2. 주요 결과

- **디코더 청킹의 효과**: $\text{Global Encoder}$를 유지한 채 디코더만 청킹했을 때, 매우 작은 청크 크기에서도 $\text{Global AED}$와 거의 동일한 $\text{WER}$을 달성하였다 (Table 1).
- **인코더-디코더 전체 청킹**: $\text{Carry-over}$ (왼쪽 컨텍스트 유지)와 $\text{Look-ahead}$ (미래 컨텍스트 참조)를 적용했을 때 성능이 향상되었다. 최적 설정에서 $\text{TED-v2}$ 7.1%, $\text{LibriSpeech}$ 6.2%의 $\text{WER}$을 기록했으며, 이는 $\text{Global AED}$ 대비 소폭의 성능 저하(상대적 증가 4~9%)가 있으나 스트리밍이 가능하다는 이점이 있다 (Table 2).
- **긴 음성 데이터 일반화**: 가장 유의미한 결과로, 입력 시퀀스가 길어질수록 $\text{Global AED}$는 성능이 급격히 하락(WER 8.2% $\rightarrow$ 62.4%)하는 반면, $\text{Chunked AED}$는 일정한 성능을 유지하거나 오히려 향상되는 모습을 보였다 (Table 4).
- **길이 편향 문제 해결**: $\text{Global AED}$는 빔 사이즈가 커질수록 $\text{Length Normalization}$ 없이는 $\text{WER}$이 급격히 증가하지만, $\text{Chunked AED}$는 $\text{EOC}$ 심볼이 길이를 모델링하므로 이러한 편향 문제 없이 일정한 성능을 유지했다 (Table 5).

## 🧠 Insights & Discussion

### 1. 강점 및 통찰

본 연구는 AED 모델의 구조를 아주 단순하게 수정($\text{EOS} \rightarrow \text{EOC}$ 및 청크 단위 처리)하는 것만으로도 스트리밍 가능성을 확보하고 $\text{Transducer}$ 모델의 강건성을 얻을 수 있음을 입증하였다. 특히, 긴 음성 데이터에 대한 일반화 성능은 $\text{Global AED}$나 기존의 세그먼트 기반 모델보다 훨씬 우수하며, 이는 상대적 위치 인코딩(relative positional encoding)과 청크 단위의 국소적 처리 덕분인 것으로 해석된다.

### 2. 한계 및 비판적 해석

- **인코더 성능 저하**: 결과 분석에 따르면, 성능 저하의 주된 원인은 청킹된 인코더에서 발생한다. 디코더만 청킹했을 때는 $\text{Global AED}$와 성능 차이가 거의 없다는 점은, 인코더의 국소적 처리가 전역적 문맥 파악 능력을 일부 희생시킨다는 것을 의미한다.
- **지연 시간(Latency)**: $\text{Look-ahead}$ 컨텍스트를 추가하면 성능은 향상되지만, 단어 생성 지연 시간이 증가하는 트레이드-오프(trade-off) 관계가 명확히 나타났다 (Table 3).

## 📌 TL;DR

본 논문은 AED 모델에 고정 크기 청킹(chunking)과 $\text{EOC}$ (End-of-Chunk) 심볼을 도입하여, 단순하면서도 강력한 스트리밍 가능 음성 인식 모델을 제안하였다. 이 모델은 수학적으로 청크 단위 $\text{Transducer}$와 동등하며, 기존 AED의 고질적인 문제인 길이 편향(length bias)을 해결하고 긴 음성 시퀀스에 대해 매우 뛰어난 일반화 성능을 보인다. 향후 실시간 ASR 시스템에서 복잡한 $\text{Transducer}$ 설계의 대안으로 AED 기반의 청킹 구조가 유효하게 사용될 가능성이 높다.
