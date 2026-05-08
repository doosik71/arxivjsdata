# EXPLORING RWKV FOR MEMORY EFFICIENT AND LOW LATENCY STREAMING ASR

Keyu An, Shiliang Zhang (2023)

## 🧩 Problem to Solve

본 논문은 실시간 음성 인식(Streaming ASR) 시스템에서 발생하는 지연 시간(Latency)과 메모리 소비 문제를 해결하고자 한다. 최근의 ASR 모델들은 기존의 RNN(Recurrent Neural Networks) 대신 Self-attention 기반의 Transformer나 Conformer를 채택하여 성능을 크게 향상시켰다. 그러나 기본적으로 Self-attention 메커니즘은 전체 시퀀스를 한 번에 처리하는 구조이므로 실시간 스트리밍에 부적합하며, 연산 비용이 시퀀스 길이에 따라 제곱으로 증가하는 특성이 있다.

이를 해결하기 위해 기존 연구들은 Chunking(데이터를 작은 단위로 나누어 처리)이나 Caching(이전 계산 결과를 저장) 방식을 도입하였다. 하지만 Chunk 기반 모델은 현재 출력을 계산하기 위해 미래의 입력 프레임을 기다려야 하므로 필연적으로 인식 지연 시간이 발생하며, 과거의 표현(Representation)을 캐시에 저장해야 하므로 추론 시 메모리 사용량이 증가한다는 한계가 있다. 따라서 본 논문의 목표는 Transformer의 높은 성능과 RNN의 추론 효율성을 동시에 갖춘 RWKV(Receptance Weighted Key Value) 아키텍처를 스트리밍 ASR에 적용하여, 매우 낮은 지연 시간과 적은 메모리 비용으로도 높은 인식 정확도를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Linear Attention Transformer의 변형인 RWKV를 스트리밍 ASR의 Acoustic Encoder로 사용하는 것이다. RWKV는 다음과 같은 설계 직관을 가진다.

1. **RNN과 Transformer의 결합**: 학습 시에는 Transformer처럼 병렬 처리가 가능하여 효율적이지만, 추론 시에는 RNN처럼 이전 상태(State)만을 유지하며 순차적으로 계산하는 구조를 가진다.
2. **선형 복잡도**: Softmax-attention의 제곱 복잡도를 선형 복잡도로 줄여, 매우 긴 문맥 정보(Long-range dependencies)를 처리하면서도 메모리 사용량을 최소화한다.
3. **제로 지연 시간(Zero Latency)**: 미래의 컨텍스트를 참조하지 않고 현재 입력과 이전 상태만으로 계산하는 인과적(Causal) 구조를 통해, 이론적으로 지연 시간을 0으로 줄일 수 있다.

## 📎 Related Works

기존의 스트리밍 ASR 접근 방식은 크게 두 가지 방향으로 나뉜다.

1. **RNN 기반 모델**: 구조적으로 스트리밍에 최적화되어 있고 메모리 비용이 낮지만, 기울기 소실(Vanishing Gradient) 문제와 표현력의 한계로 인해 Transformer 계열보다 성능이 떨어진다.
2. **Self-attention 기반 모델**:
    * **Causal Self-attention**: 현재 프레임이 왼쪽 컨텍스트만 참조하게 하여 스트리밍을 가능하게 한다.
    * **Chunk Self-attention**: 현재 프레임이 현재 청크 내부의 미래 프레임과 이전 청크들의 정보를 참조한다. 성능은 더 좋지만, 미래 프레임을 기다려야 하는 지연 시간과 방대한 캐시 메모리 저장 비용이 발생한다.

본 연구는 이러한 기존 방식들의 한계를 극복하기 위해, 최근 NLP 분야에서 성공을 거둔 Linear Attention 기반의 RWKV를 ASR에 처음으로 적용하고, 이를 RNN-T(RNN Transducer) 및 BAT(Boundary-Aware Transducer) 구조와 결합하여 성능을 검증한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서는 RWKV를 Encoder로 사용하며, 이를 **Neural Transducer(RNN-T)**와 **Boundary-aware Transducer(BAT)** 프레임워크에 적용한다. 전체 파이프라인은 `입력 신호 $\rightarrow$ Convolution Subsampling $\rightarrow$ RWKV Blocks $\rightarrow$ Joint Network` 순으로 구성된다.

### RWKV Block 구성

하나의 RWKV 블록은 **Time Mixing** 모듈과 **Channel Mixing** 모듈로 구성되며, 각 모듈 뒤에는 LayerNorm, Dropout, 그리고 Residual Connection이 연결된다.

#### 1. Time Mixing Module

이 모듈은 Transformer의 Self-attention 역할을 수행하며, 시간축에 따른 정보 혼합을 담당한다. 입력 시퀀스 $x$에 대해 출력 $o_t$는 다음과 같이 계산된다.

$$o_t = W_o \cdot (\sigma(r_t) \odot wkv_t)$$

여기서 $\sigma(r_t)$는 Receptance 벡터이며, $r_t$는 다음과 같이 계산된다.
$$r_t = W_r \cdot (\mu_r x_t + (1-\mu_r)x_{t-1})$$

핵심이 되는 $wkv_t$는 과거의 모든 입력에 대한 가중 합산으로 정의된다.
$$wkv_t = \frac{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i}v_i + e^{u+k_t}v_t}{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}}$$

이 식은 추론 시 다음과 같은 재귀적(Recursive) 형태로 변환될 수 있어 RNN과 같은 효율적인 계산이 가능하다.
$$wkv_t = \frac{a_{t-1} + e^{u+k_t}v_t}{b_{t-1} + e^{u+k_t}}$$
단, $a_t = e^{-w}a_{t-1} + e^{k_t}v_t$ 이고 $b_t = e^{-w}b_{t-1} + e^{k_t}$ 이다. 여기서 $w$는 시간 감쇠(Time decay) 벡터이며, $u$는 현재 입력에 적용되는 가중치이다.

#### 2. Channel Mixing Module

이 모듈은 각 타임스텝 내에서 채널 간의 정보를 혼합한다.
$$o'_t = \sigma(r'_t) \cdot (W'_v \odot \max(k'_t, 0)^2)$$
여기서 $r'_t$와 $k'_t$는 현재 입력 $x'_t$와 이전 입력 $x'_{t-1}$의 선형 결합으로 계산된다. 이 과정 역시 인과적(Causal)이므로 미래 정보를 참조하지 않는다.

### 학습 프레임워크

* **RNN-T**: 입력 시퀀스와 레이블 시퀀스의 정렬을 통해 로그 확률을 최대화하도록 학습한다.
* **BAT**: RNN-T의 막대한 학습 연산량을 줄이기 위해, CIF(Continuous Integrate-and-Fire) 모듈을 이용해 오디오-텍스트 정렬 범위를 제한하여 메모리 사용량과 학습 시간을 단축시킨다.

## 📊 Results

### 실험 설정

* **데이터셋**: AISHELL-1(중국어), LibriSpeech(영어), WenetSpeech(중국어), GigaSpeech(영어) 등 다양한 규모(100h $\sim$ 10,000h)의 데이터 사용.
* **비교 대상**: Chunk-based Conformer Transducer (Chunk size 8 또는 16).
* **평가 지표**: CER(중국어), WER(영어), Latency(지연 시간), Left Context(추론 시 필요한 과거 프레임 수).

### 주요 결과

1. **지연 시간 및 메모리 효율성**:
    * **RWKV**: Latency가 $0\text{ms}$이며, Left Context가 $1$ 프레임에 불과하다.
    * **Chunk Conformer**: Chunk size에 따라 $320\text{ms}$ 또는 $640\text{ms}$의 지연 시간이 발생하며, 훨씬 더 많은 Left Context 캐시가 필요하다.

2. **인식 정확도**:
    * RWKV 기반 모델은 Chunk Conformer와 비교했을 때 대등하거나 오히려 더 나은 성능을 보였다. 특히, Chunk 기반 모델이 작은 Chunk size나 제한된 Left Context를 사용할 때 RWKV가 훨씬 뛰어난 성능을 나타냈다.
    * LibriSpeech와 GigaSpeech 데이터셋에서는 2-pass CTC + Attention 모델과 비교해도 경쟁력 있는 결과를 보여주었다.

3. **BAT vs Transducer**:
    * 두 모델의 정확도는 비슷하지만, BAT는 전체 학습 메모리 비용을 약 $40\%$, 학습 시간을 약 $25\%$ 감소시켰다. 다만, 회의(Meeting) 데이터와 같이 경계 찾기가 어려운 환경에서는 일반 Transducer의 성능이 더 좋았다.

4. **LSTM과의 비교**:
    * LSTM 기반 스트리밍 모델 역시 지연 시간과 메모리 이점이 있지만, 정확도는 RWKV보다 훨씬 낮았다. 이는 RWKV(Linear Attention)가 장기 의존성(Long-term dependencies)을 모델링하는 능력이 훨씬 뛰어남을 입증한다.

## 🧠 Insights & Discussion

본 논문은 RWKV가 스트리밍 ASR에서 매우 강력한 대안이 될 수 있음을 보여준다. 특히 **"성능(Transformer 수준) $\approx$ 효율성(RNN 수준)"**이라는 명제를 ASR 도메인에서 실증적으로 증명하였다.

**강점**:

* 추론 시 미래 프레임을 기다릴 필요가 없어 실시간성이 극대화된다.
* 과거 상태를 고정된 크기의 벡터로 유지하므로, 입력 길이가 길어져도 메모리 사용량이 일정하다.

**한계 및 비판적 해석**:

* **절대적 성능의 한계**: 매우 큰 Chunk size를 사용하고 무제한의 Left Context를 허용하는 Chunk Conformer보다는 정확도가 낮을 수 있다. 즉, 효율성을 위해 일부 정확도를 희생한 측면이 있다.
* **국소적 정보 캡처**: RWKV는 전역적인 문맥 파악에는 능하지만, 음성 신호 특유의 국소적 패턴(Local context)을 잡는 능력은 부족할 수 있다. 저자들 또한 이를 보완하기 위해 Convolution 레이어를 추가하거나 Chunk 기반 모델과 결합하는 방향을 제시하고 있다.

## 📌 TL;DR

이 논문은 Linear Attention의 일종인 **RWKV를 스트리밍 ASR의 인코더로 도입**하여, Transformer의 높은 성능과 RNN의 낮은 추론 비용 및 지연 시간을 동시에 달성하였다. 실험 결과, RWKV는 기존 Chunk 기반 Conformer 모델 대비 **지연 시간을 0으로 줄이고 메모리 사용량을 획기적으로 낮추면서도 대등한 인식 정확도**를 기록하였다. 이 연구는 특히 메모리와 지연 시간이 엄격하게 제한된 온디바이스(On-device) 실시간 음성 인식 시스템 구현에 매우 중요한 기여를 할 것으로 평가된다.
