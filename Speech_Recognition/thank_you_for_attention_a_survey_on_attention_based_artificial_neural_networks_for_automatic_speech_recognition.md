# Thank you for Attention: A survey on Attention-based Artificial Neural Networks for Automatic Speech Recognition

Priyabrata Karmakar, Shyh Wei Teng, Guojun Lu (2021)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 분야에서 Attention 메커니즘을 기반으로 한 인공 신경망 모델들의 발전 과정과 구조를 분석하는 것을 목표로 한다.

전통적인 ASR 시스템은 음향 모델(Acoustic), 발음 모델(Pronunciation), 언어 모델(Language)의 세 가지 모듈이 각각 독립적으로 학습되어 통합되는 구조였다. 이러한 방식은 모듈 간의 불일치(Incompatibility) 문제를 야기하고 학습 시간이 오래 걸리며, 특히 발음 모델의 경우 전문가가 작성한 사전(Dictionary)에 의존하기 때문에 인간의 오류가 개입될 가능성이 크다는 한계가 있었다.

최근에는 이러한 한계를 극복하기 위해 전체 파이프라인을 하나의 신경망으로 학습시키는 End-to-End(E2E) ASR 시스템이 도입되었다. E2E ASR은 주로 Connectionist Temporal Classification(CTC) 기반 모델과 Attention 기반 모델로 나뉘는데, 본 논문은 특히 Attention 메커니즘이 음성 인식의 시퀀스 정렬(Alignment) 문제를 어떻게 해결하고 발전해 왔는지를 체계적으로 정리하여 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 ASR 시스템에서 사용되는 다양한 Attention 모델들을 **학습 모드(Offline vs. Streaming)**와 **기반 아키텍처(RNN vs. Transformer)**라는 두 가지 핵심 축을 중심으로 분류하고 종합적으로 리뷰한 것에 있다.

단순한 나열이 아니라, 전역적(Global) Attention에서 지역적(Local) Attention으로, 그리고 재귀적 구조(RNN)에서 병렬적 구조(Transformer)로 진화하는 기술적 흐름을 분석하였다. 특히, 실시간성이 중요한 Streaming ASR을 구현하기 위해 제안된 다양한 Monotonic Attention 및 Chunk-based 접근 방식들을 상세히 분석하여, 연구자들이 자신의 목적에 맞는 모델을 선택할 수 있는 가이드라인을 제공한다.

## 📎 Related Works

논문은 기존의 Attention 관련 서베이 논문들이 주로 자연어 처리(NLP) 분야의 기계 번역, 텍스트 분류, 요약 등에 집중되어 있음을 지적한다. NLP 분야의 Attention 연구는 매우 활발하지만, 음성 데이터라는 특수한 시퀀스(매우 긴 입력 길이와 단조로운 정렬 특성)를 다루는 ASR 특화 Attention 모델의 진화 과정을 전문적으로 다룬 문헌은 부족한 상태였다. 따라서 본 논문은 ASR이라는 특정 도메인에 집중하여 Attention 모델의 계보를 정리함으로써 기존 서베이 연구들과 차별점을 둔다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 새로운 알고리즘을 제안하기보다, 기존 모델들의 핵심 방법론을 체계적으로 설명한다.

### 1. RNN 기반 Encoder-Decoder 및 Attention

기본 구조는 입력 음성 프레임을 고차원 표현으로 변환하는 Encoder와 이를 바탕으로 심볼을 예측하는 Decoder로 구성된다.

- **Context Vector ($c_i$):** Decoder의 $i$번째 타임스텝에서 예측을 위해 참고하는 입력 시퀀스의 가중합이다.
  $$c_i = \sum_{j=1}^{L} \alpha_{i,j} h_j$$
  여기서 $h_j$는 Encoder의 $j$번째 hidden state이며, $\alpha_{i,j}$는 Attention 확률이다.

- **Attention Probability ($\alpha_{i,j}$):** Softmax 함수를 통해 계산되며, $j$번째 입력 프레임이 $i$번째 출력 예측에 얼마나 중요한지를 나타낸다.
  $$\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{j=1}^{L} \exp(e_{i,j})}$$

- **Matching Score ($e_{i,j}$):** Decoder state $s_{i-1}$과 Encoder state $h_j$ 사이의 유사도를 측정한다. Hybrid Attention의 경우 위치 정보($\alpha_{i-1}$)와 내용 정보($h_j$)를 모두 사용하여 다음과 같이 계산된다.
  $$e_{i,j} = w^T \tanh(Ws_{i-1} + Vh_j + Uf_{i,j} + b)$$

### 2. Transformer 기반 Encoder-Decoder

RNN의 순차적 처리 한계를 극복하기 위해 Self-Attention 메커니즘을 사용하며, 전체 시퀀스를 병렬로 처리한다.

- **Scaled Dot-Product Attention:** Query($Q$), Key($K$), Value($V$)를 이용하여 다음과 같이 계산한다.
  $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Multi-Head Attention:** 서로 다른 표현 부분 공간(Representation Subspaces)에서 정보를 동시에 포착하기 위해 위 과정을 $h$번 병렬로 수행하고 이를 연결(Concatenation)하여 최종 출력한다.

### 3. Offline ASR의 진화

- **Global $\rightarrow$ Local:** 모든 프레임을 보는 Global Attention의 연산 복잡도($O(L^2)$)를 줄이기 위해, 특정 윈도우 내의 프레임만 참조하는 Local Attention이 도입되었다.
- **Joint CTC-Attention:** Attention의 유연함(비단조적 정렬 가능성)으로 인한 오정렬 문제를 해결하기 위해, 엄격한 단조 정렬을 강제하는 CTC 손실 함수를 결합하여 학습한다.
  $$\mathcal{L} = \lambda \log p_{ctc}(Y|X) + (1-\lambda) \log p_{att}(Y|X)$$

### 4. Streaming ASR의 진화

실시간 처리를 위해 미래의 프레임을 볼 수 없는 제약 조건을 해결하는 방법론들이 중심이 된다.

- **Monotonic Attention:** 왼쪽에서 오른쪽으로만 정렬이 이동하도록 강제하여 지연 시간(Latency)을 줄인다.
- **MoChA (Monotonic Chunkwise Attention):** 단일 프레임이 아닌 고정된 크기의 '청크(Chunk)' 단위로 Attention을 수행하여 효율성과 성능을 동시에 잡는다.
- **Transformer-based Streaming:** Chunk-hopping, Truncated Self-Attention, 또는 Monotonic Multihead Attention(MMA) 등을 통해 Transformer의 전역 참조 특성을 지역적 참조로 제한하여 구현한다.

## 📊 Results

본 논문은 서베이 논문이므로 저자가 직접 수행한 실험 결과는 포함되어 있지 않다. 대신 리뷰 대상 논문들에서 보고된 주요 정량적/정성적 경향성을 다음과 같이 요약한다.

- **Transformer의 깊이:** Deep Transformer 연구([49], [50])에 따르면, Encoder-Decoder 층을 36-12 레이어 혹은 최대 42 레이어까지 확장했을 때 ASR 성능이 지속적으로 향상됨을 확인하였다.
- **연산 복잡도:** Self-Attention의 $O(L^2)$ 복잡도를 선형 복잡도로 줄이려는 시도([54])나, Downsampling을 통해 메모리 소비를 $a^2$배 줄이는 방법([53])이 효과적임이 제시되었다.
- **Streaming 성능:** MoChA 기반 모델이 상용 On-device ASR 시스템에 적용될 정도로 실용적인 성능과 지연 시간을 보였음을 언급한다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 ASR 모델의 진화 과정을 '제약 조건의 해결' 관점에서 매우 논리적으로 설명하고 있다. 특히, RNN에서 Transformer로의 전환이 단순히 유행을 따른 것이 아니라, 병렬 처리(Parallelism)와 장거리 의존성(Long-range dependency) 해결이라는 실질적인 이점을 제공했음을 명확히 한다. 또한, Offline 모델의 강력한 성능과 Streaming 모델의 낮은 지연 시간 사이의 Trade-off를 Attention 윈도우 조절 및 단조성(Monotonicity) 강제라는 방법론으로 풀어낸 과정을 상세히 다루었다.

### 한계 및 비판적 해석

- **정량적 비교의 부족:** 수많은 모델을 리뷰하고 있으나, 동일한 데이터셋(예: LibriSpeech)에서 각 모델의 성능(WER, Word Error Rate)을 직접적으로 비교한 요약 표가 없어, 어떤 모델이 현재 시점에서 가장 우수한지 한눈에 파악하기 어렵다.
- **최신 트렌드 반영의 한계:** 2021년 초에 작성된 논문으로, 최근의 Conformer나 Whisper와 같은 거대 사전 학습 모델(Large-scale Pre-trained Models)의 영향력이 충분히 반영되지 않았다.
- **구현 디테일 부족:** 수식은 제시되었으나, 실제 학습 시 사용되는 하이퍼파라미터나 최적화 기법(Optimizer, Learning rate schedule)에 대한 공통적인 분석은 누락되어 있다.

## 📌 TL;DR

본 논문은 ASR 시스템에서 핵심적인 역할을 하는 **Attention 메커니즘의 변천사를 RNN과 Transformer 구조, 그리고 Offline과 Streaming 환경으로 나누어 분석한 종합 서베이 보고서**이다.

전통적인 모듈형 ASR에서 End-to-End 구조로의 전환, 그리고 Global Attention에서 연산 효율과 실시간성을 고려한 Local/Monotonic Attention으로의 발전을 체계적으로 정리하였다. 이 연구는 향후 저지연(Low-latency) 실시간 음성 인식 시스템을 설계하려는 연구자들에게 적합한 Attention 구조를 선택하고 설계하는 데 중요한 이론적 배경을 제공한다.
