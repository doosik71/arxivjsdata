# Attention-based Transducer for Online Speech Recognition

Bin Wang, Yan Yin, Hui Lin (2020)

## 🧩 Problem to Solve

본 논문은 실시간 음성 인식(Online Speech Recognition)을 위한 End-to-End(E2E) 모델인 RNN-T(Recurrent Neural Network Transducer)가 가진 세 가지 주요 문제점을 해결하고자 한다.

첫째, **레이블 불균형(Label Imbalance)** 문제이다. RNN-T는 입력 시퀀스 $T$와 출력 시퀀스 $U$ 사이의 정렬을 위해 많은 양의 blank($\emptyset$) 심볼을 삽입한다. 일반적으로 $T \gg U$이기 때문에 blank 심볼이 압도적으로 많아지며, 이는 모델이 blank를 예측하는 쪽으로 편향되게 만들어 결과적으로 삭제 오류(Deletion Error)를 증가시킨다.

둘째, **학습 속도 및 메모리 효율성** 문제이다. RNN-T의 Joint Network 출력은 $T \times U \times (|Y|+1)$ 크기의 3차원 행렬 형태를 띤다. 이러한 거대한 행렬 구조는 메모리 소비를 극심하게 하며, 결과적으로 학습 시 미니 배치(Mini-batch) 크기를 크게 설정하지 못하게 하여 AED(Attention Encoder-Decoder)나 CTC(Connectionist Temporal Classification) 모델보다 학습 속도가 현저히 느리다.

셋째, **문맥 정보 활용의 부족**이다. 기존 RNN-T의 Joint Network는 현재 타임스텝의 인코더 출력 $h_t$만을 사용하며, 주변 문맥 정보(Contextual Information)를 충분히 고려하지 못해 인식 정확도 향상에 한계가 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 RNN-T의 구조에 **Attention 메커니즘을 결합**하여 효율성과 정확도를 동시에 높이는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **Joint Network 내 Chunk-wise Attention 도입**: 인코더의 출력을 고정된 길이의 청크(Chunk) 단위로 나누고, 디코더의 출력이 이 청크 내에서 Attention을 수행하게 한다. 이를 통해 정렬 그리드(Alignment Grid)의 크기를 획기적으로 줄여 메모리 효율을 높이고, blank 심볼의 절대적인 수를 줄여 레이블 불균형 문제를 완화한다.
2. **Encoder 내 Self-Attention 도입**: 인코더 상단에 Multi-head Self-Attention 층을 추가하여 음성 신호의 전역적/지역적 문맥 의존성을 더 잘 모델링함으로써 인식 정확도를 높인다.
3. **학습 및 추론 효율 최적화**: 청크 단위 처리와 인코더의 서브샘플링(Sub-sampling)을 통해 학습 속도를 1.7배 이상 향상시켰으며, 8-bit 양자화를 통해 CPU 환경에서도 낮은 지연 시간(Latency)과 실시간 계수(RTF)를 달성하였다.

## 📎 Related Works

논문에서는 E2E 음성 인식의 세 가지 주요 접근 방식을 언급한다.

- **CTC**: 실시간 처리에 유리하지만, 프레임 독립성 가정(Frame-independency assumption)으로 인해 성능 향상을 위해 별도의 언어 모델(LM)이 필수적이다.
- **AED**: 학습 효율과 성능이 뛰어나지만, 기본적으로 전체 시퀀스를 읽어야 하므로 실시간 처리가 어렵다. 이를 위해 Chunk-wise attention이나 Triggered attention 등이 제안되었으나, 대개 성능 저하가 동반된다.
- **RNN-T**: 실시간 스트리밍을 지원하면서 프레임 독립성 가정 문제를 해결한 우아한 솔루션으로 평가받으며 모바일 및 서비스 환경에서 널리 사용되고 있다.

본 연구는 기존의 Full Self-attention Transducer나 Transformer Transducer와 궤를 같이하지만, 특히 **Joint Network의 그리드 크기를 줄여 학습 속도를 높이고 레이블 불균형을 해결**하려는 Chunk-wise Attention의 적용 방식에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

제안된 Attention-based Transducer는 인코더, 디코더, 그리고 수정된 Joint Network로 구성된다.

### 2. 구성 요소별 상세 설명

#### (1) Encoder

- **Pyramid LSTM (pLSTM)**: 입력 시퀀스 $x$를 처리하며, 각 층에서 인접한 2개의 출력을 결합(Concatenate)하여 다음 층으로 전달한다. $n_p$개의 pLSTM 층을 거치면 전체 시퀀스 길이는 $\mu = 2^{n_p}$ 배만큼 감소(Down-sampling)한다.
- **Multi-head Self-Attention**: pLSTM과 LSTM 층 이후에 위치하며, 지역적 문맥 $\tau$를 고려하여 계산된다.
  - 스코어 계산: $s_i = \frac{(Q h_{lstm}^t)^T K h_{lstm}^i}{\sqrt{d/n_{att}}}$ (단, $i = t-\tau, \dots, t+\tau$)
  - 가중치 계산: $\alpha_i = \frac{\exp(s_i)}{\sum_{j=t-\tau}^{t+\tau} \exp(s_j)}$
  - 최종 출력: $h_t = \text{LayerNorm}(\sum_{i=t-\tau}^{t+\tau} \alpha_i V h_{lstm}^i + h_{lstm}^t)$

#### (2) Decoder

- 2개의 LSTM 층으로 구성된다.
- **Scheduled Sampling**: 학습과 추론 사이의 괴리를 줄이기 위해, 매 스텝마다 실제 정답 레이블과 보카불러리에서 샘플링한 무작위 레이블을 확률 $p_{ss}$에 따라 섞어서 입력으로 사용한다.

#### (3) Joint Network (핵심 수정 사항)

기존 RNN-T는 $h_t$ 하나만을 입력으로 받았으나, 제안 모델은 인코더 출력을 겹치지 않는 고정 길이 $w$(Chunk width)의 청크들로 나눈다.

- **Chunk-wise Attention**: 디코더 출력 $s_u$가 $c$번째 청크 내의 벡터들 $h_i^c$에 대해 Attention을 수행한다.
  - 어텐션 가중치: $a_{c,u}^i = \frac{(\hat{Q} s_u)^T \hat{K} h_i^c}{\sqrt{d/n_{att}}}$
  - 소프트맥스 가중치: $\alpha_{c,u}^i = \frac{\exp(a_{c,u}^i)}{\sum_{j=1}^w \exp(a_{c,u}^j)}$
  - 가중 합산 출력: $o_{c,u} = \sum_{i=1}^w \alpha_{c,u}^i \hat{V} h_i^c$
- 이후 $o_{c,u}$와 $s_u$를 결합하여 소프트맥스 층을 통과시키면 출력 유닛 및 blank에 대한 분포 $p_{c,u}$가 생성된다.

### 3. 학습 및 추론 절차

- **학습**: Forward-backward 알고리즘을 사용하여 로그 가능도 $\log p(y|x)$를 최대화하도록 학습한다.
- **추론**: Beam Search를 사용하여 청크 단위로 최적의 가설 시퀀스를 생성한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 중국인 L2 영어 학습자 데이터 (500시간 및 10,000시간).
- **비교 대상**: LAS(Listen, Attend and Spell), Baseline RNN-T, Kaldi Hybrid ASR (TDNN-f).
- **평가 지표**: Word Error Rate (WER), Real Time Factor (RTF), Latency.
- **환경**: 단일 CPU 코어 (Intel Xeon Platinum 8269CY), 8-bit 가중치 양자화 적용.

### 2. 주요 결과

- **500시간 데이터 실험**:
  - Proposed Model ($\tau=4, w=4$)은 WER 15.98%를 기록하여, Baseline RNN-T 대비 **10.7%의 상대적 WER 감소**를 보였다.
  - LAS 모델 대비로는 14.4%의 WER 감소 효과가 있었다.
  - 특히, 삭제 오류(Deletion Error)가 Baseline RNN-T 대비 약 30% 감소하여 레이블 불균형 문제가 완화되었음을 증명하였다.
- **10,000시간 데이터 실험**:
  - 최적 설정의 제안 모델은 Kaldi의 TDNN-f 시스템보다 **약 5.5% 낮은 WER**을 달성하였다.
- **효율성**:
  - 학습 속도가 1.7배 이상 빨라졌다.
  - 양자화 후 RTF는 $0.34 \sim 0.36$, Latency는 $268 \sim 409\text{ms}$ 수준으로 매우 낮게 유지되었다.

## 🧠 Insights & Discussion

본 논문은 RNN-T의 고질적인 문제인 레이블 불균형과 연산 복잡도를 **Chunk-wise Attention**이라는 단순하지만 강력한 구조적 변경을 통해 해결하였다.

**강점 및 해석**:

- 인코더의 서브샘플링($\mu$)과 Joint Network의 청크 너비($w$)가 결합되어, 결과적으로 정렬 그리드의 크기가 $\mu w$배만큼 줄어들었다. 이는 단순한 연산량 감소를 넘어, 학습 시 더 큰 배치 사이즈를 사용할 수 있게 하여 학습 안정성과 속도를 동시에 잡은 전략이다.
- blank 심볼의 수를 강제로 줄임으로써 모델이 더 이상 blank 예측에 편향되지 않고 실제 토큰을 더 적극적으로 예측하게 된 점이 삭제 오류 감소의 핵심 원인으로 분석된다.

**한계 및 논의**:

- Context length $\tau$를 늘리면 WER은 낮아지지만, 그만큼 look-ahead 시간이 늘어나 Latency가 증가하는 Trade-off 관계가 명확히 나타난다. 실시간 서비스 적용 시 서비스 요구사항에 맞는 최적의 $\tau$ 값을 찾는 것이 중요하다.
- 본 논문은 L2 영어라는 특수한 데이터셋에서 실험되었으므로, 일반적인 네이티브 화자나 타 언어에서도 동일한 수준의 성능 향상이 있을지는 추가 검증이 필요하다.

## 📌 TL;DR

본 연구는 RNN-T의 학습 속도 저하와 레이블 불균형 문제를 해결하기 위해 **Joint Network에 Chunk-wise Attention을 도입하고 Encoder에 Self-Attention을 추가**한 모델을 제안하였다. 이를 통해 메모리 사용량을 줄여 학습 속도를 1.7배 높였으며, 삭제 오류를 획기적으로 줄여 Baseline RNN-T 및 Kaldi 시스템 대비 우수한 WER 성능을 달성하였다. 이 연구는 RNN-T와 AED의 장점을 결합한 하이브리드 형태의 프레임워크를 제시했다는 점에서 향후 실시간 E2E 음성 인식 시스템 최적화에 중요한 기여를 할 것으로 보인다.
