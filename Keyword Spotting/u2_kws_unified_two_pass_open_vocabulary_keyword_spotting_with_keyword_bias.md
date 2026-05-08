# U2-KWS: UNIFIED TWO-PASS OPEN-VOCABULARY KEYWORD SPOTTING WITH KEYWORD BIAS

Ao Zhang, Pan Zhou, Kaixun Huang, Yong Zou, Ming Liu, Lei Xie (2023)

## 🧩 Problem to Solve

본 논문은 사용자가 키워드를 직접 설정할 수 있는 Open-vocabulary Keyword Spotting (KWS) 시스템의 성능 향상을 목표로 한다. 기존의 Open-vocabulary KWS 방식은 주로 자동 음성 인식(ASR)을 위한 Acoustic Model을 활용한 후 후처리 과정을 거치는 방식을 사용하였다. 그러나 이러한 방식은 모든 음소(Phoneme)를 모델링하는 ASR의 학습 기준을 따르기 때문에, 특정 키워드만을 탐지해야 하는 KWS 작업에는 최적화되지 않는 Under-optimization 문제를 야기한다.

또한, 오경보(False Alarm)를 줄이기 위해 가벼운 탐지기가 1단계로 동작하고 정밀한 검증기가 2단계로 동작하는 Multi-stage 전략이 사용되어 왔으나, 이는 시스템 구조를 복잡하게 만들어 구축 및 유지보수가 어렵다는 단점이 있다. 따라서 본 연구는 ASR의 Two-pass 구조에서 영감을 얻어, 탐지와 검증을 하나의 단일 모델로 통합하면서도 사용자 정의 키워드에 민감하게 반응하는 최적화된 KWS 프레임워크를 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Streaming 성능과 정밀도를 동시에 확보하기 위해 **서로 다른 특성의 Keyword Bias 메커니즘을 적용한 통합 Two-pass 구조**를 설계한 것이다.

1. **Unified Two-Pass Architecture**: CTC 브랜치를 통한 1차 후보 탐지(First-pass)와 Attention Decoder를 통한 2차 검증(Second-pass)을 하나의 모델로 통합하여 구조적 효율성을 높였다.
2. **Dual Keyword Bias Strategy**: 스트리밍 추론이 필요한 1단계에서는 Acoustic-query attention을 사용하여 실시간성을 확보하고, 정밀 검증이 필요한 2단계에서는 Keyword-query attention을 사용하여 오경보를 획기적으로 줄였다.
3. **KWS-Specific Training & Clipping**: ASR 기준이 아닌 KWS 작업에 특화된 데이터 샘플링 전략과, CTC의 타임스탬프 정보를 활용하여 디코더가 키워드 구간에만 집중하게 하는 Spike-based clipping 기법을 제안하였다.

## 📎 Related Works

기존의 Open-vocabulary KWS는 크게 두 가지 방향으로 발전해 왔다. 첫 번째는 Query-by-Example (QbyE) 방식으로, 텍스트가 아닌 오디오 신호 자체를 참조하여 비교하는 방법이다. 하지만 이는 사용자가 직접 키워드를 녹음해야 하며, 참조 음성의 품질에 따라 성능 편차가 크다는 한계가 있다.

두 번째는 텍스트 기반 방식으로, ASR의 Acoustic Model을 통해 음성 데이터를 phonetic posteriorgrams로 변환하고 HMM 등의 후처리 기법을 사용하는 방법이다. 최근에는 CATT-KWS와 같이 Two-pass ASR 모델을 KWS에 응용하여 탐지와 검증을 통합하려는 시도가 있었으나, 여전히 모델이 ASR 학습 기준(Criterion)으로 훈련되어 KWS 작업에 특화된 최적화가 이루어지지 않았다는 점이 한계로 지적된다. 본 논문은 이러한 한계를 극복하기 위해 모델 내부로 키워드 정보를 직접 주입하는 Keyword Bias 방식을 도입하였다.

## 🛠️ Methodology

### 전체 시스템 구조

U2-KWS는 Shared Encoder, Bias Module, CTC Decoder, Attention Decoder, 그리고 Keyword Encoder의 다섯 가지 주요 구성 요소로 이루어져 있다. 전체 파이프라인은 다음과 같이 동작한다.

1. **Shared Encoder**: Conformer 레이어로 구성되며, Dynamic chunk 전략을 사용하여 지연 시간(Latency)을 제어한다.
2. **Keyword Encoder**: 단일 LSTM으로 구성되며, 사용자가 입력한 텍스트 키워드를 고차원 표현으로 인코딩한다.
3. **First-pass (Streaming Branch)**: Shared Encoder의 출력에 Bias Module을 적용하고 CTC Decoder를 통해 잠재적 키워드 후보를 빠르게 탐지한다.
4. **Second-pass (Non-streaming Branch)**: 1단계에서 후보가 발견되면, 해당 구간의 오디오 표현을 Clipping 하여 Attention Decoder에 입력하고 최종적으로 키워드 존재 여부를 검증(Rescoring)한다.

### Keyword Bias 메커니즘

본 논문은 서로 다른 쿼리(Query) 구성을 가진 두 가지 Cross-attention을 사용하여 키워드 정보를 주입한다. 기본 Attention 수식은 다음과 같다.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Acoustic-query Attention (1단계)**: 오디오 표현 $h_a$를 Query로, 키워드 표현 $h_k$를 Key와 Value로 사용한다.
  $$\tilde{h}_a = \text{Attention}(Q_a, K_k, V_k)$$
  이 방식은 프레임 단위 처리가 가능하여 스트리밍 추론에 적합하며, 오디오 표현에 키워드 정보를 더해 phonetic posteriorgrams를 예측한다.

- **Keyword-query Attention (2단계)**: 키워드 표현 $h_k$를 Query로, 오디오 표현 $h_a$를 Key와 Value로 사용한다.
  $$\tilde{h}_k = \text{Attention}(Q_k, K_a, V_a)$$
  이 방식은 전체 오디오와 키워드 간의 전역적 상관관계를 모델링할 수 있어 정밀한 검증에 유리하지만, 계산 비용이 높아 2단계에서만 사용된다.

### 학습 절차 및 손실 함수

학습은 두 단계로 나누어 진행된다.

1. **1단계 학습**: Streaming 브랜치만을 먼저 학습시킨다. 텍스트 전사 데이터에서 무작위로 연속된 단어 시퀀스를 추출하여 긍정 샘플을 만들고, 렉시콘에서 무작위로 단어를 조합해 부정 샘플을 생성한다. 특히 키워드 뒤에 `<eok>`(end of keyword) 토큰을 추가하여 모델이 키워드 정보에 집중하도록 유도한다.
2. **2단계 학습**: 전체 모델을 공동 최적화한다. Attention Decoder의 입력은 `<sos> + keyword`로 고정하며, 긍정 샘플은 `keyword + <eok>`를, 부정 샘플은 `</s>`를 예측하도록 학습시킨다.

최종 손실 함수는 다음과 같은 결합 손실 함수를 사용한다.
$$L = \lambda L_{ctc} + (1-\lambda) L_{att}$$

### Encoder Clipping (Spike-based)

디코더의 복잡도를 줄이기 위해 1단계에서 탐지된 키워드의 타임스탬프를 기반으로 인코더 출력을 자르는(Clip) 기법을 사용한다. `<eok>` 토큰의 확률이 가장 높은 프레임을 종료 지점으로 설정하고, 종료 지점부터 역순으로 비-블랭크(non-blank) 토큰의 확률이 임계값을 넘는 'Spike'의 개수가 키워드 길이와 일치할 때를 시작 지점으로 설정하는 **Spike-based clipping** 방식을 제안하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 1,000시간의 차량 내 음성 데이터(Internal Corpus)와 공개 데이터셋인 AISHELL-1을 사용하였다.
- **평가 지표**: ROC 곡선 및 F1-score를 통해 성능을 측정하였다. 특히 False Alarm rate를 시간당 0.5회로 고정했을 때의 Wake-up rate(재현율)를 주요 지표로 삼았다.

### 주요 결과

- **성능 향상**: 제안된 U2-KWS는 기존의 Customized KWS 시스템 대비 False Alarm rate가 0.5회/h일 때 Wake-up rate가 상대적으로 **41% 향상**되는 결과를 보였다.
- **디코더 전략의 효과**: Baseline(CTC 전용) $\rightarrow$ Keyword Bias 추가(1단계) $\rightarrow$ Causal Decoder 추가(2단계) $\rightarrow$ Full Decoder 추가 순으로 성능이 지속적으로 향상되었다. 특히 Full Decoder(전체 컨텍스트 활용) 방식이 가장 높은 정밀도를 보였다.
- **키워드 길이에 따른 영향**: 키워드의 길이가 길수록 F1-score가 높아지는 경향을 보였다. 이는 짧은 키워드일수록 오경보 발생 확률이 높으며, 긴 키워드일수록 디코더의 변별력이 더 크게 작용하기 때문이다.
- **Clipping 전략 비교**: Clipping을 하지 않은 경우보다 Spike-based clipping을 적용했을 때 가장 높은 성능을 보였다. 이는 디코더가 불필요한 구간을 제외하고 키워드 검증 작업에만 집중할 수 있게 하기 때문이다.

## 🧠 Insights & Discussion

본 논문은 ASR 모델을 단순 활용하는 것을 넘어, KWS라는 특수 목적에 맞게 모델의 구조와 학습 방식을 재설계했다는 점에서 강점이 있다. 특히 스트리밍과 정밀 검증이라는 상충하는 요구사항을 두 종류의 Cross-attention(Acoustic-query vs Keyword-query)으로 해결한 점이 인상적이다.

또한, 하드웨어 리소스가 제한적인 엣지 디바이스 환경을 고려하여, 항상 켜져 있어야 하는(Always-on) 1단계 모델의 추가 파라미터를 0.04M 수준으로 최소화하면서도, 필요할 때만 정밀 디코더를 활성화하는 구조를 취함으로써 효율성과 성능의 균형을 맞추었다.

다만, 본 연구에서는 키워드 샘플링을 통해 가상의 커스텀 키워드 환경을 구축하여 학습시켰으나, 실제 사용자가 입력하는 매우 다양한 도메인의 키워드에 대해서도 동일한 일반화 성능을 유지할 수 있을지에 대해서는 추가적인 분석이 필요할 것으로 보인다.

## 📌 TL;DR

U2-KWS는 탐지(CTC)와 검증(Attention Decoder)을 통합한 Two-pass 구조의 Open-vocabulary KWS 프레임워크이다. 스트리밍 브랜치에는 Acoustic-query attention을, 검증 브랜치에는 Keyword-query attention을 적용하여 실시간성과 정확도를 모두 잡았으며, Spike-based clipping 기법으로 검증 효율을 극대화하였다. 결과적으로 기존 시스템 대비 오경보율을 낮추면서도 웨이크업 성공률을 41% 향상시켰으며, 이는 저전력 엣지 디바이스의 맞춤형 키워드 탐지 시스템에 매우 유용한 접근 방식이다.
