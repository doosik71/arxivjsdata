# Multi-Channel Transformer Transducer for Speech Recognition

Feng-Ju Chang, Martin Radfar, Athanasios Mouchtaris, Maurizio Omologo (2021)

## 🧩 Problem to Solve

본 논문은 온디바이스(on-device) 환경에서의 원거리 음성 인식(Far-field Speech Recognition) 성능을 향상시키기 위해 다채널(Multi-channel) 입력을 효율적으로 처리하는 모델을 설계하는 것을 목표로 한다.

다채널 입력은 단일 채널보다 잡음 환경에서 강건함(robustness)을 제공하지만, 기존의 다채널 트랜스포머(Multi-channel Transformer, MCT) 모델은 다음과 같은 심각한 한계를 가지고 있다.

1. **높은 계산 복잡도**: 어텐션 메커니즘의 특성상 입력 시퀀스 길이에 따라 계산량이 제곱으로 증가한다.
2. **스트리밍 불가**: 인코더-디코더 어텐션(Encoder-Decoder Attention)과 양방향 인코더를 사용하므로 전체 발화가 입력될 때까지 기다려야 하며, 이는 높은 지연 시간(latency)을 초래한다.
3. **모델 크기의 가변성**: 마이크 개수나 프레임 수에 따라 모델 파라미터 수가 증가하는 구조를 가지고 있어, 메모리가 제한적인 온디바이스 시스템에 부적합하다.

따라서 본 연구의 목표는 낮은 계산 비용과 낮은 지연 시간을 유지하면서도, 다채널 정보를 효과적으로 통합하여 인식 정확도를 높인 스트리밍 가능한 음성 인식 모델인 MCTT(Multi-Channel Transformer Transducer)를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 다채널 정보 통합 과정에서의 효율성 극대화와 스트리밍 가능 구조의 도입이다.

1. **Transducer 프레임워크 채택**: 기존의 Encoder-Decoder 구조 대신 Transducer 구조를 사용하여 스트리밍 디코딩을 가능하게 하고 지연 시간을 줄였다.
2. **효율적인 Cross-Channel Attention (CCA) 설계**: 마이크 개수($C$)에 상관없이 모델 크기가 일정하게 유지되도록, 단순한 결합기(Combiner)인 평균(Average) 또는 연결(Concatenation) 방식을 제안하였다. 이는 기존 MCT의 아핀 변환(Affine transformation) 방식보다 훨씬 가볍다.
3. **문맥 제한(Context Limiting) 전략**: 어텐션 계산 시 과거(Left)와 미래(Right)의 참조 범위를 제한함으로써 계산 복잡도를 낮추고 실시간 처리가 가능하도록 구현하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **빔포밍(Beamforming)**: 고전적인 빔포밍(Delay-and-sum, Super-directive) 방식은 표준적인 전처리 모듈로 사용되어 왔으나, 환경 변화에 대한 적응력이 부족하다.
2. **신경망 기반 빔포밍(Neural Beamforming)**: 고정 빔포밍(FBF)과 적응형 빔포밍(ABF)으로 나뉘며, 최근의 추세는 신경망을 통해 가중치를 최적화하는 방식이다. 하지만 이러한 방식들은 대부분 단계별 학습(Stagewise training)을 거치며, 합성 데이터와 실제 데이터 간의 통계적 불일치로 인해 성능 저하가 발생할 수 있다.
3. **Multi-channel Transformer (MCT)**: 채널 간 어텐션을 통해 빔포밍과 음향 모델링을 통합한 End-to-End 모델이다. 성능은 우수하지만, 앞서 언급한 계산 복잡도와 지연 시간 문제로 인해 온디바이스 적용이 불가능하다는 한계가 있다.

### 차별점

MCTT는 Transducer 구조를 결합하여 스트리밍 능력을 확보함과 동시에, CCA의 구조적 개선을 통해 마이크 개수 증가에 따른 파라미터 증가 문제를 해결함으로써 실용적인 온디바이스 다채널 ASR을 구현하였다.

## 🛠️ Methodology

### 전체 시스템 구조

MCTT는 그림 1(a)와 같이 **Multi-channel Audio Encoder**, **Label Encoder**, 그리고 **Joint Network**의 세 가지 주요 구성 요소로 이루어져 있다.

### 1. Multi-channel Audio Encoder

다채널 오디오 입력 $X = (X_1, \dots, X_C)$를 처리하며, 다음 두 층이 반복되는 구조이다.

* **Channel-wise Self-Attention (CSA)**:
    각 채널 내에서 시간축에 따른 상관관계를 학습한다. 입력 특징(log-STFT magnitude and phase)에 Positional Encoding을 더한 후 Multi-head Attention (MHA)을 적용하여 각 채널의 독립적인 특징을 추출한다.
* **Cross-Channel Attention (CCA)**:
    채널 간의 문맥적 관계를 학습한다. $i$번째 채널이 Query($Q_{CC}^i$)가 되고, 나머지 채널들은 **Combiner**를 통해 Key($K_{CC}^i$)와 Value($V_{CC}^i$)가 된다.
  * **Avg Combiner**: 다른 채널들의 특징을 시간 및 임베딩 축으로 평균낸다.
      $$H_{CC}^i = \frac{1}{C} \sum_{j \neq i} \hat{X}_j$$
  * **Concat Combiner**: 다른 채널들을 시간 축으로 연결한다.
      $$H_{CC}^i = [\hat{X}_1; \dots; \hat{X}_j; \dots; \hat{X}_C]_{j \neq i}$$
    이 방식을 통해 마이크 개수 $C$가 변해도 모델의 파라미터 수는 일정하게 유지된다.

### 2. Label Encoder 및 Joint Network

* **Label Encoder**: 이전에 예측된 레이블 시퀀스를 입력으로 받는 Transformer 구조이다. 인과성(Causality)을 보장하기 위해 미래 프레임을 가리는 Masked MHA를 사용한다.
* **Joint Network**: Audio Encoder의 출력 $h_t$와 Label Encoder의 출력 $h_{u-1}$을 연결(Concatenate)하여 입력으로 받는 단일 은닉층 기반의 Feed-forward 신경망이다. 활성화 함수로는 $\tanh$를 사용하며, 최종적으로 Softmax를 통해 다음 토큰이나 blank 기호 $\langle b \rangle$를 예측한다.

### 3. 학습 및 추론 절차

모델은 다음과 같은 조건부 확률 분포를 최적화한다.
$$P(\hat{y}|X) = \prod_{i=1}^{T+U} P(\hat{y}_i | X, t_i, y_0, \dots, y_{u_{i-1}})$$
여기서 $T$는 blank 기호의 수, $U$는 레이블의 수이다. Forward-backward 알고리즘을 통해 모든 가능한 정렬 경로(alignments)에 대해 확률을 합산하여 최적화한다.

### 4. 스트리밍을 위한 문맥 제한 (Context Limiting)

계산 복잡도를 $O(T^2)$에서 상수 시간으로 줄이기 위해, 어텐션 계산 시 현재 프레임 $t$를 기준으로 왼쪽(과거) $L$ 프레임과 오른쪽(미래) $R$ 프레임만을 참조하도록 제한한다.

## 📊 Results

### 실험 설정

* **데이터셋**: 내부 Far-field 데이터셋 (학습 2,000시간, 검증 24시간, 테스트 233시간).
* **입력**: 7개 마이크 중 2개 채널 및 Super-directive 빔포밍 신호 사용.
* **지표**: 상대적 단어 오류율 감소량(WERR, Relative Word Error Rate Reduction). 값이 높을수록 성능이 좋음을 의미한다.

### 주요 결과

1. **단계별 모델 대비 성능**:
   * MCTT-2(2채널)는 단일 채널(SC-TT) 대비 **7.1%**, 신경망 빔포머(NBF-TT, NMBF-TT) 대비 **6.0%**의 WERR 개선을 보였다.
   * 빔포밍 신호를 추가한 MCTT-3는 모든 베이스라인보다 평균 4% 더 높은 성능 향상을 기록하였다.

2. **Multi-channel Transformer(MCT) 대비 성능 및 속도**:
   * MCTT-2는 MCT-2보다 특히 `test-clean` 세트에서 성능이 우수했다.
   * **추론 속도**: TP50(중간값) 기준으로 MCTT는 MCT보다 **15.8배 빠른** 추론 속도를 기록하였다. (MCT: 4.26s $\to$ MCTT: 0.27s)

3. **문맥 제한의 영향**:
   * **Label Encoder**: 과거 문맥 $L=4$만 사용해도 전체 문맥을 사용한 것과 성능 차이가 거의 없었다.
   * **Audio Encoder**: 미래 문맥을 완전히 제거($R=0$)하면 성능이 급격히 하락하지만, 소량의 미래 프레임(예: $R=10 \sim 20$)만 허용해도 Full-attention 모델과의 성능 격차를 크게 줄일 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 강점

* **효율적인 다채널 통합**: Avg/Concat Combiner를 도입하여 마이크 개수 증가에 따른 모델 크기 증가 문제를 해결한 점이 매우 실용적이다.
* **실시간성 확보**: Transducer 구조와 문맥 제한 전략을 통해, 성능 손실을 최소화하면서도 온디바이스에 적합한 낮은 지연 시간과 계산 복잡도를 달성하였다.

### 한계 및 해석

* **미래 문맥의 의존성**: 실험 결과에서 $R=0$일 때 성능이 크게 하락하는 것은, 다채널 신호의 공간적/시간적 특징을 파악하기 위해 어느 정도의 look-ahead(미래 참조)가 필수적임을 시사한다.
* **데이터셋의 제한**: 자체 데이터셋(in-house dataset)만을 사용하였으므로, 다른 공개 데이터셋이나 다양한 마이크 배열 구조에서도 동일한 성능 향상이 나타날지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 온디바이스 음성 인식을 위해 다채널 입력을 효율적으로 처리하는 **Multi-Channel Transformer Transducer (MCTT)**를 제안한다. 기존 MCT 모델의 높은 계산 복잡도와 지연 시간 문제를 해결하기 위해 **Transducer 프레임워크**, **단순 결합기(Avg/Concat) 기반의 CCA**, 그리고 **문맥 제한 어텐션**을 도입하였다. 실험 결과, 기존 빔포밍 기반 모델 및 MCT보다 높은 정확도를 보이면서도 추론 속도는 약 **15.8배 향상**되었으며, 이는 실시간 다채널 음성 인식 시스템 구축에 중요한 기여를 할 수 있다.
