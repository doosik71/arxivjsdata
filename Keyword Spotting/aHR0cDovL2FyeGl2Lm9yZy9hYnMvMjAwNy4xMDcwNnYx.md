# Very Fast Keyword Spotting System with Real Time Factor below 0.01

Jan Nouza, Petr Cerva, and Jindrich Zdansky (2020)

## 🧩 Problem to Solve

본 논문은 음성 데이터에서 특정 단어나 구절을 탐지하는 Keyword Spotting (KWS) 시스템의 성능과 속도를 동시에 최적화하는 문제를 다룬다. KWS 시스템의 성능은 크게 두 가지 관점에서 평가된다. 첫째는 탐지 신뢰성으로, 미검출률(Miss Detection, MD)과 오경보율(False Alarm, FA)을 최소화하는 것이며, 둘째는 처리 속도이다.

특히 대규모 음성 기록(수천 시간 분량)을 검색하거나 실시간 알림 서비스에 적용하기 위해서는 Real-Time (RT) factor가 1보다 훨씬 작아야 하며, 기존의 LVCSR(Large Vocabulary Continuous Speech Recognition) 기반 방식은 전사(Transcription) 과정이 필요하여 속도가 느리다는 한계가 있다. 따라서 본 논문의 목표는 다양한 음성 데이터에서 양호한 성능을 유지하면서도 RT factor를 0.01 미만으로 낮춘 매우 빠른 KWS 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 KWS 파이프라인의 모든 단계(신호 처리, Viterbi 디코딩, 후보 탐지 및 신뢰도 계산)에 걸쳐 속도 최적화를 적용한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **Quasi-monophones 도입**: Triphone의 정밀도와 Monophone의 속도라는 상충 관계를 해결하기 위해, Triphone 상태를 Monophone 구조로 매핑하는 Quasi-monophone 방식을 제안한다. 이는 신경망의 추가 레이어를 통해 구현되며, 디코더가 처리해야 할 Filler 상태의 수를 획기적으로 줄여 속도를 향상시킨다.
2. **BFSMN (Bidirectional Feedforward Sequential Memory Networks) 활용**: RNN의 대안으로 BFSMN을 사용하여 메모리 효율성과 계산 속도를 높이면서도, 넓은 컨텍스트 범위를 통해 높은 인식 성능을 유지한다.
3. **단일 패스(Single Pass) 및 전방향 디코딩**: 중간 단계의 Lattice를 생성하고 저장할 필요 없이 프레임 동기 방식으로 데이터를 처리하여 Look-back 최소화 및 처리 시간을 단축한다.
4. **단순화된 신뢰도 계산**: 복잡한 Viterbi 연산 대신 $D_{best}$ 값의 차이를 이용한 근사치를 사용하여 후보 단어의 신뢰도를 빠르게 계산한다.

## 📎 Related Works

기존의 KWS 접근 방식은 크게 세 가지로 분류된다.

- **Acoustic Approach**: 제한된 키워드 어휘집만 사용하여 연속 음성 인식과 유사하게 작동하며, 나머지 음성은 Filler unit으로 처리한다. 가장 단순하고 빠르다.
- **LVCSR Approach**: 전체 음성을 텍스트로 전사한 후 텍스트나 Word Lattice에서 키워드를 검색한다. 언어 모델(LM)을 활용하므로 정확하지만 속도가 매우 느리고, 어휘집에 없는 단어는 탐지하지 못한다.
- **Phoneme Lattice Approach**: 음소(주로 Triphone) 단위의 Lattice를 생성하여 검색한다.

최근에는 DNN, CNN, RNN(특히 LSTM-CTC) 기반의 음향 모델이 도입되어 인식 정확도가 크게 향상되었다. 본 논문은 이러한 현대적 신경망 아키텍처를 수용하면서도, 처리 속도를 극대화하기 위해 BFSMN과 최적화된 디코딩 전략을 결합한 형태를 취한다.

## 🛠️ Methodology

### 전체 시스템 구조

시스템은 세 가지 모듈이 프레임 동기 방식으로 동작하는 파이프라인으로 구성된다.

1. **Signal Processing Module**: 입력 프레임으로부터 모든 HMM 상태에 대한 Log-likelihood를 계산한다.
2. **State Processing Module**: 활성화된 키워드 및 Filler 상태들에 대해 Viterbi 재결합(Recombination)을 제어한다.
3. **Spot Managing Module**: 키워드 모델의 마지막 상태에 도달한 후보들의 점수를 계산하고 신뢰도를 평가하여 최종 탐지 여부를 결정한다.

### 주요 구성 요소 및 최적화

#### 1. Quasi-monophones

디코더의 속도는 매 프레임 처리해야 하는 Unit의 수에 비례한다. Triphone을 사용하면 Filler의 수가 수천 개에 달해 속도가 느려지고, Monophone을 사용하면 속도는 빠르지만 성능이 떨어진다. 이를 해결하기 위해 Triphone 상태들을 Monophone 구조로 매핑하는 Quasi-monophone을 제안한다. 신경망의 마지막 층에서 매핑된 노드들 중 최댓값(Max value)을 취하는 레이어를 추가함으로써 디코더가 처리해야 할 상태 수를 크게 줄인다.

#### 2. State Processing 및 Viterbi 디코딩

각 상태 $s$의 점수 $d$는 다음과 같이 업데이트된다.
$$d(u, s, t) = L(s, t) + \max_{i=0,1} [d(u, s-i, t-1)]$$
여기서 $L(s, t)$는 신경망이 제공하는 log-likelihood이다. Unit $u$의 마지막 상태 점수를 $D(u, t)$라 하고, 전체 상태 중 최댓값을 $d_{best}(t)$, Unit들의 끝 상태 중 최댓값을 $D_{best}(t)$라고 정의하여 가지치기(Pruning) 및 다음 프레임의 시작 점수로 활용한다.

#### 3. Spot Managing 및 신뢰도 계산

키워드 $w$가 탐지되었을 때, 해당 구간의 최적 Filler 문자열 점수 $S(v_{string}, t)$와 키워드 점수 $S(w, t)$의 차이를 통해 신뢰도를 계산한다. 본 논문에서는 다음과 같은 근사식을 제안한다.
$$R(w, t) = D_{best}(t) - D(w, t)$$
이 $R$ 값을 기반으로 정규화된 신뢰도 점수 $C(w, t)$를 산출한다.
$$C(w, t) = 100 - k \frac{R(w, t)}{(t - T(w, t)) N_s(w)}$$
여기서 $T(w, t)$는 단어의 시작 프레임, $N_s(w)$는 HMM 상태 수, $k$는 사용자 해석을 돕기 위한 상수이다.

#### 4. 반복 실행 최적화 (Optimized Repeated Run)

동일한 오디오 데이터에서 키워드 리스트만 바꾸어 반복 검색하는 경우, 신호 처리 단계에서 계산된 Likelihood와 $d_{best}, D_{best}$ 값을 저장해 두었다가 재사용함으로써 신호 처리 모듈을 생략하고 디코더만 빠르게 구동할 수 있다.

## 📊 Results

### 실험 설정

- **데이터셋**: 체코어 데이터셋 3종 (Interview: 스튜디오, Stream: 소음 심함, Call: 전화 통화) 총 17시간 분량.
- **키워드 리스트**: 156개 표제어 및 555개 파생형.
- **모델**: Triphone DNN, BFSMN(Monophone, Quasi-monophone, Triphone).
- **평가 지표**: DET (Detection Error Tradeoff) 다이어그램, EER (Equal Error Rate), RT factor.

### 정량적 결과

1. **인식 정확도**:
    - **BFSMN-tri** 모델이 가장 우수한 성능을 보였으며, 이는 넓은 컨텍스트 범위 덕분이다.
    - **BFSMN-quasi-mono** 모델은 BFSMN-tri보다는 약간 낮지만, DNN-tri보다 우수하거나 유사한 성능을 보였다. (Interview 데이터셋 기준 EER: BFSMN-tri 9%, BFSMN-quasi-mono 11%, DNN-tri 16%).
2. **처리 속도 (RT Factor)**:
    - 신호 처리 모듈을 CPU에서 실행할 경우 0.12, GPU에서 실행할 경우 0.0005까지 단축된다.
    - 디코더 부분은 BFSMN-tri(0.012)보다 BFSMN-quasi-mono(0.002)가 훨씬 빠르다.
    - **최종 결과**: GPU를 사용하고 Quasi-monophone 방식을 적용하면 전체 시스템의 RT factor를 **0.001** 근처까지 낮출 수 있음을 확인하였다.
    - 키워드 개수를 555개에서 10,000개로 늘려도 속도는 약 2배 정도만 느려져, 키워드 수의 영향이 적음을 보였다.

## 🧠 Insights & Discussion

본 논문은 정확도와 속도 사이의 트레이드오프를 공학적으로 매우 효율적으로 해결하였다. 특히 BFSMN 아키텍처는 RNN/LSTM과 유사한 성능을 내면서도 학습과 추론 속도가 빨라 실용성이 높다.

Quasi-monophone 접근 방식은 정확도를 크게 희생하지 않으면서 디코더의 계산 복잡도를 획기적으로 줄였다는 점에서 매우 영리한 설계이다. 또한, 반복 실행 최적화(Repeated Run)는 수사 기관의 전화 통화 분석과 같이 동일 데이터에서 여러 키워드를 검색해야 하는 실제 시나리오에서 매우 강력한 이점이 된다.

한계점으로는 체코어 데이터셋만으로 평가되었다는 점이 있으나, 제안된 아키텍처가 언어 독립적(Language Independent)이므로 다른 언어에도 적용 가능하다는 주장은 타당해 보인다. 또한, BFSMN의 성능이 Triphone 기반 모델에 의존하므로, 기본 음향 모델의 품질이 전체 시스템의 하한선을 결정하게 된다.

## 📌 TL;DR

본 논문은 BFSMN 신경망과 Quasi-monophone 상태 매핑, 그리고 최적화된 Viterbi 디코딩 파이프라인을 통해 **RT factor 0.001 수준의 초고속 키워드 탐지 시스템**을 제안한다. 특히 GPU 가속과 Quasi-monophone의 결합은 대규모 음성 아카이브를 실시간보다 수천 배 빠르게 검색할 수 있게 하며, 이는 법집행 기관의 음성 분석이나 실시간 미디어 모니터링 시스템에 즉각적으로 적용 가능한 수준의 성능 향상을 가져온다.
