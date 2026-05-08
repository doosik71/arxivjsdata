# Predicting detection filters for small footprint open-vocabulary keyword spotting

Theodore Bluche, Thibault Gisselbrecht (2020)

## 🧩 Problem to Solve

본 논문은 리소스가 제한된 환경(예: 마이크로컨트롤러, MCU)에서 동작 가능한 소형 footprint의 **Open-vocabulary Keyword Spotting (KWS)** 시스템을 구축하는 것을 목표로 한다.

일반적인 KWS 시스템은 특정 키워드를 탐지하기 위해 해당 키워드가 포함된 전용 학습 데이터가 필요한 **Closed-vocabulary** 방식이다. 이는 새로운 키워드를 추가할 때마다 데이터를 수집하고 모델을 다시 학습시켜야 하는 유연성 부족의 문제를 야기한다. 반면, 기존의 Open-vocabulary 접근 방식은 ASR(자동 음성 인식) 시스템의 결과물에 의존하거나 복잡한 후처리 및 신뢰도 점수 보정(confidence score calibration) 과정이 필요하다는 한계가 있다.

따라서 본 연구의 목적은 특정 키워드 데이터 없이도 임의의 키워드를 탐지할 수 있으며, 연산 비용이 매우 낮아 온디바이스(on-device) 구현이 가능한 완전 신경망 기반의 KWS 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 키워드의 발음 정보(phone sequence)를 입력으로 받아, KWS 네트워크의 최상위 컨볼루션 필터 가중치를 직접 예측하는 **보조 네트워크(Auxiliary Network)**를 도입하는 것이다.

즉, 탐지하고자 하는 키워드가 결정되면 **Keyword Encoder**가 해당 키워드에 최적화된 필터 가중치를 생성하고, 이 필터가 음성 특징 맵에 적용되어 키워드 존재 여부를 판별한다. 이러한 방식은 컴퓨터 비전 분야의 Dynamic Convolution Filter 개념을 KWS에 적용한 것으로, 모델을 완전히 다시 학습시키지 않고도 새로운 키워드에 대응할 수 있는 유연성을 제공한다.

## 📎 Related Works

논문에서는 기존 KWS 접근 방식을 다음과 같이 분류하고 한계를 지적한다.

1. **End-to-End KWS**: 매우 높은 성능을 보이고 MCU에서 동작 가능하지만, 추론 시 탐지할 키워드에 대한 전용 학습 데이터가 필수적이다.
2. **ASR 기반 KWS**: ASR의 텍스트 전사(transcript)나 Lattice에서 키워드를 검색하는 방식이다. 전용 데이터는 필요 없으나, 프레임 단위의 폰(phone) 점수를 키워드 신뢰도로 변환하기 위한 복잡한 후처리와 보정이 필요하다.
3. **ASR-free Embedding 기반 KWS**: 오디오와 키워드 발음을 각각 임베딩 공간으로 투영하여 거리를 측정하는 방식이다. 전용 데이터가 필요 없다는 장점이 있으나, 본 논문이 제안하는 필터 예측 방식과는 구조적으로 차이가 있다.

본 제안 방식은 전용 데이터 없이 유연하게 키워드를 추가할 수 있으면서도, ASR 기반 방식의 복잡한 후처리 과정을 생략하고 직접적으로 키워드 탐지 점수를 도출한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 시스템은 크게 세 가지 구성 요소로 이루어져 있다.

1. **Acoustic Encoder ($F$):** 입력 음성 $x$로부터 특징을 추출하는 신경망으로, 양방향이 아닌 단방향 LSTM 층의 스택으로 구성된다. 이는 사전에 양자화된 ASR 음향 모델로 사전 학습된다.
2. **Keyword Encoder ($E$):** 키워드의 폰 시퀀스 $\pi_k$를 입력받아 최상위 컨볼루션 필터의 가중치 $\theta_k$를 예측하는 Bi-LSTM 네트워크와 Affine Transform 층으로 구성된다.
3. **Keyword Detector:** Acoustic Encoder의 출력에 Keyword Encoder가 생성한 필터 $\theta_k$를 적용하여 최종 확률값을 계산한다.

### 주요 방정식 및 연산 과정

키워드 $k$에 대한 출력 시퀀스 $y_k$는 다음과 같이 계산된다.
$$y_k = \sigma(\theta_k * F(x; \theta_F))$$
여기서 $\sigma$는 sigmoid 함수이며, $*$는 컨볼루션 연산이다. 이때 필터 $\theta_k$는 다음과 같이 Keyword Encoder에 의해 결정된다.
$$\theta_k = E(\pi_k; \theta_E)$$
결과적으로 전체 연산은 $y_k = \sigma(E(\pi_k; \theta_E) * F(x; \theta_F))$로 요약되며, 이는 키워드의 발음 정보가 직접적으로 탐지 필터를 결정함을 의미한다.

### 학습 절차 및 손실 함수

모델은 특정 키워드 데이터셋이 아닌 일반적인 음성 학습 데이터셋 $D = \{(x^{(i)}, \pi^{(i)})\}$를 사용하여 $F$와 $E$를 공동 학습시킨다.

1. **Positive/Negative 샘플 생성:** 각 학습 샘플 $x^{(i)}$의 시점 $t$에서, 해당 시점에 끝나는 폰 시퀀스의 접미사(suffix)들을 무작위로 샘플링하여 정답 세트 $K^+_{i,t}$를 구성한다. 반면, 동일 배치 내 다른 샘플들의 정답 세트를 오답 세트 $K^-_{i,t}$로 활용한다.
2. **손실 함수:** 다음과 같은 Cross-Entropy 손실 함수 $L_{KWS}$를 최소화한다.
$$L_{KWS} = \sum_{x^{(i)}, \pi^{(i)} \in D} \sum_{t} \left( -\sum_{k \in K^-_{i,t}} \log(1 - y^{(i)}_{k,t}) - \sum_{k \in K^+_{i,t}} \log y^{(i)}_{k,t} \right)$$
3. **데이터 정렬:** 정답 폰 시퀀스를 생성하기 위해 CTC(Connectionist Temporal Classification) 손실 함수를 사용하여 사전 학습된 LSTM 모델로 Forced Alignment를 수행한다.

### 특정 데이터셋에 대한 적응(Adaptation)

특정 키워드에 대한 학습 데이터 $D_K$가 존재하는 경우, 예측된 필터에 데이터 기반의 오프셋 $\theta^{(data)}_k$를 더해 미세 조정(Fine-tuning)을 수행할 수 있다.
$$\theta_k = E(\pi_k; \theta_E) + \theta^{(data)}_k$$
이 방식은 특정 키워드의 성능을 극대화하면서도, $\theta_E$는 그대로 유지하므로 새로운 임의의 키워드를 탐지하는 Open-vocabulary 능력을 상실하지 않는다.

## 📊 Results

### 실험 설정

- **모델 크기:** 5x64 LSTM 기반 모델(약 208.8k 파라미터)과 5x96 LSTM 기반 모델(약 440.7k 파라미터)을 사용하였으며, 모든 가중치는 8비트로 양자화되었다.
- **평가 지표:** 정확히 파싱된 쿼리의 비율(Exact rate)과 키워드 레벨의 F1 score를 측정하였다.

### 주요 결과

1. **KWS 성능:** 스마트 조명(lights) 및 세탁기(washing) 시나리오의 데이터셋에서 Viterbi, Lattice, Filler, Greedy, Sequence 등 5가지 베이스라인보다 월등한 성능을 보였다. 특히 더 작은 모델(5x64)을 사용했음에도 불구하고 기존의 Sequence post-processing 방식보다 높은 F1 score를 기록하였다.
2. **학습된 필터 분석:** 예측된 필터 간의 유클리드 거리를 측정한 결과, 발음이 유사하거나 접미사가 비슷한 단어들끼리 필터 값이 가깝게 형성됨을 확인하였다. 예를 들어 'increase'와 'decrease'는 필터 거리가 매우 가까웠으며, 이는 실제 모델의 혼동(confusion) 결과와 일치하였다.
3. **음성 명령(Speech Commands) 결과:** Google Speech Commands 및 내부 데이터셋에서 평가한 결과, 특정 데이터로만 학습된 모델(with data)보다는 낮았으나, 기존의 Acoustic KWS 베이스라인보다는 우수한 성능을 보였다.
4. **미세 조정 효과:** 특정 키워드 데이터로 미세 조정을 수행한 결과, 전용 데이터로 학습된 모델에 근접하는 성능 향상을 보였다.

## 🧠 Insights & Discussion

본 논문은 Open-vocabulary KWS의 유연성과 Closed-vocabulary KWS의 고성능이라는 두 마리 토끼를 잡기 위해 **필터 생성 네트워크**라는 구조적 접근을 취하였다.

**강점:**

- **극도로 낮은 Footprint:** 모델 크기가 250KB 미만으로 매우 작아 MCU 환경에 적합하다.
- **유연성:** 새로운 키워드를 추가할 때 재학습 없이 폰 시퀀스만 입력하면 즉시 탐지 필터를 생성할 수 있다.
- **신뢰도 보정 불필요:** ASR 기반 방식과 달리 신경망이 직접 확률값을 출력하므로 복잡한 후처리가 필요 없다.

**한계 및 논의:**

- **고립어(Isolated speech) 성능 차이:** 미세 조정 전후의 성능 차이가 큰 이유는 모델이 단순히 키워드 주변의 침묵(silence)을 학습했기 때문일 가능성이 제기되었다. 이는 향후 연구 과제로 남겨져 있다.
- **단순 폰 시퀀스 의존성:** 필터 예측이 주로 폰 시퀀스의 접미사에 의존하는 경향이 있어, 발음이 매우 유사한 단어 간의 변별력을 높이는 것이 향후 과제가 될 것으로 보인다.

## 📌 TL;DR

본 연구는 키워드의 발음 정보로부터 탐지 필터를 직접 예측하는 **Keyword Encoder**를 도입하여, 전용 학습 데이터 없이도 동작하는 **소형 Open-vocabulary KWS 시스템**을 제안하였다. 제안된 모델은 매우 작은 메모리 footprint를 가지면서도 기존의 음향 기반 KWS 방식을 상회하는 성능을 보였으며, 필요한 경우 특정 데이터로 미세 조정하여 전용 모델 수준의 성능을 확보할 수 있다. 이 연구는 가전제품이나 웨어러블 기기와 같이 리소스가 극히 제한된 환경에서 사용자 맞춤형 음성 인터페이스를 구현하는 데 중요한 역할을 할 것으로 기대된다.
