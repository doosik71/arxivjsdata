# SMALL-FOOTPRINT KEYWORD SPOTTING USING DEEP NEURAL NETWORK AND CONNECTIONIST TEMPORAL CLASSIFIER

Zhiming Wang, Xiaolong Li, Jun Zhou (2017)

## 🧩 Problem to Solve

본 논문은 전력 및 자원이 제한된 소형 모바일 기기(small-footprint mobile devices)에서 동작하는 효율적인 키워드 검출(Keyword Spotting, KWS) 시스템 구축을 목표로 한다. 

기존의 DNN 기반 KWS(Deep-KWS) 시스템은 성능은 뛰어나지만, 특정 키워드에 특화된 학습 데이터(keyword-specific data)에 크게 의존한다는 치명적인 단점이 있다. 이러한 특정 데이터의 부족은 데이터 수집에 많은 비용과 인력을 소모하게 만들며, 데이터 양이 적은 상태에서 모델의 파라미터 수를 늘릴 경우 오히려 성능이 저하되는 오버피팅 문제가 발생한다. 또한, 사용자가 원하는 임의의 커스텀 키워드를 유연하게 추가하기 어렵다는 실무적인 제약이 존재한다.

따라서 본 연구의 목표는 특정 키워드 데이터에 대한 의존도를 낮추고, 이미 대량으로 존재하는 일반 대규모 단어 연속 음성 인식(Large Vocabulary Continuous Speech Recognition, LVCSR) 코퍼스를 최대한 활용하여 높은 정확도와 낮은 연산 비용을 동시에 달성하는 KWS 시스템을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep Neural Network(DNN)와 Connectionist Temporal Classifier(CTC)를 결합**하여, 키워드별 전용 데이터 없이도 일반 음성 인식 데이터를 통해 학습된 모델을 KWS에 활용하는 것이다.

- **일반 코퍼스 활용:** LVCSR 코퍼스를 사용하여 일반적인 음소(phoneme) 예측 모델을 학습시킨다.
- **유연한 결정 메커니즘:** DNN이 음소의 사후 확률을 예측하면, 그 상단에 위치한 CTC가 주어진 음소 시퀀스에 대한 신뢰도 점수(confidence score)를 계산함으로써, 특정 키워드 데이터 없이도 임의의 커스텀 키워드를 검출할 수 있는 구조를 설계하였다.
- **효율성 유지:** RNN이나 LSTM 대신 피드포워드 DNN을 사용하여 모바일 기기에서의 연산 복잡도와 메모리 점유율을 낮게 유지하면서도, CTC의 시퀀스 라벨링 능력을 통해 경쟁력 있는 성능을 확보하였다.

## 📎 Related Works

전통적인 KWS 방식은 은닉 마르코프 모델(Hidden Markov Models, HMMs)을 사용하였으나, HMM의 토폴로지에 따라 연산 비용이 매우 커질 수 있다는 한계가 있었다. 이후 등장한 Deep-KWS는 DNN을 통해 서브워드(sub-word)나 전체 단어(full-word) 단위를 예측하여 성능을 높였으며, CNN을 도입해 파라미터 수를 줄이려는 시도가 있었다.

그러나 이러한 Deep-KWS 계열의 방식들은 모두 **키워드 특화 데이터(keyword-specific speech data)**에 과도하게 의존한다. 반면, RNN 기반의 KWS는 성능은 좋으나 계산 비용이 높은 bidirectional LSTM 등을 사용하며, 모델링 대상이 되는 키워드 수가 제한적이라는 한계가 있다. 본 논문의 CTC-KWS는 일반 코퍼스를 활용함으로써 데이터 부족 문제를 해결하고, 연산 비용을 DNN 수준으로 유지하며 커스텀 키워드 지원이라는 유연성을 제공한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인
시스템은 크게 세 가지 모듈로 구성된다: **특징 추출(Feature Extraction) $\rightarrow$ 심층 신경망(DNN) $\rightarrow$ 연결주의 시간 분류기(CTC)**.

### 1. 특징 추출 및 모델링 단위
- **특징 추출:** 25ms 윈도우, 10ms 이동 간격으로 40차원의 log filter bank energies를 생성한다. 지연 시간(latency)을 줄이기 위해 비대칭 컨텍스트(과거 10프레임, 미래 5프레임)를 사용하며, 평균 및 분산 정규화를 적용한다.
- **모델링 단위:** 중국어의 CI(Context Independent) 음소 72개와 CTC를 위한 blank 유닛 1개를 포함하여 총 73개의 단위를 사용한다.

### 2. Deep Neural Network (DNN)
DNN은 입력된 음향 특징 벡터를 음소 단위의 사후 확률로 매핑하는 역할을 한다. 각 층의 연산은 다음과 같다.

$$x_i = \sigma(\hat{W}_i^T x_{i-1} + B_i), \quad 1 \le i \le N$$
$$z = \hat{W}_{N+1}^T x_N + B_{N+1}$$
$$y_j = \frac{\exp(z_j)}{\sum_k \exp(z_k)}$$

여기서 $\sigma$는 ReLU 활성화 함수이며, 마지막 층의 Softmax 함수 결과인 $y_j$는 음소 $j$에 대한 예측 확률(posterior)을 의미한다. 연산 효율을 위해 각 은닉층의 노드 수는 수백 개 수준으로 제한하였다.

### 3. Connectionist Temporal Classifier (CTC)
CTC는 입력 신호와 라벨 시퀀스 간의 강제 정렬(forced alignment) 없이도 자동으로 정렬을 학습하는 시퀀스 라벨링 기법이다.

- **Blank 유닛:** 확신이 없는 프레임에서 예측을 유보하기 위해 blank($-$) 유닛을 도입한다.
- **매핑 함수 $\tau$:** CTC 경로 $\pi$에서 반복되는 라벨을 제거하고 blank를 삭제하여 최종 라벨 시퀀스 $l$로 변환하는 함수이다. (예: $\tau(\text{"aa-b-c"}) = \text{"abc"}$)
- **확률 계산:** 각 타임 스텝의 확률이 독립적이라고 가정할 때, 경로 $\pi$의 확률은 다음과 같다.
  $$p(\pi|x; \theta) = \prod_{t=0}^{T-1} y_{t, \pi_t}$$
- **라벨 가능도:** 특정 라벨 $l$이 나올 확률은 $l$로 매핑되는 모든 경로 $\pi$의 확률 합으로 계산한다.
  $$p(l|x; \theta) = \sum_{\pi \in \tau^{-1}(l)} p(\pi|x; \theta)$$
- **학습 목표:** 다음의 음의 로그 가능도(negative log-likelihood)를 최소화하는 $\theta$를 찾는다.
  $$\theta^* = \text{argmin}_\theta \sum_{(x,l) \in S} -\log(p(l|x; \theta))$$

### 4. 추론 및 배포
- **결정 메커니즘:** 일정 윈도우(100프레임) 내에서 $\log(p(l|x; \theta^*))$ 값이 미리 정의된 임계값(threshold)보다 크면 키워드가 검출된 것으로 판단한다.
- **최적화:** 모바일 기기 배포를 위해 ReLU를 사용하여 연산을 가속하고, ARM의 NEON 벡터 명령어를 사용하여 가산 및 승산 연산을 최적화하였다.

## 📊 Results

### 실험 설정
- **데이터셋:** 2,500시간 분량의 중국어 LVCSR 코퍼스를 사용하여 DNN/CTC를 학습시켰다.
- **비교 대상:** 키워드 특화 데이터를 사용하는 Deep-KWS를 베이스라인으로 설정하였다.
- **평가 지표:** 수정된 ROC 커브를 사용하며, Y축은 False Reject Rate(FRR), X축은 False Alarm Rate(FAR)로 설정하여 곡선이 낮을수록 성능이 좋은 것으로 간주한다.

### 정량적 결과
- **성능 비교:** CTC-KWS는 모든 파라미터 규모에서 Deep-KWS보다 우수한 성능을 보였다. 특히 모델의 깊이나 노드 수를 늘렸을 때, Deep-KWS는 데이터 부족으로 인해 성능이 저하되었으나, CTC-KWS는 안정적으로 성능이 유지되거나 향상되었다.
- **적응형 학습(Adaptive Training):** 일반 모델을 소량의 키워드 특화 데이터로 미세 조정(fine-tuning)했을 때 성능이 추가로 향상되었으며, 특히 낮은 FAR(1.5%) 지점에서 큰 이득을 보였다.
- **실시간 성능 (RTF):** 4개 층, 각 256개 노드 모델 기준, ARM A8(1GHz)에서 $0.2218$, MIPS(1GHz)에서 $0.3$의 Real Time Factor를 기록하여 모바일 환경에서 충분히 실시간 동작이 가능함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 KWS 시스템에서 가장 큰 병목이었던 '키워드 특화 데이터의 부족' 문제를 CTC의 시퀀스 라벨링 능력을 통해 효과적으로 해결하였다. DNN이 음향 특징을 음소 수준의 추상적 표현으로 인코딩하고, CTC가 이를 시퀀스로 엮어 신뢰도를 계산하는 구조는 데이터 효율성을 극대화한다.

특히, 모델 규모를 키웠을 때 Deep-KWS는 과적합으로 인해 성능이 떨어지는 반면, CTC-KWS는 일반 코퍼스를 통해 풍부한 음향 정보를 학습했기에 더 큰 모델의 용량을 충분히 활용할 수 있다는 점이 인상적이다.

다만, 본 연구는 정제된 코퍼스 환경에서의 성능 검증에 집중되어 있으며, 실제 환경에서 발생할 수 있는 소음(noise)이나 원거리 음성 인식(far-field) 환경에서의 강건성(robustness)에 대해서는 명시적인 실험 결과가 제시되지 않았다. 이는 향후 연구 과제로 남아 있다.

## 📌 TL;DR

이 논문은 특정 키워드 학습 데이터 없이 **일반 LVCSR 코퍼스**만을 활용하여 모바일 기기에서 동작하는 **DNN-CTC 기반의 키워드 검출(KWS) 시스템**을 제안한다. DNN으로 음소를 예측하고 CTC로 시퀀스 신뢰도를 계산함으로써, 연산 비용을 낮게 유지하면서도 사용자 정의 키워드를 유연하게 지원하고 기존 Deep-KWS보다 뛰어난 성능을 달성하였다. 이 연구는 데이터 수집 비용을 획기적으로 줄이면서도 고성능의 맞춤형 음성 명령 시스템을 구축할 수 있는 가능성을 제시한다.