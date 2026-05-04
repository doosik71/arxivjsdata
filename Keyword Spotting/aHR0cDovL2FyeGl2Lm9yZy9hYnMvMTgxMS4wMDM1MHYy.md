# END-TO-END MODELS WITH AUDITORY ATTENTION IN MULTI-CHANNEL KEYWORD SPOTTING

Haitong Zhang, Junbo Zhang, Yujun Wang (2018)

## 🧩 Problem to Solve

본 논문은 다채널(Multi-channel) 입력 환경에서의 키워드 검출(Keyword Spotting, KWS) 문제를 해결하고자 한다. KWS는 연속적인 음성 스트림에서 미리 정의된 특정 키워드를 탐지하는 작업으로, 주로 모바일 기기의 웨이크업 워드(wake-up word) 기능으로 사용된다. 따라서 높은 정확도, 낮은 지연 시간(low-latency), 그리고 작은 모델 크기(small-footprint)라는 요구 조건을 충족해야 한다.

기존의 다채널 접근 방식은 빔포밍(Beamforming)이나 음향 에코 제거(Acoustic Echo Cancellation, AEC)와 같은 신호 전처리 기술을 사용하여 다채널 신호를 단일 채널 신호로 변환한 뒤 KWS 모델에 입력하는 방식을 취했다. 그러나 이러한 전처리 단계는 최종 목표인 KWS 결과에 최적화되어 설계된 것이 아니기 때문에, 전처리 과정에서 정보 손실이 발생하거나 최적의 성능을 내지 못하는 sub-optimal한 문제가 발생한다. 본 연구의 목표는 다채널 입력을 직접 처리하여 KWS 결과를 직접 최적화하는 attention 기반의 end-to-end 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다채널 KWS를 위한 attention 기반의 end-to-end 모델을 제안하고, 특히 소음 환경에서의 강건성(robustness)을 높이기 위해 다음과 같은 설계 아이디어를 도입한 것이다.

1. **Auditory Attention Mechanism**: 여러 채널의 입력 중 중요한 정보에 집중하여 가중합을 구하는 soft attention 메커니즘을 도입하여, 연산 비용을 낮추면서도 효율적으로 다채널 특징을 통합하였다.
2. **Multi-task Learning 기반 Spectral Mapping**: 다채널 특징을 단일 채널 특징으로 매핑하는 보조 작업(auxiliary task)을 통해 모델이 더 유용한 표현을 학습하도록 유도하였다.
3. **Transfer Learning 및 Multi-target Spectral Mapping**: 깨끗한 데이터로 사전 학습 후 소음 데이터로 미세 조정(fine-tuning)하는 전이 학습과, 다양한 수준의 소음 타겟을 학습하는 다중 타겟 매핑을 통해 극심한 소음 환경에서의 성능을 대폭 향상시켰다.

## 📎 Related Works

기존의 KWS 연구는 주로 다음과 같은 방향으로 진행되었다.

- **LVCSR 기반 방식**: 대규모 어휘 연속 음성 인식 시스템을 사용하며, 오프라인 처리에 적합하나 지연 시간이 길어 모바일 기기에는 부적합하다.
- **HMM 기반 방식**: 키워드와 non-keyword 세그먼트를 각각 학습하며, 런타임에 Viterbi 탐색이 필요하여 계산 비용이 많이 든다.
- **Deep Learning 기반 방식**: DNN, CNN, RNN 등을 사용하여 키워드의 사후 확률(posterior probability)을 출력하며, 최근에는 sub-word 단위가 아닌 전체 키워드를 직접 출력하는 end-to-end 모델이 제안되었다.

하지만 위의 연구들은 대부분 단일 채널(single-channel) KWS에 집중되어 있다. 다채널 입력의 경우 기존에는 전처리-인식의 분리된 구조를 사용했으나, 본 논문은 이를 통합한 end-to-end 구조를 제안함으로써 기존 방식의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

제안된 모델은 크게 **Attention Mechanism $\rightarrow$ Encoder $\rightarrow$ Decoding**의 세 단계로 구성된다. 다채널 입력 특징이 들어오면 attention mechanism이 이를 단일 벡터로 통합하고, GRU 기반의 encoder가 고차원 표현을 학습하며, 마지막으로 softmax를 통해 키워드 존재 확률을 예측한다.

### 2. Attention Mechanism

본 모델은 soft attention을 사용하여 각 타임스텝 $t$마다 6개의 채널에 대한 가중치를 계산한다. 입력 특징 행렬을 $x_{ch}^t$라고 할 때, attention 가중치 $A_{ch}^t$는 다음과 같이 계산된다.

$$A_{ch}^t = \text{softmax}(\max(V^t * \tanh(Wx_{ch}^t + b)))$$

여기서 $x_{ch}^t$는 $6 \times 40$ 크기의 입력 특징 행렬이며, $W$와 $b$는 학습 가능한 가중치와 편향이다. 이후 가중치가 적용된 통합 특징 $x'_t$는 다음과 같이 계산된다.

$$x'_t = \sum_{c=1}^{ch} A_{jt} x_{jt}$$

### 3. Sequence-to-Sequence Training 및 Encoder

통합된 특징 $x'_t$는 Encoder로 전달된다. Encoder는 2개의 GRU(Gated Recurrent Units) 층과 1개의 Fully-connected(FC) 층으로 구성되며, 각 층은 128개의 유닛을 가진다. 최종적으로 linear transformation과 softmax 함수를 통해 매 프레임마다 전체 키워드의 발생 확률을 예측한다.

### 4. Multi-task 및 Transfer Learning

- **Multi-task Learning**: KWS 작업과 함께 다채널 특징을 단일 채널 특징으로 변환하는 spectral mapping을 보조 작업으로 수행한다. 손실 함수는 다음과 같다.
  $$\text{Loss}_{total} = \alpha \cdot \text{Loss}_{KWS} + (1-\alpha)\text{Loss}_{Map}^{clean}$$
- **Multi-target Mapping**: 소음 환경에서의 강건성을 위해 단일 타겟이 아닌 여러 수준의 소음 타겟(Target 1, 2, 3)을 동시에 학습한다. 이때의 손실 함수는 다음과 같다.
  $$\text{Loss}_{total} = \alpha \cdot \text{Loss}_{KWS} + \beta \cdot \text{Loss}_{Map}^{clean} + \theta \cdot \text{Loss}_{Map}^{noise1} + \delta \cdot \text{Loss}_{Map}^{noise2}$$
  (단, $\alpha + \beta + \theta + \delta = 1$)
- **Transfer Learning**: 깨끗한 데이터로 학습된 모델의 파라미터를 초기값으로 사용하여 소음 데이터로 미세 조정을 수행한다.

### 5. Decoding

추론 시에는 매 프레임의 확률값을 출력하며, 사후 확률 평활화(posterior probability smoothing) 방법을 적용한다. 최종 결정은 $n$개 프레임의 평균 확률을 기반으로 내려진다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 키워드 발화 240k 개, negative examples 200시간. 테스트 데이터는 filler 데이터 50시간 및 키워드 데이터 48k 개를 포함한다.
- **소음 데이터**: SNR 약 $-20\text{dB}$(hard-noisy)와 $-18\text{dB}$(easy-noisy)의 환경을 구축하여 테스트하였다.
- **기준선(Baseline)**: 빔포밍(Beamforming) 및 AEC 전처리를 거친 단일 채널 KWS 모델을 사용하였다.
- **지표**: 시간당 오경보 횟수(False Alarm per hour, FA) 대비 웨이크업 확률(Wake-up rate)을 측정하였다.

### 2. 주요 결과

- **Attention의 효과**: 제안된 Attention 모델은 모든 테스트 데이터에서 baseline보다 우수한 성능을 보였다. 특히 소음 데이터에서 성능 향상이 두드러졌는데, hard-noisy 데이터에서 baseline 대비 각각 $40\%$ 및 $60\%$의 성능 향상을 기록하였다.
- **Multi-task Learning**: spectral mapping을 추가한 모델(Mapping)은 학습 데이터와 테스트 데이터가 유사한 경우 attention 단독 모델보다 약간 높은 성능을 보였으나, 소음 데이터에서는 오히려 성능이 하락하는 경향을 보였다.
- **Transfer 및 Multi-target Mapping**: 전이 학습과 다중 타겟 매핑을 결합한 `TranMultiMap` 모델은 소음 환경에서 압도적인 성능을 보였다. $0.5\text{ FA/hr}$ 기준, hard-noisy 데이터에서 Attention 모델 대비 절대적으로 $30\%$의 웨이크업 확률 향상을 달성하였다.

## 🧠 Insights & Discussion

본 논문은 전통적인 신호 처리 기반의 전처리 방식이 KWS라는 최종 목적함수와 분리되어 있어 최적의 성능을 내기 어렵다는 점을 지적하고, 이를 end-to-end 구조의 attention 메커니즘으로 해결함으로써 그 효용성을 입증하였다. 특히 attention 기반 모델이 전처리 기반 baseline보다 소음 환경에서 훨씬 강건하다는 결과는, 신경망 기반의 특징 통합이 전통적인 빔포밍보다 복잡한 소음 환경을 더 잘 처리할 수 있음을 시사한다.

다만, 전이 학습을 적용했을 때 깨끗한 데이터에 대한 성능이 약간 하락하는 현상이 관찰되었다. 이는 학습 데이터와 테스트 데이터 간의 분포 차이(mismatch)로 인한 결과로 해석되며, 소음 강건성과 깨끗한 환경에서의 성능 사이의 trade-off가 존재함을 보여준다. 또한, 단순히 단일 타겟으로 매핑하는 것보다 여러 단계의 소음 타겟을 학습시키는 `Multi-target Mapping`이 모델의 수렴과 강건성 향상에 결정적인 역할을 했다는 점은 향후 음성 인식의 전처리 신경망 설계에 중요한 통찰을 제공한다.

## 📌 TL;DR

본 논문은 다채널 입력의 KWS를 위해 전처리 과정 없이 직접 최적화하는 **Attention 기반 end-to-end 모델**을 제안하였다. 특히 **전이 학습(Transfer Learning)**과 **다중 타겟 스펙트럼 매핑(Multi-target Spectral Mapping)**을 결합하여, SNR $-20\text{dB}$의 극심한 소음 환경에서도 웨이크업 확률을 $30\%$ 이상 향상시키는 성과를 거두었다. 이 연구는 다채널 음성 인식에서 신경망 기반의 적응형 특징 통합이 전통적인 신호 처리 방식보다 우수함을 증명하였으며, 향후 소음 강건한 음성 인식 시스템 구축에 중요한 기초가 될 것으로 보인다.
