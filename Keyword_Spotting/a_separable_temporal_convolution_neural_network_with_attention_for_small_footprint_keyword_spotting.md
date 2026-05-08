# A Separable Temporal Convolution Neural Network with Attention for Small-Footprint Keyword Spotting

Shenghua Hu, Jing Wang, Yujun Wang, Lidong Yang, Wenjing Yang (2020)

## 🧩 Problem to Solve

본 논문은 모바일 기기에서 동작하는 소형 footprint(small-footprint) Keyword Spotting (KWS) 시스템의 효율성 문제를 해결하고자 한다. KWS는 오디오 신호에서 미리 정의된 특정 키워드를 탐지하는 작업으로, 모바일 기기의 핸즈프리 제어(예: Siri, Alexa, Google Assistant)에 널리 사용된다.

이 문제의 핵심 중요성은 모바일 기기가 가진 제한된 하드웨어 자원(메모리 및 계산 능력)에 있다. 기존의 고성능 KWS 모델들은 높은 정확도를 유지하기 위해 방대한 수의 파라미터를 유지하며, 이는 모바일 기기에서의 실시간 처리 및 오프라인 동작을 어렵게 만든다. 따라서 본 논문의 목표는 모델의 파라미터 수를 획기적으로 줄이면서도 최신 모델(State-of-the-art) 수준의 높은 정확도를 유지하는 경량화된 신경망 구조를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 **Separable Temporal Convolution**과 **Attention 메커니즘**을 결합하여, 계산 복잡도를 낮추면서도 수용 영역(Receptive Field)을 효율적으로 확장하는 것이다.

주요 기여 사항은 다음과 같다.

1. **ST-Conv 구조 제안**: 주파수 영역의 정보를 먼저 추출한 후 시간 축으로 수용 영역을 확장하는 Separable Temporal Convolution 구조를 제안하여 파라미터 수를 크게 줄였다.
2. **글로벌 특징 캡처**: Bidirectional Gated Recurrent Unit (BGRU)와 Shared Weight Self-Attention (SWSA)을 도입하여 음성 신호의 장기 의존성과 핵심 정보를 효과적으로 추출하였다.
3. **초경량 모델 구현**: 약 32.2K의 파라미터만으로 기존의 Res15(239K)와 유사한 95.7%의 정확도를 달성하여, 매우 작은 footprint로도 고성능 KWS가 가능함을 입증하였다.

## 📎 Related Works

기존의 KWS 접근 방식은 크게 두 가지 흐름으로 나뉜다. 하나는 대규모 어휘 연속 음성 인식(LVCSR) 기반이며, 다른 하나는 은닉 마르코프 모델(HMMs) 기반이다. 그러나 이 방법들은 메모리 비용과 계산량이 너무 커서 모바일 기기에 적용하기 어렵다는 한계가 있다.

최근에는 심층 신경망, 특히 Convolutional Neural Network (CNN) 기반의 KWS가 주류를 이루고 있다. 예를 들어, ResNet 기반 모델은 잔차 연결(Residual Connection)을 통해 층을 깊게 쌓아 높은 정확도를 얻었으나, 여전히 파라미터 수가 많아 무겁다는 단점이 있다. 또한, Depthwise Separable Convolution (DSCNN)은 파라미터를 줄이는 효과적인 방법이지만, 충분한 수용 영역을 확보하기 위해서는 많은 수의 은닉층이 필요하게 되어 결국 다시 파라미터가 증가하는 모순이 발생한다. Att-RNN과 같은 모델은 Average Pooling 대신 Attention을 사용하여 정확도를 높였으나, 여전히 CNN과 BLSTM 구조가 비대하여 경량화 측면에서 한계가 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인

본 모델은 MFCC(Mel-frequency cepstral coefficients) 특징을 입력으로 받아 **[Separable Temporal Convolution $\rightarrow$ BGRU $\rightarrow$ SWSA $\rightarrow$ FC $\rightarrow$ Softmax]** 순으로 처리하는 구조를 가진다.

### 2. 주요 구성 요소 및 상세 설명

#### (1) Separable Temporal Convolution

전통적인 2D CNN은 이미지 처리에는 유리하지만, 음성 시퀀스 작업에서는 수용 영역을 확장하는 비용이 너무 크다. 이를 해결하기 위해 본 논문은 다음과 같은 전략을 사용한다.

- **첫 번째 레이어**: $1 \times 40$ 크기의 커널을 사용하여 MFCC의 주파수 영역 정보를 먼저 추출한다. 이를 통해 이후 레이어들이 모든 주파수 정보를 활용할 수 있게 하며, 이후 커널 크기를 $3 \times 1$로 줄여 파라미터를 $2/3$만큼 절감한다.
- **Dilated Convolution**: 수용 영역을 효율적으로 넓히기 위해 지수적 크기 일정(exponential sizing schedule)을 적용한 Atrous Convolution을 사용한다. $i$번째 레이어의 dilation rate $d$는 다음과 같다.
  $$d = \lfloor \frac{3^i}{2} \rfloor$$
- **Depthwise Separable Convolution**: Depthwise 및 Pointwise convolution으로 분해하여 채널 수가 많을 때의 계산량을 획기적으로 줄인다.

#### (2) BGRU (Bidirectional Gated Recurrent Unit)

CNN 단계에서 추출된 특징 벡터들을 BGRU에 입력하여 음성 신호의 전역적인 문맥과 장기 의존성(Long-term dependence)을 캡처한다.

#### (3) SWSA (Shared Weight Self-Attention)

BGRU의 출력에서 가장 중요한 부분에 집중하기 위해 공유 가중치 셀프 어텐션을 사용한다.

- **Query ($q$)**: BGRU 출력의 49번째 벡터(중간 지점)를 가중치 행렬 $w$와 곱하여 생성한다.
- **Key & Value ($u$)**: BGRU의 전체 출력 시퀀스를 동일한 가중치 행렬 $w$와 곱하여 생성한다.
- **작동 원리**: Query를 이용해 전체 시퀀스 중 어떤 부분이 가장 관련이 깊은지 결정하고, 이를 통해 최종적으로 압축된 특징 벡터를 추출한다.

### 3. 훈련 절차 및 손실 함수

- **손실 함수**: Cross Entropy loss를 사용한다.
- **최적화 알고리즘**: Adam optimizer를 사용하며, 초기 학습률은 0.001로 설정한다.
- **학습 전략**: 검증 세트(Development set)의 손실이 3% 이상 감소하지 않을 경우 학습률을 60%로 낮추는 스케줄링과 Early Stopping 전략을 적용하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Google Speech Commands V1 (총 64,752개 녹음, 30개 단어 중 10개 키워드와 20개 필러로 구성).
- **입력 데이터**: 40차원 MFCC (프레임 길이 25ms, 시프트 10ms, 총 99프레임).
- **평가 지표**: Test Accuracy 및 ROC curve (AUC 기반).

### 2. 정량적 결과 (Table 3 참조)

제안 모델인 ST-Conv는 기존 모델들과 비교하여 다음과 같은 성능을 보였다.

| 모델 | 테스트 정확도 | 파라미터 수 | 곱셈 연산량 (Mult.) |
| :--- | :---: | :---: | :---: |
| Res15 (SOTA) | $95.8\% \pm 0.256$ | 239K | 894M |
| TC-ResNet14 | $95.7\% \pm 0.324$ | 137K | 3.26M |
| **ST-Conv (제안)** | $\mathbf{95.7\% \pm 0.295}$ | $\mathbf{31K}$ | $\mathbf{3.09M}$ |

- **정확도**: Res15 및 TC-ResNet14와 거의 동일한 수준의 정확도($95.7\%$)를 달성하였다.
- **경량성**: 파라미터 수는 Res15의 약 1/8 수준이며, TC-ResNet14와 비교해서도 약 1/4 수준으로 매우 작다.

### 3. 어블레이션 연구 (Ablation Study)

- **ST-Conv-Narrow**: 채널 수를 40에서 20으로 줄였을 때, 파라미터는 2/3 감소하지만 정확도가 $1.5\%$ 하락하여 적절한 채널 크기 유지의 중요성을 보여주었다.
- **ST-Conv-Avg**: SWSA를 Average Pooling으로 대체했을 때 정확도가 $0.8\%$ 하락하였다. 이는 단순 평균보다 Attention 메커니즘이 중요한 음성 특징을 잡는 데 더 효과적임을 입증한다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **주파수 도메인 우선 추출 $\rightarrow$ 시간 도메인 확장**이라는 직관적인 설계를 통해 CNN의 고질적인 문제인 '수용 영역 확장 시 파라미터 급증' 문제를 해결했다는 점이다. 특히 $1 \times 40$ 커널을 첫 층에 배치함으로써 이후 층의 커널 크기를 $3 \times 1$로 고정할 수 있게 한 점이 매우 효율적이다.

또한, BGRU 이후에 단순한 풀링이 아닌 Shared Weight Self-Attention을 도입함으로써, 모델의 크기를 거의 늘리지 않으면서도(가중치 공유) 성능을 최적화한 점이 돋보인다.

다만, 본 논문은 고정된 길이(1초)의 오디오 데이터를 처리하는 배치 방식에 집중하고 있다. 실제 환경에서는 실시간 스트리밍 데이터가 입력되는데, 이에 대한 처리 방식(Streaming KWS)에 대해서는 향후 과제로 남겨두어 실제 적용 시의 지연 시간(Latency)이나 버퍼링 전략에 대한 분석이 추가로 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 모바일 기기용 초경량 키워드 탐지(KWS) 모델인 **ST-Conv**를 제안한다. $1 \times 40$ 초기 컨볼루션과 Dilated Separable Temporal Convolution, 그리고 BGRU 및 Shared Weight Self-Attention을 결합하여, **파라미터 수를 32.2K까지 줄이면서도 SOTA 모델(Res15)에 근접한 95.7%의 정확도**를 달성하였다. 이는 하드웨어 자원이 매우 제한된 임베디드 시스템이나 모바일 기기에서의 온디바이스(On-device) AI 구현에 매우 중요한 기여를 할 것으로 평가된다.
