# Small Footprint Multi-channel ConvMixer for Keyword Spotting with Centroid Based Awareness

Dianwen Ng, Jin Hui Pang, Yang Xiao, Biao Tian, Qiang Fu, Eng Siong Chng (2022)

## 🧩 Problem to Solve

본 논문은 온디바이스(on-device) 환경에서 동작하는 키워드 검출(Keyword Spotting, KWS) 모델의 효율성과 강건성을 동시에 확보하는 문제를 해결하고자 한다. KWS 모델은 전력과 연산 자원이 제한된 기기에서 실행되어야 하므로 작은 모델 크기(small footprint)가 필수적이다. 그러나 모델 크기를 줄이면서 기존의 최신 성능(SOTA)을 유지하는 것은 매우 어려운 과제이다.

특히, 다수의 신호 간섭이 발생하는 원거리(far-field) 및 소음 환경에서는 음성 신호의 품질이 저하되어 인식 정확도가 급격히 떨어진다. 기존의 단일 채널(single-channel) 접근 방식이나 단순한 전처리 방식으로는 이러한 원거리 소음 환경의 복잡한 특성을 극복하는 데 한계가 있으며, 이를 해결하기 위한 다채널(multi-channel) 시스템은 대개 연산 복잡도가 높아 소형 모델에 적용하기 어렵다는 문제가 있다. 따라서 본 연구의 목표는 연산 효율성을 유지하면서도 다채널 오디오의 공간적 특성을 활용하여 소음에 강건한 소형 KWS 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 연산 효율적인 다채널 특징 추출 구조와 잠재 공간(latent space)의 기하학적 정보를 활용한 인식률 향상에 있다.

첫째, **Multi-channel ConvMixer** 아키텍처를 제안한다. 이는 기존의 Attention 메커니즘이 가지는 높은 연산 비용을 줄이기 위해 Convolution-mixer 모듈을 도입한 것이다. 특히 시간(temporal), 주파수(frequency) 영역뿐만 아니라 오디오 채널(audio channel) 간의 상호작용을 수행하는 '채널 믹싱' 단계를 추가하여, 적은 연산량으로도 다채널 입력 데이터에서 노이즈에 강건한 특징을 추출할 수 있게 한다.

둘째, **Centroid Based Awareness** 컴포넌트를 제안한다. 이는 모델이 단순히 특징을 추출하는 것을 넘어, 잠재 특징 투영 공간 내에서 클래스별 중심점(centroid)과의 거리 정보를 활용하도록 하는 설계이다. 이를 통해 네트워크에 공간적 유도 편향(spatial inductive bias)을 제공함으로써, 소음으로 인해 왜곡된 오디오 입력에 대해서도 더 안정적이고 확신 있는 예측을 수행할 수 있도록 돕는다.

## 📎 Related Works

기존의 소형 KWS 연구들은 주로 깨끗한 환경이나 근접 대화 환경에서 우수한 성능을 보였으나, 원거리 환경에서는 성능이 크게 저하되는 경향이 있었다. 이를 해결하기 위해 배경 소음을 조건으로 주는 multi-conditioning이나, 전단(front-end)에서 신호를 정제하는 speech enhancement 기법들이 사용되었다. 하지만 multi-conditioning은 광범위한 소음 환경에 적응하기 어렵고, 단일 채널 기반의 enhancement는 원거리 환경의 잔향(reverberation)과 다중 간섭 신호로 인해 스펙트럼 정보가 뭉개지는 한계가 있다.

다채널 시스템은 빔포밍(beamforming)이나 소음 억제 기술을 통해 이러한 문제를 완화해 왔으며, 최근에는 신경망 기반의 빔포밍과 음향 모델링을 결합한 연구들이 등장했다. 그러나 이러한 다채널 신경망 구조는 대개 메모리 사용량이 많아 소형 모델(small footprint)에 적용하는 사례가 적었다. 본 논문은 이러한 배경에서 Attention 대신 효율적인 ConvMixer 구조를 채택하여 소형 모델에서도 다채널의 이점을 누릴 수 있도록 차별화하였다.

## 🛠️ Methodology

### 1. Multi-channel ConvMixer 구조
제안된 모델은 여러 개의 독립적인 단일 채널 형태의 ConvMixer로 구성되며, 전체 파이프라인은 'Convolutional Encoder $\rightarrow$ Multi-channel ConvMixer Block $\rightarrow$ Post-convolutional Block' 순으로 이어진다. raw 마이크로폰 배열 입력은 먼저 스펙트로그램으로 변환되어 인코더로 전달된다.

핵심인 ConvMixer 블록은 시간, 주파수, 그리고 오디오 채널의 세 가지 차원에서 순차적으로 믹싱을 수행한다. 각 단계의 연산은 다음과 같은 잔차 연결(residual connection) 구조를 가진다.

- **Temporal Mixing**: 모든 주파수 $f$에 대해 공유되는 가중치를 사용하여 시간축의 정보를 섞는다.
  $$u_{*,t,*} = x_{*,i,*} + W_2 \cdot \delta[W_1 \cdot \text{LayerNorm}(x)_{*,t,*}]$$
- **Frequency Mixing**: 모든 시간 $t$에 대해 공유되는 가중치를 사용하여 주파수축의 정보를 섞는다.
  $$y_{f,*,*} = u_{f,*,*} + W_4 \cdot \delta[W_3 \cdot \text{LayerNorm}(u)_{f,*,*}]$$
- **Audio Channel Mixing**: 모든 채널 $c$에 대해 공유되는 가중치를 사용하여 마이크로폰 채널 간의 상호작용을 수행한다.
  $$z_{*,*,c} = y_{*,*,c} + W_6 \cdot \delta[W_5 \cdot \text{LayerNorm}(y)_{*,*,c}]$$

여기서 $\delta$는 GELU 활성화 함수를 의미하며, $W_1 \sim W_6$는 학습 가능한 선형 층의 가중치이다. 이 과정을 $N=4$회 반복하여 특징을 더욱 정교화한 뒤, 최종적으로 모든 채널의 특징을 통합하여 $D$-차원의 잠재 벡터 $X_{feat}$를 생성한다.

### 2. Centroid Based Awareness
모델이 학습한 잠재 공간에서 각 키워드 클래스의 중심점(centroid)과 입력 벡터 사이의 거리를 측정하여 예측기에 추가 정보로 제공하는 방식이다.

- **수학적 근거**: 교차 엔트로피 손실 함수 $H(q, \hat{q})$를 Bregman divergence 기반으로 분해하면 Bias와 Variance 항으로 나눌 수 있다. 본 연구에서는 $L_2$-norm 거리를 통해 공간적 유도 편향을 추가함으로써 $\bar{q}$와 $\hat{q}$ 사이의 발산을 줄여 모델의 추정 편향과 분산을 감소시키고자 한다.
- **예측 헤드 구성**: 최종 예측 확률 $\hat{q}$는 다음과 같이 특징 벡터와 거리 정보의 결합으로 계산된다.
  $$\hat{q} \propto \exp(W_{feat}X_{feat} + W_{L^2\text{-norm}}X_{L^2\text{-norm}})$$
  여기서 $X_{L^2\text{-norm}}$은 $X_{feat}$와 각 클래스 중심점 간의 $L_2$-norm 유클리드 거리이다.
- **중심점 학습**: 모든 학습 데이터를 매번 계산하는 것은 비용이 크므로, 학습 가능한 임베딩 벡터 $V$를 설정하고 Stochastic Gradient Descent(SGD)를 통해 각 클래스 내부의 평균 제곱 오차(MSE)를 최소화하는 방향으로 중심점을 업데이트한다.

## 📊 Results

### 실험 설정
- **데이터셋**: MISP Challenge 2021 (Task 1)을 사용하였다. 3-5m 거리의 원거리 환경에서 6채널 마이크로폰 배열을 통해 "Xiao T Xiao T"라는 키워드를 검출하는 작업이다.
- **비교 대상**: 공식 Baseline(CNN-LSTM, 2.68M 파라미터), 단일 채널 ConvMixer(124K), MVDR 빔포머 적용 모델 등을 비교하였다.
- **평가 지표**: Accuracy와 함께 오경보율(FAR)과 오거부율(FRR)의 합인 $\text{Score} = \text{FAR} + \text{FRR}$를 사용하였다. (Score가 낮을수록 우수)

### 주요 결과
- **정량적 성능**: 
    - 공식 Baseline(Score 0.344) 대비 제안 모델(Centroid Awareness ConvMixer, Score 0.152)은 약 **55%의 Score 개선**과 **10%의 정확도 향상**을 달성하였다.
    - 특히, 단일 채널 ConvMixer(Score 0.177)보다 다채널 ConvMixer(Score 0.161)가 우수했으며, 여기에 Centroid Awareness를 추가했을 때 가장 좋은 성능을 보였다.
    - 파라미터 수 측면에서도 Baseline(2.68M)보다 훨씬 작은 622K 수준에서 더 높은 성능을 기록하였다.
- **전처리 결합 효과**: 전단에 WPE(Weighted Prediction Error) 역잔향 처리와 3-look MVDR 빔포머를 결합했을 때, Score가 0.126까지 낮아지며 Baseline 대비 **63%의 개선**을 보였다.

## 🧠 Insights & Discussion

본 논문은 소형 모델이 원거리 다채널 데이터에서 겪는 가장 큰 어려움이 '효율적인 공간 필터링'과 '강건한 특징 추출'의 부재라는 점을 정확히 짚어냈다. 

**강점 및 해석**: 
- ConvMixer를 통해 Attention의 무거운 연산을 대체하면서도 채널 간 상호작용을 구현함으로써, 연산 효율성과 다채널 활용 능력을 동시에 잡았다.
- 특히 Centroid Based Awareness는 단순히 신경망의 깊이를 늘리는 것이 아니라, 거리 기반의 기하학적 정보를 명시적으로 주입함으로써 클래스 간 분별력을 높였다. 이는 시각화된 잠재 표현 분포에서도 클래스 간 겹침(overlapping)이 줄어드는 것으로 확인되었다.

**한계 및 논의**:
- 본 연구에서 사용한 MVDR 빔포머는 선형적인 처리 방식이므로, 향후 비선형 신경망 기반의 enhancement 기법을 통합한다면 더 큰 성능 향상이 있을 것으로 기대된다.
- WPE 단독 적용 시에는 오히려 성능이 저하되는 모습이 관찰되었는데, 이는 맹목적인 역잔향 처리가 신호에 왜곡을 일으킬 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 원거리 소음 환경에서 동작하는 초소형 KWS 모델을 위해 **다채널 ConvMixer** 구조와 **중심점 기반 인식(Centroid Based Awareness)** 기법을 제안하였다. 연산 효율적인 믹싱 구조로 다채널 특징을 추출하고, 잠재 공간 내 클래스 중심점과의 거리 정보를 추가하여 인식의 안정성을 높였다. 그 결과, 기존 Baseline 대비 모델 크기는 크게 줄이면서도 성능(Score)은 최대 63% 향상시키는 성과를 거두었으며, 이는 저전력 온디바이스 음성 인식 시스템 구현에 중요한 기여를 할 것으로 보인다.