# TF-Mamba: A Time-Frequency Network for Sound Source Localization

Yang Xiao, Rohan Kumar Das (2025)

## 🧩 Problem to Solve

본 논문은 다채널 오디오 데이터를 이용하여 음원의 위치를 추정하는 Sound Source Localization (SSL) 문제를 다룬다. SSL은 음원 분리(speech separation) 및 음성 향상(speech enhancement)의 성능을 높이는 데 필수적인 전처리 단계로 활용된다.

특히, 실제 환경에서는 다음과 같은 요인들로 인해 정확한 음원 위치 추정이 어렵다.

- **소음 및 잔향(Noise and Reverberation):** 외부 소음과 잔향은 신호를 왜곡하거나 마스킹하여 공간적 특성 추출을 방해한다.
- **이동하는 음원(Moving Sources):** 음원이 움직일 경우 공간적 단서(spatial cues)가 시간에 따라 계속 변화하므로, 이를 효과적으로 추적하고 모델링하는 것이 중요하다.

따라서 본 연구의 목표는 시간-주파수 영역의 특성을 모두 효과적으로 융합하고, 특히 긴 시퀀스 데이터에서 장기 의존성(long-range dependencies)을 효율적으로 모델링할 수 있는 새로운 SSL 시스템인 TF-Mamba를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 시퀀스 모델링에서 뛰어난 성능을 보인 **Mamba (State-Space Model, SSM)** 구조를 SSL 태스크에 최초로 도입한 것이다.

중심적인 설계 아이디어는 다음과 같다.

1. **Dual-Dimension Approach:** Mamba를 시간(Time) 축과 주파수(Frequency) 축 모두에 적용하여, 공간적 단서와 그 시간적 진화 과정을 동시에 캡처한다.
2. **Bidirectional Mamba (BiMamba):** 단방향 성질을 가진 기존 Mamba의 한계를 극복하기 위해, 정방향과 역방향 시퀀스를 동시에 처리하는 BiMamba 블록을 사용하여 비인과적(non-causal) 문맥을 학습한다.
3. **TF-Mamba Architecture:** 시간 및 주파수 Mamba 층을 교차 배치하고 Skip Connection을 추가하여 정보 손실을 방지하고 효율적인 특징 추출을 가능하게 한다.

## 📎 Related Works

### 기존 연구 및 한계

- **전통적 방법:** SRP-PHAT과 같이 시간 지연, Inter-channel Phase/Level Difference (IPD/ILD) 등의 공간적 특징을 이용한다. 이는 이상적인 환경에서는 효과적이지만, 소음과 잔향이 심한 실제 환경에서는 성능이 급격히 저하된다.
- **딥러닝 기반 방법:** CNN은 국소적인 공간 특징을, RNN(LSTM, GRU)은 장기적인 시간 문맥을 캡처한다. 최근의 FN-SSL과 같은 모델은 풀밴드(full-band)와 내로우밴드(narrow-band) 도메인을 융합하여 성능을 높였으나, 여전히 RNN 계열의 계산 복잡도와 장기 의존성 모델링의 한계가 존재한다.

### 차별점

본 제안 방법은 Transformer 수준의 성능을 내면서도 계산 복잡도가 선형적으로 증가하는 Mamba 구조를 채택하였다. 특히, 기존의 음성 처리 Mamba 연구들이 주로 음성 향상이나 분리에 집중했던 것과 달리, 이를 SSL 분야에 적용하여 시간-주파수 차원의 융합 모델을 구축했다는 점에서 독창성을 가진다.

## 🛠️ Methodology

### 1. BiMamba (Bidirectional Mamba)

Mamba의 핵심인 선형 선택적 SSM(Linear Selective SSM)은 입력 $x_t$를 은닉 상태 $h_t$를 거쳐 출력 $y_t$로 변환한다. 연속 시간 SSM을 이산화(discretization)하여 다음과 같은 상태 방정식으로 표현한다.

$$h_t = \tilde{A}h_{t-1} + \tilde{B}x_t, \quad y_t = Ch_t$$

여기서 $\tilde{A}$와 $\tilde{B}$는 학습 가능한 행렬이다. 이 구조는 다음과 같은 컨볼루션(convolution) 형태로도 표현 가능하여 효율적인 처리가 가능하다.

$$y = x * K, \quad \text{where } K = (C\tilde{B}, C\tilde{A}\tilde{B}, \dots, C\tilde{A}^{L-1}\tilde{B})$$

하지만 Mamba는 기본적으로 단방향(unidirectional) 모델이므로, 과거와 미래의 문맥을 모두 고려해야 하는 SSL 태스크를 위해 **BiMamba**를 사용한다. 이는 원본 시퀀스와 역순 시퀀스를 각각의 SSM으로 처리한 후 그 결과를 평균 내어 통합하는 구조이다.

### 2. 전체 네트워크 아키텍처

TF-Mamba는 크게 세 가지 모듈로 구성된다.

#### (1) Feature Encoder

- 입력값으로 STFT 계수의 실수부와 허수부를 사용하며, 입력 채널 수는 (마이크 수 $\times 2$)가 된다.
- Dilated DenseNet 코어와 두 개의 Convolutional 레이어를 통해 스펙트럼 특성을 강화한 특징 맵을 생성한다.

#### (2) Time-Frequency Mamba (TF-Mamba Block)

- **Temporal Mamba (T-BiMamba):** 단일 시간 프레임 내의 주파수 축을 따라 시퀀스를 처리한다. 이를 통해 공간 및 위치 단서와 관련된 주파수 간의 의존성(inter-frequency dependencies)을 학습한다.
- **Frequency Mamba (F-BiMamba):** 단일 주파수 성분에 대해 시간 축을 따라 시퀀스를 처리한다. 이는 직접 경로(direct-path) 위치 특징의 시간적 진화(temporal evolution)를 학습하며, 특히 이동하는 음원의 궤적을 파악하는 데 중요하다.
- **Skip Connection:** T-BiMamba와 F-BiMamba 층을 통과할 때 발생할 수 있는 정보 손실을 막기 위해 각 층의 입력을 출력에 더해주는 잔차 연결을 적용한다.

#### (3) Output Decoder

- TF-Mamba 블록의 출력을 Average Pooling을 통해 프레임 레이트를 줄인 후, $\tanh$ 활성화 함수를 가진 Fully Connected 레이어를 통과시킨다.
- 최종적으로 181개 지점의 맵으로 구성된 **Spatial Spectrum**을 예측하며, 정답(Ground Truth)과의 **Mean Squared Error (MSE) Loss**를 최소화하는 방향으로 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋:**
  - **Simulated Data:** LibriSpeech 음성과 gpuRIR로 생성한 방 임펄스 응답(RIR)을 합성하였다. RT60은 0.2~0.6s, 방 크기는 $4\times2\times2\text{m}$에서 $10\times8\times5\text{m}$까지 다양하게 설정하였으며, $-10\text{dB}$에서 $10\text{dB}$ 사이의 SNR을 가진 소음을 추가하였다.
  - **Real-world Data:** LOCATA 데이터셋의 Task 3, 5를 사용하였으며, 시뮬레이션 데이터로 학습된 모델을 직접 테스트하여 일반화 성능을 측정하였다.
- **평가 지표:** Mean Absolute Error (MAE)와 Accuracy (ACC, 오차 범위 $10^\circ$ 및 $15^\circ$ 기준)를 사용하였다.
- **비교 대상:** Cross3D, SELDnet, SE-ResNet, SALSA-Lite, FN-SSL.

### 주요 결과

- **시뮬레이션 데이터:** Clean 환경에서 TF-Mamba는 $\text{ACC}(10^\circ) = 96.9\%$, $\text{MAE} = 2.5^\circ$를 기록하며 모든 베이스라인 모델을 능가하였다. 특히 $\text{SNR} = -10\text{dB}$의 극한 소음 환경에서도 TF-Mamba는 $\text{ACC}(10^\circ) = 72.5\%$를 기록하여, FN-SSL($68.4\%$)이나 SALSA-Lite($65.7\%$)보다 훨씬 강건한 성능을 보였다.
- **실제 데이터 (LOCATA):** 시뮬레이션 데이터로만 학습된 모델을 적용했을 때, TF-Mamba는 $\text{ACC}(10^\circ) = 94.3\%$, $\text{MAE} = 3.2^\circ$로 가장 우수한 성능을 기록하였다.
- **Ablation Study:**
  - 블록 수를 3개에서 5개로 늘렸을 때 $\text{ACC}(10^\circ)$가 $92.6\% \rightarrow 96.9\%$로 향상되어 깊은 네트워크의 효과를 입증하였다.
  - BiMamba를 단방향 Mamba로 대체하거나 Skip Connection을 제거했을 때 성능이 하락하여, 양방향 문맥 파악과 정보 보존의 중요성이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

TF-Mamba는 Mamba의 선형 복잡도와 강력한 시퀀스 모델링 능력을 활용하여, 기존 RNN 기반 모델(FN-SSL 등)보다 장기 의존성을 더 잘 캡처한다. 특히 시간과 주파수라는 두 가지 차원을 독립적으로 처리하고 융합함으로써, 이동하는 음원의 복잡한 공간적 특성을 정밀하게 모델링할 수 있었다. 실제 데이터 실험에서 나타난 'Over-smoothing' 문제의 완화는 TF-Mamba가 급격한 음원 방향 전환 시에도 빠르게 반응할 수 있음을 시사한다.

### 한계 및 논의사항

본 논문은 SSM 기반 모델을 SSL에 적용한 최초의 시도라는 점에서는 의의가 크나, 몇 가지 논의할 점이 있다.

- **계산 효율성의 실제 측정:** Mamba가 이론적으로 선형 복잡도를 가지지만, 실제 추론 시간(Inference Time)이나 메모리 사용량에 대한 정량적 비교 수치가 구체적으로 제시되지 않았다.
- **데이터 의존성:** 시뮬레이션 데이터로 학습하여 실제 데이터에 적용하는 전이 학습(Transfer Learning) 성능은 좋았으나, 실제 데이터로 직접 학습했을 때의 성능 향상 폭에 대해서는 언급되지 않았다.

## 📌 TL;DR

본 논문은 Mamba(State-Space Model) 구조를 활용하여 시간-주파수 도메인의 특징을 효율적으로 융합한 **TF-Mamba** 네트워크를 제안한다. BiMamba를 통해 양방향 문맥을 학습하고 시간/주파수 축을 교차 모델링함으로써, 소음과 잔향이 심한 환경 및 음원이 이동하는 상황에서도 기존 SOTA 모델들을 뛰어넘는 높은 정밀도의 음원 위치 추정(SSL) 성능을 달성하였다. 이는 SSM이 오디오의 공간적 특징 추출 및 실시간 추적 시스템에 매우 효과적인 대안이 될 수 있음을 보여준다.
