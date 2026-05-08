# Multi-task WaveNet: A Multi-task Generative Model for Statistical Parametric Speech Synthesis without Fundamental Frequency Conditions

Yu Gu, Yongguo Kang (2018)

## 🧩 Problem to Solve

본 논문은 통계적 파라미터 음성 합성(Statistical Parametric Speech Synthesis, SPSS)에서 WaveNet 모델을 사용할 때 발생하는 의존성 문제와 그로 인한 품질 저하 문제를 해결하고자 한다.

기존의 WaveNet 기반 SPSS 시스템은 고품질의 음성 파형을 생성하기 위해 언어적 특징(Linguistic features)뿐만 아니라 외부 모델을 통해 예측된 기본 주파수(Fundamental Frequency, $\log F0$) 값을 조건부 입력으로 사용한다. 하지만 이러한 구조는 다음과 같은 세 가지 핵심적인 문제를 야기한다.

1. **추론 과정의 복잡성**: 파형 생성 전 단계에서 별도의 $F0$ 예측 모델을 거쳐야 하므로 추론 파이프라인이 복잡해진다.
2. **오차 누적 문제**: 외부 $F0$ 예측 모델에서 발생하는 피치 결정 오류나 유성음/무성음(Voiced/Unvoiced) 판별 오류가 WaveNet의 입력으로 전달되어, WaveNet 자체의 성능과 관계없이 최종 합성 음성의 자연스러움을 저하시킨다.
3. **학습-테스트 불일치**: 학습 시에는 실제 정답(Ground-truth) $F0$를 사용하지만, 테스트(추론) 시에는 예측된 $F0$를 사용함으로써 발생하는 데이터 분포의 불일치 문제가 존재한다.

따라서 본 논문의 목표는 외부 $F0$ 예측 모델 없이 언어적 특징만으로도 고품질의 음성을 생성할 수 있는 Multi-task WaveNet 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **다중 작업 학습(Multi-task Learning, MTL)** 프레임워크를 WaveNet에 도입하는 것이다.

단순히 파형을 생성하는 메인 작업(Main task) 외에, 프레임 수준의 음향 특징(Acoustic features)을 예측하는 보조 작업(Secondary task)을 동시에 학습시킨다. 이 보조 작업이 메인 작업과 조건부 네트워크(Conditional network)를 공유하게 함으로써, 네트워크가 언어적 특징으로부터 피치 정보를 내재적으로 학습하도록 유도한다. 결과적으로 추론 단계에서는 보조 작업 부분을 제거하고 언어적 특징만으로도 $F0$ 정보가 반영된 자연스러운 음성을 생성할 수 있게 된다.

## 📎 Related Works

기존의 SPSS는 주로 HMM 기반 방식이나 DNN, LSTM과 같은 신경망 기반 방식을 사용해왔다. 최근 DeepMind에서 제안한 WaveNet은 Vocoder 없이 Dilated Causal Convolutional Neural Networks를 통해 직접 파형을 생성함으로써 기존 SPSS의 고질적인 문제인 과도한 평활화(Over-smoothing) 현상을 극복하고 매우 자연스러운 음질을 구현하였다.

그러나 앞서 언급했듯이, 기존 WaveNet 기반 SPSS는 $\log F0$ 조건에 강하게 의존한다. $F0$ 조건이 없을 경우 억양 오류가 심각하게 발생하며, 이를 해결하기 위한 외부 $F0$ 예측 모델의 도입은 시스템의 효율성과 안정성을 떨어뜨리는 한계가 있었다. 본 연구는 이러한 외부 의존성을 MTL 구조를 통해 네트워크 내부로 통합함으로써 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Multi-task WaveNet은 크게 **조건부 네트워크(Conditional Network)**와 **WaveNet 생성 모델**로 구성된다.

- **조건부 네트워크**: 입력된 언어적 특징을 인코딩하여 고차원 표현을 생성하며, 이는 메인 작업과 보조 작업에 공통으로 제공된다.
- **Main Task (Waveform Generation)**: 인코딩된 특징을 조건으로 하여 $\mu$-law 알고리즘으로 양자화된 음성 샘플을 자기회귀(Autoregressive) 방식으로 생성한다.
- **Secondary Task (Acoustic Feature Prediction)**: 동일한 조건부 네트워크를 공유하며, 멜-케프스트럼 계수(MCCs), $\log F0$, 유성/무성 플래그(V/UV flag)를 예측한다.

### 2. 주요 구성 요소 및 수식 설명

#### A. WaveNet 생성 모델

WaveNet은 Dilated Causal Convolution을 사용하여 수신 영역(Receptive field)을 지수적으로 확장한다. 출력 샘플 시퀀스 $x = \{x_1, x_2, \dots, x_T\}$의 조건부 확률 분포는 다음과 같이 정의된다.

$$p(x|c) = \prod_{i=1}^{T} p(x_i | x_{i-N+1}, x_{i-N+2}, \dots, x_{i-1}, c)$$

여기서 $N$은 수신 영역의 길이이며, $c$는 조건부 시퀀스이다. 각 층의 Gated Activation Unit은 다음과 같이 계산된다.

$$\hat{h}_k = \tanh(W_{f,k} * h_k + V_{f,k} * c) \odot \sigma(W_{g,k} * h_k + V_{g,k} * c)$$

$\sigma$는 시그모이드 함수, $\odot$은 원소별 곱셈, $*$는 컨볼루션 연산을 의미한다.

#### B. Quasi-Recurrent Neural Network (QRNN) 조건부 네트워크

언어적 특징을 인코딩하기 위해 QRNN을 사용한다. QRNN은 RNN의 순차적 특성과 CNN의 병렬 계산 능력을 결합한 구조이다. 단방향 QRNN 층의 연산은 다음과 같다.

$$\hat{h} = \tanh(W_h * x + B_h)$$
$$o = \sigma(W_o * x + B_o)$$
$$f = \sigma(W_f * x + B_f)$$
$$h_t = f_t \cdot h_{t-1} + (1 - f_t) \cdot \hat{h}_t$$
$$z_t = o_t \cdot h_t$$

본 모델은 이를 양방향(Bidirectional)으로 구성하여 문맥 정보를 충분히 반영하며, 이후 Nearest neighbor upsampling을 통해 오디오 샘플링 레이트와 시간 해상도를 맞춘다.

### 3. 학습 및 추론 절차

- **학습 단계**: 메인 작업은 교차 엔트로피(Cross-entropy) 손실 함수를 사용하여 양자화된 샘플 값을 분류하도록 학습하며, 보조 작업은 최소 평균 제곱 오차(Minimum Mean Squared Error, MMSE) 기준을 사용하여 MCC, $\log F0$, V/UV 플래그를 예측하도록 학습한다.
- **추론 단계**: 텍스트 분석을 통해 얻은 언어적 특징이 QRNN을 통과하여 WaveNet으로 전달된다. 이때 **보조 작업 네트워크는 제거**되며, 오직 메인 작업의 WaveNet만이 동작하여 파형을 생성한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 중국어 여성 화자의 데이터셋 두 종류(Corpus A: 16.2시간, Corpus B: 2.1시간)를 사용하였다.
- **비교 대상**:
  - `LSTM`: 전통적인 신경망 기반 SPSS.
  - `Concatenative`: HMM 기반 유닛 선택 합성.
  - `WaveNet`: $F0$와 언어적 특징을 모두 조건으로 사용하는 오리지널 WaveNet.
  - `WaveNet-lin`: 언어적 특징만 조건으로 사용하는 오리지널 WaveNet.
  - `MTL-WaveNet`: 제안된 다중 작업 WaveNet.
- **평가 지표**: MCD(Mel-cepstral distortion), BAP, $F0$ RMSE, $F0$ 상관계수(Corr.), V/UV 에러 등을 측정하였다.

### 2. 정량적 결과

표 1의 결과에 따르면, `MTL-WaveNet`은 모든 시스템 중 가장 낮은 $F0$ RMSE와 MCD 값을 기록하였다. 특히 `WaveNet-lin`의 $F0$ RMSE가 가장 높게 나타난 것은 오리지널 WaveNet에서 $\log F0$ 조건이 얼마나 필수적이었는지를 보여준다. 또한 `MTL-WaveNet`이 기존 `WaveNet`보다 우수한 지표를 보인 것은, 외부 $F0$ 모델의 오차가 누적되는 문제를 MTL 구조가 효과적으로 해결했음을 시사한다.

### 3. 정성적 결과 (주관적 선호도 테스트)

5명의 청취자를 통한 선호도 테스트 결과, `MTL-WaveNet`은 `WaveNet` 및 `Concatenative` 시스템보다 더 높은 선호도를 얻었다. 특히 데이터 양이 적은 Corpus B에서 유닛 선택 방식(`Concatenative`)과의 격차가 매우 크게 벌어졌는데, 이는 WaveNet 기반 방식이 소량의 데이터에서도 더 안정적인 품질을 제공함을 의미한다. 또한 `WaveNet-lin`에서 발생하던 부자연스러운 톤(Strange tones) 문제가 `MTL-WaveNet`에서 해결되었음이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 다중 작업 학습을 통해 신경망이 입력 특징으로부터 음성 생성에 필요한 핵심 정보(피치)를 스스로 추출하도록 강제함으로써, 외부 모델에 대한 의존성을 제거하는 전략을 성공적으로 입증하였다.

**강점:**

- **추론 효율성 증대**: 외부 $F0$ 예측 모델을 제거하여 전체 파이프라인을 단순화하였다.
- **품질 향상**: 학습-테스트 간의 데이터 불일치 문제와 외부 모델의 오차 누적 문제를 근본적으로 해결하여 더 자연스러운 억양을 구현하였다.
- **학습 가속화**: 보조 작업이 메인 작업의 학습 과정에서 가이드 역할을 하여 수렴 속도를 높이는 효과를 가져왔다.

**한계 및 논의사항:**

- 본 연구는 여전히 Duration 모델을 별도로 사용하고 있어 완전한 End-to-end 구조는 아니다. 저자들 또한 향후 연구에서 Duration 모델을 대체할 수 있는 end-to-end 모델 도입의 필요성을 언급하고 있다.
- 주관적 테스트에서 `MTL-WaveNet`과 `WaveNet`의 차이가 정량적 지표만큼 극적으로 나타나지 않은 점은, 이미 `WaveNet` 자체의 품질이 충분히 높기 때문에 미세한 디테일에서만 차이가 발생했기 때문으로 해석된다.

## 📌 TL;DR

본 논문은 WaveNet 기반 음성 합성에서 외부 $F0$ 예측 모델에 의존하던 기존 방식을 개선하여, **음향 특징 예측을 보조 작업으로 하는 다중 작업 학습(MTL) 구조**를 제안하였다. 이를 통해 외부 모델 없이 언어적 특징만으로도 고품질의 음성 파형을 생성할 수 있게 되었으며, 추론 과정의 단순화와 피치 오차 누적 문제 해결이라는 성과를 거두었다. 이 연구는 향후 실시간 음성 합성 시스템의 효율성과 자연스러움을 동시에 잡을 수 있는 실용적인 방향성을 제시한다.
