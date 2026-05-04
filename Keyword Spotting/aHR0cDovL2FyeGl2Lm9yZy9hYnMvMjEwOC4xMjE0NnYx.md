# Separable Temporal Convolution plus Temporally Pooled Attention for Lightweight High-performance Keyword Spotting

Shenghua Hu, Jing Wang, Yujun Wang, Wenjing Yang (Year not specified, references up to 2019)

## 🧩 Problem to Solve

본 논문은 모바일 기기 환경에서 동작하는 Keyword Spotting(KWS) 시스템의 효율성 문제를 해결하고자 한다. KWS는 오디오 신호에서 미리 정의된 특정 키워드를 검출하는 기술로, 모바일 기기의 한정된 하드웨어 자원 내에서 구동되어야 하므로 매우 작은 메모리 점유율(memory footprint)이 필수적이다.

기존의 딥러닝 기반 KWS 모델들은 높은 성능을 유지하기 위해 여전히 수십만 개의 파라미터를 사용하고 있으며, 이는 모바일 기기에서의 실시간 구현과 저전력 동작에 제약이 된다. 따라서 본 연구의 목표는 높은 정확도를 유지하면서도 파라미터 수와 연산량을 획기적으로 줄인 경량화된 고성능 KWS 신경망 구조를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Separable Temporal Convolution**과 **Temporally Pooled Attention**을 결합하여 모델의 복잡도를 낮추면서도 전역적 특징 추출 능력을 강화하는 것이다.

1. **Separable Temporal Convolution Network**: Depthwise Separable Convolution과 Temporal Convolution을 결합하여 파라미터 수와 연산량을 줄이는 동시에 시간적 특징을 효과적으로 추출한다.
2. **Temporally Pooled Attention Module**: 기존의 Average Pooling이 모든 시간 프레임에 동일한 가중치를 부여하는 한계를 극복하기 위해, Attention 메커니즘을 통해 시간축의 전역적 특징을 적응적으로 캡처하는 모듈을 제안한다.
3. **ST-AttNet 설계**: 위 두 가지 요소를 결합한 ST-AttNet 구조를 통해, 최신 모델인 TC-ResNet14-1.5 대비 파라미터 수를 1/6 수준(48K)으로 줄이면서도 동일한 수준의 정확도(96.6%)를 달성하였다.

## 📎 Related Works

KWS를 구현하는 기존 방식은 크게 대규모 어휘 연속 음성 인식(LVCSR) 기반 방식과 은닉 마르코프 모델(HMMs) 기반 방식으로 나뉘나, 두 방식 모두 메모리 비용과 연산량이 너무 커 모바일 적용이 어렵다.

최근에는 신경망 기반의 KWS가 각 키워드를 하나의 클래스로 보는 오디오 분류 문제로 접근하며 인기를 끌고 있다. 특히 ResNet 기반의 KWS 시스템은 잔차 연결(residual connections)을 통해 네트워크 깊이를 늘려 성능을 높였으나, 여전히 200K 이상의 파라미터를 필요로 한다. 또한 Temporal Convolutional Neural Network(TCNN)는 1D 컨볼루션을 통해 파라미터를 줄이고 정확도를 높였지만, 채널 수로 인한 모델의 비대함이 여전히 문제로 남아 있었다. 본 논문은 이러한 기존 모델들이 전역 특징 추출을 위해 단순한 Average Pooling을 사용한다는 점에 주목하여, 이를 Attention 메커니즘으로 대체함으로써 효율성과 성능을 동시에 잡고자 하였다.

## 🛠️ Methodology

### 1. 데이터 전처리 (Data Process)

입력 오디오 신호에 대해 $20\text{Hz}/7.8\text{kHz}$ 대역 통과 필터(band-pass filter)를 적용하여 노이즈를 제거한다. 이후 $30\text{ms}$ 윈도우 크기와 $10\text{ms}$ 프레임 시프트를 사용하여 40차원의 Mel-Frequency Cepstrum Coefficient(MFCC) 특징을 추출한다. 최종 입력 데이터는 $I \in \mathbb{R}^{T \times F}$ 형태로 표현되며, 여기서 $F$는 MFCC 차원, $T$는 프레임 수이다.

### 2. Temporally Pooled Attention

기존의 Self-Attention은 2D 행렬을 입력받아 2D 행렬을 출력하지만, KWS 분류를 위해서는 Softmax 층 이전에 2D 행렬을 1D 벡터로 축소해야 한다. 기존 연구들은 단순 AveragePool을 사용했으나, 이는 모든 시간 특징에 동일한 가중치를 주는 비합리적인 방식이다. 본 논문은 다음과 같은 Temporally Pooled Attention을 제안한다.

$$\text{Attend}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D_k}}\right)V$$

여기서 쿼리 $Q$를 생성하기 위해 입력 $U$에 가중치 행렬 $W$를 곱한 후 AveragePool을 적용하여 1D 벡터로 변환한다. 즉, $Q = \text{AveragePool}(UW)$로 정의하여 출력값이 1D 벡터가 되도록 설계하였다. 또한, 이를 확장하여 여러 개의 헤드를 사용하는 Multi-head Attention 구조를 적용하였다.

$$\text{Multihead}(Q, K, V) = \text{Cat}(\text{Attend}(Q_1, K_1, V_1), \dots, \text{Attend}(Q_n, K_n, V_n))$$

### 3. 모델 아키텍처 (Model Architecture)

전체 시스템인 ST-AttNet은 다음과 같은 구조적 특징을 가진다.

- **Separable Convolution**: 모든 컨볼루션 층은 $3 \times 1$ kernels를 사용하는 Depthwise Convolution과 $1 \times 1$ kernels를 사용하는 Pointwise Convolution으로 구성된다. 각 층 이후에는 Batch Normalization(BN)과 ReLU 활성화 함수가 배치된다.
- **Dilated Convolution**: 시간축의 수용 영역(receptive field)을 효율적으로 확장하기 위해 Dilated Convolution을 적용하며, 층 $i$에서의 확장 비율 $d$는 $d = 2^{\lfloor i/3 \rfloor}$의 지수적 스케줄을 따른다.
- **Residual Block**: 2개의 Separable Convolution 층과 입력-출력을 직접 연결하는 잔차 연결(shortcut)로 구성된다. 모든 컨볼루션의 stride는 1로 설정하여 입력과 출력의 차원을 일치시켰다.
- **전체 파이프라인**: MFCC 입력 $\rightarrow$ Separable Temporal Conv 층들 (Residual Blocks) $\rightarrow$ Temporally Pooled Attention $\rightarrow$ Softmax $\rightarrow$ 최종 클래스 분류 순으로 진행된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Google Speech Commands Dataset V1을 사용하였다. 총 30개 단어 중 10개 키워드와 filler 단어들을 구분하여 총 12개 클래스(yes, no, up, down, left, right, on, off, stop, go, unknown, silence)를 분류한다.
- **학습 설정**: TensorFlow를 사용하여 학습하였으며, Cross Entropy 손실 함수와 Adam 옵티마이저를 사용하였다. 초기 학습률은 $0.001$이며, 성능 향상이 정체될 경우 $60\%$ 수준으로 감소시키는 전략을 사용하였다.

### 2. 정량적 결과 및 비교

제안 모델의 변형인 ST-AttNet4, ST-AttNet4-wide, ST-AttNet7의 성능을 기존 baseline 모델들과 비교하였다.

- **정확도 및 효율성**: ST-AttNet4-wide 모델은 **96.6%의 정확도**를 기록하였는데, 이는 SOTA 모델인 TC-ResNet14-1.5와 동일한 수치이다. 그러나 파라미터 수는 **48K**로, TC-ResNet14-1.5(305K)의 약 $1/6$ 수준에 불과하다.
- **Attention의 효과**: Temporally Pooled Attention을 제거하고 AveragePool로 대체한 ST-Net4(95.4%)와 비교했을 때, ST-AttNet4(96.3%)가 **0.9% 더 높은 정확도**를 보였다.
- **기타 비교**: Res15 모델과 비교하여 ST-AttNet4는 파라미터를 10배 줄였음에도 더 높은 성능을 냈으며, DS-CNN-L 대비로는 파라미터를 17배 줄이면서 정확도를 0.9% 향상시켰다.

### 3. ROC 곡선 분석

ROC 곡선의 AUC(Area Under the Curve)가 작을수록 우수한 모델임을 의미하며, 실험 결과 ST-AttNet4-wide가 다른 비교 모델들보다 가장 우수한 성능 곡선을 나타내었다.

## 🧠 Insights & Discussion

본 연구는 KWS 모델의 경량화를 위해 단순한 파라미터 제거가 아닌, **구조적 효율성(Separable Conv)**과 **전역 특징 추출의 최적화(Pooled Attention)**라는 두 가지 방향에서 접근하였다.

특히, 기존 모델들이 시간축의 정보를 압축하기 위해 사용하던 Average Pooling이 정보의 손실을 초래한다는 점을 지적하고, 이를 Attention 메커니즘으로 대체함으로써 아주 적은 파라미터 추가만으로도 상당한 성능 향상을 이끌어낸 점이 인상적이다. 또한 Dilated Convolution을 통해 층을 깊게 쌓지 않고도 넓은 시간적 맥락을 파악할 수 있게 설계하여 연산 효율성을 극대화하였다.

다만, 본 논문은 Google Speech Commands V1 데이터셋에 국한하여 검증하였으므로, 더 다양한 환경의 소음이 포함된 데이터셋이나 더 많은 단어 수를 가진 환경에서도 동일한 효율성이 유지되는지에 대한 추가 검증이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 모바일 기기용 경량 고성능 KWS를 위해 **Separable Temporal Convolution**과 **Temporally Pooled Attention**을 결합한 **ST-AttNet**을 제안한다. 제안된 모델(ST-AttNet4-wide)은 기존 SOTA 모델인 TC-ResNet14-1.5와 동일한 **96.6%의 정확도**를 달성하면서도 파라미터 수는 **1/6 수준인 48K**로 획기적으로 줄였다. 이는 초소형 임베디드 환경에서의 실시간 음성 인식 시스템 구현에 매우 중요한 기여를 할 수 있을 것으로 평가된다.
