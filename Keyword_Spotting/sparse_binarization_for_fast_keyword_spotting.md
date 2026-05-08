# Sparse Binarization for Fast Keyword Spotting

Jonathan Svirsky, Uri Shaham, Ofir Lindenbaum (2024)

## 🧩 Problem to Solve

본 논문은 스마트폰이나 임베디드 시스템과 같은 엣지 디바이스(Edge Device)에서 동작하는 키워드 검출(Keyword Spotting, KWS) 모델의 효율성 문제를 해결하고자 한다. KWS 모델은 항상 켜져 있는(Always-on) 상태로 특정 트리거 단어를 감지해야 하므로, 전력 소비를 줄이고 개인정보 보호 및 응답 속도를 높이기 위해 디바이스 로컬에서 실행되는 것이 필수적이다.

그러나 엣지 디바이스는 매우 제한된 계산 능력과 메모리 자원을 가지고 있다. 구체적으로, 온칩 RAM(SRAM)은 20 KB에서 512 KB 수준이며, CPU(Cortex-M)의 동작 주파수는 72 MHz에서 216 MHz 사이로 제한적이다. 이러한 하드웨어 제약은 모델의 파라미터 수와 연산량(Operations)을 엄격하게 제한하며, 이는 모델의 정확도를 유지하면서도 연산 효율성을 극대화해야 하는 상충 관계(Trade-off) 문제를 야기한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 입력 신호를 희소한 이진 표현(Sparse Binarized Representation)으로 변환하여, 매우 가벼운 선형 분류기만으로도 높은 정확도를 달성하는 **SparkNet** 아키텍처를 제안하는 것이다.

핵심적인 직관은 입력 데이터에서 예측에 불필요한 노이즈나 비정보성 특징(Non-informative features)은 제거하고, 중요한 특징만을 보존하는 동적 이진화(Dynamic Binarization) 모델을 학습시키는 것이다. 기존의 동적 희소화 방식들이 입력 샘플과 게이트(Gate)를 요소별로 곱하는 과정을 거쳤던 것과 달리, SparkNet은 학습된 이진 표현 그 자체를 분류기의 입력으로 사용하여 계산 복잡도를 획기적으로 낮추었다.

## 📎 Related Works

기존의 KWS 최적화 연구들은 주로 작은 크기의 Convolutional Neural Networks(CNNs)나 Recurrent Neural Networks(RNNs)를 설계하고, 양자화(Quantization) 및 가지치기(Pruning) 기법을 통해 모델 크기를 줄이는 방식에 집중하였다. 또한, 2차원 컨볼루션이나 Attention 블록을 사용하여 성능을 높이려는 시도가 있었다.

본 논문은 특히 음성 활동 감지(Voice Activity Detection, VAD)를 위해 제안된 SG-VAD 모델에서 영감을 받았다. SG-VAD는 이진 마스킹을 통해 입력 신호의 노이즈를 줄이는 방식을 사용하였으나, 본 연구는 이를 KWS 태스크로 확장하여 단순한 선형 투영 층(Linear Projection Layer)만으로도 키워드 분류가 가능함을 보였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

SparkNet은 입력 데이터인 MFCC(Mel-frequency cepstral coefficients) 특징 행렬 $x_i \in \mathbb{R}^{F \times T}$를 받아, 이를 이진 표현 $z_i$로 변환한 뒤, 시간 축으로 평균 풀링(Average Pooling)을 거쳐 단일 선형 층을 통해 최종 클래스를 예측한다.

### 2. SparkNet 아키텍처

모델의 기본 빌딩 블록은 **1D Time-Channel Separable Convolution (TCS)**이다. 이는 Depth-wise Separable Convolution의 구현체로, 다음과 같이 두 단계로 분리된다:

- **Depth-wise Convolution:** 각 채널별로 커널 길이 $K$만큼 시간 축으로 연산한다.
- **Point-wise Convolution:** 각 시간 프레임에서 모든 채널에 대해 독립적으로 연산한다.

전체 구조는 4개의 블록으로 구성되며, 각 블록은 `TCS Convolution $\rightarrow$ Batch Normalization $\rightarrow$ ReLU` 순으로 배치된다. 뒤의 3개 블록에는 잔차 연결(Residual Connections)이 포함되며, 커널 크기는 $K \in \{11, 15, 19, 29\}$로 점진적으로 증가하여 수용 영역(Receptive Field)을 넓힌다. 마지막으로 $1 \times 1$ Convolution, BN, $\tanh$ 활성화 함수를 거쳐 최종 표현을 생성한다.

### 3. 희소 이진 표현 학습 (Sparse Binarized Representation Learning)

CNN 백본은 먼저 $\mu_i \in [-1, 1]^{F \times T}$를 학습하며, 이를 가우시안 기반의 이완(Gaussian-based relaxation) 기법을 통해 근사적 베르누이 변수 $z_i$로 변환한다. 재매개변수화 트릭(Reparameterization trick)을 사용하여 학습 중 다음과 같이 계산한다:

$$z_i = \max(0, \min(1, 0.5 + \mu_i + \varepsilon_i))$$

여기서 $\varepsilon_i$는 $\mathcal{N}(0, \sigma^2)$에서 추출된 랜덤 노이즈이며, $\sigma=0.5$로 고정된다.

또한, $z_i$ 행렬이 희소(Sparse)해지도록 유도하기 위해 $L_0$ 노름을 근사한 희소성 손실 함수 $L_{sparse}$를 도입한다. 이는 가우시안 오차 함수($\text{erf}$)를 사용하여 다음과 같이 정의된다:

$$L_{sparse}(\mu_i) = \frac{1}{F \times T} \sum_{f=1}^{F} \sum_{t=1}^{T} \left( \frac{1}{2} - \frac{1}{2} \text{erf} \left( \frac{-\mu_{f,t} + 0.5}{\sqrt{2}\sigma} \right) \right)$$

### 4. 학습 목표 및 손실 함수

최종 손실 함수는 희소성 손실과 교차 엔트로피 손실(Cross-Entropy Loss)의 가중 합으로 정의된다:

$$L = L_{sparse}(z) + \lambda L_{ce}(\hat{y}, y)$$

여기서 $\lambda=100$으로 설정되어 희소성을 강력하게 유도한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Google Speech Commands v1 및 v2. 10개의 타겟 단어, 'Unknown', 'Silence'를 포함한 총 12개 클래스로 구성된다.
- **특징 추출:** $F=32$ 빈(bins)의 MFCC를 사용한다.
- **평가 지표:** Top-1 Accuracy 및 MACs(Multiply-Accumulate operations)를 통해 효율성을 측정한다.

### 2. 정량적 결과

SparkNet은 기존 SOTA 소형 모델들과 비교하여 압도적인 연산 효율성을 보였다.

- **연산 효율성:** $C=16$ (파라미터 $\sim 4.6\text{K}$) 모델의 경우, 유사한 규모의 BC-ResNet-0.625보다 **약 4배 빠르며(MACs 기준)**, 정확도는 소폭 향상되었다. $C=32$ 모델 역시 BC-ResNet-1보다 3배, DS-ResNet10보다 5배 빠른 속도를 기록하였다.
- **노이즈 강건성:** 다양한 SNR(Signal-to-Noise Ratio) 환경에서 테스트한 결과, SparkNet이 BC-ResNet보다 높은 정확도를 유지하며 더 강건한 성능을 보였다. 이는 이진화 과정이 정보가 없는 노이즈 영역을 효과적으로 제거하기 때문으로 분석된다.
- **초소형 모델 가능성:** 채널 수를 $C=4$까지 줄인 초소형 모델(파라미터 $\sim 1.4\text{K}$, MACs $105\text{K}$)에서도 약 83%의 정확도를 달성하였다.

### 3. 절제 연구 (Ablation Study)

- **보조 분류기/디코더:** 보조 분류기(Auxiliary Classifier)나 재구성 디코더(Reconstruction Decoder)를 추가하여 학습시켜 보았으나, 정확도 향상은 거의 없었다. 이는 KWS 태스크에서 이진 마스크 자체가 이미 충분한 정보를 담고 있음을 시사한다.
- **이진화의 효과:** $L_{sparse}$와 베르누이 근사 과정을 제거했을 때 성능이 하락함을 확인하여, 제안한 희소 이진 표현 학습이 성능 향상의 핵심임을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

SparkNet의 가장 큰 성과는 **'계산 비용의 획기적 감소'**와 **'강건성 확보'**를 동시에 달성한 점이다. 일반적으로 모델의 크기를 줄이면 정확도가 낮아지지만, 본 논문은 입력 데이터를 적응적으로 필터링하는 '이진 게이트' 개념을 도입함으로써, 복잡한 연산 없이도 유의미한 특징만을 추출하여 선형 분류기에 전달하는 구조를 설계하였다. 특히 MFCC의 시공간적 영역에서 중요한 부분만을 활성화하는 패턴을 학습함으로써 노이즈에 강한 특성을 갖게 되었다.

### 한계 및 논의사항

논문에서는 정성적 분석을 통해 모델이 유의미한 특징을 잘 포착함을 보였으나, 어떤 구체적인 음향적 특징이 이진화 과정에서 보존되는지에 대한 심층적인 음성학적 분석은 부족하다. 또한, 지도 학습(Supervised Learning) 기반으로 제안되었으므로, 데이터 라벨이 부족한 환경에서의 성능은 미지수이다.

## 📌 TL;DR

본 논문은 엣지 디바이스용 초경량 키워드 검출 모델인 **SparkNet**을 제안한다. 1D Time-Channel Separable Convolution과 가우시안 이완 기반의 희소 이진화(Sparse Binarization)를 통해, 입력 신호에서 핵심 정보만 남기고 노이즈를 제거하는 이진 마스크를 학습한다. 결과적으로 기존 SOTA 모델 대비 **연산량(MACs)을 최대 4~5배 줄이면서도 동등하거나 더 높은 정확도와 노이즈 강건성을 달성**하였다. 이 연구는 극도로 자원이 제한된 환경에서의 실시간 음성 인터페이스 구현에 매우 중요한 기여를 할 것으로 평가된다.
