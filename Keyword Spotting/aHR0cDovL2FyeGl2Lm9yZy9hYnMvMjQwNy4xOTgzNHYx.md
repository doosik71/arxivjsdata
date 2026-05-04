# Frequency & Channel Attention Network for Small Footprint Noisy Spoken Keyword Spotting

Yuanxi Lin and Yuriy Evgenyevich Gapanyuk (2024)

## 🧩 Problem to Solve

본 논문은 저전력 및 저사양 임베디드 기기에서 동작하는 **Small-footprint Keyword Spotting (KWS)** 시스템의 소음 강건성(Noise Robustness)을 향상시키는 문제를 해결하고자 한다. KWS 기술은 스마트 스피커, 웨어러블 기기, IoT 장치 등에서 물리적 접촉 없이 명령을 수행하게 하는 핵심 기술이다.

Small-footprint KWS 모델은 항상 전원이 켜져 있어야 하며 제한된 메모리와 연산 자원을 가진 환경에서 동작해야 하므로 모델의 경량성이 필수적이다. 기존의 딥러닝 기반 KWS 모델들은 깨끗한 오디오 데이터셋에서는 높은 성능을 보이지만, 실제 환경의 배경 소음이 섞인 상황에서는 오탐지(False activation)나 미탐지(Missed detection)가 빈번하게 발생하는 한계가 있다. 따라서 본 연구의 목표는 모델의 메모리 점유율을 낮게 유지하면서도, 소음이 심한 환경에서 키워드를 정확하게 인식할 수 있는 강건한 신경망 구조를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **FCA-Net**이라는 새로운 CNN 구조를 제안한 것으로, 주요 설계 아이디어는 다음과 같다.

1.  **C2D (Convolution-based Two-Dimensional) Attention 모듈 제안**: 기존의 채널 어텐션(Channel Attention) 방식에서 더 나아가, 채널과 주파수(Frequency) 차원을 동시에 고려하여 세밀한 가중치를 생성하는 2D 컨볼루션 기반의 어텐션 메커니즘을 도입하였다.
2.  **ConvMixer 기반의 특성 상호작용 결합**: 글로벌 채널 간의 특성 상호작용을 계산하는 ConvMixer 구조와 제안한 C2D 어텐션 모듈을 결합하여, 국소적 특징 추출 능력과 전역적 정보 흐름을 동시에 강화하였다.
3.  **커리큘럼 기반 다중 조건 학습 전략(Curriculum-based multi-condition training strategy)**: 학습 초기에는 깨끗한 샘플로 시작하여 점진적으로 소음 수준이 높은 어려운 샘플을 학습시키는 방식을 적용하여 모델의 소음 적응력을 높였다.

## 📎 Related Works

본 논문에서는 KWS 및 어텐션 메커니즘과 관련된 기존 연구들을 다음과 같이 설명한다.

- **Small Footprint KWS**: TC-ResNet은 1D temporal convolution을 통해 효율성을 높였고, BC-ResNet은 broadcasted residual learning을 도입하였다. 하지만 이러한 연구들은 대부분 소음 강건성 문제를 깊게 다루지 않았다.
- **Noise Robust KWS**: 소음 혼합 학습(Noise mixture training)과 커리큘럼 학습(Curriculum learning)이 제안되었으나, 모델의 크기가 매우 작은 경우 고소음 환경에서 학습에 어려움을 겪는 한계가 있다.
- **Attention Mechanisms**: SE-Net은 채널 어텐션을 통해 성능을 높였고, CBAM은 채널과 공간 어텐션을 결합하였다. 최근에는 연산 비용을 줄인 ECA-Net이 제안되었으나, 이를 소음 강건 KWS 시스템에 통합하여 적용한 연구는 아직 부족한 상태이다.

## 🛠️ Methodology

### 1. ConvMixer Architecture
FCA-Net의 기본 뼈대가 되는 ConvMixer는 Pre-convolutional block, ConvMixer block, Post-convolutional block의 세 부분으로 구성된다. 각 블록은 1D depthwise separable (DWS) convolution, Batch Normalization (BN), Swish activation을 포함한다.

특히 ConvMixer block은 주파수 도메인과 시간 도메인의 특징을 순차적으로 추출하며, 다음과 같은 수식으로 정의된다.
주파수 도메인 특징 $z$는 2D-DWS convolution $f$와 활성화 함수 $\sigma$를 통해 다음과 같이 계산된다.
$$z = \sigma \circ f(\sigma \circ f(x))$$
이후 시간 도메인 특징 $y_1$과 처리된 특징 $y_2$를 다음과 같이 순차적으로 구한다.
$$y_1 = \sigma \circ \text{BatchNorm}(f(z))$$
$$y_2 = \sigma \circ \text{BatchNorm}(f_2(y_1))$$
최종 출력 $\tilde{y}$는 입력 $x$와 중간 특징들의 합으로 계산되며, 여기서 $f_3$는 mixer layer(MLP) 역할을 한다.
$$\tilde{y} = x + y_1 + f_3(y_2)$$

### 2. Efficient Attention Modules
논문에서는 기존의 SE, ECA 모듈과 제안하는 C2D 모듈을 비교 분석한다.
- **SE (Squeeze-and-Excitation)**: Global Average Pooling (GAP) 후 두 개의 FC 레이어를 통해 채널별 가중치를 생성한다.
- **ECA (Efficient Channel Attention)**: GAP 후 1D 컨볼루션을 사용하여 차원 축소 없이 지역적 채널 상호작용을 캡처한다.
- **C2D (Convolution-based 2D Attention)**: 시간 축에 대해 GAP를 수행하여 채널-주파수 평면 $z$를 생성하고, 이를 2D 컨볼루션 레이어에 통과시켜 세밀한 가중치 $\omega$를 생성한다.
  $$z(c, f) = \frac{1}{T} \sum_{t=1}^{T} X(c, f, t)$$
  $$\omega = \sigma(\text{Conv}_2(\text{ReLU}(\text{BN}(\text{Conv}_1(z)))))$$
  최종 결과물 $X'$는 입력 특징 맵 $X$와 가중치 $\omega$의 원소별 곱(element-wise product)으로 계산된다.
  $$X' = \omega \odot X$$

### 3. Proposed FCA-Net
FCA-Net은 ConvMixer 블록 뒤에 C2D 어텐션 모듈을 배치한 구조이다. 논문에서는 C2D 모듈의 위치에 따라 FCA-Net-all(모든 블록 뒤), FCA-Net-pre(전처리 블록 뒤), FCA-Net-post(후처리 블록 뒤), FCA-Net-final(최종 선형 레이어 전)의 변형 구조를 실험하였다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Commands V2 (12개 클래스) 및 소음 합성을 위한 MUSAN 데이터셋을 사용하였다.
- **소음 조건**: 깨끗한 상태(Clean)부터 SNR 20dB, 0dB, -5dB, -10dB까지 단계적으로 소음을 추가하여 테스트하였다.
- **구현 세부사항**: 40-channel MFCC를 입력으로 사용하였으며, Mixup, SpecAugment, 커리큘럼 학습을 적용하였다. Adam 옵티마이저와 Cosine Annealing Warm Restarts 스케줄러를 사용하였다.

### 2. 정량적 결과
FCA-Net은 다른 소형 모델 및 대형 트랜스포머 모델과 비교하여 뛰어난 성능을 보였다.

- **정확도**: 공식 V2-12 테스트셋에서 약 96.96%의 정확도를 기록하여 ConvMixer보다 2~3% 향상된 성능을 보였다.
- **소음 강건성**: 가장 가혹한 조건인 -10dB 소음 환경에서 **80.40%**의 정확도를 달성하였다. 이는 유사한 크기의 ConvMixer(77.47%)나 BC-ResNet-6(79.40%)보다 우수하며, 훨씬 더 큰 모델인 AST-Tiny(72.32%)나 KWT-3(71.08%)보다 월등히 높은 수치이다.
- **효율성**: 모델 파라미터 수와 MACs(연산량) 측면에서도 매우 효율적이며, 메모리 제약이 심한 환경에 적합함을 입증하였다.

### 3. 절제 연구 (Ablation Study)
다양한 어텐션 모듈과 배치 위치를 실험한 결과, **C2D 모듈을 모든 블록에 배치한 'all' 설정**이 모든 소음 수준에서 가장 높은 성능을 보였다. 특히 C2D는 SE나 ECA보다 소음 환경에서 더 강력한 성능 향상을 이끌어냈다.

## 🧠 Insights & Discussion

본 연구의 결과는 KWS 시스템에서 단순히 채널 정보를 압축하는 것보다, **주파수-채널의 2차원적 관계를 보존하며 가중치를 부여하는 것이 소음 제거 및 핵심 특징 추출에 훨씬 효과적**임을 시사한다. 소음은 특정 주파수 대역에 집중되어 나타나는 경향이 있는데, C2D 모듈이 이를 정밀하게 억제하고 유의미한 음성 성분을 강조했기 때문으로 해석된다.

또한, 소형 모델일수록 고소음 환경에서 학습이 불안정하다는 점을 커리큘럼 학습을 통해 극복한 점이 인상적이다. 다만, 본 논문에서는 특정 데이터셋(Google Speech Commands V2)과 특정 소음셋(MUSAN)에 한정하여 실험을 진행하였으므로, 실제 환경의 더 다양한 변칙적 소음(예: 갑작스러운 충격음 등)에 대해서도 동일한 강건성을 유지할지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 소형 KWS 모델의 소음 강건성을 높이기 위해 **ConvMixer 구조에 2D 컨볼루션 기반의 C2D 어텐션 모듈을 결합한 FCA-Net**을 제안하였다. 커리큘럼 학습 전략을 통해 학습 효율을 높였으며, 실험 결과 매우 적은 파라미터로도 -10dB의 고소음 환경에서 SOTA 모델들을 능가하는 성능(80.40% 정확도)을 달성하였다. 이 연구는 리소스가 제한된 엣지 디바이스에서 실제 세계의 소음 문제를 해결할 수 있는 실용적인 아키텍처를 제시하였다는 점에서 중요한 의미를 갖는다.