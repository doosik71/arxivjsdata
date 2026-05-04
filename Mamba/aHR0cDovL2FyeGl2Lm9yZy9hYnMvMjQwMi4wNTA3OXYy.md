# Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation

Ziyang Wang, Jian-Qing Zheng, Yichi Zhang, Ge Cui, Lei Li (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서는 국소적 특징(local features)을 잘 포착하는 합성곱 신경망(CNN)과 전역적 문맥(global context) 이해 능력이 뛰어난 비전 트랜스포머(ViT)가 널리 사용되어 왔다. 그러나 CNN은 장거리 의존성(long-range dependencies)을 모델링하는 데 한계가 있으며, ViT는 입력 크기에 따라 연산 비용이 제곱으로 증가하는 self-attention 메커니즘의 특성상 고해상도 의료 영상을 효율적으로 처리하기 어렵다는 문제점이 있다.

본 논문의 목표는 이러한 연산 효율성과 전역적 문맥 파악 능력 사이의 트레이드오프를 해결하는 것이다. 즉, 트랜스포머의 전역적 모델링 능력은 유지하면서도 연산 복잡도를 낮추어 고해상도 의료 영상에서도 정밀한 분할 성능을 낼 수 있는 새로운 아키텍처를 제안하는 것에 목적이 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 상태 공간 모델(State Space Model, SSM)의 효율적인 시퀀스 모델링 능력을 갖춘 Mamba 아키텍처를 U-Net 구조에 통합하는 것이다. 특히, 시각적 데이터를 처리하기 위해 제안된 Visual Mamba(VMamba)를 기반으로 한 **Mamba-UNet**을 제안한다. 이 모델은 순수하게 Visual Mamba 블록(VSS block)으로 구성된 인코더-디코더 구조를 가지며, 스킵 연결(skip connections)을 통해 다양한 스케일의 공간 정보를 보존함으로써 세밀한 디테일과 넓은 의미적 문맥을 동시에 학습할 수 있도록 설계되었다.

## 📎 Related Works

기존의 의료 영상 분할은 주로 U-Net의 대칭적 구조와 스킵 연결을 기본으로 하여 발전해 왔다. 최근에는 트랜스포머를 통합한 TransUNet, UNETR, Swin-UNet 등이 등장하여 전역적 의존성을 모델링하려 시도했다. 하지만 이러한 Transformer 기반 모델들은 입력 해상도가 높아질수록 연산량이 급격히 증가하는 한계가 있다.

반면, 최근 주목받는 SSM(State Space Model)과 그 발전 형태인 Mamba는 시퀀스 길이에 대해 선형적인 연산 복잡도를 가지면서도 매우 긴 시퀀스를 효율적으로 처리할 수 있다. 본 연구는 이러한 Mamba의 특성과 VMamba의 Cross-Scan Module(CSM)을 활용하여, 기존의 CNN 기반 U-Net보다는 강력한 전역 모델링을, Transformer 기반 U-Net보다는 효율적인 연산을 수행하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인
Mamba-UNet은 전형적인 U-Net의 인코더-디코더 구조를 따른다. 입력 영상은 패치(patch) 단위로 분할되어 1차원 시퀀스로 변환된 후, 여러 단계의 VSS 블록과 패치 병합(patch merging) 층을 거치며 계층적 특징을 추출한다. 이후 디코더에서 패치 확장(patch expanding) 층과 VSS 블록을 통해 해상도를 복원하며, 이때 인코더의 특징 맵을 스킵 연결로 전달받아 정밀한 복원을 수행한다.

### Visual State Space (VSS) Block
VSS 블록은 본 모델의 핵심 구성 요소로, 다음과 같은 구조를 가진다.
- 입력 특징은 먼저 선형 임베딩 층을 거친 후 두 갈래 경로로 나뉜다.
- 한 경로에서는 Depth-wise Convolution과 SiLU 활성화 함수를 거쳐 **SS2D 모듈**로 진입하며, 이후 Layer Normalization을 통해 다시 합쳐진다.
- 일반적인 ViT와 달리 positional embedding이나 MLP 단계가 없으며, 이는 동일한 연산 예산 내에서 더 많은 블록을 쌓기 위함이다.

### 수학적 배경: State Space Model (SSM)
Mamba의 기초가 되는 SSM은 선형 시불변 시스템(linear time-invariant system)으로, 입력 $x(t)$를 출력 $y(t)$로 매핑하기 위해 은닉 상태 $h(t)$를 사용한다. 이는 다음과 같은 상미분 방정식(ODE)으로 정의된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

여기서 $A$는 진화 파라미터, $B$와 $C$는 투영 파라미터, $D$는 스킵 연결 파라미터이다. 이를 디지털 시스템에서 처리하기 위해 이산화(discretization)하면 다음과 같이 표현된다.

$$h_t = \bar{A}h_{k-1} + \bar{B}x_k$$
$$y_t = C h_k + D x_k$$

이때 $\bar{A} = e^{\Delta A}$, $\bar{B} = (\Delta A)(e^{\Delta A}-I)^{-1}\Delta B \approx \Delta B$ 와 같이 근사화되어 계산 효율성을 높인다.

### Encoder, Decoder 및 Bottleneck
- **Encoder**: 각 단계에서 2개의 VSS 블록을 사용하며, 패치 병합(patch merging)을 통해 해상도를 $\frac{1}{2}$로 줄이고 채널 차원을 2배로 늘린다. 총 3번의 다운샘플링을 거친다.
- **Bottleneck**: 인코더와 디코더 사이에서 2개의 VSS 블록을 사용하여 최상위 수준의 특징을 학습한다.
- **Decoder**: 패치 확장(patch expanding) 층을 통해 해상도를 2배로 높이고 채널을 절반으로 줄이며, 인코더의 동일 레벨 특징을 결합하여 공간적 디테일을 보완한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ACDC MRI 심장 분할 데이터셋(100명 환자, 4개 ROI 클래스)과 Synapse CT 복부 분할 데이터셋(30개 스캔, 9개 ROI 클래스)을 사용하였다. 모든 이미지는 $224 \times 224$ 크기로 조정되었다.
- **비교 대상**: UNet, Attention UNet, TransUNet, Swin-UNet.
- **평가 지표**: Dice, IoU, Accuracy, Precision, Sensitivity, Specificity (높을수록 우수) 및 HD95, ASD (낮을수록 우수).
- **학습 환경**: SGD 옵티마이저, 학습률 0.01, 배치 크기 24, 10,000회 반복 학습을 수행하였다.

### 결과 분석
정량적 결과(Table 1, 2)에 따르면, Mamba-UNet은 두 데이터셋 모두에서 대부분의 지표에서 가장 우수한 성능을 보였다.
- **ACDC 데이터셋**: Dice 계수 $0.9281$을 기록하며 다른 모델들(UNet: $0.9248$, Swin-UNet: $0.9188$)보다 높은 성능을 보였으며, 거리 기반 지표인 HD95($2.4645$)와 ASD($0.7677$)에서도 가장 낮은 수치를 기록하여 예측 마스크의 정밀도가 높음을 입증했다.
- **Synapse 데이터셋**: Dice 계수 $0.6429$로 UNet($0.6299$) 및 Swin-UNet($0.6178$)을 유의미하게 앞질렀으며, 특히 HD95($24.4725$) 지표에서 타 모델 대비 월등한 성능 향상을 보였다.

정성적 분석(Figure 4, 5)에서도 Mamba-UNet의 결과물이 Ground Truth에 가장 근접하며, 특히 복잡한 장기의 경계선을 더 정확하게 예측하는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 Visual Mamba 블록을 U-Net 구조에 성공적으로 통합함으로써, 트랜스포머의 전역적 수용 야드(receptive field)와 CNN의 효율성을 동시에 확보할 수 있음을 보여주었다. 특히, 연산 복잡도가 선형적임에도 불구하고 Swin-UNet과 같은 최신 트랜스포머 기반 모델보다 우수한 성능을 낸다는 점은 매우 고무적이다.

다만, 본 연구는 2D 이미지 분할에 한정되어 수행되었다는 한계가 있다. 의료 영상의 특성상 3D 볼륨 데이터 처리가 매우 중요함에도 불구하고 이에 대한 검증은 이루어지지 않았다. 또한, 구체적인 손실 함수(loss function)의 수식이나 하이퍼파라미터 튜닝 과정에 대한 상세한 설명이 부족하여, 동일한 조건에서의 재현성을 확인하기 위해서는 추가적인 정보가 필요하다.

결론적으로, Mamba-UNet은 SSM이라는 새로운 패러다임을 의료 영상 분할에 적용하여 효율적인 전역 컨텍스트 모델링의 가능성을 제시했다는 점에서 학술적 가치가 높다.

## 📌 TL;DR

Mamba-UNet은 전역적 의존성을 선형 시간 복잡도로 처리할 수 있는 Visual Mamba(VSS 블록)를 U-Net 구조에 적용한 순수 Mamba 기반 의료 영상 분할 모델이다. ACDC 및 Synapse 데이터셋 실험을 통해 기존의 CNN 및 Transformer 기반 U-Net 모델들보다 정밀한 분할 성능을 보였으며, 이는 고해상도 의료 영상 처리에서 Mamba 아키텍처가 매우 효율적인 대안이 될 수 있음을 시사한다. 향후 3D 확장 및 준지도 학습(semi-supervised learning) 적용 시 더 큰 파급력이 있을 것으로 기대된다.