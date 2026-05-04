# The Fully Convolutional Transformer for Medical Image Segmentation

Athanasios Tragakis, Chaitanya Kaul, Roderick Murray-Smith, Dirk Husmeier (2022)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 병변의 경계를 정밀하게 탐지하여 종양이나 암 영역을 식별하는 데 필수적이며, 이는 진단 속도와 환자의 예후를 개선하는 데 매우 중요하다. 기존의 접근 방식은 크게 두 가지 흐름으로 나뉜다. 첫째는 UNet과 같은 CNN 기반 모델로, 지역적 특징 추출에는 뛰어나지만 Convolution 연산의 본질적인 국소성(locality)으로 인해 입력 영상의 장거리 의미론적 의존성(long-range semantic dependencies)을 포착하지 못한다는 한계가 있다. 둘째는 Transformer 기반 모델로, 전역적 문맥(global context) 포착에는 유리하지만 CNN과 같은 공간적 문맥(spatial context) 활용 능력이 부족하고, 모델이 비대하며 많은 데이터를 필요로 하는 경향이 있다.

본 논문의 목표는 CNN의 효율적인 이미지 표현 학습 능력과 Transformer의 장거리 의존성 포착 능력을 결합하여, 의료 영상의 정밀한 특성을 반영하면서도 가볍고 강력한 성능을 가진 '완전 합성곱 트랜스포머(Fully Convolutional Transformer, FCT)' 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 구조 내에서 선형 투영(linear projection)과 같은 비-합성곱 연산을 제거하고, 이를 Depthwise-Convolution으로 대체하여 모델의 전 과정을 완전 합성곱 형태로 구성하는 것이다. 이를 통해 다음과 같은 기여를 한다.

- **Fully Convolutional Transformer (FCT) 제안**: 의료 영상 분할을 위한 최초의 완전 합성곱 트랜스포머 모델을 제안하며, 기존의 CNN 및 Transformer 기반 아키텍처보다 우수한 성능을 보인다.
- **Convolutional Attention 모듈**: Depthwise-Convolution을 사용하여 중첩된 패치(overlapping patches)를 생성함으로써 Positional Encoding 없이도 공간적 문맥을 유지하며 장거리 의미론적 문맥을 학습한다.
- **Wide-Focus 모듈**: 다중 해상도 확장 합성곱(multi-resolution dilated convolutions)을 통해 국소적 영역에서 전역적 영역으로 확장되는 계층적 문맥을 캡처하여 정밀한 분할 성능을 높인다.
- **효율성 및 성능 입증**: 기존 모델(예: nnFormer) 대비 파라미터 수를 최대 5배 줄이면서도 다양한 모달리티의 데이터셋에서 SOTA(State-of-the-art) 성능을 달성하였다.

## 📎 Related Works

- **초기 CNN 및 Attention 모델**: UNet은 의료 영상 분할의 표준이 되었으며, Attention UNet이나 FocusNet 등은 gating function이나 attention gating을 통해 특징 전파를 개선하려 했다. nnUNet은 데이터 전처리와 아키텍처 선택을 자동화하여 높은 성능을 보였다.
- **Transformer 모델**: ViT의 등장 이후 CvT, CCT, Swin Transformer 등이 이미지의 공간적 문맥을 보완하며 발전했다. 의료 분야에서는 TransUNet, UNETR, Swin UNETR와 같은 CNN-Transformer 하이브리드 모델이 제안되었으나, 이는 모델을 비대하고 계산 복잡하게 만드는 단점이 있다.
- **최근 연구 (Concurrent Works)**: Swin UNet과 같이 순수 Transformer 기반 모델이나, nnFormer, D-Former처럼 Transformer 블록 자체를 의료 영상에 맞게 개선하려는 시도가 있었다. 그러나 이러한 모델들은 여전히 Attention 투영 및 특징 처리 과정에서 선형적인(linear) 성격이 강해 공간적 문맥 활용에 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조
FCT는 전형적인 UNet의 대칭적 인코더-디코더 구조를 따른다. 인코더는 4개의 FCT 레이어로 구성되어 특징을 추출하고 전파하며, 디코더는 이를 다시 업샘플링하여 분할 맵을 생성한다. 인코더와 디코더 사이에는 동일 해상도의 특징 맵을 결합하는 Skip Connection이 존재한다.

### FCT 레이어의 구성 요소
FCT 레이어는 크게 **Convolutional Attention**과 **Wide-Focus** 모듈로 구성된다.

**1. Convolutional Attention**
- **패치 임베딩**: $\text{LayerNormalization} \to \text{Conv} \to \text{Conv} \to \text{MaxPool}$ 과정을 거친 후, Depthwise-Convolution을 통해 토큰 맵을 생성한다. 이때 stride를 조절하여 패치들이 서로 중첩되게 함으로써 Positional Encoding 없이도 공간 정보를 유지한다.
- **Convolutional MHSA**: 표준 Transformer의 Multi-Head Self Attention(MHSA)에서 사용되는 선형 투영(linear projection)을 Depthwise-Convolution으로 대체한다.
- MHSA의 연산 과정은 다음과 같다:
  $$\text{MHSA}(z^{l-1}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$
  여기서 $Q, K, V$는 Depthwise-Convolution을 통해 생성된 쿼리, 키, 값 벡터이다.

**2. Wide-Focus 모듈**
MHSA의 출력 결과에 대해 다양한 수용 영역(receptive field)을 갖도록 설계된 다중 분기 합성곱 레이어를 적용한다.
- 한 분기는 일반적인 공간 합성곱(spatial convolution)을 수행하고, 다른 분기들은 선형적으로 증가하는 확장률(dilation rate)을 가진 확장 합성곱(dilated convolutions)을 수행한다.
- 이렇게 얻어진 특징들을 합산(summation)한 후, 최종적으로 공간 합성곱 연산자를 통해 특징을 집계(feature aggregation)한다.

### 학습 및 추론 절차
- **인코더**: 이미지 피라미드 형태의 입력을 사용하여 다양한 스케일에서 ROI(Region of Interest) 특징을 강조한다.
- **디코더**: bottleneck 표현을 입력으로 받아 업샘플링과 FCT 레이어를 통해 분할 맵을 재구성한다.
- **Deep Supervision**: 최하위 스케일(28x28)을 제외한 각 스케일에서 중간 분할 맵을 출력하여 추가적인 감독 학습을 수행한다. 최하위 스케일을 제외한 이유는 ROI가 너무 작아 배경으로 오인될 가능성이 크기 때문이다.

## 📊 Results

### 실험 설정
- **데이터셋**: ACDC(MRI), Synapse(CT), Spleen(CT), ISIC 2017(Dermoscopy)의 4가지 다양한 모달리티 데이터셋을 사용하였다.
- **지표 및 구현**: Dice coefficient를 주 지표로 사용하였다. Loss 함수는 Cross Entropy와 Dice Loss를 동일 가중치로 결합하여 사용했으며, Adam 옵티마이저(LR 1e-3)를 적용하였다.
- **모델 규모**: FCT는 31.7M의 파라미터와 7.87 GFLOPs의 연산량을 가진다.

### 정량적 결과
- **ACDC 데이터셋**: $384 \times 384$ 입력 크기에서 93.02%의 Average Dice score를 기록하며 SOTA를 달성하였다. 특히 nnFormer(158.92M 파라미터)보다 모델 크기는 5배 작으면서 성능은 더 뛰어났다.
- **Synapse 데이터셋**: TransUNet, LeViT-UNet, Swin UNet 등 기존 모델들을 상당한 차이로 앞질렀다 (Average Dice 83.53%).
- **이진 분할(Spleen, ISIC 2017)**: Spleen 데이터셋에서 SETR, CoTr 등을 1.2% 이상 앞섰으며, ISIC 2017에서는 Boundary Aware Transformer보다 1.1% 높은 Dice score와 더 높은 민감도(Sensitivity)를 보였다.
- **ACDC 온라인 리더보드**: 512x512 해상도로 학습된 $\text{FCT}_{512}$ 모델이 Average Dice 93.13%로 기존 Top 5 기록을 경신하며 새로운 SOTA를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 Transformer의 강력한 전역 문맥 포착 능력과 CNN의 정밀한 지역 특징 추출 능력을 '완전 합성곱'이라는 틀 안에서 성공적으로 통합하였다.

- **Wide-Focus 모듈의 중요성**: 특히 ISIC 2017의 암 경계 분할에서 높은 민감도를 보인 것은, Wide-Focus 모듈이 다양한 수용 영역을 통해 계층적 특징 정보를 효과적으로 캡처했기 때문으로 분석된다.
- **확장 합성곱의 한계**: Ablation study를 통해 확장 분기가 3개를 넘어갈 경우 오히려 성능이 포화되거나 감소함을 발견하였다. 이는 확장 커널이 깊은 레이어에서 전역 커널을 제대로 근사하지 못하고 핵심 특징 정보를 놓치기 때문으로 해석된다.
- **Deep Supervision의 전략적 적용**: 모든 스케일에 Deep Supervision을 적용하는 것보다, 너무 작은 해상도의 최하위 스케일을 제외하는 것이 성능 향상에 도움이 됨을 확인하였다. 이는 모델이 작은 ROI를 배경으로 예측하려는 강한 편향(bias)을 방지하기 위함이다.

## 📌 TL;DR

본 연구는 의료 영상 분할을 위해 Depthwise-Convolution 기반의 **Convolutional Attention**과 다중 해상도 확장 합성곱 기반의 **Wide-Focus** 모듈을 결합한 **Fully Convolutional Transformer (FCT)**를 제안한다. 이 모델은 기존의 하이브리드 또는 순수 Transformer 모델보다 훨씬 적은 파라미터(nnFormer 대비 1/5 수준)만으로도 다양한 의료 영상 데이터셋에서 SOTA 성능을 달성하였다. 특히 공간적 문맥 유지와 계층적 특징 추출 능력이 뛰어나, 향후 다양한 의료 영상 처리 작업의 효율적인 백본(backbone) 네트워크로 활용될 가능성이 매우 높다.