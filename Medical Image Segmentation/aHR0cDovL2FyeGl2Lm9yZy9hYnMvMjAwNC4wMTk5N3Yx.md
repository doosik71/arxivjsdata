# Volumetric Attention for 3D Medical Image Segmentation and Detection

Xudong Wang, Shizhong Han, Yunqiang Chen, Dashan Gao, and Nuno Vasconcelos (2019)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상의 분할(Segmentation) 및 검출(Detection)에서 발생하는 고해상도 데이터 처리와 문맥 정보(Contextual Information) 활용 사이의 트레이드오프 문제를 해결하고자 한다.

일반적으로 3D 의료 영상 처리를 위해 3D Convolutional Neural Networks(CNN)를 사용할 수 있으나, 현재의 GPU 메모리 제한으로 인해 고해상도 3D 볼륨 전체를 처리하는 것이 불가능하다. 만약 메모리 한계로 인해 저해상도 볼륨을 사용하게 되면, 작은 병변이나 종양을 놓치거나 경계 부분이 흐릿하게 예측되는 정밀도 저하 문제가 발생한다.

이를 해결하기 위해 2D 또는 2.5D(인접 슬라이스 활용) 네트워크를 사용하는 방식이 대안으로 제시되었으나, 이러한 방식은 $z$축 방향의 문맥 정보가 부족하여 전문가가 여러 슬라이스를 동시에 확인하며 판단하는 것과 같은 수준의 정확도를 얻기 어렵다는 한계가 있다. 따라서 본 논문의 목표는 고해상도의 공간 정밀도를 유지하면서도 $z$축 방향의 3D 문맥 정보를 효율적으로 통합할 수 있는 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 비디오 처리 분야의 최신 연구에서 영감을 얻은 **Volumetric Attention (VA)** 모듈을 제안한 것이다. VA 모듈의 중심적인 설계 아이디어는 다음과 같다.

1. **2.5D 네트워크의 3D 문맥 활용**: 2D 기반 네트워크를 유지하면서 $z$축 방향의 인접 슬라이스들로부터 추출한 특징을 Attention 메커니즘을 통해 통합함으로써 3D 문맥 정보를 활용한다.
2. **사전 학습 모델(Pre-trained Models)의 활용**: 2D 네트워크 구조를 기반으로 하므로, 의료 영상 데이터셋의 부족 문제를 해결하기 위해 이미 대량의 데이터로 학습된 2D 검출 모델의 가중치를 전이 학습(Transfer Learning)에 사용할 수 있다.
3. **계산 효율성**: Global Spatial Pooling과 Global Channel Pooling을 사용하여 연산 비용을 낮추었으며, 이를 통해 어떤 CNN 아키텍처(One-stage, Two-stage detector, Segmentation network)와도 유연하게 결합 가능하다.
4. **RPN 성능 향상**: VA 모듈이 Region Proposal Network(RPN) 이전에 적용될 수 있어, 3D 문맥을 바탕으로 더 정확한 후보 영역(Proposal)을 생성함으로써 검출 누락을 줄인다.

## 📎 Related Works

기존의 3D 의료 영상 분할 및 검출 접근 방식은 크게 두 가지로 나뉜다.

- **3D CNN 기반 방식**: 3D U-Net과 같은 구조가 대표적이며, 볼륨 전체를 직접 처리한다. 그러나 앞서 언급한 GPU 메모리 제한으로 인해 고해상도 입력이 어렵다는 치명적인 한계가 있다.
- **2D/2.5D 기반 방식**: 2D U-Net이나 인접 슬라이스를 결합한 2.5D Residual U-Net 등이 제안되었다. 이들은 메모리 효율적이고 고해상도 처리가 가능하지만, $z$축 방향의 장거리 의존성(Long-range dependency)을 모델링하지 못해 성능의 상한선이 낮다.

일부 연구에서는 3D Context Enhanced Region-based CNN을 제안하였으나, 이는 RPN 기반의 구조여서 SSD나 YOLO 같은 단일 단계 검출기나 U-Net 같은 분할 네트워크에 직접 적용하기 어렵다. 또한, 제안된 방식은 RPN이 후보 영역을 생성하는 단계에서는 3D 문맥을 활용하지 못하므로, 초기에 놓친 후보 영역을 나중에 회복할 수 없다는 한계가 있다.

## 🛠️ Methodology

본 논문에서는 VA 모듈을 Mask R-CNN에 통합한 **VA Mask-RCNN** 아키텍처를 제안한다. VA 모듈은 타겟이 되는 2.5D 이미지와 그 주변의 문맥적(Contextual) 2.5D 이미지들 사이의 관계를 분석한다. 여기서 2.5D 이미지는 3개의 인접한 슬라이스를 채널 방향으로 쌓은 형태를 의미한다.

### 1. Bag of Long-range Features

$z$축 방향의 의존성을 고려하기 위해 타겟 이미지 주변의 인접 이미지들로부터 특징을 추출하여 'Bag' 형태로 구성한다.
$$X_{long}^i = [X_1, X_2, \dots, X_N] \in \mathbb{R}^{N \times C_i \times H_i \times W_i}$$
여기서 $i$는 피라미드 레벨, $N$은 문맥 이미지의 수, $C_i, H_i, W_i$는 각각 채널, 높이, 너비를 나타낸다.

### 2. Volumetric Channel Attention (VCA)

채널 간의 비선형 상호작용을 학습하여 중요한 채널을 강조하는 메커니즘이다.

- **특징 임베딩**: Global Average Pooling($F_{avg}^c$)을 적용한 후, 연산량 감소를 위해 reduction ratio가 16인 두 개의 $1 \times 1$ Convolution 레이어를 사용하여 임베딩한다.
  $$F_{emb}^c(X) = W_2 \delta(W_1 F_{avg}^c(X))$$
  ($\delta$는 ReLU 함수)
- **슬라이스 Attention 신호 계산**: 타겟 이미지와 문맥 이미지들의 임베딩 간 행렬 곱을 수행하고 Softmax를 적용하여 각 슬라이스의 중요도를 계산한다.
  $$S_{att}^c = \text{softmax}(F_{emb}^c(X_{tgt}) \cdot F_{emb}^c(X_{long})) \in \mathbb{R}^{1 \times N}$$
- **최종 적용**: 계산된 $S_{att}^c$를 $F_{emb}^c(X_{long})$에 적용한 후, ReLU $\rightarrow$ $1 \times 1$ Conv $\rightarrow$ Sigmoid 과정을 거쳐 채널 가중치 $S^c \in \mathbb{R}^{C \times 1 \times 1}$를 생성하고, 이를 타겟 특징 맵 $X_{tgt}$에 채널별 곱셈(Channel-wise multiplication)으로 적용한다.

### 3. Volumetric Spatial Attention (VSA)

공간적인 위치 정보를 바탕으로 중요한 영역을 강조하는 메커니즘이다.

- **특징 압축**: Max pooling과 Average pooling을 통해 채널 차원을 축소하여 두 개의 채널 특징 맵을 생성한다.
  $$F_{pool}^s(X) = [F_{max}^s(X), F_{avg}^s(X)] \in \mathbb{R}^{2 \times H \times W}$$
- **슬라이스 Attention 신호 계산**: 학습 가능한 컨볼루션 가중치 $W$를 통해 임베딩한 후, VCA와 유사하게 Softmax를 통해 슬라이스별 중요도를 계산한다.
  $$S_{att}^s = \text{softmax}(F_{emb}^s(X_{tgt}) \cdot F_{emb}^s(X_{long})) \in \mathbb{R}^{1 \times N}$$
- **최종 적용**: 생성된 공간 Attention 맵 $S^s \in \mathbb{R}^{1 \times H \times W}$를 $X_{tgt}$에 요소별 곱셈(Element-wise multiplication)으로 적용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Liver Tumor Segmentation (LiTS) 및 DeepLesion 데이터셋 사용.
- **지표**: LiTS는 볼륨당 평균 Dice coefficient를, DeepLesion은 이미지당 False Positives (FPs) 대비 Sensitivity(민감도)를 측정하였다.
- **전처리**: CT 영상의 HU(Hounsfield unit)를 $[-200, 300]$ 범위로 클램핑하고 정규화하였으며, 해상도는 $1024 \times 1024$로 조정하였다.

### 주요 결과

1. **LiTS 챌린지 성능**: VA Mask-RCNN은 74.1의 Dice score를 기록하며, 이전 챌린지 우승자보다 3.9 포인트 높은 성능을 보였으며 제출 당시 리더보드 최상위 성능을 달성하였다.
2. **DeepLesion 검출 성능**: ResNet50 백본의 Deformable Faster-RCNN에 VA를 추가했을 때, 0.5 FPs/image에서 69.1의 민감도를 기록하여 기존 최고 결과보다 6.6 포인트 향상되었다.
3. **소형 병변 검출**: Ablation study 결과, VA 모듈 추가 시 특히 작은 크기의 병변(diameter < 15mm)에 대한 Dice score가 크게 향상됨을 확인하였다. 이는 3D 문맥 정보가 작은 객체를 식별하는 데 결정적임을 시사한다.

### Ablation Study 결과

- **사전 학습 영향**: ImageNet $\rightarrow$ MS-COCO $\rightarrow$ DeepLesion 순으로 사전 학습을 진행했을 때 성능이 단계적으로 향상되었으며, 이는 의료 영상의 데이터 부족 문제를 극복하는 데 효과적이다.
- **VA 위치**: VA 모듈을 RPN 이전에 배치했을 때가 RCNN(ROI align 이후)에 배치했을 때보다 성능 향상 폭이 훨씬 컸다($\text{Dice } +5.1 \text{ vs } +1.7$). 이는 3D 문맥 정보가 고품질의 후보 영역을 생성하는 단계에서 필수적임을 의미한다.
- **특징 Bag 크기**: 슬라이스 수를 9개까지 늘릴 때 성능이 향상되었으며, 그 이상의 증가(예: 11개)는 오히려 소형/중형 병변 성능을 저하시키는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 2D/2.5D 네트워크가 가진 '문맥 부족' 문제와 3D 네트워크가 가진 '메모리 제한' 문제를 VA 모듈이라는 효율적인 Attention 메커니즘으로 해결하였다.

**강점 및 분석**:

- **전략적 배치**: VA 모듈을 RPN 앞에 배치함으로써, 2D 단독 모델이 완전히 놓쳤을 가능성이 있는 병변(False Negative)을 3D 문맥을 통해 사전에 포착할 수 있게 하였다. 이는 단순히 사후에 필터링하는 것보다 훨씬 강력한 성능 향상을 가져온다.
- **유연한 통합**: 특정 아키텍처에 종속되지 않고 특징 추출기 이후에 추가하는 형태이므로, 다양한 검출 및 분할 모델에 즉시 적용 가능하다는 범용성을 갖는다.

**한계 및 논의**:

- **데이터 의존성**: 사전 학습 데이터셋의 순서(Domain shift)에 따라 성능 차이가 발생하는 점은 여전히 의료 영상 분석에서 전이 학습의 설계가 중요함을 보여준다.
- **추론 시간**: Bag size가 커질수록 성능은 어느 정도 향상되지만 연산 시간이 증가하므로, 실시간 응용 분야에서는 적절한 Bag size의 선택이 필요하다.

## 📌 TL;DR

본 연구는 3D 의료 영상의 고해상도 정밀도와 $z$축 문맥 정보를 동시에 확보하기 위해 **Volumetric Attention (VA)** 모듈을 제안하였다. 이 모듈은 2.5D 네트워크가 3D 정보를 효율적으로 활용하게 하며, 특히 RPN 이전에 배치될 때 소형 병변 검출 성능을 획기적으로 높인다. 결과적으로 LiTS와 DeepLesion 데이터셋에서 SOTA 성능을 달성하였으며, 이는 데이터가 제한적인 의료 환경에서 사전 학습된 2D 모델을 활용하면서도 3D의 이점을 챙길 수 있는 실용적인 방법론을 제시한 것이다.
