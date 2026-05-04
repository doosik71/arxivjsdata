# VIG-UNET: VISION GRAPH NEURAL NETWORKS FOR MEDICAL IMAGE SEGMENTATION

Juntao Jiang, Xiyu Chen, Guanzhong Tian and Yong Liu (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 기존 딥러닝 모델들이 가진 표현력의 한계를 해결하고자 한다. 의료 영상 분할은 장기나 병변의 픽셀을 배경으로부터 식별하는 작업으로, 컴퓨터 보조 진단 및 치료의 효율성과 정확성을 높이는 데 매우 중요하다.

기존의 주류 모델인 CNN(Convolutional Neural Networks)은 이미지를 유클리드 공간(Euclidean space) 상의 픽셀 그리드로 처리하며, 최근 주목받는 Transformer 기반 모델들은 이미지를 패치(patch)의 시퀀스로 인식한다. 그러나 이러한 방식들은 이미지 내의 각 부분 간의 복잡하고 일반적인 관계를 충분히 반영하지 못하는 한계가 있다. 따라서 본 연구의 목표는 이미지의 각 부분을 노드로 설정하고 그들 사이의 연결성을 구축할 수 있는 그래프 기반 표현(Graph-based representation)을 U-shaped 아키텍처에 통합하여, 더 일반화된 특징 추출과 정교한 분할 성능을 달성하는 ViG-UNet을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지 인식 작업에서 우수한 성능을 보인 Vision GNN(ViG)의 강력한 그래프 처리 능력을 의료 영상 분할을 위한 U-Net 구조에 접목시킨 것이다. 

중심적인 설계 직관은 이미지를 단순한 그리드나 시퀀스가 아닌 그래프로 표현함으로써, 이미지 내 서로 다른 영역 간의 관계를 보다 유연하게 모델링하는 것이다. 이를 위해 인코더, 디코더, 병목 지점(bottleneck) 및 스킵 연결(skip connection)을 포함하는 U-shaped 구조 내에 Grapher Module과 Feed-forward Networks(FFNs)를 배치하여, 전역적인 맥락 정보와 지역적인 세부 정보를 동시에 효과적으로 포착하도록 설계하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 흐름과 한계를 언급한다.

1.  **CNN 기반 모델**: U-Net 및 그 변형 모델(Attention-UNet, UNet++ 등)은 대칭적인 인코더-디코더 구조와 스킵 연결을 통해 특징 추출과 픽셀 수준의 분류에서 큰 성공을 거두었다. 하지만 CNN은 기본적으로 국소적인 수용 영역(receptive field)에 의존하므로 전역적인 문맥 파악에 한계가 있다.
2.  **Transformer 기반 모델**: ViT의 성공 이후 Trans-UNet, Swin-UNet 등이 제안되었다. 이들은 이미지를 패치 시퀀스로 처리하여 전역적인 이해도를 높였으며 경쟁력 있는 성능을 보여주었다.
3.  **GNN 기반 모델**: 그래프 신경망(GNN)은 데이터 간의 관계를 일반화하여 표현할 수 있다. 특히 최근 제안된 Vision GNN(ViG)은 이미지를 블록 단위의 노드로 나누고 최근접 이웃(K-nearest neighbors)을 연결하여 그래프로 처리함으로써 이미지 인식 작업에서 높은 정확도를 기록하였다.

본 연구는 이러한 ViG의 그래프 표현 능력을 분할 작업에 적용하여, 기존의 CNN 및 Transformer 기반 U-Net 변형 모델들보다 더 뛰어난 성능을 내고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조
ViG-UNet은 대칭적인 U-shape 구조를 가지며, 크게 **Stem Block $\rightarrow$ Encoder $\rightarrow$ Bottleneck $\rightarrow$ Decoder $\rightarrow$ Final Layer**의 순서로 구성된다. 각 단계에서 특징 맵의 해상도는 조절되며, 인코더의 각 단계 출력은 디코더의 대응하는 단계 입력에 더해지는 스킵 연결(skip connection)을 통해 공간 정보를 보존한다.

### 주요 구성 요소 및 역할
1.  **Stem Block**: 입력 이미지에 두 개의 합성곱 계층(stride 1, 2)을 적용하여 해상도를 $\frac{H}{2} \times \frac{W}{2}$로 줄이고, 위치 임베딩(position embedding)을 추가하여 초기 시각적 임베딩을 생성한다.
2.  **Grapher Module**: 이미지의 패치를 노드로 간주하고 $K$개의 최근접 이웃(KNN)을 연결하여 그래프 구조 $G=(V, E)$를 구축한다. 이후 그래프 합성곱을 통해 이웃 노드의 정보를 집계(aggregation)하고 업데이트(update)한다.
3.  **Feed-forward Networks (FFNs)**: Grapher Module 이후에 배치되어 특징 변환 능력을 높이고, GNN에서 흔히 발생하는 오버스무딩(over-smoothing) 현상을 완화한다.
4.  **Downsampling & Upsampling**: 인코더에서는 stride 2의 합성곱 계층을 통해 다운샘플링을 수행하며, 디코더에서는 Bilinear interpolation과 합성곱 계층을 결합하여 업샘플링을 수행한다.

### 상세 알고리즘 및 방정식

#### 1. Grapher Module의 작동 원리
노드 $v_i$의 업데이트된 특징 $x'_i$는 다음과 같이 집계 함수 $g(\cdot)$와 업데이트 함수 $h(\cdot)$의 조합으로 정의된다.

$$x'_i = h(x_i, g(x_i, N(x_i); W_{aggregate}); W_{update})$$

여기서 집계 함수 $g(\cdot)$는 **Max-relative graph convolution**을 사용하여 이웃 노드와의 차이 중 최대값을 추출한다.

$$g(\cdot) = x''_i = [x_i, \max(\{x_j - x_i | j \in N(x_i)\})]$$

업데이트 함수 $h(\cdot)$는 학습 가능한 가중치 $W_{update}$와 편향 $b_h$를 적용한다.

$$h(\cdot) = x'_i = x''_i W_{update} + b_h$$

또한, 멀티헤드(multi-head) 구조를 채택하여 각 헤드별로 서로 다른 가중치를 적용한 후 이를 다시 연결(concatenate)하여 최종 특징을 얻는다.

최종적으로 Grapher Module의 출력 $Y$는 다음과 같은 잔차 연결(shortcut)과 활성화 함수를 거친다.

$$Y = \text{Droppath}(\text{GELU}(\text{GraphConv}(X_1)W_{out} + b_{out}) + X)$$

#### 2. Feed-forward Network (FFN)
FFN은 다음과 같이 두 번의 선형 변환과 비선형 활성화 함수를 거쳐 특징을 변환한다.

$$Z = \text{Droppath}(\text{GELU}(Y W_1 + b_1)W_2 + b_2) + Y$$

### 훈련 절차 및 손실 함수
모델은 ADAM 옵티마이저와 CosineAnnealingLR 스케줄러를 사용하여 학습되었다. 손실 함수로는 이진 교차 엔트로피(BCE) 손실과 Dice 손실을 결합한 혼합 손실 함수를 사용한다.

$$L = 0.5\text{BCE}(\hat{y}, y) + \text{Dice}(\hat{y}, y)$$

## 📊 Results

### 실험 설정
- **데이터셋**: ISIC 2016 (피부 병변), ISIC 2017 (피부 병변), Kvasir-SEG (폴립 이미지).
- **비교 모델**: UNet, Attention-UNet, UNet++, Trans-UNet, Swin-UNet, UNext.
- **측정 지표**: IoU (Intersection over Union), Dice Coefficient.
- **입력 크기**: $512 \times 512$.

### 정량적 결과
실험 결과, ViG-UNet은 세 가지 모든 데이터셋에서 비교 대상 모델들보다 우수한 성능을 보였다.

| Methods | ISIC 2016 (IoU) | ISIC 2017 (IoU) | Kvasir-SEG (IoU) |
| :--- | :---: | :---: | :---: |
| UNet | 0.8209 | 0.6410 | 0.6913 |
| Trans-UNet | 0.8481 | 0.7147 | 0.4943 |
| Swin-UNet | 0.7559 | 0.6676 | 0.3405 |
| **ViG-UNet** | **0.8558** | **0.7211** | **0.7104** |

Dice 지표에서도 마찬가지로 ViG-UNet이 가장 높은 수치를 기록하며 최첨단(SOTA) 성능을 입증하였다.

### 모델 복잡도
파라미터 수 분석 결과, ViG-UNet은 약 $0.7\text{G}$ (7억 개)의 파라미터를 가지고 있어, UNet(7.8M)이나 UNext(1.5M) 등 다른 모델들에 비해 월등히 크다.

## 🧠 Insights & Discussion

### 강점
ViG-UNet은 그래프 기반의 관계 모델링을 통해 이미지의 전역적 문맥을 효과적으로 포착함으로써, 기존의 CNN 및 Transformer 기반 모델보다 정교한 분할 성능을 구현하였다. 특히 데이터셋의 특성에 관계없이 일관되게 높은 성능 향상을 보였다는 점이 고무적이다.

### 한계 및 비판적 해석
1.  **계산 비용의 급격한 증가**: 성능 향상은 뚜렷하지만, 파라미터 수가 다른 모델들에 비해 압도적으로 많다($0.7\text{G}$). 이는 메모리 사용량 증가와 추론 속도 저하로 이어지며, 실제 의료 현장의 실시간 진단 시스템에 적용하기에는 상당한 부담이 될 수 있다.
2.  **사전 학습의 부재**: 본 실험에서는 ViG-UNet을 처음부터 학습시켰으나, 저자들은 ImageNet으로 사전 학습된 ViG 모델을 사용한다면 성능이 더 향상될 가능성이 있다고 언급한다. 이는 현재의 결과가 모델의 잠재력을 완전히 끌어낸 것이 아닐 수 있음을 시사한다.
3.  **데이터 효율성**: 비교적 작은 규모의 데이터셋에서도 높은 성능을 냈으나, 모델의 크기가 매우 크기 때문에 과적합(overfitting)의 위험이 존재한다. 이에 대한 구체적인 정규화 분석이나 효율적인 파라미터 최적화 방안이 제시되지 않은 점은 아쉽다.

## 📌 TL;DR

본 논문은 이미지의 관계를 그래프로 모델링하는 **Vision GNN(ViG)**을 **U-Net** 구조에 통합한 **ViG-UNet**을 제안한다. 이 모델은 피부 병변 및 폴립 분할 데이터셋에서 기존 CNN 및 Transformer 기반 모델들을 능가하는 최고의 IoU 및 Dice 성능을 달성하였다. 비록 파라미터 수가 매우 많다는 단점이 있으나, 그래프 기반 표현이 의료 영상 분할의 정확도를 획기적으로 높일 수 있음을 증명하였으며, 향후 사전 학습 모델의 적용이나 경량화 연구를 통해 실용성을 높일 수 있을 것으로 기대된다.