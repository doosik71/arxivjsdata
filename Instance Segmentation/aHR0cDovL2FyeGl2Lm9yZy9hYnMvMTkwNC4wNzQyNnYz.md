# SPRNet: Single Pixel Reconstruction for One-stage Instance Segmentation

Jun Yu, Jinghan Yao, Jian Zhang, Zhou Yu, and Dacheng Tao (2019)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전의 핵심 과제 중 하나인 인스턴스 분할(Instance Segmentation)의 효율성과 정확도 사이의 트레이드오프 문제를 해결하고자 한다. 기존의 인스턴스 분할 방식은 대부분 Mask R-CNN과 같이 Region Proposal Network(RPN)를 사용하는 2단계(Two-stage) 구조를 채택하고 있다. 이러한 방식은 정교한 분할 결과를 생성하지만, 다수의 RoI(Region of Interest)를 처리해야 하는 구조적 특성상 추론 속도가 느려 실제 실시간 응용 분야에 적용하기에는 한계가 있다.

반면, 1단계(One-stage) 검출기는 속도가 매우 빠르지만, RoI의 도움 없이 합성곱 특징 맵(Convolution feature map)에서 클래스 간(between classes) 및 클래스 내(within classes) 인스턴스를 동시에 구분하여 픽셀 단위의 마스크를 생성하는 것이 매우 어렵다. 따라서 본 논문의 목표는 2단계 방식의 정확도에 근접하면서도 1단계 방식의 효율성을 갖춘 새로운 인스턴스 분할 프레임워크를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **단일 픽셀 재구성(Single Pixel Reconstruction, SPR)** 브랜치를 도입하여 1단계 검출기에서도 효율적인 인스턴스 분할을 수행하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **SPRNet 프레임워크 제안**: 별도의 RoI Pooling이나 RoI Align 과정 없이, 특징 맵의 단일 픽셀로부터 인스턴스 마스크를 직접 재구성하는 1단계 인스턴스 분할 모델을 제안한다.
2. **Gate-FPN (GFPN) 설계**: 기존 Feature Pyramid Network(FPN)의 단순 합산(element-wise summation) 방식이 유발하는 공간적 편이(Spatial shift)와 원치 않는 기울기 전파(Gradient propagation) 문제를 해결하기 위해, 적응형 게이팅 메커니즘을 도입한 GFPN을 제안하여 검출 및 분할 성능을 전반적으로 향상시킨다.
3. **효율성과 정확도의 조화**: ResNet-50 백본을 사용할 때, Mask R-CNN과 비교하여 경쟁력 있는 Mask AP를 유지하면서도 더 빠른 추론 속도를 달성하며, RetinaNet 대비 BBox AP를 개선한다.

## 📎 Related Works

인스턴스 분할 연구는 크게 두 가지 방향으로 나뉜다.
- **검출 의존적 방법(Detection-dependent)**: Faster R-CNN 등의 검출 모델에 마스크 예측 브랜치를 추가하는 방식이다. Mask R-CNN이 대표적이며 높은 정확도를 보이지만, RoI 제안 단계로 인해 속도가 느리다. FCIS 같은 모델은 제안 단계를 통합하려 했으나 겹쳐진 인스턴스 분리나 경계 생성에서 오류가 발생하는 한계가 있었다.
- **검출 독립적 방법(Detection-free)**: 시맨틱 분할(Semantic Segmentation) 모델을 확장하여 픽셀을 분류한 후, 클러스터링 알고리즘을 통해 인스턴스를 구분하는 방식이다.

또한, 다중 스케일 특징 학습을 위해 FPN이 널리 사용된다. FPN은 하향식(Top-down) 경로를 통해 고수준의 시맨틱 정보를 저수준 특징 맵에 전달하지만, 단순 합산 과정에서 고수준 특징의 부정확한 공간 정보가 저수준 특징의 정교한 정보를 훼손하는 문제가 존재한다.

## 🛠️ Methodology

SPRNet은 인코더-디코더 구조를 가진 1단계 프레임워크로, RetinaNet을 기반으로 하여 클래스, 박스, 마스크 예측을 위한 세 개의 병렬 브랜치를 구성한다.

### 1. Gate-FPN (GFPN)
특징 맵의 각 픽셀이 충분한 정보를 담도록 하기 위해 GFPN을 도입한다. 기존 FPN의 단순 합산을 대체하여, 공유된 분리 가능 합성곱(Shared separable convolution)과 시그모이드(Sigmoid) 활성화 함수를 통해 두 특징 맵의 가중치를 결정하는 게이트를 생성한다.

수식으로 표현하면 다음과 같다. 백본 특징 맵 $x_1, x_2$에 대해:
$$f_1 \leftarrow x_1 \cdot \text{sigmoid}(x_1 w_s + b_s)$$
$$f_2 \leftarrow x_2 \cdot \text{sigmoid}(x_2 w_s + b_s)$$
$$y \leftarrow f_1 + f_2$$
여기서 $w_s$와 $b_s$는 공유된 가중치와 편향이다. 이 메커니즘은 각 레벨의 특징 학습에서 어떤 백본 특징 맵이 주된 기여를 할지 네트워크가 적응적으로 결정하게 하여, 공간적 편이를 방지하고 기울기 전파를 제어한다.

### 2. Mask Branch (SPR 과정)
마스크 브랜치는 단일 픽셀로부터 $32 \times 32$ 크기의 마스크를 재구성하는 디코더 구조로 설계되었다.

- **Multi-scale Fusion**: 단일 픽셀에 충분한 수용 영역(Receptive field)을 부여하기 위해, $1 \times 1$ 합성곱 1개와 서로 다른 다이레이션(Dilation rate: 2, 4, 6)을 가진 $3 \times 3$ 합성곱 3개를 병렬로 적용한 뒤 채널 방향으로 결합(Concatenation)한다.
- **Positive Pixel Sampling**: 앵커 박스와 인스턴스 간의 IoU가 $0.7$ 이상인 픽셀을 양성 샘플(Positive pixel)로 정의한다. 이 임계값은 학습의 안정성과 추론 시의 강건함을 조절하는 핵심 요소이다.
- **Single Pixel Reconstruction**: 선택된 양성 픽셀을 입력으로 하여 연속적인 디컨볼루션(Deconvolution) 층을 통과시킨다. 처음 3개 층은 활성화 함수 없이 $8 \times 8$ 맵을 생성하고, 이후 ReLU를 포함한 2개 층을 더 거쳐 최종 $32 \times 32$ 스코어 맵을 생성한다. 또한 $8 \times 8$ 맵에서 최종 층으로 이어지는 최근접 보간(Nearest interpolation) 숏컷을 추가하여 성능을 높였다.

### 3. 학습 및 추론 절차
- **손실 함수**: 클래스 분류에는 Focal Loss, 박스 회귀에는 Smoothed $L_1$ Loss, 마스크 예측에는 Binary Cross-Entropy Loss를 사용한다.
- **추론**: 분류 브랜치에서 점수가 가장 높은 상위 100개 픽셀을 샘플링하여 마스크를 재구성한 뒤, 회귀 브랜치에서 예측된 실제 박스 크기에 맞게 쌍선형 보간(Bilinear interpolation)으로 리사이징한다.

## 📊 Results

### 실험 설정
- **데이터셋**: MS-COCO 2017
- **지표**: AP, $\text{AP}_{50}$, $\text{AP}_{75}$, $\text{AP}_S$, $\text{AP}_M$, $\text{AP}_L$
- **비교 대상**: Mask R-CNN, FCIS, MNC 및 RetinaNet(GFPN 효과 검증용)

### 주요 결과
1. **인스턴스 분할 성능**: ResNet-50-GFPN 백본을 사용한 SPRNet은 Mask R-CNN(ResNet-50-FPN) 대비 Mask AP가 약간 낮으나, 추론 속도는 약 30% 더 빠르다 (Mask R-CNN: 7 fps $\rightarrow$ SPRNet: 9 fps).
2. **객체 검출 성능**: GFPN의 도입으로 RetinaNet보다 우수한 BBox AP를 기록했다. 특히 작은 객체($\text{AP}_S$) 검출에서 괄목할만한 향상을 보였으며, 일부 설정에서는 ResNet-101 기반의 2단계 검출기보다 높은 $\text{AP}_S$를 달성하였다.
3. **재현율(Recall)**: 1단계 프레임워크의 특성상 Mask R-CNN보다 더 많은 객체를 검출하는 경향(Higher Average Recall)을 보였다.
4. **GFPN 효과**: FPN과 GFPN을 비교한 실험에서 GFPN이 모든 스케일의 객체 검출에서 일관되게 높은 성능을 보였으며, 특히 큰 객체 검출에서 기울기 차단 효과로 인해 성능이 크게 향상되었다.

## 🧠 Insights & Discussion

### 강점
- **효율적인 구조**: RoI 기반의 무거운 연산을 제거하고 단일 픽셀 재구성 방식을 도입함으로써 실시간성에 가까운 속도를 확보했다.
- **특징 융합의 최적화**: GFPN을 통해 단순 합산의 문제점을 해결하고, 다이레이션 합성곱을 통해 단일 픽셀에 광범위한 컨텍스트 정보를 압축적으로 담아내는 데 성공했다.
- **높은 재현율**: 2단계 모델이 놓치기 쉬운 객체들을 더 많이 찾아내는 특성을 보인다.

### 한계 및 비판적 해석
- **경계 정밀도 부족**: Mask R-CNN은 최종 박스 내부에서 정교한 이진 분할을 수행하므로 경계선이 매우 정확하지만, SPRNet은 단일 픽셀로부터 마스크를 '복원'하는 방식이기에 세밀한 경계 묘사 능력은 상대적으로 떨어진다.
- **박스 회귀 의존성**: 1단계 구조이므로 박스 회귀 결과가 부정확할 경우, 리사이징 과정에서 마스크가 정렬되지 않는(Mis-aligned) 문제가 발생할 가능성이 크다.
- **하이퍼파라미터 민감도**: 양성 픽셀 샘플링을 위한 IoU 임계값($0.7$) 설정이 학습 안정성에 큰 영향을 미친다는 점은 모델의 일반화 성능에 제약이 될 수 있다.

## 📌 TL;DR

본 논문은 RoI Pooling 없이 단일 픽셀로부터 마스크를 재구성하는 1단계 인스턴스 분할 모델인 **SPRNet**을 제안한다. 특히 적응형 게이팅 메커니즘을 갖춘 **Gate-FPN**을 통해 특징 융합 성능을 높였으며, 다이레이션 합성곱 기반의 인코더와 디컨볼루션 기반의 디코더를 통해 효율적인 마스크 생성을 구현하였다. 결과적으로 Mask R-CNN 수준의 정확도에 근접하면서도 추론 속도를 향상시켰으며, 특히 객체 검출 재현율과 작은 객체 인식 성능을 크게 개선하였다. 이 연구는 실시간 인스턴스 분할 시스템 구축을 위한 실용적인 방향성을 제시한다.