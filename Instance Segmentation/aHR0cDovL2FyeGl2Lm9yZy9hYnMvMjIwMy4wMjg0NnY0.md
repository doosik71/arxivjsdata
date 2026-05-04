# Region Proposal Rectification Towards Robust Instance Segmentation of Biological Images

Qilong Zhangli et al. (2022)

## 🧩 Problem to Solve

본 논문은 생물학적 이미지(biological images)의 인스턴스 분할(instance segmentation)에서 발생하는 **불완전한 분할(incomplete segmentation)** 문제를 해결하고자 한다.

생물학적 이미지는 텍스처가 불균일하고 경계가 불분명하며, 객체들이 서로 밀착되어 있는 특성이 있다. 현재의 Top-down 방식의 인스턴스 분할 프레임워크는 객체 검출(object detection) 성능은 뛰어나지만, **과도한 크롭(over-crop)** 문제에 취약하다. 즉, 초기 단계에서 생성된 Region Proposal(바운딩 박스)이 객체의 경계를 정확하게 포함하지 못하고 일부를 잘라낼 경우, 이후 단계에서 수행되는 마스크 생성 과정에서 객체의 일부가 유실된 불완전한 마스크가 생성되는 문제가 발생한다.

생물학적 이미지 분석에서는 객체의 모양, 부피와 같은 형태학적 특성(morphological properties)을 정확히 측정하는 것이 매우 중요하므로, 이러한 불완전한 분할 문제를 해결하여 완전한 세그멘테이션 마스크를 얻는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Region Proposal Rectification (RPR)** 모듈을 도입하여, 초기 제안된 바운딩 박스의 위치를 정밀하게 교정하는 것이다.

RPR 모듈의 중심 직관은 **"제한된 ROI 내부의 정보만으로는 박스를 정확히 교정하기 어려우며, 주변 이웃 정보(neighbor information)를 함께 고려해야 한다"**는 점이다. 이를 위해 본 논문은 다음 두 가지 핵심 설계를 제안한다.

1. **Progressive ROIAlign**: ROI 영역을 점진적으로 확장하여 주변 맥락 정보를 단계적으로 수집한다.
2. **Attentive Feed-Forward Network (FFN)**: Self-attention 메커니즘을 통해 원본 ROI와 확장된 ROI 간의 공간적 관계를 분석하여 바운딩 박스의 좌표를 최적화한다.

## 📎 Related Works

인스턴스 분할의 접근 방식은 크게 Bottom-up과 Top-down으로 나뉜다.

- **Bottom-up 방식**: 전체 이미지에 대해 먼저 시맨틱 분할(semantic segmentation)을 수행한 후, 윤곽선(contours)이나 픽셀 유사도 등의 특징을 이용해 개별 인스턴스를 분리한다. 이 방식은 과잉 분할(over-segmentation)이나 과소 분할(under-segmentation)을 방지하기 위해 정교한 특징 설계가 필요하다는 한계가 있다.
- **Top-down 방식**: 먼저 객체성(objectness) 특징을 이용해 Region Proposal(바운딩 박스)을 생성하고, 해당 ROI 내부에서 마스크를 예측한다. 이 방식은 객체 검출에는 유리하지만, 앞서 언급한 것처럼 제안된 박스가 객체를 완전히 포함하지 못할 경우 마스크가 잘려 나가는 문제가 발생한다.

본 논문은 기존의 Top-down 방식이 가진 '제한된 뷰(limited view)' 문제를 해결하기 위해 Transformer 기반의 Attention 구조를 제안 박스 교정에 도입했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

전체 시스템은 **Baseline Framework $\rightarrow$ RPR Module $\rightarrow$ Mask Head** 순으로 구성된다.

### 1. Baseline Framework

본 논문은 앵커 기반(anchor-based)인 **Mask R-CNN**과 앵커 프리(anchor-free) 방식인 **CenterMask** 두 가지를 베이스라인으로 사용한다.

- **구조**: Backbone(ResNet50)과 FPN을 통해 다중 스케일 특징 맵 $F$를 추출하고, Box Head를 통해 초기 Region Proposal 박스들을 생성한다.

### 2. Region Proposal Rectification (RPR) Module

RPR 모듈은 초기 제안된 박스 $B$를 입력받아 교정된 박스 $B_{rec}$를 출력한다.

#### (1) Progressive ROIAlign

단일 ROI의 한계를 극복하기 위해 박스 영역을 점진적으로 확장하여 $K$개의 ROI 세트를 생성한다.

- **확장 원리**: 확장 비율 $r_i$를 다음과 같이 정의하여 $K$번의 반복을 통해 박스를 넓힌다.
  $$r_i = r_{i-1} + \frac{\rho}{K}, \quad r_0 = 0$$
- **좌표 계산**: 확장된 박스 $B^{expand}_i = (\hat{x}^i_1, \hat{y}^i_1, \hat{x}^i_2, \hat{y}^i_2)$는 다음과 같이 계산된다.
  $$\hat{x}^i_1 = \max(x_1 \times (1-r_i), 0), \quad \hat{y}^i_1 = \max(y_1 \times (1-r_i), 0)$$
  $$\hat{x}^i_2 = \min(x_2 \times (1+r_i), \hat{W}), \quad \hat{y}^i_2 = \min(y_2 \times (1+r_i), \hat{H})$$
  (여기서 $\rho=0.4, K=5$를 사용하며, $\hat{W}, \hat{H}$는 이미지의 너비와 높이이다.)
- 이후 각 확장된 박스에 대해 $\text{ROIAlign}$을 수행하여 $K$개의 ROI 특징들을 추출한다.

#### (2) Attentive FFN

추출된 $K$개의 ROI 특징 중 원본 ROI($\text{ROI}_{ori}$)를 쿼리(Query)로, 확장된 ROI들($\text{ROIs}_{expand}$)을 키(Key)와 밸류(Value)로 사용하여 Self-attention을 수행한다.

- **어텐션 메커니즘**: 채널 차원을 $C=256$에서 $\hat{C}=64$로 투영한 후, 다음 식을 통해 어텐션 특징 $\text{ROI}_{att}$를 계산한다.
  $$\text{ROI}_{att} = Q + V \cdot \text{softmax}\left(\frac{(Q^T K)^T}{\sqrt{\hat{C}}}\right)$$
- 최종적으로 $\text{ROI}_{att}$는 컨볼루션 층과 전결합 층(FC layer)을 거쳐 교정된 박스 좌표 $B_{rec}$로 변환된다.

### 3. 학습 절차 및 손실 함수

전체 네트워크는 다음과 같은 통합 손실 함수를 통해 학습된다.
$$L = L_{box} + L_{mask} + L_{RPR}$$
여기서 $L_{box}$와 $L_{mask}$는 베이스라인 모델의 손실 함수를 따르며, $L_{RPR}$은 교정된 박스의 정확도를 높이기 위해 **Smooth $L_1$ loss**를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Urothelial Cell(요로상피세포), Plant Phenotyping(식물 표현형), DSB2018(세포핵)의 3가지 생물학적 데이터셋을 사용하였다.
- **지표**: $\text{AP}_{bbox}$ (바운딩 박스 정밀도)와 $\text{AP}_{mask}$ (마스크 정밀도)를 사용하였으며, IoU 임계값 0.5에서 0.95까지 0.05 간격으로 평균을 낸 AP 값을 측정하였다.

### 정량적 결과

RPR 모듈을 적용했을 때 모든 데이터셋에서 성능 향상이 관찰되었다.

- **Mask R-CNN + RPR**: Plant 데이터셋에서 $\text{AP}_{bbox}$가 2.59pt, $\text{AP}_{mask}$가 1.41pt 상승하였다.
- **CenterMask + RPR**: Urothelial Cell 데이터셋에서 $\text{AP}_{bbox}$가 4.65pt, $\text{AP}_{mask}$가 1.94pt 상승하며 뚜렷한 개선을 보였다.

### 정성적 및 분석 결과

- **마스크 완전성**: 시각적 결과 분석 시, 베이스라인에서 나타나던 '잘려 나간 마스크' 문제가 RPR 적용 후 현저히 줄어들고 객체의 전체 형태가 온전하게 보존되었다.
- **IoU 분석**: RPR 적용 후 GT 박스와 제안 박스 간의 IoU가 크게 상승하였다. 특히 Mask R-CNN에서 개선 폭이 컸는데, 이는 Mask R-CNN의 초기 제안 박스들이 CenterMask보다 덜 정교했기 때문에 교정 모듈의 효과가 더 크게 나타난 것으로 해석된다.
- **교정 비율 분석**: 교정된 박스의 비율이 높아질수록 $\text{AP}_{bbox}$와 $\text{AP}_{mask}$가 모두 상승하는 경향을 보였으며, 특히 $\text{AP}_{bbox}$의 상승 폭이 더 컸다.

## 🧠 Insights & Discussion

### 강점

본 논문은 Top-down 방식의 고질적인 문제인 '불완전한 마스크' 문제를 해결하기 위해, 단순히 네트워크를 깊게 쌓는 것이 아니라 **ROI의 시야(View)를 확장하고 Attention을 통해 필요한 정보를 선택적으로 수집**하는 구조적 접근을 취했다는 점에서 강점이 있다. 또한, 앵커 기반과 앵커 프리 방식 모두에 적용 가능하다는 범용성을 입증하였다.

### 한계 및 미해결 과제

실험 결과 RPR 모듈이 주로 객체의 경계 부분(boundary)을 교정하는 데 효과적임이 밝혀졌다. 그러나 세포의 경계가 매우 흐릿하여(blurred) 객체 구분이 어려운 극단적인 케이스에 대해서는 여전히 강건성(robustness)을 확보해야 하는 과제가 남아 있다.

### 비판적 해석

RPR 모듈은 일종의 '두 번째 확인(second-look)' 과정으로 볼 수 있다. 초기 Box Head가 예측한 좌표를 맹신하지 않고, 주변 맥락을 포함한 특징을 다시 살펴봄으로써 좌표를 보정하는 방식이다. 이는 추론 시 약간의 연산 오버헤드를 발생시키지만, 생물학적 이미지 분석처럼 형태학적 정밀도가 최우선인 작업에서는 매우 실용적인 트레이드-오프라고 판단된다.

## 📌 TL;DR

- **요약**: Top-down 인스턴스 분할에서 발생하는 '마스크 잘림' 문제를 해결하기 위해, ROI를 점진적으로 확장해 주변 정보를 수집하고 이를 Attention 네트워크로 처리하여 박스 좌표를 정밀하게 교정하는 **RPR(Region Proposal Rectification)** 모듈을 제안하였다.
- **의의**: Mask R-CNN, CenterMask 등 다양한 모델에 결합하여 생물학적 이미지의 분할 성능을 높였으며, 향후 정밀한 형태학적 분석이 필요한 의료 및 생물학 이미지 처리 연구에 중요한 기초 baseline이 될 가능성이 높다.
