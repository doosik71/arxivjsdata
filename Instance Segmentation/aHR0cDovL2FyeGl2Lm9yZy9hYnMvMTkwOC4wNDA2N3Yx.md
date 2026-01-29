# Explicit Shape Encoding for Real-Time Instance Segmentation

Wenqiang Xu, Haiyang Wang, Fubo Qi, Cewu Lu

## 🧩 Problem to Solve

인스턴스 분할(Instance Segmentation)은 객체의 위치와 형태를 동시에 예측해야 하므로, 일반적으로 객체 탐지(Object Detection)보다 계산 비용이 훨씬 높고 느립니다. 기존의 주류 인스턴스 분할 프레임워크(예: Mask R-CNN)는 각 객체 인스턴스마다 업샘플링 네트워크를 통해 마스크를 생성해야 하므로, 이미지 내 객체 수가 많을수록 처리 속도가 저하됩니다. 또한, 암묵적(implicit) 형상 인코딩 방식은 디코더 네트워크를 필요로 하여 여러 인스턴스에 대해 여러 번의 순방향 전파(forwarding)를 야기합니다.

본 논문은 객체 탐지와 거의 동일한 속도로 인스턴스 분할을 수행하기 위해, 객체 형태를 짧은 벡터로 회귀(regression)하고 이를 간단한 텐서 연산으로 효율적으로 디코딩하는 명시적(explicit) 형상 인코딩 방식을 제안합니다. 짧으면서도 충분한 정보를 담고, 잡음에 강하며, 효율적으로 디코딩 가능한 형상 벡터를 설계하는 것이 핵심 과제입니다.

## ✨ Key Contributions

* **ESE-Seg 프레임워크 제안:** 명시적 형상 인코딩 기반의 새로운 하향식(top-down) 인스턴스 분할 프레임워크 ESE-Seg를 제안합니다. 이는 여러 객체의 형태를 단일 패스(one pass)로 재구성하여 계산 비용을 크게 줄이고, 객체 탐지 수준의 속도를 달성합니다.
* **Inner-center Radius (IR) 형상 시그니처 도입:** 객체 형태를 표현하기 위한 새로운 형상 시그니처인 'Inner-center Radius (IR)'를 소개합니다. 이는 객체 내부의 이너 센터(inner-center)를 기준으로 윤곽점들을 극좌표계로 변환하여 $f(\theta)$ 함수로 나타냅니다.
* **Chebyshev 다항식 피팅 활용:** IR 시그니처를 짧고 잡음에 강한 계수 벡터로 변환하기 위해 Chebyshev 다항식 피팅을 적용합니다. 이 계수들이 네트워크가 예측할 형상 벡터가 됩니다.
* **다양한 객체 탐지기와 호환성:** ESE-Seg는 기존의 바운딩 박스 기반 객체 탐지 프레임워크(YOLOv3, Faster R-CNN, RetinaNet, YOLOv3-tiny 등)와 호환됩니다.
* **우수한 성능 및 속도:** YOLOv3 기반 ESE-Seg는 Pascal VOC 2012 데이터셋에서 Mask R-CNN보다 높은 mAP$_{r}$@0.5 성능을 달성하면서도 7배 더 빠릅니다. YOLOv3-tiny 기반으로는 GTX 1080Ti에서 약 130fps의 속도를 보여줍니다.

## 📎 Related Works

* **명시적 vs. 암묵적 형상 표현:**
  * **암묵적:** Jetley et al. [16]은 YOLO를 인코더로 사용하고 사전 훈련된 오토인코더 디코더로 마스크를 재구성하는 방식을 제안했습니다. ESE-Seg는 이와 달리 추가적인 디코더 네트워크 없이 간단한 텐서 연산으로 병렬 디코딩을 수행하며, 인코더-디코더 간 입력 불일치 문제를 피합니다.
  * **명시적:** 중심 반경(centroid radius), 복소 좌표(complex coordinates), 누적 각도(cumulative angle) [5, 34, 39] 등 다양한 기존 형상 시그니처가 있지만, 형태 재구성이 가능한 것은 소수에 불과합니다.
* **객체 탐지:**
  * **2단계 탐지:** Faster R-CNN [32], R-FCN [4], Cascade R-CNN [1].
  * **1단계 탐지:** SSD [23], YOLO [30], RetinaNet [20], CornerNet [18].
  * ESE-Seg는 바운딩 박스 기반의 모든 탐지 네트워크와 호환됩니다.
* **인스턴스 분할:**
  * **하향식(Top-down) 방식:** MNC [3], FCIS [19], Mask R-CNN [12] 등. 객체 수에 따라 성능 저하 문제가 있었으나, ESE-Seg는 이를 극복합니다.
  * **상향식(Bottom-up) 방식:** Associative Embedding [26].
  * 데이터 증강(data augmentation) [7, 36] 및 스케일 정규화(scale normalization) [33] 기술은 ESE-Seg 시스템에 쉽게 통합될 수 있습니다.

## 🛠️ Methodology

1. **전체 파이프라인:**
    * 객체 인스턴스 분할을 위해 명시적 형상 인코딩 기반 탐지 방식을 사용합니다.
    * 객체 인스턴스 마스크가 주어지면, 윤곽선을 **Inner-center Radius (IR) 형상 시그니처**로 매개변수화합니다.
    * IR 시그니처 벡터는 **Chebyshev 다항식**을 사용하여 소수의 계수로 근사됩니다. 이 계수들이 네트워크가 예측할 형상 디스크립터(shape descriptor)가 됩니다.
    * 최종적으로, 예측된 형상 디스크립터는 일반적인 객체 탐지 프레임워크 내에서 간단한 텐서 연산으로 디코딩됩니다.
2. **Inner-center Radius (IR) 형상 시그니처:**
    * **이너 센터(Inner Center) 찾기:** 윤곽선에서 가장 먼 점으로 정의되며, 거리 변환(distance transform) [25]을 통해 얻어집니다. 이는 질량 중심이나 바운딩 박스 중심과 달리 항상 객체 내부에 존재함을 보장합니다. 분리된 영역의 경우, 팽창(dilate) 및 윤곽선 재정렬 과정을 거쳐 단일 형태로 만듭니다.
    * **밀집 윤곽 샘플링(Dense Contour Sampling):** 이너 센터를 중심으로 각도 $\theta$ 간격으로 윤곽점들을 샘플링합니다(예: $\tau = \pi/180$ 이면 $N=360$ 점). 이를 통해 반경 $f(\theta)$를 각도 $\theta$에 대한 함수로 나타냅니다.
3. **Chebyshev 다항식 피팅:**
    * IR 시그니처는 여전히 길고 노이즈에 민감하므로, Chebyshev 다항식으로 근사하여 벡터 길이를 줄이고 강건성(robustness)을 높입니다.
    * $n$차 다항식으로 근사된 $\tilde{f}(\theta) = \sum_{i=0}^{n} c_i T_i(\theta)$에서 계수 $k = (c_0, ..., c_n)$가 형상 시그니처 벡터가 됩니다.
    * 다른 형상 시그니처(예: 2D 윤곽점 좌표 'XY') 및 피팅 방법(다항식 회귀, 푸리에 급수)과의 비교를 통해 Chebyshev 다항식이 재구성 오차, 노이즈 민감도, 계수 분포 측면에서 가장 우수함을 보였습니다.
4. **객체 탐지 프레임워크에서의 회귀:**
    * 네트워크는 객체 바운딩 박스와 함께 이너 센터 $\hat{p}$, 형상 벡터 $\hat{k}$를 예측합니다.
    * 전체 손실 함수는 $L = \lambda_{cls}L_{cls} + \lambda_{bbox}L_{bbox} + \lambda_{shape}L_{shape}$이며, 여기서 형상 손실 $L_{shape} = \frac{1}{\text{obj}} ||(\hat{p}-p) + (\hat{k}-k)||^2_2$입니다.
5. **형상 벡터를 형상으로 디코딩:**
    * 예측된 형상 벡터 $\hat{k} = (\hat{k}_0, ..., \hat{k}_{l-1})^T$를 사용하여 Chebyshev 다항식 $\hat{f}(\theta) = \sum_{i=0}^{l-1} \hat{k}_i T_i(\theta)$를 재구성합니다.
    * 윤곽점 $\hat{P}_i = \hat{p}_c + \hat{f}(\theta) \odot u(\theta)$ (여기서 $u(\theta) = (\cos\theta, \sin\theta)$는 Hadamard 곱)를 통해 윤곽선을 복원합니다.
    * 이 계산은 GPU에서 매우 빠르게 수행되는 텐서 연산 $\hat{P} = \hat{P}_c + \hat{C}T(\Theta) \odot u^T(\Theta)$으로 구현됩니다.

## 📊 Results

* **Explicit vs. Implicit (Pascal SBD 2012, Pascal VOC 2012, COCO 2017):**
  * YOLO-Cheby(20)는 기존 암묵적 방법(Embedding (20))보다 높은 mAP$_{r}$@0.5를 달성했습니다.
  * YOLOv3-Cheby(20) (COCO 사전 학습)는 Pascal VOC 2012에서 Mask R-CNN보다 69.3 mAP$_{r}$@0.5를 기록하며 우수했으며, Mask R-CNN (180ms)보다 7배 빠른 26ms의 추론 시간을 보였습니다. COCO 2017에서도 경쟁력 있는 성능(48.7 mAP)과 훨씬 빠른 속도(26ms)를 입증했습니다.
* **형상 시그니처 및 피팅 방법 비교 (Pascal VOC 2012):**
  * IR-Cheby(20)는 62.6 mAP$_{r}$@0.5를 달성하여, 직접 회귀 방식(IR(40), 52.6 mAP$_{r}$@0.5)이나 2D 윤곽점 기반 방식(XY-Cheby(20+20), 53.1 mAP$_{r}$@0.5)보다 훨씬 뛰어났습니다.
  * 다양한 함수 근사 방법 중 Chebyshev(20)가 62.6 mAP$_{r}$@0.5로 Fourier(20) (37.5 mAP$_{r}$@0.5)나 Poly(20) (26.3 mAP$_{r}$@0.5)에 비해 월등히 우수한 성능을 보였습니다.
* **다양한 객체 탐지기 기반 성능 (Pascal VOC 2012):**
  * ESE-Seg는 Faster R-CNN (63.4 mAP$_{r}$@0.5), RetinaNet (65.9 mAP$_{r}$@0.5), YOLOv3 (62.6 mAP$_{r}$@0.5) 등 다양한 바운딩 박스 기반 탐지기에서 안정적인 성능을 유지했습니다.
  * YOLOv3-tiny 기반 ESE-Seg는 53.2 mAP$_{r}$@0.5의 성능으로 8ms (약 130fps)의 매우 빠른 속도를 기록했습니다.
* **정성적 결과:** 예측된 형상 벡터가 윤곽선의 특성을 성공적으로 포착하며 노이즈를 발생시키지 않음을 확인했습니다.

## 🧠 Insights & Discussion

* **의의:** ESE-Seg는 인스턴스 분할의 계산 비용을 객체 탐지 수준으로 낮추어 실시간 인스턴스 분할을 가능하게 합니다. 이는 형태를 짧은 벡터로 압축하고 이를 GPU의 고속 텐서 연산으로 디코딩하는 새로운 메커니즘 덕분입니다. 이로써 기존 인스턴스 분할의 주요 병목 현상이었던 속도 문제를 해결했습니다.
* **한계점:** 현재 ESE-Seg는 낮은 IoU 임계값(예: 0.5)에서는 좋은 성능을 보이지만, 더 높은 IoU 임계값(예: 0.7)에서는 성능 하락이 큽니다. 이는 형상 벡터 표현의 부정확성 및 CNN 회귀에서 발생하는 노이즈 때문으로 분석됩니다.
* **향후 연구:** 향후에는 더 나은 명시적 형상 표현 방법과 고(高)IoU 임계값에서 높은 성능을 달성할 수 있는 CNN 회귀 훈련 방법에 대한 연구가 필요합니다.

## 📌 TL;DR

**문제:** 기존 인스턴스 분할은 객체 형태 디코딩으로 인해 느림.
**방법:** ESE-Seg는 "Inner-center Radius (IR)" 형상 시그니처와 Chebyshev 다항식 피팅을 사용하여 객체 형태를 짧은 계수 벡터로 명시적 인코딩합니다. 이 벡터는 YOLOv3와 같은 표준 객체 탐지기에서 바운딩 박스와 함께 회귀되며, 극도로 빠른 텐서 연산으로 실시간 디코딩됩니다.
**주요 발견:** ESE-Seg는 Mask R-CNN보다 7배 빠르고 Pascal VOC 2012에서 더 나은 mAP$_{r}$@0.5 성능을 달성하며, 객체 탐지 수준의 속도(예: YOLOv3-tiny로 130fps)로 실시간 인스턴스 분할을 가능하게 합니다.
