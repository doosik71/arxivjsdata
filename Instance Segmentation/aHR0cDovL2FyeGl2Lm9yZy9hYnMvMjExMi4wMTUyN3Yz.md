# Masked-attention Mask Transformer for Universal Image Segmentation

Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar

## 🧩 Problem to Solve

이미지 분할(Image segmentation)은 픽셀을 범주나 인스턴스 멤버십과 같은 다양한 의미론에 따라 그룹화하는 문제입니다. 기존에는 Panoptic, Instance, Semantic 분할과 같은 각기 다른 분할 작업을 위해 특화된 아키텍처(specialized architectures)를 설계해왔습니다. 이러한 방식은 연구 노력을 중복시키고 유연성이 부족하며, 최근 제안된 범용 아키텍처(universal architectures)들은 유연하지만, 특정 작업에서 최고의 특화된 아키텍처보다 성능이 뒤처지거나 학습 효율이 낮다는 한계를 가집니다. 본 논문은 이러한 분열을 해결하고 모든 이미지 분할 작업을 우수한 성능으로 처리할 수 있는 범용 아키텍처를 제안합니다.

## ✨ Key Contributions

* **범용 이미지 분할의 SOTA 달성**: Masked-attention Mask Transformer (Mask2Former)는 Panoptic, Instance, Semantic 분할의 세 가지 주요 이미지 분할 작업에서 기존의 최고 특화 모델들을 능가하는 최신 기술(State-of-the-art, SOTA) 성능을 달성한 최초의 범용 아키텍처입니다.
* **마스크드 어텐션(Masked Attention) 도입**: 트랜스포머 디코더(Transformer decoder)에서 예측된 마스크 영역 내로 교차 어텐션(cross-attention)을 제한하는 마스크드 어텐션을 제안하여, 지역화된 특징을 효율적으로 추출하고 모델의 수렴 속도 및 성능을 크게 향상시켰습니다.
* **효율적인 다중 스케일 고해상도 특징 활용**: 작은 객체/영역 분할을 개선하기 위해 픽셀 디코더(pixel decoder)에서 생성된 다중 스케일 고해상도 특징을 트랜스포머 디코더 계층에 효율적으로 공급하는 전략을 제안했습니다.
* **최적화 개선**: 자기 어텐션(self-attention)과 교차 어텐션의 순서 변경, 학습 가능한 쿼리 특징(learnable query features) 사용, 드롭아웃(dropout) 제거 등 계산 비용을 추가하지 않으면서 성능을 향상시키는 여러 최적화 기법을 적용했습니다.
* **학습 효율성 증대**: 마스크 손실(mask loss) 계산 시 전체 마스크 대신 무작위로 샘플링된 점들을 사용하여 학습 메모리를 3배 절감함으로써, 제한된 컴퓨팅 자원을 가진 사용자들도 범용 아키텍처에 쉽게 접근할 수 있도록 했습니다.

## 📎 Related Works

* **특화된 분할 아키텍처**: Semantic 분할을 위한 FCN(Fully Convolutional Networks) 기반 모델 [37] 및 컨텍스트 모듈 [7, 8, 63], Instance 분할을 위한 Mask R-CNN [24] 및 동적 마스크 생성 방법 [3, 49, 56] 등이 연구되었습니다. Panoptic 분할은 Semantic과 Instance를 통합하려는 시도 [28]로, 두 작업의 특화 아키텍처를 결합하거나 새로운 목적 함수를 설계하는 방식 [5, 52]이 있었습니다.
* **범용 아키텍처**: DETR [5]은 종단 간(end-to-end) 집합 예측(set prediction)을 통해 마스크 분류 아키텍처가 범용적일 수 있음을 보였습니다. MaskFormer [14]는 DETR 기반 마스크 분류가 Panoptic 및 Semantic 분할에서 SOTA를 달성했음을 보여주었고, K-Net [62]은 Instance 분할까지 확장했습니다. 그러나 이들 범용 아키텍처는 여전히 특화 모델 대비 성능 격차가 존재했으며, 특히 MaskFormer는 Instance 분할에 취약했습니다. Mask2Former는 이러한 성능 격차를 해소하는 데 중점을 둡니다.

## 🛠️ Methodology

Mask2Former는 백본(backbone), 픽셀 디코더(pixel decoder), 트랜스포머 디코더(Transformer decoder)로 구성된 MaskFormer [14]와 유사한 메타 아키텍처를 기반으로 합니다. 주요 방법론은 다음과 같습니다.

1. **백본**: ResNet [25] 또는 Swin Transformer [36]와 같은 표준 백본을 사용하여 저해상도 이미지 특징을 추출합니다.
2. **픽셀 디코더**: 백본의 출력에서 저해상도 특징을 점진적으로 업샘플링하여 고해상도 픽셀별 임베딩(per-pixel embeddings)을 생성합니다. 기본적으로 MSDeformAttn [66]을 사용합니다.
3. **트랜스포머 디코더**: 이미지 특징을 활용하여 객체 쿼리(object queries)를 처리합니다. 본 논문의 핵심 개선 사항이 여기에 집중됩니다.
    * **마스크드 어텐션(Masked Attention)**: 표준 교차 어텐션과 달리, 예측된 마스크의 전경(foreground) 영역 내로 어텐션을 제한합니다. 이는 트랜스포머 디코더의 $l$번째 레이어에서 쿼리 특징 $X_l$을 다음과 같이 계산합니다.
        $$
        X_l = \text{softmax}(M_{l-1} + Q_l K_l^{\text{T}})V_l + X_{l-1}
        $$
        여기서 $M_{l-1}$은 이전 레이어에서 예측된 마스크를 이진화(binarized)하여 생성된 어텐션 마스크이며, 전경 픽셀은 $0$, 배경 픽셀은 $-\infty$로 설정하여 어텐션이 배경으로 확산되는 것을 방지합니다.
        $$
        M_{l-1}(x,y) =
        \begin{cases}
        0 & \text{if } M_{l-1}(x,y) = 1 \\
        -\infty & \text{otherwise}
        \end{cases}
        $$
    * **다중 스케일 고해상도 특징 활용**: 픽셀 디코더에서 생성된 $1/32, 1/16, 1/8$ 스케일의 특징 피라미드(feature pyramid)를 활용합니다. 이를 라운드 로빈(round-robin) 방식으로 트랜스포머 디코더의 연속적인 레이어에 하나씩 공급하여 계산량 증가를 억제하면서 작은 객체 분할 성능을 개선합니다.
    * **최적화 개선**:
        * 자기 어텐션과 교차 어텐션의 순서를 교환하여(교차 어텐션을 먼저 수행) 쿼리 특징이 이미지 정보를 먼저 얻도록 합니다.
        * 객체 쿼리 특징($X_0$)을 학습 가능하게 만들고, 트랜스포머 디코더에 들어가기 전 마스크 예측에 직접 사용되어 손실을 통해 지도 학습되도록 합니다.
        * 드롭아웃을 완전히 제거하여 성능 저하를 방지합니다.
4. **학습 효율성 개선**:
    * 마스크 손실 계산 시 전체 마스크 대신 $K=12544$개의 무작위 샘플링된 점들을 사용합니다. 이 방법은 이분 매칭(bipartite matching) 손실과 최종 손실 계산 모두에 적용되며, 학습 메모리를 3배 절감합니다.

## 📊 Results

Mask2Former는 COCO, ADE20K, Cityscapes, Mapillary Vistas 등 4가지 주요 데이터셋에서 범용 아키텍처로는 처음으로 모든 이미지 분할 작업에서 SOTA 성능을 달성했습니다.

* **COCO Panoptic Segmentation**: 57.8 PQ 달성 (이전 SOTA MaskFormer 대비 5.1 PQ, K-Net 대비 3.2 PQ 향상).
* **COCO Instance Segmentation**: 50.1 AP 달성 (이전 SOTA HTC++ 대비 0.6 AP 향상, 경계(boundary) 품질은 2.1 AP$_{boundary}$ 향상).
* **ADE20K Semantic Segmentation**: 57.7 mIoU 달성 (이전 SOTA MaskFormer 대비 큰 폭으로 향상).
* **학습 효율성**: MaskFormer가 300 epoch가 필요했던 반면, Mask2Former는 50 epoch 만에 더 높은 성능에 수렴하여 학습 시간을 크게 단축했습니다.
* **메모리 절감**: 포인트 기반 마스크 손실 계산을 통해 학습 메모리를 이미지당 18GB에서 6GB로 3배 절감했습니다.
* **Cityscapes 및 Mapillary Vistas**: 이들 데이터셋에서도 경쟁력 있는 성능을 보이며 모델의 범용성을 입증했습니다.

## 🧠 Insights & Discussion

Mask2Former는 이미지 분할 분야에서 범용 아키텍처가 특정 작업에 특화된 모델들을 능가할 수 있음을 최초로 보여주며, 연구 노력을 크게 절감할 수 있는 잠재력을 제시합니다. 특히, 마스크드 어텐션은 트랜스포머 디코더가 지역화된 특징에 집중하게 하여 성능과 수렴 속도 개선에 가장 큰 기여를 했습니다. 학습 가능한 쿼리 특징은 마스크 제안(mask proposals)을 생성하는 데 효과적임이 확인되었습니다.

**한계점**:

* **개별 작업별 학습 필요**: Mask2Former는 범용 아키텍처임에도 불구하고, Panoptic 분할용으로만 학습된 모델이 Instance 또는 Semantic 분할용으로 개별적으로 학습된 동일 모델보다 성능이 약간 떨어지는 경향을 보였습니다. 이는 여전히 각 특정 작업에 대한 별도의 학습이 필요함을 시사합니다.
* **작은 객체 분할**: Mask2Former는 기존 모델 대비 작은 객체 분할 성능이 향상되었지만, 여전히 개선의 여지가 있으며, 멀티 스케일 특징을 완전히 활용하지 못하고 있습니다.

**향후 연구 방향**: 하나의 모델로 여러 작업과 데이터셋에 대해 한 번만 학습할 수 있는 진정한 의미의 범용 모델 개발, 작은 객체 분할 성능 개선 및 특징 피라미드 활용 최적화 등이 기대됩니다.

## 📌 TL;DR

Mask2Former는 마스크드 어텐션(masked attention)과 효율적인 학습 기법을 도입하여 범용 이미지 분할의 새로운 기준을 제시합니다. 이 모델은 Panoptic, Instance, Semantic 분할의 세 가지 주요 작업에서 각 분야의 최고 특화 모델들을 능가하는 SOTA 성능을 달성했으며, 동시에 학습 효율성(더 빠른 수렴, 3배 메모리 절감)을 크게 개선하여 범용 모델 설계의 가능성을 열었습니다.
