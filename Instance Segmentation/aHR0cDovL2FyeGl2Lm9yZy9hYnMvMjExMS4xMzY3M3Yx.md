# Mask Transfiner for High-Quality Instance Segmentation

Lei Ke, Martin Danelljan, Xia Li, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu

## 🧩 Problem to Solve

현재의 인스턴스 세그멘테이션(instance segmentation) 방법들은 객체 탐지(object detection) 능력은 향상되었으나, 생성되는 마스크의 품질이 **여전히 거칠고 객체 경계를 과도하게 부드럽게(over-smoothing) 처리**하는 경향이 있습니다. 특히 Mask R-CNN이나 DETR 기반의 최신 방법들도 마스크 품질에서 탐지 성능과의 상당한 격차를 보입니다. 고해상도의 세밀한 마스크를 예측하려면 높은 계산 및 메모리 비용이 요구되어, 효율적이면서도 정확한 마스크 예측이 어려운 문제에 직면해 있습니다.

## ✨ Key Contributions

* **고품질 및 효율적인 Mask Transfiner 제안**: 거친 마스크 문제를 해결하고 기존 방법들을 크게 능가하는 트랜스포머(Transformer) 기반의 인스턴스 세그멘테이션 방법을 제시합니다.
* **Quadtree 기반 희소(Sparse) 이미지 영역 표현 및 처리**: 기존의 밀집(dense) 텐서 대신, 객체 경계와 같은 오류 발생 가능성이 높은 '비일관성 영역(incoherent regions)'을 쿼드트리(quadtree)로 분해하여 표현하고, 이 희소한 노드들만을 선택적으로 처리하여 계산 효율성을 극대화합니다.
* **Transformer 기반 정제 네트워크 설계**: 쿼드트리 노드들의 마스크 레이블을 병렬로 자기 수정(self-correct)하는 노드 인코더(Node Encoder), 시퀀스 인코더(Sequence Encoder), 픽셀 디코더(Pixel Decoder)로 구성된 트랜스포머 아키텍처를 개발했습니다. 이는 희소한 픽셀 간의 전역적(global) 및 스케일 간(inter-scale) 관계 추론에 효과적입니다.
* **계층적 마스크 전파(Hierarchical Mask Propagation) 스키마**: 쿼드트리 구조를 활용하여 정제된 마스크 레이블을 거친 스케일에서 미세한 스케일로 효율적으로 전파하여 최종 마스크 품질을 향상시킵니다.
* **SOTA 성능 달성 및 효율성 입증**: COCO 및 BDD100K에서 +3.0 mask AP, Cityscapes에서 +6.6 boundary AP를 포함하여 주요 벤치마크에서 기존 인스턴스 세그멘테이션 방법을 큰 폭으로 능가하며, 낮은 계산 및 메모리 비용을 유지합니다.

## 📎 Related Works

* **Two-stage Instance Segmentation**: Mask R-CNN [19]과 같은 방법들은 바운딩 박스 탐지 후 각 RoI(Region of Interest) 내에서 마스크를 예측합니다. Mask Transfiner는 이러한 프레임워크에 쉽게 통합되어 마스크 품질을 개선합니다.
* **Query-based Instance Segmentation**: DETR [4]에서 영감을 받은 방법들 [14, 15]은 세그멘테이션을 집합 예측 문제로 다루지만, Mask Transfiner는 이들 방법의 마스크 품질 한계를 지적하고 개선합니다.
* **Refinement for Instance Segmentation**: PointRend [25]는 낮은 신뢰도 포인트를 샘플링하여 MLP로 정제하고, RefineMask [47]는 fine-grained 특징을 사용합니다. Mask Transfiner는 이러한 방법들과 달리 경량 FCN으로 비일관성 영역을 탐지하고, Transformer의 전역적 처리 능력을 활용하여 쿼드트리 구조에 맞는 효율적인 정제를 수행합니다.

## 🛠️ Methodology

Mask Transfiner는 기존 객체 탐지 네트워크와 함께 작동하여 고품질 인스턴스 마스크를 예측합니다.

1. **비일관성 영역(Incoherent Regions) 정의 및 탐지**:
    * **정의**: 마스크를 다운샘플링($S_\downarrow$) 후 다시 업샘플링($S_\uparrow$)하여 원본 마스크 $M_{l-1}$를 재구성할 때 정보 손실이 발생하는 영역으로 정의됩니다. 이는 $D_l = O_\downarrow(M_{l-1} \oplus S_\uparrow(S_\downarrow(M_{l-1})))$로 계산되며, 주로 객체 경계나 고주파 영역에 분포합니다.
    * **탐지**: 다중 스케일 특징과 초기 거친 마스크를 입력으로 받아, $3 \times 3$ 컨볼루션 레이어 4개와 이진 분류기로 구성된 경량 FCN(Fully Convolutional Network)을 통해 비일관성 마스크를 예측합니다. 이때 하위 스케일에서 탐지된 마스크가 상위 스케일의 탐지를 안내하는 계단식 디자인을 채택하여 효율성을 높입니다.

2. **쿼드트리(Quadtree) 기반 마스크 정제**:
    * **쿼드트리 구축**: 탐지된 비일관성 영역을 분해하는 '포인트 쿼드트리'를 사용합니다. FPN(Feature Pyramid Network)의 각 레벨에 걸쳐 비일관성 포인트들을 기준으로 계층적으로 분할하여 구축합니다.
    * **쿼드트리 정제**: 구축된 쿼드트리의 모든 비일관성 노드의 마스크 예측을 후술할 Mask Transfiner 아키텍처로 공동(jointly) 정제합니다.

3. **Mask Transfiner 아키텍처**:
    * **RoI Feature Pyramid**: CNN 백본과 FPN을 통해 $P_2$부터 $P_5$까지의 계층적 특징 맵을 추출하고, RoI Align을 통해 각 객체의 $28 \times 28, 56 \times 56, 112 \times 112$ 스케일의 RoI 특징 피라미드를 생성합니다.
    * **Input Node Sequence**: 쿼드트리의 세 레벨에서 선택된 모든 비일관성 노드들을 하나의 시퀀스로 구성하여 트랜스포머의 입력으로 사용합니다. 이 시퀀스는 희소하며 순서에 무관합니다.
    * **Node Encoder**: 각 쿼드트리 노드의 특징을 풍부하게 합니다. 네 가지 정보(1) FPN의 fine-grained 특징, (2) 초기 거친 마스크 예측, (3) RoI 내의 상대적 위치 임베딩, (4) $3 \times 3$ 주변 컨텍스트 특징)를 통합합니다.
    * **Sequence Encoder**: Node Encoder에서 인코딩된 노드 시퀀스를 입력으로 받아 다중 헤드 자기 주의(multi-head self-attention) 및 FFN으로 구성된 표준 트랜스포머 레이어를 통해 전역적 공간 및 스케일 간 추론을 수행합니다.
    * **Pixel Decoder**: 트랜스포머 디코더와 달리, 각 노드의 출력 쿼리(query)를 디코딩하여 최종 마스크 레이블을 예측하는 작은 2-레이어 MLP입니다.
    * **훈련 및 추론**: 전체 Mask Transfiner 프레임워크는 객체 탐지 손실($L_{Detect}$), 거친 마스크 손실($L_{Coarse}$), 정제 손실($L_{Refine}$), 비일관성 영역 탐지 손실($L_{Inc}$)을 포함하는 다중 작업 손실로 엔드-투-엔드(end-to-end) 훈련됩니다. 추론 시에는 쿼드트리 전파 스키마를 따릅니다.

4. **쿼드트리 마스크 전파(Quadtree Propagation)**:
    * 정제된 마스크 예측을 기반으로 계층적(coarse-to-fine) 마스크 전파 방식을 사용합니다. 쿼드트리의 루트 레벨에서 시작하여 보정된 포인트 레이블을 니어리스트 이웃 보간법을 통해 다음 미세 레벨의 4개 quadrant로 전파합니다. 이 과정은 가장 미세한 쿼드트리 레벨에 도달할 때까지 반복됩니다.

## 📊 Results

Mask Transfiner는 주요 인스턴스 세그멘테이션 벤치마크에서 SOTA(State-Of-The-Art) 성능을 달성하며 효율성을 입증했습니다.

* **COCO**:
  * 두 단계(two-stage) 및 쿼리 기반(query-based) 프레임워크 모두에서 Mask AP를 크게 향상시켰습니다. 예를 들어, R50-FPN 백본 사용 시 Mask AP를 **+3.0** 개선했습니다.
  * 특히 쿼리 기반 DETR [4] 탐지기와 결합했을 때, Mask Transfiner는 41.6 Mask AP를 달성하여 SOLQ [14] 및 QueryInst [15]를 큰 폭으로 능가했습니다.
  * Boundary IoU (AP$_B$) 지표에서 2점 이상 개선되어, Mask Transfiner의 fine-grained 마스크 품질 향상 기여를 입증했습니다.
* **Cityscapes**:
  * Mask AP 37.9, Boundary AP$_B$ 18.0을 달성하며 PointRend [25] 및 BMask R-CNN [12]과 같은 기존 SOTA 방법들을 각각 1.3 AP$_B$, 2.3 AP$_B$만큼 능가했습니다.
* **BDD100K**:
  * 23.6 Mask AP를 기록하여 baseline [20] 대비 3점 향상된 성능을 보였습니다.
* **효율성**:
  * 희소한 픽셀 처리를 통해 동일한 출력 크기에서 Non-local attention [39]보다 3배 적은 메모리를 사용합니다.
  * 표준 Transformer가 $56 \times 56$ RoI에서 동작할 때보다 절반의 FLOPs로 $224 \times 224$ 고해상도 예측을 생성합니다.
  * $112 \times 112$ 출력 크기에서 7.1 FPS로 동작하여, Cascade Mask R-CNN (4.8 FPS) 및 HTC (2.1 FPS)보다 빠르면서도 더 정확합니다.
* **정성적 결과**: Cityscapes 이미지에서 자동차의 후면 거울, 하이힐과 같은 작은 부위나 복잡한 객체 경계 영역에서 이전 방법들보다 훨씬 정밀하고 고품질의 마스크를 생성하는 것을 확인했습니다.

## 🧠 Insights & Discussion

* **고품질 마스크 예측의 중요성과 효율성**: 인스턴스 세그멘테이션에서 마스크 품질 향상이 객체 탐지 능력의 발전을 따라가지 못하는 문제를 Mask Transfiner가 효과적으로 해결했습니다. 전체 오류 픽셀의 43%가 바운딩 박스 면적의 14%만을 차지하는 '비일관성 영역'에 집중되어 있다는 분석은, 이러한 오류가 많은 희소 영역에 집중하는 것이 효율적임을 시사합니다.
* **Transformer와 쿼드트리 결합의 시너지**: 트랜스포머의 강력한 전역적 추론 능력과 쿼드트리의 계층적이고 희소한 데이터 표현이 결합하여, 기존 CNN 기반 방식의 균일한 그리드 제약을 뛰어넘어 고해상도에서 높은 정확도를 효율적으로 달성할 수 있습니다. 특히 노드 인코딩의 위치 임베딩(positional encoding)은 트랜스포머의 순열 불변성(permutation-invariance)을 보완하여 마스크 품질 향상에 중요한 역할을 합니다.
* **계층적 정제 및 전파의 효과**: 쿼드트리 깊이가 깊어질수록, 즉 더 높은 해상도에서 정제할수록 마스크 AP가 꾸준히 증가하며 다단계 정제의 효과를 입증했습니다. 또한 계층적 마스크 전파 전략은 중간 쿼드트리 레벨의 리프 노드까지 정제된 레이블을 확장하여 미미한 추가 계산 비용으로 정제 영역을 넓히고 성능을 향상시킵니다.
* **한계 및 미래 연구**: 현재 Mask Transfiner는 다른 경쟁 방법들과 마찬가지로 완전히 지도 학습(fully supervised training)을 필요로 합니다. 향후 연구에서는 이러한 지도 학습에 대한 가정을 완화하는 방향으로 발전할 수 있을 것입니다.

## 📌 TL;DR

Mask Transfiner는 거친 마스크 품질 문제를 해결하기 위해 **쿼드트리 기반의 희소(sparse) 영역 처리와 Transformer 아키텍처를 결합한 고품질 인스턴스 세그멘테이션 방법**입니다. 이 방법은 객체 경계의 **'비일관성 영역'만을 효율적으로 탐지**하여 쿼드트리로 표현하고, **Node Encoder, Sequence Encoder, Pixel Decoder로 구성된 Transformer 기반 네트워크로 이 노드들의 마스크 레이블을 병렬 정제**합니다. 정제된 레이블은 계층적 전파를 통해 최종 마스크를 형성합니다. 그 결과, Mask Transfiner는 COCO, Cityscapes, BDD100K 벤치마크에서 기존 방법들을 뛰어넘는 **탁월한 마스크 품질(예: COCO +3.0 mask AP, Cityscapes +6.6 boundary AP)을 달성**하며, 희소한 처리로 **낮은 계산 및 메모리 비용**을 유지하는 효율성을 입증했습니다.
