# Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers

Zhiqi Li, Wenhai Wang, Enze Xie, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo, Tong Lu

## 🧩 Problem to Solve

본 논문은 Panoptic Segmentation(파놉틱 분할) 작업에서 기존 Transformer 기반 모델, 특히 DETR의 한계를 해결하고자 합니다. 주요 문제점은 다음과 같습니다:

* **느린 수렴 속도**: DETR은 긴 학습 시간을 필요로 합니다.
* **낮은 마스크 품질**: DETR의 FPN(Feature Pyramid Network) 스타일 파놉틱 헤드는 마스크 경계의 충실도가 낮습니다.
* **"사물(things)"과 "배경(stuff)" 처리의 비효율성**: DETR은 셀 수 있는 객체(things)와 셀 수 없는 영역(stuff)을 동등하게 처리하여, 특히 stuff에 대해 최적의 성능을 내지 못합니다.
* **후처리 방법의 한계**: 픽셀 단위 argmax와 같은 기존 후처리 방식은 오탐(false-positive) 결과를 자주 생성합니다.

## ✨ Key Contributions

본 논문은 위 문제점들을 해결하기 위해 다음과 같은 혁신적인 기여를 합니다:

* **효율적인 심층 감독 마스크 디코더(Efficient Deeply-Supervised Mask Decoder)**: 다중 스케일 어텐션 맵을 활용하여 고품질 마스크를 생성하며, 중간 레이어에 대한 심층 감독을 통해 더 나은 마스크 품질과 빠른 수렴을 가능하게 합니다.
* **쿼리 디커플링 전략(Query Decoupling Strategy)**: 쿼리 세트를 "things" 쿼리 세트와 "stuff" 쿼리 세트로 분리하여 서로 간의 간섭을 피하고 특히 stuff 분할의 품질을 크게 향상시킵니다.
* **개선된 마스크 단위 병합 후처리 방법(Improved Mask-Wise Merging Post-Processing Method)**: 분류 확률과 예측된 마스크 품질을 동시에 고려하여 충돌하는 마스크 중첩을 해결하고, 기존 픽셀 단위 argmax 방법보다 효율적이며 성능을 향상시킵니다.
* **Deformable DETR 활용**: 효율적인 멀티스케일 피처 처리를 위해 Deformable DETR을 사용하여 계산 비용과 메모리를 절감하고 수렴 속도를 높입니다.

## 📎 Related Works

* **Panoptic Segmentation 초기 연구**: Kirillov et al. [6]이 Panoptic Segmentation 개념과 벤치마크를 제안했으며, Panoptic FPN [7], UPSNet [9], AUNet [20] 등이 개별적인 인스턴스/의미 분할 모델의 출력을 결합하여 성능을 개선했습니다.
* **통합 프레임워크 연구**: Li et al. [21]의 Panoptic FCN은 "top-down meets bottom-up" 디자인으로 파이프라인을 간소화했습니다.
* **Transformer 기반 Panoptic Segmentation**:
  * **DETR [1]**: 쿼리 세트를 통해 things와 stuff를 모두 처리하는 초기 Transformer 기반 모델로, 파놉틱 헤드를 추가하여 워크플로우를 단순화했습니다.
  * **Max-Deeplab [2]**: 이중 경로 Transformer를 통해 객체 범주와 마스크를 직접 예측합니다.
  * **MaskFormer [3]**: DETR 위에 픽셀 디코더를 추가하여 고해상도 피처를 정제합니다.
  * **K-Net [4]**: 동적 커널을 사용하여 인스턴스 및 의미 분할을 수행하는 동시 연구입니다.
* **End-to-end 객체 감지**: DETR [1]은 NMS 및 앵커와 같은 수작업 구성 요소 없이 객체 감지를 단순화했습니다. Deformable DETR [12]은 변형 어텐션 레이어를 통해 메모리 및 계산 비용을 더욱 절감했습니다.

## 🛠️ Methodology

Panoptic SegFormer는 크게 백본(Backbone), 인코더(Encoder), 디코더(Decoder)로 구성됩니다.

1. **전체 아키텍처**:
    * 입력 이미지 ($X \in R^{H \times W \times 3}$)는 백본 네트워크를 통해 멀티스케일 피처 맵 ($C_3, C_4, C_5$)을 얻습니다.
    * 이 피처 맵들은 FC 레이어를 통해 256 채널로 투영되고 평탄화되어 피처 토큰($C'_3, C'_4, C'_5$)이 됩니다.
    * Transformer 인코더는 이 피처 토큰을 입력으로 받아 정제된 피처를 출력합니다.
    * 초기화된 $N_{th}$ thing 쿼리와 $N_{st}$ stuff 쿼리가 "things"와 "stuff"를 각각 나타내는 데 사용됩니다.
    * 위치 디코더(Location Decoder)는 $N_{th}$ thing 쿼리를 정제하여 위치 정보를 학습합니다.
    * 마스크 디코더(Mask Decoder)는 thing 쿼리와 stuff 쿼리를 모두 입력으로 받아 최종 범주와 마스크를 예측합니다.
    * 추론 시에는 마스크 단위 병합 전략을 사용하여 파놉틱 분할 결과를 생성합니다.
2. **Transformer 인코더**:
    * Deformable Attention [12]을 사용하여 기존 Transformer 기반 방법의 한계였던 고해상도 및 멀티스케일 피처 맵을 효율적으로 처리합니다.
3. **디코더**:
    * **쿼리 디커플링 전략**:
        * $N_{th}$ thing 쿼리는 이분 매칭(bipartite matching)을 통해 things를 예측하는 데 사용됩니다.
        * $N_{st}$ stuff 쿼리는 클래스 고정 할당(class-fixed assign) 전략을 통해 stuff만 처리합니다.
        * 이 분리된 쿼리 세트는 things와 stuff 간의 상호 간섭을 방지하고, 동일한 파이프라인으로 처리될 수 있습니다.
    * **위치 디코더**:
        * $N_{th}$ thing 쿼리에 things의 위치 정보를 도입하기 위해 사용됩니다.
        * 훈련 단계에서는 보조 MLP 헤드를 사용하여 바운딩 박스와 범주를 예측하고 감지 손실 ($L_{det}$)로 감독합니다. 추론 시에는 이 MLP 헤드를 제거할 수 있습니다.
    * **마스크 디코더**:
        * 주어진 쿼리($Q$)와 Transformer 인코더의 정제된 피처($F$)로부터 범주와 마스크를 예측합니다.
        * 멀티스케일 어텐션 맵 ($A_3, A_4, A_5$)을 업샘플링하고 연결($A_{fused}$)하여 1x1 컨볼루션을 통해 이진 마스크를 예측합니다.
        * **심층 감독 (Deep Supervision)**: 마스크 디코더의 각 레이어에서 어텐션 맵이 실제 마스크에 의해 감독되어 어텐션 모듈이 의미 있는 영역에 빠르게 집중하도록 합니다.
        * **초경량 FC 헤드**: 어텐션 맵에서 마스크를 생성하는 데 사용되어 어텐션 모듈이 실제 마스크에 의해 효과적으로 학습될 수 있도록 합니다.
4. **손실 함수**:
    * 전체 손실 ($L = \lambda_{things}L_{things} + \lambda_{stuff}L_{stuff}$)은 things 손실과 stuff 손실의 합으로 구성됩니다.
    * **Things 손실 ($L_{things}$)**: 분류 손실 ($L_{cls}$), 감지 손실 ($L_{det}$), 분할 손실 ($L_{seg}$)의 합으로, 헝가리안 알고리즘 [31]을 통한 이분 매칭으로 예측과 GT를 매칭합니다.
    * **Stuff 손실 ($L_{stuff}$)**: 클래스 고정 매칭 전략을 사용하며, 분류 손실과 분할 손실로 구성됩니다.
5. **마스크 단위 병합 추론**:
    * 기존 픽셀 단위 argmax와 달리, 예측된 마스크들의 중첩 문제를 마스크 단위로 해결합니다.
    * 마스크의 신뢰도 점수($s_i = p_i^\alpha \times \text{average}(1_{\{m_i[h,w]>0.5\}}m_i[h,w])^\beta$)는 분류 확률과 예측 마스크 품질을 모두 고려합니다.
    * 신뢰도 점수가 높은 순서대로 마스크를 정렬하고, 낮은 신뢰도 점수를 가진 중첩 영역을 제거하여 최종 비중첩 파놉틱 결과를 생성합니다.

## 📊 Results

* **COCO 데이터셋 성능**:
  * COCO val 세트에서 ResNet-50 백본으로 49.6% PQ를 달성하여 기존 DETR (43.4% PQ) 및 MaskFormer (46.5% PQ)를 각각 6.2% PQ 및 3.1% PQ 초과합니다.
  * Swin-L 백본으로 COCO test-dev에서 56.2% PQ의 새로운 SOTA를 달성하여 MaskFormer (Swin-L)를 2.9% PQ 앞지릅니다.
  * PVTv2-B5 백본 사용 시 더 적은 파라미터와 FLOPs로 경쟁력 있는 성능(55.4% PQ on COCO val)을 보입니다.
* **ADE20K 데이터셋 성능**: ADE20K val 세트에서 MaskFormer를 1.7% PQ 앞선 36.4% PQ를 달성합니다.
* **효율성**:
  * 단 24 에폭 학습으로 49.6% PQ를 달성하여 MaskFormer (300 에폭)보다 빠르고 효율적인 수렴을 보입니다 (12 에폭에서도 MaskFormer 300 에폭보다 높은 48.0% PQ 달성).
  * DETR 대비 더 적은 계산 비용과 빠른 추론 속도를 보여줍니다.
* **인스턴스 분할 성능**: stuff 쿼리를 제외하여 인스턴스 분할 모델로 전환 가능하며, QueryInst [13] 및 HTC [14]와 유사한 경쟁력 있는 성능을 달성합니다.
* **모듈별 효과**:
  * **위치 디코더**: things 쿼리의 위치 정보 학습에 기여하여 things 성능을 향상시킵니다.
  * **마스크 단위 병합**: 모든 모델에서 픽셀 단위 argmax보다 항상 더 나은 Mask PQ 및 Boundary PQ [41]를 제공하며, DETR의 성능을 1.3% PQ 향상시킵니다. 또한 추론 속도가 더 빠릅니다.
  * **쿼리 디커플링 전략**: PQ$_{st}$를 크게 향상시키며, 기존 공동 매칭(joint matching) 방식 대비 2.9% PQ 향상을 달성합니다.
* **자연적 손상에 대한 강건성**: COCO-C 데이터셋에서 다른 SOTA 모델들보다 평균적으로 더 나은 강건성을 보입니다.

## 🧠 Insights & Discussion

* **Task-Specific 디자인의 중요성**: Panoptic SegFormer는 "things"와 "stuff"를 쿼리로 표현하여 동일한 패러다임으로 처리하면서도, 쿼리 디커플링 전략을 통해 그들의 상이한 특성을 고려한 맞춤형 파이프라인을 유연하게 설계할 수 있음을 보여줍니다. 이는 Panoptic Segmentation을 인스턴스/의미 분할의 하위 작업으로 모델링하는 기존 방식이나 모든 작업을 단일 파이프라인으로 통합하려는 시도보다 "차이점을 보존하면서 공통점을 추구하는" 더 합리적인 접근 방식이라고 주장합니다.
* **효율성과 성능의 균형**: Deformable Attention과 심층 감독 마스크 디코더 덕분에 Panoptic SegFormer는 SOTA 성능을 달성하면서도, 기존 Transformer 기반 모델 대비 훨씬 적은 학습 시간으로 빠르게 수렴합니다.
* **강건성 향상**: Transformer 기반 백본(Swin-L, PVTv2-B5)이 강건성 향상에 기여할 뿐만 아니라, Transformer 기반 마스크 디코더와 같은 작업 헤드의 설계 또한 모델의 강건성에 중요한 역할을 함을 COCO-C 데이터셋 실험을 통해 입증합니다.
* **한계**: Deformable Attention의 사용으로 인한 속도 저하가 여전히 존재하며, 더 큰 공간적 형태의 피처를 처리하거나 작은 객체에 대해 성능이 좋지 않을 수 있습니다.

## 📌 TL;DR

Panoptic SegFormer는 Transformer 기반 파놉틱 분할 모델의 한계를 극복하기 위해 **심층 감독 마스크 디코더, 쿼리 디커플링 전략, 마스크 단위 병합 후처리**라는 세 가지 혁신적인 요소를 제안합니다. 이 모델은 things와 stuff를 분리된 쿼리로 처리하여 상호 간섭을 줄이고, 마스크 디코더에 심층 감독을 적용하여 빠른 수렴과 고품질 마스크 생성을 가능하게 합니다. 또한 분류 확률과 마스크 품질을 통합한 새로운 후처리 방법을 통해 중첩 문제를 효과적으로 해결합니다. 결과적으로 Panoptic SegFormer는 COCO test-dev에서 56.2% PQ를 달성하며 SOTA 성능을 기록하고, 기존 모델 대비 학습 에폭을 절반으로 줄이는 등 뛰어난 효율성과 강건성을 입증했습니다.
