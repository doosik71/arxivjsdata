# FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation

Junjie He, Pengyu Li, Yifeng Geng, Xuansong Xie

## 🧩 Problem to Solve

최근 인스턴스 분할(Instance Segmentation) 분야에서는 NMS(Non-Maximum Suppression)를 사용하지 않는 엔드-투-엔드(end-to-end) 쿼리 기반 모델(query-based models)에 대한 관심이 높아지고 있습니다. 하지만 이러한 모델들은 고정확도 실시간 벤치마크에서 우수성이 충분히 입증되지 못했습니다. 특히, 쿼리 정제에 많은 디코더 레이어가 필요하고, 무거운 픽셀 디코더에 의존하며, 마스크 어텐션(masked attention)이 최적화되지 않은 쿼리 업데이트로 이어질 수 있다는 문제점이 있습니다. 이로 인해 효율적인 실시간 인스턴스 분할 벤치마크에서는 여전히 컨볼루션 기반(convolution-based) 모델이 주류를 이루고 있습니다.

## ✨ Key Contributions

* **FastInst 프레임워크 제안**: 실시간 인스턴스 분할을 위한 간단하고 효과적인 쿼리 기반 프레임워크인 FastInst를 제안했습니다. 이 모델은 추가적인 복잡성 없이 COCO test-dev에서 32.5 FPS의 속도로 40.5 AP를 달성하며, 대부분의 최첨단 실시간 모델들을 속도와 정확도 면에서 능가합니다.
* **Instance Activation-Guided Queries (IA-guided queries)**: 동적으로 높은 의미론적 정보를 가진 픽셀 임베딩을 초기 쿼리로 선택하여 Transformer 디코더의 반복 업데이트 부담을 크게 줄였습니다.
* **Dual-Path Update Strategy**: Transformer 디코더 내에서 쿼리 피처(query features)와 픽셀 피처(pixel features)를 교대로 업데이트하는 전략을 도입했습니다. 이는 가벼운 픽셀 디코더 사용을 가능하게 하고, 업데이트 수렴 속도를 높이며, 더 적은 디코더 레이어로도 우수한 성능을 달성하게 합니다.
* **Ground Truth Mask-Guided Learning (GT mask-guided learning)**: 훈련 시 마지막 레이어의 GT 마스크를 마스크 어텐션에 활용하여 각 쿼리가 대상 객체의 전체 영역을 보도록 유도함으로써, 마스크 어텐션이 더 적절한 전경 영역에 집중하고 모델 성능을 향상시켰습니다.
* 이러한 핵심 설계를 통해 FastInst는 가벼운 픽셀 디코더와 더 적은 Transformer 디코더 레이어를 사용하면서도 더 나은 성능을 달성했습니다.

## 📎 Related Works

* **영역 기반(Region-based) 방법**: Mask R-CNN [16], Faster R-CNN [35]과 같이 객체 바운딩 박스를 먼저 감지하고 마스크를 생성하는 방식입니다. 하지만 중복되는 영역 제안으로 인해 비효율적인 문제가 있습니다.
* **인스턴스 활성화 기반(Instance activation-based) 방법**: SOLO [42, 43], MEInst [47], CondInst [38], SparseInst [11] 등 특정 픽셀을 객체 대표로 사용하여 마스크를 예측하는 방식입니다. 실시간 성능은 뛰어나지만 NMS와 같은 후처리 단계에 의존하는 경향이 있습니다.
* **쿼리 기반(Query-based) 방법**: DETR [4]의 성공 이후 등장했으며, Mask2Former [9, 10], SOLQ [14], ISTR [19], Panoptic SegFormer [26], Mask DINO [25] 등이 있습니다. NMS-free 및 엔드-투-엔드 방식으로 동작하지만, 일반적으로 높은 계산 비용으로 인해 실시간 적용에 한계가 있었습니다. FastInst는 Mask2Former의 메타 아키텍처를 기반으로 이러한 효율성 문제를 해결하고자 합니다.

## 🛠️ Methodology

FastInst는 백본(backbone), 픽셀 디코더(pixel decoder), Transformer 디코더(Transformer decoder)의 세 가지 모듈로 구성됩니다.

1. **Lightweight Pixel Decoder**:
    * 입력 이미지에서 백본을 통해 `C_3, C_4, C_5` 피처 맵을 얻습니다.
    * 이 피처 맵들을 256 채널로 프로젝트한 후 픽셀 디코더에 입력하여 향상된 멀티스케일 피처 맵 `E_3, E_4, E_5`를 생성합니다.
    * 기존 방법들과 달리, 무거운 컨텍스트 집계가 필요 없는 PPM-FPN [11]과 같은 경량 픽셀 디코더를 사용하여 효율성을 높였습니다.

2. **Instance Activation-Guided Queries (IA-guided queries)**:
    * 픽셀 디코더의 출력 피처 맵 `E_4`에 보조 분류 헤드와 softmax 활성화를 적용하여 각 픽셀에 대한 클래스 확률 `p_i`를 얻습니다.
    * 각 픽셀 `i`에 대해 `p_{i, k_i}`가 해당 클래스 평면($k_i$는 해당 픽셀의 예측 클래스)에서 *지역 최대값(local maximum)*을 갖는 픽셀들을 먼저 선택합니다.
    * 이들 중 가장 높은 전경 확률을 갖는 상위 $N_a$개의 픽셀 임베딩을 IA-guided 쿼리로 선택합니다.
    * 훈련 시에는 헝가리 손실 [23]과 위치 비용 $L_{loc}$ (픽셀이 객체 영역 내부에 있으면 0, 아니면 1)을 사용하여 이 보조 분류 헤드를 감독합니다. 이를 통해 초기에 풍부한 객체 정보를 가진 쿼리를 생성하여 Transformer 디코더의 부담을 줄입니다.

3. **Dual-Path Transformer Decoder**:
    * $N_a$개의 IA-guided 쿼리와 $N_b$개의 보조 학습 가능한 쿼리를 결합하여 전체 쿼리 $Q$를 구성합니다.
    * $Q$와 평탄화된 고해상도 픽셀 피처 $X$를 Transformer 디코더에 입력합니다.
    * 각 Transformer 디코더 레이어는 **픽셀 피처 업데이트**와 **쿼리 업데이트**를 교대로 수행합니다 (EM 클러스터링과 유사).
        * **픽셀 피처 업데이트**: 교차-어텐션 레이어와 피드포워드 레이어를 통해 픽셀 피처 `X`를 업데이트합니다.
        * **쿼리 업데이트**: Masked attention, self-attention, 피드포워드 네트워크를 통해 쿼리 `Q`를 업데이트합니다.
    * 고정된 sinusoidal 위치 임베딩 대신 학습 가능한 위치 임베딩을 사용하여 모델 추론 속도를 향상시킵니다.
    * 각 디코더 레이어의 최종 쿼리와 픽셀 피처를 사용하여 객체 클래스와 분할 마스크를 예측합니다.

4. **Ground Truth Mask-Guided Learning**:
    * 훈련 중 $l$번째 레이어의 마스크 어텐션에 사용되는 예측 마스크를 마지막 Transformer 디코더 레이어의 이분 매칭된 GT 마스크 $M_{gt}$로 대체합니다.
    * 이를 통해 각 쿼리가 훈련 시 대상 객체의 전체 영역을 보도록 하여 마스크 어텐션이 더 정확한 전경 영역에 집중하도록 돕습니다.
    * 새로운 출력은 마지막 레이어의 이분 매칭 결과와 일관된 고정된 매칭 $\sigma$에 따라 감독됩니다.

5. **Loss Function**: 전체 손실 함수는 IA-guided 쿼리의 인스턴스 활성화 손실 $L_{IA-q}$, 예측 손실 $L_{pred}$, GT 마스크 유도 손실 $L'_{pred}$의 합으로 구성됩니다: $L = L_{IA-q} + L_{pred} + L'_{pred}$.

## 📊 Results

* **COCO test-dev 성능**: FastInst는 COCO test-dev 벤치마크에서 대부분의 기존 최첨단 실시간 인스턴스 분할 알고리즘을 속도와 정확도 면에서 능가합니다.
  * ResNet-50 백본을 사용하는 **FastInst-D1** 모델은 53.8 FPS에서 35.6 AP를 달성하여 SparseInst [11]보다 0.9 AP 더 높은 성능을 보였습니다.
  * ResNet-50-d-DCN 백본을 사용하는 **FastInst-D3** 모델은 32.5 FPS에서 40.5 AP를 달성하여, 40 AP를 넘는 동시에 실시간 속도(≥ 30 FPS)를 유지하는 유일한 알고리즘입니다 (표 1).
* **Mask2Former [9]와의 비교**: 경량 픽셀 디코더를 사용한 Mask2Former [9]와 비교했을 때, FastInst는 정확도와 속도 모두에서 우수한 성능을 보여 실시간 벤치마크에서의 효율성을 입증했습니다.
* **Ablation Studies**:
  * **IA-guided queries**: 제안된 IA-guided 쿼리가 다른 쿼리 선택 방식보다 더 나은 결과를 얻었으며, 특히 Transformer 디코더 레이어 수가 적을 때 효율성이 두드러졌습니다 (표 2).
  * **Dual-path update strategy**: 쿼리 및 픽셀 피처의 공동 최적화 덕분에 단일 경로 업데이트 전략보다 일관되게 우수한 성능을 보였습니다 (표 3).
  * **GT mask-guided learning**: 이 기술은 모델 성능을 최대 0.5 AP 향상시키는 것으로 나타나, 마스크 어텐션 학습에 효과적임을 입증했습니다 (표 4).
  * **픽셀 디코더**: PPM-FPN [11]은 정확도와 속도 사이에서 좋은 절충점을 제공하는 것으로 확인되었습니다 (표 5).
  * **Transformer 디코더 레이어 수**: 레이어 수가 증가함에 따라 성능이 향상되지만, 몇 개의 레이어만으로도 좋은 성능을 얻을 수 있으며 6개 레이어에서 성능이 포화되는 경향을 보였습니다 (표 6a).
  * **IA-guided 쿼리 수**: 쿼리 수를 늘리면 객체 리콜(recall) 향상을 통해 분할 성능이 향상되었습니다 (표 6b, 6c).
  * **학습 가능한 위치 임베딩**: 비매개변수적인 sinusoidal 위치 임베딩 대신 학습 가능한 위치 임베딩을 사용하면 성능 저하 없이 모델 추론 속도가 향상되었습니다 (표 7).

## 🧠 Insights & Discussion

FastInst는 쿼리 기반 모델이 효율적인 실시간 인스턴스 분할에서도 뛰어난 성능을 발휘할 수 있음을 강력하게 보여주었습니다. 제안된 세 가지 핵심 기술(IA-guided queries, dual-path update strategy, GT mask-guided learning)은 모두 모델의 정확도와 효율성을 높이는 데 중요한 역할을 했습니다. 특히, IA-guided queries는 Transformer 디코더의 초기 쿼리에 풍부한 객체 정보를 제공하여 적은 수의 디코더 레이어로도 뛰어난 성능을 가능하게 하며, dual-path update strategy는 픽셀 디코더의 부담을 줄이고 빠른 수렴을 돕습니다. GT mask-guided learning은 마스크 어텐션의 학습 과정을 개선하여 정확도를 높입니다.

**한계점**:

* 일반적인 쿼리 기반 모델과 유사하게, FastInst는 작은 객체에 대한 분할 성능이 아직 만족스럽지 않습니다. 더 강력한 픽셀 디코더를 사용하거나 더 큰 피처 맵을 활용하면 성능이 개선되지만, 이는 더 많은 계산 비용을 초래합니다.
* GT mask-guided learning은 성능 향상에 기여하지만 훈련 비용이 증가합니다. 향후 더 효율적이고 우아한 방법이 필요합니다.

## 📌 TL;DR

**문제**: 쿼리 기반 인스턴스 분할 모델은 NMS-free 및 엔드-투-엔드 방식이지만, 실시간 고정확도 성능과 효율성(과도한 디코더 레이어, 무거운 픽셀 디코더) 면에서 한계가 있었습니다.

**해결책**: FastInst는 Mask2Former의 메타 아키텍처를 기반으로 세 가지 핵심 기술을 도입하여 이러한 문제를 해결합니다. 1) **Instance Activation-Guided Queries($N_a$)**는 피처 맵($E_4$)에서 지역 최대값의 전경 확률을 가진 픽셀 임베딩을 동적으로 선택하여 초기 쿼리에 풍부한 객체 정보를 제공하고 Transformer 디코더의 부담을 줄입니다. 2) **Dual-Path Update Strategy**는 쿼리 피처와 픽셀 피처를 교대로 업데이트하여 가벼운 픽셀 디코더를 사용하면서도 빠른 수렴과 효율적인 학습을 가능하게 합니다. 3) **Ground Truth Mask-Guided Learning**은 훈련 시 마스크 어텐션에 예측 마스크 대신 GT 마스크를 활용하여 쿼리가 대상 객체의 전체 영역을 학습하도록 유도하여 성능을 향상시킵니다.

**결과**: FastInst는 COCO test-dev에서 32.5 FPS의 속도로 40.5 AP를 달성하는 등 대부분의 최첨단 실시간 인스턴스 분할 모델들을 속도와 정확도 면에서 능가합니다. 이는 쿼리 기반 모델이 실시간 애플리케이션에 강력한 잠재력을 가짐을 입증합니다.
