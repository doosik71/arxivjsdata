# Transformer-Based Visual Segmentation: A Survey

Xiangtai Li, Henghui Ding, Haobo Yuan, Wenwei Zhang, Jiangmiao Pang, Guangliang Cheng, Kai Chen, Ziwei Liu, Chen Change Loy

## 🧩 Problem to Solve

시각 분할(Visual segmentation)은 이미지, 비디오 프레임, 또는 포인트 클라우드를 여러 세그먼트나 그룹으로 분할하는 컴퓨터 비전의 핵심 과제입니다. 지난 10년간 딥러닝 기반 방법론들이 상당한 발전을 이루었으며, 특히 최근에는 자연어 처리(NLP)에서 시작된 트랜스포머(self-attention 기반)가 기존의 CNN(Convolutional Neural Network) 또는 RNN(Recurrent Neural Network) 기반 접근 방식을 다양한 비전 태스크에서 뛰어넘었습니다. 특히 비전 트랜스포머(Vision Transformers)는 다양한 분할 태스크에 대해 강력하고 통합적이며 심지어 더 단순한 해결책을 제시합니다. 하지만 트랜스포머 기반 시각 분할의 급속한 발전에 비해, 이 분야의 최신 발전을 체계적으로 요약하고 정리하는 포괄적인 조사가 부족한 상황입니다.

## ✨ Key Contributions

이 논문은 트랜스포머 기반 시각 분할의 최신 발전에 대한 포괄적인 조사를 제공하며, 주요 기여는 다음과 같습니다:

* **체계적인 개요:** 트랜스포머 기반 시각 분할 방법론의 최신 발전 사항을 배경 지식(문제 정의, 데이터셋, 이전 CNN 방법)부터 현재의 트랜스포머 기반 접근 방식까지 체계적으로 소개합니다.
* **메타-아키텍처 요약:** DETR(DEtection TRansformer) [22]의 확장 개념인, 기존 트랜스포머 기반 접근 방식들을 통합하는 핵심 메타-아키텍처를 제안합니다.
* **방법론 분류:** 제안된 메타-아키텍처의 구성 요소 수정 및 관련 응용 프로그램을 기반으로, 기존 방법론들을 강력한 표현 학습, 디코더의 교차-어텐션 디자인, 객체 쿼리 최적화, 쿼리를 연관성 추론에 활용, 조건부 쿼리 생성의 5가지 핵심 기술 범주로 분류합니다.
* **특정 하위 분야 조사:** 3D 포인트 클라우드 분할, 파운데이션 모델 튜닝, 도메인 인식 분할, 효율적인 분할, 의료 영상 분할 등 밀접하게 관련된 여러 하위 분야에서의 트랜스포머 활용을 다룹니다.
* **성능 벤치마킹 및 재평가:** 여러 주요 데이터셋에서 영향력 있는 방법론들의 성능을 비교하고, 특히 공정한 비교를 위해 일부 대표적인 작업들을 동일한 설정으로 재-벤치마킹합니다.
* **미해결 과제 및 미래 연구 방향 제시:** 이 분야의 현재 미해결 과제들을 식별하고, 일반화되고 통합된 분할, 다중 모달리티 학습, 평생 학습, 긴 비디오 분할, 생성 분할, 시각적 추론과의 통합 등 미래 연구 방향을 제시합니다.

## 📎 Related Works

이 논문은 트랜스포머 기반 시각 분할을 다루기 위해 광범위한 관련 연구를 참조하며, 주요 범주는 다음과 같습니다:

* **트랜스포머 이전의 딥러닝 기반 분할:**
  * **Semantic Segmentation (SS):** FCN [9]을 시작으로, 인코더-디코더 프레임워크 [58], [59], 더 큰 커널 [60], [61], 멀티스케일 풀링 [11], [62], 멀티스케일 특징 융합 [12], [63], 비지역 모델링(Non-local modeling) [18], [66] 등이 발전했습니다.
  * **Instance Segmentation (IS):** 객체 탐지기를 마스크 헤드(예: Mask R-CNN [76])로 확장하는 탑-다운(top-down) 방식과, 시맨틱 분할 맵에서 인스턴스 클러스터링을 수행하는 바텀-업(bottom-up) 방식 [78], [79]으로 나뉩니다.
  * **Panoptic Segmentation (PS):** SS와 IS 결과를 융합하는 복잡한 파이프라인에 주로 초점을 맞췄으며, 이 역시 탑-다운 [87], [88] 및 바텀-업 [84], [89] 접근 방식이 있었습니다.
  * **Video Segmentation:** VSS (공간-시간 융합 [90]), VIS (인스턴스별 공간-시간 관계 학습 [93]), VPS (탑-다운 및 바텀-업 [52]) 등이 다루어졌습니다.
  * **Point Cloud Segmentation:** PointNet [100], [101]과 같은 포인트 기반 및 voxel 기반 방법 [107], [108]이 사용되었습니다.
* **초기 Vision Transformer 연구:**
  * **CNN과 Self-Attention 결합:** 초기 방법 [17], [18]은 CNN을 보강하기 위해 self-attention 레이어를 사용했습니다.
  * **순수 Self-Attention:** ViT [21]는 이미지 패치 시퀀스를 직접 분류하는 순수 트랜스포머로, 이미지 인식에서 SOTA(State-Of-The-Art) 성능을 달성했습니다.
  * **Object Query 도입:** DETR [22]은 객체 쿼리(object query) 개념을 도입하여 복잡한 앵커 디자인을 대체하고 탐지 및 분할 파이프라인을 단순화했습니다.
* **기존 트랜스포머 관련 설문조사:** 시각 트랜스포머 [31], [32] 및 딥러닝 기반 분할 [37], [38]에 대한 이전 설문조사들이 있었으나, 본 논문은 비전 트랜스포머를 시각 분할 또는 쿼리 기반 객체 탐지에 초점을 맞춰 요약하는 최초의 시도입니다.

## 🛠️ Methodology

이 논문은 DETR(DEtection TRansformer) [22]에서 영감을 받은 메타-아키텍처를 기반으로 트랜스포머 기반 분할 방법론들을 체계적으로 분류하고 설명합니다.

1. **메타-아키텍처($Meta-Architecture$) 구성:**
    * **백본(Backbone):** 이미지 ($I \in \mathbb{R}^{H \times W \times 3}$)를 고수준 특징 ($F \in \mathbb{R}^{H' \times W' \times C}$)으로 추출합니다. 초기에는 CNN (예: ResNet50 [7])을 사용했지만, ViT [21]는 이미지를 패치 ($I_p \in \mathbb{R}^{N \times P^2 \times 3}$)로 분할하고 위치 임베딩($P$)을 더한 후 표준 트랜스포머 인코더를 사용하여 특징을 생성합니다.
    * **넥(Neck):** FPN (Feature Pyramid Network) [116] 아키텍처를 사용하여 백본에서 나온 다양한 스케일의 특징들을 통합하고 디코더가 활용할 수 있도록 동일한 채널 차원 ($C$)으로 매핑합니다. Deformable DETR [25]의 Deformable FPN과 같이 스케일 간 모델링을 강화하는 방법도 있습니다.
    * **객체 쿼리(Object Query):** DETR [22]에서 처음 도입된, 학습 가능한 임베딩($Q_{obj} \in \mathbb{R}^{N_{ins} \times d}$) 집합입니다. 각 쿼리는 하나의 객체 인스턴스를 대표하며, 기존 객체 탐지기의 NMS(Non-Maximum Suppression) 같은 복잡한 수작업 후처리 과정을 제거합니다.
    * **트랜스포머 디코더(Transformer Decoder):** 객체 쿼리($Q_{obj}$)와 이미지/비디오 특징($F$) 간의 cross-attention을 반복적으로 수행하여 객체 쿼리를 정제($Q_{out}$)합니다. 정제된 쿼리는 예측 FFN(Feed-Forward Network)을 통해 최종 예측(클래스, 바운딩 박스, 이진 마스크 로짓)을 생성합니다.
    * **마스크 예측 표현($Mask Prediction Representation$):** FCNs처럼 픽셀 단위 예측 (SS, VSS) 또는 DETR처럼 각 쿼리가 각 인스턴스를 대표하는 마스크별 예측 (IS, VIS, VPS)을 사용합니다.
    * **이분 매칭 및 손실 함수($Bipartite Matching \& Loss Function$):** 훈련 중 예측과 정답 간의 1대1 매칭을 위해 헝가리안 알고리즘 [121]을 사용합니다. 객체 분류 및 바운딩 박스 회귀 손실과 함께 마스크 분류 및 분할 손실 (이진 교차 엔트로피, Dice 손실 [122])을 적용합니다.

2. **방법론 분류($Method Categorization$):** 메타-아키텍처의 구성 요소 개선에 따라 5가지 범주로 분류합니다.
    * **강력한 표현 학습($Strong Representations$):**
        * **더 나은 ViT 디자인:** MViT [131], Pyramid ViT [135], XCiT [134] 등.
        * **하이브리드 CNNs/Transformers/MLPs:** Swin [23], Segformer [123], ConvNeXt [143] 등.
        * **Self-Supervised Learning (SSL):** MAE [24], BEiT [151], DINO [217] 등.
    * **디코더의 교차-어텐션 디자인($Cross-Attention Design in Decoder$):**
        * **향상된 교차-어텐션 (이미지):** Deformable DETR [25], MaskFormer [164], Mask2Former [226], K-Net [163] 등.
        * **시공간 교차-어텐션 (비디오):** VisTR [166], Video K-Net [172], TubeFormer [173] 등.
    * **객체 쿼리 최적화($Optimizing Object Query$):**
        * **쿼리에 위치 정보 추가:** Conditional DETR [174], DAB-DETR [177] 등.
        * **쿼리에 추가적인 감독($supervision$) 추가:** DN-DETR [178], Mask DINO [180] 등.
    * **쿼리를 연관성 추론에 활용($Using Query For Association$):**
        * **인스턴스 연관성 (비디오):** TrackFormer [185], MOTR [187], MiniVIS [188] 등.
        * **다중 작업 연결 (멀티태스크):** Panoptic-PartFormer [190], Polyphonicformer [128] 등.
    * **조건부 쿼리 생성($Conditional Query Generation$):**
        * **언어 특징 기반 조건부 쿼리:** VLT [195], LAVT [196], MTTR [199] 등 (RIS, RVOS).
        * **교차 이미지 특징 기반 조건부 쿼리:** CyCTR [203], BATMAN [208] 등 (Few-shot SS, VOS).

3. **특정 하위 분야($Specific Subfields$):** 포인트 클라우드 분할, 파운데이션 모델 튜닝(vision adapter, open vocabulary learning), 도메인 인식 분할(domain adaptation, multi-dataset segmentation), 레이블 및 모델 효율적 분할(weakly supervised, unsupervised, mobile segmentation), 클래스 불가지론적 분할 및 트래킹, 의료 영상 분할을 다룹니다.

## 📊 Results

이 논문은 이미지 및 비디오 분할 데이터셋에서 트랜스포머 기반 방법론들의 성능을 종합적으로 벤치마킹하고 재평가했습니다. 주요 결과는 다음과 같습니다:

* **이미지 분할 데이터셋:**
  * **Semantic Segmentation (SS):** Mask2Former [226]와 OneFormer [272]는 Cityscapes 및 ADE20K 데이터셋에서, SegNext [145]는 COCO-Stuff 및 Pascal-Context 데이터셋에서 최고의 성능을 보였습니다.
  * **Instance Segmentation (IS):** Mask DINO [180]는 ResNet 및 Swin-L 백본 모두에서 COCO 인스턴스 분할에서 가장 좋은 결과를 달성했습니다.
  * **Panoptic Segmentation (PS):** Mask DINO [180]와 k-means Mask Transformer (kMaX-DeepLab) [229]는 COCO 데이터셋에서, kMaX-DeepLab은 Cityscapes에서, OneFormer [272]는 ADE20K에서 최고 성능을 기록했습니다.
* **비디오 분할 데이터셋:**
  * **Video Semantic Segmentation (VSS):** TubeFormer [173]는 VPSW 데이터셋에서 최고의 mIoU 성능을 달성했습니다.
  * **Video Instance Segmentation (VIS):** CTVIS [358]는 YT-VIS-2019 및 YT-VIS-2021에서, GenVIS [359]는 OVIS에서 최고의 mAP 성능을 보였습니다.
  * **Video Panoptic Segmentation (VPS):** SLOT-VPS [360]는 Cityscapes-VPS에서, TubeLink [241]는 VIP-Seg에서, Video K-Net [172]는 KITTI-STEP 데이터셋에서 최고 성능을 달성했습니다.
* **이미지 분할 재-벤치마킹:**
  * 동일한 ResNet50 백본과 Deformable FPN [25] 넥 아키텍처를 사용하여 SS, IS, PS에 대한 재-벤치마킹을 수행했습니다. Mask2Former [226]는 모든 세 가지 작업에서 전반적으로 최고의 성능을 보였습니다.
  * 재-벤치마킹 결과, 강력한 데이터 증강(LSJ)과 Deformable FPN 같은 강력한 특징 피라미드 네트워크가 COCO와 같은 복잡한 데이터셋에서 성능 향상에 중요한 역할을 함을 확인했습니다.

## 🧠 Insights & Discussion

* **트랜스포머 기반 분할의 패러다임 변화:** 트랜스포머는 DETR에서 시작된 쿼리 기반 디자인을 통해 NMS와 같은 복잡한 수작업 후처리 과정을 제거하고, 단순하면서도 강력한 분할 파이프라인을 구축했습니다. 이는 이전 CNN 기반 방식에 비해 모델 설계의 복잡성을 줄이고 성능을 크게 향상시켰습니다.
* **메타-아키텍처의 유연성:** 제안된 백본, 넥, 객체 쿼리, 트랜스포머 디코더로 구성된 메타-아키텍처는 다양한 시각 분할 작업을 통합하고 새로운 방법론을 개발하는 데 매우 유연한 기반을 제공합니다. 이는 단일 프레임워크 내에서 시맨틱, 인스턴스, 파놉틱 분할을 통합하는 데 성공적인 적용 사례를 낳았습니다.
* **오픈 과제 및 미래 연구 방향:**
  * **범용적이고 통합된 분할:** 이미지와 비디오 분할 작업을 단일 모델로 통합하여 다양한 시나리오(예: 희귀 클래스 감지)에서 범용적이고 강력한 분할 능력을 달성하는 것이 중요한 목표입니다.
  * **다중 양식(Multi-Modality)과의 공동 학습:** 트랜스포머의 유연성을 활용하여 시각-언어 태스크(예: 텍스트-이미지 검색, 캡션 생성)와 분할을 공동 학습함으로써 상호 이점을 얻을 수 있습니다.
  * **평생 학습(Life-Long Learning):** 실제 세계의 개방적이고 비정상적인 시나리오에서 모델이 새로운 클래스를 지속적으로 학습하고 기존 지식 기반에 통합하는 능력을 개발해야 합니다.
  * **긴 비디오 분할:** 긴 비디오에서 인스턴스 연관성, 마스크 일관성, 심한 가림 처리, 다양한 장면 입력에 대한 도메인 강건성을 유지하는 새로운 방법론이 필요합니다.
  * **생성 분할(Generative Segmentation):** 확산 모델(diffusion model)과 같은 강력한 생성 모델을 활용하여 분할 문제를 생성 모델링 관점에서 접근함으로써 전체 프레임워크를 단순화할 수 있습니다.
  * **시각적 추론(Visual Reasoning)과의 통합:** 시각적 추론과 분할을 공동 학습하여 객체 간의 연결을 이해함으로써 분할 정확도를 높이고, 로봇 모션 계획 등 응용 분야를 강화할 수 있습니다.

## 📌 TL;DR

트랜스포머 기반 시각 분할에 대한 최초의 종합적인 조사 논문으로, DETR [22]에서 영감을 받은 백본, 넥, 객체 쿼리, 트랜스포머 디코더로 구성된 **메타-아키텍처**를 제안합니다. 이 메타-아키텍처를 기반으로 방법론들을 **강력한 표현 학습, 디코더의 교차-어텐션 디자인, 객체 쿼리 최적화, 쿼리를 연관성 추론에 활용, 조건부 쿼리 생성**의 5가지 주요 범주로 분류합니다. 또한 3D 포인트 클라우드, 파운데이션 모델 튜닝, 의료 영상 등 **특정 하위 분야**의 트랜스포머 활용을 다룹니다. **Mask2Former [226] 및 Mask DINO [180]**와 같은 모델들이 다양한 이미지 및 비디오 분할 작업에서 최첨단 성능을 달성했으며, 강력한 데이터 증강과 Deformable FPN [25] 같은 요소들이 성능 향상에 크게 기여함을 확인했습니다. 미래 연구는 **범용적이고 통합된 모델, 다중 양식 공동 학습, 평생 학습, 긴 비디오 및 생성 분할, 시각적 추론**과의 통합에 초점을 맞출 것으로 전망됩니다.
