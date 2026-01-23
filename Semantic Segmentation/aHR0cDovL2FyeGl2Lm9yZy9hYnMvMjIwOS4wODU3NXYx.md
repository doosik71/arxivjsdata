# SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu

## 🧩 Problem to Solve

최근 시맨틱 분할 분야는 공간 정보를 효과적으로 인코딩하는 셀프 어텐션 메커니즘을 사용하는 트랜스포머 기반 모델들이 지배적이었습니다. 하지만 트랜스포머는 고해상도 이미지 처리 시 높은 연산 복잡도($O(n^2)$) 문제를 가집니다. 이 논문은 컨볼루션 어텐션이 트랜스포머의 셀프 어텐션보다 컨텍스트 정보를 인코딩하는 데 더 효율적이고 효과적인 방법임을 보여줌으로써, 효율적이고 강력한 시맨틱 분할 모델을 설계하는 것을 목표로 합니다. 특히, 성공적인 분할 모델이 가져야 할 핵심 특성들(강력한 인코더, 멀티 스케일 상호작용, 공간 어텐션, 낮은 연산 복잡도)을 통합하는 새로운 컨볼루션 기반 아키텍처를 제안합니다.

## ✨ Key Contributions

* 성공적인 시맨틱 분할 모델이 갖춰야 할 특성들을 재정의하고, 멀티 스케일 컨볼루션 특징을 통해 공간 어텐션을 유도하는 새로운 네트워크 아키텍처인 SegNeXt를 제안했습니다.
* 단순하고 저렴한 컨볼루션 연산을 사용하는 인코더가 비전 트랜스포머보다 적은 연산 비용으로, 특히 객체 디테일 처리에서 더 나은 성능을 달성할 수 있음을 입증했습니다.
* ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, iSAID를 포함한 다양한 벤치마크에서 기존 최첨단 시맨틱 분할 방법들보다 월등히 향상된 성능을 보여주었습니다.

## 📎 Related Works

* **시맨틱 분할 (Semantic Segmentation):** FCN 및 DeepLab 시리즈와 같은 초기 CNN 기반 모델부터 SETR, SegFormer와 같은 최근 트랜스포머 기반 모델에 이르기까지 아키텍처의 발전을 다룹니다. 인코더-디코더 구조는 ResNet, HRNet, Swin Transformer 등 다양한 백본을 사용해 발전해왔습니다.
* **멀티 스케일 네트워크 (Multi-Scale Networks):** GoogleNet, HRNet과 같이 인코더 및 디코더에서 멀티 스케일 특징을 추출하고 통합하는 방법들이 연구되었습니다.
* **어텐션 메커니즘 (Attention Mechanisms):** 채널 어텐션 (Squeeze-and-Excitation Networks), 공간 어텐션 (Vision Transformers) 등 네트워크가 중요한 부분에 집중하도록 돕는 메커니즘을 다룹니다. 특히 Visual Attention Network (VAN)는 대형 커널 어텐션(LKA)을 사용하여 채널 및 공간 어텐션을 구축한 점에서 SegNeXt와 유사하지만, 멀티 스케일 특징 통합의 중요성을 간과했습니다.

## 🛠️ Methodology

SegNeXt는 인코더-디코더 아키텍처를 따릅니다.

* **컨볼루션 인코더 (Convolutional Encoder, MSCAN):**
  * 대부분의 기존 연구와 같이 피라미드 구조를 채택했습니다.
  * **멀티 스케일 컨볼루션 어텐션 (Multi-Scale Convolutional Attention, MSCA) 모듈:** 셀프 어텐션 대신 MSCA를 사용합니다.
    * 로컬 정보 집계를 위한 깊이별 컨볼루션 (Depth-wise Convolution).
    * 멀티 스케일 컨텍스트 캡처를 위한 여러 갈래의 깊이별 스트립 컨볼루션 (Strip Convolution, 커널 크기 7, 11, 21). 스트립 컨볼루션은 가볍고, 큰 커널 2D 컨볼루션을 모방하며, 스트립 형태의 특징을 추출하는 데 효과적입니다.
    * 채널 간 관계 모델링을 위한 $1 \times 1$ 컨볼루션.
    * 수학적으로:
            $$ Att = Conv_{1 \times 1} \left( \sum_{i=0}^{3} Scale_i(DW\text{-}Conv(F)) \right) $$
            $$ Out = Att \otimes F $$
            여기서 $F$는 입력 특징, $Att$는 어텐션 맵, $Out$은 출력, $\otimes$는 요소별 곱셈을 의미합니다. $Scale_0$는 항등 연결입니다.
  * **계층적 구조:** 4개의 스테이지로 구성되며, 공간 해상도는 $H/4 \times W/4$, $H/8 \times W/8$, $H/16 \times W/16$, $H/32 \times W/32$로 점진적으로 감소합니다.
  * 레이어 정규화 대신 배치 정규화(Batch Normalization)를 사용합니다.
  * MSCAN-T, -S, -B, -L의 네 가지 모델 크기를 설계했습니다.

* **디코더 (Decoder):**
  * 마지막 세 스테이지($H/8 \times W/8$, $H/16 \times W/16$, $H/32 \times W/32$)의 특징만을 집계하여 사용합니다. 첫 번째 스테이지의 특징은 낮은 수준의 정보가 많고 계산 오버헤드가 커서 제외했습니다.
  * 경량 Hamburger [21] 모듈을 사용하여 전역 컨텍스트를 추가로 모델링합니다. 이 조합은 성능-연산 효율성 면에서 우수함을 발견했습니다.

## 📊 Results

* **ImageNet 분류 성능:** MSCAN 인코더는 ImageNet-1K 사전 학습에서 ConvNeXt, Swin Transformer, MiT 등 최신 CNN 및 트랜스포머 기반 모델보다 우수한 Top-1 정확도를 달성했습니다.
* **성능-연산 트레이드오프:** Cityscapes 및 ADE20K 검증 세트에서 SegNeXt는 SegFormer, HRFormer, MaskFormer 등 최첨단 모델 대비 가장 우수한 성능-연산 트레이드오프를 보여주었습니다.
  * SegNeXt-S는 SegFormer-B2와 유사한 성능을 달성하면서 Cityscapes에서 약 $1/6$의 FLOPs와 $1/2$의 파라미터만을 사용했습니다.
* **최첨단 트랜스포머 모델과의 비교:**
  * ADE20K에서 SegNeXt-L은 Mask2Former (Swin-T 백본)보다 3.3 mIoU (51.0% vs 47.7%) 높았으며, 파라미터 및 연산 비용은 유사했습니다.
  * ADE20K에서 SegNeXt-B는 SegFormer-B2보다 2.0 mIoU (48.5% vs 46.5%) 높았으며, 56% 적은 연산 비용을 사용했습니다.
  * 특히 Cityscapes와 같은 고해상도 이미지 처리에서 SegNeXt-B는 SegFormer-B2보다 1.6 mIoU (82.6% vs 81.0%) 높았고, 연산은 40% 적게 사용했습니다.
* **최첨단 CNN 모델과의 비교:**
  * Pascal VOC 2012 테스트 리더보드에서 SegNeXt-L은 EfficientNet-L2 w/ NAS-FPN (추가 3억 이미지 사전 학습)을 90.6% mIoU로 능가했으며, 파라미터는 약 $1/10$에 불과했습니다 (48.7M vs 485M).
  * HRNet (OCR)과 같은 시맨틱 분할 전용 모델보다 더 적은 파라미터와 연산으로 더 높은 mIoU를 달성했습니다.
* **실시간 성능:** SegNeXt-T는 Cityscapes 테스트 세트에서 768x1,536 이미지 처리 시 단일 RTX 3090 GPU로 25 FPS를 달성하며 실시간 시맨틱 분할에서 새로운 최첨단 성능을 기록했습니다.
* **어블레이션 연구:** MSCA의 각 구성 요소 (멀티 브랜치 스트립 컨볼루션, $1 \times 1$ 컨볼루션, 어텐션)와 멀티 스케일 특징 통합이 성능 향상에 필수적임을 보여주었습니다. 디코더에서는 Hamburger 모듈이 복잡도와 성능 면에서 최상의 트레이드오프를 제공했습니다.

## 🧠 Insights & Discussion

이 논문은 트랜스포머 기반 모델이 지배하는 시맨틱 분할 분야에서 적절한 설계가 이루어진다면 컨볼루션 신경망(CNN) 기반 방법도 여전히 트랜스포머보다 더 나은 성능을 낼 수 있음을 강력하게 시사합니다. 특히, 제안된 멀티 스케일 컨볼루션 어텐션(MSCA) 모듈은 효율적인 방식으로 공간 정보를 인코딩하여, 고해상도 이미지의 객체 세부 정보를 처리하는 데 매우 효과적임을 입증했습니다. 이는 컨볼루션 연산의 잠재력을 재조명하고, 연구자들이 CNN의 가능성을 더욱 탐구하도록 장려합니다.

제한점으로는 1억 개 이상의 파라미터를 가진 대규모 모델로의 확장 가능성이나 다른 비전 또는 NLP 태스크에서의 성능 검증이 미래 연구 과제로 남아있습니다.

## 📌 TL;DR

SegNeXt는 트랜스포머 중심의 시맨틱 분할 분야에서 컨볼루션 어텐션의 효율성과 효과를 재조명합니다. 강력한 인코더, 멀티 스케일 상호작용, 공간 어텐션, 낮은 연산 복잡도를 만족하는 새로운 멀티 스케일 컨볼루션 어텐션(MSCA) 모듈과 경량 디코더를 갖춘 CNN 기반 아키텍처를 제안합니다. SegNeXt는 기존 최첨단 트랜스포머 및 CNN 모델들을 훨씬 적은 연산 비용으로 능가하며, 특히 고해상도 이미지의 디테일 처리에서 우수한 성능을 보여주며 실시간 애플리케이션에도 적합합니다.
