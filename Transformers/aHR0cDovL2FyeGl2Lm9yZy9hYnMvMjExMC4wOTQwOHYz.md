# HRFormer: High-Resolution Transformer for Dense Prediction

Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, Jingdong Wang

## 🧩 Problem to Solve

- **Vision Transformer (ViT)의 저해상도 표현 문제**: 기존 ViT는 이미지를 $16 \times 16$ 크기의 패치로 분할하여 저해상도 특징 표현을 생성합니다. 이는 인간 자세 추정, 의미론적 분할과 같은 Dense Prediction(밀집 예측) 작업에 필수적인 미세한 공간적 세부 정보(fine-grained spatial details)를 손실시킵니다.
- **단일 스케일 특징의 한계**: ViT는 단일 스케일의 특징 표현만을 출력하여, 이미지 내에 존재하는 다중 스케일 변화(multi-scale variation)를 효과적으로 다루기 어렵습니다.
- **높은 메모리 및 계산 비용**: 원본 ViT는 self-attention의 2차($O(N^2)$) 복잡도로 인해 고해상도 입력에 대해 높은 메모리 소비와 계산 비용을 요구합니다.

## ✨ Key Contributions

- **고해상도 트랜스포머 (HRFormer) 제안**: Dense Prediction 작업을 위해 고해상도 표현을 학습하는 효율적인 트랜스포머 아키텍처를 제시합니다.
- **HRNet의 다중 해상도 병렬 설계 통합**: High-Resolution Convolutional Networks (HRNet) [46]에서 영감을 받은 다중 해상도 병렬 구조를 트랜스포머에 적용하여 네트워크 전반에 걸쳐 고해상도 스트림을 유지하고 다중 스케일 정보를 효율적으로 교환합니다.
- **Local-Window Self-Attention 도입**: 이미지 특징 맵을 겹치지 않는 작은 윈도우로 분할하고 각 윈도우 내에서 독립적으로 self-attention을 수행하여 메모리 및 계산 복잡도를 공간 크기에 대해 2차($O(N^2)$)에서 선형($O(N)$)으로 감소시킵니다.
- **FFN 내 Depth-wise Convolution 삽입**: Local-window self-attention으로 인해 윈도우 간 정보 교환이 단절되는 문제를 해결하기 위해 Feed-Forward Network (FFN)에 $3 \times 3$ depth-wise convolution을 추가하여 수용장(receptive field)을 확장하고 윈도우 간 정보 교환을 가능하게 합니다.
- **최고 수준의 성능 및 효율성 달성**: COCO 자세 추정 및 의미론적 분할 벤치마크에서 기존 SOTA CNN 및 트랜스포머 모델 대비 더 적은 파라미터와 FLOPs로 경쟁력 있는 성능을 달성합니다. 예를 들어, COCO 자세 추정에서 Swin Transformer [27]보다 1.3 AP 높으면서 파라미터는 50% 적고 FLOPs는 30% 적습니다.

## 📎 Related Works

- **Vision Transformers**: ViT [13] 및 DeiT [42] 이후 다양한 Vision Transformer 변형들이 제안되었습니다.
  - **다중 스케일 특징 계층**: MViT [14], PVT [47], Swin [27] 등은 ResNet과 같은 CNN 아키텍처의 공간 구성을 따라 다중 스케일 특징 계층을 도입했습니다. HRFormer는 HRNet의 다중 해상도 병렬 설계를 활용하여 차별화됩니다.
  - **컨볼루션 통합**: CvT [48], CeiT [53], LocalViT [25] 등은 self-attention 또는 FFN 내에 depth-wise convolution을 삽입하여 트랜스포머의 지역성(locality)을 강화했습니다. HRFormer의 컨볼루션은 지역성 강화뿐만 아니라 **단절된 윈도우 간의 정보 교환**을 보장한다는 점에서 다릅니다.
  - **로컬 self-attention**: [36, 19] 등은 겹치는 로컬 윈도우를 사용했으나 계산 비용이 높았습니다. HRFormer는 [21, 44, 27]과 유사하게 **겹치지 않는 윈도우** 내에서 self-attention을 적용하여 효율성을 크게 향상시켰습니다.
  - **밀집 예측을 위한 ViT**: [63, 37] 등은 ViT를 밀집 예측에 적용하며 고해상도 표현의 중요성을 강조했습니다. HRFormer는 다중 해상도 병렬 트랜스포머 스키마를 통해 저해상도 문제를 해결하는 새로운 길을 제시합니다.
- **고해상도 CNN (HRCNN) for Dense Prediction**: 밀집 예측에서 성공한 고해상도 컨볼루션 방식 중, HRFormer는 HRNet [46]과 같이 **네트워크 전체에서 고해상도 표현을 유지**하는 방식에 속하며, Vision Transformer와 HRNet의 장점을 모두 결합합니다.

## 🛠️ Methodology

HRFormer는 HRNet [46]의 설계 원칙과 트랜스포머의 강점을 결합합니다.

1. **다중 해상도 병렬 트랜스포머**:

   - **고해상도 컨볼루션 스템**: 첫 번째 단계는 고해상도 컨볼루션 스템으로 시작하여 고해상도 특징을 효과적으로 초기화합니다.
   - **병렬 스트림**: 고해상도 스트림과 병렬로 중간 및 저해상도 스트림을 점진적으로 추가하여, 여러 해상도의 특징 맵을 동시에 유지합니다.
   - **다중 스케일 융합 모듈**: 각 해상도 스트림의 특징은 여러 트랜스포머 블록으로 독립적으로 업데이트되며, HRNet과 동일한 컨볼루션 기반의 다중 스케일 융합 모듈을 통해 해상도 간 정보가 반복적으로 교환됩니다. 이는 단거리(short-range) 및 장거리(long-range) 어텐션을 혼합하는 효과를 줍니다.

2. **HRFormer 블록**: 각 HRFormer 블록은 다음 두 가지 핵심 구성 요소로 이루어집니다 (그림 1 참조).

   - **Local-Window Self-Attention (LSA)**:
     - 입력 특징 맵 $X \in \mathbb{R}^{N \times D}$를 겹치지 않는 $K \times K$ 크기의 작은 윈도우들 $X_1, X_2, \dots, X_P$로 나눕니다.
     - 각 윈도우 $X_p$ 내에서 Multi-Head Self-Attention (MHSA)을 독립적으로 수행합니다.
     - MHSA 수식:
       $$ \text{MultiHead}(X_p) = \text{Concat}[\text{head}(X_p)_1, \dots, \text{head}(X_p)_H] \in \mathbb{R}^{K^2 \times D} $$
            $$ \text{head}(X_p)\_h = \text{Softmax}\left[\frac{(X_p W_q^h)(X_p W_k^h)^T}{\sqrt{D/H}}\right] X_p W_v^h \in \mathbb{R}^{K^2 \times D/H} $$
            $$ \hat{X}\_p = X_p + \text{MultiHead}(X_p)W_o \in \mathbb{R}^{K^2 \times D/H} $$
       상대 위치 임베딩(relative position embedding)을 사용하여 윈도우 내의 상대 위치 정보를 인코딩합니다. 이 방식은 메모리 및 계산 복잡도를 공간 크기에 대해 선형으로 줄여 효율성을 크게 향상시킵니다.
   - **FFN (Feed-Forward Network) with Depth-wise Convolution**:
     - Local-window self-attention으로 인해 윈도우 간 정보 교환이 불가능한 문제를 해결합니다.
     - 기존 트랜스포머의 FFN (MLP(MLP())) 사이에 $3 \times 3$ depth-wise convolution을 삽입합니다: MLP(DW-Conv.(MLP())).
     - 이 컨볼루션은 지역성(locality)을 강화하고, 겹치지 않는 윈도우 간의 정보 교환을 가능하게 하여 유효 수용장(effective receptive field)을 확장합니다 (그림 3 참조).

3. **Representation Head 디자인**: HRFormer의 다중 해상도 출력을 다양한 작업에 맞게 활용합니다.
   - **ImageNet 분류**: 4가지 해상도 특징 맵을 병목층(bottleneck)으로 변환 후, 스트라이드 컨볼루션으로 융합하여 단일 저해상도 특징 맵을 생성하고, 전역 평균 풀링(global average pooling) 후 분류합니다.
   - **자세 추정**: 최고 해상도 특징 맵에 직접 회귀 헤드(regression head)를 적용합니다.
   - **의미론적 분할**: 모든 저해상도 특징 맵을 최고 해상도로 업샘플링한 후, 연결(concatenate)하여 의미론적 분할 헤드에 입력합니다.

## 📊 Results

HRFormer는 ImageNet 분류, 인간 자세 추정, 의미론적 분할에서 모두 뛰어난 성능과 효율성을 입증했습니다.

- **인간 자세 추정 (COCO 데이터셋)**:
  - **HRFormer-B**: COCO val 세트에서 HRNet-W48 (입력 크기 $384 \times 288$)보다 0.9% AP 높은 77.2% AP를 달성했습니다. 이는 HRNet-W48보다 파라미터는 32% 적고, FLOPs는 19% 적은 결과입니다.
  - 다른 트랜스포머 기반 모델(ViT-Large, DeiT-B, Swin-B)과의 비교에서 HRFormer-B는 더 적은 파라미터와 FLOPs로 더 높은 AP를 기록했습니다 (예: Swin-B보다 1.3 AP 높으면서 파라미터 50%↓, FLOPs 30%↓).
- **의미론적 분할 (Cityscapes, PASCAL-Context, COCO-Stuff)**:
  - **HRFormer-B + OCR**: Cityscapes에서 SETR-PUP과 유사한 82.6% mIoU를 달성하며 파라미터 70%, FLOPs 50%를 절감했습니다.
  - PASCAL-Context에서 HRNet-W48 + OCR보다 1.1% mIoU 높은 58.5% mIoU를 달성했습니다.
  - COCO-Stuff에서 이전 최고 성능인 HRNet-W48 + OCR보다 약 2% mIoU 높은 43.3% mIoU를 기록했습니다.
- **ImageNet 분류**:
  - **HRFormer-B**: ImageNet-1K val 세트에서 DeiT-B보다 1.0% 높은 82.8% top-1 정확도를 달성했으며, 파라미터는 약 40% 적고, FLOPs는 약 20% 적습니다.
  - HRNet-T와 비교하여 HRFormer-T는 파라미터와 FLOPs가 약 50% 적으면서도 ImageNet에서 2.0%, PASCAL-Context에서 1.5%, COCO 자세 추정에서 1.6% 더 높은 성능을 보였습니다.

**어블레이션 스터디**:

- **FFN 내 $3 \times 3$ depth-wise convolution의 영향**: 이 컨볼루션은 ImageNet에서 0.65%, PASCAL-Context에서 2.9%, COCO 자세 추정에서 4.04%의 성능 향상을 가져와, 윈도우 간 정보 교환 및 지역성 강화에 핵심적인 기여를 함을 입증했습니다.
- **Shifted Window Scheme (Swin-T) vs. FFN 내 $3 \times 3$ depth-wise convolution**: Swin-T 기반 실험에서 FFN 내 $3 \times 3$ depth-wise convolution을 적용한 `IntraWin-T`가 Swin-T (shifted window 사용)보다 높은 성능을 보였습니다. HRFormer-T 기반 실험에서도 FFN 내 DW-Conv 방식이 shifted window 방식보다 모든 작업에서 크게 우수함을 확인했습니다.

## 🧠 Insights & Discussion

- **고해상도 표현 유지의 중요성**: HRFormer는 Dense Prediction 작업에 고해상도 정보가 필수적임을 다시 한번 입증하며, HRNet의 병렬 다중 해상도 디자인을 트랜스포머에 성공적으로 통합하여 이 문제를 효과적으로 해결했습니다.
- **효율적인 어텐션과 지역성 강화**: Local-window self-attention과 FFN 내 depth-wise convolution의 조합은 트랜스포머의 계산 효율성을 크게 높이면서도, 윈도우 간 정보 교환을 통해 모델의 수용장을 확장하고 지역성을 강화함으로써 Dense Prediction에 필요한 미세한 특징을 효과적으로 포착할 수 있게 합니다.
- **컨볼루션과 트랜스포머의 시너지**: 초기 컨볼루션 스템, FFN 내 컨볼루션, 그리고 다중 스케일 융합 모듈에서의 컨볼루션 사용은 트랜스포머의 장점과 컨볼루션의 지역성 및 효율성을 성공적으로 결합하여, 두 패러다임의 시너지를 극대화했음을 보여줍니다.
- **효율성과 성능의 균형**: HRFormer는 기존의 SOTA 모델들보다 적은 연산량과 파라미터로 더 나은 성능을 달성하여, 효율적인 아키텍처 설계가 대규모 모델의 대안이 될 수 있음을 시사합니다. 이는 특히 자원 제약이 있는 환경에서 큰 장점이 될 수 있습니다.
- **확장 가능성**: 본 논문에서는 UDP나 DARK와 같은 추가적인 고급 기술을 사용하지 않고도 경쟁력 있는 성능을 달성했으므로, 이러한 기술들을 적용하면 추가적인 성능 향상을 기대할 수 있습니다.

## 📌 TL;DR

HRFormer는 Vision Transformer가 Dense Prediction 작업에서 겪는 저해상도 표현 및 높은 계산 비용 문제를 해결하기 위해 고안되었습니다. 이 모델은 HRNet의 다중 해상도 병렬 설계에 Local-Window Self-Attention과 FFN 내 $3 \times 3$ Depth-wise Convolution을 결합하여, 네트워크 전반에 걸쳐 고해상도 특징을 효율적으로 유지하고 윈도우 간 정보 교환을 가능하게 합니다. 결과적으로, HRFormer는 인간 자세 추정 및 의미론적 분할에서 기존 최첨단 모델들보다 더 적은 파라미터와 FLOPs로 우수한 성능을 달성하며, 트랜스포머의 장점과 HRNet의 효율성을 성공적으로 통합했음을 입증했습니다.
